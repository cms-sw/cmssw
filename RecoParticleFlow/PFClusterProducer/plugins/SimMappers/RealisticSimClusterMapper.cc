/////////////////////////
// Author: Felice Pantaleo
// Date:   30/06/2017
// Email: felice@cern.ch
/////////////////////////
#include <unordered_map>
#include "RealisticSimClusterMapper.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "FWCore/Framework/interface/Event.h"

#include "SimDataFormats/CaloAnalysis/interface/SimCluster.h"
#include "DataFormats/ForwardDetId/interface/HGCalDetId.h"
#include "RealisticHitToClusterAssociator.h"

#ifdef PFLOW_DEBUG
#define LOGVERB(x) edm::LogVerbatim(x)
#define LOGWARN(x) edm::LogWarning(x)
#define LOGERR(x) edm::LogError(x)
#define LOGDRESSED(x) edm::LogInfo(x)
#else
#define LOGVERB(x) LogTrace(x)
#define LOGWARN(x) edm::LogWarning(x)
#define LOGERR(x) edm::LogError(x)
#define LOGDRESSED(x) LogDebug(x)
#endif

void RealisticSimClusterMapper::updateEvent(const edm::Event& ev)
{
    ev.getByToken(simClusterToken_, simClusterH_);
}

void RealisticSimClusterMapper::update(const edm::EventSetup& es)
{
    rhtools_.getEventSetup(es);

}

void RealisticSimClusterMapper::buildClusters(const edm::Handle<reco::PFRecHitCollection>& input,
        const std::vector<bool>& rechitMask, const std::vector<bool>& seedable,
        reco::PFClusterCollection& output)
{
    const SimClusterCollection& simClusters = *simClusterH_;
    auto const& hits = *input;
    RealisticHitToClusterAssociator realisticAssociator;
    const int numberOfLayers = rhtools_.getLayer(ForwardSubdetector::ForwardEmpty);
    realisticAssociator.init(hits.size(), simClusters.size(), numberOfLayers + 1);
    // for quick indexing back to hit energy
    std::unordered_map < uint32_t, size_t > detIdToIndex(hits.size());
    for (uint32_t i = 0; i < hits.size(); ++i)
    {
        detIdToIndex[hits[i].detId()] = i;
        auto ref = makeRefhit(input, i);
        const auto& hitPos = rhtools_.getPosition(ref->detId());

        realisticAssociator.insertHitPosition(hitPos.x(), hitPos.y(), hitPos.z(), i);
        realisticAssociator.insertHitEnergy(ref->energy(), i);
        realisticAssociator.insertLayerId(rhtools_.getLayerWithOffset(ref->detId()), i);

    }

    for (unsigned int ic = 0; ic < simClusters.size(); ++ic)
    {
        const auto & sc = simClusters[ic];
        const auto& hitsAndFractions = sc.hits_and_fractions();
        for (const auto& hAndF : hitsAndFractions)
        {
            auto itr = detIdToIndex.find(hAndF.first);
            if (itr == detIdToIndex.end())
            {
                continue; // hit wasn't saved in reco or did not pass the SNR threshold
            }
            auto hitId = itr->second;
            auto ref = makeRefhit(input, hitId);
            float fraction = hAndF.second;
            float associatedEnergy = fraction * ref->energy();
            realisticAssociator.insertSimClusterIdAndFraction(ic, fraction, hitId,
                    associatedEnergy);
        }
    }
    realisticAssociator.computeAssociation(exclusiveFraction_, useMCFractionsForExclEnergy_,
            rhtools_.lastLayerEE(), rhtools_.lastLayerFH());
    realisticAssociator.findAndMergeInvisibleClusters(invisibleFraction_, exclusiveFraction_);
    realisticAssociator.findCentersOfGravity();
    if(maxDistanceFilter_)
        realisticAssociator.filterHitsByDistance(maxDistance_);

    const auto& realisticClusters = realisticAssociator.realisticClusters();
    unsigned int nClusters = realisticClusters.size();
    for (unsigned ic = 0; ic < nClusters; ++ic)
    {
        float highest_energy = 0.0f;
        output.emplace_back();
        reco::PFCluster& back = output.back();
        edm::Ref < std::vector<reco::PFRecHit> > seed;
        if (realisticClusters[ic].isVisible())
        {
            const auto& hitsIdsAndFractions = realisticClusters[ic].hitsIdsAndFractions();
            for (const auto& idAndF : hitsIdsAndFractions)
            {
                auto fraction = idAndF.second;
                if (fraction > 0.f)
                {
                    auto ref = makeRefhit(input, idAndF.first);
                    back.addRecHitFraction(reco::PFRecHitFraction(ref, fraction));
                    const float hit_energy = fraction * ref->energy();
                    if (hit_energy > highest_energy || highest_energy == 0.0)
                    {
                        highest_energy = hit_energy;
                        seed = ref;
                    }
                }
            }
        }
        if (back.hitsAndFractions().size() != 0)
        {
            back.setSeed(seed->detId());
            back.setEnergy(realisticClusters[ic].getEnergy());
            back.setCorrectedEnergy(realisticClusters[ic].getEnergy());
        }
        else
        {
            back.setSeed(-1);
            back.setEnergy(0.f);
        }
    }
}

