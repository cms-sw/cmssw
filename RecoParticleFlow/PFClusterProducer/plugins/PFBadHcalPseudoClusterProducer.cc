// system include files
#include <iostream>
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

#include "DataFormats/ParticleFlowReco/interface/PFCluster.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecHitFraction.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecHit.h"

#include "DataFormats/HcalDetId/interface/HcalDetId.h"

#include "CondFormats/HcalObjects/interface/HcalChannelStatus.h"
#include "CondFormats/HcalObjects/interface/HcalChannelQuality.h"
#include "CondFormats/HcalObjects/interface/HcalCondObjectContainer.h"
#include "CondFormats/DataRecord/interface/HcalChannelQualityRcd.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/CaloTopology/interface/HcalTopology.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"

#include "FWCore/Framework/interface/ESHandle.h"


using namespace std;
using namespace edm;

//
// class declaration
//

class PFBadHcalPseudoClusterProducer : public edm::stream::EDProducer<> {
    public:
        explicit PFBadHcalPseudoClusterProducer(const edm::ParameterSet&);
        ~PFBadHcalPseudoClusterProducer() override;

        static void fillDescriptions(edm::ConfigurationDescriptions & descriptions);

    private:
        virtual void init(const EventSetup& c) ;
        void produce(edm::Event&, const edm::EventSetup&) override;

        bool enabled_;
        bool debug_;

        edm::ESHandle<HcalChannelQuality> hQuality_;
        edm::ESHandle<CaloGeometry> hGeom_;
        unsigned long long cacheId_quality_, cacheId_geom_;
        std::vector<reco::PFCluster> badAreasC_;
        std::vector<reco::PFRecHit> badAreasRH_;
};


PFBadHcalPseudoClusterProducer::PFBadHcalPseudoClusterProducer(const edm::ParameterSet& ps) :
   enabled_(ps.getParameter<bool>("enable")),
   debug_(ps.getUntrackedParameter<bool>("debug",false)),
   cacheId_quality_(0), cacheId_geom_(0)
{
    produces<std::vector<reco::PFCluster>>();
    produces<std::vector<reco::PFRecHit>>("hits");
}


PFBadHcalPseudoClusterProducer::~PFBadHcalPseudoClusterProducer()
{
}

void PFBadHcalPseudoClusterProducer::init(const EventSetup& iSetup)
{
    badAreasC_.clear();
    badAreasRH_.clear();

    edm::ESHandle<HcalChannelQuality> hQuality_;
    iSetup.get<HcalChannelQualityRcd>().get("withTopo",hQuality_);
    const HcalChannelQuality & chanquality = *hQuality_;

    edm::ESHandle<CaloGeometry> hGeom_;
    iSetup.get<CaloGeometryRecord>().get(hGeom_);
    const CaloGeometry& caloGeom = *hGeom_;
    const CaloSubdetectorGeometry *hbGeom = caloGeom.getSubdetectorGeometry(DetId::Hcal, HcalBarrel);
    const CaloSubdetectorGeometry *heGeom = caloGeom.getSubdetectorGeometry(DetId::Hcal, HcalEndcap);

    int statusMask = ((1<<HcalChannelStatus::HcalCellOff) | (1<<HcalChannelStatus::HcalCellMask) | (1<<HcalChannelStatus::HcalCellDead));
    // histogram the number of bad depths at each ieta, iphi
    std::map<std::pair<int,int>, int> good, bads;
    std::map<std::pair<int,int>, std::pair<int,HcalSubdetector>> minDepths;
    for (const DetId & i : chanquality.getAllChannels()) {
        if (i.det()!=DetId::Hcal) continue; // not an hcal cell
        HcalDetId id = HcalDetId(i);
        if (id.subdet() != HcalBarrel && id.subdet() != HcalEndcap) continue; // we don't deal with HO and HF here
        int status = chanquality.getValues(id)->getValue();
        auto tower = std::make_pair(id.ieta(), id.iphi());
        if (status & statusMask) {
            bads[tower]++;
            if (debug_) std::cout << "Channel " << i() << " (subdet " << id.subdet() << ", zside " << id.zside() << ", ieta " << id.ieta() << ", iphi " << id.iphi() << " depth " << id.depth() << " has status " << status << "  masked " << (status & statusMask) << std::endl;
        } else {
            good[tower]++;
        }
        auto & minD = minDepths[tower];
        if (minD.second == HcalEmpty || minD.first > id.depth()) {
            minD.first  = id.depth();
            minD.second = id.subdet();
        }
    }

    const float dummyEnergy = 1e-5; // non-zero, but small (even if it shouldn't ever be used)
    for (const auto & rec : bads) {
        int ieta = rec.first.first, iphi = rec.first.second, nbad = rec.second, ngood = good[rec.first];
        auto minDepth = minDepths[rec.first];   
        bool barrel = minDepth.second == HcalBarrel;
        HcalDetId id(minDepth.second, ieta, iphi, minDepth.first);
        bool isBad = (nbad > 0 && nbad >= ngood);
        if (debug_) std::cout << "At ieta " << id.ieta() << ", iphi " << id.iphi() << " I have " << nbad << " bad depths, " << ngood << " good depths. First depth is in " << (barrel ? "HB" : "HE") << " depth " << minDepth.first <<"; " << (isBad ? " MARK BAD": " ignore") << std::endl;
        if (!isBad) continue;
        PFLayer::Layer layer = (barrel ? PFLayer::HCAL_BARREL1 : PFLayer::HCAL_ENDCAP);
        // make a PF RecHit
        std::shared_ptr<const CaloCellGeometry> thisCell = (barrel ? hbGeom : heGeom)->getGeometry(id);
        const GlobalPoint & pos = thisCell->getPosition();
        badAreasRH_.emplace_back( thisCell, id(), layer, dummyEnergy );
        // make a PF cluster (but can't add the rechit, as for that I need an edm::Ref)
        badAreasC_.emplace_back( layer, dummyEnergy, pos.x(), pos.y(), pos.z() );
        badAreasC_.back().setFlags(reco::CaloCluster::badHcalMarker);
    }

    cacheId_quality_ = iSetup.get<HcalChannelQualityRcd>().cacheIdentifier();
    cacheId_geom_ = iSetup.get<CaloGeometryRecord>().cacheIdentifier();
}

// ------------ method called to produce the data  ------------
    void
PFBadHcalPseudoClusterProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
    if (enabled_) {
        if (cacheId_quality_ != iSetup.get<HcalChannelQualityRcd>().cacheIdentifier() ||
               cacheId_geom_ != iSetup.get<CaloGeometryRecord>().cacheIdentifier()) {
            init(iSetup);
        }
    }


    auto outRH = std::make_unique<std::vector<reco::PFRecHit>>(badAreasRH_);
    auto rhHandle = iEvent.put(std::move(outRH), "hits");

    auto outC = std::make_unique<std::vector<reco::PFCluster>>(badAreasC_);

    // now we go set references
    for (unsigned int i = 0, n = rhHandle->size(); i < n; ++i) {
        (*outC)[i].addRecHitFraction( reco::PFRecHitFraction(reco::PFRecHitRef(rhHandle,i), 1.0) );
    }

    iEvent.put(std::move(outC));
}

void PFBadHcalPseudoClusterProducer::fillDescriptions(edm::ConfigurationDescriptions & descriptions) {
    edm::ParameterSetDescription desc;
    desc.add<bool>("enable", false)->setComment("activate the module (if false, it doesn't check the DB and produces an empty collection)");
    desc.addUntracked<bool>("debug", false);
    descriptions.add("particleFlowBadHcalPseudoCluster", desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(PFBadHcalPseudoClusterProducer);

