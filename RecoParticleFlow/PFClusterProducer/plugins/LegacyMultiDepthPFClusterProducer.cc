#include <Eigen/Core>
#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <memory>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Utilities/interface/StreamID.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Utilities/interface/EDPutToken.h"

#include "CondFormats/DataRecord/interface/HcalPFCutsRcd.h"
#include "CondTools/Hcal/interface/HcalPFCutsHandler.h"

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecHitHostCollection.h"
#include "DataFormats/ParticleFlowReco/interface/PFClusterHostCollection.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecHitFractionHostCollection.h"
#include "RecoParticleFlow/PFClusterProducer/interface/PFCPositionCalculatorBase.h"

#include "RecoParticleFlow/PFClusterProducer/interface/PFClusterEnergyCorrectorBase.h"

class LegacyMultiDepthPFClusterProducer : public edm::stream::EDProducer<> {
public:
  LegacyMultiDepthPFClusterProducer(edm::ParameterSet const& config)
      : pfClusterSoAToken_(consumes(config.getParameter<edm::InputTag>("pfClusterSoA"))),
        pfRecHitFractionSoAToken_(consumes(config.getParameter<edm::InputTag>("pfRecHitFractionSoA"))),
        inputPFRecHitSoA_Token_{consumes(config.getParameter<edm::InputTag>("pfRecHitsSoA"))},
        legacyPfClustersToken_(produces()),
        recHitsLabel_(consumes(config.getParameter<edm::InputTag>("recHitsSource"))),
        hcalCutsToken_(esConsumes<HcalPFCuts, HcalPFCutsRcd>(edm::ESInputTag("", "withTopo"))),
        cutsFromDB_(config.getParameter<bool>("usePFThresholdsFromDB")) {
    edm::ConsumesCollector cc = consumesCollector();

    //setup pf cluster builder if requested
    const edm::ParameterSet& pfcConf = config.getParameterSet("pfClusterBuilder");
    if (!pfcConf.empty()) {
      const auto& acConf = pfcConf.getParameterSet("positionCalc");
      if (!acConf.empty()) {
        const std::string& algoac = acConf.getParameter<std::string>("algoName");
        if (!algoac.empty())
          positionCalc_ = PFCPositionCalculatorFactory::get()->create(algoac, acConf, cc);
      }

      const auto& acConf2 = pfcConf.getParameterSet("allCellsPositionCalc");
      if (!acConf2.empty()) {
        const std::string& algoac = acConf2.getParameter<std::string>("algoName");
        if (!algoac.empty())
          allCellsPositionCalc_ = PFCPositionCalculatorFactory::get()->create(algoac, acConf2, cc);
      }
      // see if new need to apply corrections, setup if there.
      const edm::ParameterSet& cConf = config.getParameterSet("energyCorrector");
      if (!cConf.empty()) {
        const std::string& cName = cConf.getParameter<std::string>("algoName");
        if (!cName.empty())
          energyCorrector_ = PFClusterEnergyCorrectorFactory::get()->create(cName, cConf);
      }
    }
  }

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;

    desc.add<edm::InputTag>("pfClusterSoA", edm::InputTag("pfClusterSoAProducer"));
    desc.add<edm::InputTag>("pfRecHitFractionSoA", edm::InputTag("pfClusterSoAProducer"));
    desc.add<edm::InputTag>("pfRecHitsSoA", edm::InputTag("pfRecHitSoAProducerHCAL"));
    desc.add<edm::InputTag>("recHitsSource", edm::InputTag("legacyPFRecHitProducer"));
    desc.add<bool>("usePFThresholdsFromDB", true);

    desc.add<edm::ParameterSetDescription>("energyCorrector", {});
    {
      edm::ParameterSetDescription pset0;
      pset0.add<std::string>("algoName", "PFMultiDepthClusterizer");
      {
        edm::ParameterSetDescription pset1;
        pset1.add<std::string>("algoName", "Basic2DGenericPFlowPositionCalc");
        {
          edm::ParameterSetDescription validator;
          validator.add<std::string>("detector", "");
          validator.add<std::vector<int>>("depths", {});
          validator.add<std::vector<double>>("logWeightDenominator", {});
          std::vector<edm::ParameterSet> vDefaults(2);
          vDefaults[0].addParameter<std::string>("detector", "HCAL_BARREL1");
          vDefaults[0].addParameter<std::vector<int>>("depths", {1, 2, 3, 4});
          vDefaults[0].addParameter<std::vector<double>>("logWeightDenominator", {0.1, 0.2, 0.3, 0.3});
          vDefaults[1].addParameter<std::string>("detector", "HCAL_ENDCAP");
          vDefaults[1].addParameter<std::vector<int>>("depths", {1, 2, 3, 4, 5, 6, 7});
          vDefaults[1].addParameter<std::vector<double>>("logWeightDenominator", {0.1, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2});
          pset1.addVPSet("logWeightDenominatorByDetector", validator, vDefaults);
        }
        pset1.add<double>("minAllowedNormalization", 1e-09);
        pset1.add<double>("minFractionInCalc", 1e-09);
        pset1.add<int>("posCalcNCrystals", -1);
        pset1.add<edm::ParameterSetDescription>("timeResolutionCalcBarrel", {});
        pset1.add<edm::ParameterSetDescription>("timeResolutionCalcEndcap", {});
        pset0.add<edm::ParameterSetDescription>("allCellsPositionCalc", pset1);
      }
      pset0.add<edm::ParameterSetDescription>("positionCalc", {});
      pset0.add<double>("minFractionToKeep", 1e-07);
      desc.add<edm::ParameterSetDescription>("pfClusterBuilder", pset0);
    }

    desc.add<edm::ParameterSetDescription>("positionReCalc", {});
    descriptions.addWithDefaultLabel(desc);
  }

private:
  void produce(edm::Event&, const edm::EventSetup&) override;
  const edm::EDGetTokenT<reco::PFClusterHostCollection> pfClusterSoAToken_;
  const edm::EDGetTokenT<reco::PFRecHitFractionHostCollection> pfRecHitFractionSoAToken_;
  const edm::EDGetTokenT<reco::PFRecHitHostCollection> inputPFRecHitSoA_Token_;
  const edm::EDPutTokenT<reco::PFClusterCollection> legacyPfClustersToken_;
  const edm::EDGetTokenT<reco::PFRecHitCollection> recHitsLabel_;
  const edm::ESGetToken<HcalPFCuts, HcalPFCutsRcd> hcalCutsToken_;
  const bool cutsFromDB_;
  // the actual algorithm
  std::unique_ptr<PFCPositionCalculatorBase> positionCalc_;
  std::unique_ptr<PFCPositionCalculatorBase> allCellsPositionCalc_;
  std::unique_ptr<PFClusterEnergyCorrectorBase> energyCorrector_;
};

void LegacyMultiDepthPFClusterProducer::produce(edm::Event& event, const edm::EventSetup& setup) {
  const reco::PFRecHitHostCollection& pfRecHits = event.get(inputPFRecHitSoA_Token_);

  HcalPFCuts const* paramPF = cutsFromDB_ ? &setup.getData(hcalCutsToken_) : nullptr;

  auto const& pfClusterSoA = event.get(pfClusterSoAToken_).const_view();
  auto const& pfRecHitFractionSoA = event.get(pfRecHitFractionSoAToken_).const_view();

  auto const nRoots = pfClusterSoA.nTopos();

  int nRH = pfRecHits->metadata().size();

  reco::PFClusterCollection out;

  out.reserve(nRoots);

  auto const rechitsHandle = event.getHandle(recHitsLabel_);

  // Build PFClusters in legacy format

  std::unordered_map<int, int> nTopoSeeds;
  nTopoSeeds.reserve(pfClusterSoA.nSeeds());

  for (int i = 0; i < pfClusterSoA.nSeeds(); ++i)
    nTopoSeeds[pfClusterSoA[i].topoId()]++;

  // Looping over SoA PFClusters to produce legacy PFCluster collection
  auto const& recHits = *rechitsHandle;

  for (int i = 0; i < pfClusterSoA.nSeeds(); i++) {
    unsigned int seedIdx = pfClusterSoA[i].seedRHIdx();

    if (seedIdx >= static_cast<unsigned>(nRH))
      continue;

    reco::PFCluster temp;

    temp.setSeed(recHits[seedIdx].detId());

    int const offset = pfClusterSoA[i].rhfracOffset();
    int const size = pfClusterSoA[i].rhfracSize();

    for (int k = offset; k < (offset + size); k++) {  // Looping over PFRecHits in the same topo cluster
      if (pfRecHitFractionSoA[k].pfrhIdx() < nRH && pfRecHitFractionSoA[k].pfrhIdx() > -1 &&
          pfRecHitFractionSoA[k].frac() > 0.0f) {
        const reco::PFRecHitRef& refhit = reco::PFRecHitRef(rechitsHandle, pfRecHitFractionSoA[k].pfrhIdx());
        temp.addRecHitFraction(reco::PFRecHitFraction(refhit, pfRecHitFractionSoA[k].frac()));
      }
    }

    // Now PFRecHitFraction of this PFCluster is set. Now compute calculateAndSetPosition (energy, position etc)
    if (nTopoSeeds[pfClusterSoA[i].topoId()] == 1 && allCellsPositionCalc_) {
      allCellsPositionCalc_->calculateAndSetPosition(temp, paramPF);
    } else {
      positionCalc_->calculateAndSetPosition(temp, paramPF);
    }
    out.emplace_back(std::move(temp));
  }

  event.emplace(legacyPfClustersToken_, std::move(out));
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(LegacyMultiDepthPFClusterProducer);
