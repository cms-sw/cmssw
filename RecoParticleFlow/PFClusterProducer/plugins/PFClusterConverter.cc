#include <Eigen/Core>
#include <Eigen/Dense>
#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
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

class PFClusterConverter : public edm::stream::EDProducer<> {
public:
  PFClusterConverter(edm::ParameterSet const& config)
      : pfClusterSoAToken_(consumes(config.getParameter<edm::InputTag>("src"))),
        pfRecHitFractionSoAToken_(consumes(config.getParameter<edm::InputTag>("src"))),
        InputPFRecHitSoA_Token_{consumes(config.getParameter<edm::InputTag>("PFRecHitsLabelIn"))},
        outPFClusterSoAToken_(produces<reco::PFClusterHostCollection>()),
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
    }
  }

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;
    desc.add<edm::InputTag>("src", edm::InputTag("pfClusterSoAProducer"));
    desc.add<edm::InputTag>("PFRecHitsLabelIn", edm::InputTag("pfRecHitSoAProducerHCAL"));
    //desc.add<edm::InputTag>("recHitsSource", edm::InputTag("legacyPFRecHitProducer"));
    desc.add<bool>("usePFThresholdsFromDB", true);
    {
      edm::ParameterSetDescription pfClusterBuilder;
      pfClusterBuilder.add<unsigned int>("maxIterations", 5);
      pfClusterBuilder.add<double>("minFracTot", 1e-20);
      pfClusterBuilder.add<double>("minFractionToKeep", 1e-7);
      pfClusterBuilder.add<bool>("excludeOtherSeeds", true);
      pfClusterBuilder.add<double>("showerSigma", 10.);
      pfClusterBuilder.add<double>("stoppingTolerance", 1e-8);
      pfClusterBuilder.add<double>("timeSigmaEB", 10.);
      pfClusterBuilder.add<double>("timeSigmaEE", 10.);
      pfClusterBuilder.add<double>("maxNSigmaTime", 10.);
      pfClusterBuilder.add<double>("minChi2Prob", 0.);
      pfClusterBuilder.add<bool>("clusterTimeResFromSeed", false);
      pfClusterBuilder.add<std::string>("algoName", "");
      pfClusterBuilder.add<edm::ParameterSetDescription>("positionCalcForConvergence", {});
      {
        edm::ParameterSetDescription validator;
        validator.add<std::string>("detector", "");
        validator.add<std::vector<int>>("depths", {});
        validator.add<std::vector<double>>("recHitEnergyNorm", {});
        std::vector<edm::ParameterSet> vDefaults(2);
        vDefaults[0].addParameter<std::string>("detector", "HCAL_BARREL1");
        vDefaults[0].addParameter<std::vector<int>>("depths", {1, 2, 3, 4});
        vDefaults[0].addParameter<std::vector<double>>("recHitEnergyNorm", {0.1, 0.2, 0.3, 0.3});
        vDefaults[1].addParameter<std::string>("detector", "HCAL_ENDCAP");
        vDefaults[1].addParameter<std::vector<int>>("depths", {1, 2, 3, 4, 5, 6, 7});
        vDefaults[1].addParameter<std::vector<double>>("recHitEnergyNorm", {0.1, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2});
        pfClusterBuilder.addVPSet("recHitEnergyNorms", validator, vDefaults);
      }
      {
        edm::ParameterSetDescription bar;
        bar.add<std::string>("algoName", "Basic2DGenericPFlowPositionCalc");
        bar.add<double>("minFractionInCalc", 1e-9);
        bar.add<int>("posCalcNCrystals", 5);
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
          bar.addVPSet("logWeightDenominatorByDetector", validator, vDefaults);
        }
        bar.add<double>("minAllowedNormalization", 1e-9);
        bar.add<edm::ParameterSetDescription>("timeResolutionCalcBarrel", {});
        bar.add<edm::ParameterSetDescription>("timeResolutionCalcEndcap", {});
        pfClusterBuilder.add("positionCalc", bar);
      }
      {
        edm::ParameterSetDescription bar;
        bar.add<std::string>("algoName", "Basic2DGenericPFlowPositionCalc");
        bar.add<double>("minFractionInCalc", 1e-9);
        bar.add<int>("posCalcNCrystals", -1);
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
          bar.addVPSet("logWeightDenominatorByDetector", validator, vDefaults);
        }
        bar.add<double>("minAllowedNormalization", 1e-9);
        bar.add<edm::ParameterSetDescription>("timeResolutionCalcBarrel", {});
        bar.add<edm::ParameterSetDescription>("timeResolutionCalcEndcap", {});
        pfClusterBuilder.add("allCellsPositionCalc", bar);
      }
      {
        edm::ParameterSetDescription bar;
        bar.add<double>("corrTermLowE", 0.);
        bar.add<double>("threshLowE", 6.);
        bar.add<double>("noiseTerm", 21.86);
        bar.add<double>("constantTermLowE", 4.24);
        bar.add<double>("noiseTermLowE", 8.);
        bar.add<double>("threshHighE", 15.);
        bar.add<double>("constantTerm", 2.82);
        pfClusterBuilder.add("timeResolutionCalcBarrel", bar);
      }
      {
        edm::ParameterSetDescription bar;
        bar.add<double>("corrTermLowE", 0.);
        bar.add<double>("threshLowE", 6.);
        bar.add<double>("noiseTerm", 21.86);
        bar.add<double>("constantTermLowE", 4.24);
        bar.add<double>("noiseTermLowE", 8.);
        bar.add<double>("threshHighE", 15.);
        bar.add<double>("constantTerm", 2.82);
        pfClusterBuilder.add("timeResolutionCalcEndcap", bar);
      }
      {
        edm::ParameterSetDescription bar;
        pfClusterBuilder.add("positionReCalc", bar);
      }
      {
        edm::ParameterSetDescription bar;
        pfClusterBuilder.add("energyCorrector", bar);
      }
      desc.add("pfClusterBuilder", pfClusterBuilder);
    }
    descriptions.addWithDefaultLabel(desc);
  }

private:
  void produce(edm::Event&, const edm::EventSetup&) override;
  const edm::EDGetTokenT<reco::PFClusterHostCollection> pfClusterSoAToken_;
  const edm::EDGetTokenT<reco::PFRecHitFractionHostCollection> pfRecHitFractionSoAToken_;
  const edm::EDGetTokenT<reco::PFRecHitHostCollection> InputPFRecHitSoA_Token_;
  const edm::EDPutTokenT<reco::PFClusterHostCollection> outPFClusterSoAToken_;
  const edm::EDGetTokenT<reco::PFRecHitCollection> recHitsLabel_;
  const edm::ESGetToken<HcalPFCuts, HcalPFCutsRcd> hcalCutsToken_;
  const bool cutsFromDB_;
  // the actual algorithm
  std::unique_ptr<PFCPositionCalculatorBase> positionCalc_;
  std::unique_ptr<PFCPositionCalculatorBase> allCellsPositionCalc_;
};

void PFClusterConverter::produce(edm::Event& event, const edm::EventSetup& setup) {
  const reco::PFRecHitHostCollection& pfRecHits = event.get(InputPFRecHitSoA_Token_);

  HcalPFCuts const* paramPF = cutsFromDB_ ? &setup.getData(hcalCutsToken_) : nullptr;

  auto const& pfClusterSoA = event.get(pfClusterSoAToken_).const_view();

  int nRH = 0;
  if (pfRecHits->metadata().size() != 0) nRH = pfRecHits.view().size();

  std::unique_ptr<reco::PFClusterHostCollection> outPFClusterSoAPtr;

  if (pfClusterSoA.nSeeds() == 0 || nRH == 0) {
    outPFClusterSoAPtr = std::make_unique<reco::PFClusterHostCollection>(0, cms::alpakatools::host());	  
    event.emplace(outPFClusterSoAToken_, std::move(*outPFClusterSoAPtr));
    return;
  }
  auto const& pfRecHitFractionSoA = event.get(pfRecHitFractionSoAToken_).const_view();

  auto const rechitsHandle = event.getHandle(recHitsLabel_);

  outPFClusterSoAPtr = std::make_unique<reco::PFClusterHostCollection>(pfClusterSoA.nSeeds(), cms::alpakatools::host());

  auto& outPFClusterSoA = outPFClusterSoAPtr->view(); 

  std::vector<int> nTopoSeeds(nRH, 0);

  for (int i = 0; i < pfClusterSoA.nSeeds(); i++) {
    nTopoSeeds[pfClusterSoA[i].topoId()]++;
  }
  // Looping over SoA PFClusters
  outPFClusterSoA.nTopos() = pfClusterSoA.nTopos();
  outPFClusterSoA.nSeeds() = pfClusterSoA.nSeeds();
  outPFClusterSoA.nRHFracs() = pfClusterSoA.nRHFracs();
  outPFClusterSoA.size() = pfClusterSoA.size();

  for (int i = 0; i < pfClusterSoA.nSeeds(); i++) {
    unsigned int n = pfClusterSoA[i].seedRHIdx();
    reco::PFCluster temp;
    temp.setSeed((*rechitsHandle)[n].detId());  // Pulling the detId of this PFRecHit from the legacy format input
    int offset = pfClusterSoA[i].rhfracOffset();
    for (int k = offset; k < (offset + pfClusterSoA[i].rhfracSize()) && k >= 0;
           k++) {  // Looping over PFRecHits in the same topo cluster
      if (pfRecHitFractionSoA[k].pfrhIdx() < nRH && pfRecHitFractionSoA[k].pfrhIdx() > -1 &&
            pfRecHitFractionSoA[k].frac() > 0.0) {
        const reco::PFRecHitRef& refhit = reco::PFRecHitRef(rechitsHandle, pfRecHitFractionSoA[k].pfrhIdx());
        temp.addRecHitFraction(reco::PFRecHitFraction(refhit, pfRecHitFractionSoA[k].frac()));
      }
    }

    if (nTopoSeeds[pfClusterSoA[i].topoId()] == 1 && allCellsPositionCalc_) {
      allCellsPositionCalc_->calculateAndSetPosition(temp, paramPF);
    } else {
      positionCalc_->calculateAndSetPosition(temp, paramPF);
    }
    
    outPFClusterSoA[i].depth() = pfClusterSoA[i].depth();
    outPFClusterSoA[i].seedRHIdx() = n;
    outPFClusterSoA[i].topoId() = pfClusterSoA[i].topoId();
    outPFClusterSoA[i].rhfracSize() = pfClusterSoA[i].rhfracSize();
    outPFClusterSoA[i].rhfracOffset() = pfClusterSoA[i].rhfracOffset();
    outPFClusterSoA[i].energy() = temp.energy();
    outPFClusterSoA[i].x() = temp.x();
    outPFClusterSoA[i].y() = temp.y();
    outPFClusterSoA[i].z() = temp.z();
    outPFClusterSoA[i].topoRHCount() = pfClusterSoA[i].topoRHCount();
  }
  event.emplace(outPFClusterSoAToken_, std::move(*outPFClusterSoAPtr));
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(PFClusterConverter);
