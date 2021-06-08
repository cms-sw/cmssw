#ifndef __Basic2DGenericPFlowPositionCalc_H__
#define __Basic2DGenericPFlowPositionCalc_H__

#include "RecoParticleFlow/PFClusterProducer/interface/PFCPositionCalculatorBase.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecHitFwd.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecHit.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "RecoParticleFlow/PFClusterProducer/interface/CaloRecHitResolutionProvider.h"
#include <memory>

#include <tuple>

class Basic2DGenericPFlowPositionCalc : public PFCPositionCalculatorBase {
public:
  Basic2DGenericPFlowPositionCalc(const edm::ParameterSet& conf, edm::ConsumesCollector& cc)
      : PFCPositionCalculatorBase(conf, cc),
        _posCalcNCrystals(conf.getParameter<int>("posCalcNCrystals")),
        _minAllowedNorm(conf.getParameter<double>("minAllowedNormalization")) {
    std::vector<int> detectorEnum;
    std::vector<int> depths;
    std::vector<double> logWeightDenom;
    std::vector<float> logWeightDenomInv;

    if (conf.exists("logWeightDenominatorByDetector")) {
      const std::vector<edm::ParameterSet>& logWeightDenominatorByDetectorPSet =
          conf.getParameterSetVector("logWeightDenominatorByDetector");

      for (const auto& pset : logWeightDenominatorByDetectorPSet) {
        if (!pset.exists("detector")) {
          throw cms::Exception("logWeightDenominatorByDetectorPSet") << "logWeightDenominator : detector not specified";
        }

        const std::string& det = pset.getParameter<std::string>("detector");

        if (det == std::string("HCAL_BARREL1") || det == std::string("HCAL_ENDCAP")) {
          std::vector<int> depthsT = pset.getParameter<std::vector<int> >("depths");
          std::vector<double> logWeightDenomT = pset.getParameter<std::vector<double> >("logWeightDenominator");
          if (logWeightDenomT.size() != depthsT.size()) {
            throw cms::Exception("logWeightDenominator") << "logWeightDenominator mismatch with the numbers of depths";
          }
          for (unsigned int i = 0; i < depthsT.size(); ++i) {
            if (det == std::string("HCAL_BARREL1"))
              detectorEnum.push_back(1);
            if (det == std::string("HCAL_ENDCAP"))
              detectorEnum.push_back(2);
            depths.push_back(depthsT[i]);
            logWeightDenom.push_back(logWeightDenomT[i]);
          }
        }
      }
    } else {
      detectorEnum.push_back(0);
      depths.push_back(0);
      logWeightDenom.push_back(conf.getParameter<double>("logWeightDenominator"));
    }

    for (unsigned int i = 0; i < depths.size(); ++i) {
      logWeightDenomInv.push_back(1. / logWeightDenom[i]);
    }

    //    _logWeightDenom = std::make_pair(depths,logWeightDenomInv);
    _logWeightDenom = std::make_tuple(detectorEnum, depths, logWeightDenomInv);

    _timeResolutionCalcBarrel.reset(nullptr);
    if (conf.exists("timeResolutionCalcBarrel")) {
      const edm::ParameterSet& timeResConf = conf.getParameterSet("timeResolutionCalcBarrel");
      _timeResolutionCalcBarrel = std::make_unique<CaloRecHitResolutionProvider>(timeResConf);
    }
    _timeResolutionCalcEndcap.reset(nullptr);
    if (conf.exists("timeResolutionCalcEndcap")) {
      const edm::ParameterSet& timeResConf = conf.getParameterSet("timeResolutionCalcEndcap");
      _timeResolutionCalcEndcap = std::make_unique<CaloRecHitResolutionProvider>(timeResConf);
    }

    switch (_posCalcNCrystals) {
      case 5:
      case 9:
      case -1:
        break;
      default:
        edm::LogError("Basic2DGenericPFlowPositionCalc") << "posCalcNCrystals not valid";
        assert(0);  // bug
    }
  }

  Basic2DGenericPFlowPositionCalc(const Basic2DGenericPFlowPositionCalc&) = delete;
  Basic2DGenericPFlowPositionCalc& operator=(const Basic2DGenericPFlowPositionCalc&) = delete;

  void calculateAndSetPosition(reco::PFCluster&) override;
  void calculateAndSetPositions(reco::PFClusterCollection&) override;

private:
  const int _posCalcNCrystals;
  std::tuple<std::vector<int>, std::vector<int>, std::vector<float> > _logWeightDenom;
  const float _minAllowedNorm;

  std::unique_ptr<CaloRecHitResolutionProvider> _timeResolutionCalcBarrel;
  std::unique_ptr<CaloRecHitResolutionProvider> _timeResolutionCalcEndcap;

  void calculateAndSetPositionActual(reco::PFCluster&) const;
};

DEFINE_EDM_PLUGIN(PFCPositionCalculatorFactory, Basic2DGenericPFlowPositionCalc, "Basic2DGenericPFlowPositionCalc");

#endif
