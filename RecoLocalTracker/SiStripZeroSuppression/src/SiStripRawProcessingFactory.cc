#include "RecoLocalTracker/SiStripZeroSuppression/interface/SiStripRawProcessingFactory.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "RecoLocalTracker/SiStripZeroSuppression/interface/SiStripRawProcessingAlgorithms.h"
#include "RecoLocalTracker/SiStripZeroSuppression/interface/SiStripPedestalsSubtractor.h"
#include "RecoLocalTracker/SiStripZeroSuppression/interface/SiStripFedZeroSuppression.h"
#include "RecoLocalTracker/SiStripZeroSuppression/interface/MedianCMNSubtractor.h"
#include "RecoLocalTracker/SiStripZeroSuppression/interface/PercentileCMNSubtractor.h"
#include "RecoLocalTracker/SiStripZeroSuppression/interface/IteratedMedianCMNSubtractor.h"
#include "RecoLocalTracker/SiStripZeroSuppression/interface/FastLinearCMNSubtractor.h"
#include "RecoLocalTracker/SiStripZeroSuppression/interface/TT6CMNSubtractor.h"


std::unique_ptr<SiStripRawProcessingAlgorithms>
SiStripRawProcessingFactory::create(const edm::ParameterSet& conf)
{
  return std::unique_ptr<SiStripRawProcessingAlgorithms>(
      new SiStripRawProcessingAlgorithms(
        std::move(create_SubtractorPed(conf)),
        std::move(create_SubtractorCMN(conf)),
        std::move(create_Suppressor(conf)),
        std::move(create_Restorer(conf)),
        conf.getParameter<bool>("doAPVRestore"),
        conf.getParameter<bool>("useCMMeanMap")
      ));
}

std::unique_ptr<SiStripPedestalsSubtractor>
SiStripRawProcessingFactory::create_SubtractorPed(const edm::ParameterSet& conf)
{
  return std::unique_ptr<SiStripPedestalsSubtractor>(new SiStripPedestalsSubtractor(
      conf.getParameter<bool>("PedestalSubtractionFedMode")));
}

std::unique_ptr<SiStripCommonModeNoiseSubtractor>
SiStripRawProcessingFactory::create_SubtractorCMN(const edm::ParameterSet& conf)
{
  const std::string mode = conf.getParameter<std::string>("CommonModeNoiseSubtractionMode");

  if ( mode == "Median")
    return std::unique_ptr<SiStripCommonModeNoiseSubtractor>(new MedianCMNSubtractor());

  if ( mode == "Percentile") {
    return std::unique_ptr<SiStripCommonModeNoiseSubtractor>(
        new PercentileCMNSubtractor(conf.getParameter<double>("Percentile")));
  }

  if ( mode == "IteratedMedian") {
    return std::unique_ptr<SiStripCommonModeNoiseSubtractor>(new IteratedMedianCMNSubtractor(
          conf.getParameter<double>("CutToAvoidSignal"), conf.getParameter<int>("Iterations")));
  }

  if ( mode == "FastLinear")
    return std::unique_ptr<SiStripCommonModeNoiseSubtractor>(new FastLinearCMNSubtractor());

  if ( mode == "TT6") {
    return std::unique_ptr<SiStripCommonModeNoiseSubtractor>(new TT6CMNSubtractor(
          conf.getParameter<double>("CutToAvoidSignal")));
  }

  edm::LogError("SiStripRawProcessingFactory::create_SubtractorCMN")
    << "Unregistered Algorithm: " << mode << ". Use one of {Median, Percentile, IteratedMedian, FastLinear, TT6}";
  return std::unique_ptr<SiStripCommonModeNoiseSubtractor>(new MedianCMNSubtractor());
}

std::unique_ptr<SiStripFedZeroSuppression>
SiStripRawProcessingFactory::create_Suppressor(const edm::ParameterSet& conf)
{
  const uint32_t mode = conf.getParameter<uint32_t>("SiStripFedZeroSuppressionMode");
  const bool trunc = conf.getParameter<bool>("TruncateInSuppressor");
  const bool trunc10bits = conf.getParameter<bool>("Use10bitsTruncation");
  switch (mode) {
  case 1: case 2: case 3:  case 4:
    return std::unique_ptr<SiStripFedZeroSuppression>(new SiStripFedZeroSuppression(mode, trunc, trunc10bits));
  default:
    edm::LogError("SiStripRawProcessingFactory::createSuppressor")
      << "Unregistered mode: " << mode << ". Use one of {1,2,3,4}.";
    return std::unique_ptr<SiStripFedZeroSuppression>(new SiStripFedZeroSuppression(4, true, trunc10bits));
  }
}

std::unique_ptr<SiStripAPVRestorer>
SiStripRawProcessingFactory::create_Restorer(const edm::ParameterSet& conf)
{
  if ( ! conf.exists("APVRestoreMode") ) {
    return std::unique_ptr<SiStripAPVRestorer>(nullptr);
  } else {
    return std::unique_ptr<SiStripAPVRestorer>(new SiStripAPVRestorer(conf));
  }
}
