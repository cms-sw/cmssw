#include "RecoLocalTracker/SiStripZeroSuppression/interface/SiStripRawProcessingFactory.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "RecoLocalTracker/SiStripZeroSuppression/interface/SiStripRawProcessingAlgorithms.h"
#include "RecoLocalTracker/SiStripZeroSuppression/interface/SiStripPedestalsSubtractor.h"
#include "RecoLocalTracker/SiStripZeroSuppression/interface/SiStripFedZeroSuppression.h"
#include "RecoLocalTracker/SiStripZeroSuppression/interface/MedianCMNSubtractor.h"
#include "RecoLocalTracker/SiStripZeroSuppression/interface/FastLinearCMNSubtractor.h"
#include "RecoLocalTracker/SiStripZeroSuppression/interface/TT6CMNSubtractor.h"

std::auto_ptr<SiStripRawProcessingAlgorithms> SiStripRawProcessingFactory::
create(const edm::ParameterSet& conf) {
  return std::auto_ptr<SiStripRawProcessingAlgorithms>(
	           new SiStripRawProcessingAlgorithms(
						      create_SubtractorPed(conf),
						      create_SubtractorCMN(conf),
						      create_Suppressor(conf)     ));
}

std::auto_ptr<SiStripPedestalsSubtractor> SiStripRawProcessingFactory::
create_SubtractorPed(const edm::ParameterSet& conf) {
  return std::auto_ptr<SiStripPedestalsSubtractor>( new SiStripPedestalsSubtractor() );
}

std::auto_ptr<SiStripCommonModeNoiseSubtractor> SiStripRawProcessingFactory::
create_SubtractorCMN(const edm::ParameterSet& conf) {
  std::string mode = conf.getParameter<std::string>("CommonModeNoiseSubtractionMode");

  if ( mode == "Median")
    return std::auto_ptr<SiStripCommonModeNoiseSubtractor>( new MedianCMNSubtractor() );

  if ( mode == "FastLinear")
    return std::auto_ptr<SiStripCommonModeNoiseSubtractor>( new FastLinearCMNSubtractor() );

  if ( mode == "TT6") {
    double cutToAvoidSignal = conf.getParameter<double>("CutToAvoidSignal");
    return std::auto_ptr<SiStripCommonModeNoiseSubtractor>( new TT6CMNSubtractor(cutToAvoidSignal) );
  }
  
  edm::LogError("SiStripRawProcessingFactory::create_SubtractorCMN")
    << "Unregistered Algorithm: "<<mode<<". Use one of {Median, FastLinear, TT6}";
  return std::auto_ptr<SiStripCommonModeNoiseSubtractor>( new MedianCMNSubtractor() );
}

std::auto_ptr<SiStripFedZeroSuppression> SiStripRawProcessingFactory::
create_Suppressor(const edm::ParameterSet& conf) {
  uint32_t mode = conf.getParameter<uint32_t>("SiStripFedZeroSuppressionMode");
  switch(mode) {
  case 1: case 2: case 3:  case 4:
    return std::auto_ptr<SiStripFedZeroSuppression>( new SiStripFedZeroSuppression(mode));
  default:
    edm::LogError("SiStripRawProcessingFactory::createSuppressor")
      << "Unregistered mode: "<<mode<<". Use one of {1,2,3,4}.";
    return std::auto_ptr<SiStripFedZeroSuppression>( new SiStripFedZeroSuppression(4));
  }
}

