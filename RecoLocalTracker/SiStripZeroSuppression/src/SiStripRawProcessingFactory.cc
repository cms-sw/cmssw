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


std::auto_ptr<SiStripRawProcessingAlgorithms> SiStripRawProcessingFactory::
create(const edm::ParameterSet& conf) {
  return std::auto_ptr<SiStripRawProcessingAlgorithms>(
	           new SiStripRawProcessingAlgorithms(
						      create_SubtractorPed(conf),
						      create_SubtractorCMN(conf),
						      create_Suppressor(conf),
                                                      create_Restorer(conf),
						      create_doAPVRestorer(conf),
						      create_useCMMeanMap(conf)));
}

bool SiStripRawProcessingFactory::create_doAPVRestorer(const edm::ParameterSet& conf){
   bool doAPVRestore = conf.getParameter<bool>("doAPVRestore");
   return doAPVRestore; 
}
  
bool SiStripRawProcessingFactory::create_useCMMeanMap(const edm::ParameterSet&conf){
   bool useCMMeanMap = conf.getParameter<bool>("useCMMeanMap");
   return useCMMeanMap; 
}

std::auto_ptr<SiStripPedestalsSubtractor> SiStripRawProcessingFactory::
create_SubtractorPed(const edm::ParameterSet& conf) {
  bool fedMode = conf.getParameter<bool>("PedestalSubtractionFedMode");
  return std::auto_ptr<SiStripPedestalsSubtractor>( new SiStripPedestalsSubtractor(fedMode) );
}

std::auto_ptr<SiStripCommonModeNoiseSubtractor> SiStripRawProcessingFactory::
create_SubtractorCMN(const edm::ParameterSet& conf) {
  std::string mode = conf.getParameter<std::string>("CommonModeNoiseSubtractionMode");

  if ( mode == "Median")
    return std::auto_ptr<SiStripCommonModeNoiseSubtractor>( new MedianCMNSubtractor() );

  if ( mode == "Percentile") {
    double percentile = conf.getParameter<double>("Percentile");
    return std::auto_ptr<SiStripCommonModeNoiseSubtractor>( new PercentileCMNSubtractor(percentile) );
  }

  if ( mode == "IteratedMedian") {
    double cutToAvoidSignal = conf.getParameter<double>("CutToAvoidSignal");
    int iterations = conf.getParameter<int>("Iterations");
    return std::auto_ptr<SiStripCommonModeNoiseSubtractor>( new IteratedMedianCMNSubtractor(cutToAvoidSignal,iterations) );
  }

  if ( mode == "FastLinear")
    return std::auto_ptr<SiStripCommonModeNoiseSubtractor>( new FastLinearCMNSubtractor() );

  if ( mode == "TT6") {
    double cutToAvoidSignal = conf.getParameter<double>("CutToAvoidSignal");
    return std::auto_ptr<SiStripCommonModeNoiseSubtractor>( new TT6CMNSubtractor(cutToAvoidSignal) );
  }
  
  edm::LogError("SiStripRawProcessingFactory::create_SubtractorCMN")
    << "Unregistered Algorithm: "<<mode<<". Use one of {Median, Percentile, IteratedMedian, FastLinear, TT6}";
  return std::auto_ptr<SiStripCommonModeNoiseSubtractor>( new MedianCMNSubtractor() );
}

std::auto_ptr<SiStripFedZeroSuppression> SiStripRawProcessingFactory::
create_Suppressor(const edm::ParameterSet& conf) {
  uint32_t mode = conf.getParameter<uint32_t>("SiStripFedZeroSuppressionMode");
  bool trunc = conf.getParameter<bool>("TruncateInSuppressor");
  switch(mode) {
  case 1: case 2: case 3:  case 4:
    return std::auto_ptr<SiStripFedZeroSuppression>( new SiStripFedZeroSuppression(mode,trunc));
  default:
    edm::LogError("SiStripRawProcessingFactory::createSuppressor")
      << "Unregistered mode: "<<mode<<". Use one of {1,2,3,4}.";
    return std::auto_ptr<SiStripFedZeroSuppression>( new SiStripFedZeroSuppression(4,true));
  }
}

std::auto_ptr<SiStripAPVRestorer> SiStripRawProcessingFactory::
create_Restorer( const edm::ParameterSet& conf) {
  if(!conf.exists("APVRestoreMode")) {
    return std::auto_ptr<SiStripAPVRestorer>( 0 );
  } else {
    return std::auto_ptr<SiStripAPVRestorer> (new SiStripAPVRestorer(conf));
  }
}

