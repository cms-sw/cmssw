#include "FWCore/Framework/interface/ModuleFactory.h"

#include "CalibMuon/CSCCalibration/interface/CSCChannelMapperESProducer.h"
#include "CalibMuon/CSCCalibration/interface/CSCChannelMapperFactory.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

CSCChannelMapperESProducer::CSCChannelMapperESProducer(const edm::ParameterSet& pset)
{
  algoName = pset.getParameter<std::string>("AlgoName");

  LogTrace("CSCChannelMapperESProducer") << " will produce: " << algoName;

  setWhatProduced(this);

}

CSCChannelMapperESProducer::~CSCChannelMapperESProducer(){
}

CSCChannelMapperESProducer::BSP_TYPE CSCChannelMapperESProducer::produce(const CSCChannelMapperRecord& )
{
  LogTrace("CSCChannelMapperESProducer") << " producing: " << algoName;

  CSCChannelMapperESProducer::BSP_TYPE theChannelMapper(CSCChannelMapperFactory::get()->create(algoName));

  return theChannelMapper ;
}

// define this as a plug-in
DEFINE_FWK_EVENTSETUP_MODULE(CSCChannelMapperESProducer);
