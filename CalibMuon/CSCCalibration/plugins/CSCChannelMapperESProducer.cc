#include "FWCore/Framework/interface/ModuleFactory.h"
#include "CalibMuon/CSCCalibration/interface/CSCChannelMapperESProducer.h"
#include "CalibMuon/CSCCalibration/interface/CSCChannelMapperFactory.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

CSCChannelMapperESProducer::CSCChannelMapperESProducer(const edm::ParameterSet &pset) {
  algoName = pset.getParameter<std::string>("AlgoName");

  LogTrace("CSCChannelMapperESProducer") << " will produce: " << algoName;

  setWhatProduced(this);
}

CSCChannelMapperESProducer::~CSCChannelMapperESProducer() {}

CSCChannelMapperESProducer::BSP_TYPE CSCChannelMapperESProducer::produce(const CSCChannelMapperRecord &) {
  LogTrace("CSCChannelMapperESProducer") << " producing: " << algoName;

  return CSCChannelMapperESProducer::BSP_TYPE(CSCChannelMapperFactory::get()->create(algoName));
}

// ---- add this ----
void CSCChannelMapperESProducer::fillDescriptions(edm::ConfigurationDescriptions &descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<std::string>("AlgoName", "CSCChannelMapperStartup");  // default
  descriptions.addWithDefaultLabel(desc);
}

// define this as a plug-in
DEFINE_FWK_EVENTSETUP_MODULE(CSCChannelMapperESProducer);
