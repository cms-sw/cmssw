#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ESProducer.h"
#include "Geometry/MTDNumberingBuilder/interface/MTDTopology.h"
#include "Geometry/Records/interface/MTDTopologyRcd.h"
#include "CondFormats/GeometryObjects/interface/PMTDParameters.h"
#include "Geometry/Records/interface/PMTDParametersRcd.h"

#include <memory>
//#define EDM_ML_DEBUG

class MTDTopologyEP : public edm::ESProducer {
public:
  MTDTopologyEP(const edm::ParameterSet&);

  using ReturnType = std::unique_ptr<MTDTopology>;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  ReturnType produce(const MTDTopologyRcd&);

private:
  void fillParameters(const PMTDParameters&, int& mtdTopologyMode);

  const edm::ESGetToken<PMTDParameters, PMTDParametersRcd> token_;
};

MTDTopologyEP::MTDTopologyEP(const edm::ParameterSet& conf)
    : token_{setWhatProduced(this).consumesFrom<PMTDParameters, PMTDParametersRcd>(edm::ESInputTag())} {}

void MTDTopologyEP::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription ttc;
  descriptions.add("mtdTopology", ttc);
}

MTDTopologyEP::ReturnType MTDTopologyEP::produce(const MTDTopologyRcd& iRecord) {
  int mtdTopologyMode;

  fillParameters(iRecord.get(token_), mtdTopologyMode);

  return std::make_unique<MTDTopology>(mtdTopologyMode);
}

void MTDTopologyEP::fillParameters(const PMTDParameters& ptp, int& mtdTopologyMode) {
  mtdTopologyMode = ptp.topologyMode_;

#ifdef EDM_ML_DEBUG

  edm::LogInfo("MTDTopologyEP") << "Topology mode = " << mtdTopologyMode;

#endif
}

DEFINE_FWK_EVENTSETUP_MODULE(MTDTopologyEP);
