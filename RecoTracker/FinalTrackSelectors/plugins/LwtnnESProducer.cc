#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"

#include "TrackingTools/Records/interface/TrackingComponentsRecord.h"

//from lwtnn
#include "lwtnn/LightweightNeuralNetwork.hh"
#include "lwtnn/parse_json.hh"
#include <fstream>

class LwtnnESProducer : public edm::ESProducer {
public:
  LwtnnESProducer(const edm::ParameterSet& iConfig);
  ~LwtnnESProducer() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  // TODO: Use of TrackingComponentsRecord is as inadequate as the
  // placement of this ESProducer in RecoTracker/FinalTrackSelectors
  // (but it works, I tried to create a new record but for some reason
  // did not get it to work). Especially if this producer gets used
  // wider we should figure out a better record and package.
  std::unique_ptr<lwt::LightweightNeuralNetwork> produce(const TrackingComponentsRecord& iRecord);

private:
  edm::FileInPath fileName_;
};

LwtnnESProducer::LwtnnESProducer(const edm::ParameterSet& iConfig)
    : fileName_(iConfig.getParameter<edm::FileInPath>("fileName")) {
  auto componentName = iConfig.getParameter<std::string>("ComponentName");
  setWhatProduced(this, componentName);
}

void LwtnnESProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<std::string>("ComponentName", "lwtnnESProducer");
  desc.add<edm::FileInPath>("fileName", edm::FileInPath());
  descriptions.add("lwtnnESProducer", desc);
}

std::unique_ptr<lwt::LightweightNeuralNetwork> LwtnnESProducer::produce(const TrackingComponentsRecord& iRecord) {
  std::ifstream jsonfile(fileName_.fullPath().c_str());
  auto config = lwt::parse_json(jsonfile);
  return std::make_unique<lwt::LightweightNeuralNetwork>(config.inputs, config.layers, config.outputs);
}

#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_EVENTSETUP_MODULE(LwtnnESProducer);
