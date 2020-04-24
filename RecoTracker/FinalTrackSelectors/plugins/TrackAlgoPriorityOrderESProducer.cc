#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "RecoTracker/FinalTrackSelectors/interface/TrackAlgoPriorityOrder.h"

#include "RecoTracker/Record/interface/CkfComponentsRecord.h"

class TrackAlgoPriorityOrderESProducer: public edm::ESProducer {
public:
  TrackAlgoPriorityOrderESProducer(const edm::ParameterSet& iConfig);
  ~TrackAlgoPriorityOrderESProducer() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  std::unique_ptr<TrackAlgoPriorityOrder> produce(const CkfComponentsRecord& iRecord);

private:
  std::vector<reco::TrackBase::TrackAlgorithm> algoOrder_;
};

TrackAlgoPriorityOrderESProducer::TrackAlgoPriorityOrderESProducer(const edm::ParameterSet& iConfig) {
  const auto& algoNames = iConfig.getParameter<std::vector<std::string> >("algoOrder");
  algoOrder_.reserve(algoNames.size());
  for(const auto& name: algoNames) {
    auto algo = reco::TrackBase::algoByName(name);
    if(algo == reco::TrackBase::undefAlgorithm && name != "undefAlgorithm") {
      throw cms::Exception("Configuration") << "Incorrect track algo " << name;
    }
    algoOrder_.push_back(algo);
  }

  auto componentName = iConfig.getParameter<std::string>("ComponentName");
  setWhatProduced(this, componentName);
}

void TrackAlgoPriorityOrderESProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<std::string>("ComponentName", "trackAlgoPriorityOrder");
  desc.add<std::vector<std::string> >("algoOrder", std::vector<std::string>());
  descriptions.add("trackAlgoPriorityOrderDefault", desc);
}

std::unique_ptr<TrackAlgoPriorityOrder> TrackAlgoPriorityOrderESProducer::produce(const CkfComponentsRecord& iRecord) {
  return std::make_unique<TrackAlgoPriorityOrder>(algoOrder_);
}

#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_EVENTSETUP_MODULE(TrackAlgoPriorityOrderESProducer);
