// user include files
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"

#include "FWCore/Framework/interface/ESHandle.h"

#include "RecoPixelVertexing/PixelLowPtUtilities/interface/TrackCleaner.h"

class TrackCleanerESProducer: public edm::ESProducer {
public:
  TrackCleanerESProducer(const edm::ParameterSet& iConfig);
  ~TrackCleanerESProducer() = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  std::unique_ptr<PixelTrackCleaner> produce(const PixelTrackCleaner::Record& iRecord);
};

TrackCleanerESProducer::TrackCleanerESProducer(const edm::ParameterSet& iConfig) {
  auto componentName = iConfig.getParameter<std::string>("ComponentName");
  setWhatProduced(this, componentName);
}

void TrackCleanerESProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<std::string>("ComponentName", "trackCleaner");
  descriptions.add("trackCleaner", desc);
}

std::unique_ptr<PixelTrackCleaner> TrackCleanerESProducer::produce(const PixelTrackCleaner::Record& iRecord) {
  return std::make_unique<TrackCleaner>();
}

#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_EVENTSETUP_MODULE(TrackCleanerESProducer);
