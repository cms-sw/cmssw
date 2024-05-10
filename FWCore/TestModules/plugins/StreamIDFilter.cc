#include "FWCore/Framework/interface/global/EDFilter.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

namespace edmtest {
  class StreamIDFilter : public edm::global::EDFilter<> {
  public:
    explicit StreamIDFilter(edm::ParameterSet const&);

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);
    bool filter(edm::StreamID, edm::Event& e, edm::EventSetup const& c) const final;

  private:
    std::vector<unsigned int> rejectStreams_;
  };

  StreamIDFilter::StreamIDFilter(edm::ParameterSet const& ps)
      : rejectStreams_(ps.getParameter<std::vector<unsigned int>>("rejectStreams")) {}

  bool StreamIDFilter::filter(edm::StreamID id, edm::Event&, edm::EventSetup const&) const {
    return std::find(rejectStreams_.begin(), rejectStreams_.end(), id) == rejectStreams_.end();
  }

  void StreamIDFilter::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;
    desc.add<std::vector<unsigned int>>("rejectStreams")
        ->setComment("Stream IDs for which to reject events. If empty, all events are accepted.");
    descriptions.addDefault(desc);
  }
}  // namespace edmtest

DEFINE_FWK_MODULE(edmtest::StreamIDFilter);
