
#include "FWCore/Framework/interface/global/EDFilter.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

namespace {

  template <typename T>
  std::vector<T> sorted(std::vector<T> v) {
    std::sort(v.begin(), v.end());
    return v;
  }

}  // namespace

namespace edm {

  class BunchCrossingFilter : public global::EDFilter<> {
  public:
    explicit BunchCrossingFilter(ParameterSet const& config);

    static void fillDescriptions(ConfigurationDescriptions& descriptions);
    bool filter(StreamID, Event& event, EventSetup const&) const final;

  private:
    const std::vector<unsigned int> bunches_;
  };

  BunchCrossingFilter::BunchCrossingFilter(ParameterSet const& config)
      : bunches_(sorted(config.getParameter<std::vector<unsigned int>>("bunches"))) {}

  bool BunchCrossingFilter::filter(StreamID, Event& event, EventSetup const&) const {
    return std::binary_search(bunches_.begin(), bunches_.end(), event.bunchCrossing());
  }

  void BunchCrossingFilter::fillDescriptions(ConfigurationDescriptions& descriptions) {
    ParameterSetDescription desc;
    desc.add<std::vector<unsigned int>>("bunches", {})
        ->setComment("List of bunch crossings for which events should be accepted [1-3564].");
    descriptions.add("bunchCrossingFilter", desc);
  }

}  // namespace edm

using edm::BunchCrossingFilter;
DEFINE_FWK_MODULE(BunchCrossingFilter);
