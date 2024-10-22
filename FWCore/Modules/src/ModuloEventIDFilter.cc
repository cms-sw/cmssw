
#include "FWCore/Framework/interface/global/EDFilter.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

namespace edm {
  class ModuloEventIDFilter : public global::EDFilter<> {
  public:
    explicit ModuloEventIDFilter(ParameterSet const&);

    static void fillDescriptions(ConfigurationDescriptions& descriptions);
    bool filter(StreamID, Event& e, EventSetup const& c) const final;

  private:
    const unsigned int n_;       // accept one in n
    const unsigned int offset_;  // with offset, ie. sequence of events does not have to start at first event
  };

  ModuloEventIDFilter::ModuloEventIDFilter(ParameterSet const& ps)
      : n_(ps.getParameter<unsigned int>("modulo")), offset_(ps.getParameter<unsigned int>("offset")) {}

  bool ModuloEventIDFilter::filter(StreamID, Event& iEvent, EventSetup const&) const {
    return (iEvent.id().event() % n_ == offset_);
  }

  void ModuloEventIDFilter::fillDescriptions(ConfigurationDescriptions& descriptions) {
    ParameterSetDescription desc;
    desc.add<unsigned int>("modulo")->setComment("Accept event if (eventID % modulo) == offset.");
    desc.add<unsigned int>("offset")->setComment("Used to shift which value of modulo to accept.");
    descriptions.add("eventIDFilter", desc);
  }
}  // namespace edm

using edm::ModuloEventIDFilter;
DEFINE_FWK_MODULE(ModuloEventIDFilter);
