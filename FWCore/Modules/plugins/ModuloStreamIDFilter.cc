
#include "FWCore/Framework/interface/global/EDFilter.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

namespace edm {
  class ModuloStreamIDFilter : public global::EDFilter<> {
  public:
    explicit ModuloStreamIDFilter(ParameterSet const&);
    ~ModuloStreamIDFilter() override;

    static void fillDescriptions(ConfigurationDescriptions& descriptions);
    bool filter(StreamID, Event& e, EventSetup const& c) const final;

  private:
    const unsigned int n_;       // accept one in n
    const unsigned int offset_;  // with offset, ie. sequence of events does not have to start at first event
  };

  ModuloStreamIDFilter::ModuloStreamIDFilter(ParameterSet const& ps)
      : n_(ps.getParameter<unsigned int>("modulo")), offset_(ps.getParameter<unsigned int>("offset")) {}

  ModuloStreamIDFilter::~ModuloStreamIDFilter() {}

  bool ModuloStreamIDFilter::filter(StreamID iStreamID, Event&, EventSetup const&) const {
    return (iStreamID.value() % n_ == offset_);
  }

  void ModuloStreamIDFilter::fillDescriptions(ConfigurationDescriptions& descriptions) {
    ParameterSetDescription desc;
    desc.add<unsigned int>("modulo")->setComment("Accept event if (streamID % modulo) == offset.");
    desc.add<unsigned int>("offset")->setComment("Used to shift which value of modulo to accept.");
    descriptions.add("streamIDFilter", desc);
  }
}  // namespace edm

using edm::ModuloStreamIDFilter;
DEFINE_FWK_MODULE(ModuloStreamIDFilter);
