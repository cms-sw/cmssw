#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/InputSourceMacros.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Sources/interface/ProducerSourceBase.h"

namespace edm {
  class EmptySource : public ProducerSourceBase {
  public:
    explicit EmptySource(ParameterSet const&, InputSourceDescription const&);
    ~EmptySource();
    static void fillDescriptions(ConfigurationDescriptions& descriptions);
  private:
    virtual bool setRunAndEventInfo(EventID& id, TimeValue_t& time) override;
    virtual void produce(Event &) override;
  };

  EmptySource::EmptySource(ParameterSet const& pset,
				       InputSourceDescription const& desc) :
    ProducerSourceBase(pset, desc, false)
  { }

  EmptySource::~EmptySource() {
  }

  bool
  EmptySource::setRunAndEventInfo(EventID&, TimeValue_t&) {
    return true;
  }

  void
  EmptySource::produce(edm::Event&) {
  }

  void
  EmptySource::fillDescriptions(ConfigurationDescriptions& descriptions) {
    ParameterSetDescription desc;
    desc.setComment("Creates runs, lumis and events containing no products.");
    ProducerSourceBase::fillDescription(desc);
    descriptions.add("source", desc);
  }
}

using edm::EmptySource;
DEFINE_FWK_INPUT_SOURCE(EmptySource);
