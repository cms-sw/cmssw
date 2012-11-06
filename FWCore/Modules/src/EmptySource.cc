#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/GeneratedInputSource.h"
#include "FWCore/Framework/interface/InputSourceMacros.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

namespace edm {
  class EmptySource : public GeneratedInputSource {
  public:
    explicit EmptySource(ParameterSet const&, InputSourceDescription const&);
    ~EmptySource();
    static void fillDescriptions(ConfigurationDescriptions& descriptions);
  private:
    virtual bool produce(Event &);
  };

  EmptySource::EmptySource(ParameterSet const& pset,
				       InputSourceDescription const& desc) :
    GeneratedInputSource(pset, desc)
  { }

  EmptySource::~EmptySource() {
  }

  bool
  EmptySource::produce(edm::Event &) {
    return true;
  }

  void
  EmptySource::fillDescriptions(ConfigurationDescriptions& descriptions) {
    ParameterSetDescription desc;
    desc.setComment("Creates runs, lumis and events containing no products.");
    GeneratedInputSource::fillDescription(desc);
    descriptions.add("source", desc);
  }
}

using edm::EmptySource;
DEFINE_FWK_INPUT_SOURCE(EmptySource);
