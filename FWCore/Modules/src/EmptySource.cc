#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/GeneratedInputSource.h"
#include "FWCore/Framework/interface/InputSourceMacros.h"

namespace edm {
  class EmptySource : public GeneratedInputSource {
  public:
    explicit EmptySource(ParameterSet const&, InputSourceDescription const&);
    ~EmptySource();
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
}

using edm::EmptySource;
DEFINE_FWK_INPUT_SOURCE(EmptySource);
