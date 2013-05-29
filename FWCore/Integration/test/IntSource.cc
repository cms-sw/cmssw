#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/InputSourceMacros.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Sources/interface/ProducerSourceBase.h"
#include "DataFormats/TestObjects/interface/ToyProducts.h"
#include "FWCore/Framework/interface/Event.h"

#include <memory>

namespace edm {

  class IntSource : public ProducerSourceBase {
  public:
    explicit IntSource(ParameterSet const&, InputSourceDescription const&);
    ~IntSource();
    static void fillDescriptions(ConfigurationDescriptions& descriptions);
  private:
    virtual bool setRunAndEventInfo(EventID& id, TimeValue_t& time);
    virtual void produce(Event &);
  };

  IntSource::IntSource(ParameterSet const& pset,
                                       InputSourceDescription const& desc) :
    ProducerSourceBase(pset, desc, false)
  { produces<edmtest::IntProduct>(); }

  IntSource::~IntSource() {
  }

  bool
  IntSource::setRunAndEventInfo(EventID&, TimeValue_t&) {
    return true;
  }

  void
  IntSource::produce(edm::Event& e) {
    std::auto_ptr<edmtest::IntProduct> p(new edmtest::IntProduct(4));
    e.put(p);
  }

  void
  IntSource::fillDescriptions(ConfigurationDescriptions& descriptions) {
    ParameterSetDescription desc;
    ProducerSourceBase::fillDescription(desc);
    descriptions.add("source", desc);
  }
}
using edm::IntSource;
DEFINE_FWK_INPUT_SOURCE(IntSource);
