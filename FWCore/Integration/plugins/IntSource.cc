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
    ~IntSource() override;
    static void fillDescriptions(ConfigurationDescriptions& descriptions);

  private:
    bool setRunAndEventInfo(EventID& id, TimeValue_t& time, edm::EventAuxiliary::ExperimentType& eType) override;
    void produce(Event&) override;
  };

  IntSource::IntSource(ParameterSet const& pset, InputSourceDescription const& desc)
      : ProducerSourceBase(pset, desc, false) {
    produces<edmtest::IntProduct>();
  }

  IntSource::~IntSource() {}

  bool IntSource::setRunAndEventInfo(EventID&, TimeValue_t&, edm::EventAuxiliary::ExperimentType&) { return true; }

  void IntSource::produce(edm::Event& e) { e.put(std::make_unique<edmtest::IntProduct>(4)); }

  void IntSource::fillDescriptions(ConfigurationDescriptions& descriptions) {
    ParameterSetDescription desc;
    ProducerSourceBase::fillDescription(desc);
    descriptions.add("source", desc);
  }
}  // namespace edm
using edm::IntSource;
DEFINE_FWK_INPUT_SOURCE(IntSource);
