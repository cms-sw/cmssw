#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/InputSourceMacros.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Framework/interface/InputSource.h"
#include "FWCore/Sources/interface/IDGeneratorSourceBase.h"

namespace edm {
  class EmptySource : public IDGeneratorSourceBase<InputSource> {
  public:
    explicit EmptySource(ParameterSet const&, InputSourceDescription const&);
    ~EmptySource() override;
    static void fillDescriptions(ConfigurationDescriptions& descriptions);

  private:
    bool setRunAndEventInfo(EventID& id, TimeValue_t& time, edm::EventAuxiliary::ExperimentType&) override;
    void readEvent_(edm::EventPrincipal&) override;
  };

  EmptySource::EmptySource(ParameterSet const& pset, InputSourceDescription const& desc)
      : IDGeneratorSourceBase<InputSource>(pset, desc, false) {}

  EmptySource::~EmptySource() {}

  bool EmptySource::setRunAndEventInfo(EventID&, TimeValue_t&, edm::EventAuxiliary::ExperimentType&) { return true; }

  void EmptySource::readEvent_(edm::EventPrincipal& e) {
    doReadEvent(e, [](auto const&) {});
  }

  void EmptySource::fillDescriptions(ConfigurationDescriptions& descriptions) {
    ParameterSetDescription desc;
    desc.setComment("Creates runs, lumis and events containing no products.");
    IDGeneratorSourceBase<InputSource>::fillDescription(desc);
    descriptions.add("source", desc);
  }
}  // namespace edm

using edm::EmptySource;
DEFINE_FWK_INPUT_SOURCE(EmptySource);
