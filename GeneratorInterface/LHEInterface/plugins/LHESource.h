#ifndef GeneratorInterface_LHEInterface_LHESource_h
#define GeneratorInterface_LHEInterface_LHESource_h

#include <memory>

#include <deque>

#include "DataFormats/Provenance/interface/ProcessHistoryID.h"
#include "GeneratorInterface/LHEInterface/plugins/LHEProvenanceHelper.h"
#include "FWCore/Sources/interface/ProducerSourceFromFiles.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"

#include "SimDataFormats/GeneratorProducts/interface/LesHouches.h"

namespace lhef {
  class LHERunInfo;
  class LHEEvent;
  class LHEReader;
}  // namespace lhef

namespace edm {
  class EventPrincipal;
  class LuminosityBlockAuxiliary;
  class LuminosityBlockPrincipal;
  class ParameterSet;
  class Run;
  class RunAuxiliary;
  class RunPrincipal;
}  // namespace edm

class LHERunInfoProduct;

class LHESource : public edm::ProducerSourceFromFiles {
public:
  explicit LHESource(const edm::ParameterSet& params, const edm::InputSourceDescription& desc);
  ~LHESource() override;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void endJob() override;
  bool setRunAndEventInfo(edm::EventID&, edm::TimeValue_t&, edm::EventAuxiliary::ExperimentType&) override;
  void readRun_(edm::RunPrincipal& runPrincipal) override;
  void readLuminosityBlock_(edm::LuminosityBlockPrincipal& lumiPrincipal) override;
  void readEvent_(edm::EventPrincipal& eventPrincipal) override;
  void produce(edm::Event&) override {}
  std::shared_ptr<edm::RunAuxiliary> readRunAuxiliary_() override;
  std::shared_ptr<edm::LuminosityBlockAuxiliary> readLuminosityBlockAuxiliary_() override;

  void nextEvent();

  void putRunInfoProduct(edm::RunPrincipal&);
  void fillRunInfoProduct(lhef::LHERunInfo const&, LHERunInfoProduct&);

  std::unique_ptr<lhef::LHEReader> reader_;

  std::shared_ptr<lhef::LHERunInfo> runInfoLast_;
  std::shared_ptr<lhef::LHEEvent> partonLevel_;

  std::unique_ptr<LHERunInfoProduct> runInfoProductLast_;
  edm::LHEProvenanceHelper lheProvenanceHelper_;
  edm::ProcessHistoryID phid_;
};

#endif  // GeneratorInterface_LHEInterface_LHESource_h
