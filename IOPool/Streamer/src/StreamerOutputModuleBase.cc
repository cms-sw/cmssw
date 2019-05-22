
#include "IOPool/Streamer/interface/StreamerOutputModuleBase.h"

#include "IOPool/Streamer/interface/InitMsgBuilder.h"
#include "IOPool/Streamer/interface/EventMsgBuilder.h"
#include "FWCore/Framework/interface/EventForOutput.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "DataFormats/Common/interface/TriggerResults.h"
#include "DataFormats/Provenance/interface/ModuleDescription.h"
#include "DataFormats/Provenance/interface/ParameterSetID.h"

#include "zlib.h"

namespace edm {
  StreamerOutputModuleBase::StreamerOutputModuleBase(ParameterSet const& ps)
      : one::OutputModuleBase::OutputModuleBase(ps),
        one::OutputModule<one::WatchRuns, one::WatchLuminosityBlocks>(ps),
        StreamerOutputModuleCommon(ps, &keptProducts()[InEvent]),
        trToken_(consumes<edm::TriggerResults>(edm::InputTag("TriggerResults"))) {}

  StreamerOutputModuleBase::~StreamerOutputModuleBase() {}

  void StreamerOutputModuleBase::beginRun(RunForOutput const&) {
    start();

    std::unique_ptr<InitMsgBuilder> init_message = serializeRegistry(*branchIDLists(),
                                                                     *thinnedAssociationsHelper(),
                                                                     OutputModule::processName(),
                                                                     description().moduleLabel(),
                                                                     moduleDescription().mainParameterSetID());

    doOutputHeader(*init_message);
    clearSerializeDataBuffer();
  }

  void StreamerOutputModuleBase::endRun(RunForOutput const&) { stop(); }

  void StreamerOutputModuleBase::beginJob() {}

  void StreamerOutputModuleBase::endJob() { stop(); }

  void StreamerOutputModuleBase::writeRun(RunForOutput const&) {}

  void StreamerOutputModuleBase::writeLuminosityBlock(LuminosityBlockForOutput const&) {}

  void StreamerOutputModuleBase::write(EventForOutput const& e) {
    Handle<TriggerResults> const& triggerResults = getTriggerResults(trToken_, e);
    std::unique_ptr<EventMsgBuilder> msg = serializeEvent(e, triggerResults, selectorConfig());
    doOutputEvent(*msg);  // You can't use msg in StreamerOutputModuleBase after this point
  }

  Trig StreamerOutputModuleBase::getTriggerResults(EDGetTokenT<TriggerResults> const& token,
                                                   EventForOutput const& e) const {
    Trig result;
    e.getByToken<TriggerResults>(token, result);
    return result;
  }

  void StreamerOutputModuleBase::fillDescription(ParameterSetDescription& desc) {
    StreamerOutputModuleCommon::fillDescription(desc);
    OutputModule::fillDescription(desc);
  }
}  // namespace edm
