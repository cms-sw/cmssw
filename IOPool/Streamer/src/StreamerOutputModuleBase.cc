
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

namespace edm::streamer {
  StreamerOutputModuleBase::StreamerOutputModuleBase(ParameterSet const& ps)
      : one::OutputModuleBase::OutputModuleBase(ps),
        one::OutputModule<one::WatchRuns, one::WatchLuminosityBlocks>(ps),
        StreamerOutputModuleCommon(ps, &keptProducts()[InEvent], description().moduleLabel()),
        trToken_(consumes<edm::TriggerResults>(edm::InputTag("TriggerResults"))),
        psetToken_(
            consumes<SendJobHeader::ParameterSetMap, edm::InRun>(ps.getUntrackedParameter<edm::InputTag>("psetMap"))) {}

  StreamerOutputModuleBase::~StreamerOutputModuleBase() {}

  void StreamerOutputModuleBase::beginRun(RunForOutput const& iRun) {
    start();

    if (not initWritten_) {
      auto psetMapHandle = iRun.getHandle(psetToken_);

      std::unique_ptr<InitMsgBuilder> init_message =
          serializeRegistry(OutputModule::processName(),
                            description().moduleLabel(),
                            moduleDescription().mainParameterSetID(),
                            psetMapHandle.isValid() ? psetMapHandle.product() : nullptr);

      doOutputHeader(*init_message);
      lastCallWasBeginRun_ = true;
      auto history = iRun.processHistory();
      lastHistory_ = history.reduce().id();
      initWritten_ = true;

      clearHeaderBuffer();
    } else {
      auto history = iRun.processHistory();
      if (lastHistory_ != history.reduce().id()) {
        throw edm::Exception(errors::FileWriteError) << "Streamer output can not handle writing a new Run if the "
                                                        "ProcessHistory changed since the last Run written.";
      }
      //need to write meta data anyway
      lastCallWasBeginRun_ = true;
    }
  }

  void StreamerOutputModuleBase::endRun(RunForOutput const&) { stop(); }

  void StreamerOutputModuleBase::beginJob() {}

  void StreamerOutputModuleBase::endJob() { stop(); }

  void StreamerOutputModuleBase::writeRun(RunForOutput const&) {}

  void StreamerOutputModuleBase::writeLuminosityBlock(LuminosityBlockForOutput const&) {}

  void StreamerOutputModuleBase::write(EventForOutput const& e) {
    Handle<TriggerResults> const& triggerResults = getTriggerResults(trToken_, e);

    if (lastCallWasBeginRun_) {
      auto msg = serializeEventMetaData(*branchIDLists(), *thinnedAssociationsHelper());
      doOutputEvent(*msg);
      lastCallWasBeginRun_ = false;
    }
    auto msg = serializeEvent(e, triggerResults, selectorConfig());

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
    desc.addUntracked<edm::InputTag>("psetMap", {"hltPSetMap"})
        ->setComment("Optionally allow the map of ParameterSets to be calculated externally.");
  }
}  // namespace edm::streamer
