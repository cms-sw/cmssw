#include "IOPool/Streamer/interface/StreamerOutputModuleCommon.h"

#include "IOPool/Streamer/interface/InitMsgBuilder.h"
#include "IOPool/Streamer/interface/EventMsgBuilder.h"
#include "FWCore/Framework/interface/EventForOutput.h"
#include "FWCore/Framework/interface/EventSelector.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/DebugMacros.h"
#include "FWCore/Version/interface/GetReleaseVersion.h"
#include "DataFormats/Common/interface/TriggerResults.h"
#include "DataFormats/Provenance/interface/ModuleDescription.h"
#include "DataFormats/Provenance/interface/ParameterSetID.h"
#include "DataFormats/Provenance/interface/SelectedProducts.h"
#include "FWCore/Framework/interface/getAllTriggerNames.h"

#include <iostream>
#include <memory>
#include <string>
#include <sys/time.h>
#include <unistd.h>
#include <vector>
#include <zlib.h>

namespace edm::streamer {
  StreamerOutputModuleCommon::Parameters StreamerOutputModuleCommon::parameters(ParameterSet const& ps) {
    return StreamerOutputMsgBuilders::parameters(ps);
  }

  StreamerOutputModuleCommon::StreamerOutputModuleCommon(Parameters const& p,
                                                         SelectedProducts const* selections,
                                                         std::string const& moduleLabel)
      : builders_(p, selections, moduleLabel) {}

  StreamerOutputModuleCommon::~StreamerOutputModuleCommon() {}

  std::unique_ptr<InitMsgBuilder> StreamerOutputModuleCommon::serializeRegistry(
      std::string const& processName,
      std::string const& moduleLabel,
      ParameterSetID const& toplevel,
      SendJobHeader::ParameterSetMap const* psetMap) {
    return builders_.serializeRegistry(buffer_, processName, moduleLabel, toplevel, psetMap);
  }

  std::unique_ptr<EventMsgBuilder> StreamerOutputModuleCommon::serializeEvent(
      EventForOutput const& e, Handle<TriggerResults> const& triggerResults, ParameterSetID const& selectorCfg) {
    return builders_.serializeEvent(buffer_, e, triggerResults, selectorCfg, eventMetaDataChecksum_);
  }

  std::unique_ptr<EventMsgBuilder> StreamerOutputModuleCommon::serializeEventMetaData(
      BranchIDLists const& branchLists, ThinnedAssociationsHelper const& helper) {
    auto ret = builders_.serializeEventMetaData(buffer_, branchLists, helper);
    eventMetaDataChecksum_ = ret.second;
    return std::move(ret.first);
  }

  void StreamerOutputModuleCommon::fillDescription(ParameterSetDescription& desc) {
    StreamerOutputMsgBuilders::fillDescription(desc);
  }

}  // namespace edm::streamer
