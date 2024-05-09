#ifndef IOPool_Streamer_StreamerOutputModuleCommon_h
#define IOPool_Streamer_StreamerOutputModuleCommon_h

#include "IOPool/Streamer/interface/MsgTools.h"
#include "FWCore/Common/interface/TriggerNames.h"
#include "IOPool/Streamer/interface/StreamerOutputMsgBuilders.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Streamer/interface/StreamedProducts.h"
#include <memory>
#include <vector>

namespace edm {
  class ParameterSet;
  class ParameterSetDescription;
  class EventForOutput;
  class ThinnedAssociationsHelper;
  class TriggerResults;

  namespace streamer {
    class InitMsgBuilder;
    class EventMsgBuilder;

    class StreamerOutputModuleCommon {
    public:
      using Parameters = StreamerOutputMsgBuilders::Parameters;

      static Parameters parameters(ParameterSet const& ps);

      explicit StreamerOutputModuleCommon(Parameters const& p,
                                          SelectedProducts const* selections,
                                          std::string const& moduleLabel);

      explicit StreamerOutputModuleCommon(ParameterSet const& ps,
                                          SelectedProducts const* selections,
                                          std::string const& moduleLabel)
          : StreamerOutputModuleCommon(parameters(ps), selections, moduleLabel) {}

      ~StreamerOutputModuleCommon();
      static void fillDescription(ParameterSetDescription& desc);

      std::unique_ptr<InitMsgBuilder> serializeRegistry(std::string const& processName,
                                                        std::string const& moduleLabel,
                                                        ParameterSetID const& toplevel,
                                                        SendJobHeader::ParameterSetMap const* psetMap);

      std::unique_ptr<EventMsgBuilder> serializeEventMetaData(BranchIDLists const& branchLists,
                                                              ThinnedAssociationsHelper const& helper);

      std::unique_ptr<EventMsgBuilder> serializeEvent(EventForOutput const& e,
                                                      Handle<TriggerResults> const& triggerResults,
                                                      ParameterSetID const& selectorCfg);

    protected:
      void clearHeaderBuffer() { buffer_.clearHeaderBuffer(); }

    private:
      SerializeDataBuffer buffer_;
      StreamerOutputMsgBuilders builders_;

      uint32_t eventMetaDataChecksum_ = 0;
    };  //end-of-class-def
  }     // namespace streamer
}  // namespace edm

#endif
