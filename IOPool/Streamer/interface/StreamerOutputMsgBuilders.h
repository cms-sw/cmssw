#ifndef IOPool_Streamer_StreamerOutputMsgBuilders_h
#define IOPool_Streamer_StreamerOutputMsgBuilders_h

#include "IOPool/Streamer/interface/MsgTools.h"
#include "FWCore/Common/interface/TriggerNames.h"
#include "IOPool/Streamer/interface/StreamSerializer.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Streamer/interface/StreamedProducts.h"
#include <memory>
#include <vector>

namespace edm {
  class EventForOutput;
  class ThinnedAssociationsHelper;
  class TriggerResults;
  class ParameterSet;
  class ParameterSetDescription;

  namespace streamer {
    class InitMsgBuilder;
    class EventMsgBuilder;

    class StreamerOutputMsgBuilders {
    public:
      struct Parameters {
        Strings hltTriggerSelections;
        std::string compressionAlgoStr;
        int compressionLevel;
        int lumiSectionInterval;
        bool useCompression;
      };

      static Parameters parameters(ParameterSet const& ps);

      explicit StreamerOutputMsgBuilders(Parameters const& p,
                                         SelectedProducts const* selections,
                                         std::string const& moduleLabel);

      ~StreamerOutputMsgBuilders();
      static void fillDescription(ParameterSetDescription& desc);

      std::unique_ptr<InitMsgBuilder> serializeRegistry(SerializeDataBuffer& sbuf,
                                                        std::string const& processName,
                                                        std::string const& moduleLabel,
                                                        ParameterSetID const& toplevel,
                                                        SendJobHeader::ParameterSetMap const* psetMap) const;

      std::pair<std::unique_ptr<EventMsgBuilder>, uint32_t> serializeEventMetaData(
          SerializeDataBuffer& sbuf, BranchIDLists const& branchLists, ThinnedAssociationsHelper const& helper) const;

      std::unique_ptr<EventMsgBuilder> serializeEvent(SerializeDataBuffer& sbuf,
                                                      EventForOutput const& e,
                                                      Handle<TriggerResults> const& triggerResults,
                                                      ParameterSetID const& selectorCfg,
                                                      uint32_t eventMetaDataChecksum) const;

    private:
      std::unique_ptr<EventMsgBuilder> serializeEventCommon(uint32 run,
                                                            uint32 lumi,
                                                            uint64 event,
                                                            std::vector<unsigned char> hltbits,
                                                            unsigned int hltsize,
                                                            SerializeDataBuffer& sbuf) const;

      void setHltMask(EventForOutput const& e,
                      Handle<TriggerResults> const& triggerResults,
                      std::vector<unsigned char>& hltbits) const;

      StreamSerializer serializer_;

      int maxEventSize_;
      bool useCompression_;
      std::string compressionAlgoStr_;
      int compressionLevel_;

      StreamerCompressionAlgo compressionAlgo_;

      // test luminosity sections
      int lumiSectionInterval_;
      double timeInSecSinceUTC;

      unsigned int hltsize_;
      char host_name_[255];

      Strings hltTriggerSelections_;
      uint32 outputModuleId_;

      uint32_t eventMetaDataChecksum_ = 0;
    };  //end-of-class-def
  }     // namespace streamer
}  // namespace edm

#endif
