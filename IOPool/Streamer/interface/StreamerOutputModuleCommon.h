#ifndef IOPool_Streamer_StreamerOutputModuleCommon_h
#define IOPool_Streamer_StreamerOutputModuleCommon_h

#include "IOPool/Streamer/interface/MsgTools.h"
#include "FWCore/Common/interface/TriggerNames.h"
#include "IOPool/Streamer/interface/StreamSerializer.h"
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
      struct Parameters {
        Strings hltTriggerSelections;
        std::string compressionAlgoStr;
        int compressionLevel;
        int lumiSectionInterval;
        bool useCompression;
      };

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

      std::unique_ptr<InitMsgBuilder> serializeRegistry(SerializeDataBuffer& sbuf,
                                                        std::string const& processName,
                                                        std::string const& moduleLabel,
                                                        ParameterSetID const& toplevel,
                                                        SendJobHeader::ParameterSetMap const* psetMap);

      std::unique_ptr<EventMsgBuilder> serializeEventMetaData(SerializeDataBuffer& sbuf,
                                                              BranchIDLists const& branchLists,
                                                              ThinnedAssociationsHelper const& helper);

      std::unique_ptr<EventMsgBuilder> serializeEvent(SerializeDataBuffer& sbuf,
                                                      EventForOutput const& e,
                                                      Handle<TriggerResults> const& triggerResults,
                                                      ParameterSetID const& selectorCfg);

      SerializeDataBuffer* getSerializerBuffer();

    protected:
      std::unique_ptr<SerializeDataBuffer> serializerBuffer_;

    private:
      std::unique_ptr<EventMsgBuilder> serializeEventCommon(uint32 run,
                                                            uint32 lumi,
                                                            uint64 event,
                                                            std::vector<unsigned char> hltbits,
                                                            unsigned int hltsize,
                                                            SerializeDataBuffer& sbuf);

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
