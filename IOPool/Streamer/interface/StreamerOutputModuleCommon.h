#ifndef IOPool_Streamer_StreamerOutputModuleCommon_h
#define IOPool_Streamer_StreamerOutputModuleCommon_h

#include "IOPool/Streamer/interface/MsgTools.h"
#include "FWCore/Common/interface/TriggerNames.h"
#include "IOPool/Streamer/interface/StreamSerializer.h"
#include "DataFormats/Common/interface/Handle.h"
#include <memory>
#include <vector>

class InitMsgBuilder;
class EventMsgBuilder;
namespace edm {
  class ParameterSet;
  class ParameterSetDescription;
  class EventForOutput;
  class ThinnedAssociationsHelper;
  class TriggerResults;

  class StreamerOutputModuleCommon {
  public:
    explicit StreamerOutputModuleCommon(ParameterSet const& ps, SelectedProducts const* selections);
    ~StreamerOutputModuleCommon();
    static void fillDescription(ParameterSetDescription& desc);

    std::unique_ptr<InitMsgBuilder> serializeRegistry(BranchIDLists const& branchLists,
                                                      ThinnedAssociationsHelper const& helper,
                                                      std::string const& processName,
                                                      std::string const& moduleLabel,
                                                      ParameterSetID const& toplevel);

    std::unique_ptr<EventMsgBuilder> serializeEvent(EventForOutput const& e,
                                                    Handle<TriggerResults> const& triggerResults,
                                                    ParameterSetID const& selectorCfg);

    void clearSerializeDataBuffer() {
      serializeDataBuffer_.header_buf_.clear();
      serializeDataBuffer_.header_buf_.shrink_to_fit();
    }

  private:
    void setHltMask(EventForOutput const& e,
                    Handle<TriggerResults> const& triggerResults,
                    std::vector<unsigned char>& hltbits) const;

    StreamSerializer serializer_;

    int maxEventSize_;
    bool useCompression_;
    int compressionLevel_;

    // test luminosity sections
    int lumiSectionInterval_;
    double timeInSecSinceUTC;

    SerializeDataBuffer serializeDataBuffer_;

    unsigned int hltsize_;
    uint32 origSize_;
    char host_name_[255];

    Strings hltTriggerSelections_;
    uint32 outputModuleId_;

  };  //end-of-class-def

}  // namespace edm

#endif
