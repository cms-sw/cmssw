#ifndef IOPool_Streamer_StreamerInputSource_h
#define IOPool_Streamer_StreamerInputSource_h

/**
 * StreamerInputSource.h
 *
 * Base class for translating streamer message objects into
 * framework objects (e.g. ProductRegistry and EventPrincipal)
 */

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Sources/interface/RawInputSource.h"
#include "FWCore/Utilities/interface/propagate_const.h"

#include "DataFormats/Streamer/interface/StreamedProducts.h"
#include "DataFormats/Common/interface/EDProductGetter.h"

#include <memory>
#include <vector>

class TClass;

namespace edm {
  class BranchIDListHelper;
  class ParameterSetDescription;

}  // namespace edm

namespace edm::streamer {
  class InitMsgView;
  class EventMsgView;

  class StreamerInputSource : public RawInputSource {
  public:
    explicit StreamerInputSource(ParameterSet const& pset, InputSourceDescription const& desc);
    ~StreamerInputSource() override;
    static void fillDescription(ParameterSetDescription& description);

    std::unique_ptr<SendJobHeader> deserializeRegistry(InitMsgView const& initView);

    void deserializeAndMergeWithRegistry(InitMsgView const& initView, bool subsequent = false);

    //If eventView is a meta data event then this returns its checksum
    uint32_t eventMetaDataChecksum(EventMsgView const& eventView) const;
    //Should be called right after this message has been read
    void deserializeEventMetaData(EventMsgView const& eventView);
    void deserializeEvent(EventMsgView const& eventView);

    uint32_t presentEventMetaDataChecksum() const { return eventMetaDataChecksum_; }
    //This can only be called during a new file transition as it updates state that requires
    // framework synchronization.
    void updateEventMetaData();

    static void mergeIntoRegistry(SendJobHeader const& header, ProductRegistry&, bool subsequent);

  protected:
    static void declareStreamers(SendDescs const& descs);
    static void buildClassCache(SendDescs const& descs);
    void resetAfterEndRun();

    class EventPrincipalHolder : public EDProductGetter {
    public:
      EventPrincipalHolder();
      ~EventPrincipalHolder() override;

      WrapperBase const* getIt(ProductID const& id) const override;
      unsigned int transitionIndex_() const override;

      void setEventPrincipal(EventPrincipal* ep);

    private:
      // We don't own the principal.  The lifetime must be managed externally.
      EventPrincipal const* eventPrincipal_;
    };

    // An event message deserialized into a SendEvent, before it is handed to the framework
    struct DecodedEvent {
      std::unique_ptr<SendEvent> sendEvent;
      std::unique_ptr<EventPrincipalHolder> eventPrincipalHolder;
      uint32_t adler32_chksum = 0;
      uint32_t lumi = 0;
    };

    // Verify the checksum, uncompress and deserialize an event message, using dest as the
    // uncompression buffer. It does not modify the state of the source, so different events
    // can be decoded concurrently, each with its own buffer.
    void decodeEvent(EventMsgView const& eventView, std::vector<unsigned char>& dest, DecodedEvent& decoded) const;

    // Make a decoded (non meta data) event the next event of the source: check the meta data
    // checksum, register the process history and update the run and lumi auxiliaries.
    // Must be called serially and in input order.
    void finishEvent(DecodedEvent&& decoded);

  private:
    void deserializeEventCommon(EventMsgView const& eventView, bool isMetaData);

    void read(EventPrincipal& eventPrincipal) override;

    void setRun(RunNumber_t r) override;

    edm::propagate_const<TClass*> tc_;
    std::vector<unsigned char> dest_;
    edm::propagate_const<std::unique_ptr<SendEvent>> sendEvent_;
    edm::propagate_const<std::unique_ptr<EventPrincipalHolder>> eventPrincipalHolder_;
    std::vector<edm::propagate_const<std::unique_ptr<EventPrincipalHolder>>> streamToEventPrincipalHolders_;

    std::string processName_;
    unsigned int protocolVersion_;
    uint32_t eventMetaDataChecksum_ = 0;
  };  //end-of-class-def
}  // namespace edm::streamer

#endif
