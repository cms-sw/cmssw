#ifndef IOPool_Streamer_StreamerInputSource_h
#define IOPool_Streamer_StreamerInputSource_h

/**
 * StreamerInputSource.h
 *
 * Base class for translating streamer message objects into
 * framework objects (e.g. ProductRegistry and EventPrincipal)
 */

#include "TBufferFile.h"

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Sources/interface/RawInputSource.h"
#include "FWCore/Utilities/interface/propagate_const.h"

#include "DataFormats/Streamer/interface/StreamedProducts.h"
#include "DataFormats/Common/interface/EDProductGetter.h"

#include <memory>
#include <vector>

namespace edm {
  class BranchIDListHelper;
  class ParameterSetDescription;
  class ThinnedAssociationsHelper;
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

    /**
     * Detect if buffer starts with "XZ\0" which means it is compressed in LZMA format
     */
    bool isBufferLZMA(unsigned char const* inputBuffer, unsigned int inputSize);

    /**
     * Detect if buffer starts with "Z\0" which means it is compressed in ZStandard format
     */
    bool isBufferZSTD(unsigned char const* inputBuffer, unsigned int inputSize);

    /**
     * Uncompresses the data in the specified input buffer into the
     * specified output buffer.  The inputSize should be set to the size
     * of the compressed data in the inputBuffer.  The expectedFullSize should
     * be set to the original size of the data (before compression).
     * Returns the actual size of the uncompressed data.
     * Errors are reported by throwing exceptions.
     */
    static unsigned int uncompressBuffer(unsigned char* inputBuffer,
                                         unsigned int inputSize,
                                         std::vector<unsigned char>& outputBuffer,
                                         unsigned int expectedFullSize);

    static unsigned int uncompressBufferLZMA(unsigned char* inputBuffer,
                                             unsigned int inputSize,
                                             std::vector<unsigned char>& outputBuffer,
                                             unsigned int expectedFullSize,
                                             bool hasHeader = true);

    static unsigned int uncompressBufferZSTD(unsigned char* inputBuffer,
                                             unsigned int inputSize,
                                             std::vector<unsigned char>& outputBuffer,
                                             unsigned int expectedFullSize,
                                             bool hasHeader = true);

  protected:
    static void declareStreamers(SendDescs const& descs);
    static void buildClassCache(SendDescs const& descs);
    void resetAfterEndRun();

  private:
    void deserializeEventCommon(EventMsgView const& eventView, bool isMetaData);

    class EventPrincipalHolder : public EDProductGetter {
    public:
      EventPrincipalHolder();
      ~EventPrincipalHolder() override;

      WrapperBase const* getIt(ProductID const& id) const override;
      std::optional<std::tuple<edm::WrapperBase const*, unsigned int>> getThinnedProduct(ProductID const&,
                                                                                         unsigned int) const override;
      void getThinnedProducts(ProductID const& pid,
                              std::vector<WrapperBase const*>& wrappers,
                              std::vector<unsigned int>& keys) const override;
      OptionalThinnedKey getThinnedKeyFrom(ProductID const&, unsigned int, ProductID const&) const override;

      unsigned int transitionIndex_() const override;

      void setEventPrincipal(EventPrincipal* ep);

    private:
      // We don't own the principal.  The lifetime must be managed externally.
      EventPrincipal const* eventPrincipal_;
    };

    void read(EventPrincipal& eventPrincipal) override;

    void setRun(RunNumber_t r) override;

    edm::propagate_const<TClass*> tc_;
    std::vector<unsigned char> dest_;
    TBufferFile xbuf_;
    edm::propagate_const<std::unique_ptr<SendEvent>> sendEvent_;
    edm::propagate_const<std::unique_ptr<EventPrincipalHolder>> eventPrincipalHolder_;
    std::vector<edm::propagate_const<std::unique_ptr<EventPrincipalHolder>>> streamToEventPrincipalHolders_;

    std::string processName_;
    unsigned int protocolVersion_;
    uint32_t eventMetaDataChecksum_ = 0;
  };  //end-of-class-def
}  // namespace edm::streamer

#endif
