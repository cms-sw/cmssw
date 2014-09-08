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

#include "DataFormats/Streamer/interface/StreamedProducts.h"
#include "DataFormats/Common/interface/EDProductGetter.h"

#include <memory>
#include <vector>

class InitMsgView;
class EventMsgView;

namespace edm {
  class BranchIDListHelper;
  class ParameterSetDescription;
  class StreamerInputSource : public RawInputSource {
  public:  
    explicit StreamerInputSource(ParameterSet const& pset,
                 InputSourceDescription const& desc);
    virtual ~StreamerInputSource();
    static void fillDescription(ParameterSetDescription& description);

    std::auto_ptr<SendJobHeader> deserializeRegistry(InitMsgView const& initView);

    void deserializeAndMergeWithRegistry(InitMsgView const& initView, bool subsequent = false);

    void deserializeEvent(EventMsgView const& eventView);

    static
    void mergeIntoRegistry(SendJobHeader const& header, ProductRegistry&, BranchIDListHelper&, bool subsequent);

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
  protected:
    static void declareStreamers(SendDescs const& descs);
    static void buildClassCache(SendDescs const& descs);
    void resetAfterEndRun();

  private:

    class EventPrincipalHolder : public EDProductGetter {
    public:
      EventPrincipalHolder();
      virtual ~EventPrincipalHolder();

      virtual WrapperBase const* getIt(edm::ProductID const& id) const override;
 
      virtual unsigned int transitionIndex_() const override;

      void setEventPrincipal(EventPrincipal* ep);

    private:
      // We don't own the principal.  The lifetime must be managed externally.
      EventPrincipal const* eventPrincipal_;
    };

    virtual void read(EventPrincipal& eventPrincipal);

    virtual void setRun(RunNumber_t r);

    virtual std::unique_ptr<FileBlock> readFile_();

    TClass* tc_;
    std::vector<unsigned char> dest_;
    TBufferFile xbuf_;
    std::unique_ptr<SendEvent> sendEvent_;
    EventPrincipalHolder eventPrincipalHolder_;
    bool adjustEventToNewProductRegistry_;

    std::string processName_;
    unsigned int protocolVersion_;
  }; //end-of-class-def
} // end of namespace-edm
  
#endif
