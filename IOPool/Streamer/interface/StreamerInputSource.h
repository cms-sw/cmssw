#ifndef IOPool_Streamer_StreamerInputSource_h
#define IOPool_Streamer_StreamerInputSource_h

/**
 * StreamerInputSource.h
 *
 * Base class for translating streamer message objects into
 * framework objects (e.g. ProductRegistry and EventPrincipal)
 */


#include "TBufferFile.h"

#include "boost/shared_ptr.hpp"

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/InputSource.h"

#include "DataFormats/Streamer/interface/StreamedProducts.h"
#include "DataFormats/Provenance/interface/ProcessHistoryID.h"
#include "DataFormats/Common/interface/EDProductGetter.h"
#include <vector>

class InitMsgView;
class EventMsgView;

namespace edm {
  class StreamerInputSource : public InputSource {
  public:  
    explicit StreamerInputSource(ParameterSet const& pset,
                 InputSourceDescription const& desc);
    virtual ~StreamerInputSource();

    static
    std::auto_ptr<SendJobHeader> deserializeRegistry(InitMsgView const& initView);

    void deserializeAndMergeWithRegistry(InitMsgView const& initView, bool subsequent);

    EventPrincipal* deserializeEvent(EventMsgView const& eventView);

    static
    void mergeIntoRegistry(SendJobHeader const& header, ProductRegistry&, bool subsequent);

    /**
     * Uncompresses the data in the specified input buffer into the
     * specified output buffer.  The inputSize should be set to the size
     * of the compressed data in the inputBuffer.  The expectedFullSize should
     * be set to the original size of the data (before compression).
     * Returns the actual size of the uncompressed data.
     * Errors are reported by throwing exceptions.
     */
    static unsigned int uncompressBuffer(unsigned char *inputBuffer,
                                         unsigned int inputSize,
                                         std::vector<unsigned char> &outputBuffer,
                                         unsigned int expectedFullSize);
  protected:
    static void declareStreamers(SendDescs const& descs);
    static void buildClassCache(SendDescs const& descs);
    void setEndRun() {runEndingFlag_ = true;}
    void resetAfterEndRun();

    bool inputFileTransitionsEachEvent_;

  private:

    class ProductGetter : public EDProductGetter {
    public:
      ProductGetter();
      virtual ~ProductGetter();

      virtual EDProduct const* getIt(edm::ProductID const& id) const;

      void setEventPrincipal(EventPrincipal *ep);

    private:
      // We don't own the principal.  The lifetime must be managed externally.
      EventPrincipal const* eventPrincipal_;
    };

    virtual EventPrincipal* read() = 0;

    virtual boost::shared_ptr<RunAuxiliary> readRunAuxiliary_();

    virtual boost::shared_ptr<LuminosityBlockAuxiliary> readLuminosityBlockAuxiliary_();

    virtual EventPrincipal*
    readEvent_();

    virtual ItemType getNextItemType();

    virtual void setRun(RunNumber_t r);

    virtual boost::shared_ptr<FileBlock> readFile_();

    bool newRun_;
    bool newLumi_;
    bool eventCached_;

    TClass* tc_;
    std::vector<unsigned char> dest_;
    TBufferFile xbuf_;
    bool runEndingFlag_;
    ProductGetter productGetter_;

    //Do not like these to be static, but no choice as deserializeRegistry() that sets it is a static memeber 
    static std::string processName_;
    static unsigned int protocolVersion_;
  }; //end-of-class-def
} // end of namespace-edm
  
#endif
