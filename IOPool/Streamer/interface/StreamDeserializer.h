#ifndef Stream_Deserializer_h
#define Stream_Deserializer_h

/**
 * StreamDeserializer.h
 *
 * Utility class for translating streamer message objects into
 * framework objects (e.g. ProductRegistry and EventPrincipal)
 */

#include "boost/shared_ptr.hpp"

#include "DataFormats/Provenance/interface/ProcessConfiguration.h"
#include "DataFormats/Provenance/interface/ProcessHistoryID.h"
#include "TBuffer.h"
#include "TClass.h"
#include <vector>

class InitMsgView;
class EventMsgView;
namespace edm {

  class SendJobHeader;
  class EventPrincipal;
  class ProductRegistry;
  class StreamDeserializer {

  public:

    StreamDeserializer();

    static std::auto_ptr<SendJobHeader>
        deserializeRegistry(InitMsgView const& initView);

    std::auto_ptr<EventPrincipal>
        deserializeEvent(EventMsgView const& eventView,
                         boost::shared_ptr<ProductRegistry const> productRegistry);
    void
    setProcessConfiguration(ProcessConfiguration pc) {
      processConfiguration_ = pc;
    }

    void
    setProcessHistoryID(ProcessHistoryID phid) {
      processHistoryID_ = phid;
    }

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

   //static std::string processName() { return processName_; }
   ///static unsigned int protocolVersion() {return protocolVersion_;}

    //Do not like this to be static, but no choice as deserializeRegistry() that sets it is a static memeber 
    static std::string processName_;
    static unsigned int protocolVersion_;

  private:
    ProcessConfiguration processConfiguration_;
    ProcessHistoryID processHistoryID_;
    TClass* tc_;
    std::vector<unsigned char> dest_;
    TBuffer xbuf_;

    //Do not like this to be static, but no choice as deserializeRegistry() that sets it is a static memeber 
    //static std::string processName_;
    //static unsigned int protocolVersion_;
  };

}

#endif
