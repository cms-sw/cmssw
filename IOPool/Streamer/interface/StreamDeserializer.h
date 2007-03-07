#ifndef Stream_Deserializer_h
#define Stream_Deserializer_h

/**
 * StreamDeserializer.h
 *
 * Utility class for translating streamer message objects into
 * framework objects (e.g. ProductRegistry and EventPrincipal)
 */

#include "DataFormats/Provenance/interface/ProcessConfiguration.h"
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
                         ProductRegistry const& productRegistry);
    void
    setProcessConfiguration(ProcessConfiguration pc) {
      processConfiguration_ = pc;
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

  private:
    ProcessConfiguration processConfiguration_;
    TClass* tc_;
    std::vector<unsigned char> dest_;
    TBuffer xbuf_;
  };

}

#endif
