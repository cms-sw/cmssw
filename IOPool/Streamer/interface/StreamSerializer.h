#ifndef Stream_Serializer_h
#define Stream_Serializer_h

/**
 * StreamSerializer.h
 *
 * Utility class for translating framework objects (e.g. ProductRegistry and
 * EventPrincipal) into streamer message objects.
 */

#include "FWCore/Framework/interface/OutputModule.h"
#include "TBuffer.h"
#include <vector>

class EventMsgBuilder;
class InitMsgBuilder;
namespace edm
{
  
  class EventPrincipal;
  class StreamSerializer
  {

  public:

    StreamSerializer(OutputModule::Selections const* selections);

    int serializeRegistry(InitMsgBuilder& initMessage);   
    int serializeEvent(EventPrincipal const& eventPrincipal,
                       bool use_compression, int compression_level);

    // This object always caches the results of the last event 
    // serialization operation.  You get access to the data using the
    // following member functions.

    unsigned char* bufferPointer() const { return ptr_; }
    unsigned int currentSpaceUsed() const { return curr_space_used_; }
    unsigned int currentEventSize() const { return curr_event_size_; }

    /**
     * Compresses the data in the specified input buffer into the
     * specified output buffer.  Returns the size of the compressed data
     * or zero if compression failed.
     */
    static unsigned int compressBuffer(unsigned char *inputBuffer,
                                       unsigned int inputSize,
                                       std::vector<unsigned char> &outputBuffer,
                                       int compressionLevel);

  private:

    // helps to keep the data in this class exception safe
    struct Arr
    {
      explicit Arr(int sz); // allocate
      ~Arr(); // free

      char* ptr_;
    };

    OutputModule::Selections const* selections_;
    // Arr data_;
    std::vector<unsigned char> comp_buf_; // space for compressed data
    unsigned int curr_event_size_;
    unsigned int curr_space_used_; // less than curr_event_size_ if compressed
    TBuffer rootbuf_;
    unsigned char* ptr_; // set to the place where the last event stored
    TClass* tc_;
  };

}

#endif
