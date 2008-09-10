#ifndef IOPool_Streamer_StreamSerializer_h
#define IOPool_Streamer_StreamSerializer_h

/**
 * StreamSerializer.h
 *
 * Utility class for translating framework objects (e.g. ProductRegistry and
 * EventPrincipal) into streamer message objects.
 */

#include "RVersion.h"
#if ROOT_VERSION_CODE >= ROOT_VERSION(5,15,0)
#include "TBufferFile.h"
typedef TBufferFile RootBuffer;
#else
#include "TBuffer.h"
typedef TBuffer RootBuffer;
#endif

const int init_size = 1024*1024;

// Data structure to be shared by all output modules for event serialization
struct SerializeDataBuffer
{
  typedef std::vector<char> SBuffer;

  SerializeDataBuffer():
    comp_buf_(init_size),
    curr_event_size_(),
    curr_space_used_(),
    rootbuf_(TBuffer::kWrite,init_size),
    ptr_((unsigned char*)rootbuf_.Buffer()),
    header_buf_(),
    bufs_()
  { }

  // This object caches the results of the last INIT or event 
  // serialization operation.  You get access to the data using the
  // following member functions.

  unsigned char* bufferPointer() const { return ptr_; }
  unsigned int currentSpaceUsed() const { return curr_space_used_; }
  unsigned int currentEventSize() const { return curr_event_size_; }

  std::vector<unsigned char> comp_buf_; // space for compressed data
  unsigned int curr_event_size_;
  unsigned int curr_space_used_; // less than curr_event_size_ if compressed
  RootBuffer rootbuf_;
  unsigned char* ptr_; // set to the place where the last event stored
  SBuffer header_buf_; // place for INIT message creation
  SBuffer bufs_;       // place for EVENT message creation
};

#include "DataFormats/Provenance/interface/Selections.h"
#include <vector>

class EventMsgBuilder;
class InitMsgBuilder;
namespace edm
{
  
  class EventPrincipal;
  class StreamSerializer
  {

  public:

    StreamSerializer(Selections const* selections);

    int serializeRegistry(SerializeDataBuffer &data_buffer);   
    int serializeEvent(EventPrincipal const& eventPrincipal,
                       bool use_compression, int compression_level,
                       SerializeDataBuffer &data_buffer);

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

    Selections const* selections_;
    // Arr data_;
    TClass* tc_;
  };

}

#endif
