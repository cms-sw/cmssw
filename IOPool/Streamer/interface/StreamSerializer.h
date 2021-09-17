#ifndef IOPool_Streamer_StreamSerializer_h
#define IOPool_Streamer_StreamSerializer_h

/**
 * StreamSerializer.h
 *
 * Utility class for translating framework objects (e.g. ProductRegistry and
 * EventForOutput) into streamer message objects.
 */

#include "TBufferFile.h"

#include <cstdint>
#include <vector>

#include "DataFormats/Provenance/interface/BranchIDList.h"
#include "DataFormats/Provenance/interface/ParameterSetID.h"
#include "DataFormats/Provenance/interface/SelectedProducts.h"
#include "DataFormats/Streamer/interface/StreamedProducts.h"
#include "FWCore/Utilities/interface/get_underlying_safe.h"

// Data structure to be shared by all output modules for event serialization
struct SerializeDataBuffer {
  typedef std::vector<char> SBuffer;
  static constexpr int init_size = 0;  //will be allocated on first event
  static constexpr unsigned int reserve_size = 50000;

  SerializeDataBuffer()
      : comp_buf_(reserve_size + init_size),
        curr_event_size_(),
        curr_space_used_(),
        rootbuf_(TBuffer::kWrite, init_size),
        ptr_((unsigned char *)rootbuf_.Buffer()),
        header_buf_(),
        adler32_chksum_(0) {}

  // This object caches the results of the last INIT or event
  // serialization operation.  You get access to the data using the
  // following member functions.

  unsigned char const *bufferPointer() const { return get_underlying_safe(ptr_); }
  unsigned char *&bufferPointer() { return get_underlying_safe(ptr_); }
  unsigned int currentSpaceUsed() const { return curr_space_used_; }
  unsigned int currentEventSize() const { return curr_event_size_; }
  uint32_t adler32_chksum() const { return adler32_chksum_; }

  void clearHeaderBuffer() {
    header_buf_.clear();
    header_buf_.shrink_to_fit();
    rootbuf_.Reset();
    rootbuf_.Expand(init_size);  //shrink TBuffer to size 0 after resetting TBuffer length
  }

  std::vector<unsigned char> comp_buf_;  // space for compressed data
  unsigned int curr_event_size_;
  unsigned int curr_space_used_;  // less than curr_event_size_ if compressed
  TBufferFile rootbuf_;
  edm::propagate_const<unsigned char *> ptr_;  // set to the place where the last event stored
  SBuffer header_buf_;                         // place for INIT message creation and streamer event header
  uint32_t adler32_chksum_;                    // adler32 check sum for the (compressed) data
};

class EventMsgBuilder;
class InitMsgBuilder;
namespace edm {
  enum StreamerCompressionAlgo { UNCOMPRESSED = 0, ZLIB = 1, LZMA = 2, ZSTD = 4 };

  class EventForOutput;
  class ModuleCallingContext;
  class ThinnedAssociationsHelper;

  class StreamSerializer {
  public:
    StreamSerializer(SelectedProducts const *selections);

    int serializeRegistry(SerializeDataBuffer &data_buffer,
                          const BranchIDLists &branchIDLists,
                          ThinnedAssociationsHelper const &thinnedAssociationsHelper);

    int serializeRegistry(SerializeDataBuffer &data_buffer,
                          const BranchIDLists &branchIDLists,
                          ThinnedAssociationsHelper const &thinnedAssociationsHelper,
                          SendJobHeader::ParameterSetMap const &psetMap);

    int serializeEvent(SerializeDataBuffer &data_buffer,
                       EventForOutput const &event,
                       ParameterSetID const &selectorConfig,
                       StreamerCompressionAlgo compressionAlgo,
                       int compression_level,
                       unsigned int reserveSize) const;

    /**
     * Compresses the data in the specified input buffer into the
     * specified output buffer.  Returns the size of the compressed data
     * or zero if compression failed.
     */
    static unsigned int compressBuffer(unsigned char *inputBuffer,
                                       unsigned int inputSize,
                                       std::vector<unsigned char> &outputBuffer,
                                       int compressionLevel,
                                       unsigned int reserveSize);

    static unsigned int compressBufferLZMA(unsigned char *inputBuffer,
                                           unsigned int inputSize,
                                           std::vector<unsigned char> &outputBuffer,
                                           int compressionLevel,
                                           unsigned int reserveSize,
                                           bool addHeader = true);

    static unsigned int compressBufferZSTD(unsigned char *inputBuffer,
                                           unsigned int inputSize,
                                           std::vector<unsigned char> &outputBuffer,
                                           int compressionLevel,
                                           unsigned int reserveSize,
                                           bool addHeader = true);

  private:
    SelectedProducts const *selections_;
    edm::propagate_const<TClass *> tc_;
  };

}  // namespace edm

#endif
