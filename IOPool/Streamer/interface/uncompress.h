#ifndef IOPool_Streamer_uncompress_h
#define IOPool_Streamer_uncompress_h

#include <vector>

namespace edm::streamer::uncompress {
  enum class Algo { kZLIB, kZSTD, kLZMA };

  Algo compressionAlgo(unsigned char const* inputBuffer, unsigned int inputSize);

  /**
     * Uncompresses the data in the specified input buffer into the
     * specified output buffer.  The inputSize should be set to the size
     * of the compressed data in the inputBuffer.  The expectedFullSize should
     * be set to the original size of the data (before compression).
     * Figures out the compression algorithm from the input content.
     * Returns the actual size of the uncompressed data.
     * Errors are reported by throwing exceptions.
     */
  unsigned int uncompressBuffer(unsigned char const* inputBuffer,
                                unsigned int inputSize,
                                std::vector<unsigned char>& outputBuffer,
                                unsigned int expectedFullSize,
                                bool hasHeader = true);
}  // namespace edm::streamer::uncompress

#endif
