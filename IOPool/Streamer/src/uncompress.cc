#include "FWCore/Utilities/interface/Exception.h"
#include "IOPool/Streamer/interface/uncompress.h"

#include "zlib.h"
#include "lzma.h"
#include "zstd.h"
#include "lz4.h"

#include <cstring>

namespace {
  /**
   * Detect if buffer starts with "XZ\0" which means it is compressed in LZMA format
   */
  bool isBufferLZMA(unsigned char const* inputBuffer, unsigned int inputSize) {
    if (inputSize >= 4 && !strcmp((const char*)inputBuffer, "XZ"))
      return true;
    else
      return false;
  }

  /**
   * Detect if buffer starts with "Z\0" which means it is compressed in ZStandard format
   */
  bool isBufferZSTD(unsigned char const* inputBuffer, unsigned int inputSize) {
    if (inputSize >= 4 && !strcmp((const char*)inputBuffer, "ZS"))
      return true;
    else
      return false;
  }

  /**
   * Detect if buffer starts with "L4\0" which means it is compressed in LZ4 format
   */
  bool isBufferLZ4(unsigned char const* inputBuffer, unsigned int inputSize) {
    if (inputSize >= 4 && !strcmp((const char*)inputBuffer, "L4"))
      return true;
    else
      return false;
  }

  unsigned int uncompressBufferZLIB(unsigned char* inputBuffer,
                                    unsigned int inputSize,
                                    std::vector<unsigned char>& outputBuffer,
                                    unsigned int expectedFullSize) {
    unsigned long origSize = expectedFullSize;
    unsigned long uncompressedSize = expectedFullSize * 1.1;
    outputBuffer.resize(uncompressedSize);
    int ret =
        ::uncompress(&outputBuffer[0], &uncompressedSize, inputBuffer, inputSize);  // do not need compression level
    //std::cout << "unCompress Return value: " << ret << " Okay = " << Z_OK << std::endl;
    if (ret == Z_OK) {
      // check the length against original uncompressed length
      if (origSize != uncompressedSize) {
        // we throw an error and return without event! null pointer
        throw cms::Exception("StreamDeserialization", "Uncompression error")
            << "mismatch event lengths should be" << origSize << " got " << uncompressedSize << "\n";
      }
    } else {
      // we throw an error and return without event! null pointer
      throw cms::Exception("StreamDeserialization", "Uncompression error") << "Error code = " << ret << "\n ";
    }
    return (unsigned int)uncompressedSize;
  }

  unsigned int uncompressBufferLZMA(unsigned char* inputBuffer,
                                    unsigned int inputSize,
                                    std::vector<unsigned char>& outputBuffer,
                                    unsigned int expectedFullSize,
                                    bool hasHeader) {
    unsigned long origSize = expectedFullSize;
    unsigned long uncompressedSize = expectedFullSize * 1.1;
    outputBuffer.resize(uncompressedSize);

    lzma_stream stream = LZMA_STREAM_INIT;
    lzma_ret returnStatus;

    returnStatus = lzma_stream_decoder(&stream, UINT64_MAX, 0U);
    if (returnStatus != LZMA_OK) {
      throw cms::Exception("StreamDeserializationLZM", "LZMA stream decoder error")
          << "Error code = " << returnStatus << "\n ";
    }

    size_t hdrSize = hasHeader ? 4 : 0;
    stream.next_in = (const uint8_t*)(inputBuffer + hdrSize);
    stream.avail_in = (size_t)(inputSize - hdrSize);
    stream.next_out = (uint8_t*)&outputBuffer[0];
    stream.avail_out = (size_t)uncompressedSize;

    returnStatus = lzma_code(&stream, LZMA_FINISH);
    if (returnStatus != LZMA_STREAM_END) {
      lzma_end(&stream);
      throw cms::Exception("StreamDeserializationLZM", "LZMA uncompression error")
          << "Error code = " << returnStatus << "\n ";
    }
    lzma_end(&stream);

    uncompressedSize = (unsigned int)stream.total_out;

    if (origSize != uncompressedSize) {
      // we throw an error and return without event! null pointer
      throw cms::Exception("StreamDeserialization", "LZMA uncompression error")
          << "mismatch event lengths should be" << origSize << " got " << uncompressedSize << "\n";
    }

    return uncompressedSize;
  }

  unsigned int uncompressBufferZSTD(unsigned char* inputBuffer,
                                    unsigned int inputSize,
                                    std::vector<unsigned char>& outputBuffer,
                                    unsigned int expectedFullSize,
                                    bool hasHeader) {
    unsigned long uncompressedSize = expectedFullSize * 1.1;
    outputBuffer.resize(uncompressedSize);

    size_t hdrSize = hasHeader ? 4 : 0;
    size_t ret = ZSTD_decompress(
        (void*)&(outputBuffer[0]), uncompressedSize, (const void*)(inputBuffer + hdrSize), inputSize - hdrSize);

    if (ZSTD_isError(ret)) {
      throw cms::Exception("StreamDeserializationZSTD", "ZSTD uncompression error")
          << "Error core " << ret << ", message:" << ZSTD_getErrorName(ret);
    }
    return (unsigned int)ret;
  }
  unsigned int uncompressBufferLZ4(unsigned char* inputBuffer,
                                   unsigned int inputSize,
                                   std::vector<unsigned char>& outputBuffer,
                                   unsigned int expectedFullSize,
                                   bool hasHeader) {
    // LZ4 blocks do not store their uncompressed size: the expected size is exact
    outputBuffer.resize(expectedFullSize);

    size_t hdrSize = hasHeader ? 4 : 0;
    int ret = LZ4_decompress_safe(
        (const char*)(inputBuffer + hdrSize), (char*)&(outputBuffer[0]), inputSize - hdrSize, expectedFullSize);

    if (ret < 0) {
      throw cms::Exception("StreamDeserializationLZ4", "LZ4 uncompression error") << "Error code " << ret;
    }
    if ((unsigned int)ret != expectedFullSize) {
      throw cms::Exception("StreamDeserialization", "LZ4 uncompression error")
          << "mismatch event lengths should be" << expectedFullSize << " got " << ret << "\n";
    }
    return (unsigned int)ret;
  }
}  // namespace

namespace edm::streamer::uncompress {
  Algo compressionAlgo(unsigned char const* inputBuffer, unsigned int inputSize) {
    if (isBufferLZMA(inputBuffer, inputSize)) {
      return Algo::kLZMA;
    } else if (isBufferZSTD(inputBuffer, inputSize)) {
      return Algo::kZSTD;
    } else if (isBufferLZ4(inputBuffer, inputSize)) {
      return Algo::kLZ4;
    } else {
      return Algo::kZLIB;
    }
  }

  unsigned int uncompressBuffer(unsigned char const* inputBuffer,
                                unsigned int inputSize,
                                std::vector<unsigned char>& outputBuffer,
                                unsigned int expectedFullSize,
                                bool hasHeader) {
    if (isBufferLZMA(inputBuffer, inputSize)) {
      return uncompressBufferLZMA(
          const_cast<unsigned char*>(inputBuffer), inputSize, outputBuffer, expectedFullSize, hasHeader);
    } else if (isBufferZSTD(inputBuffer, inputSize)) {
      return uncompressBufferZSTD(
          const_cast<unsigned char*>(inputBuffer), inputSize, outputBuffer, expectedFullSize, hasHeader);
    } else if (isBufferLZ4(inputBuffer, inputSize)) {
      return uncompressBufferLZ4(
          const_cast<unsigned char*>(inputBuffer), inputSize, outputBuffer, expectedFullSize, hasHeader);
    } else
      return uncompressBufferZLIB(const_cast<unsigned char*>(inputBuffer), inputSize, outputBuffer, expectedFullSize);
  }
}  // namespace edm::streamer::uncompress
