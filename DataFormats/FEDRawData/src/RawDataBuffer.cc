/** \file
 *  implementation of DaqRawDataBuffer
 *
 *  \author N. Amapane - S. Argiro'
 */

#include <DataFormats/FEDRawData/interface/RawDataBuffer.h>
#include <DataFormats/FEDRawData/interface/FEDNumbering.h>
#include <DataFormats/FEDRawData/interface/SourceIDNumbering.h>
#include "FWCore/Utilities/interface/Exception.h"
#include <cstring>

RawDataBuffer::RawDataBuffer() {}
RawDataBuffer::RawDataBuffer(uint32_t preallocSize) : usedSize_(0), data_(preallocSize) {}

unsigned char* RawDataBuffer::addSource(uint32_t sourceId, unsigned char const* buf, uint32_t size) {
  auto maxSize = data_.size();
  if ((uint64_t)usedSize_ + (uint64_t)size > maxSize)
    throw cms::Exception("RawDataBuffer") << "RawDataBuffer size overflow adding ID " << sourceId << ": " << usedSize_
                                          << " + " << size << " > " << maxSize;

  if (phase1Range_) {
    if (sourceId > FEDNumbering::lastFEDId())
      throw cms::Exception("RawDataBuffer") << "FED ID " << sourceId << " out of range";
  } else {
    if (size % 16)
      throw cms::Exception("RawDataBuffer")
          << "source " << sourceId << " data not multiple of 16 bytes (" << size << ")";
    if (size < 32)
      throw cms::Exception("RawDataBuffer") << "source " << sourceId << " data is too small: " << size << " bytes";
  }

  void* targetAddr = (void*)(&data_[usedSize_]);
  //allow also buffer preparation without copy if source pointer is null
  if (buf != nullptr)
    memcpy(targetAddr, (void*)buf, size);
  auto usedSize = usedSize_;
  map_.emplace(sourceId, std::make_pair(usedSize, size));
  usedSize_ += size;
  return (unsigned char*)targetAddr;
}

const RawFragmentWrapper RawDataBuffer::fragmentData(uint32_t sourceId) const {
  if (phase1Range_ && sourceId > FEDNumbering::lastFEDId())
    throw cms::Exception("RawDataBuffer") << "can not fetch out of range FED ID " << sourceId;
  auto it = map_.find(sourceId);
  if (it == map_.end()) {
    return RawFragmentWrapper();
  }
  const auto& desc = it->second;
  std::span<const unsigned char>&& byte_span{reinterpret_cast<const unsigned char*>(this->data_.data() + desc.first),
                                             desc.second};
  return RawFragmentWrapper(sourceId, std::move(byte_span));
}

const RawFragmentWrapper RawDataBuffer::fragmentData(
    std::map<uint32_t, std::pair<uint32_t, uint32_t>>::const_iterator const& it) const {
  if (it == map_.end()) {
    return RawFragmentWrapper();
  }
  const auto& desc = it->second;
  std::span<const unsigned char>&& byte_span{reinterpret_cast<const unsigned char*>(this->data_.data() + desc.first),
                                             desc.second};
  return RawFragmentWrapper(it->first, std::move(byte_span));
}
