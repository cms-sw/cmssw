#ifndef DataFormats_FEDRawData_RawDataBuffer_h
#define DataFormats_FEDRawData_RawDataBuffer_h

/** \class RawDataBuffer
 *  A product storing raw data for all SourceIDs in a Event, intended
 *  for persistent storage in ROOT Streamer or root files.
 *  For Phase-2 source ID is 32-bit, so map is used instead of linear vector
 *  as in the FEDRawDataCollection.  Legacy mode is added to use it for
 *  Phase-1 data.
 *  IMPORTANT: this is not a final format version for Phase-2 and only intended
 *  for detector development. It will not be frozen until the end of LS3.
 *  \author S. Morovic
 */

#include "DataFormats/Common/interface/traits.h"
#include "FWCore/Utilities/interface/Exception.h"

#include <vector>
#include <map>
#include <span>

/*
 * Phase2 source or FED wrapper pointing to a span of RawDataBuffer data
 * */

class RawFragmentWrapper {
private:
  uint32_t sourceId_;
  bool valid_;
  uint32_t size_;
  std::span<const unsigned char> span_;

public:
  RawFragmentWrapper() : sourceId_(0), valid_(false) {}
  RawFragmentWrapper(uint32_t sourceId, std::span<const unsigned char>&& span)
      : sourceId_(sourceId), valid_(true), span_(span) {}
  std::span<const unsigned char> const& data() const { return span_; }
  std::span<const unsigned char> dataHeader(uint32_t expSize) const {
    if (expSize > span_.size())
      throw cms::Exception("RawFragmentWrapper") << "Expected trailer too large: " << expSize << " > " << span_.size();
    return span_.subspan(0, expSize);
  }
  std::span<const unsigned char> dataTrailer(uint32_t expSize) const {
    if (expSize > span_.size())
      throw cms::Exception("RawFragmentWrapper") << "Expected trailer too large: " << expSize << " > " << span_.size();
    return span_.subspan(span_.size() - expSize, expSize);
  }
  std::span<const unsigned char> payload(uint32_t expSizeHeader, uint32_t expSizeTrailer) const {
    if (expSizeHeader + expSizeTrailer > span_.size())
      throw cms::Exception("RawFragmentWrapper")
          << "Trailer and header too large: " << expSizeHeader << " + " << expSizeTrailer << " > " << span_.size();
    return span_.subspan(expSizeHeader, span_.size() - expSizeTrailer - expSizeHeader);
  }

  uint32_t size() const { return span_.size(); }
  uint32_t sourceId() const { return sourceId_; }
  bool isValid() const { return valid_; }
};

/*
 * Contains metadata vector with source ID or FED ID offsets and size
 * and data buffer as vector of bytes
 */

class RawDataBuffer : public edm::DoNotRecordParents {
public:
  RawDataBuffer();
  RawDataBuffer(uint32_t preallocSize);

  unsigned char* addSource(uint32_t sourceId, unsigned char const* buf, uint32_t size);
  void setPhase1Range() { phase1Range_ = true; }

  const RawFragmentWrapper fragmentData(uint32_t sourceId) const;
  const RawFragmentWrapper fragmentData(
      std::map<uint32_t, std::pair<uint32_t, uint32_t>>::const_iterator const& it) const;

  unsigned char getByte(unsigned int pos) const { return data_.at(pos); }
  std::vector<unsigned char> data() const { return data_; }
  std::map<uint32_t, std::pair<uint32_t, uint32_t>> const& map() const { return map_; }

private:
  uint32_t usedSize_ = 0;
  std::map<uint32_t, std::pair<uint32_t, uint32_t>> map_;  //map of source id fragment offset and size pairs
  std::vector<unsigned char> data_;                        //raw data byte vector
  bool phase1Range_ = false;
};

#endif
