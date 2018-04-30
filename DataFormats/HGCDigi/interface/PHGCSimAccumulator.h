#ifndef DataFormats_HGCDigi_PHGCSimAccumulator_h
#define DataFormats_HGCDigi_PHGCSimAccumulator_h

#include "DataFormats/DetId/interface/DetId.h"

#include <vector>
#include <cassert>

class PHGCSimAccumulator {
public:
  // These two structs are public only because of dictionary generation
  class DetIdSize {
  public:
    // Use top 27 bits to hold the details of the DetId
    constexpr static unsigned detIdOffset = 5;
    constexpr static unsigned detIdMask = 0x7ffffff;
    // Use the last 5 bits to index 2x15 elements
    constexpr static unsigned sizeOffset = 27;
    constexpr static unsigned sizeMask = 0x1f;
    // With this arrangement increasing the size component is faster

    DetIdSize(): detIdSize_(0) {}
    DetIdSize(unsigned int detId): detIdSize_(detId << detIdOffset) {}

    void increaseSize() {
      assert(size()+1 < 1<<detIdOffset);
      ++detIdSize_;
    }

    unsigned int detIdDetails() const { return detIdSize_ >> detIdOffset; }
    unsigned int size() const { return detIdSize_ & sizeMask; }

  private:
    unsigned int detIdSize_;
  };
  class Data {
  public:
    constexpr static unsigned energyOffset = 15;
    constexpr static unsigned energyMask = 0x1;
    constexpr static unsigned sampleOffset = 11;
    constexpr static unsigned sampleMask = 0xf;
    constexpr static unsigned dataOffset = 0;
    constexpr static unsigned dataMask = 0x7ff;

    Data(): data_(0) {}
    Data(unsigned short ei, unsigned short si, unsigned short d):
      data_((ei << energyOffset) | (si << sampleOffset) | d)
    {}

    unsigned int energyIndex() const { return data_ >> energyOffset; }
    unsigned int sampleIndex() const { return (data_ >> sampleOffset) & sampleMask; }
    unsigned int data() const { return data_ & dataMask; }

  private:
    unsigned short data_;
  };

  PHGCSimAccumulator() = default;
  PHGCSimAccumulator(unsigned int detId): detSubdetId_(detId >> DetId::kSubdetOffset) {}
  ~PHGCSimAccumulator() = default;

  void reserve(size_t size) {
    detIdSize_.reserve(size);
    data_.reserve(size);
  }

  void shrink_to_fit() {
    detIdSize_.shrink_to_fit();
    data_.shrink_to_fit();
  }

  /**
   * Adds data for a given detId, energyIndex, and sampleIndex.
   *
   * It is the caller's responsibility to ensure that energyIndex,
   * sampleIndex, and data fit in the space reserved for them in the
   * Data bitfield above.
   */
  void emplace_back(unsigned int detId, unsigned short energyIndex, unsigned short sampleIndex, unsigned short data) {
    assert( (detId >> DetId::kSubdetOffset) == detSubdetId_ );

    if(detIdSize_.empty() || detIdSize_.back().detIdDetails() != (detId & DetIdSize::detIdMask)) {
      detIdSize_.emplace_back(detId);
    }
    data_.emplace_back(energyIndex, sampleIndex, data);
    detIdSize_.back().increaseSize();
  }

  class TmpElem {
  public:
    TmpElem(unsigned int detId, Data data): detId_(detId), data_(data) {}

    unsigned int detId() const { return detId_; }
    unsigned short energyIndex() const { return data_.energyIndex(); }
    unsigned short sampleIndex() const { return data_.sampleIndex(); }
    unsigned short data() const { return data_.data(); }
  private:
    unsigned int detId_;
    Data data_;
  };

  class const_iterator {
  public:
    // begin
    const_iterator(const PHGCSimAccumulator *acc):
      acc_(acc), iDet_(0), iData_(0),
      endData_(acc->detIdSize_.empty() ? 0 : acc->detIdSize_.front().size())
    {}

    // end
    const_iterator(const PHGCSimAccumulator *acc, unsigned int detSize, unsigned int dataSize):
      acc_(acc), iDet_(detSize), iData_(dataSize), endData_(0)
    {}

    bool operator==(const const_iterator& other) const {
      return iDet_ == other.iDet_ && iData_ == other.iData_;
    }
    bool operator!=(const const_iterator& other) const {
      return !operator==(other);
    }
    const_iterator& operator++() {
      ++iData_;
      if(iData_ == endData_) {
        ++iDet_;
        endData_ += (iDet_ == acc_->detIdSize_.size()) ? 0 : acc_->detIdSize_[iDet_].size();
      }
      return *this;
    }
    const_iterator operator++(int) {
      auto tmp = *this;
      ++(*this);
      return tmp;
    }
    TmpElem operator*() {
      return TmpElem((acc_->detSubdetId_ << DetId::kSubdetOffset) | acc_->detIdSize_[iDet_].detIdDetails(),
                     acc_->data_[iData_]);
    }

  private:
    const PHGCSimAccumulator *acc_;
    unsigned int iDet_;
    unsigned int iData_;
    unsigned int endData_;
  };

  TmpElem back() const {
    return TmpElem((detSubdetId_ << DetId::kSubdetOffset) | detIdSize_.back().detIdDetails(),
                   data_.back());
  }

  const_iterator cbegin() const { return const_iterator(this); }
  const_iterator begin() const { return cbegin(); }
  const_iterator cend() const { return const_iterator(this, detIdSize_.size(), data_.size()); }
  const_iterator end() const { return cend(); }

private:
  std::vector<DetIdSize> detIdSize_;
  std::vector<Data> data_;
  unsigned short detSubdetId_ = 0;
};

#endif
