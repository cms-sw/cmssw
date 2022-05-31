#ifndef DataFormats_HGCalDigis_PHGCSimAccumulator_h
#define DataFormats_HGCalDigis_PHGCSimAccumulator_h

#include "DataFormats/DetId/interface/DetId.h"
#include <iostream>
#include <vector>

class PHGCSimAccumulator {
public:
  class DetIdSize {
  public:
    DetIdSize() {}
    DetIdSize(unsigned int detId) : detId_(detId) {}
    void increaseSize() { ++size_; }
    unsigned int detId() const { return detId_; }
    unsigned int size() const { return size_; }

  private:
    unsigned int detId_ = 0;
    unsigned char size_ = 0;
  };

  class SimHitCollection {
  public:
    constexpr static unsigned energyOffset = 15;
    constexpr static unsigned energyMask = 0x1;
    constexpr static unsigned sampleOffset = 11;
    constexpr static unsigned sampleMask = 0xf;
    constexpr static unsigned dataOffset = 0;
    constexpr static unsigned dataMask = 0x7ff;

    SimHitCollection() {}
    SimHitCollection(unsigned int nhits) : nhits_(nhits) {}
    SimHitCollection(const unsigned short si,
                     const std::vector<unsigned short>& accCharge,
                     const std::vector<unsigned short>& time)
        : nhits_(accCharge.size()) {
      chargeArray_.reserve(nhits_);
      timeArray_.reserve(nhits_);
      for (size_t i = 0; i < nhits_; ++i) {
        unsigned short ei = 0;
        unsigned short d = accCharge[i];
        unsigned short data = ((ei << energyOffset) | (si << sampleOffset) | d);
        chargeArray_.emplace_back(data);
      }
      for (size_t i = 0; i < nhits_; ++i) {
        unsigned short ei = 1;
        unsigned short d = time[i];
        unsigned short data = ((ei << energyOffset) | (si << sampleOffset) | d);
        timeArray_.emplace_back(data);
      }
    }
    SimHitCollection(const SimHitCollection& simhitcollection) = default;
    unsigned int nhits() const { return nhits_; }
    unsigned int sampleIndex() const { return (chargeArray_[0] >> sampleOffset) & sampleMask; }
    const std::vector<unsigned short>& chargeArray() const { return chargeArray_; }
    const std::vector<unsigned short>& timeArray() const { return timeArray_; }

  private:
    unsigned int nhits_;
    std::vector<unsigned short> chargeArray_;
    std::vector<unsigned short> timeArray_;
  };

  PHGCSimAccumulator() = default;
  ~PHGCSimAccumulator() = default;

  void reserve(size_t size) {
    detIdSize_.reserve(size);
    simhitCollection_.reserve(size);
  }
  void shrink_to_fit() {
    detIdSize_.shrink_to_fit();
    simhitCollection_.shrink_to_fit();
  }

  void emplace_back(unsigned int detId,
                    unsigned short sampleIndex,
                    const std::vector<unsigned short>& accCharge,
                    const std::vector<unsigned short>& timing) {
    if (detIdSize_.empty() || detIdSize_.back().detId() != detId) {
      detIdSize_.emplace_back(detId);
    }
    simhitCollection_.emplace_back(sampleIndex, accCharge, timing);
    detIdSize_.back().increaseSize();
  }

  class TmpElem {
  public:
    TmpElem(const unsigned int detId, const SimHitCollection& simhitCollection)
        : detId_(detId), simhitcollection_(simhitCollection) {}

    unsigned int detId() const { return detId_; }
    unsigned short sampleIndex() const { return simhitcollection_.sampleIndex(); }
    unsigned int nhits() const { return simhitcollection_.nhits(); }
    const std::vector<unsigned short> chargeArray() const { return simhitcollection_.chargeArray(); }
    const std::vector<unsigned short> timeArray() const { return simhitcollection_.timeArray(); }

  private:
    unsigned int detId_;
    SimHitCollection simhitcollection_;
  };

  class const_iterator {
  public:
    // begin
    const_iterator(const PHGCSimAccumulator* ncc)
        : ncc_(ncc), iDet_(0), iData_(0), endData_(ncc->detIdSize_.empty() ? 0 : ncc->detIdSize_.front().size()) {}
    // end
    const_iterator(const PHGCSimAccumulator* ncc, unsigned int detSize, unsigned int dataSize)
        : ncc_(ncc), iDet_(detSize), iData_(dataSize), endData_(0) {}

    bool operator==(const const_iterator& other) const { return iDet_ == other.iDet_ && iData_ == other.iData_; }
    bool operator!=(const const_iterator& other) const { return !operator==(other); }
    const_iterator& operator++() {
      ++iData_;
      if (iData_ == endData_) {
        ++iDet_;
        endData_ += (iDet_ == ncc_->detIdSize_.size()) ? 0 : ncc_->detIdSize_[iDet_].size();
      }
      return *this;
    }
    const_iterator operator++(int) {
      auto tmp = *this;
      ++(*this);
      return tmp;
    }

    TmpElem operator*() { return TmpElem(ncc_->detIdSize_[iDet_].detId(), ncc_->simhitCollection_[iData_]); }

  private:
    const PHGCSimAccumulator* ncc_;
    unsigned int iDet_;
    unsigned int iData_;
    unsigned int endData_;
  };

  TmpElem back() const { return TmpElem(detIdSize_.back().detId(), simhitCollection_.back()); }

  const_iterator cbegin() const { return const_iterator(this); }
  const_iterator begin() const { return cbegin(); }
  const_iterator cend() const { return const_iterator(this, detIdSize_.size(), simhitCollection_.size()); }
  const_iterator end() const { return cend(); }

private:
  std::vector<DetIdSize> detIdSize_;
  std::vector<SimHitCollection> simhitCollection_;
};

#endif
