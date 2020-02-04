#ifndef DataFormats_FTLDigi_PMTDSimAccumulator_h
#define DataFormats_FTLDigi_PMTDSimAccumulator_h

#include "DataFormats/DetId/interface/DetId.h"

#include <vector>
#include <cassert>

class PMTDSimAccumulator {
public:
  // These two structs are public only because of dictionary generation
  class DetIdSize {
  public:
    DetIdSize() {}
    DetIdSize(unsigned int detId, unsigned char row, unsigned char col) : detId_(detId), row_(row), column_(col) {}

    void increaseSize() { ++size_; }

    unsigned int detId() const { return detId_; }
    unsigned char row() const { return row_; }
    unsigned char column() const { return column_; }
    unsigned int size() const { return size_; }

  private:
    unsigned int detId_ = 0;
    unsigned char row_ = 0;
    unsigned char column_ = 0;
    unsigned char size_ = 0;
  };
  class Data {
  public:
    constexpr static unsigned energyOffset = 14;
    constexpr static unsigned energyMask = 0x3;
    constexpr static unsigned sampleOffset = 10;
    constexpr static unsigned sampleMask = 0xf;
    constexpr static unsigned dataOffset = 0;
    constexpr static unsigned dataMask = 0x3ff;

    Data() : data_(0) {}
    Data(unsigned short ei, unsigned short si, unsigned short d)
        : data_((ei << energyOffset) | (si << sampleOffset) | d) {}

    unsigned int energyIndex() const { return data_ >> energyOffset; }
    unsigned int sampleIndex() const { return (data_ >> sampleOffset) & sampleMask; }
    unsigned int data() const { return data_ & dataMask; }

  private:
    unsigned short data_;
  };

  PMTDSimAccumulator() = default;
  ~PMTDSimAccumulator() = default;

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
  void emplace_back(unsigned int detId,
                    unsigned char row,
                    unsigned char column,
                    unsigned short energyIndex,
                    unsigned short sampleIndex,
                    unsigned short data) {
    if (detIdSize_.empty() || detIdSize_.back().detId() != detId || detIdSize_.back().row() != row ||
        detIdSize_.back().column() != column) {
      detIdSize_.emplace_back(detId, row, column);
    }
    data_.emplace_back(energyIndex, sampleIndex, data);
    detIdSize_.back().increaseSize();
  }

  class TmpElem {
  public:
    TmpElem(unsigned int detId, unsigned char row, unsigned char column, Data data)
        : detId_(detId), data_(data), row_(row), column_(column) {}

    unsigned int detId() const { return detId_; }
    unsigned char row() const { return row_; }
    unsigned char column() const { return column_; }
    unsigned short energyIndex() const { return data_.energyIndex(); }
    unsigned short sampleIndex() const { return data_.sampleIndex(); }
    unsigned short data() const { return data_.data(); }

  private:
    unsigned int detId_;
    Data data_;
    unsigned char row_;
    unsigned char column_;
  };

  class const_iterator {
  public:
    // begin
    const_iterator(const PMTDSimAccumulator* acc)
        : acc_(acc), iDet_(0), iData_(0), endData_(acc->detIdSize_.empty() ? 0 : acc->detIdSize_.front().size()) {}

    // end
    const_iterator(const PMTDSimAccumulator* acc, unsigned int detSize, unsigned int dataSize)
        : acc_(acc), iDet_(detSize), iData_(dataSize), endData_(0) {}

    bool operator==(const const_iterator& other) const { return iDet_ == other.iDet_ && iData_ == other.iData_; }
    bool operator!=(const const_iterator& other) const { return !operator==(other); }
    const_iterator& operator++() {
      ++iData_;
      if (iData_ == endData_) {
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
      const auto& id = acc_->detIdSize_[iDet_];
      return TmpElem(id.detId(), id.row(), id.column(), acc_->data_[iData_]);
    }

  private:
    const PMTDSimAccumulator* acc_;
    unsigned int iDet_;
    unsigned int iData_;
    unsigned int endData_;
  };

  TmpElem back() const {
    const auto& id = detIdSize_.back();
    return TmpElem(id.detId(), id.row(), id.column(), data_.back());
  }

  const_iterator cbegin() const { return const_iterator(this); }
  const_iterator begin() const { return cbegin(); }
  const_iterator cend() const { return const_iterator(this, detIdSize_.size(), data_.size()); }
  const_iterator end() const { return cend(); }

private:
  std::vector<DetIdSize> detIdSize_;
  std::vector<Data> data_;
};

#endif
