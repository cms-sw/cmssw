#ifndef EventFilter_SiStripRawToDigi_SiStripDetSetVectorFiller_H
#define EventFilter_SiStripRawToDigi_SiStripDetSetVectorFiller_H

#include "DataFormats/Common/interface/DetSetVector.h"
#include <vector>
#include <algorithm>
#include <memory>
#include <cstdint>
class SiStripRawDigi;
class SiStripDigi;

namespace sistrip {

  //A class to fill a DetSetVector from data from different channels only sorting once

  //T is type of object in DetSetVector (SiStripRawDigi, SiStripDigi), dsvIsSparse should be false for raw digis so null digis are inserted
  template <typename T, bool dsvIsSparse>
  class DetSetVectorFiller {
  public:
    DetSetVectorFiller(const size_t registrySize, const size_t dataSize);
    ~DetSetVectorFiller();

    void newChannel(const uint32_t key, const uint16_t firstItem = 0);
    void addItem(const T& item);
    std::unique_ptr<edm::DetSetVector<T> > createDetSetVector();

  private:
    struct ChannelRegistryItem {
      ChannelRegistryItem(const uint32_t theKey,
                          const uint32_t theStartIndex,
                          const uint16_t theLength,
                          const uint16_t theFirstItem)
          : key(theKey), startIndex(theStartIndex), length(theLength), firstItem(theFirstItem) {}
      bool operator<(const ChannelRegistryItem& other) const {
        return ((this->key != other.key) ? (this->key < other.key) : (this->firstItem < other.firstItem));
      }
      uint32_t key;         //ID of DetSet in DetSetVector
      uint32_t startIndex;  //index of first item in data container
      uint16_t length;      //number of items
      uint16_t firstItem;   //index of first item in final DetSet
    };
    typedef std::vector<ChannelRegistryItem> Registry;
    typedef std::vector<T> Data;

    Registry registry_;
    Data data_;
  };

  template <typename T, bool dsvIsSparse>
  inline DetSetVectorFiller<T, dsvIsSparse>::DetSetVectorFiller(const size_t registrySize, const size_t dataSize) {
    registry_.reserve(registrySize);
    data_.reserve(dataSize);
  }

  template <typename T, bool dsvIsSparse>
  inline DetSetVectorFiller<T, dsvIsSparse>::~DetSetVectorFiller() {}

  template <typename T, bool dsvIsSparse>
  inline void DetSetVectorFiller<T, dsvIsSparse>::newChannel(const uint32_t key, const uint16_t firstItem) {
    registry_.push_back(ChannelRegistryItem(key, data_.size(), 0, firstItem));
  }

  template <typename T, bool dsvIsSparse>
  inline void DetSetVectorFiller<T, dsvIsSparse>::addItem(const T& item) {
    data_.push_back(item);
    registry_.back().length++;
  }

  template <typename T, bool dsvIsSparse>
  std::unique_ptr<edm::DetSetVector<T> > DetSetVectorFiller<T, dsvIsSparse>::createDetSetVector() {
    std::sort(registry_.begin(), registry_.end());
    std::vector<edm::DetSet<T> > sorted_and_merged;
    sorted_and_merged.reserve(registry_.size());
    typename Registry::const_iterator iReg = registry_.begin();
    const typename Registry::const_iterator endReg = registry_.end();
    while (iReg != endReg) {
      sorted_and_merged.push_back(edm::DetSet<T>(iReg->key));
      std::vector<T>& detSetData = sorted_and_merged.back().data;
      typename Registry::const_iterator jReg;
      if (dsvIsSparse) {
        uint16_t length = 0;
        for (jReg = iReg; (jReg != endReg) && (jReg->key == iReg->key); ++jReg)
          length += jReg->length;
        detSetData.reserve(length);
        for (jReg = iReg; (jReg != endReg) && (jReg->key == iReg->key); ++jReg) {
          detSetData.insert(
              detSetData.end(), data_.begin() + jReg->startIndex, data_.begin() + jReg->startIndex + jReg->length);
        }
      } else {
        uint16_t detLength = 0;
        uint16_t firstItemOfLastChannel = 0;
        for (jReg = iReg; (jReg != endReg) && (jReg->key == iReg->key); ++jReg) {
          if (!detLength)
            detLength = jReg->length;
          else if (detLength != jReg->length)
            throw cms::Exception("DetSetVectorFiller")
                << "Cannot fill non-sparse DetSet if channels are not unformly sized.";
          firstItemOfLastChannel = jReg->firstItem;
        }
        detSetData.resize(firstItemOfLastChannel + detLength);
        for (jReg = iReg; (jReg != endReg) && (jReg->key == iReg->key); ++jReg) {
          std::copy(data_.begin() + jReg->startIndex,
                    data_.begin() + jReg->startIndex + jReg->length,
                    detSetData.begin() + jReg->firstItem);
        }
      }
      iReg = jReg;
    }
    return typename std::unique_ptr<edm::DetSetVector<T> >(new edm::DetSetVector<T>(sorted_and_merged, true));
  }

  typedef DetSetVectorFiller<SiStripRawDigi, false> RawDigiDetSetVectorFiller;
  typedef DetSetVectorFiller<SiStripDigi, true> ZSDigiDetSetVectorFiller;
}  // namespace sistrip

#endif  // EventFilter_SiStripRawToDigi_SiStripDetSetVectorFiller_H
