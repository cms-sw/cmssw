#ifndef CondFormats_HGCalObjects_interface_HGCalMappingCellParameterIndex_h
#define CondFormats_HGCalObjects_interface_HGCalMappingCellParameterIndex_h

#include <iostream>
#include <cstdint>
#include <vector>
#include <numeric>
#include "DataFormats/HGCalDigi/interface/HGCalElectronicsId.h"
#include "CondFormats/Serialization/interface/Serializable.h"
#include "CondFormats/HGCalObjects/interface/HGCalDenseIndexerBase.h"

/**
   @short utility class to assign dense readout cell indexing
 */
class HGCalMappingCellIndexer {
public:
  // typedef HGCalDenseIndexerBase WaferDenseIndexerBase;

  HGCalMappingCellIndexer() = default;

  /**
     adds to map of type codes (= module types) to handle and updatest the max. number of eRx 
   */
  void processNewCell(std::string typecode, uint16_t chip, uint16_t half) {
    //assign index to this typecode and resize the max e-Rx vector
    if (typeCodeIndexer_.count(typecode) == 0) {
      typeCodeIndexer_[typecode] = typeCodeIndexer_.size();
      maxErx_.resize(typeCodeIndexer_.size(), 0);
    }

    size_t idx = typeCodeIndexer_[typecode];
    uint16_t erx = chip * 2 + half + 1;  //use the number not the index here
    maxErx_[idx] = std::max(maxErx_[idx], erx);
  }

  /**
     @short process the current list of type codes handled and updates the dense indexers
  */
  void update() {
    uint32_t n = typeCodeIndexer_.size();
    offsets_ = std::vector<uint32_t>(n, 0);
    di_ = std::vector<HGCalDenseIndexerBase>(n, HGCalDenseIndexerBase(2));
    for (uint32_t idx = 0; idx < n; idx++) {
      uint16_t nerx = maxErx_[idx];
      di_[idx].updateRanges({{nerx, maxChPerErx_}});
      if (idx < n - 1)
        offsets_[idx + 1] = di_[idx].getMaxIndex();
    }

    //accumulate the offsets in the array
    std::partial_sum(offsets_.begin(), offsets_.end(), offsets_.begin());
  }

  /**
     @short gets index given typecode string
   */
  size_t getEnumFromTypecode(std::string typecode) const {
    auto it = typeCodeIndexer_.find(typecode);
    if (it == typeCodeIndexer_.end())
      throw cms::Exception("ValueError") << " unable to find typecode=" << typecode << " in cell indexer";
    return it->second;
  }

  /**
     @short checks if there is a typecode corresponding to an index
   */
  std::string getTypecodeFromEnum(size_t idx) const {
    for (const auto& it : typeCodeIndexer_)
      if (it.second == idx)
        return it.first;
    throw cms::Exception("ValueError") << " unable to find typecode corresponding to idx=" << idx;
  }

  /**
     @short returns the dense indexer for a typecode
   */
  HGCalDenseIndexerBase getDenseIndexFor(std::string typecode) const {
    return getDenseIndexerFor(getEnumFromTypecode(typecode));
  }

  /**
     @short returns the dense indexer for a given internal index
  */
  HGCalDenseIndexerBase getDenseIndexerFor(size_t idx) const {
    if (idx >= di_.size())
      throw cms::Exception("ValueError") << " index requested for cell dense indexer (i=" << idx
                                         << ") is larger than allocated";
    return di_[idx];
  }

  /**
     @short builders for the dense index
   */
  uint32_t denseIndex(std::string typecode, uint32_t chip, uint32_t half, uint32_t seq) const {
    return denseIndex(getEnumFromTypecode(typecode), chip, half, seq);
  }
  uint32_t denseIndex(std::string typecode, uint32_t erx, uint32_t seq) const {
    return denseIndex(getEnumFromTypecode(typecode), erx, seq);
  }
  uint32_t denseIndex(size_t idx, uint32_t chip, uint32_t half, uint32_t seq) const {
    uint16_t erx = chip * maxHalfPerROC_ + half;
    return denseIndex(idx, erx, seq);
  }
  uint32_t denseIndex(size_t idx, uint32_t erx, uint32_t seq) const {
    return di_[idx].denseIndex({{erx, seq}}) + offsets_[idx];
  }

  /**
     @short decodes the dense index code
   */
  uint32_t elecIdFromIndex(uint32_t rtn, std::string typecode) const {
    return elecIdFromIndex(rtn, getEnumFromTypecode(typecode));
  }
  uint32_t elecIdFromIndex(uint32_t rtn, size_t idx) const {
    if (idx >= di_.size())
      throw cms::Exception("ValueError") << " index requested for cell dense indexer (i=" << idx
                                         << ") is larger than allocated";
    rtn -= offsets_[idx];
    auto rtn_codes = di_[idx].unpackDenseIndex(rtn);
    return HGCalElectronicsId(false, 0, 0, 0, rtn_codes[0], rtn_codes[1]).raw();
  }

  /**
     @short returns the max. dense index expected
   */
  uint32_t maxDenseIndex() const {
    size_t i = maxErx_.size();
    if (i == 0)
      return 0;
    return offsets_.back() + maxErx_.back() * maxChPerErx_;
  }

  /**
     @short gets the number of words for a given typecode
  */
  size_t getNWordsExpectedFor(std::string typecode) const {
    auto it = getEnumFromTypecode(typecode);
    return getNWordsExpectedFor(it);
  }
  size_t getNWordsExpectedFor(size_t typecodeidx) const { return maxErx_[typecodeidx] * maxChPerErx_; }

  /**
     @short gets the number of e-Rx for a given typecode
  */
  size_t getNErxExpectedFor(std::string typecode) const {
    auto it = getEnumFromTypecode(typecode);
    return getNErxExpectedFor(it);
  }
  size_t getNErxExpectedFor(size_t typecodeidx) const { return maxErx_[typecodeidx]; }

  constexpr static char maxHalfPerROC_ = 2;
  constexpr static uint16_t maxChPerErx_ = 37;  //36 channels + 1 calib

  std::map<std::string, size_t> typeCodeIndexer_;
  std::vector<uint16_t> maxErx_;
  std::vector<uint32_t> offsets_;
  std::vector<HGCalDenseIndexerBase> di_;

  ~HGCalMappingCellIndexer() {}

  COND_SERIALIZABLE;
};

#endif
