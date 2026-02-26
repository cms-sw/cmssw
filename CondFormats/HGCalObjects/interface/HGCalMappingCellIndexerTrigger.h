#ifndef CondFormats_HGCalObjects_interface_HGCalMappingCellIndexerTrigger_h
#define CondFormats_HGCalObjects_interface_HGCalMappingCellIndexerTrigger_h

#include <iostream>
#include <cstdint>
#include <vector>
#include <numeric>
#include "CondFormats/Serialization/interface/Serializable.h"
#include "CondFormats/HGCalObjects/interface/HGCalDenseIndexerBase.h"

/**
   @short utility class to assign dense readout trigger cell indexing
 */
class HGCalMappingCellIndexerTrigger {
public:
  // typedef HGCalDenseIndexerBase WaferDenseIndexerBase;

  HGCalMappingCellIndexerTrigger() = default;

  /**
     adds to map of type codes (= module types) to handle and updatest the max. number of eRx
   */
  void processNewCell(std::string const& typecode, uint16_t ROC, uint16_t trLink, uint16_t trCell) {
    // Skip if trLink, trCell not connected
    if (trLink == uint16_t(-1) || trCell == uint16_t(-1))
      return;
    //assign index to this typecode and resize the max vectors
    if (typeCodeIndexer_.count(typecode) == 0) {
      typeCodeIndexer_[typecode] = typeCodeIndexer_.size();
      maxROC_.resize(typeCodeIndexer_.size(), 0);
      maxTrLink_.resize(typeCodeIndexer_.size(), 0);
      maxTCPerLink_.resize(typeCodeIndexer_.size(), 0);
    }

    size_t idx = typeCodeIndexer_[typecode];

    /*High density modules have links {0, 2} and low {0, 1, 2, 3} so to make it work I need to divide by 2*/
    if (typecode[1] == 'H')
      trLink /= 2;
    /*SiPM tiles have trCells indexed from 1 instead from 0*/
    if (typecode[0] == 'T')
      trCell--;
    maxROC_[idx] = std::max(maxROC_[idx], static_cast<uint16_t>(ROC + 1));
    maxTrLink_[idx] = std::max(maxTrLink_[idx], static_cast<uint16_t>(trLink + 1));
    maxTCPerLink_[idx] = std::max(maxTCPerLink_[idx], static_cast<uint16_t>(trCell + 1));
  }

  /**
     @short process the current list of type codes handled and updates the dense indexers
  */
  void update() {
    uint32_t n = typeCodeIndexer_.size();
    offsets_ = std::vector<uint32_t>(n, 0);
    di_ = std::vector<HGCalDenseIndexerBase>(n, HGCalDenseIndexerBase(3)); /* The indices are {ROC, trLink, TC}*/
    for (uint32_t idx = 0; idx < n; idx++) {
      uint16_t maxROCs = maxROC_[idx];
      uint16_t maxLinks = maxTrLink_[idx];
      uint16_t maxTCPerLink = maxTCPerLink_[idx];
      di_[idx].updateRanges({{maxROCs, maxLinks, maxTCPerLink}});
      if (idx < n - 1)
        offsets_[idx + 1] = di_[idx].maxIndex() + offsets_[idx];
    }
  }

  /**
     @short gets index given typecode string
   */
  size_t getEnumFromTypecode(std::string const& typecode) const {
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
  HGCalDenseIndexerBase getDenseIndexFor(std::string const& typecode) const {
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
  uint16_t denseIndex(std::string const& typecode, uint32_t ROC, uint32_t trLink, uint32_t trCell) const {
    return denseIndex(getEnumFromTypecode(typecode), ROC, trLink, trCell);
  }
  uint32_t denseIndex(size_t idx, uint32_t ROC, uint32_t trLink, uint32_t trCell) const {
    return di_[idx].denseIndex({{ROC, trLink, trCell}}) + offsets_[idx];
  }

  /**
     @short returns the max. dense index expected
   */
  uint32_t maxDenseIndex() const {
    size_t i = maxTrLink_.size();
    if (i == 0)
      return 0;
    return offsets_.back() + maxROC_.back() * maxTrLink_.back() * maxTCPerLink_.back();
  }

  /**
     @short gets the number of words (cells) for a given typecode
     Note : some partials are rounded to the closest multiplie of 16 or 8 depending on the density
     That is done because not all the TrigLinks,TrgCells possible are assigned in practice
     e.g.: ML-T has 22 TCs but this will return 32 or MH-T has 19 but it will return 24
     It results in a small mem overhead over the totall memory needed to be allocated
  */
  size_t getNWordsExpectedFor(std::string const& typecode) const {
    auto it = getEnumFromTypecode(typecode);
    return getNWordsExpectedFor(it);
  }
  size_t getNWordsExpectedFor(size_t typecodeidx) const { return getDenseIndexerFor(typecodeidx).maxIndex(); }

  /**
     @short gets the number of Trigger Links for a given typecode
  */
  size_t getNTrLinkExpectedFor(std::string const& typecode) const {
    auto it = getEnumFromTypecode(typecode);
    return getNTrLinkExpectedFor(it);
  }
  size_t getNTrLinkExpectedFor(size_t typecodeidx) const { return maxTrLink_[typecodeidx] * maxROC_[typecodeidx]; }

  std::unordered_map<std::string, size_t> typeCodeIndexer_;
  std::vector<uint16_t> maxROC_;
  std::vector<uint16_t> maxTrLink_;
  std::vector<uint16_t> maxTCPerLink_;
  std::vector<uint32_t> offsets_;
  std::vector<HGCalDenseIndexerBase> di_;

  ~HGCalMappingCellIndexerTrigger() = default;

  COND_SERIALIZABLE;
};

#endif
