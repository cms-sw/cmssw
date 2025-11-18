#ifndef CondFormats_HGCalObjects_interface_HGCalMappingParameterIndex_h
#define CondFormats_HGCalObjects_interface_HGCalMappingParameterIndex_h

#include <cstdint>
#include <vector>
#include <map>
#include <algorithm>  // for std::min
#include <utility>    // for std::pair, std::make_pair
#include <iterator>   // for std::next and std::advance

#include "DataFormats/HGCalDigi/interface/HGCalElectronicsId.h"
#include "CondFormats/Serialization/interface/Serializable.h"
#include "CondFormats/HGCalObjects/interface/HGCalDenseIndexerBase.h"
#include "CondFormats/HGCalObjects/interface/HGCalMappingCellIndexer.h"
#include "FWCore/Utilities/interface/Exception.h"

/**
 * @short this structure holds the indices and types in the readout sequence
 * as the 12 capture blocks may not all be used and the each capture block may also be under-utilized
 * a lookup table is used to hold the compact index
 */
struct HGCalFEDReadoutSequence {
  uint32_t id;
  //overall counters for total number of ECONs and Capture Blocks (useful to steer the unpacker behavior)
  size_t totalECONs_, totalCBs_;
  /// look-up table (capture block, econd idx) -> internal dense index
  std::vector<int> moduleLUT_;
  /// dense sequence of modules in the readout: the type is the one in use in the cell mapping
  std::vector<int> readoutTypes_;
  /// dense sequence of offsets for modules, e-Rx and channel data
  std::vector<uint32_t> modOffsets_, erxOffsets_, chDataOffsets_, enabledErx_;
  COND_SERIALIZABLE;
};

/**
 * @short utility class to assign dense readout module indexing
 * the class holds the information on the expected readout sequence (module types) per FED and their offset in the SoAs of data
 */
class HGCalMappingModuleIndexer {
public:
  HGCalMappingModuleIndexer() : modFedIndexer_({maxCBperFED_, maxECONDperCB_}) {}

  ~HGCalMappingModuleIndexer() = default;

  /**
   * @short for a new module it adds it's type to the readaout sequence vector
   * if the fed id is not yet existing in the mapping it's added
   * a dense indexer is used to create the necessary indices for the new module
   * unused indices will be set with -1
   */
  void processNewModule(uint32_t fedid,
                        uint16_t captureblockIdx,
                        uint16_t econdIdx,
                        uint32_t typecodeIdx,
                        uint32_t nerx,
                        uint32_t nwords,
                        std::string const &typecode);

  /**
   * @short to be called after all the modules have been processed
   */
  void finalize();

  /**
   * @short decode silicon or sipm type and cell type for the detector id
   * from the typecode string: "M[LH]-X[123]X-*" for Si, "T[LH]-L*S*[PN]" for SiPm
   */
  static std::pair<bool, int8_t> getCellType(std::string_view typecode);

  /**
   * @short returns the index for the n-th module in the readout sequence of a FED
   * if the index in the readout sequence is unknown alternative methods which take the (capture block, econd idx) are provided
   * which will find first what should be the internal dense index (index in the readout sequence)
   */
  uint32_t getIndexForModule(uint32_t fedid, uint32_t modid) const {
    return fedReadoutSequences_[fedid].modOffsets_[modid];
  };
  uint32_t getIndexForModule(uint32_t fedid, uint16_t captureblockIdx, uint16_t econdIdx) const {
    uint32_t modid = denseIndexingFor(fedid, captureblockIdx, econdIdx);
    return getIndexForModule(fedid, modid);
  };
  //uint32_t getIndexForModule(HGCalElectronicsId id) const {
  //  return getIndexForModule(id.localFEDId(),id.captureBlock(),id.econdIdx());
  //};
  uint32_t getIndexForModule(std::string const &typecode) const {
    const auto &[fedid, modid] = getIndexForFedAndModule(typecode);  // (fedId,modId)
    return getIndexForModule(fedid, modid);
  };
  uint32_t getIndexForModuleErx(uint32_t fedid, uint32_t modid, uint32_t erxidx) const {
    return fedReadoutSequences_[fedid].erxOffsets_[modid] + erxidx;
  };
  uint32_t getIndexForModuleErx(uint32_t fedid, uint16_t captureblockIdx, uint16_t econdIdx, uint32_t erxidx) const {
    uint32_t modid = denseIndexingFor(fedid, captureblockIdx, econdIdx);
    return getIndexForModuleErx(fedid, modid, erxidx);
  }
  uint32_t getIndexForModuleData(uint32_t fedid, uint32_t modid, uint32_t erxidx, uint32_t chidx) const {
    return fedReadoutSequences_[fedid].chDataOffsets_[modid] + erxidx * HGCalMappingCellIndexer::maxChPerErx_ + chidx;
  };
  uint32_t getIndexForModuleData(
      uint32_t fedid, uint16_t captureblockIdx, uint16_t econdIdx, uint32_t erxidx, uint32_t chidx) const {
    uint32_t modid = denseIndexingFor(fedid, captureblockIdx, econdIdx);
    return getIndexForModuleData(fedid, modid, erxidx, chidx);
  };
  uint32_t getIndexForModuleData(HGCalElectronicsId id) const {
    return id.isCM() ? getIndexForModuleErx(id.localFEDId(), id.captureBlock(), id.econdIdx(), id.econdeRx())
                     : getIndexForModuleData(
                           id.localFEDId(), id.captureBlock(), id.econdIdx(), id.econdeRx(), id.halfrocChannel());
  };
  uint32_t getIndexForModuleData(std::string const &typecode) const {
    const auto &[fedid, modid] = getIndexForFedAndModule(typecode);
    return getIndexForModuleData(fedid, modid, 0, 0);
  };
  std::pair<uint32_t, uint32_t> getIndexForFedAndModule(std::string const &typecode) const;

  /**
   * @short return number maximum index of FED, ECON-D Module, eRx ROC
   */
  uint32_t numFEDs() const {
    return count_if(fedReadoutSequences_.begin(), fedReadoutSequences_.end(), [](auto fedrs) {
      return fedrs.readoutTypes_.size() != 0;
    });
  }  ///< return total number of FEDs that actually exist
  uint32_t maxFEDSize() const {
    return fedReadoutSequences_.size();
  }  ///< maximum FED index (fedReadoutSequences_ includes non existing FED IDs)
  uint32_t maxModuleSize() const {
    return maxModulesIdx_;
  }  ///< total number of ECON-Ds (useful for setting ECON-D SoA size)
  uint32_t getNumModules(uint32_t fedid) const {
    return fedReadoutSequences_[fedid].readoutTypes_.size();
  }  ///< number of ECON-Ds for given FED id
  uint32_t maxERxSize() const {
    return maxErxIdx_;
  }  ///< total number of eRx half-ROCs (useful for setting config SoA size)
  uint32_t getNumERxs(uint32_t fedid, uint32_t modid) const {
    auto modtype_val = fedReadoutSequences_[fedid].readoutTypes_[modid];
    return globalTypesNErx_[modtype_val];
  }  ///< number of eRx half-ROCs for given FED & ECON-D ids
  uint32_t getNumERxs(std::string const &typecode) const {
    const auto &[fedid, modid] = getIndexForFedAndModule(typecode);
    return getNumERxs(fedid, modid);
  }  ///< number of eRx half-ROCs for a given ECON-D typecode
  uint32_t maxDataSize() const {
    return maxDataIdx_;
  }  ///< total number of channels (useful for setting calib SoA size)
  uint32_t getNumChannels(uint32_t fedid, uint32_t modid) const {
    return HGCalMappingCellIndexer::maxChPerErx_ * getNumERxs(fedid, modid);
  }  ///< total number of channels for given FED & ECON-D ids
  uint32_t getNumChannels(std::string const &typecode) const {
    const auto &[fedid, modid] = getIndexForFedAndModule(typecode);
    return getNumChannels(fedid, modid);
  }  ///< total number of channels for a given ECON-D typecode

  /**
   * @short return type ECON-D Module
   */
  int getTypeForModule(uint32_t fedid, uint32_t modid) const {
    return fedReadoutSequences_[fedid].readoutTypes_[modid];
  }
  int getTypeForModule(uint32_t fedid, uint16_t captureblockIdx, uint16_t econdIdx) const {
    uint32_t modid = denseIndexingFor(fedid, captureblockIdx, econdIdx);
    return getTypeForModule(fedid, modid);
  }

  /**
   * @short getters for private members
   */
  HGCalDenseIndexerBase const &fedIndexer() const { return modFedIndexer_; }
  std::vector<HGCalFEDReadoutSequence> const &fedReadoutSequences() const { return fedReadoutSequences_; }
  std::vector<uint32_t> const &globalTypesCounter() const { return globalTypesCounter_; }
  std::vector<uint32_t> const &globalTypesNErx() const { return globalTypesNErx_; }
  std::vector<uint32_t> const &globalTypesNWords() const { return globalTypesNWords_; }
  std::vector<uint32_t> const &moduleOffsets() const { return moduleOffsets_; }
  std::vector<uint32_t> const &erxOffsets() const { return erxOffsets_; }
  std::vector<uint32_t> const &dataOffsets() const { return dataOffsets_; }
  uint32_t fedCount() const { return nfeds_; }
  uint32_t maxDataIndex() const { return maxDataIdx_; }
  uint32_t maxErxIndex() const { return maxErxIdx_; }
  uint32_t maxModulesIndex() const { return maxModulesIdx_; }
  std::map<std::string, std::pair<uint32_t, uint32_t>> const &typecodeMap() const { return typecodeMap_; }
  uint32_t maxModulesCount() const { return maxModulesCount_; }

  /// max number of main buffers/capture blocks per FED
  constexpr static uint32_t maxCBperFED_ = 10;
  /// max number of ECON-Ds processed by a main buffer/capture block
  constexpr static uint32_t maxECONDperCB_ = 12;

private:
  /// internal indexer
  HGCalDenseIndexerBase modFedIndexer_;
  /// the sequence of FED readout sequence descriptors
  std::vector<HGCalFEDReadoutSequence> fedReadoutSequences_;
  /// global counters for types of modules, number of e-Rx and words
  std::vector<uint32_t> globalTypesCounter_, globalTypesNErx_, globalTypesNWords_;
  /// base offsets to apply per module type with different granularity : module, e-Rx, channel data
  std::vector<uint32_t> moduleOffsets_, erxOffsets_, dataOffsets_;
  /// global counters (sizes of vectors)
  uint32_t nfeds_, maxDataIdx_, maxErxIdx_, maxModulesIdx_, maxModulesCount_;
  /// map from module type code string to (fedIdx,modIdx) pair (implemented to retrieve dense index offset)
  std::map<std::string, std::pair<uint32_t, uint32_t>> typecodeMap_;

  /**
   * @short given capture block and econd indices returns the dense indexer
   */
  uint32_t denseIndexingFor(uint32_t fedid, uint16_t captureblockIdx, uint16_t econdIdx) const {
    if (fedid > nfeds_)
      throw cms::Exception("ValueError") << "FED ID=" << fedid << " is unknown to current mapping";
    uint32_t idx = modFedIndexer_.denseIndex({{captureblockIdx, econdIdx}});
    auto dense_idx = fedReadoutSequences_[fedid].moduleLUT_[idx];
    if (dense_idx < 0)
      throw cms::Exception("ValueError") << "FED ID=" << fedid << " capture block=" << captureblockIdx
                                         << " econ=" << econdIdx << "has not been assigned a dense indexing"
                                         << std::endl;
    return uint32_t(dense_idx);
  }

  /**
   * @short when finalize is called, empty entries are removed and they may need to be re-assigned for the real final number of modules
   */
  void reassignTypecodeLocation(uint32_t fedid, uint32_t cur_modIdx, uint32_t new_modIx) {
    std::pair<uint32_t, uint32_t> val(fedid, cur_modIdx), newval(fedid, new_modIx);
    for (const auto &it : typecodeMap_) {
      if (it.second != val)
        continue;
      typecodeMap_[it.first] = newval;
      break;
    }
  }

  COND_SERIALIZABLE;
};

#endif
