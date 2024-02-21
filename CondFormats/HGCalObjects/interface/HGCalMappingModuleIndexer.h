#ifndef CondFormats_HGCalObjects_interface_HGCalMappingParameterIndex_h
#define CondFormats_HGCalObjects_interface_HGCalMappingParameterIndex_h

#include <cstdint>
#include <vector>
#include <algorithm>

#include "DataFormats/HGCalDigi/interface/HGCalElectronicsId.h"
#include "CondFormats/Serialization/interface/Serializable.h"
#include "CondFormats/HGCalObjects/interface/HGCalDenseIndexerBase.h"
#include "CondFormats/HGCalObjects/interface/HGCalMappingCellIndexer.h"
#include "FWCore/Utilities/interface/Exception.h"

/**
   @short this structure holds the indices and types in the readout sequence
   as the 12 capture blocks may not all be used and the each capture block may also be under-utilized
   a lookup table is used to hold the compact index
 */
struct HGCalFEDReadoutSequence_t {
  uint32_t id;
  ///>look-up table (capture block, econd idx) -> internal dense index
  std::vector<int> moduleLUT_;
  ///>dense sequence of modules in the readout: the type is the one in use in the cell mapping
  std::vector<int> readoutTypes_;
  ///>dense sequence of offsets for modules, e-Rx and channel data
  std::vector<uint32_t> modOffsets_, erxOffsets_, chDataOffsets_;
  COND_SERIALIZABLE;
};

/**
   @short utility class to assign dense readout module indexing
   the class holds the information on the expected readout sequence (module types) per FED and their offset in the SoAs of data
 */
class HGCalMappingModuleIndexer {
public:
  HGCalMappingModuleIndexer() : modFedIndexer_({maxCBperFED_, maxECONDperCB_}) {}

  ~HGCalMappingModuleIndexer() = default;

  /**
     @short for a new module it adds it's type to the readaout sequence vector
     if the fed id is not yet existing in the mapping it's added
     a dense indexer is used to create the necessary indices for the new module
     unused indices will be set with -1
   */
  void processNewModule(uint32_t fedid,
                        uint16_t captureblockIdx,
                        uint16_t econdIdx,
                        uint32_t typecodeIdx,
                        uint32_t nerx,
                        uint32_t nwords) {
    //add fed if needed
    if (fedid >= fedReadoutSequences_.size()) {
      fedReadoutSequences_.resize(fedid + 1);
    }
    HGCalFEDReadoutSequence_t &frs = fedReadoutSequences_[fedid];
    frs.id = fedid;

    //assign position, resize if needed, and fill the type code
    uint32_t idx = modFedIndexer_.denseIndex({{captureblockIdx, econdIdx}});
    if (idx >= frs.readoutTypes_.size()) {
      frs.readoutTypes_.resize(idx + 1, -1);
    }
    frs.readoutTypes_[idx] = typecodeIdx;

    //count another typecodein the global list
    if (typecodeIdx >= globalTypesCounter_.size()) {
      globalTypesCounter_.resize(typecodeIdx + 1, 0);
      globalTypesNErx_.resize(typecodeIdx + 1, 0);
      globalTypesNWords_.resize(typecodeIdx + 1, 0);
      dataOffsets_.resize(typecodeIdx + 1, 0);
    }
    globalTypesCounter_[typecodeIdx]++;
    globalTypesNErx_[typecodeIdx] = nerx;
    globalTypesNWords_[typecodeIdx] = nwords;
  }

  /**
     @short
   */
  void finalize() {
    //max indices at different levels
    nfeds_ = fedReadoutSequences_.size();
    maxModulesIdx_ = std::accumulate(globalTypesCounter_.begin(), globalTypesCounter_.end(), 0);
    maxErxIdx_ =
        std::inner_product(globalTypesCounter_.begin(), globalTypesCounter_.end(), globalTypesNErx_.begin(), 0);
    maxDataIdx_ =
        std::inner_product(globalTypesCounter_.begin(), globalTypesCounter_.end(), globalTypesNWords_.begin(), 0);

    //compute the global offset to assign per board type, eRx and channel data
    moduleOffsets_.resize(maxModulesIdx_, 0);
    erxOffsets_.resize(maxModulesIdx_, 0);
    dataOffsets_.resize(maxModulesIdx_, 0);
    for (size_t i = 1; i < globalTypesCounter_.size(); i++) {
      moduleOffsets_[i] = globalTypesCounter_[i - 1];
      erxOffsets_[i] = globalTypesCounter_[i - 1] * globalTypesNErx_[i - 1];
      dataOffsets_[i] = globalTypesCounter_[i - 1] * globalTypesNWords_[i - 1];
    }
    std::partial_sum(moduleOffsets_.begin(), moduleOffsets_.end(), moduleOffsets_.begin());
    std::partial_sum(erxOffsets_.begin(), erxOffsets_.end(), erxOffsets_.begin());
    std::partial_sum(dataOffsets_.begin(), dataOffsets_.end(), dataOffsets_.begin());

    //now go through the FEDs and ascribe the offsets per module in the readout sequence
    std::vector<uint32_t> typeCounters(globalTypesCounter_.size(), 0);
    for (auto &fedit : fedReadoutSequences_) {
      //assign the indexing in the look-up table
      size_t nconn(0);
      fedit.moduleLUT_.resize(fedit.readoutTypes_.size(), -1);
      for (size_t i = 0; i < fedit.readoutTypes_.size(); i++) {
        if (fedit.readoutTypes_[i] == -1)
          continue;  //unexisting
        fedit.moduleLUT_[i] = nconn;
        nconn++;
      }

      //remove unexisting ECONs building a final compact readout sequence
      std::remove_if(
          fedit.readoutTypes_.begin(), fedit.readoutTypes_.end(), [&](int val) -> bool { return val == -1; });

      //assign the final offsets at the different levels
      size_t nmods = fedit.readoutTypes_.size();
      fedit.modOffsets_.resize(nmods, 0);
      fedit.erxOffsets_.resize(nmods, 0);
      fedit.chDataOffsets_.resize(nmods, 0);
      for (size_t i = 0; i < nmods; i++) {
        uint32_t type_val = fedit.readoutTypes_[i];

        //module offset : global offset for this type + current index for this type
        uint32_t baseMod_offset = moduleOffsets_[type_val] + typeCounters[type_val];
        fedit.modOffsets_[i] = baseMod_offset;  // + internalMod_offset;

        //erx-level offset : global offset of e-Rx of this type + #e-Rrx * current index for this type
        uint32_t baseErx_offset = erxOffsets_[type_val];
        uint32_t internalErx_offset = globalTypesNErx_[type_val] * typeCounters[type_val];
        fedit.erxOffsets_[i] = baseErx_offset + internalErx_offset;

        //channel data offset: global offset for data of this type + #words * current index for this type
        uint32_t baseData_offset = dataOffsets_[type_val];
        uint32_t internalData_offset = globalTypesNWords_[type_val] * typeCounters[type_val];
        fedit.chDataOffsets_[i] = baseData_offset + internalData_offset;

        typeCounters[type_val]++;
      }
    }
  }

  /**
     @short decodes silicon or sipm type and cell type for the detector id 
     from the typecode string
   */
  static std::pair<bool, int> convertTypeCode(std::string_view typecode) {
    if (typecode.size() < 5)
      throw cms::Exception("InvalidHGCALTypeCode") << typecode << " is invalid for decoding readout cell type";

    bool isSiPM = {typecode.find("TM") != std::string::npos ? true : false};
    int celltype;
    if (isSiPM) {
      celltype = 0;  // Assign SiPM type coarse or molded with next version of modulelocator
    } else {
      celltype = {typecode[4] == '1' ? 0 : typecode[4] == '2' ? 1 : 2};
    }
    return std::pair<bool, bool>(isSiPM, celltype);
  }

  /**
     @short returns the index for the n-th module in the readout sequence of a FED
     if the index in the readout sequence is unknown alternative methods which take the (capture block, econd idx) are provided
     which will find first what should be the internal dense index (index in the readout sequence)
   */
  uint32_t getIndexForModule(uint32_t fedid, uint32_t nmod) const {
    return fedReadoutSequences_[fedid].modOffsets_[nmod];
  };
  uint32_t getIndexForModule(uint32_t fedid, uint16_t captureblockIdx, uint16_t econdIdx) const {
    uint32_t nmod = denseIndexingFor(fedid, captureblockIdx, econdIdx);
    return getIndexForModule(fedid, nmod);
  };
  uint32_t getIndexForModuleErx(uint32_t fedid, uint32_t nmod, uint32_t erxidx) const {
    return fedReadoutSequences_[fedid].erxOffsets_[nmod] + erxidx;
  };
  uint32_t getIndexForModuleErx(uint32_t fedid, uint16_t captureblockIdx, uint16_t econdIdx, uint32_t erxidx) const {
    uint32_t nmod = denseIndexingFor(fedid, captureblockIdx, econdIdx);
    return getIndexForModuleErx(fedid, nmod, erxidx);
  }
  uint32_t getIndexForModuleData(uint32_t fedid, uint32_t nmod, uint32_t erxidx, uint32_t chidx) const {
    return fedReadoutSequences_[fedid].chDataOffsets_[nmod] + erxidx * HGCalMappingCellIndexer::maxChPerErx_ + chidx;
  };
  uint32_t getIndexForModuleData(
      uint32_t fedid, uint16_t captureblockIdx, uint16_t econdIdx, uint32_t erxidx, uint32_t chidx) const {
    uint32_t nmod = denseIndexingFor(fedid, captureblockIdx, econdIdx);
    return getIndexForModuleData(fedid, nmod, erxidx, chidx);
  };

  int getTypeForModule(uint32_t fedid, uint32_t nmod) const { return fedReadoutSequences_[fedid].readoutTypes_[nmod]; }
  int getTypeForModule(uint32_t fedid, uint16_t captureblockIdx, uint16_t econdIdx) const {
    uint32_t nmod = denseIndexingFor(fedid, captureblockIdx, econdIdx);
    return getTypeForModule(fedid, nmod);
  }

  ///< internal indexer
  HGCalDenseIndexerBase modFedIndexer_;
  ///< the sequence of FED readout sequence descriptors
  std::vector<HGCalFEDReadoutSequence_t> fedReadoutSequences_;
  ///< global counters for types of modules, number of e-Rx and words
  std::vector<uint32_t> globalTypesCounter_, globalTypesNErx_, globalTypesNWords_;
  ///< base offsets to apply per module type with different granularity : module, e-Rx, channel data
  std::vector<uint32_t> moduleOffsets_, erxOffsets_, dataOffsets_;
  ///< global counters (sizes of vectors)
  uint32_t nfeds_, maxDataIdx_, maxErxIdx_, maxModulesIdx_;

  ///< max number of main buffers/capture blocks per FED
  constexpr static uint32_t maxCBperFED_ = 10;
  ///< max number of ECON-Ds processed by a main buffer/capture block
  constexpr static uint32_t maxECONDperCB_ = 12;

private:
  /**
     @short given capture block and econd indices returns the dense indexer
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

  COND_SERIALIZABLE;
};

#endif
