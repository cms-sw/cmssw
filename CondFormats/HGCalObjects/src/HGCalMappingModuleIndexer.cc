#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "CondFormats/HGCalObjects/interface/HGCalMappingModuleIndexer.h"
#include "DataFormats/ForwardDetId/interface/HGCSiliconDetId.h"       // for HGCSiliconDetId::waferType
#include "DataFormats/ForwardDetId/interface/HGCScintillatorDetId.h"  // for HGCScintillatorDetId::tileGranularity

/**
 * @short for a new module it adds it's type to the readaout sequence vector
 * if the fed id is not yet existing in the mapping it's added
 * a dense indexer is used to create the necessary indices for the new module
 * unused indices will be set with -1
 */
void HGCalMappingModuleIndexer::processNewModule(uint32_t fedid,
                                                 uint16_t captureblockIdx,
                                                 uint16_t econdIdx,
                                                 uint32_t typecodeIdx,
                                                 uint32_t nerx,
                                                 uint32_t nwords,
                                                 std::string const& typecode) {
  //add fed if needed
  if (fedid >= fedReadoutSequences_.size()) {
    fedReadoutSequences_.resize(fedid + 1);
  }
  HGCalFEDReadoutSequence& frs = fedReadoutSequences_[fedid];
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

  //add typecode string to map to retrieve fedId & modId later
  if (!typecode.empty()) {
    if (typecodeMap_.find(typecode) != typecodeMap_.end()) {     // found key
      const auto& [fedid_, modid_] = typecodeMap_.at(typecode);  // (fedId,modId)
      edm::LogWarning("HGCalMappingModuleIndexer")
          << "Found typecode " << typecode << " already in map (fedid,modid)=(" << fedid_ << "," << modid_
          << ")! Overwriting with (" << fedid << "," << idx << ")...";
    }
    LogDebug("HGCalMappingModuleIndexer")
        << "HGCalMappingModuleIndexer::processNewModule: Adding typecode=\"" << typecode << "\" with fedid=" << fedid
        << ", idx=" << idx << " (will be re-indexed after finalize)";
    typecodeMap_[typecode] = std::make_pair(fedid, idx);
  }
}

/// @short to be called after all the modules have been processed
void HGCalMappingModuleIndexer::finalize() {
  //max indices at different levels
  nfeds_ = fedReadoutSequences_.size();
  maxModulesCount_ = std::accumulate(globalTypesCounter_.begin(), globalTypesCounter_.end(), 0);
  maxModulesIdx_ = globalTypesCounter_.size();
  maxErxIdx_ = std::inner_product(globalTypesCounter_.begin(), globalTypesCounter_.end(), globalTypesNErx_.begin(), 0);
  maxDataIdx_ =
      std::inner_product(globalTypesCounter_.begin(), globalTypesCounter_.end(), globalTypesNWords_.begin(), 0);

  //compute the global offset to assign per board type, eRx and channel data
  moduleOffsets_.resize(maxModulesIdx_, 0);
  erxOffsets_.resize(maxModulesIdx_, 0);
  dataOffsets_.resize(maxModulesIdx_, 0);
  for (size_t i = 1; i < maxModulesIdx_; i++) {
    moduleOffsets_[i] = globalTypesCounter_[i - 1] + moduleOffsets_[i - 1];
    erxOffsets_[i] = globalTypesCounter_[i - 1] * globalTypesNErx_[i - 1] + erxOffsets_[i - 1];
    dataOffsets_[i] = globalTypesCounter_[i - 1] * globalTypesNWords_[i - 1] + dataOffsets_[i - 1];
  }

  //now go through the FEDs and ascribe the offsets per module in the readout sequence
  std::vector<uint32_t> typeCounters(globalTypesCounter_.size(), 0);
  for (auto& fedit : fedReadoutSequences_) {
    //assign the final indexing in the look-up table depending on which ECON-D's are really present
    //count also the the number of capture blocks present
    size_t nconn(0);
    fedit.moduleLUT_.resize(fedit.readoutTypes_.size(), -1);
    std::set<uint32_t> uniqueCB;
    for (size_t i = 0; i < fedit.readoutTypes_.size(); i++) {
      if (fedit.readoutTypes_[i] == -1)
        continue;  //unexisting

      reassignTypecodeLocation(fedit.id, i, nconn);
      fedit.moduleLUT_[i] = nconn;
      nconn++;

      uniqueCB.insert(modFedIndexer_.unpackDenseIndex(i)[0]);
    }
    fedit.totalECONs_ = nconn;
    fedit.totalCBs_ = uniqueCB.size();

    //remove unexisting ECONs building a final compact readout sequence
    fedit.readoutTypes_.erase(
        std::remove_if(
            fedit.readoutTypes_.begin(), fedit.readoutTypes_.end(), [&](int val) -> bool { return val == -1; }),
        fedit.readoutTypes_.end());

    //resize vectors to their final size and set final values
    size_t nmods = fedit.readoutTypes_.size();
    fedit.modOffsets_.resize(nmods, 0);
    fedit.erxOffsets_.resize(nmods, 0);
    fedit.chDataOffsets_.resize(nmods, 0);
    fedit.enabledErx_.resize(nmods, 0);

    for (size_t i = 0; i < nmods; i++) {
      int type_val = fedit.readoutTypes_[i];

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

      //enabled erx flags
      //FIXME: assume all eRx are enabled now
      fedit.enabledErx_[i] = (0b1 << globalTypesNErx_[type_val]) - 0b1;
      typeCounters[type_val]++;
    }
  }
}

/**
 * @short decode silicon or sipm type and cell type for the detector id 
 * from the typecode string: "M[LH]-X[123]X-*" for Si, "T[LH]-L*S*[PN]" for SiPm
 */
std::pair<bool, int8_t> HGCalMappingModuleIndexer::getCellType(std::string_view typecode) {
  if (typecode.size() < 5) {
    cms::Exception ex("InvalidHGCALTypeCode");
    ex << "'" << typecode << "' is invalid for decoding readout cell type";
    ex.addContext("Calling HGCalMappingModuleIndexer::getCellType()");
    throw ex;
  }
  int8_t celltype = -1;
  const bool isSiPM = (typecode[0] == 'T');
  const bool isHD = (typecode[1] == 'H');
  if (isSiPM) {  // assign SiPM type coarse or molded with next version of modulelocator
    if (isHD)
      celltype = HGCScintillatorDetId::tileGranularity::HGCalTileFine;
    else
      celltype = HGCScintillatorDetId::tileGranularity::HGCalTileNormal;
  } else {  // assign Si wafer type low/high density and thickness (120, 200, 300 um)
    const char thickness = typecode[4];
    if (isHD) {
      if (thickness == '1')
        celltype = HGCSiliconDetId::waferType::HGCalHD120;
      else if (thickness == '2')
        celltype = HGCSiliconDetId::waferType::HGCalHD200;
    } else {
      if (thickness == '2')
        celltype = HGCSiliconDetId::waferType::HGCalLD200;
      else if (thickness == '3')
        celltype = HGCSiliconDetId::waferType::HGCalLD300;
    }
  }
  if (celltype == -1) {
    cms::Exception ex("InvalidHGCALTypeCode");
    ex << "Could not parse cell type from typecode='" << typecode << "'";
    ex.addContext("Calling HGCalMappingModuleIndexer::getCellType()");
    throw ex;
  }
  return std::pair<bool, int8_t>(isSiPM, celltype);
}

//
std::pair<uint32_t, uint32_t> HGCalMappingModuleIndexer::getIndexForFedAndModule(std::string const& typecode) const {
  auto it = typecodeMap_.find(typecode);
  if (it == typecodeMap_.end()) {       // did not find key
    std::size_t nmax = 100;             // maximum number of keys to print
    auto maxit = typecodeMap_.begin();  // limit printout to prevent gigantic print out
    std::advance(maxit, std::min(typecodeMap_.size(), nmax));
    std::string allkeys = std::accumulate(
        std::next(typecodeMap_.begin()), maxit, typecodeMap_.begin()->first, [](const std::string& a, const auto& b) {
          return a + ',' + b.first;
        });
    if (typecodeMap_.size() > nmax)
      allkeys += ", ...";
    throw cms::Exception("HGCalMappingModuleIndexer")
        << "Could not find typecode '" << typecode << "' in map (size=" << typecodeMap_.size()
        << ")! Found the following modules (from the module locator file): " << allkeys;
  }
  return it->second;  // (fedid,modid)
};
