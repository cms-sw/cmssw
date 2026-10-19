#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "CondFormats/HGCalObjects/interface/HGCalMappingModuleIndexerTrigger.h"
#include "DataFormats/ForwardDetId/interface/HGCSiliconDetId.h"       // for HGCSiliconDetId::waferType
#include "DataFormats/ForwardDetId/interface/HGCScintillatorDetId.h"  // for HGCScintillatorDetId::tileGranularity

/**
 * @short for a new module it adds its type to the readout sequence vector
 * if the fed id is not yet existing in the mapping it is added
 * a dense indexer is used to create the necessary indices for the new module
 * unused indices will be set with -1
 */
void HGCalMappingModuleIndexerTrigger::processNewModule(uint32_t fedid,
                                                        uint16_t econtIdx,
                                                        uint32_t typecodeIdx,
                                                        uint32_t nTrLinks,
                                                        uint32_t nTCs,
                                                        std::string const& typecode) {
  // add fed if needed
  if (fedid >= fedReadoutSequences_.size()) {
    fedReadoutSequences_.resize(fedid + 1);
  }
  HGCalTriggerFEDReadoutSequence& frs = fedReadoutSequences_[fedid];
  frs.id = fedid;

  // asin position, resize if needed, and fill the type code
  uint32_t idx = modFedIndexer_.denseIndex({{econtIdx, 0}});  // This will give back the econt*1+0 = econt
  if (idx >= frs.readoutTypes_.size()) {
    frs.readoutTypes_.resize(idx + 1, -1);
  }
  frs.readoutTypes_[idx] = typecodeIdx;

  // std::count another typecodein the global list
  if (typecodeIdx >= globalTypesCounter_.size()) {
    globalTypesCounter_.resize(typecodeIdx + 1, 0);
    globalTypesNTrLinks_.resize(typecodeIdx + 1, 0);
    globalTypesNTCs_.resize(typecodeIdx + 1, 0);
    offsetsTC_.resize(typecodeIdx + 1, 0);
  }
  globalTypesCounter_[typecodeIdx]++;
  globalTypesNTrLinks_[typecodeIdx] = nTrLinks;
  globalTypesNTCs_[typecodeIdx] = nTCs;

  // add typecode string to map to retrieve fedId & modId later
  if (!typecode.empty()) {
    if (typecodeMap_.find(typecode) != typecodeMap_.end()) {     // found key
      const auto& [fedid_, modid_] = typecodeMap_.at(typecode);  // (fedId,modId)
      edm::LogWarning("HGCalMappingModuleIndexerTrigger")
          << "Found typecode " << typecode << " already in map (fedid,modid)=(" << fedid_ << "," << modid_
          << ")! Overwriting with (" << fedid << "," << idx << ")...";
    }
    LogDebug("HGCalMappingModuleIndexerTrigger")
        << "HGCalMappingModuleIndexerTrigger::processNewModule: Adding typecode=\"" << typecode
        << "\" with fedid=" << fedid << ", idx=" << idx << " (will be re-indexed after finalize)";
    typecodeMap_[typecode] = std::make_pair(fedid, idx);
  }
}

// @short to be called after all the modules have been processed
void HGCalMappingModuleIndexerTrigger::finalize() {
  // max indices at different levels
  nfeds_ = fedReadoutSequences_.size();
  maxModulesIdx_ = std::accumulate(globalTypesCounter_.begin(), globalTypesCounter_.end(), 0);
  maxTrLinksIdx_ =
      std::inner_product(globalTypesCounter_.begin(), globalTypesCounter_.end(), globalTypesNTrLinks_.begin(), 0);
  maxNTCIdx_ = std::inner_product(globalTypesCounter_.begin(), globalTypesCounter_.end(), globalTypesNTCs_.begin(), 0);

  // compute the global offset to assign per board type, eRx and channel data
  offsetsModule_.resize(maxModulesIdx_, 0);
  offsetsTrLink_.resize(maxModulesIdx_, 0);
  offsetsTC_.resize(maxModulesIdx_, 0);
  for (size_t i = 1; i < globalTypesCounter_.size(); i++) {
    offsetsModule_[i] = globalTypesCounter_[i - 1] + offsetsModule_[i - 1];
    offsetsTrLink_[i] = globalTypesCounter_[i - 1] * globalTypesNTrLinks_[i - 1] + offsetsTrLink_[i - 1];
    offsetsTC_[i] = globalTypesCounter_[i - 1] * globalTypesNTCs_[i - 1] + offsetsTC_[i - 1];
  }

  // now go through the FEDs and ascribe the offsets per module in the readout sequence
  std::vector<uint32_t> typeCounters(globalTypesCounter_.size(), 0);
  for (auto& fedit : fedReadoutSequences_) {
    // assign the final indexing in the look-up table depending on which ECON-D's are really present
    size_t nconn(0);
    fedit.moduleLUT_.resize(fedit.readoutTypes_.size(), -1);
    for (size_t i = 0; i < fedit.readoutTypes_.size(); i++) {
      if (fedit.readoutTypes_[i] == -1)
        continue;  //unexisting

      reassignTypecodeLocation(fedit.id, i, nconn);
      fedit.moduleLUT_[i] = nconn;
      nconn++;
    }

    // remove unexisting ECONs building a final compact readout sequence
    fedit.readoutTypes_.erase(
        std::remove_if(
            fedit.readoutTypes_.begin(), fedit.readoutTypes_.end(), [&](int val) -> bool { return val == -1; }),
        fedit.readoutTypes_.end());

    // resize vectors to their final size and set final values
    size_t nmods = fedit.readoutTypes_.size();
    fedit.modOffsets_.resize(nmods, 0);
    fedit.TrLinkOffsets_.resize(nmods, 0);
    fedit.TCOffsets_.resize(nmods, 0);
    fedit.enabledLink_.resize(nmods, 0);

    for (size_t i = 0; i < nmods; i++) {
      int type_val = fedit.readoutTypes_[i];

      // module offset : global offset for this type + current index for this type
      uint32_t baseMod_offset = offsetsModule_[type_val] + typeCounters[type_val];
      fedit.modOffsets_[i] = baseMod_offset;  // + internalMod_offset;

      // erx-level offset : global offset of e-Rx of this type + #e-Rrx * current index for this type
      uint32_t baseTrLink_offset = offsetsTrLink_[type_val];
      uint32_t internalTrLink_offset = globalTypesNTrLinks_[type_val] * typeCounters[type_val];
      fedit.TrLinkOffsets_[i] = baseTrLink_offset + internalTrLink_offset;

      // channel data offset: global offset for data of this type + #words * current index for this type
      uint32_t baseData_offset = offsetsTC_[type_val];
      uint32_t internalData_offset = globalTypesNTCs_[type_val] * typeCounters[type_val];
      fedit.TCOffsets_[i] = baseData_offset + internalData_offset;

      // enabled erx flags : we assume all eRx are enabled
      // this results in a small mem overhead for a few partial wafers but it's not a show stopper
      fedit.enabledLink_[i] = (0b1 << globalTypesNTrLinks_[type_val]) - 0b1;
      typeCounters[type_val]++;
    }
  }
}

/**
 * @short decode silicon or sipm type and cell type for the detector id 
 * from the typecode string: "M[LH]-X[123]X-*" for Si, "T[LH]-L*S*[PN]" for SiPm
 * details can be found in the EDMS documents under https://edms.cern.ch/document/2718867/1
 */
std::pair<bool, int8_t> HGCalMappingModuleIndexerTrigger::getCellType(std::string_view typecode) {
  if (typecode.size() < 5) {
    cms::Exception ex("InvalidHGCALTypeCode");
    ex << "'" << typecode << "' is invalid for decoding readout cell type";
    ex.addContext("Calling HGCalMappingModuleIndexerTrigger::getCellType()");
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
    ex.addContext("Calling HGCalMappingModuleIndexerTrigger::getCellType()");
    throw ex;
  }
  return std::pair<bool, int8_t>(isSiPM, celltype);
}

//
std::pair<uint32_t, uint32_t> HGCalMappingModuleIndexerTrigger::getIndexForFedAndModule(
    std::string const& typecode) const {
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
    throw cms::Exception("HGCalMappingModuleIndexerTrigger")
        << "Could not find typecode '" << typecode << "' in map (size=" << typecodeMap_.size()
        << ")! Found the following modules (from the module locator file): " << allkeys;
  }
  return it->second;  // (fedid,modid)
};
