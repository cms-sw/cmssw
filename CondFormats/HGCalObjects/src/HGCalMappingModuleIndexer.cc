#include "CondFormats/HGCalObjects/interface/HGCalMappingModuleIndexer.h"

//
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
  if (typecode != "") {
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

//
void HGCalMappingModuleIndexer::finalize() {
  //max indices at different levels
  nfeds_ = fedReadoutSequences_.size();
  maxModulesIdx_ = std::accumulate(globalTypesCounter_.begin(), globalTypesCounter_.end(), 0);
  maxErxIdx_ = std::inner_product(globalTypesCounter_.begin(), globalTypesCounter_.end(), globalTypesNErx_.begin(), 0);
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
  for (auto& fedit : fedReadoutSequences_) {
    //assign the final indexing in the look-up table depending on which ECON-D's are really present
    size_t nconn(0);
    fedit.moduleLUT_.resize(fedit.readoutTypes_.size(), -1);
    for (size_t i = 0; i < fedit.readoutTypes_.size(); i++) {
      if (fedit.readoutTypes_[i] == -1)
        continue;  //unexisting

      reassignTypecodeLocation(fedit.id, i, nconn);
      fedit.moduleLUT_[i] = nconn;
      nconn++;
    }

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
