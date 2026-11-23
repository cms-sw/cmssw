#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "CondFormats/MTDObjects/interface/BTLReadoutMap.h"

#include <algorithm>
#include <stdexcept>
#include <ostream>
#include <utility>
#include <vector>

BTLReadoutMap::BTLReadoutMap() {}

BTLReadoutMap::~BTLReadoutMap() {}

// ------------------------------------------------------------------
// Initialize - to build inverse map which is not stored persistently
// ------------------------------------------------------------------
void BTLReadoutMap::initialize() {
  elecToDet_.clear();

  for (const auto& entry : detToElec_) {
    const uint32_t detId = entry.first;
    const BTLElectronicsIdPair& elecIds = entry.second;

    auto ret = elecToDet_.emplace(elecIds.minus.rawId(), detId);
    if (!ret.second) {
      throw cms::Exception("BTLReadoutMap::initialize()") << "Duplicate BTLElectronicsId (minus side)" << std::endl;
    }

    ret = elecToDet_.emplace(elecIds.plus.rawId(), detId);
    if (!ret.second) {
      throw cms::Exception("BTLReadoutMap::initialize()") << "Duplicate BTLElectronicsId (plus side)" << std::endl;
    }
  }
}

// ------------------------------------------------------------
// Add full 2-channel mapping
// ------------------------------------------------------------
void BTLReadoutMap::add(const BTLDetId& detId, const BTLElectronicsIdPair& elecIds) {
  const uint32_t detKey = detId.rawId();

  // ----------------------------
  // Forward: det -> 2 channels
  // ----------------------------
  if (detToElec_.find(detKey) != detToElec_.end()) {
    throw cms::Exception("BTLReadoutMap::add()") << "Duplicate BTLDetId entry" << std::endl;
  }

  detToElec_[detKey] = elecIds;

  // ----------------------------
  // Reverse: each channel -> det
  // ----------------------------

  // -- minus side
  const uint32_t minusKey = elecIds.minus.rawId();
  if (elecToDet_.find(minusKey) != elecToDet_.end()) {
    throw cms::Exception("BTLReadoutMap::add()") << "Duplicate BTLElectronicsId entry (minus side): " << elecIds.minus;
  }
  elecToDet_[minusKey] = detKey;

  // -- plus side
  const uint32_t plusKey = elecIds.plus.rawId();
  if (elecToDet_.find(plusKey) != elecToDet_.end()) {
    throw cms::Exception("BTLReadoutMap::add()") << "Duplicate BTLElectronicsId entry (plus side): " << elecIds.plus;
  }
  elecToDet_[plusKey] = detKey;
}

// ------------------------------------------------------------
// Forward lookup
// ------------------------------------------------------------
BTLElectronicsIdPair BTLReadoutMap::getElectronicsId(const BTLDetId& detId) const {
  const auto it = detToElec_.find(detId.rawId());

  if (it == detToElec_.end()) {
    throw cms::Exception("BTLReadoutMap::getElectronicsId") << "BTLDetId " << detId.rawId() << " not found! ";
  }

  return it->second;
}

// ------------------------------------------------------------
// Reverse lookup
// ------------------------------------------------------------
BTLDetId BTLReadoutMap::getDetId(const BTLElectronicsId& elecId) const {
  const auto it = elecToDet_.find(elecId.rawId());

  if (it == elecToDet_.end()) {
    throw cms::Exception("BTLReadoutMap::getElectronicsId") << "BTLElectronicsId " << elecId.rawId() << " not found! ";
  }

  return BTLDetId(it->second);
}

// ------------------------------------------------------------
// clear
// ------------------------------------------------------------
void BTLReadoutMap::clear() {
  detToElec_.clear();
  elecToDet_.clear();
}

// ------------------------------------------------------------
// Return list of detIds
// ------------------------------------------------------------
std::vector<BTLDetId> BTLReadoutMap::getListOfDetIds() const {
  std::vector<BTLDetId> detIds;
  detIds.reserve(detToElec_.size());

  for (const auto& [rawDetId, elecIds] : detToElec_) {
    detIds.emplace_back(rawDetId);
  }

  return detIds;
}

// ------------------------------------------------------------
// Check dense channel indexes and their ordering w.r.t. rawId
// ------------------------------------------------------------
bool BTLReadoutMap::checkDenseChannelIndex() const {
  // (dense index, rawId) of each electronics channel
  std::vector<std::pair<uint32_t, uint32_t>> pairs;
  pairs.reserve(2 * detToElec_.size());
  for (const auto& [detId, elecIds] : detToElec_) {
    pairs.emplace_back(elecIds.minus.denseChannelIndex(), elecIds.minus.rawId());
    pairs.emplace_back(elecIds.plus.denseChannelIndex(), elecIds.plus.rawId());
  }

  std::sort(pairs.begin(), pairs.end());

  for (std::size_t i = 0; i < pairs.size(); ++i) {
    if (pairs[i].first != i) {
      edm::LogError("BTLReadoutMap") << "Dense channel index " << i << " missing or duplicated (found "
                                     << pairs[i].first << ")";
      return false;
    }
    if (i > 0 && pairs[i - 1].second >= pairs[i].second) {
      edm::LogError("BTLReadoutMap") << "rawId not increasing with dense channel index at index " << i << ": "
                                     << BTLElectronicsId(pairs[i - 1].second) << " vs "
                                     << BTLElectronicsId(pairs[i].second);
      return false;
    }
  }
  return true;
}
