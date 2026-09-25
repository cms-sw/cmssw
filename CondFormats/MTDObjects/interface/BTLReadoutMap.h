#ifndef CondFormats_MTDObjects_BTLReadoutMap_h
#define CondFormats_MTDObjects_BTLReadoutMap_h

#include <cstdint>
#include <unordered_map>

#include "DataFormats/ForwardDetId/interface/BTLDetId.h"
#include "CondFormats/MTDObjects/interface/BTLElectronicsId.h"
#include "CondFormats/Serialization/interface/Serializable.h"

//------------------------------------------------------------
// Electronics IDs corresponding to the two sides of one crystal
//------------------------------------------------------------
struct BTLElectronicsIdPair {
  BTLElectronicsId minus;
  BTLElectronicsId plus;

  COND_SERIALIZABLE;
};

// ------------------------------------------------------------
// Readout map: BTLDetId <-> BTLElectronicsId
// ------------------------------------------------------------
class BTLReadoutMap {
public:
  BTLReadoutMap();
  virtual ~BTLReadoutMap();

  // ----------------------------
  // Fill interface - inserts a new record in the readout map
  // ----------------------------
  void add(const BTLDetId& detId, const BTLElectronicsIdPair& elecIds);

  // ----------------------------
  // Forward lookup: DetId -> electronics
  // ----------------------------
  BTLElectronicsIdPair getElectronicsId(const BTLDetId& detId) const;

  // ----------------------------
  // Reverse lookup: electronics -> DetId
  // ----------------------------
  BTLDetId getDetId(const BTLElectronicsId& elecId) const;

  // ----------------------------
  // Utilities
  // ----------------------------
  void initialize();

  void clear();

  int size() const { return detToElec_.size(); };

private:
  // forward mapping
  std::unordered_map<uint32_t, BTLElectronicsIdPair> detToElec_;

  // reverse mapping (packed electronics key -> detid)
  std::unordered_map<uint32_t, uint32_t> elecToDet_ COND_TRANSIENT;

  COND_SERIALIZABLE;
};

#endif
