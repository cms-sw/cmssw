#ifndef CalibFormats_SiStripObjects_SiStripHashedDetId_H
#define CalibFormats_SiStripObjects_SiStripHashedDetId_H

#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/SiStripCommon/interface/SiStripConstants.h"
#include <algorithm>
#include <iomanip>
#include <vector>
#include <cstdint>

class SiStripHashedDetId;
std::ostream &operator<<(std::ostream &os, const SiStripHashedDetId &);

/**
    @class SiStripHashedDetId
    @author R.Bainbridge
    @brief Provides dense hash map in place of DetId
*/
class SiStripHashedDetId {
public:
  // ---------- constructors ----------

  /** Constructor taking raw DetIds as input. */
  SiStripHashedDetId(const std::vector<uint32_t> &);

  /** Constructor taking DetIds as input. */
  SiStripHashedDetId(const std::vector<DetId> &);

  /** Copy constructor. */
  SiStripHashedDetId(const SiStripHashedDetId &);

  /** Public default constructor. */
  SiStripHashedDetId();

  /** Default destructor. */
  ~SiStripHashedDetId();

  // ---------- typedefs ----------

  typedef std::vector<uint32_t>::const_iterator const_iterator;

  typedef std::vector<uint32_t>::iterator iterator;

  // ---------- public interface ----------

  /** Returns hashed index for given DetId. */
  inline uint32_t hashedIndex(uint32_t det_id);

  /** Returns raw (32-bit) DetId for given hashed index. */
  inline uint32_t unhashIndex(uint32_t hashed_index) const;

  /** Returns DetId object for given hashed index. */
  // inline DetId detId( uint32_t index ) const;

  inline const_iterator begin() const;

  inline const_iterator end() const;

private:
  void init(const std::vector<uint32_t> &);

  /** Sorted list of all silicon strip tracker DetIds. */
  std::vector<uint32_t> detIds_;

  uint32_t id_;

  const_iterator iter_;
};

uint32_t SiStripHashedDetId::hashedIndex(uint32_t det_id) {
  const_iterator iter = end();
  if (det_id > id_) {
    iter = find(iter_, end(), det_id);
  } else {
    iter = find(begin(), iter_, det_id);
  }
  if (iter != end()) {
    id_ = det_id;
    iter_ = iter;
    return iter - begin();
  } else {
    id_ = 0;
    iter_ = begin();
    return sistrip::invalid32_;
  }
}
uint32_t SiStripHashedDetId::unhashIndex(uint32_t hashed_index) const {
  if (hashed_index < static_cast<uint32_t>(end() - begin())) {
    return detIds_[hashed_index];
  } else {
    return sistrip::invalid32_;
  }
}
SiStripHashedDetId::const_iterator SiStripHashedDetId::begin() const { return detIds_.begin(); }
SiStripHashedDetId::const_iterator SiStripHashedDetId::end() const { return detIds_.end(); }

#endif  // CalibFormats_SiStripObjects_SiStripHashedDetId_H
