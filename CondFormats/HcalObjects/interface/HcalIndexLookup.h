#ifndef CondFormats_HcalObjects_HcalIndexLookup_h
#define CondFormats_HcalObjects_HcalIndexLookup_h

#include "FWCore/Utilities/interface/Exception.h"

#include "boost/serialization/access.hpp"
#include "boost/serialization/version.hpp"
#include "boost/serialization/vector.hpp"
#include "boost/serialization/utility.hpp"

#include <cstdint>
#include <climits>
#include <vector>

//
// Storable lookup of unsigned index values by unsigned key
// (raw detId, ieta, etc)
//
class HcalIndexLookup {
public:
  static const unsigned InvalidIndex = UINT_MAX;

  inline HcalIndexLookup() : sorted_(true) {}

  // Add an index for lookup. All "transformedId" numbers should be
  // unique and "index" argument must not be equal InvalidIndex.
  void add(unsigned transformedId, unsigned index);

  void sort();
  void clear();
  inline void reserve(const unsigned n) { data_.reserve(n); }

  // After filling up the table, it is recommended to verify that
  // there are no duplicate ids. The collection will also become
  // sorted as a side effect of the following function call.
  bool hasDuplicateIds();

  // Some trivial inspectors
  inline std::size_t size() const { return data_.size(); }
  inline bool empty() const { return data_.empty(); }

  // Largest index number. Returns InvalidIndex for empty collection.
  unsigned largestIndex() const;

  // "find" returns InvalidIndex in case argument detId
  // is not in the collection. Note that the object should be
  // sorted (or even better, checked for duplicate ids) before
  // performing index lookups.
  unsigned find(unsigned detId) const;

  // Comparison is only really useful for testing sorted lookups
  // (serialized lookups will be sorted)
  inline bool operator==(const HcalIndexLookup& r) const { return data_ == r.data_ && sorted_ == r.sorted_; }

  inline bool operator!=(const HcalIndexLookup& r) const { return !(*this == r); }

private:
  std::vector<std::pair<uint32_t, uint32_t> > data_;
  bool sorted_;

  friend class boost::serialization::access;

  template <class Archive>
  inline void save(Archive& ar, const unsigned /* version */) const {
    // Make sure that there are no duplicate ids
    if ((const_cast<HcalIndexLookup*>(this))->hasDuplicateIds())
      throw cms::Exception("In HcalIndexLookup::save: invalid data");
    ar& data_& sorted_;
  }

  template <class Archive>
  inline void load(Archive& ar, const unsigned /* version */) {
    ar& data_& sorted_;
    if (hasDuplicateIds())
      throw cms::Exception("In HcalIndexLookup::load: invalid data");
  }

  BOOST_SERIALIZATION_SPLIT_MEMBER()
};

BOOST_CLASS_VERSION(HcalIndexLookup, 1)

#endif  // CondFormats_HcalObjects_HcalIndexLookup_h
