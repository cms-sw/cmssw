#ifndef CondFormats_HcalObjects_HBHEChannelGroups_h_
#define CondFormats_HcalObjects_HBHEChannelGroups_h_

#include "FWCore/Utilities/interface/Exception.h"

#include "boost/serialization/access.hpp"
#include "boost/serialization/vector.hpp"

#include "CondFormats/HcalObjects/interface/HBHELinearMap.h"
#include <cstdint>

class HBHEChannelGroups {
public:
  inline HBHEChannelGroups() : group_(HBHELinearMap::ChannelCount, 0U) {}

  //
  // Main constructor. It is expected that "len" equals
  // HBHELinearMap::ChannelCount and that every element of "data"
  // indicates to which group that particular channel should belong.
  //
  inline HBHEChannelGroups(const unsigned* data, const unsigned len) : group_(data, data + len) {
    if (!validate())
      throw cms::Exception("In HBHEChannelGroups constructor: invalid input data");
  }

  //
  // Set the group number for the given HBHE linear channel number.
  // Linear channel numbers are calculated by HBHELinearMap.
  //
  inline void setGroup(const unsigned linearChannel, const unsigned groupNum) { group_.at(linearChannel) = groupNum; }

  // Inspectors
  inline unsigned size() const { return group_.size(); }

  inline const uint32_t* groupData() const { return group_.empty() ? nullptr : &group_[0]; }

  inline unsigned getGroup(const unsigned linearChannel) const { return group_.at(linearChannel); }

  inline unsigned largestGroupNumber() const {
    unsigned lg = 0;
    const unsigned sz = group_.size();
    const uint32_t* dat = sz ? &group_[0] : nullptr;
    for (unsigned i = 0; i < sz; ++i)
      if (dat[i] > lg)
        lg = dat[i];
    return lg;
  }

  // Comparators
  inline bool operator==(const HBHEChannelGroups& r) const { return group_ == r.group_; }

  inline bool operator!=(const HBHEChannelGroups& r) const { return !(*this == r); }

private:
  std::vector<uint32_t> group_;

  inline bool validate() const { return group_.size() == HBHELinearMap::ChannelCount; }

  friend class boost::serialization::access;

  template <class Archive>
  inline void save(Archive& ar, const unsigned /* version */) const {
    if (!validate())
      throw cms::Exception("In HBHEChannelGroups::save: invalid data");
    ar& group_;
  }

  template <class Archive>
  inline void load(Archive& ar, const unsigned /* version */) {
    ar& group_;
    if (!validate())
      throw cms::Exception("In HBHEChannelGroups::load: invalid data");
  }

  BOOST_SERIALIZATION_SPLIT_MEMBER()
};

BOOST_CLASS_VERSION(HBHEChannelGroups, 1)

#endif  // CondFormats_HcalObjects_HBHEChannelGroups_h_
