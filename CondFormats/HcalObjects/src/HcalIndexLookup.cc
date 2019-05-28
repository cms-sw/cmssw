#include <algorithm>

#include "CondFormats/HcalObjects/interface/HcalIndexLookup.h"

void HcalIndexLookup::clear() {
  data_.clear();
  sorted_ = true;
}

void HcalIndexLookup::sort() {
  if (!sorted_) {
    std::sort(data_.begin(), data_.end());
    sorted_ = true;
  }
}

bool HcalIndexLookup::hasDuplicateIds() {
  const std::size_t sz = data_.size();
  if (sz) {
    sort();
    const std::size_t szm1 = sz - 1;
    for (std::size_t i = 0; i < szm1; ++i)
      if (data_[i].first == data_[i + 1].first)
        return true;
  }
  return false;
}

void HcalIndexLookup::add(const unsigned detId, const unsigned index) {
  if (index == InvalidIndex)
    throw cms::Exception("In HcalIndexLookup::add: invalid index");
  data_.push_back(std::pair<uint32_t, uint32_t>(detId, index));
  sorted_ = false;
}

unsigned HcalIndexLookup::find(const unsigned detId) const {
  if (data_.empty())
    return InvalidIndex;
  if (!sorted_)
    throw cms::Exception(
        "In HcalIndexLookup::lookup:"
        " collection is not sorted");
  std::pair<uint32_t, uint32_t> search(detId, 0U);
  auto end = data_.end();
  auto it = std::lower_bound(data_.begin(), end, search);
  if (it == end)
    return InvalidIndex;
  if (it->first == detId)
    return it->second;
  else
    return InvalidIndex;
}

unsigned HcalIndexLookup::largestIndex() const {
  const std::size_t sz = data_.size();
  if (sz) {
    uint32_t largest = 0;
    for (std::size_t i = 0; i < sz; ++i)
      if (data_[i].second > largest)
        largest = data_[i].second;
    return largest;
  } else
    return InvalidIndex;
}
