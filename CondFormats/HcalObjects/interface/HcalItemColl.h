#ifndef CondFormats_HcalObjects_HcalItemColl_h
#define CondFormats_HcalObjects_HcalItemColl_h

#include <memory>

#include "boost/serialization/access.hpp"
#include "boost/serialization/version.hpp"
#include "boost/serialization/shared_ptr.hpp"
#include "boost/serialization/vector.hpp"

//
// This collection manages only pointers and references.
// In particular, it can be used for storing objects in
// an inheritance hierarchy by their base pointers.
// The pointee objects are owned by this collection.
// If copies of this collection are made, all copies
// will share the ownership of the same set of objects.
//
template <typename Item>
class HcalItemColl {
public:
  typedef Item value_type;

  // The following method adds a new item to the collection
  inline void push_back(std::unique_ptr<Item> ptr) { data_.push_back(std::shared_ptr<Item>(ptr.release())); }

  // Other modifiers
  inline void clear() { data_.clear(); }
  inline void reserve(const unsigned n) { data_.reserve(n); }

  // Some inspectors
  inline std::size_t size() const { return data_.size(); }
  inline bool empty() const { return data_.empty(); }

  // The following function returns nullptr if the index is out of range
  inline const Item* get(const unsigned index) const {
    if (index < data_.size())
      return data_[index].get();
    else
      return nullptr;
  }

  // The following function throws an exception if the index is out of range
  inline const Item& at(const unsigned index) const { return *data_.at(index); }

  // Deep comparison for equality is useful for testing serialization
  bool operator==(const HcalItemColl& r) const {
    const std::size_t sz = data_.size();
    if (sz != r.data_.size())
      return false;
    for (std::size_t i = 0; i < sz; ++i)
      if (!(*data_[i] == *r.data_[i]))
        return false;
    return true;
  }

  inline bool operator!=(const HcalItemColl& r) const { return !(*this == r); }

private:
  std::vector<std::shared_ptr<Item> > data_;

  friend class boost::serialization::access;

  template <class Archive>
  inline void serialize(Archive& ar, unsigned /* version */) {
    ar& data_;
  }
};

// boost serialization version number for this template
namespace boost {
  namespace serialization {
    template <typename Item>
    struct version<HcalItemColl<Item> > {
      BOOST_STATIC_CONSTANT(int, value = 1);
    };
  }  // namespace serialization
}  // namespace boost

#endif  // CondFormats_HcalObjects_HcalItemColl_h
