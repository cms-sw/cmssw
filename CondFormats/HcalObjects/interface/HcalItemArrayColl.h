#ifndef CondFormats_HcalObjects_HcalItemArrayColl_h
#define CondFormats_HcalObjects_HcalItemArrayColl_h

#include <memory>
#include <array>

#include "boost/array.hpp"
#include "boost/serialization/access.hpp"
#include "boost/serialization/version.hpp"
#include "boost/serialization/shared_ptr.hpp"
#include "boost/serialization/vector.hpp"
#include "boost/version.hpp"
#if BOOST_VERSION < 106400
#include "boost/serialization/array.hpp"
#else
#include "boost/serialization/boost_array.hpp"
#endif

//
// This collection manages arrays of pointers and references.
// In particular, it can be used for storing objects in
// an inheritance hierarchy by their base pointers.
// The pointee objects are owned by this collection.
//
template <typename Item, unsigned N>
class HcalItemArrayColl {
public:
  typedef Item value_type;
  typedef std::array<std::unique_ptr<Item>, N> InputArray;
  static constexpr unsigned arraySize() { return N; }

  // The following method adds a new array of pointers to the collection.
  // This class will take ownership of the pointee objects.
  void push_back(InputArray& arr) {
    StoredArray st;
    for (unsigned i = 0; i < N; ++i)
      st[i] = std::shared_ptr<Item>(arr[i].release());
    data_.push_back(st);
  }

  // Other modifiers
  inline void clear() { data_.clear(); }
  inline void reserve(const unsigned n) { data_.reserve(n); }

  // Some inspectors
  inline std::size_t size() const { return data_.size(); }
  inline bool empty() const { return data_.empty(); }

  // The following function returns nullptr if
  // one of the argument indices is out of range
  inline const Item* get(const unsigned itemIndex, const unsigned arrayIndex) const {
    if (itemIndex < data_.size() && arrayIndex < N)
      return data_[itemIndex][arrayIndex].get();
    else
      return nullptr;
  }

  // The following function throws an exception if
  // one of the argument indices is out of range
  inline Item& at(const unsigned itemIndex, const unsigned arrayIndex) const {
    return *data_.at(itemIndex).at(arrayIndex);
  }

  // Deep comparison for equality is useful for testing serialization
  bool operator==(const HcalItemArrayColl& r) const {
    const std::size_t sz = data_.size();
    if (sz != r.data_.size())
      return false;
    for (std::size_t i = 0; i < sz; ++i)
      for (unsigned j = 0; j < N; ++j)
        if (!(*data_[i][j] == *r.data_[i][j]))
          return false;
    return true;
  }

  inline bool operator!=(const HcalItemArrayColl& r) const { return !(*this == r); }

private:
  typedef boost::array<std::shared_ptr<Item>, N> StoredArray;
  std::vector<StoredArray> data_;

  friend class boost::serialization::access;

  template <class Archive>
  inline void serialize(Archive& ar, unsigned /* version */) {
    ar& data_;
  }
};

// boost serialization version number for this template
namespace boost {
  namespace serialization {
    template <typename Item, unsigned N>
    struct version<HcalItemArrayColl<Item, N> > {
      BOOST_STATIC_CONSTANT(int, value = 1);
    };
  }  // namespace serialization
}  // namespace boost

#endif  // CondFormats_HcalObjects_HcalItemArrayColl_h
