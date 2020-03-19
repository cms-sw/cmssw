#ifndef CondFormats_HcalObjects_HcalItemArrayCollById_h
#define CondFormats_HcalObjects_HcalItemArrayCollById_h

#include <cstdint>

#include "FWCore/Utilities/interface/Exception.h"

#include "CondFormats/HcalObjects/interface/HcalItemArrayColl.h"
#include "CondFormats/HcalObjects/interface/HcalIndexLookup.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "CondFormats/HcalObjects/interface/HcalDetIdTransform.h"
#include "CondFormats/HcalObjects/interface/AbsHcalAlgoData.h"

//
// This collection allows lookup of arrays of items by HcalDetId.
// If the given HcalDetId is not explicitly listed in the
// lookup table, default item is returned.
//
// Just like HcalItemArrayColl, this collection works with pointers
// and references only, so it can be used with the inheritance
// scenarios. Note that the ownership of objects is shared with
// the collection provided in the constructor. The default array
// is owned by this collection. Its ownership will also become
// shared if a copy of this collection is made.
//
template <typename Item, unsigned N>
class HcalItemArrayCollById : public AbsHcalAlgoData {
public:
  typedef Item value_type;
  typedef typename HcalItemArrayColl<Item, N>::InputArray InputArray;
  static constexpr unsigned arraySize() { return N; }

  // Dummy constructor. To be used for deserialization only.
  inline HcalItemArrayCollById() : transformCode_(HcalDetIdTransform::N_TRANSFORMS) {}

  // Normal constructor
  HcalItemArrayCollById(const HcalItemArrayColl<Item, N>& coll,
                        const HcalIndexLookup& indexLookupTable,
                        const unsigned detIdTransformCode,
                        InputArray& defaultFunctors)
      : coll_(coll), lookup_(indexLookupTable), transformCode_(detIdTransformCode) {
    // Check that the lookup table is valid for this application
    if (lookup_.hasDuplicateIds())
      throw cms::Exception(
          "In HcalItemArrayCollById constructor:"
          " invalid lookup table");

    // Check that the lookup table is consistent with the size
    // of the collection
    const unsigned maxIndex = lookup_.largestIndex();
    if (maxIndex != HcalIndexLookup::InvalidIndex && maxIndex >= coll_.size())
      throw cms::Exception(
          "In HcalItemArrayCollById constructor:"
          " collection and lookup table are inconsistent");

    HcalDetIdTransform::validateCode(transformCode_);

    // Take care of the default array
    setDefault(defaultFunctors);
  }

  inline virtual ~HcalItemArrayCollById() {}

  // Modifier for the default array of items
  inline void setDefault(InputArray& arr) {
    for (unsigned i = 0; i < N; ++i)
      default_[i] = std::shared_ptr<Item>(arr[i].release());
  }

  // Size of the internal collection, not counting the default
  inline std::size_t size() const { return coll_.size(); }

  // Look up the index into the collection by detector id
  inline unsigned getIndex(const HcalDetId& id) const {
    return lookup_.find(HcalDetIdTransform::transform(id, transformCode_));
  }

  // Item lookup by its index and array index. If item lookup
  // by index fails and the array index is not out of bounds,
  // default item is returned.
  inline const Item* getByIndex(const unsigned itemIndex, const unsigned arrayIndex) const {
    const Item* f = coll_.get(itemIndex, arrayIndex);
    if (f == nullptr && arrayIndex < N)
      f = default_[arrayIndex].get();
    return f;
  }

  // The following method will return nullptr if
  // there is no corresponding default
  inline const Item* getDefault(const unsigned arrayIndex) const {
    if (arrayIndex < N)
      return default_[arrayIndex].get();
    else
      return nullptr;
  }

  // Convenience function for getting what we need by id.
  // Note that, if you are simply cycling over array indices,
  // it will be more efficient to retrieve the item index
  // first and then use "getByIndex" method.
  inline const Item* get(const HcalDetId& id, const unsigned arrayIndex) const {
    return getByIndex(getIndex(id), arrayIndex);
  }

  // Similar comment applies here if you are just cycling over array indices
  inline const Item& at(const HcalDetId& id, const unsigned arrayIndex) const {
    const Item* f = getByIndex(getIndex(id), arrayIndex);
    if (f == nullptr)
      throw cms::Exception("In HcalItemArrayCollById::at: invalid detector id");
    return *f;
  }

protected:
  virtual bool isEqual(const AbsHcalAlgoData& other) const override {
    const HcalItemArrayCollById& r = static_cast<const HcalItemArrayCollById&>(other);
    if (coll_ != r.coll_)
      return false;
    if (lookup_ != r.lookup_)
      return false;
    if (transformCode_ != r.transformCode_)
      return false;
    for (unsigned j = 0; j < N; ++j) {
      // The default may or may not be there
      const bool ld = default_[j].get();
      const bool rd = r.default_[j].get();
      if (ld != rd)
        return false;
      if (ld)
        if (!(*default_[j] == *r.default_[j]))
          return false;
    }
    return true;
  }

private:
  typedef boost::array<std::shared_ptr<Item>, N> StoredArray;

  HcalItemArrayColl<Item, N> coll_;
  HcalIndexLookup lookup_;
  StoredArray default_;
  uint32_t transformCode_;

  friend class boost::serialization::access;

  template <class Archive>
  inline void serialize(Archive& ar, unsigned /* version */) {
    ar& coll_& lookup_& default_& transformCode_;
  }
};

// boost serialization version number for this template
namespace boost {
  namespace serialization {
    template <typename Item, unsigned N>
    struct version<HcalItemArrayCollById<Item, N> > {
      BOOST_STATIC_CONSTANT(int, value = 1);
    };
  }  // namespace serialization
}  // namespace boost

#endif  // CondFormats_HcalObjects_HcalItemArrayCollById_h
