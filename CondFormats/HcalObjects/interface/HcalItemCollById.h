#ifndef CondFormats_HcalObjects_HcalItemCollById_h
#define CondFormats_HcalObjects_HcalItemCollById_h

#include <cstdint>

#include "FWCore/Utilities/interface/Exception.h"

#include "CondFormats/HcalObjects/interface/HcalItemColl.h"
#include "CondFormats/HcalObjects/interface/HcalIndexLookup.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "CondFormats/HcalObjects/interface/HcalDetIdTransform.h"
#include "CondFormats/HcalObjects/interface/AbsHcalAlgoData.h"

//
// This collection allows lookup of items by HcalDetId.
// If the given HcalDetId is not explicitly listed in the
// lookup table, default item is returned.
//
// Just like HcalItemColl, this collection works with pointers
// and references only, so it can be used with the inheritance
// scenarios. Note that the ownership of objects is shared with
// the collection provided in the constructor. The default item
// is owned by this collection. Its ownership will also become
// shared if a copy of this collection is made.
//
template <typename Item>
class HcalItemCollById : public AbsHcalAlgoData {
public:
  typedef Item value_type;

  // Dummy default constructor. To be used only for deserialization.
  inline HcalItemCollById() : transformCode_(HcalDetIdTransform::N_TRANSFORMS) {}

  // Normal constructor
  HcalItemCollById(const HcalItemColl<Item>& coll,
                   const HcalIndexLookup& indexLookupTable,
                   const unsigned detIdTransformCode,
                   std::unique_ptr<Item> defaultItem)
      : coll_(coll), lookup_(indexLookupTable), default_(defaultItem.release()), transformCode_(detIdTransformCode) {
    // Check that the lookup table is valid for this application
    if (lookup_.hasDuplicateIds())
      throw cms::Exception(
          "In HcalItemCollById constructor:"
          " invalid lookup table");

    // Check that the lookup table is consistent with the size
    // of the collection
    const unsigned maxIndex = lookup_.largestIndex();
    if (maxIndex != HcalIndexLookup::InvalidIndex && maxIndex >= coll_.size())
      throw cms::Exception(
          "In HcalItemCollById constructor:"
          " collection and lookup table are inconsistent");

    HcalDetIdTransform::validateCode(transformCode_);
  }

  inline ~HcalItemCollById() override {}

  // Modifier for the default item
  inline void setDefault(std::unique_ptr<Item> f) { default_ = std::shared_ptr<Item>(f.release()); }

  // Size of the internal collection, not counting the default
  inline std::size_t size() const { return coll_.size(); }

  // The following method will return nullptr if there is no default
  inline const Item* getDefault() const { return default_.get(); }

  // Look up the index into the collection by detector id
  inline unsigned getIndex(const HcalDetId& id) const {
    return lookup_.find(HcalDetIdTransform::transform(id, transformCode_));
  }

  // Get an item by its index in the collection. If the index
  // is out of range, the default item is returned. If the
  // index is out of range and there is no default, nullptr
  // is returned.
  inline const Item* getByIndex(const unsigned index) const {
    if (index < coll_.size())
      return coll_.get(index);
    else
      return default_.get();
  }

  // Convenience function for getting what we need by id.
  // This method can return nullptr.
  inline const Item* get(const HcalDetId& id) const { return getByIndex(getIndex(id)); }

  // The following method will throw an exception if the id is not
  // in the lookup table and, in addition, there is no default
  inline const Item& at(const HcalDetId& id) const {
    const Item* ptr = getByIndex(getIndex(id));
    if (ptr == nullptr)
      throw cms::Exception("In HcalItemCollById::at: invalid detector id");
    return *ptr;
  }

protected:
  bool isEqual(const AbsHcalAlgoData& other) const override {
    const HcalItemCollById& r = static_cast<const HcalItemCollById&>(other);
    if (coll_ != r.coll_)
      return false;
    if (lookup_ != r.lookup_)
      return false;
    if (transformCode_ != r.transformCode_)
      return false;
    // The default may or may not be there
    const bool ld = default_.get();
    const bool rd = r.default_.get();
    if (ld != rd)
      return false;
    if (ld)
      if (!(*default_ == *r.default_))
        return false;
    return true;
  }

private:
  HcalItemColl<Item> coll_;
  HcalIndexLookup lookup_;
  std::shared_ptr<Item> default_;
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
    template <typename Item>
    struct version<HcalItemCollById<Item> > {
      BOOST_STATIC_CONSTANT(int, value = 1);
    };
  }  // namespace serialization
}  // namespace boost

#endif  // CondFormats_HcalObjects_HcalItemCollById_h
