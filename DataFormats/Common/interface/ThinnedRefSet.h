#ifndef DataFormats_Common_ThinnedRefSet_h
#define DataFormats_Common_ThinnedRefSet_h
//
// Package:     DataFormats/Common
// Class  :     ThinnedRefSet
//
/**\class ThinnedRefSet ThinnedRefSet.h "ThinnedRefSet.h"

 Description: A minimal set interface (insertion, contains(), clear()) for a set of Ref keys to a thinned collection

 Usage:

    A ThinnedRefSet contains Ref keys to a thinned collection. The
keys are inserted as Refs to the thinned collection itself, or to any
parent collections of the thinned collection.

The main use case are Selector classes for edm::ThinningProducer.
There, an object of the class would be stored as a member of the
Selector class, filled in Selector::preChoose(), calling contains() in
Selector::choose(), and cleared in Selector::reset().

Example of filling
\code
class ExampleSelector {
  ...
  edm::ThinnedRefSet<ThingCollection> keysToSave_;
};

void ExampleSelector::preChoose(edm::Handle<ThingCollection> tc, edm::Event const& e, edm::EventSetup const&) {
  auto filler = keysToSave_.fill(edm::RefProd(tc), event.productGetter());
  for (auto const& object : event.get(objectCollectionToken_) {
    filler.insert(object.refToThing());
  }
}
\endcode

Example of querying if a key is present
\code
bool ExampleSelector::choose(unsigned int iIndex, Thing const& iItem) const {
  return keysToSave_.contains(iIndex);
}
\endcode
 */

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefItemGet.h"

#include <unordered_set>

namespace edm {
  class EDProductGetter;

  enum class ThinnedRefSetMode { throwOnInvalidParentRef, ignoreInvalidParentRef };

  template <typename C>
  class ThinnedRefSet {
  public:
    class Filler {
    public:
      explicit Filler(ThinnedRefSet<C>* set, RefProd<C> thinned, edm::EDProductGetter const& prodGetter)
          : set_(set), thinnedRefProd_(thinned), prodGetter_(prodGetter) {}

      template <typename T, typename F>
      void insert(Ref<C, T, F> const& ref) {
        if (ref.isNonnull()) {
          Ref<C, T, F> thinnedRef;
          if (set_->invalidParentRefMode_ == ThinnedRefSetMode::ignoreInvalidParentRef) {
            thinnedRef = tryThinnedRefFrom(ref, thinnedRefProd_, prodGetter_);
          } else {
            thinnedRef = thinnedRefFrom(ref, thinnedRefProd_, prodGetter_);
          }
          if (thinnedRef.isNonnull()) {
            set_->keys_.insert(thinnedRef.key());
          }
        }
      }

    private:
      ThinnedRefSet<C>* set_;
      RefProd<C> thinnedRefProd_;
      edm::EDProductGetter const& prodGetter_;
    };

    explicit ThinnedRefSet(ThinnedRefSetMode mode = ThinnedRefSetMode::throwOnInvalidParentRef)
        : invalidParentRefMode_(mode) {}

    Filler fill(RefProd<C> thinned, edm::EDProductGetter const& prodGetter) {
      return Filler(this, thinned, prodGetter);
    }

    void clear() { keys_.clear(); }

    bool contains(unsigned int key) const { return keys_.find(key) != keys_.end(); }

  private:
    std::unordered_set<unsigned int> keys_;
    ThinnedRefSetMode invalidParentRefMode_;
  };
}  // namespace edm

#endif
