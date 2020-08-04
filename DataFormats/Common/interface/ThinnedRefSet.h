#ifndef DataFormats_Common_ThinnedRefSet_h
#define DataFormats_Common_ThinnedRefSet_h

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefItemGet.h"

#include <unordered_set>

namespace edm {
  class EDProductGetter;

  template <typename C>
  class ThinnedRefSet {
  public:
    class Filler {
    public:
      Filler(ThinnedRefSet<C>* set, RefProd<C> refProd, edm::EDProductGetter const& prodGetter)
          : set_(set), refProd_(refProd), prodGetter_(prodGetter) {}

      template <typename T, typename F>
      void insert(Ref<C, T, F> const& ref) {
        if (ref.isNull())
          return;
        auto thinnedRef = thinnedRefFrom(ref, refProd_, prodGetter_);
        set_->keys_.insert(thinnedRef.key());
      }

    private:
      ThinnedRefSet<C>* set_;
      RefProd<C> refProd_;
      edm::EDProductGetter const& prodGetter_;
    };

    ThinnedRefSet() = default;

    Filler fill(RefProd<C> refProd, edm::EDProductGetter const& prodGetter) {
      return Filler(this, refProd, prodGetter);
    }

    void clear() { keys_.clear(); }

    bool contains(unsigned int key) const { return keys_.find(key) != keys_.end(); }

  private:
    std::unordered_set<unsigned int> keys_;
  };
}  // namespace edm

#endif
