#ifndef FWCore_Framework_ThinningSelectorByRefBase_h
#define FWCore_Framework_ThinningSelectorByRefBase_h

#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefVector.h"

namespace edm {

  template <typename RefT>
  class ThinningSelectorByRefBase {
  public:
    virtual ~ThinningSelectorByRefBase<RefT>() {}
    void preChoose(Handle<typename RefT::product_type> inputCollection, Event const& event, EventSetup const& es);
    bool choose(unsigned int iIndex, typename RefT::value_type const& iItem);
    virtual void preChooseRefs(Handle<typename RefT::product_type> inputCollection, Event const& event, EventSetup const& es) = 0;
    virtual void modify(typename RefT::value_type& iItem) {}

  protected:
    void addRef(RefT const& ref);
    void addRefVector(RefVector<typename RefT::product_type, typename RefT::value_type, typename RefT::finder_type> const& refv);

  private:
    std::map<ProductID, std::vector<unsigned int> > idxMap_;
    std::set<unsigned int> keysToKeep_;
  };

  template <typename RefT>
  void ThinningSelectorByRefBase<RefT>::preChoose(Handle<typename RefT::product_type> inputCollection,
                                                           Event const& event,
                                                           EventSetup const& es) {
    //clear map
    idxMap_.clear();
    //reset set
    keysToKeep_.clear();

    //run derived class method which should call addRef and/or addRefVector on relevant refs which may point to collection being thinned
    preChooseRefs(inputCollection, event, es);

    //find indices to thinned collection from refs and fill the set of items to keep
    for (auto& entry : idxMap_) {
      ProductID const& id = entry.first;
      std::vector<unsigned int>& keys = entry.second;
      unsigned int nkeys = keys.size();
      if (id == inputCollection.id()) {
        //collections match, use indices directly
        for (unsigned int i = 0; i < nkeys; ++i) {
          keysToKeep_.insert(keys[i]);
        }
        continue;
      }
      //check for matching thinned collections
      std::vector<WrapperBase const*> thinnedprods(nkeys, nullptr);
      event.productGetter().getThinnedProducts(id, thinnedprods, keys, inputCollection.id());
      for (unsigned int i = 0; i < nkeys; ++i) {
        if (thinnedprods[i] == nullptr) {
          continue;
        }
        keysToKeep_.insert(keys[i]);
      }
    }
  }

  template <typename RefT>
  bool ThinningSelectorByRefBase<RefT>::choose(unsigned int iIndex, typename RefT::value_type const& iItem) {
    return keysToKeep_.count(iIndex) > 0;
  }

  template <typename RefT>
  void ThinningSelectorByRefBase<RefT>::addRef(RefT const& ref) {
    if (ref.isNonnull()) {
      idxMap_[ref.id()].push_back(ref.key());
    }
  }

  template <typename RefT>
  void ThinningSelectorByRefBase<RefT>::addRefVector(RefVector<typename RefT::product_type, typename RefT::value_type, typename RefT::finder_type> const& refv) {
    for (RefT const& ref : refv) {
      idxMap_[ref.id()].push_back(ref.key());
    }
  }
}  // namespace edm
#endif
