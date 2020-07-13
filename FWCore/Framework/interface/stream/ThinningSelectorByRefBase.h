#ifndef FWCore_Framework_ThinningSelectorByRefBase_h
#define FWCore_Framework_ThinningSelectorByRefBase_h

#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefVector.h"

namespace edm {

  template <typename Collection, typename T = typename Collection::value_type>
  class ThinningSelectorByRefBase {
  public:
    virtual ~ThinningSelectorByRefBase<Collection, T>() {}
    void preChoose(Handle<Collection> inputCollection, Event const& event, EventSetup const& es);
    bool choose(unsigned int iIndex, T const& iItem);
    virtual void preChooseRefs(Handle<Collection> inputCollection, Event const& event, EventSetup const& es) = 0;
    virtual void modify(T& iItem) {}

  protected:
    void addRef(Ref<Collection> const& ref);
    void addRefVector(RefVector<Collection> const& refv);

  private:
    std::map<ProductID, std::vector<unsigned int> > idxMap_;
    std::vector<bool> keepMask_;
  };

  template <typename Collection, typename T>
  void ThinningSelectorByRefBase<Collection, T>::preChoose(Handle<Collection> inputCollection,
                                                           Event const& event,
                                                           EventSetup const& es) {
    //clear map
    idxMap_.clear();
    //reset mask
    keepMask_.clear();
    keepMask_.resize(inputCollection->size(), false);

    //run derived class method which should call addRef and/or addRefVector on relevant refs which may point to collection being thinned
    preChooseRefs(inputCollection, event, es);

    //find indices to thinned collection from refs and set the mask of items to keep
    for (auto& entry : idxMap_) {
      ProductID const& id = entry.first;
      std::vector<unsigned int>& keys = entry.second;
      unsigned int nkeys = keys.size();
      if (id == inputCollection.id()) {
        //collections match, use indices directly
        for (unsigned int i = 0; i < nkeys; ++i) {
          keepMask_[keys[i]] = true;
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
        keepMask_[keys[i]] = true;
      }
    }
  }

  template <typename Collection, typename T>
  bool ThinningSelectorByRefBase<Collection, T>::choose(unsigned int iIndex, T const& iItem) {
    return keepMask_[iIndex];
  }

  template <typename Collection, typename T>
  void ThinningSelectorByRefBase<Collection, T>::addRef(Ref<Collection> const& ref) {
    idxMap_[ref.id()].push_back(ref.key());
  }

  template <typename Collection, typename T>
  void ThinningSelectorByRefBase<Collection, T>::addRefVector(RefVector<Collection> const& refv) {
    for (edm::Ref<Collection> const& ref : refv) {
      idxMap_[ref.id()].push_back(ref.key());
    }
  }
}  // namespace edm
#endif
