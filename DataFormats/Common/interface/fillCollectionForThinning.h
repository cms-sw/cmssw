#ifndef DataFormats_Common_fillCollectionForThinning_h
#define DataFormats_Common_fillCollectionForThinning_h

// Implementation detail of thinning
//
// Need to be declared here in order to provide a customization hook
// for edmNew::DetSetVector. Definition is in ThinningProducer.h
namespace edm {
  class ThinnedAssociation;

  namespace detail {
    template <typename Item, typename Selector, typename Collection>
    void fillCollectionForThinning(
        Item const& item, Selector& selector, unsigned int iIndex, Collection& output, ThinnedAssociation& association);
  }
}  // namespace edm

#endif
