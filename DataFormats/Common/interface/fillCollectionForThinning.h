#ifndef DataFormats_Common_fillCollectionForThinning_h
#define DataFormats_Common_fillCollectionForThinning_h

#include <type_traits>

// Implementation detail of thinning
//
// Need to be declared here in order to provide a customization hooks
// for edmNew::DetSetVector.
namespace edm {
  class ThinnedAssociation;

  namespace detail {
    // by default a linear container
    template <typename Collection>
    struct ElementType {
      using type = typename std::remove_reference<decltype(*std::declval<Collection>().begin())>::type;
    };

    // Defined in ThinningProducer.h
    template <typename Item, typename Selector, typename Collection>
    void fillCollectionForThinning(
        Item const& item, Selector& selector, unsigned int iIndex, Collection& output, ThinnedAssociation& association);
  }  // namespace detail
}  // namespace edm

#endif
