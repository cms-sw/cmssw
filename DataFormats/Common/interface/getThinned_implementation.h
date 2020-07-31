#ifndef DataFormats_Common_getThinned_implementation_h
#define DataFormats_Common_getThinned_implementation_h

#include <optional>
#include <tuple>

#include "DataFormats/Common/interface/ThinnedAssociation.h"
#include "DataFormats/Provenance/interface/BranchID.h"
#include "DataFormats/Provenance/interface/ProductID.h"
#include "DataFormats/Provenance/interface/ThinnedAssociationsHelper.h"

namespace edm {
  class WrapperBase;

  namespace detail {
    // This function provides a common implementation of
    // EDProductGetter::getThinnedProduct() for EventPrincipal,
    // DataGetterHelper, and BareRootProductGetter.
    //
    // getThinnedProduct assumes getIt was already called and failed to find
    // the product. The input key is the index of the desired element in the
    // container identified by ProductID (which cannot be found).
    // If the return value is not null, then the desired element was
    // found in a thinned container. If the desired element is not
    // found, then an optional without a value is returned.
    template <typename F1, typename F2, typename F3>
    std::optional<std::tuple<WrapperBase const*, unsigned int> > getThinnedProduct(
        ProductID const& pid,
        unsigned int key,
        ThinnedAssociationsHelper const& thinnedAssociationsHelper,
        F1 pidToBid,
        F2 getThinnedAssociation,
        F3 getByProductID) {
      BranchID parent = pidToBid(pid);

      // Loop over thinned containers which were made by selecting elements from the parent container
      for (auto associatedBranches = thinnedAssociationsHelper.parentBegin(parent),
                iEnd = thinnedAssociationsHelper.parentEnd(parent);
           associatedBranches != iEnd;
           ++associatedBranches) {
        ThinnedAssociation const* thinnedAssociation = getThinnedAssociation(associatedBranches->association());
        if (thinnedAssociation == nullptr)
          continue;

        if (associatedBranches->parent() != pidToBid(thinnedAssociation->parentCollectionID())) {
          continue;
        }

        // Does this thinned container have the element referenced by key?
        auto thinnedIndex = thinnedAssociation->getThinnedIndex(key);
        if (not thinnedIndex.has_value()) {
          continue;
        }

        // Get the thinned container and return a pointer if we can find it
        ProductID const& thinnedCollectionPID = thinnedAssociation->thinnedCollectionID();
        WrapperBase const* thinnedCollection = getByProductID(thinnedCollectionPID);
        if (thinnedCollection == nullptr) {
          // Thinned container is not found, try looking recursively in thinned containers
          // which were made by selecting elements from this thinned container.
          auto thinnedCollectionKey = getThinnedProduct(thinnedCollectionPID,
                                                        *thinnedIndex,
                                                        thinnedAssociationsHelper,
                                                        pidToBid,
                                                        getThinnedAssociation,
                                                        getByProductID);
          if (thinnedCollectionKey.has_value()) {
            return thinnedCollectionKey;
          } else {
            continue;
          }
        }
        return std::tuple(thinnedCollection, *thinnedIndex);
      }
      return std::nullopt;
    }

    // This function provides a common implementation of
    // EDProductGetter::getThinnedProducts() for EventPrincipal,
    // DataGetterHelper, and BareRootProductGetter.
    //
    // getThinnedProducts assumes getIt was already called and failed to find
    // the product. The input keys are the indexes into the container identified
    // by ProductID (which cannot be found). On input the WrapperBase pointers
    // must all be set to nullptr (except when the function calls itself
    // recursively where non-null pointers mark already found elements).
    // Thinned containers derived from the product are searched to see
    // if they contain the desired elements. For each that is
    // found, the corresponding WrapperBase pointer is set and the key
    // is modified to be the key into the container where the element
    // was found. The WrapperBase pointers might or might not all point
    // to the same thinned container.
    template <typename F1, typename F2, typename F3>
    void getThinnedProducts(ProductID const& pid,
                            ThinnedAssociationsHelper const& thinnedAssociationsHelper,
                            F1 pidToBid,
                            F2 getThinnedAssociation,
                            F3 getByProductID,
                            std::vector<WrapperBase const*>& foundContainers,
                            std::vector<unsigned int>& keys) {
      BranchID parent = pidToBid(pid);

      // Loop over thinned containers which were made by selecting elements from the parent container
      for (auto associatedBranches = thinnedAssociationsHelper.parentBegin(parent),
                iEnd = thinnedAssociationsHelper.parentEnd(parent);
           associatedBranches != iEnd;
           ++associatedBranches) {
        ThinnedAssociation const* thinnedAssociation = getThinnedAssociation(associatedBranches->association());
        if (thinnedAssociation == nullptr)
          continue;

        if (associatedBranches->parent() != pidToBid(thinnedAssociation->parentCollectionID())) {
          continue;
        }

        unsigned nKeys = keys.size();
        unsigned int doNotLookForThisIndex = std::numeric_limits<unsigned int>::max();
        std::vector<unsigned int> thinnedIndexes(nKeys, doNotLookForThisIndex);
        bool hasAny = false;
        for (unsigned k = 0; k < nKeys; ++k) {
          // Already found this one
          if (foundContainers[k] != nullptr)
            continue;
          // Already know this one is not in this thinned container
          if (keys[k] == doNotLookForThisIndex)
            continue;
          // Does the thinned container hold the entry of interest?
          if (auto thinnedIndex = thinnedAssociation->getThinnedIndex(keys[k]); thinnedIndex.has_value()) {
            thinnedIndexes[k] = *thinnedIndex;
            hasAny = true;
          }
        }
        if (!hasAny) {
          continue;
        }
        // Get the thinned container and set the pointers and indexes into
        // it (if we can find it)
        ProductID thinnedCollectionPID = thinnedAssociation->thinnedCollectionID();
        WrapperBase const* thinnedCollection = getByProductID(thinnedCollectionPID);
        if (thinnedCollection == nullptr) {
          // Thinned container is not found, try looking recursively in thinned containers
          // which were made by selecting elements from this thinned container.
          getThinnedProducts(thinnedCollectionPID,
                             thinnedAssociationsHelper,
                             pidToBid,
                             getThinnedAssociation,
                             getByProductID,
                             foundContainers,
                             thinnedIndexes);
          for (unsigned k = 0; k < nKeys; ++k) {
            if (foundContainers[k] == nullptr)
              continue;
            if (thinnedIndexes[k] == doNotLookForThisIndex)
              continue;
            keys[k] = thinnedIndexes[k];
          }
        } else {
          for (unsigned k = 0; k < nKeys; ++k) {
            if (thinnedIndexes[k] == doNotLookForThisIndex)
              continue;
            keys[k] = thinnedIndexes[k];
            foundContainers[k] = thinnedCollection;
          }
        }
      }
    }

  }  // namespace detail
}  // namespace edm

#endif
