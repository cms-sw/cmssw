#ifndef DataFormats_Common_getThinned_implementation_h
#define DataFormats_Common_getThinned_implementation_h

#include <algorithm>
#include <cassert>
#include <functional>
#include <optional>
#include <tuple>
#include <variant>

#include "DataFormats/Common/interface/ThinnedAssociation.h"
#include "DataFormats/Provenance/interface/BranchID.h"
#include "DataFormats/Provenance/interface/ProductID.h"
#include "DataFormats/Provenance/interface/ThinnedAssociationsHelper.h"
#include "FWCore/Utilities/interface/EDMException.h"

namespace edm {
  class WrapperBase;

  namespace detail {
    constexpr unsigned int kThinningDoNotLookForThisIndex = std::numeric_limits<unsigned int>::max();

    class ThinnedOrSlimmedProduct {
    public:
      ThinnedOrSlimmedProduct() = default;
      explicit ThinnedOrSlimmedProduct(WrapperBase const* thinned, unsigned int key)
          : thinnedProduct_{thinned}, thinnedKey_{key} {}
      explicit ThinnedOrSlimmedProduct(ThinnedAssociation const* slimmed, unsigned int key)
          : slimmedAssociation_{slimmed}, thinnedKey_{key} {}

      bool hasThinned() const { return thinnedProduct_ != nullptr; }
      bool hasSlimmed() const { return slimmedAssociation_ != nullptr; }

      std::tuple<WrapperBase const*, unsigned int> thinnedProduct() const {
        return std::tuple(thinnedProduct_, thinnedKey_);
      }

      std::tuple<ThinnedAssociation const*, unsigned int> slimmedAssociation() const {
        return std::tuple(slimmedAssociation_, thinnedKey_);
      }

    private:
      WrapperBase const* thinnedProduct_ = nullptr;
      ThinnedAssociation const* slimmedAssociation_ = nullptr;
      unsigned int thinnedKey_ = 0;
    };

    // This is a helper function to recursively search for a thinned
    // product containing the parent key on the same "slimming depth". That
    // means that when the recursion encounters a slimmed collection,
    // the tree traversal does not proceed onto the children of the
    // slimmed collection. Instead, the slimmed ThinnedAssociation is
    // recorded for the case that the entire tree on a given "slimming
    // depth" does not have any thinned-only collections.
    //
    // Returns
    // - (WrapperBase, unsigned) in case a thinned collection
    //   containing the parent key was found.
    // - (ThinnedAssociation, unsigned) in case no thinned collections
    //   were encountered, but a slimmed collection containing the
    //   parent key was found
    // - otherwise "null" (i.e. only thinned collections without the
    //   parent key, or in absence of thinned collections the slimmed
    //   collection without the parent key, or no thinned or slimmed
    //   collections)
    template <typename F1, typename F2, typename F3>
    ThinnedOrSlimmedProduct getThinnedProductOnSlimmingDepth(ProductID const& pid,
                                                             unsigned int key,
                                                             ThinnedAssociationsHelper const& thinnedAssociationsHelper,
                                                             F1 pidToBid,
                                                             F2 getThinnedAssociation,
                                                             F3 getByProductID) {
      BranchID parent = pidToBid(pid);

      auto associatedBranches = thinnedAssociationsHelper.parentBegin(parent);
      auto const iEnd = thinnedAssociationsHelper.parentEnd(parent);

      if (associatedBranches == iEnd) {
        return ThinnedOrSlimmedProduct();
      }
      bool const slimmedAllowed = (associatedBranches + 1 == iEnd);
      if (slimmedAllowed and associatedBranches->isSlimmed()) {
        // Slimmed container can be considered only if it has no (thinned) siblings
        ThinnedAssociation const* slimmedAssociation = getThinnedAssociation(associatedBranches->association());
        if (slimmedAssociation == nullptr or
            associatedBranches->parent() != pidToBid(slimmedAssociation->parentCollectionID())) {
          return ThinnedOrSlimmedProduct();
        }

        // Does this slimmed container have the element referenced by key?
        auto slimmedIndex = slimmedAssociation->getThinnedIndex(key);
        if (slimmedIndex.has_value()) {
          return ThinnedOrSlimmedProduct(slimmedAssociation, *slimmedIndex);
        } else {
          return ThinnedOrSlimmedProduct();
        }
      }

      // Loop over thinned containers which were made by selecting elements from the parent container
      for (; associatedBranches != iEnd; ++associatedBranches) {
        if (associatedBranches->isSlimmed()) {
          continue;
        }

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

        // Return a pointer to thinned container if we can find it
        ProductID const& thinnedCollectionPID = thinnedAssociation->thinnedCollectionID();
        WrapperBase const* thinnedCollection = getByProductID(thinnedCollectionPID);
        if (thinnedCollection != nullptr) {
          return ThinnedOrSlimmedProduct(thinnedCollection, *thinnedIndex);
        }

        // Thinned container is not found, try looking recursively in thinned containers
        // which were made by selecting elements from this thinned container.
        auto thinnedOrSlimmed = getThinnedProductOnSlimmingDepth(thinnedCollectionPID,
                                                                 *thinnedIndex,
                                                                 thinnedAssociationsHelper,
                                                                 pidToBid,
                                                                 getThinnedAssociation,
                                                                 getByProductID);
        if (thinnedOrSlimmed.hasThinned() or (slimmedAllowed and thinnedOrSlimmed.hasSlimmed())) {
          return thinnedOrSlimmed;
        }
      }

      return ThinnedOrSlimmedProduct();
    }

    inline auto makeThinnedIndexes(std::vector<unsigned int> const& keys,
                                   std::vector<WrapperBase const*> const& foundContainers,
                                   ThinnedAssociation const* thinnedAssociation) {
      unsigned const nKeys = keys.size();
      std::vector<unsigned int> thinnedIndexes(nKeys, kThinningDoNotLookForThisIndex);
      bool hasAny = false;
      for (unsigned k = 0; k < nKeys; ++k) {
        // Already found this one
        if (foundContainers[k] != nullptr) {
          continue;
        }
        // Already know this one is not in this thinned container
        if (keys[k] == kThinningDoNotLookForThisIndex) {
          continue;
        }
        // Does the thinned container hold the entry of interest?
        if (auto thinnedIndex = thinnedAssociation->getThinnedIndex(keys[k]); thinnedIndex.has_value()) {
          thinnedIndexes[k] = *thinnedIndex;
          hasAny = true;
        }
      }
      return std::tuple(std::move(thinnedIndexes), hasAny);
    }

    // This is a helper function to recursive search ffor thinned
    // collections that contain some of the parent keys on the same
    // "slimming depth". That means that when the recursion encounters
    // a slimmed colleciton, the tree traversal does not proceed onto
    // the children of the slimmed collection. Instead, the slimmed
    // ThinnedAssociation is recorded for the case that the entire
    // tree on a given "slimming depth" does not have any thinned-only
    // collections.
    //
    // Returns a (ThinnedAssociation, vector<unsigned>) in case no
    // thinned collections were encountered, but a slimmed collection
    // containing at least one of the parent keys was found. The
    // returned vector contains keys to the slimmed collection, and
    // the output arguments foundContainers and keys are not modified
    // in this case.
    //
    // Otherwise returns a null optional (i.e. any thinned collection
    // was encountered, or in absence of thinned collections the
    // slimmed collection did not contain any parent keys, or there
    // were no thinned or slimmed collections)
    template <typename F1, typename F2, typename F3>
    std::optional<std::tuple<ThinnedAssociation const*, std::vector<unsigned int>>> getThinnedProductsOnSlimmingDepth(
        ProductID const& pid,
        ThinnedAssociationsHelper const& thinnedAssociationsHelper,
        F1 pidToBid,
        F2 getThinnedAssociation,
        F3 getByProductID,
        std::vector<WrapperBase const*>& foundContainers,
        std::vector<unsigned int>& keys) {
      BranchID parent = pidToBid(pid);

      auto associatedBranches = thinnedAssociationsHelper.parentBegin(parent);
      auto const iEnd = thinnedAssociationsHelper.parentEnd(parent);

      if (associatedBranches == iEnd) {
        return std::nullopt;
      }
      bool const slimmedAllowed = associatedBranches + 1 == iEnd;
      if (slimmedAllowed and associatedBranches->isSlimmed()) {
        // Slimmed container can be considered only if it has no (thinned) siblings
        ThinnedAssociation const* slimmedAssociation = getThinnedAssociation(associatedBranches->association());
        if (slimmedAssociation == nullptr or
            associatedBranches->parent() != pidToBid(slimmedAssociation->parentCollectionID())) {
          return std::nullopt;
        }

        auto [slimmedIndexes, hasAny] = makeThinnedIndexes(keys, foundContainers, slimmedAssociation);
        // Does this slimmed container have any of the elements referenced by keys?
        if (hasAny) {
          return std::tuple(slimmedAssociation, std::move(slimmedIndexes));
        } else {
          return std::nullopt;
        }
      }

      // Loop over thinned containers which were made by selecting elements from the parent container
      for (; associatedBranches != iEnd; ++associatedBranches) {
        if (associatedBranches->isSlimmed()) {
          continue;
        }

        ThinnedAssociation const* thinnedAssociation = getThinnedAssociation(associatedBranches->association());
        if (thinnedAssociation == nullptr)
          continue;

        if (associatedBranches->parent() != pidToBid(thinnedAssociation->parentCollectionID())) {
          continue;
        }

        auto [thinnedIndexes, hasAny] = makeThinnedIndexes(keys, foundContainers, thinnedAssociation);
        if (!hasAny) {
          continue;
        }

        // Set the pointers and indexes into the thinned container (if we can find it)
        ProductID thinnedCollectionPID = thinnedAssociation->thinnedCollectionID();
        WrapperBase const* thinnedCollection = getByProductID(thinnedCollectionPID);
        unsigned const nKeys = keys.size();
        if (thinnedCollection == nullptr) {
          // Thinned container is not found, try looking recursively in thinned containers
          // which were made by selecting elements from this thinned container.
          auto slimmed = getThinnedProductsOnSlimmingDepth(thinnedCollectionPID,
                                                           thinnedAssociationsHelper,
                                                           pidToBid,
                                                           getThinnedAssociation,
                                                           getByProductID,
                                                           foundContainers,
                                                           thinnedIndexes);
          if (slimmedAllowed and slimmed.has_value()) {
            return slimmed;
          }
          for (unsigned k = 0; k < nKeys; ++k) {
            if (foundContainers[k] == nullptr)
              continue;
            if (thinnedIndexes[k] == kThinningDoNotLookForThisIndex)
              continue;
            keys[k] = thinnedIndexes[k];
          }
        } else {
          for (unsigned k = 0; k < nKeys; ++k) {
            if (thinnedIndexes[k] == kThinningDoNotLookForThisIndex)
              continue;
            keys[k] = thinnedIndexes[k];
            foundContainers[k] = thinnedCollection;
          }
        }
      }
      return std::nullopt;
    }

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
    std::optional<std::tuple<WrapperBase const*, unsigned int>> getThinnedProduct(
        ProductID const& pid,
        unsigned int key,
        ThinnedAssociationsHelper const& thinnedAssociationsHelper,
        F1 pidToBid,
        F2 getThinnedAssociation,
        F3 getByProductID) {
      auto thinnedOrSlimmed = getThinnedProductOnSlimmingDepth(
          pid, key, thinnedAssociationsHelper, pidToBid, getThinnedAssociation, getByProductID);

      if (thinnedOrSlimmed.hasThinned()) {
        return thinnedOrSlimmed.thinnedProduct();
      } else if (thinnedOrSlimmed.hasSlimmed()) {
        auto [slimmedAssociation, slimmedIndex] = thinnedOrSlimmed.slimmedAssociation();
        ProductID const& slimmedCollectionPID = slimmedAssociation->thinnedCollectionID();
        WrapperBase const* slimmedCollection = getByProductID(slimmedCollectionPID);
        if (slimmedCollection == nullptr) {
          // Slimmed container is not found, try looking recursively in thinned containers
          // which were made by selecting elements from this thinned container.
          return getThinnedProduct(slimmedCollectionPID,
                                   slimmedIndex,
                                   thinnedAssociationsHelper,
                                   pidToBid,
                                   getThinnedAssociation,
                                   getByProductID);
        }
        return std::tuple(slimmedCollection, slimmedIndex);
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
      auto slimmed = getThinnedProductsOnSlimmingDepth(
          pid, thinnedAssociationsHelper, pidToBid, getThinnedAssociation, getByProductID, foundContainers, keys);
      if (slimmed.has_value()) {
        // no thinned procucts found, try out slimmed next if one is available
        auto [slimmedAssociation, slimmedIndexes] = std::move(*slimmed);
        ProductID const& slimmedCollectionPID = slimmedAssociation->thinnedCollectionID();
        WrapperBase const* slimmedCollection = getByProductID(slimmedCollectionPID);
        unsigned const nKeys = keys.size();
        if (slimmedCollection == nullptr) {
          getThinnedProducts(slimmedCollectionPID,
                             thinnedAssociationsHelper,
                             pidToBid,
                             getThinnedAssociation,
                             getByProductID,
                             foundContainers,
                             slimmedIndexes);
          for (unsigned k = 0; k < nKeys; ++k) {
            if (foundContainers[k] == nullptr)
              continue;
            if (slimmedIndexes[k] == kThinningDoNotLookForThisIndex)
              continue;
            keys[k] = slimmedIndexes[k];
          }
        } else {
          for (unsigned k = 0; k < nKeys; ++k) {
            if (slimmedIndexes[k] == kThinningDoNotLookForThisIndex)
              continue;
            keys[k] = slimmedIndexes[k];
            foundContainers[k] = slimmedCollection;
          }
        }
      }
    }

    using GetThinnedKeyFromExceptionFactory = std::function<edm::Exception()>;

    // This function provides a common implementation of
    // EDProductGetter::getThinnedKeyFrom() for EventPrincipal,
    // DataGetterHelper, and BareRootProductGetter.
    //
    // The thinned ProductID must come from an existing RefCore. The
    // input key is the index of the desired element in the container
    // identified by the parent ProductID. Returns an std::variant
    // whose contents can be
    // - unsigned int for the index in the thinned collection if the
    //   desired element was found in the thinned collection
    // - function creating an edm::Exception if parent is not a parent
    //   of any thinned collection, thinned is not really a thinned
    //   collection, or parent and thinned have no thinning
    //   relationship
    // - std::monostate if thinned is thinned from parent, but the key
    //   is not found in the thinned collection
    template <typename F>
    std::variant<unsigned int, GetThinnedKeyFromExceptionFactory, std::monostate> getThinnedKeyFrom_implementation(
        ProductID const& parentID,
        BranchID const& parent,
        unsigned int key,
        ProductID const& thinnedID,
        BranchID thinned,
        ThinnedAssociationsHelper const& thinnedAssociationsHelper,
        F&& getThinnedAssociation) {
      // need to explicitly check for equality of parent BranchID,
      // because ThinnedAssociationsHelper::parentBegin() uses
      // std::lower_bound() that returns a valid iterator in case the
      // parent is not found
      if (auto iParent = thinnedAssociationsHelper.parentBegin(parent);
          iParent == thinnedAssociationsHelper.parentEnd(parent) or iParent->parent() != parent) {
        return [parentID]() {
          return Exception(errors::InvalidReference)
                 << "Parent collection with ProductID " << parentID << " has not been thinned";
        };
      }

      bool foundParent = false;
      std::vector<ThinnedAssociation const*> thinnedAssociationParentage;
      while (not foundParent) {
        // TODO: be smarter than linear search every time?
        auto branchesToThinned = std::find_if(
            thinnedAssociationsHelper.begin(), thinnedAssociationsHelper.end(), [&thinned](auto& associatedBranches) {
              return associatedBranches.thinned() == thinned;
            });
        if (branchesToThinned == thinnedAssociationsHelper.end()) {
          return [parentID, thinnedID, thinnedIsThinned = not thinnedAssociationParentage.empty()]() {
            Exception ex(errors::InvalidReference);
            ex << "Requested thinned collection with ProductID " << thinnedID
               << " is not thinned from the parent collection with ProductID " << parentID
               << " or from any collection thinned from it.";
            if (not thinnedIsThinned) {
              ex << " In fact, the collection " << thinnedID
                 << " passed in as a 'thinned' collection has not been thinned at all.";
            }
            return ex;
          };
        }

        ThinnedAssociation const* thinnedAssociation = getThinnedAssociation(branchesToThinned->association());
        if (thinnedAssociation == nullptr) {
          Exception ex{errors::LogicError};
          if (thinnedAssociationParentage.empty()) {
            ex << "ThinnedAssociation corresponding to thinned collection with ProductID " << thinnedID
               << " not found.";
          } else {
            ex << "Intermediate ThinnedAssociation between the requested thinned ProductID " << thinnedID
               << " and parent " << parentID << " not found.";
          }
          ex << " This should not happen.\nPlease contact the core framework developers.";
          throw ex;
        }

        thinnedAssociationParentage.push_back(thinnedAssociation);
        if (branchesToThinned->parent() == parent) {
          foundParent = true;
        } else {
          // next iteration with current parent as the thinned collection
          thinned = branchesToThinned->parent();
        }
      }

      // found the parent, now need to rewind the parentage chain to
      // find the index in the requested thinned collection
      unsigned int thinnedIndex = key;
      for (auto iAssociation = thinnedAssociationParentage.rbegin(), iEnd = thinnedAssociationParentage.rend();
           iAssociation != iEnd;
           ++iAssociation) {
        auto optIndex = (*iAssociation)->getThinnedIndex(thinnedIndex);
        if (optIndex) {
          thinnedIndex = *optIndex;
        } else {
          return std::monostate{};
        }
      }
      return thinnedIndex;
    }

  }  // namespace detail
}  // namespace edm

#endif
