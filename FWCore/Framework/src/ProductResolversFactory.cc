#include "FWCore/Framework/interface/ProductResolversFactory.h"
#include "FWCore/Framework/interface/ProductResolverBase.h"
#include "DataFormats/Provenance/interface/ProductResolverIndexHelper.h"
#include "ProductResolvers.h"
#include "DroppedDataProductResolver.h"

#include <memory>

namespace edm::productResolversFactory {
  namespace {
    std::shared_ptr<ProductResolverBase> makeScheduledProduct(std::shared_ptr<BranchDescription const> bd) {
      return std::make_shared<PuttableProductResolver>(std::move(bd));
    }
    std::shared_ptr<ProductResolverBase> makeSourceProduct(std::shared_ptr<BranchDescription const> bd) {
      return std::make_shared<PuttableProductResolver>(std::move(bd));
    }
    std::shared_ptr<ProductResolverBase> makeDelayedReaderInputProduct(std::shared_ptr<BranchDescription const> bd) {
      return std::make_shared<DelayedReaderInputProductResolver>(std::move(bd));
    }
    std::shared_ptr<ProductResolverBase> makePutOnReadInputProduct(std::shared_ptr<BranchDescription const> bd) {
      return std::make_shared<PutOnReadInputProductResolver>(std::move(bd));
    }
    std::shared_ptr<ProductResolverBase> makeUnscheduledProduct(std::shared_ptr<BranchDescription const> bd) {
      return std::make_shared<UnscheduledProductResolver>(std::move(bd));
    }
    std::shared_ptr<ProductResolverBase> makeTransformProduct(std::shared_ptr<BranchDescription const> bd) {
      return std::make_shared<TransformingProductResolver>(std::move(bd));
    }
    std::shared_ptr<ProductResolverBase> makeDroppedProduct(std::shared_ptr<BranchDescription const> bd) {
      return std::make_shared<DroppedDataProductResolver>(std::move(bd));
    }

    std::shared_ptr<ProductResolverBase> makeAliasedProduct(
        std::shared_ptr<BranchDescription const> bd,
        ProductRegistry const& iReg,
        std::vector<std::shared_ptr<ProductResolverBase>> const& iResolvers) {
      ProductResolverIndex index = iReg.indexFrom(bd->originalBranchID());
      assert(index != ProductResolverIndexInvalid);
      return std::make_shared<AliasProductResolver>(
          std::move(bd), dynamic_cast<DataManagingOrAliasProductResolver&>(*iResolvers[index]));
    }
    std::shared_ptr<ProductResolverBase> makeSwitchProducerProduct(
        std::shared_ptr<BranchDescription const> bd,
        ProductRegistry const& iReg,
        std::vector<std::shared_ptr<ProductResolverBase>> const& iResolvers) {
      ProductResolverIndex index = iReg.indexFrom(bd->switchAliasForBranchID());
      assert(index != ProductResolverIndexInvalid);

      return std::make_shared<SwitchProducerProductResolver>(
          std::move(bd), dynamic_cast<DataManagingOrAliasProductResolver&>(*iResolvers[index]));
    }

    std::shared_ptr<ProductResolverBase> makeSwitchAliasProduct(
        std::shared_ptr<BranchDescription const> bd,
        ProductRegistry const& iReg,
        std::vector<std::shared_ptr<ProductResolverBase>> const& iResolvers) {
      ProductResolverIndex index = iReg.indexFrom(bd->switchAliasForBranchID());
      assert(index != ProductResolverIndexInvalid);

      return std::make_shared<SwitchAliasProductResolver>(
          std::move(bd), dynamic_cast<DataManagingOrAliasProductResolver&>(*iResolvers[index]));
    }

    std::shared_ptr<ProductResolverBase> makeParentProcessProduct(std::shared_ptr<BranchDescription const> bd) {
      return std::make_shared<ParentProcessProductResolver>(std::move(bd));
    }

    void addProductOrThrow(std::shared_ptr<ProductResolverBase> iResolver,
                           std::vector<std::shared_ptr<ProductResolverBase>>& oResolvers,
                           ProductRegistry const& iReg) {
      assert(bool(iResolver));
      BranchDescription const& bd = iResolver->branchDescription();
      assert(!bd.className().empty());
      assert(!bd.friendlyClassName().empty());
      assert(!bd.moduleLabel().empty());
      assert(!bd.processName().empty());
      auto index = iReg.indexFrom(bd.branchID());
      if (oResolvers[index]) {
        throw Exception(errors::InsertFailure, "AlreadyPresent")
            << "Problem found while adding a new ProductResolver, "
            << "product already exists for (" << bd.friendlyClassName() << "," << bd.moduleLabel() << ","
            << bd.productInstanceName() << "," << bd.processName() << ")\n";
      }
      oResolvers[index] = std::move(iResolver);
    }

    std::shared_ptr<ProductResolverBase> makeForPrimary(BranchDescription const& bd,
                                                        ProductRegistry const& iReg,
                                                        ProductResolverIndexHelper const& iHelper) {
      auto cbd = std::make_shared<BranchDescription const>(bd);
      if (bd.produced()) {
        using namespace std::literals;
        if (bd.moduleLabel() == "source"sv) {
          return makeSourceProduct(cbd);
        } else if (bd.onDemand()) {
          assert(bd.branchType() == InEvent);
          if (bd.isTransform()) {
            return makeTransformProduct(cbd);
          } else {
            return makeUnscheduledProduct(cbd);
          }
        }
        return makeScheduledProduct(cbd);
      }
      /* not produced so comes from source */
      if (bd.dropped()) {
        //this allows access to provenance for the dropped product
        return makeDroppedProduct(cbd);
      }
      if (bd.onDemand()) {
        return makeDelayedReaderInputProduct(cbd);
      }
      return makePutOnReadInputProduct(cbd);
    }
    bool isProductMadeAtEnd(edm::BranchIDList const& matchingHolders,
                            std::vector<bool> const& ambiguous,
                            std::vector<std::shared_ptr<edm::ProductResolverBase>> const& productResolvers) {
      for (unsigned int j = 0; j < matchingHolders.size(); ++j) {
        if ((not ambiguous[j]) and ProductResolverIndexInvalid != matchingHolders[j] and
            productResolvers[matchingHolders[j]]->branchDescription().availableOnlyAtEndTransition()) {
          return true;
        }
      }
      return false;
    }
    std::shared_ptr<ProductResolverBase> makeNoProcess(
        unsigned int numberOfMatches,
        ProductResolverIndex lastMatchIndex,
        edm::BranchIDList const& matchingHolders,
        std::vector<bool> const& ambiguous,
        std::vector<std::shared_ptr<edm::ProductResolverBase>> const& productResolvers) {
      if ((numberOfMatches == 1) and (lastMatchIndex != ProductResolverIndexAmbiguous)) {
        //only one choice so use a special resolver
        return std::make_shared<SingleChoiceNoProcessProductResolver>(lastMatchIndex);
      }
      //Need to know if the product from this processes is added at end of transition
      bool productMadeAtEnd = isProductMadeAtEnd(matchingHolders, ambiguous, productResolvers);
      return std::make_shared<NoProcessProductResolver>(matchingHolders, ambiguous, productMadeAtEnd);
    }

    void addUnspecifiedProcess(ProductResolverIndexHelper const& iHelper,
                               std::vector<std::shared_ptr<ProductResolverBase>>& productResolvers) {
      // Now create the ProductResolvers that search in reverse process
      // order and are used for queries where the process name is the
      // empty string
      std::vector<std::string> const& lookupProcessNames = iHelper.lookupProcessNames();
      std::vector<ProductResolverIndex> matchingHolders(lookupProcessNames.size(), ProductResolverIndexInvalid);
      std::vector<bool> ambiguous(lookupProcessNames.size(), false);
      unsigned int beginElements = iHelper.beginElements();
      std::vector<TypeID> const& sortedTypeIDs = iHelper.sortedTypeIDs();
      std::vector<ProductResolverIndexHelper::Range> const& ranges = iHelper.ranges();
      std::vector<ProductResolverIndexHelper::IndexAndNames> const& indexAndNames = iHelper.indexAndNames();
      std::vector<char> const& processNamesCharArray = iHelper.processNames();

      unsigned int numberOfMatches = 0;
      ProductResolverIndex lastMatchIndex = ProductResolverIndexInvalid;
      if (!sortedTypeIDs.empty()) {
        ProductResolverIndex productResolverIndex = ProductResolverIndexInvalid;
        for (unsigned int k = 0, kEnd = sortedTypeIDs.size(); k < kEnd; ++k) {
          ProductResolverIndexHelper::Range const& range = ranges.at(k);
          for (unsigned int i = range.begin(); i < range.end(); ++i) {
            ProductResolverIndexHelper::IndexAndNames const& product = indexAndNames.at(i);
            if (product.startInProcessNames() == 0) {
              if (productResolverIndex != ProductResolverIndexInvalid) {
                productResolvers.at(productResolverIndex) =
                    makeNoProcess(numberOfMatches, lastMatchIndex, matchingHolders, ambiguous, productResolvers);
                matchingHolders.assign(lookupProcessNames.size(), ProductResolverIndexInvalid);
                ambiguous.assign(lookupProcessNames.size(), false);
                numberOfMatches = 0;
                lastMatchIndex = ProductResolverIndexInvalid;
              }
              productResolverIndex = product.index();
            } else {
              const std::string_view process(&processNamesCharArray.at(product.startInProcessNames()));
              auto iter = std::find(lookupProcessNames.begin(), lookupProcessNames.end(), process);
              assert(iter != lookupProcessNames.end());
              ProductResolverIndex iMatchingIndex = product.index();
              lastMatchIndex = iMatchingIndex;
              assert(iMatchingIndex != ProductResolverIndexInvalid);
              ++numberOfMatches;
              if (iMatchingIndex == ProductResolverIndexAmbiguous) {
                assert(k >= beginElements);
                ambiguous.at(iter - lookupProcessNames.begin()) = true;
              } else {
                matchingHolders.at(iter - lookupProcessNames.begin()) = iMatchingIndex;
              }
            }
          }
        }
        productResolvers.at(productResolverIndex) =
            makeNoProcess(numberOfMatches, lastMatchIndex, matchingHolders, ambiguous, productResolvers);
      }
    }
  }  // namespace

  std::vector<std::shared_ptr<ProductResolverBase>> make(BranchType bt,
                                                         std::string_view iProcessName,
                                                         ProductRegistry const& iReg,
                                                         bool isForPrimaryProcess) {
    auto const& helper = iReg.productLookup(bt);
    std::vector<std::shared_ptr<ProductResolverBase>> productResolvers(iReg.getNextIndexValue(bt));
    ProductRegistry::ProductList const& prodsList = iReg.productList();
    // The constructor of an alias product holder takes as an argument the product holder for which it is an alias.
    // So, the non-alias product holders must be created first.
    // Therefore, on this first pass, skip current EDAliases.
    bool hasAliases = false;
    bool hasSwitchAliases = false;
    for (auto const& prod : prodsList) {
      BranchDescription const& bd = prod.second;
      if (bd.branchType() == bt) {
        if (isForPrimaryProcess or bd.processName() == iProcessName) {
          if (bd.isAlias()) {
            hasAliases = true;
          } else if (bd.isSwitchAlias()) {
            hasSwitchAliases = true;
          } else {
            addProductOrThrow(makeForPrimary(bd, iReg, *helper), productResolvers, iReg);
          }
        } else {
          //We are in a SubProcess and this branch is from the parent
          auto cbd = std::make_shared<BranchDescription const>(bd);
          if (bd.dropped()) {
            addProductOrThrow(makeDroppedProduct(cbd), productResolvers, iReg);
          } else {
            addProductOrThrow(makeParentProcessProduct(cbd), productResolvers, iReg);
          }
        }
      }
    }
    // Now process any EDAliases
    if (hasAliases) {
      for (auto const& prod : prodsList) {
        BranchDescription const& bd = prod.second;
        if (bd.isAlias() && bd.branchType() == bt) {
          addProductOrThrow(makeAliasedProduct(std::make_shared<BranchDescription const>(bd), iReg, productResolvers),
                            productResolvers,
                            iReg);
        }
      }
    }
    // Finally process any SwitchProducer aliases
    if (hasSwitchAliases) {
      for (auto const& prod : prodsList) {
        BranchDescription const& bd = prod.second;
        if (bd.isSwitchAlias() && bd.branchType() == bt) {
          assert(bt == InEvent);
          auto cbd = std::make_shared<BranchDescription const>(bd);
          // Need different implementation for SwitchProducers not
          // in any Path (onDemand) and for those in a Path in order
          // to prevent the switch-aliased-for EDProducers from
          // being run when the SwitchProducer is in a Path after a
          // failing EDFilter.
          if (bd.onDemand()) {
            addProductOrThrow(makeSwitchAliasProduct(cbd, iReg, productResolvers), productResolvers, iReg);
          } else {
            addProductOrThrow(makeSwitchProducerProduct(cbd, iReg, productResolvers), productResolvers, iReg);
          }
        }
      }
    }

    addUnspecifiedProcess(*helper, productResolvers);

    return productResolvers;
  }
}  // namespace edm::productResolversFactory