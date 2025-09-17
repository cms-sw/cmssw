#include "FWCore/Framework/interface/ProductResolversFactory.h"
#include "FWCore/Framework/interface/ProductResolverBase.h"
#include "DataFormats/Provenance/interface/ProductResolverIndexHelper.h"
#include "DataFormats/Provenance/interface/BranchIDList.h"
#include "ProductResolvers.h"
#include "DroppedDataProductResolver.h"

#include <memory>

namespace edm::productResolversFactory {
  namespace {
    std::shared_ptr<ProductResolverBase> makeScheduledProduct(std::shared_ptr<ProductDescription const> bd) {
      return std::make_shared<PuttableProductResolver>(std::move(bd));
    }
    std::shared_ptr<ProductResolverBase> makeSourceProduct(std::shared_ptr<ProductDescription const> bd) {
      return std::make_shared<PuttableProductResolver>(std::move(bd));
    }
    std::shared_ptr<ProductResolverBase> makeDelayedReaderInputProduct(std::shared_ptr<ProductDescription const> bd) {
      return std::make_shared<DelayedReaderInputProductResolver>(std::move(bd));
    }
    std::shared_ptr<ProductResolverBase> makePutOnReadInputProduct(std::shared_ptr<ProductDescription const> bd) {
      return std::make_shared<PutOnReadInputProductResolver>(std::move(bd));
    }
    std::shared_ptr<ProductResolverBase> makeUnscheduledProduct(std::shared_ptr<ProductDescription const> bd) {
      return std::make_shared<UnscheduledProductResolver>(std::move(bd));
    }
    std::shared_ptr<ProductResolverBase> makeTransformProduct(std::shared_ptr<ProductDescription const> bd) {
      return std::make_shared<TransformingProductResolver>(std::move(bd));
    }
    std::shared_ptr<ProductResolverBase> makeDroppedProduct(std::shared_ptr<ProductDescription const> bd) {
      return std::make_shared<DroppedDataProductResolver>(std::move(bd));
    }

    std::shared_ptr<ProductResolverBase> makeAliasedProduct(
        std::shared_ptr<ProductDescription const> bd,
        ProductRegistry const& iReg,
        std::vector<std::shared_ptr<ProductResolverBase>> const& iResolvers) {
      ProductResolverIndex index = iReg.indexFrom(bd->originalBranchID());
      assert(index != ProductResolverIndexInvalid);
      return std::make_shared<AliasProductResolver>(
          std::move(bd), dynamic_cast<DataManagingOrAliasProductResolver&>(*iResolvers[index]));
    }

    void addProductOrThrow(std::shared_ptr<ProductResolverBase> iResolver,
                           std::vector<std::shared_ptr<ProductResolverBase>>& oResolvers,
                           ProductRegistry const& iReg) {
      assert(bool(iResolver));
      ProductDescription const& bd = iResolver->productDescription();
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

    std::shared_ptr<ProductResolverBase> makeForPrimary(ProductDescription const& bd,
                                                        ProductRegistry const& iReg,
                                                        ProductResolverIndexHelper const& iHelper) {
      auto cbd = std::make_shared<ProductDescription const>(bd);
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
  }  // namespace

  std::vector<std::shared_ptr<ProductResolverBase>> make(BranchType bt,
                                                         std::string_view iProcessName,
                                                         ProductRegistry const& iReg) {
    auto const& helper = iReg.productLookup(bt);
    std::vector<std::shared_ptr<ProductResolverBase>> productResolvers(iReg.getNextIndexValue(bt));
    ProductRegistry::ProductList const& prodsList = iReg.productList();
    // The constructor of an alias product holder takes as an argument the product holder for which it is an alias.
    // So, the non-alias product holders must be created first.
    // Therefore, on this first pass, skip current EDAliases.
    bool hasAliases = false;
    for (auto const& prod : prodsList) {
      ProductDescription const& bd = prod.second;
      if (bd.branchType() == bt) {
        if (bd.isAlias()) {
          hasAliases = true;
        } else {
          addProductOrThrow(makeForPrimary(bd, iReg, *helper), productResolvers, iReg);
        }
      }
    }
    // Now process any EDAliases
    if (hasAliases) {
      for (auto const& prod : prodsList) {
        ProductDescription const& bd = prod.second;
        if (bd.isAlias() && bd.branchType() == bt) {
          addProductOrThrow(makeAliasedProduct(std::make_shared<ProductDescription const>(bd), iReg, productResolvers),
                            productResolvers,
                            iReg);
        }
      }
    }

    return productResolvers;
  }
}  // namespace edm::productResolversFactory
