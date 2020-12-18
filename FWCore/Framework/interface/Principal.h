#ifndef FWCore_Framework_Principal_h
#define FWCore_Framework_Principal_h

/*----------------------------------------------------------------------

Principal: This is the implementation of the classes responsible
for management of EDProducts. It is not seen by reconstruction code.

The major internal component of the Principal is the ProductResolver, which
contains an EDProduct and its associated Provenance, along with
ancillary transient information regarding the two. ProductResolvers are handled
through shared pointers.

The Principal returns BasicHandle, rather than a shared
pointer to a ProductResolver, when queried.

(Historical note: prior to April 2007 this class was named DataBlockImpl)

----------------------------------------------------------------------*/
#include "DataFormats/Common/interface/BasicHandle.h"
#include "DataFormats/Common/interface/ConvertHandle.h"
#include "DataFormats/Common/interface/WrapperBase.h"
#include "DataFormats/Common/interface/EDProductGetter.h"
#include "DataFormats/Common/interface/Wrapper.h"
#include "DataFormats/Provenance/interface/ProcessHistory.h"
#include "DataFormats/Provenance/interface/ProvenanceFwd.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/ProductResolverBase.h"
#include "FWCore/Concurrency/interface/WaitingTaskHolder.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Utilities/interface/ProductKindOfType.h"
#include "FWCore/Utilities/interface/propagate_const.h"

#include "boost/iterator/filter_iterator.hpp"

#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>

namespace edm {

  class HistoryAppender;
  class MergeableRunProductMetadata;
  class ModuleCallingContext;
  class ProductResolverIndexHelper;
  class EDConsumerBase;
  class SharedResourcesAcquirer;
  class InputProductResolver;
  class UnscheduledConfigurator;

  struct FilledProductPtr {
    bool operator()(propagate_const<std::shared_ptr<ProductResolverBase>> const& iObj) { return bool(iObj); }
  };

  class Principal : public EDProductGetter {
  public:
    typedef std::vector<propagate_const<std::shared_ptr<ProductResolverBase>>> ProductResolverCollection;
    typedef boost::filter_iterator<FilledProductPtr, ProductResolverCollection::const_iterator> const_iterator;
    typedef boost::filter_iterator<FilledProductPtr, ProductResolverCollection::iterator> iterator;
    typedef ProcessHistory::const_iterator ProcessNameConstIterator;
    typedef ProductResolverBase const* ConstProductResolverPtr;
    typedef std::vector<BasicHandle> BasicHandleVec;
    typedef ProductResolverCollection::size_type size_type;

    typedef std::shared_ptr<ProductResolverBase> SharedProductPtr;
    typedef std::string ProcessName;

    Principal(std::shared_ptr<ProductRegistry const> reg,
              std::shared_ptr<ProductResolverIndexHelper const> productLookup,
              ProcessConfiguration const& pc,
              BranchType bt,
              HistoryAppender* historyAppender,
              bool isForPrimaryProcess = true);

    ~Principal() override;

    bool adjustToNewProductRegistry(ProductRegistry const& reg);

    void adjustIndexesAfterProductRegistryAddition();

    void fillPrincipal(DelayedReader* reader);
    void fillPrincipal(ProcessHistoryID const& hist, ProcessHistory const* phr, DelayedReader* reader);
    void fillPrincipal(std::string const& processNameOfBlock, DelayedReader* reader);

    void clearPrincipal();

    void setupUnscheduled(UnscheduledConfigurator const&);

    void deleteProduct(BranchID const& id) const;

    EDProductGetter const* prodGetter() const { return this; }

    // Return a BasicHandle to the product which:
    //   1. matches the given label, instance, and process
    //   (if process if empty gets the match from the most recent process)
    //   2. If kindOfType is PRODUCT, then the type of the product matches typeID
    //   3. If kindOfType is ELEMENT
    //      a.  the product is a sequence,
    //      b.  the sequence has the nested type 'value_type'
    //      c.  typeID is the same as or a public base of
    //      this value_type,

    BasicHandle getByLabel(KindOfType kindOfType,
                           TypeID const& typeID,
                           InputTag const& inputTag,
                           EDConsumerBase const* consumes,
                           SharedResourcesAcquirer* sra,
                           ModuleCallingContext const* mcc) const;

    BasicHandle getByLabel(KindOfType kindOfType,
                           TypeID const& typeID,
                           std::string const& label,
                           std::string const& instance,
                           std::string const& process,
                           EDConsumerBase const* consumes,
                           SharedResourcesAcquirer* sra,
                           ModuleCallingContext const* mcc) const;

    BasicHandle getByToken(KindOfType kindOfType,
                           TypeID const& typeID,
                           ProductResolverIndex index,
                           bool skipCurrentProcess,
                           bool& ambiguous,
                           SharedResourcesAcquirer* sra,
                           ModuleCallingContext const* mcc) const;

    void prefetchAsync(WaitingTaskHolder waitTask,
                       ProductResolverIndex index,
                       bool skipCurrentProcess,
                       ServiceToken const& token,
                       ModuleCallingContext const* mcc) const;

    void getManyByType(TypeID const& typeID,
                       BasicHandleVec& results,
                       EDConsumerBase const* consumes,
                       SharedResourcesAcquirer* sra,
                       ModuleCallingContext const* mcc) const;

    ProcessHistory const& processHistory() const { return *processHistoryPtr_; }

    ProcessHistoryID const& processHistoryID() const { return processHistoryID_; }

    ProcessConfiguration const& processConfiguration() const { return *processConfiguration_; }

    ProductRegistry const& productRegistry() const { return *preg_; }

    ProductResolverIndexHelper const& productLookup() const { return *productLookup_; }

    // merge Principals containing different products.
    void recombine(Principal& other, std::vector<BranchID> const& bids);

    ProductResolverBase* getModifiableProductResolver(BranchID const& oid) {
      return const_cast<ProductResolverBase*>(const_cast<const Principal*>(this)->getProductResolver(oid));
    }

    size_t size() const;

    // These iterators skip over any null shared pointers
    const_iterator begin() const {
      return boost::make_filter_iterator<FilledProductPtr>(productResolvers_.begin(), productResolvers_.end());
    }
    const_iterator end() const {
      return boost::make_filter_iterator<FilledProductPtr>(productResolvers_.end(), productResolvers_.end());
    }

    iterator begin() {
      return boost::make_filter_iterator<FilledProductPtr>(productResolvers_.begin(), productResolvers_.end());
    }
    iterator end() {
      return boost::make_filter_iterator<FilledProductPtr>(productResolvers_.end(), productResolvers_.end());
    }

    Provenance getProvenance(BranchID const& bid, ModuleCallingContext const* mcc) const;

    void getAllProvenance(std::vector<Provenance const*>& provenances) const;

    void getAllStableProvenance(std::vector<StableProvenance const*>& provenances) const;

    BranchType const& branchType() const { return branchType_; }

    //This will never return 0 so you can use 0 to mean unset
    typedef unsigned long CacheIdentifier_t;
    CacheIdentifier_t cacheIdentifier() const { return cacheIdentifier_; }

    DelayedReader* reader() const { return reader_; }

    ConstProductResolverPtr getProductResolver(BranchID const& oid) const;

    ProductData const* findProductByTag(TypeID const& typeID,
                                        InputTag const& tag,
                                        ModuleCallingContext const* mcc) const;

    void readAllFromSourceAndMergeImmediately(MergeableRunProductMetadata const* mergeableRunProductMetadata = nullptr);

    std::vector<unsigned int> const& lookupProcessOrder() const { return lookupProcessOrder_; }

    ConstProductResolverPtr getProductResolverByIndex(ProductResolverIndex const& oid) const;

  protected:
    // ----- Add a new ProductResolver
    // *this takes ownership of the ProductResolver, which in turn owns its
    // data.
    void addProduct_(std::unique_ptr<ProductResolverBase> phb);
    void addProductOrThrow(std::unique_ptr<ProductResolverBase> phb);
    ProductResolverBase* getExistingProduct(BranchID const& branchID);
    ProductResolverBase const* getExistingProduct(BranchID const& branchID) const;
    ProductResolverBase const* getExistingProduct(ProductResolverBase const& phb) const;

    void putOrMerge(BranchDescription const& bd, std::unique_ptr<WrapperBase> edp) const;

    //F must take an argument of type ProductResolverBase*
    template <typename F>
    void applyToResolvers(F iFunc) {
      for (auto& resolver : productResolvers_) {
        iFunc(resolver.get());
      }
    }

  private:
    //called by adjustIndexesAfterProductRegistryAddition only if an index actually changed
    virtual void changedIndexes_() {}

    void addScheduledProduct(std::shared_ptr<BranchDescription const> bd);
    void addSourceProduct(std::shared_ptr<BranchDescription const> bd);
    void addInputProduct(std::shared_ptr<BranchDescription const> bd);
    void addUnscheduledProduct(std::shared_ptr<BranchDescription const> bd);
    void addAliasedProduct(std::shared_ptr<BranchDescription const> bd);
    void addSwitchProducerProduct(std::shared_ptr<BranchDescription const> bd);
    void addSwitchAliasProduct(std::shared_ptr<BranchDescription const> bd);
    void addParentProcessProduct(std::shared_ptr<BranchDescription const> bd);

    WrapperBase const* getIt(ProductID const&) const override;
    std::optional<std::tuple<WrapperBase const*, unsigned int>> getThinnedProduct(ProductID const&,
                                                                                  unsigned int) const override;
    void getThinnedProducts(ProductID const&,
                            std::vector<WrapperBase const*>&,
                            std::vector<unsigned int>&) const override;
    OptionalThinnedKey getThinnedKeyFrom(ProductID const& parent,
                                         unsigned int key,
                                         ProductID const& thinned) const override;

    void findProducts(std::vector<ProductResolverBase const*> const& holders,
                      TypeID const& typeID,
                      BasicHandleVec& results,
                      SharedResourcesAcquirer* sra,
                      ModuleCallingContext const* mcc) const;

    ProductData const* findProductByLabel(KindOfType kindOfType,
                                          TypeID const& typeID,
                                          InputTag const& inputTag,
                                          EDConsumerBase const* consumer,
                                          SharedResourcesAcquirer* sra,
                                          ModuleCallingContext const* mcc) const;

    ProductData const* findProductByLabel(KindOfType kindOfType,
                                          TypeID const& typeID,
                                          std::string const& label,
                                          std::string const& instance,
                                          std::string const& process,
                                          EDConsumerBase const* consumer,
                                          SharedResourcesAcquirer* sra,
                                          ModuleCallingContext const* mcc) const;

    void putOrMerge(std::unique_ptr<WrapperBase> prod, ProductResolverBase const* productResolver) const;

    std::shared_ptr<ProcessHistory const> processHistoryPtr_;

    ProcessHistoryID processHistoryID_;
    ProcessHistoryID processHistoryIDBeforeConfig_;

    ProcessConfiguration const* processConfiguration_;

    // A vector of product holders.
    ProductResolverCollection productResolvers_;  // products and provenances are persistent

    // Pointer to the product registry. There is one entry in the registry
    // for each EDProduct in the event.
    std::shared_ptr<ProductRegistry const> preg_;
    std::shared_ptr<ProductResolverIndexHelper const> productLookup_;

    std::vector<unsigned int> lookupProcessOrder_;
    ProcessHistoryID orderProcessHistoryID_;

    // Pointer to the 'source' that will be used to obtain EDProducts
    // from the persistent store. This 'source' is owned by the input source.
    DelayedReader* reader_;

    BranchType branchType_;

    // In use cases where the new process should not be appended to
    // input ProcessHistory, the following pointer should be null.
    // The Principal does not own this object.
    edm::propagate_const<HistoryAppender*> historyAppender_;

    CacheIdentifier_t cacheIdentifier_;
  };

  template <typename PROD>
  inline std::shared_ptr<Wrapper<PROD> const> getProductByTag(Principal const& ep,
                                                              InputTag const& tag,
                                                              ModuleCallingContext const* mcc) {
    TypeID tid = TypeID(typeid(PROD));
    ProductData const* result = ep.findProductByTag(tid, tag, mcc);
    if (result == nullptr) {
      return std::shared_ptr<Wrapper<PROD> const>();
    }

    if (!(result->wrapper()->dynamicTypeInfo() == typeid(PROD))) {
      handleimpl::throwConvertTypeError(typeid(PROD), result->wrapper()->dynamicTypeInfo());
    }
    return std::static_pointer_cast<Wrapper<PROD> const>(result->sharedConstWrapper());
  }
}  // namespace edm
#endif
