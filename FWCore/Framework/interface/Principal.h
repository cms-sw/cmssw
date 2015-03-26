#ifndef FWCore_Framework_Principal_h
#define FWCore_Framework_Principal_h

/*----------------------------------------------------------------------

Principal: This is the implementation of the classes responsible
for management of EDProducts. It is not seen by reconstruction code.

The major internal component of the Principal is the ProductHolder, which
contains an EDProduct and its associated Provenance, along with
ancillary transient information regarding the two. ProductHolders are handled
through shared pointers.

The Principal returns BasicHandle, rather than a shared
pointer to a ProductHolder, when queried.

(Historical note: prior to April 2007 this class was named DataBlockImpl)

----------------------------------------------------------------------*/
#include "DataFormats/Common/interface/BasicHandle.h"
#include "DataFormats/Common/interface/ConvertHandle.h"
#include "DataFormats/Common/interface/WrapperBase.h"
#include "DataFormats/Common/interface/EDProductGetter.h"
#include "DataFormats/Common/interface/OutputHandle.h"
#include "DataFormats/Common/interface/Wrapper.h"
#include "DataFormats/Provenance/interface/ProcessHistory.h"
#include "DataFormats/Provenance/interface/ProvenanceFwd.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/ProductHolder.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Utilities/interface/ProductKindOfType.h"

#include "boost/iterator/filter_iterator.hpp"

#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>

namespace edm {

  class HistoryAppender;
  class ModuleCallingContext;
  class ProcessHistoryRegistry;
  class ProductHolderIndexHelper;
  class EDConsumerBase;

  struct FilledProductPtr {
    bool operator()(std::shared_ptr<ProductHolderBase> const& iObj) { return bool(iObj);}
  };

  class Principal : public EDProductGetter {
  public:
    typedef std::vector<std::shared_ptr<ProductHolderBase> > ProductHolderCollection;
    typedef boost::filter_iterator<FilledProductPtr, ProductHolderCollection::const_iterator> const_iterator;
    typedef ProcessHistory::const_iterator ProcessNameConstIterator;
    typedef ProductHolderBase const* ConstProductHolderPtr;
    typedef std::vector<BasicHandle> BasicHandleVec;
    typedef ProductHolderCollection::size_type size_type;

    typedef std::shared_ptr<ProductHolderBase> SharedProductPtr;
    typedef std::string ProcessName;

    Principal(std::shared_ptr<ProductRegistry const> reg,
              std::shared_ptr<ProductHolderIndexHelper const> productLookup,
              ProcessConfiguration const& pc,
              BranchType bt,
              HistoryAppender* historyAppender);

    virtual ~Principal();

    bool adjustToNewProductRegistry(ProductRegistry const& reg);

    void adjustIndexesAfterProductRegistryAddition();

    void addScheduledProduct(std::shared_ptr<BranchDescription const> bd);

    void addSourceProduct(std::shared_ptr<BranchDescription const> bd);

    void addInputProduct(std::shared_ptr<BranchDescription const> bd);

    void addUnscheduledProduct(std::shared_ptr<BranchDescription const> bd);

    void addAliasedProduct(std::shared_ptr<BranchDescription const> bd);

    void fillPrincipal(ProcessHistoryID const& hist, ProcessHistoryRegistry const& phr, DelayedReader* reader);

    void clearPrincipal();

    void deleteProduct(BranchID const& id);
    
    EDProductGetter const* prodGetter() const {return this;}

    OutputHandle getForOutput(BranchID const& bid, bool getProd, ModuleCallingContext const* mcc) const;

    // Return a BasicHandle to the product which:
    //   1. matches the given label, instance, and process
    //   (if process if empty gets the match from the most recent process)
    //   2. If kindOfType is PRODUCT, then the type of the product matches typeID
    //   3. If kindOfType is ELEMENT
    //      a.  the product is a sequence,
    //      b.  the sequence has the nested type 'value_type'
    //      c.  typeID is the same as or a public base of
    //      this value_type,

    BasicHandle  getByLabel(KindOfType kindOfType,
                            TypeID const& typeID,
                            InputTag const& inputTag,
                            EDConsumerBase const* consumes,
                            ModuleCallingContext const* mcc) const;

    BasicHandle  getByLabel(KindOfType kindOfType,
                            TypeID const& typeID,
                            std::string const& label,
                            std::string const& instance,
                            std::string const& process,
                            EDConsumerBase const* consumes,
                            ModuleCallingContext const* mcc) const;
    
    BasicHandle getByToken(KindOfType kindOfType,
                           TypeID const& typeID,
                           ProductHolderIndex index,
                           bool skipCurrentProcess,
                           bool& ambiguous,
                           ModuleCallingContext const* mcc) const;

    void prefetch(ProductHolderIndex index,
                  bool skipCurrentProcess,
                  ModuleCallingContext const* mcc) const;

    void getManyByType(TypeID const& typeID,
                       BasicHandleVec& results,
                       EDConsumerBase const* consumes,
                       ModuleCallingContext const* mcc) const;

    ProcessHistory const& processHistory() const {
      return *processHistoryPtr_;
    }

    ProcessHistoryID const& processHistoryID() const {
      return processHistoryID_;
    }

    ProcessConfiguration const& processConfiguration() const {return *processConfiguration_;}

    ProductRegistry const& productRegistry() const {return *preg_;}

    ProductHolderIndexHelper const& productLookup() const {return *productLookup_;}

    // merge Principals containing different products.
    void recombine(Principal& other, std::vector<BranchID> const& bids);

    size_t size() const;

    // These iterators skip over any null shared pointers
    const_iterator begin() const {return boost::make_filter_iterator<FilledProductPtr>(productHolders_.begin(), productHolders_.end());}
    const_iterator end() const {return  boost::make_filter_iterator<FilledProductPtr>(productHolders_.end(), productHolders_.end());}

    Provenance getProvenance(BranchID const& bid,
                             ModuleCallingContext const* mcc) const;

    void getAllProvenance(std::vector<Provenance const*>& provenances) const;

    BranchType const& branchType() const {return branchType_;}
    
    //This will never return 0 so you can use 0 to mean unset
    typedef unsigned long CacheIdentifier_t;
    CacheIdentifier_t cacheIdentifier() const {return cacheIdentifier_;}

    DelayedReader* reader() const {return reader_;}

    ConstProductHolderPtr getProductHolder(BranchID const& oid) const;

    ProductData const* findProductByTag(TypeID const& typeID, InputTag const& tag, ModuleCallingContext const* mcc) const;

    // Make my DelayedReader get the EDProduct for a ProductHolder.
    // The ProductHolder is a cache, and so can be modified through the const
    // reference.
    // We do not change the *number* of products through this call, and so
    // *this is const.
    void readFromSource(ProductHolderBase const& phb, ModuleCallingContext const* mcc) const {
      readFromSource_(phb, mcc);
    }

    virtual bool unscheduledFill(std::string const& moduleLabel,
                                 ModuleCallingContext const* mcc) const = 0;

    std::vector<unsigned int> const& lookupProcessOrder() const { return lookupProcessOrder_; }

    ConstProductHolderPtr getProductHolderByIndex(ProductHolderIndex const& oid) const;

    bool isComplete() const {return isComplete_();}

  protected:

    // ----- Add a new ProductHolder
    // *this takes ownership of the ProductHolder, which in turn owns its
    // data.
    void addProduct_(std::auto_ptr<ProductHolderBase> phb);
    void addProductOrThrow(std::auto_ptr<ProductHolderBase> phb);
    ProductHolderBase* getExistingProduct(BranchID const& branchID);
    ProductHolderBase* getExistingProduct(ProductHolderBase const& phb);

    // throws if the pointed to product is already in the Principal.
    void checkUniquenessAndType(WrapperBase const* prod, ProductHolderBase const* productHolder) const;

    void putOrMerge(std::unique_ptr<WrapperBase> prod, ProductHolderBase const* productHolder) const;

    void putOrMerge(std::unique_ptr<WrapperBase> prod, ProductProvenance& prov, ProductHolderBase* productHolder);

  private:

    virtual WrapperBase const* getIt(ProductID const&) const override;
    virtual WrapperBase const* getThinnedProduct(ProductID const&, unsigned int&) const override;
    virtual void getThinnedProducts(ProductID const&,
                                    std::vector<WrapperBase const*>&,
                                    std::vector<unsigned int>&) const override;

    void findProducts(std::vector<ProductHolderBase const*> const& holders,
                      TypeID const& typeID,
                      BasicHandleVec& results,
                      ModuleCallingContext const* mcc) const;

    ProductData const* findProductByLabel(KindOfType kindOfType,
                                          TypeID const& typeID,
                                          InputTag const& inputTag,
                                          EDConsumerBase const* consumer,
                                          ModuleCallingContext const* mcc) const;

    ProductData const* findProductByLabel(KindOfType kindOfType,
                                          TypeID const& typeID,
                                          std::string const& label,
                                          std::string const& instance,
                                          std::string const& process,
                                          EDConsumerBase const* consumer,
                                          ModuleCallingContext const* mcc) const;

    virtual void readFromSource_(ProductHolderBase const& /* phb */, ModuleCallingContext const* /* mcc */) const {}

    virtual bool isComplete_() const {return true;}

    std::shared_ptr<ProcessHistory const> processHistoryPtr_;

    ProcessHistoryID processHistoryID_;

    ProcessConfiguration const* processConfiguration_;

    // A vector of product holders.
    ProductHolderCollection productHolders_; // products and provenances are persistent

    // Pointer to the product registry. There is one entry in the registry
    // for each EDProduct in the event.
    std::shared_ptr<ProductRegistry const> preg_;
    std::shared_ptr<ProductHolderIndexHelper const> productLookup_;

    std::vector<unsigned int> lookupProcessOrder_;
    ProcessHistoryID orderProcessHistoryID_;

    // Pointer to the 'source' that will be used to obtain EDProducts
    // from the persistent store. This 'source' is owned by the input source.
    DelayedReader* reader_;

    // Used to check for duplicates.  The same product instance must not be in more than one product holder
    mutable std::set<void const*> productPtrs_;

    BranchType branchType_;

    // In use cases where the new process should not be appended to
    // input ProcessHistory, the following pointer should be null.
    // The Principal does not own this object.
    HistoryAppender* historyAppender_;
    
    CacheIdentifier_t cacheIdentifier_;

  };

  template <typename PROD>
  inline
  std::shared_ptr<Wrapper<PROD> const>
    getProductByTag(Principal const& ep, InputTag const& tag, ModuleCallingContext const* mcc) {
    TypeID tid = TypeID(typeid(PROD));
    ProductData const* result = ep.findProductByTag(tid, tag, mcc);
    if(result == nullptr) {
      return std::shared_ptr<Wrapper<PROD> const>(); 
    }

    if(!(result->wrapper_->dynamicTypeInfo() == typeid(PROD))) {
      handleimpl::throwConvertTypeError(typeid(PROD), result->wrapper_->dynamicTypeInfo());
    }
    return std::static_pointer_cast<Wrapper<PROD> const>(result->wrapper_);
  }
}
#endif
