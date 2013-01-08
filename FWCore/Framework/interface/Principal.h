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
#include "DataFormats/Common/interface/EDProductGetter.h"
#include "DataFormats/Common/interface/OutputHandle.h"
#include "DataFormats/Common/interface/Wrapper.h"
#include "DataFormats/Common/interface/WrapperHolder.h"
#include "DataFormats/Common/interface/WrapperOwningHolder.h"
#include "DataFormats/Provenance/interface/ProcessHistory.h"
#include "DataFormats/Provenance/interface/ProductTransientIndex.h"
#include "DataFormats/Provenance/interface/ProvenanceFwd.h"
#include "DataFormats/Provenance/interface/TransientProductLookupMap.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/ProductHolder.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "boost/iterator/filter_iterator.hpp"
#include "boost/shared_ptr.hpp"

#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>

namespace edm {

   class HistoryAppender;

   struct FilledProductPtr {
      bool operator()(boost::shared_ptr<ProductHolderBase> const& iObj) { return bool(iObj);}
   };

  class Principal : public EDProductGetter {
  public:
    typedef std::vector<boost::shared_ptr<ProductHolderBase> > ProductHolderCollection;
    typedef boost::filter_iterator<FilledProductPtr, ProductHolderCollection::const_iterator> const_iterator;
    typedef ProcessHistory::const_iterator ProcessNameConstIterator;
    typedef ProductHolderBase const* ConstProductPtr;
    typedef std::vector<BasicHandle> BasicHandleVec;
    typedef ProductHolderCollection::size_type      size_type;

    typedef boost::shared_ptr<ProductHolderBase> SharedProductPtr;
    typedef std::string ProcessName;

    Principal(boost::shared_ptr<ProductRegistry const> reg,
              ProcessConfiguration const& pc,
              BranchType bt,
              HistoryAppender* historyAppender);

    virtual ~Principal();

    bool adjustToNewProductRegistry(ProductRegistry const& reg);

    void adjustIndexesAfterProductRegistryAddition();

    void addScheduledProduct(boost::shared_ptr<ConstBranchDescription> bd);

    void addSourceProduct(boost::shared_ptr<ConstBranchDescription> bd);

    void addInputProduct(boost::shared_ptr<ConstBranchDescription> bd);

    void addUnscheduledProduct(boost::shared_ptr<ConstBranchDescription> bd);

    void addAliasedProduct(boost::shared_ptr<ConstBranchDescription> bd);

    void fillPrincipal(ProcessHistoryID const& hist, DelayedReader* reader);

    void clearPrincipal();

    void deleteProduct(BranchID const& id);
    
    EDProductGetter const* prodGetter() const {return this;}

    OutputHandle getForOutput(BranchID const& bid, bool getProd) const;

    BasicHandle  getByLabel(TypeID const& tid,
                            std::string const& label,
                            std::string const& productInstanceName,
                            std::string const& processName,
                            size_t& cachedOffset,
                            int& fillCount) const;

    void getManyByType(TypeID const& tid,
                 BasicHandleVec& results) const;

    // Return a BasicHandle to the product which:
    //   1. is a sequence,
    //   2. and has the nested type 'value_type'
    //   3. and for which typeID is the same as or a public base of
    //      this value_type,
    //   4. and which matches the given label, instance, and process
    size_t getMatchingSequence(TypeID const& typeID,
                               std::string const& moduleLabel,
                               std::string const& productInstanceName,
                               std::string const& processName,
                               BasicHandle& result) const;

    ProcessHistory const& processHistory() const {
      return *processHistoryPtr_;
    }

    ProcessHistoryID const& processHistoryID() const {
      return processHistoryID_;
    }

    ProcessConfiguration const& processConfiguration() const {return *processConfiguration_;}

    ProductRegistry const& productRegistry() const {return *preg_;}

    // merge Principals containing different products.
    void recombine(Principal& other, std::vector<BranchID> const& bids);

    size_t size() const;

    // These iterators skip over any null shared pointers
    const_iterator begin() const {return boost::make_filter_iterator<FilledProductPtr>(productHolders_.begin(), productHolders_.end());}
    const_iterator end() const {return  boost::make_filter_iterator<FilledProductPtr>(productHolders_.end(), productHolders_.end());}

    Provenance getProvenance(BranchID const& bid) const;

    void getAllProvenance(std::vector<Provenance const*>& provenances) const;

    BranchType const& branchType() const {return branchType_;}

    DelayedReader* reader() const {return reader_;}

    void maybeFlushCache(TypeID const& tid, InputTag const& tag) const;

    ConstProductPtr getProductHolder(BranchID const& oid,
                           bool resolveProd,
                           bool fillOnDemand) const;

    ProductData const* findProductByTag(TypeID const& typeID, InputTag const& tag) const;

  protected:

    // ----- Add a new ProductHolder
    // *this takes ownership of the ProductHolder, which in turn owns its
    // data.
    void addProduct_(std::auto_ptr<ProductHolderBase> phb);
    void addProductOrThrow(std::auto_ptr<ProductHolderBase> phb);
    ProductHolderBase* getExistingProduct(BranchID const& branchID);
    ProductHolderBase* getExistingProduct(ProductHolderBase const& phb);

    ConstProductPtr getProductByIndex(ProductTransientIndex const& oid,
                                  bool resolveProd,
                                  bool fillOnDemand) const;

    // Make my DelayedReader get the EDProduct for a ProductHolder or
    // trigger unscheduled execution if required.  The ProductHolder is
    // a cache, and so can be modified through the const reference.
    // We do not change the *number* of products through this call, and so
    // *this is const.
    void resolveProduct(ProductHolderBase const& phb, bool fillOnDemand) const {resolveProduct_(phb, fillOnDemand);}

    // throws if the pointed to product is already in the Principal.
    void checkUniquenessAndType(WrapperOwningHolder const& prod, ProductHolderBase const* productHolder) const;

    void putOrMerge(WrapperOwningHolder const& prod, ProductHolderBase const* productHolder) const;

    void putOrMerge(WrapperOwningHolder const& prod, ProductProvenance& prov, ProductHolderBase* productHolder);

  private:
    virtual WrapperHolder getIt(ProductID const&) const;

    virtual bool unscheduledFill(std::string const& moduleLabel) const = 0;

    // Used for indices to find products by type and process
    typedef TransientProductLookupMap TypeLookup;

    size_t findProduct(TypeID const& typeID,
                     TypeLookup const& typeLookup,
                     std::string const& moduleLabel,
                     std::string const& productInstanceName,
                     std::string const& processName,
                     BasicHandle& result) const;

    ProductData const* findProductByLabel(TypeID const& typeID,
                                        TypeLookup const& typeLookup,
                                        std::string const& moduleLabel,
                                        std::string const& productInstanceName,
                                        std::string const& processName,
                                        size_t& cachedOffset,
                                        int& fillCount) const;

    size_t findProducts(TypeID const& typeID,
                      TypeLookup const& typeLookup,
                      BasicHandleVec& results) const;

    // defaults to no-op unless overridden in derived class.
    virtual void resolveProduct_(ProductHolderBase const&, bool /*fillOnDemand*/) const {}

    ProcessHistory const* processHistoryPtr_;

    ProcessHistoryID processHistoryID_;

    ProcessConfiguration const* processConfiguration_;

    // A vector of product holders.
    ProductHolderCollection productHolders_; // products and provenances are persistent

    // Pointer to the product registry. There is one entry in the registry
    // for each EDProduct in the event.
    boost::shared_ptr<ProductRegistry const> preg_;

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

    static const ProcessHistory emptyProcessHistory_;
  };

  template <typename PROD>
  inline
  boost::shared_ptr<Wrapper<PROD> const>
  getProductByTag(Principal const& ep, InputTag const& tag) {
    TypeID tid = TypeID(typeid(PROD));
    ep.maybeFlushCache(tid, tag);
    ProductData const* result = ep.findProductByTag(tid, tag);

    if(result->getInterface() &&
       (!(result->getInterface()->dynamicTypeInfo() == typeid(PROD)))) {
      handleimpl::throwConvertTypeError(typeid(PROD), result->getInterface()->dynamicTypeInfo());
    }
    return boost::static_pointer_cast<Wrapper<PROD> const>(result->wrapper_);
  }
}
#endif
