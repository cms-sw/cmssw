#ifndef FWCore_Framework_Principal_h
#define FWCore_Framework_Principal_h

/*----------------------------------------------------------------------

Principal: This is the implementation of the classes responsible
for management of EDProducts. It is not seen by reconstruction code.

The major internal component of the Principal is the Group, which
contains an EDProduct and its associated Provenance, along with
ancillary transient information regarding the two. Groups are handled
through shared pointers.

The Principal returns BasicHandle, rather than a shared
pointer to a Group, when queried.

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
#include "FWCore/Framework/interface/Group.h"
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

   struct FilledGroupPtr {
      bool operator()(boost::shared_ptr<Group> const& iObj) { return bool(iObj);}
   };

  class Principal : public EDProductGetter {
  public:
    typedef std::vector<boost::shared_ptr<Group> > GroupCollection;
    typedef boost::filter_iterator<FilledGroupPtr, GroupCollection::const_iterator> const_iterator;
    typedef ProcessHistory::const_iterator ProcessNameConstIterator;
    typedef Group const* ConstGroupPtr;
    typedef std::vector<BasicHandle> BasicHandleVec;
    typedef GroupCollection::size_type      size_type;

    typedef boost::shared_ptr<Group> SharedGroupPtr;
    typedef std::string ProcessName;

    Principal(boost::shared_ptr<ProductRegistry const> reg,
              ProcessConfiguration const& pc,
              BranchType bt,
              HistoryAppender* historyAppender);

    virtual ~Principal();

    bool adjustToNewProductRegistry(ProductRegistry const& reg);

    void adjustIndexesAfterProductRegistryAddition();

    void addGroupScheduled(boost::shared_ptr<ConstBranchDescription> bd);

    void addGroupSource(boost::shared_ptr<ConstBranchDescription> bd);

    void addGroupInput(boost::shared_ptr<ConstBranchDescription> bd);

    void addOnDemandGroup(boost::shared_ptr<ConstBranchDescription> bd);

    void addGroupAliased(boost::shared_ptr<ConstBranchDescription> bd);

    void fillPrincipal(ProcessHistoryID const& hist, DelayedReader* reader);

    void clearPrincipal();

    void deleteProduct(BranchID const& id);
    
    EDProductGetter const* prodGetter() const {return this;}

    OutputHandle getForOutput(BranchID const& bid, bool getProd) const;

    BasicHandle  getBySelector(TypeID const& tid,
                               SelectorBase const& s) const;

    BasicHandle  getByLabel(TypeID const& tid,
                            std::string const& label,
                            std::string const& productInstanceName,
                            std::string const& processName,
                            size_t& cachedOffset,
                            int& fillCount) const;

    void getMany(TypeID const& tid,
                 SelectorBase const&,
                 BasicHandleVec& results) const;

    BasicHandle  getByType(TypeID const& tid) const;

    void getManyByType(TypeID const& tid,
                 BasicHandleVec& results) const;

    // Return a BasicHandle to the product which:
    //   1. is a sequence,
    //   2. and has the nested type 'value_type'
    //   3. and for which typeID is the same as or a public base of
    //      this value_type,
    //   4. and which matches the given selector
    size_t getMatchingSequence(TypeID const& typeID,
                               SelectorBase const& selector,
                               BasicHandle& result) const;

    ProcessHistory const& processHistory() const {
      return *processHistoryPtr_;
    }

    ProcessHistoryID const& processHistoryID() const {
      return processHistoryID_;
    }

    ProcessConfiguration const& processConfiguration() const {return *processConfiguration_;}

    ProductRegistry const& productRegistry() const {return *preg_;}

    // merge Principals containing different groups.
    void recombine(Principal& other, std::vector<BranchID> const& bids);

    size_t size() const;

    // These iterators skip over any null shared pointers
    const_iterator begin() const {return boost::make_filter_iterator<FilledGroupPtr>(groups_.begin(), groups_.end());}
    const_iterator end() const {return  boost::make_filter_iterator<FilledGroupPtr>(groups_.end(), groups_.end());}

    Provenance getProvenance(BranchID const& bid) const;

    void getAllProvenance(std::vector<Provenance const*>& provenances) const;

    BranchType const& branchType() const {return branchType_;}

    DelayedReader* reader() const {return reader_;}

    void maybeFlushCache(TypeID const& tid, InputTag const& tag) const;

    ConstGroupPtr getGroup(BranchID const& oid,
                           bool resolveProd,
                           bool fillOnDemand) const;

    ProductData const* findGroupByTag(TypeID const& typeID, InputTag const& tag) const;

  protected:

    // ----- Add a new Group
    // *this takes ownership of the Group, which in turn owns its
    // data.
    void addGroup_(std::auto_ptr<Group> g);
    void addGroupOrThrow(std::auto_ptr<Group> g);
    Group* getExistingGroup(BranchID const& branchID);
    Group* getExistingGroup(Group const& g);

    ConstGroupPtr getGroupByIndex(ProductTransientIndex const& oid,
                                  bool resolveProd,
                                  bool fillOnDemand) const;

    // Make my DelayedReader get the EDProduct for a Group or
    // trigger unscheduled execution if required.  The Group is
    // a cache, and so can be modified through the const reference.
    // We do not change the *number* of groups through this call, and so
    // *this is const.
    void resolveProduct(Group const& g, bool fillOnDemand) const {resolveProduct_(g, fillOnDemand);}

    // throws if the pointed to product is already in the Principal.
    void checkUniquenessAndType(WrapperOwningHolder const& prod, Group const* group) const;

    void putOrMerge(WrapperOwningHolder const& prod, Group const* group) const;

    void putOrMerge(WrapperOwningHolder const& prod, ProductProvenance& prov, Group* group);

  private:
    virtual WrapperHolder getIt(ProductID const&) const;

    virtual bool unscheduledFill(std::string const& moduleLabel) const = 0;

    // Used for indices to find groups by type and process
    typedef TransientProductLookupMap TypeLookup;

    size_t findGroup(TypeID const& typeID,
                     TypeLookup const& typeLookup,
                     SelectorBase const& selector,
                     BasicHandle& result) const;

    ProductData const* findGroupByLabel(TypeID const& typeID,
                                        TypeLookup const& typeLookup,
                                        std::string const& moduleLabel,
                                        std::string const& productInstanceName,
                                        std::string const& processName,
                                        size_t& cachedOffset,
                                        int& fillCount) const;

    size_t findGroups(TypeID const& typeID,
                      TypeLookup const& typeLookup,
                      SelectorBase const& selector,
                      BasicHandleVec& results) const;

    // defaults to no-op unless overridden in derived class.
    virtual void resolveProduct_(Group const&, bool /*fillOnDemand*/) const {}

    ProcessHistory const* processHistoryPtr_;

    ProcessHistoryID processHistoryID_;

    ProcessConfiguration const* processConfiguration_;

    // A vector of groups.
    GroupCollection groups_; // products and provenances are persistent

    // Pointer to the product registry. There is one entry in the registry
    // for each EDProduct in the event.
    boost::shared_ptr<ProductRegistry const> preg_;

    // Pointer to the 'source' that will be used to obtain EDProducts
    // from the persistent store. This 'source' is owned by the input source.
    DelayedReader* reader_;

    // Used to check for duplicates.  The same product instance must not be in more than one group.
    mutable std::set<void const*> productPtrs_;

    BranchType branchType_;

    // In use cases where the new process should not be appended to
    // input ProcessHistory, the following pointer should be null.
    // The Principal does not own this object.
    HistoryAppender* historyAppender_;

    static ProcessHistory emptyProcessHistory_;
  };

  template <typename PROD>
  inline
  boost::shared_ptr<Wrapper<PROD> const>
  getProductByTag(Principal const& ep, InputTag const& tag) {
    TypeID tid = TypeID(typeid(PROD));
    ep.maybeFlushCache(tid, tag);
    ProductData const* result = ep.findGroupByTag(tid, tag);

    if(result->getInterface() &&
       (!(result->getInterface()->dynamicTypeInfo() == typeid(PROD)))) {
      handleimpl::throwConvertTypeError(typeid(PROD), result->getInterface()->dynamicTypeInfo());
    }
    return boost::static_pointer_cast<Wrapper<PROD> const>(result->wrapper_);
  }
}
#endif
