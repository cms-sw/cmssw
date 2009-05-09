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
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "boost/shared_ptr.hpp"
#include "boost/iterator/filter_iterator.hpp"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "DataFormats/Provenance/interface/ProvenanceFwd.h"
#include "DataFormats/Provenance/interface/BranchID.h"
#include "DataFormats/Provenance/interface/ProductTransientIndex.h"
#include "DataFormats/Provenance/interface/BranchMapper.h"
#include "DataFormats/Provenance/interface/ProductProvenance.h"
#include "DataFormats/Common/interface/EDProductGetter.h"
#include "DataFormats/Common/interface/BasicHandle.h"
#include "DataFormats/Common/interface/OutputHandle.h"
#include "DataFormats/Common/interface/Wrapper.h"
#include "DataFormats/Provenance/interface/ProcessHistory.h"
#include "DataFormats/Provenance/interface/ProductStatus.h"
#include "FWCore/Framework/interface/Group.h"
#include "FWCore/Framework/interface/NoDelayedReader.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Utilities/interface/TypeID.h"
#include "DataFormats/Provenance/interface/TransientProductLookupMap.h"

namespace edm {
   struct FilledGroupPtr {
      bool operator()(boost::shared_ptr<Group> const& iObj) { return bool(iObj);}
   };
   
  class Principal : public EDProductGetter {
  public:
    typedef std::vector<boost::shared_ptr<Group> > GroupCollection;
     typedef boost::filter_iterator<FilledGroupPtr, GroupCollection::const_iterator> const_iterator;
    typedef ProcessHistory::const_iterator ProcessNameConstIterator;
    typedef boost::shared_ptr<Group const> SharedConstGroupPtr;
    typedef std::vector<BasicHandle> BasicHandleVec;
    typedef GroupCollection::size_type      size_type;

    typedef boost::shared_ptr<Group> SharedGroupPtr;
    typedef std::string ProcessName;

    Principal(boost::shared_ptr<ProductRegistry const> reg,
	      ProcessConfiguration const& pc,
              BranchType bt,
	      ProcessHistoryID const& hist = ProcessHistoryID(),
              boost::shared_ptr<BranchMapper> mapper = boost::shared_ptr<BranchMapper>(new BranchMapper),
              boost::shared_ptr<DelayedReader> rtrv = boost::shared_ptr<DelayedReader>(new NoDelayedReader));

    virtual ~Principal();

    EDProductGetter const* prodGetter() const {return this;}

    OutputHandle getForOutput(BranchID const& bid, bool getProd) const;

    BasicHandle  getBySelector(TypeID const& tid,
                               SelectorBase const& s) const;

    BasicHandle  getByLabel(TypeID const& tid,
			    std::string const& label,
			    std::string const& productInstanceName,
			    std::string const& processName) const;

    void getMany(TypeID const& tid, 
		 SelectorBase const&,
		 BasicHandleVec& results) const;

    BasicHandle  getByType(TypeID const& tid) const;

    void getManyByType(TypeID const& tid, 
		 BasicHandleVec& results) const;

    // Return a vector of BasicHandles to the products which:
    //   1. are sequences,
    //   2. and have the nested type 'value_type'
    //   3. and for which typeID is the same as or a public base of
    //      this value_type,
    //   4. and which matches the given selector
    size_t getMatchingSequence(TypeID const& typeID,
			       SelectorBase const& selector,
			       BasicHandleVec& results,
			       bool stopIfProcessHasMatch) const;

    void
    readImmediate() const;

    void
    readProvenanceImmediate() const;

    ProcessHistory const& processHistory() const;    

    ProcessConfiguration const& processConfiguration() const {return *processConfiguration_;}

    ProductRegistry const& productRegistry() const {return *preg_;}

    boost::shared_ptr<DelayedReader> store() const {return store_;}

    boost::shared_ptr<BranchMapper> branchMapperPtr() const {return branchMapperPtr_;}

    // ----- Mark this Principal as having been updated in the
    // current Process.
    void addToProcessHistory() const;

    // merge Principals containing different groups.
    void recombine(Principal & other, std::vector<BranchID> const& bids);

    size_t  size() const { return size_; }

    const_iterator begin() const {return boost::make_filter_iterator<FilledGroupPtr>(groups_.begin(), groups_.end());}
    const_iterator end() const {return  boost::make_filter_iterator<FilledGroupPtr>(groups_.end(), groups_.end());}

    Provenance
    getProvenance(BranchID const& bid) const;

    void
    getAllProvenance(std::vector<Provenance const*> & provenances) const;

  protected:
    // ----- Add a new Group
    // *this takes ownership of the Group, which in turn owns its
    // data.
    void addGroup_(std::auto_ptr<Group> g);
    Group* getExistingGroup(Group const& g);
    void replaceGroup(std::auto_ptr<Group> g);

    //deprecated
    SharedConstGroupPtr const getGroup(BranchID const& oid,
                                       bool resolveProd,
                                       bool resolveProv,
				       bool fillOnDemand) const;
    SharedConstGroupPtr const getGroupByIndex(ProductTransientIndex const& oid,
                                        bool resolveProd,
                                        bool resolveProv,
                                        bool fillOnDemand) const;
     
    void resolveProvenance(Group const& g) const;

    void swapBase(Principal&);
  private:
    virtual EDProduct const* getIt(ProductID const&) const;

    virtual void addOrReplaceGroup(std::auto_ptr<Group> g) = 0;


    virtual ProcessHistoryID const& processHistoryID() const = 0;

    virtual void setProcessHistoryID(ProcessHistoryID const& phid) const = 0;

    virtual bool unscheduledFill(std::string const& moduleLabel) const = 0;

    // Used for indices to find groups by type and process
    typedef TransientProductLookupMap TypeLookup;

    size_t findGroups(TypeID const& typeID,
		      TypeLookup const& typeLookup,
		      SelectorBase const& selector,
		      BasicHandleVec& results,
		      bool stopIfProcessHasMatch) const;

    // Make my DelayedReader get the EDProduct for a Group or
    // trigger unscheduled execution if required.  The Group is
    // a cache, and so can be modified through the const reference.
    // We do not change the *number* of groups through this call, and so
    // *this is const.
    void resolveProduct(Group const& g, bool fillOnDemand) const;

    boost::shared_ptr<ProcessHistory> processHistoryPtr_;

    ProcessConfiguration const* processConfiguration_;

    mutable bool processHistoryModified_;

    // A vector of groups.
    GroupCollection groups_; // products and provenances are persistent
    //how many non-null group pointers are in groups_
    size_t size_;

    // Pointer to the product registry. There is one entry in the registry
    // for each EDProduct in the event.
    boost::shared_ptr<ProductRegistry const> preg_;

    // Pointer to the 'mapper' that will get provenance information
    // from the persistent store.
    boost::shared_ptr<BranchMapper> branchMapperPtr_;

    // Pointer to the 'source' that will be used to obtain EDProducts
    // from the persistent store.
    boost::shared_ptr<DelayedReader> store_;
     
    BranchType branchType_;
  };

  template <typename PROD>
  inline
  boost::shared_ptr<Wrapper<PROD> const> 	 
  getProductByTag(Principal const& ep, InputTag const& tag) {
    return boost::dynamic_pointer_cast<Wrapper<PROD> const>(ep.getByLabel(TypeID(typeid(PROD)), tag.label(), tag.instance(), tag.process()).product());
  }
}
#endif
