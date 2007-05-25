#ifndef Framework_Principal_h
#define Framework_Principal_h

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

$Id: Principal.h,v 1.4 2007/05/10 12:27:03 wmtan Exp $

----------------------------------------------------------------------*/
#include <list>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <typeinfo>
#include <vector>

#include "boost/shared_ptr.hpp"

#include "DataFormats/Provenance/interface/BranchKey.h"
#include "DataFormats/Provenance/interface/ProductID.h"
#include "DataFormats/Common/interface/EDProduct.h"
#include "DataFormats/Common/interface/EDProductGetter.h"
#include "DataFormats/Provenance/interface/ProcessHistory.h"
#include "DataFormats/Provenance/interface/ProcessHistoryID.h"
#include "DataFormats/Common/interface/BasicHandle.h"
#include "FWCore/Framework/interface/NoDelayedReader.h"
#include "FWCore/Framework/interface/DelayedReader.h"
#include "FWCore/Framework/interface/SelectorBase.h"

#include "FWCore/Framework/interface/Group.h"
#include "FWCore/Utilities/interface/TypeID.h"

namespace edm {
  class ProductRegistry;
  class Principal : public EDProductGetter {
  public:
    typedef std::vector<boost::shared_ptr<Group> > GroupVec;
    typedef GroupVec::const_iterator               const_iterator;
    typedef ProcessHistory::const_iterator         ProcessNameConstIterator;
    typedef boost::shared_ptr<const Group>         SharedConstGroupPtr;
    typedef std::vector<BasicHandle>               BasicHandleVec;
    typedef GroupVec::size_type                    size_type;

    typedef boost::shared_ptr<Group> SharedGroupPtr;
    typedef std::string ProcessName;

    Principal(ProductRegistry const& reg,
	      ProcessConfiguration const& pc,
              ProcessHistoryID const& hist = ProcessHistoryID(),
              boost::shared_ptr<DelayedReader> rtrv = boost::shared_ptr<DelayedReader>(new NoDelayedReader));

    virtual ~Principal();
    size_t  size() const { return groups_.size(); }

    EDProductGetter const* prodGetter() const {return this;}

    Principal const& groupGetter() const {return *this;}

    Principal & groupGetter() {return *this;}

    // Return the number of EDProducts contained.
    size_type numEDProducts() const;
    
    void put(std::auto_ptr<EDProduct> edp,
	     std::auto_ptr<Provenance> prov);

    SharedConstGroupPtr const getGroup(ProductID const& oid,
                                       bool resolve = true) const;

    BasicHandle  get(ProductID const& oid) const;

    BasicHandle  getBySelector(TypeID const& tid,
                               SelectorBase const& s) const;

    BasicHandle  getByLabel(TypeID const& tid,
			    std::string const& label,
                            std::string const& productInstanceName) const;

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

    Provenance const&
    getProvenance(ProductID const& oid) const;

    void
    getAllProvenance(std::vector<Provenance const *> & provenances) const;

    // ----- access to all products

    const_iterator begin() const { return groups_.begin(); }
    const_iterator end() const { return groups_.end(); }

    ProcessHistory const& processHistory() const;    

    ProcessHistoryID const& processHistoryID() const {
      return processHistoryID_;   
    }

    // ----- Add a new Group
    // *this takes ownership of the Group, which in turn owns its
    // data.
    void addGroup(std::auto_ptr<Group> g);

    ProductRegistry const& productRegistry() const {return *preg_;}

    boost::shared_ptr<DelayedReader> store() const {return store_;}

    virtual EDProduct const* getIt(ProductID const& oid) const;

    // ----- Mark this Principal as having been updated in the
    // current Process.
    void addToProcessHistory() const;

  private:

    virtual bool unscheduledFill(Group const& group) const = 0;

    // Used for indices to find groups by type and process
    typedef std::map<std::string, std::vector<ProductID> > ProcessLookup;
    typedef std::map<std::string, ProcessLookup> TypeLookup;

    size_t findGroups(TypeID const& typeID,
		      TypeLookup const& typeLookup,
		      SelectorBase const& selector,
		      BasicHandleVec& results,
		      bool stopIfProcessHasMatch) const;

    void findGroupsForProcess(std::string const& processName,
                              ProcessLookup const& processLookup,
                              SelectorBase const& selector,
                              BasicHandleVec& results) const;

    // Make my DelayedReader get the EDProduct for a Group or
    // trigger unscheduled execution if required.  The Group is
    // a cache, and so can be modified through the const reference.
    // We do not change the *number* of groups through this call, and so
    // *this is const.
    void resolveProduct(Group const& g) const;

    // Make my DelayedReader get the BranchEntryDescription
    // for a group.
    void resolveProvenance(Group const& g) const;

    mutable ProcessHistoryID processHistoryID_;

    boost::shared_ptr<ProcessHistory> processHistoryPtr_;

    ProcessConfiguration const& processConfiguration_;

    mutable bool processHistoryModified_;

    // A vector of groups.
    GroupVec groups_; // products and provenances are persistent

    // indices used to quickly find a group in the vector groups_
    // by branch (branch includes module label, instance name,
    // friendly class name (type), and process name.
    typedef std::map<BranchKey, int> BranchDict;
    BranchDict branchDict_; // 1->1

    // indices used to quickly find a group in the vector groups_
    // by the product ID.  Each EDProduct has a unique product ID.
    typedef std::map<ProductID, int> ProductDict;
    ProductDict productDict_; // 1->1
    
    // Pointer to the product registry. There is one entry in the registry
    // for each EDProduct in the event.
    ProductRegistry const* preg_;

    // Pointer to the 'source' that will be used to obtain EDProducts
    // from the persistent store.
    boost::shared_ptr<DelayedReader> store_;
  };
}
#endif
