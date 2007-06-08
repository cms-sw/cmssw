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

$Id: Principal.h,v 1.8 2007/06/06 23:33:48 wmtan Exp $

----------------------------------------------------------------------*/
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "boost/shared_ptr.hpp"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "DataFormats/Provenance/interface/ProvenanceFwd.h"
#include "DataFormats/Common/interface/EDProductGetter.h"
#include "DataFormats/Provenance/interface/ProcessHistory.h"
#include "FWCore/Framework/interface/NoDelayedReader.h"


namespace edm {
  class Principal : public EDProductGetter {
  public:
    typedef std::vector<boost::shared_ptr<Group> > GroupVec;
    typedef ProcessHistory::const_iterator         ProcessNameConstIterator;
    typedef boost::shared_ptr<const Group>         SharedConstGroupPtr;
    typedef std::vector<BasicHandle>               BasicHandleVec;
    typedef GroupVec::size_type                    size_type;

    typedef boost::shared_ptr<Group> SharedGroupPtr;
    typedef std::string ProcessName;

    Principal(boost::shared_ptr<ProductRegistry const> reg,
	      ProcessConfiguration const& pc,
              ProcessHistoryID const& hist = ProcessHistoryID(),
              boost::shared_ptr<DelayedReader> rtrv = boost::shared_ptr<DelayedReader>(new NoDelayedReader));

    virtual ~Principal();
    size_t  size() const { return size_; }

    EDProductGetter const* prodGetter() const {return this;}

    Principal const& groupGetter() const {return *this;}

    Principal & groupGetter() {return *this;}

    // Return the number of EDProducts contained.
    size_type numEDProducts() const;
    
    void put(std::auto_ptr<EDProduct> edp,
	     std::auto_ptr<Provenance> prov);

    BasicHandle  get(ProductID const& oid) const;

    BasicHandle  getForOutput(ProductID const& oid, bool selected) const;

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

    ProcessHistory const& processHistory() const;    

    ProcessHistoryID const& processHistoryID() const {
      return processHistoryID_;   
    }

    void addGroup(ConstBranchDescription const& bd);

    void addGroup(std::auto_ptr<Provenance>, bool onDemand = false);

    void addGroup(std::auto_ptr<EDProduct> prod, std::auto_ptr<Provenance> prov);

    ProductRegistry const& productRegistry() const {return *preg_;}

    boost::shared_ptr<DelayedReader> store() const {return store_;}

    virtual EDProduct const* getIt(ProductID const& oid) const;

    // ----- Mark this Principal as having been updated in the
    // current Process.
    void addToProcessHistory() const;

  private:
    // We need a custom iterator to skip non-existent groups.
    class const_iterator : public std::iterator <std::forward_iterator_tag, boost::shared_ptr<Group> > {
    public:
      typedef GroupVec::value_type value_type;
      typedef GroupVec::const_iterator Iter;
      const_iterator(Iter const& it, Iter const& itEnd) : iter_(it), iterEnd_(itEnd) {}
      value_type const& operator*() const { return *iter_; }
      value_type const * operator->() const { return &*iter_; }
      const_iterator & operator++() {
        ++iter_; while (iter_ != iterEnd_ && iter_->get() == 0) ++iter_; return *this;
      }
      const_iterator operator++(int) {
        const_iterator it(*this); ++iter_; while (iter_ != iterEnd_ && iter_->get() == 0) ++iter_; return it;
      }
      bool operator==(const_iterator const& rhs) const {return this->iter_ == rhs.iter_;}
      bool operator!=(const_iterator const& rhs) const {return this->iter_ != rhs.iter_;}
    private:
      Iter iter_;
      Iter iterEnd_;
    };

    // ----- access to all products

    const_iterator begin() const {
      GroupVec::const_iterator iter(groups_.begin());
      while (iter != groups_.end() && iter->get() == 0) ++iter; 
      return const_iterator(iter, groups_.end());
    }

    const_iterator end() const { return const_iterator(groups_.end(), groups_.end()); }

    // ----- Add a new Group
    // *this takes ownership of the Group, which in turn owns its
    // data.
    void addGroup_(std::auto_ptr<Group> g);

    SharedConstGroupPtr const getGroup(ProductID const& oid,
                                       bool resolveProd,
                                       bool resolveProv,
				       bool fillOnDemand) const;

    virtual bool unscheduledFill(Provenance const& prov) const = 0;

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
    void resolveProduct(Group const& g, bool fillOnDemand) const;

    // Make my DelayedReader get the BranchEntryDescription
    // for a group.
    void resolveProvenance(Group const& g) const;

    mutable ProcessHistoryID processHistoryID_;

    boost::shared_ptr<ProcessHistory> processHistoryPtr_;

    ProcessConfiguration const& processConfiguration_;

    mutable bool processHistoryModified_;

    // A vector of groups.
    GroupVec groups_; // products and provenances are persistent

    // Pointer to the product registry. There is one entry in the registry
    // for each EDProduct in the event.
    boost::shared_ptr<ProductRegistry const> preg_;

    // Pointer to the 'source' that will be used to obtain EDProducts
    // from the persistent store.
    boost::shared_ptr<DelayedReader> store_;

    // Number of groups in the event (excluding on-demand groups not yet produced).
    size_t size_;
  };
}
#endif
