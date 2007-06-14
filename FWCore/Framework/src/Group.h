#ifndef FWCore_Framework_Group_h
#define FWCore_Framework_Group_h

/*----------------------------------------------------------------------
  
Group: A collection of information related to a single EDProduct. This
is the storage unit of such information.

$Id: Group.h,v 1.21 2007/05/29 19:27:01 wmtan Exp $

----------------------------------------------------------------------*/

#include <memory>

#include "boost/shared_ptr.hpp"

#include "Reflex/Type.h"

#include "DataFormats/Common/interface/EDProduct.h"
#include "DataFormats/Provenance/interface/Provenance.h"

namespace edm {
  class BasicHandle;
  class Group {
  public:
    explicit Group(std::auto_ptr<Provenance> prov,
 	  bool onDemand = false);

    explicit Group(ConstBranchDescription const& bd);

    Group(std::auto_ptr<EDProduct> edp,
	  std::auto_ptr<Provenance> prov);

    ~Group();

    void swap(Group& other);

    // drop (write) on output: choose not to write, output module

    // product is not available (dropped or never created)
    bool productUnavailable() const;

    // provenance is currently available
    bool provenanceAvailable() const;

    // True if and only if this group's product has not been produced yet
    // and an unscheduled module in this process declared it produces it
    bool onDemand() const { return onDemand_; }

    EDProduct const* product() const { return product_.get(); }

    Provenance & provenance() {return *provenance_;} 

    Provenance const& provenance() const {return *provenance_;} 

    BranchEntryDescription const* branchEntryDescription() const {return provenance_->branchEntryDescription().get();}

    BranchDescription const& productDescription() const { return provenance_->product(); }

    std::string const& moduleLabel() const 
    { return provenance_->moduleLabel(); }

    std::string const& productInstanceName() const 
    { return provenance_->productInstanceName(); }

    std::string const& processName() const
    { return provenance_->processName(); }

    // The following is const because we can add an EDProduct to the
    // cache after creation of the Group, without changing the meaning
    // of the Group.
    void setProduct(std::auto_ptr<EDProduct> prod) const;

    // The following is const because we can add a BranchEntryDescription
    // to the cache after creation of the Group, without changing the meaning
    // of the Group.
    void setProvenance(std::auto_ptr<BranchEntryDescription> prov) const;

    // Write the group to the stream.
    void write(std::ostream& os) const;

    // Figure out what to do if a duplicate group is created.
    bool replace(Group& g);

    // Return the type of the product stored in this Group.
    // We are relying on the fact that Type instances are small, and
    // so we are free to copy them at will.
    ROOT::Reflex::Type productType() const;


    // Return true if this group's product is a sequence, and if the
    // sequence has a 'value_type' that 'matches' the given type.
    // 'Matches' in this context means the sequence's value_type is
    // either the same as the given type, or has the given type as a
    // public base type.
    bool isMatchingSequence(ROOT::Reflex::Type const& wanted) const;


    // Return a BasicHandle to this Group.
    BasicHandle makeBasicHandle() const;

  private:
    Group(const Group&);
    void operator=(const Group&);

    mutable boost::shared_ptr<EDProduct> product_;
    // mutable boost::shared_ptr<BranchEntryDescription> branchEntryDescription_;
    // BranchDescription branchDescription_;
    mutable boost::shared_ptr<Provenance> provenance_;
    mutable bool      unavailable_;
    bool              onDemand_;
  };

  // Free swap function
  inline
  void
  swap(Group& a, Group& b) {
    a.swap(b);
  }

  inline
  std::ostream&
  operator<<(std::ostream& os, Group const& g) {
    g.write(os);
    return os;
  }

}
#endif
