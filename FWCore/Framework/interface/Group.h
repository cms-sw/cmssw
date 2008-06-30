#ifndef FWCore_Framework_GroupT_h
#define FWCore_Framework_GroupT_h

/*----------------------------------------------------------------------
  
GroupT: A collection of information related to a single EDProduct. This
is the storage unit of such information.

----------------------------------------------------------------------*/

#include <memory>

#include "boost/shared_ptr.hpp"

#include "Reflex/Type.h"

#include "DataFormats/Common/interface/EDProduct.h"
#include "DataFormats/Provenance/interface/ConstBranchDescription.h"
#include "DataFormats/Provenance/interface/Provenance.h"

// In the future, the untemplated Provenance class should no longer be used here.
// We still need it for now.

namespace edm {
  template <typename T>
  class GroupT {
  public:

    GroupT();

    GroupT(ConstBranchDescription const& bd, bool demand);

    explicit GroupT(ConstBranchDescription const& bd);

    GroupT(std::auto_ptr<EDProduct> edp,
	  ConstBranchDescription const& bd,
	  std::auto_ptr<T> entryInfo);

    GroupT(ConstBranchDescription const& bd,
	  std::auto_ptr<T> entryInfo);

    GroupT(std::auto_ptr<EDProduct> edp,
	  ConstBranchDescription const& bd,
	  boost::shared_ptr<T> entryInfo);

    GroupT(ConstBranchDescription const& bd,
	  boost::shared_ptr<T> entryInfo);

    ~GroupT();

    void swap(GroupT& other);

    // product is not available (dropped or never created)
    bool productUnavailable() const;

    // provenance is currently available
    bool provenanceAvailable() const;

    // Scheduled for on demand production
    bool onDemand() const;

    EDProduct const* product() const { return product_.get(); }

    boost::shared_ptr<T> entryInfoPtr() const {return entryInfo_;}

    ConstBranchDescription const& productDescription() const {return *branchDescription_;}

    std::string const& moduleLabel() const {return branchDescription_->moduleLabel();}

    std::string const& productInstanceName() const {return branchDescription_->productInstanceName();}

    std::string const& processName() const {return branchDescription_->processName();}

    Provenance const * provenance() const;

    ProductStatus status() const;

    // The following is const because we can add an EDProduct to the
    // cache after creation of the Group, without changing the meaning
    // of the Group.
    void setProduct(std::auto_ptr<EDProduct> prod) const;

    // The following is const because we can add the provenance
    // to the cache after creation of the Group, without changing the meaning
    // of the Group.
    void setProvenance(boost::shared_ptr<T> entryInfo) const;

    // Write the group to the stream.
    void write(std::ostream& os) const;

    // Replace the existing group with a new one
    void replace(GroupT& g);

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

    void mergeGroup(GroupT * newGroup);

  private:
    GroupT(const GroupT&);
    void operator=(const GroupT&);

    mutable boost::shared_ptr<EDProduct> product_;
    mutable boost::shared_ptr<ConstBranchDescription> branchDescription_;
    mutable boost::shared_ptr<T> entryInfo_;
    mutable boost::shared_ptr<Provenance> prov_;
    bool    dropped_;
    bool    onDemand_;
  };

  // Free swap function
  template <typename T>
  inline
  void
  swap(GroupT<T>& a, GroupT<T>& b) {
    a.swap(b);
  }

  template <typename T>
  inline
  std::ostream&
  operator<<(std::ostream& os, GroupT<T> const& g) {
    g.write(os);
    return os;
  }

}

#include "Group.icc"
#endif
