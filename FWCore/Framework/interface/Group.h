#ifndef FWCore_Framework_Group_h
#define FWCore_Framework_Group_h

/*----------------------------------------------------------------------
  
Group: A collection of information related to a single EDProduct. This
is the storage unit of such information.

----------------------------------------------------------------------*/

#include <memory>

#include "boost/shared_ptr.hpp"

#include "Reflex/Type.h"

#include "DataFormats/Common/interface/EDProduct.h"
#include "DataFormats/Provenance/interface/ConstBranchDescription.h"
#include "DataFormats/Provenance/interface/ProductProvenance.h"
#include "DataFormats/Provenance/interface/Provenance.h"
#include "DataFormats/Provenance/interface/ProductID.h"

namespace edm {
  class Group {
  public:
    Group();

    Group(ConstBranchDescription const& bd, ProductID const& pid, bool demand);

    Group(ConstBranchDescription const& bd, ProductID const& pid);

    Group(std::auto_ptr<EDProduct> edp,
	  ConstBranchDescription const& bd,
	  ProductID const& pid,
	  std::auto_ptr<ProductProvenance> productProvenance);

    Group(ConstBranchDescription const& bd,
	  ProductID const& pid,
	  std::auto_ptr<ProductProvenance> productProvenance);

    Group(std::auto_ptr<EDProduct> edp,
	  ConstBranchDescription const& bd,
	  ProductID const& pid,
	  boost::shared_ptr<ProductProvenance> productProvenance);

    Group(ConstBranchDescription const& bd,
	  ProductID const& pid,
	  boost::shared_ptr<ProductProvenance> productProvenance);

    ~Group();

    void swap(Group& other);

    // product is not available (dropped or never created)
    bool productUnavailable() const;

    // provenance is currently available
    bool provenanceAvailable() const;

    // Scheduled for on demand production
    bool onDemand() const;

    boost::shared_ptr<EDProduct> product() const { return product_; }

    boost::shared_ptr<ProductProvenance> productProvenancePtr() const {return productProvenance_;}

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
    void setProvenance(boost::shared_ptr<ProductProvenance> productProvenance) const;

    // Write the group to the stream.
    void write(std::ostream& os) const;

    // Replace the existing group with a new one
    void replace(Group& g);

    // Return the type of the product stored in this Group.
    // We are relying on the fact that Type instances are small, and
    // so we are free to copy them at will.
    Reflex::Type productType() const;

    // Return true if this group's product is a sequence, and if the
    // sequence has a 'value_type' that 'matches' the given type.
    // 'Matches' in this context means the sequence's value_type is
    // either the same as the given type, or has the given type as a
    // public base type.
    bool isMatchingSequence(Reflex::Type const& wanted) const;

    void mergeGroup(Group * newGroup);

    ProductID const& productID() const {return pid_;};

  private:
    Group(const Group&);
    void operator=(const Group&);

    mutable boost::shared_ptr<EDProduct> product_;
    boost::shared_ptr<ConstBranchDescription> branchDescription_;
    mutable ProductID pid_;
    mutable boost::shared_ptr<ProductProvenance> productProvenance_;
    mutable boost::shared_ptr<Provenance> prov_;
    bool    dropped_;
    bool    onDemand_;
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
