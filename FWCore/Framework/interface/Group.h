#ifndef FWCore_Framework_Group_h
#define FWCore_Framework_Group_h

/*----------------------------------------------------------------------
  
Group: A collection of information related to a single EDProduct. This
is the storage unit of such information.

----------------------------------------------------------------------*/

#include <memory>

#include "boost/shared_ptr.hpp"
#include "boost/utility.hpp"

#include "Reflex/Type.h"

#include "DataFormats/Common/interface/EDProduct.h"
#include "DataFormats/Provenance/interface/ConstBranchDescription.h"
#include "DataFormats/Provenance/interface/Provenance.h"
#include "DataFormats/Provenance/interface/ProductID.h"

namespace edm {
  class DelayedReader;
  class BranchMapper;
  
  struct GroupData {
    explicit GroupData(boost::shared_ptr<ConstBranchDescription> bd) : branchDescription_(bd), pid_(), product_(), prov_() {}
    ~GroupData() {}

    void swap(GroupData& other) {
       branchDescription_.swap(other.branchDescription_);
       edm::swap(pid_, other.pid_);
       product_.swap(other.product_);
       prov_.swap(other.prov_);
    }
    void resetBranchDescription(boost::shared_ptr<ConstBranchDescription> bd) {
      branchDescription_.swap(bd);
    }
    void resetGroupData() {
      pid_.reset();
      product_.reset();
      prov_.reset();
    }

    // "const" data (some data may change only when a new input file is merged)
    boost::shared_ptr<ConstBranchDescription> branchDescription_;

    // "non-const data" (updated every event)
    ProductID pid_;
    mutable boost::shared_ptr<EDProduct> product_;
    mutable boost::shared_ptr<Provenance> prov_;
  };

  // Free swap function
  inline void swap(GroupData& a, GroupData& b) {
    a.swap(b);
  }

  class Group : private boost::noncopyable {
  public:
    Group();
    virtual ~Group();
    void resetGroupData() {
      groupData().resetGroupData();
      resetStatus();
    }

    // product is not available (dropped or never created)
    bool productUnavailable() const {return productUnavailable_();}

    // provenance is currently available
    bool provenanceAvailable() const;

    // Scheduled for on demand production
    bool onDemand() const {return onDemand_();}

    // Retrieves shared pointer to the product. 
    boost::shared_ptr<EDProduct> product() const { return groupData().product_; }

    // Sets the pointer to the product
    void setProduct(std::auto_ptr<EDProduct> prod) const;
    void setProduct(boost::shared_ptr<EDProduct> prod) const;

    // Retrieves shared pointer to the per event(lumi)(run) provenance. 
    boost::shared_ptr<ProductProvenance> productProvenancePtr() const {return provenance()->productProvenanceSharedPtr();}

    // Sets the pointer to the per event(lumi)(run) provenance. 
    void setProvenance(boost::shared_ptr<ProductProvenance> prov) const;

    // Retrieves a reference to the event independent provenance.
    ConstBranchDescription const& branchDescription() const {return *groupData().branchDescription_;}

    // Sets the pointer to the event independent provenance.
    void resetBranchDescription(boost::shared_ptr<ConstBranchDescription> bd) {groupData().resetBranchDescription(bd);}

    // Retrieves  a reference to the module label.
    std::string const& moduleLabel() const {return branchDescription().moduleLabel();}

    // Retrieves  a reference to the product instance name
    std::string const& productInstanceName() const {return branchDescription().productInstanceName();}

    // Retrieves  a reference to the process name
    std::string const& processName() const {return branchDescription().processName();}

    // Retrieves pointer to a class containing both the event independent and the per even provenance.
    Provenance* provenance() const;

    // Write the group to the stream.
    void write(std::ostream& os) const;

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

    // Retrieves the product ID of the product.
    ProductID const& productID() const {return groupData().pid_;};

    // Sets the product ID of the product.
    void setProductID(ProductID const& pid) {groupData().pid_ = pid;};

    // Merges two instances of the product.
    void mergeProduct(boost::shared_ptr<EDProduct> edp) const;

    // Puts the product and its per event(lumi)(run) provenance into the Group.
    void putProduct(boost::shared_ptr<EDProduct> edp, boost::shared_ptr<ProductProvenance> productProvenance) {
      putProduct_(edp, productProvenance);
    }
    void putProduct(boost::shared_ptr<EDProduct> edp, std::auto_ptr<ProductProvenance> productProvenance) {
      putProduct_(edp, boost::shared_ptr<ProductProvenance>(productProvenance.release()));
    }

    // Puts the product into the Group.
    void putProduct(boost::shared_ptr<EDProduct> edp) const {
      putProduct_(edp);
    }

    // Puts the product and its per event(lumi)(run) provenance into the Group, or merges it with the pre-existing product
    void putOrMergeProduct(boost::shared_ptr<EDProduct> edp, boost::shared_ptr<ProductProvenance> productProvenance) {
      putOrMergeProduct_(edp, productProvenance);
    }
    void putOrMergeProduct(boost::shared_ptr<EDProduct> edp, std::auto_ptr<ProductProvenance> productProvenance) {
      putOrMergeProduct_(edp, boost::shared_ptr<ProductProvenance>(productProvenance.release()));
    }

    // Puts the product into the Group, or merges it with the pre-existing product
    void putOrMergeProduct(boost::shared_ptr<EDProduct> edp) const {
      putOrMergeProduct_(edp);
    }

    // Resolves the per event(lumi)(run) provenance (read from input).
    void resolveProvenance(boost::shared_ptr<BranchMapper> store) const {
      resolveProvenance_(store);
    }
 
  private:
    virtual bool onDemand_() const = 0;
    virtual bool productUnavailable_() const = 0;
    virtual void putProduct_(boost::shared_ptr<EDProduct> edp, boost::shared_ptr<ProductProvenance> productProvenance) = 0;
    virtual void putProduct_(boost::shared_ptr<EDProduct> edp) const = 0;
    virtual void putOrMergeProduct_(boost::shared_ptr<EDProduct> edp, boost::shared_ptr<ProductProvenance> productProvenance) = 0;
    virtual void putOrMergeProduct_(boost::shared_ptr<EDProduct> edp) const = 0;
    virtual void resolveProvenance_(boost::shared_ptr<BranchMapper> store) const = 0;
    virtual void resetStatus() = 0;
    virtual GroupData const& groupData() const = 0;
    virtual GroupData& groupData() = 0;
  };

  inline
  std::ostream&
  operator<<(std::ostream& os, Group const& g) {
    g.write(os);
    return os;
  }

  class InputGroup : public Group {
    enum GroupStatus {
      Present = 0,
      NeverCreated = 1,
      Dropped = 2,
      NotRun = 3,
      NotCompleted = 4,
      NotPut = 5,
      UnscheduledNotRun = 6,
      Unknown = 0xfe,
      Uninitialized = 0xff
    };
    public:
      explicit InputGroup(boost::shared_ptr<ConstBranchDescription> bd) :
	Group(), groupData_(bd), theStatus_(Uninitialized) {}
      virtual ~InputGroup();
      GroupStatus const& status() const {return theStatus_;}
      void swap(InputGroup& other) {
	edm::swap(groupData_, other.groupData_);
	std::swap(theStatus_, other.theStatus_);
      }
      // The following are is because we can add an EDProduct to the
      // cache after creation of the Group, without changing the meaning
      // of the Group.
      void updateStatus() const;
      void setStatus(ProductStatus const&) const;
    private:
      virtual void putProduct_(boost::shared_ptr<EDProduct> edp, boost::shared_ptr<ProductProvenance> productProvenance);
      virtual void putProduct_(boost::shared_ptr<EDProduct> edp) const;
      virtual void putOrMergeProduct_(boost::shared_ptr<EDProduct> edp, boost::shared_ptr<ProductProvenance> productProvenance);
      virtual void putOrMergeProduct_(boost::shared_ptr<EDProduct> edp) const;
      virtual void resolveProvenance_(boost::shared_ptr<BranchMapper> store) const;
      virtual void resetStatus() {theStatus_ = Uninitialized;}
      virtual bool onDemand_() const {return false;}
      virtual bool productUnavailable_() const;
      virtual GroupData const& groupData() const {return groupData_;} 
      virtual GroupData& groupData() {return groupData_;} 
      GroupData groupData_;
      mutable GroupStatus theStatus_;
  };

  // Free swap function
  inline void swap(InputGroup& a, InputGroup& b) {
    a.swap(b);
  }
  
  class ProducedGroup : public Group {
    protected:
    enum GroupStatus {
      Present = 0,
      NotRun = 3,
      NotCompleted = 4,
      NotPut = 5,
      UnscheduledNotRun = 6,
      Uninitialized = 0xff
    };
    public:
      ProducedGroup() : Group() {}
      virtual ~ProducedGroup();
      void producerStarted();
      void producerCompleted();
      GroupStatus const& status() const {return status_();}
      GroupStatus& status() {return status_();}
    private:
      virtual void putProduct_(boost::shared_ptr<EDProduct> edp, boost::shared_ptr<ProductProvenance> productProvenance);
      virtual void putProduct_(boost::shared_ptr<EDProduct> edp) const;
      virtual void putOrMergeProduct_(boost::shared_ptr<EDProduct> edp, boost::shared_ptr<ProductProvenance> productProvenance);
      virtual void putOrMergeProduct_(boost::shared_ptr<EDProduct> edp) const;
      virtual void resolveProvenance_(boost::shared_ptr<BranchMapper> store) const;
      virtual GroupStatus const& status_() const = 0;
      virtual GroupStatus& status_() = 0;
      virtual bool productUnavailable_() const;
  };
  
  class ScheduledGroup : public ProducedGroup {
    public:
      explicit ScheduledGroup(boost::shared_ptr<ConstBranchDescription> bd) : ProducedGroup(), groupData_(bd), theStatus_(NotRun) {}
      virtual ~ScheduledGroup();
      void swap(ScheduledGroup& other) {
	edm::swap(groupData_, other.groupData_);
	std::swap(theStatus_, other.theStatus_);
      }
    private:
      virtual void resetStatus() {theStatus_ = NotRun;}
      virtual bool onDemand_() const {return false;}
      virtual GroupData const& groupData() const {return groupData_;} 
      virtual GroupData& groupData() {return groupData_;} 
      virtual GroupStatus const& status_() const {return theStatus_;}
      virtual GroupStatus& status_() {return theStatus_;}
      GroupData groupData_;
      GroupStatus theStatus_;
  };

  // Free swap function
  inline void swap(ScheduledGroup& a, ScheduledGroup& b) {
    a.swap(b);
  }
  
  class UnscheduledGroup : public ProducedGroup {
    public:
      explicit UnscheduledGroup(boost::shared_ptr<ConstBranchDescription> bd) : ProducedGroup(), groupData_(bd), theStatus_(UnscheduledNotRun) {}
      virtual ~UnscheduledGroup();
      void swap(UnscheduledGroup& other) {
	edm::swap(groupData_, other.groupData_);
	std::swap(theStatus_, other.theStatus_);
      }
    private:
      virtual void resetStatus() {theStatus_ = UnscheduledNotRun;}
      virtual bool onDemand_() const {return status() == UnscheduledNotRun;}
      virtual GroupData const& groupData() const {return groupData_;} 
      virtual GroupData& groupData() {return groupData_;} 
      virtual GroupStatus const& status_() const {return theStatus_;}
      virtual GroupStatus& status_() {return theStatus_;}
      GroupData groupData_;
      GroupStatus theStatus_;
  };

  // Free swap function
  inline void swap(UnscheduledGroup& a, UnscheduledGroup& b) {
    a.swap(b);
  }
  
  class SourceGroup : public ProducedGroup {
    public:
      explicit SourceGroup(boost::shared_ptr<ConstBranchDescription> bd) : ProducedGroup(), groupData_(bd), theStatus_(NotPut) {}
      virtual ~SourceGroup();
      void swap(SourceGroup& other) {
	edm::swap(groupData_, other.groupData_);
	std::swap(theStatus_, other.theStatus_);
      }
    private:
      virtual void resetStatus() {theStatus_ = NotPut;}
      virtual bool onDemand_() const {return false;}
      virtual GroupData const& groupData() const {return groupData_;} 
      virtual GroupData& groupData() {return groupData_;} 
      virtual GroupStatus const& status_() const {return theStatus_;}
      virtual GroupStatus& status_() {return theStatus_;}
      GroupData groupData_;
      GroupStatus theStatus_;
  };

  // Free swap function
  inline void swap(edm::SourceGroup& a, edm::SourceGroup& b) {
    a.swap(b);
  }
}

#endif
