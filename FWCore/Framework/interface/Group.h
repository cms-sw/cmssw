#ifndef FWCore_Framework_Group_h
#define FWCore_Framework_Group_h

/*----------------------------------------------------------------------

Group: A collection of information related to a single EDProduct. This
is the storage unit of such information.

----------------------------------------------------------------------*/

#include "DataFormats/Common/interface/ProductData.h"
#include "DataFormats/Common/interface/WrapperHolder.h"
#include "DataFormats/Common/interface/WrapperOwningHolder.h"
#include "DataFormats/Provenance/interface/ConstBranchDescription.h"
#include "DataFormats/Provenance/interface/Provenance.h"
#include "FWCore/Utilities/interface/TypeID.h"

#include "boost/shared_ptr.hpp"
#include "boost/utility.hpp"

#include <string>

namespace edm {
  class BranchMapper;
  class DelayedReader;
  class WrapperInterfaceBase;

  class Group : private boost::noncopyable {
  public:
    Group();
    virtual ~Group();

    ProductData const& productData() const {
      return getProductData();
    }

    ProductData& productData() {
      return getProductData();
    }

    void resetStatus () {
      resetStatus_();
    }

    void setProductDeleted () {
      setProductDeleted_();
    }

    void resetProductData() {
      getProductData().resetProductData();
      resetStatus_();
    }

    void deleteProduct() {
      getProductData().resetProductData();
      setProductDeleted_();
    }
    
    // product is not available (dropped or never created)
    bool productUnavailable() const {return productUnavailable_();}

    // provenance is currently available
    bool provenanceAvailable() const;

    // Scheduled for on demand production
    bool onDemand() const {return onDemand_();}
    
    // Product was deleted early in order to save memory
    bool productWasDeleted() const {return productWasDeleted_();}

    // Retrieves a shared pointer to the wrapped product.
    boost::shared_ptr<void const> product() const { return getProductData().wrapper_; }

    // Retrieves the wrapped product and type. (non-owning);
    WrapperHolder wrapper() const { return WrapperHolder(getProductData().wrapper_.get(), getProductData().getInterface()); }

    // Retrieves pointer to the per event(lumi)(run) provenance.
    ProductProvenance* productProvenancePtr() const {
      return provenance()->productProvenance();
    }

    // Sets the the per event(lumi)(run) provenance.
    void setProductProvenance(ProductProvenance const& prov) const;

    // Retrieves a reference to the event independent provenance.
    ConstBranchDescription const& branchDescription() const {return branchDescription_();}

    // Sets the pointer to the event independent provenance.
    void resetBranchDescription(boost::shared_ptr<ConstBranchDescription> bd) {resetBranchDescription_(bd);}

    // Retrieves a reference to the module label.
    std::string const& moduleLabel() const {return branchDescription().moduleLabel();}

    // Retrieves a reference to the product instance name
    std::string const& productInstanceName() const {return branchDescription().productInstanceName();}

    // Retrieves a reference to the process name
    std::string const& processName() const {return branchDescription().processName();}

    // Retrieves pointer to a class containing both the event independent and the per even provenance.
    Provenance* provenance() const;

    // Initializes the event independent portion of the provenance, plus the process history ID, the product ID, and the mapper.
    void setProvenance(boost::shared_ptr<BranchMapper> mapper, ProcessHistoryID const& phid, ProductID const& pid);

    // Initializes the event independent portion of the provenance, plus the process history ID and the mapper.
    void setProvenance(boost::shared_ptr<BranchMapper> mapper, ProcessHistoryID const& phid);

    // Initializes the process history ID.
    void setProcessHistoryID(ProcessHistoryID const& phid);

    // Write the group to the stream.
    void write(std::ostream& os) const;

    // Return the type of the product stored in this Group.
    // We are relying on the fact that Type instances are small, and
    // so we are free to copy them at will.
    TypeID productType() const;

    // Retrieves the product ID of the product.
    ProductID const& productID() const {return getProductData().prov_.productID();}

    // Puts the product and its per event(lumi)(run) provenance into the Group.
    void putProduct(WrapperOwningHolder const& edp, ProductProvenance const& productProvenance) {
      putProduct_(edp, productProvenance);
    }

    // Puts the product into the Group.
    void putProduct(WrapperOwningHolder const& edp) const {
      putProduct_(edp);
    }

    // This returns true if it will be put, false if it will be merged
    bool putOrMergeProduct() const {
      return putOrMergeProduct_();
    }

    // merges the product with the pre-existing product
    void mergeProduct(WrapperOwningHolder const& edp, ProductProvenance& productProvenance) {
      mergeProduct_(edp, productProvenance);
    }

    void mergeProduct(WrapperOwningHolder const& edp) const {
      mergeProduct_(edp);
    }

    // Merges two instances of the product.
    void mergeTheProduct(WrapperOwningHolder const& edp) const;

    void reallyCheckType(WrapperOwningHolder const& prod) const;

    void checkType(WrapperOwningHolder const& prod) const {
      checkType_(prod);
    }

    void swap(Group& rhs) {swap_(rhs);}

  private:
    virtual ProductData const& getProductData() const = 0;
    virtual ProductData& getProductData() = 0;
    virtual void swap_(Group& rhs) = 0;
    virtual bool onDemand_() const = 0;
    virtual bool productUnavailable_() const = 0;
    virtual bool productWasDeleted_() const = 0;
    virtual void putProduct_(WrapperOwningHolder const& edp, ProductProvenance const& productProvenance) = 0;
    virtual void putProduct_(WrapperOwningHolder const& edp) const = 0;
    virtual void mergeProduct_(WrapperOwningHolder const&  edp, ProductProvenance& productProvenance) = 0;
    virtual void mergeProduct_(WrapperOwningHolder const& edp) const = 0;
    virtual bool putOrMergeProduct_() const = 0;
    virtual void checkType_(WrapperOwningHolder const& prod) const = 0;
    virtual void resetStatus_() = 0;
    virtual void setProductDeleted_() = 0;
    virtual ConstBranchDescription const& branchDescription_() const = 0;
    virtual void resetBranchDescription_(boost::shared_ptr<ConstBranchDescription> bd) = 0;
  };

  inline
  std::ostream&
  operator<<(std::ostream& os, Group const& g) {
    g.write(os);
    return os;
  }

  class InputGroup : public Group {
    public:
      explicit InputGroup(boost::shared_ptr<ConstBranchDescription> bd) :
        Group(), productData_(bd), productIsUnavailable_(false),
        productHasBeenDeleted_(false) {}
      virtual ~InputGroup();

      // The following is const because we can add an EDProduct to the
      // cache after creation of the Group, without changing the meaning
      // of the Group.
      void setProduct(WrapperOwningHolder const& prod) const;
      bool productIsUnavailable() const {return productIsUnavailable_;}
      void setProductUnavailable() const {productIsUnavailable_ = true;}

    private:
      virtual void swap_(Group& rhs) {
        InputGroup& other = dynamic_cast<InputGroup&>(rhs);
        edm::swap(productData_, other.productData_);
        std::swap(productIsUnavailable_, other.productIsUnavailable_);
      }
      virtual void putProduct_(WrapperOwningHolder const& edp, ProductProvenance const& productProvenance);
      virtual void putProduct_(WrapperOwningHolder const& edp) const;
      virtual void mergeProduct_(WrapperOwningHolder const& edp, ProductProvenance& productProvenance);
      virtual void mergeProduct_(WrapperOwningHolder const& edp) const;
      virtual bool putOrMergeProduct_() const;
      virtual void checkType_(WrapperOwningHolder const&) const {}
      virtual void resetStatus_() {productIsUnavailable_ = false;
        productHasBeenDeleted_=false;}
      virtual bool onDemand_() const {return false;}
      virtual bool productUnavailable_() const;
      virtual bool productWasDeleted_() const {return productHasBeenDeleted_;}
      virtual ProductData const& getProductData() const {return productData_;}
      virtual ProductData& getProductData() {return productData_;}
      virtual void setProductDeleted_() {productHasBeenDeleted_ = true;}
      virtual ConstBranchDescription const& branchDescription_() const {return *productData().branchDescription();}
      virtual void resetBranchDescription_(boost::shared_ptr<ConstBranchDescription> bd) {productData().resetBranchDescription(bd);}

      ProductData productData_;
      mutable bool productIsUnavailable_;
      mutable bool productHasBeenDeleted_;
  };

  // Free swap function
  inline void swap(InputGroup& a, InputGroup& b) {
    a.swap(b);
  }

  class ProducedGroup : public Group {
    public:
    enum GroupStatus {
      Present = 0,
      NotRun = 3,
      NotCompleted = 4,
      NotPut = 5,
      UnscheduledNotRun = 6,
      ProductDeleted =7,
      Uninitialized = 0xff
    };
      ProducedGroup() : Group() {}
      virtual ~ProducedGroup();
      void producerStarted();
      void producerCompleted();
      GroupStatus& status() const {return status_();}
    private:
      virtual void putProduct_(WrapperOwningHolder const& edp, ProductProvenance const& productProvenance);
      virtual void putProduct_(WrapperOwningHolder const& edp) const;
      virtual void mergeProduct_(WrapperOwningHolder const& edp, ProductProvenance& productProvenance);
      virtual void mergeProduct_(WrapperOwningHolder const& edp) const;
      virtual bool putOrMergeProduct_() const;
      virtual void checkType_(WrapperOwningHolder const& prod) const {
        reallyCheckType(prod);
      }
      virtual GroupStatus& status_() const = 0;
      virtual bool productUnavailable_() const;
      virtual bool productWasDeleted_() const;
      virtual void setProductDeleted_();
      virtual ConstBranchDescription const& branchDescription_() const {return *productData().branchDescription();}
      virtual void resetBranchDescription_(boost::shared_ptr<ConstBranchDescription> bd) {productData().resetBranchDescription(bd);}
  };

  class ScheduledGroup : public ProducedGroup {
    public:
      explicit ScheduledGroup(boost::shared_ptr<ConstBranchDescription> bd) : ProducedGroup(), productData_(bd), theStatus_(NotRun) {}
      virtual ~ScheduledGroup();
    private:
      virtual void swap_(Group& rhs) {
        ScheduledGroup& other = dynamic_cast<ScheduledGroup&>(rhs);
        edm::swap(productData_, other.productData_);
        std::swap(theStatus_, other.theStatus_);
      }
      virtual void resetStatus_() {theStatus_ = NotRun;}
      virtual bool onDemand_() const {return false;}
      virtual ProductData const& getProductData() const {return productData_;}
      virtual ProductData& getProductData() {return productData_;}
      virtual GroupStatus& status_() const {return theStatus_;}
      ProductData productData_;
      mutable GroupStatus theStatus_;
  };

  // Free swap function
  inline void swap(ScheduledGroup& a, ScheduledGroup& b) {
    a.swap(b);
  }

  class UnscheduledGroup : public ProducedGroup {
    public:
      explicit UnscheduledGroup(boost::shared_ptr<ConstBranchDescription> bd) : ProducedGroup(), productData_(bd), theStatus_(UnscheduledNotRun) {}
      virtual ~UnscheduledGroup();
    private:
      virtual void swap_(Group& rhs) {
        UnscheduledGroup& other = dynamic_cast<UnscheduledGroup&>(rhs);
        edm::swap(productData_, other.productData_);
        std::swap(theStatus_, other.theStatus_);
      }
      virtual void resetStatus_() {theStatus_ = UnscheduledNotRun;}
      virtual bool onDemand_() const {return status() == UnscheduledNotRun;}
      virtual ProductData const& getProductData() const {return productData_;}
      virtual ProductData& getProductData() {return productData_;}
      virtual GroupStatus& status_() const {return theStatus_;}
      ProductData productData_;
      mutable GroupStatus theStatus_;
  };

  // Free swap function
  inline void swap(UnscheduledGroup& a, UnscheduledGroup& b) {
    a.swap(b);
  }

  class SourceGroup : public ProducedGroup {
    public:
      explicit SourceGroup(boost::shared_ptr<ConstBranchDescription> bd) : ProducedGroup(), productData_(bd), theStatus_(NotPut) {}
      virtual ~SourceGroup();
    private:
      virtual void swap_(Group& rhs) {
        SourceGroup& other = dynamic_cast<SourceGroup&>(rhs);
        edm::swap(productData_, other.productData_);
        std::swap(theStatus_, other.theStatus_);
      }
      virtual void resetStatus_() {theStatus_ = NotPut;}
      virtual bool onDemand_() const {return false;}
      virtual ProductData const& getProductData() const {return productData_;}
      virtual ProductData& getProductData() {return productData_;}
      virtual GroupStatus& status_() const {return theStatus_;}
      ProductData productData_;
      mutable GroupStatus theStatus_;
  };

  class AliasGroup : public Group {
    public:
      typedef ProducedGroup::GroupStatus GroupStatus;
      explicit AliasGroup(boost::shared_ptr<ConstBranchDescription> bd, ProducedGroup& realGroup) : Group(), realGroup_(realGroup), bd_(bd) {}
      virtual ~AliasGroup();
    private:
      virtual void swap_(Group& rhs) {
        AliasGroup& other = dynamic_cast<AliasGroup&>(rhs);
        realGroup_.swap(other.realGroup_);
        std::swap(bd_, other.bd_);
      }
      virtual bool onDemand_() const {return realGroup_.onDemand();}
      virtual GroupStatus& status_() const {return realGroup_.status();}
      virtual void resetStatus_() {realGroup_.resetStatus();}
      virtual bool productUnavailable_() const {return realGroup_.productUnavailable();}
      virtual bool productWasDeleted_() const {return realGroup_.productWasDeleted();}
      virtual void checkType_(WrapperOwningHolder const& prod) const {realGroup_.checkType(prod);}
      virtual ProductData const& getProductData() const {return realGroup_.productData();}
      virtual ProductData& getProductData() {return realGroup_.productData();}
      virtual void setProductDeleted_() {realGroup_.setProductDeleted();}
      virtual void putProduct_(WrapperOwningHolder const& edp, ProductProvenance const& productProvenance) {
        realGroup_.putProduct(edp, productProvenance);
      }
      virtual void putProduct_(WrapperOwningHolder const& edp) const {
        realGroup_.putProduct(edp);
      }
      virtual void mergeProduct_(WrapperOwningHolder const& edp, ProductProvenance& productProvenance) {
        realGroup_.mergeProduct(edp, productProvenance);
      }
      virtual void mergeProduct_(WrapperOwningHolder const& edp) const {
        realGroup_.mergeProduct(edp);
      }
      virtual bool putOrMergeProduct_() const {
        return realGroup_.putOrMergeProduct();
      }
      virtual ConstBranchDescription const& branchDescription_() const {return *bd_;}
      virtual void resetBranchDescription_(boost::shared_ptr<ConstBranchDescription> bd) {bd_ = bd;}

      ProducedGroup& realGroup_;
      boost::shared_ptr<ConstBranchDescription> bd_;
  };

  // Free swap function
  inline void swap(SourceGroup& a, SourceGroup& b) {
    a.swap(b);
  }
}

#endif
