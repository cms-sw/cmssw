#ifndef FWCore_Framework_ProductHolder_h
#define FWCore_Framework_ProductHolder_h

/*----------------------------------------------------------------------

ProductHolder: A collection of information related to a single EDProduct or
a set of related EDProducts. This is the storage unit of such information.

----------------------------------------------------------------------*/

#include "DataFormats/Common/interface/ProductData.h"
#include "DataFormats/Common/interface/WrapperHolder.h"
#include "DataFormats/Common/interface/WrapperOwningHolder.h"
#include "DataFormats/Provenance/interface/BranchID.h"
#include "DataFormats/Provenance/interface/ConstBranchDescription.h"
#include "DataFormats/Provenance/interface/Provenance.h"
#include "FWCore/Utilities/interface/ProductHolderIndex.h"
#include "FWCore/Utilities/interface/TypeID.h"

#include "boost/shared_ptr.hpp"

#include <string>

namespace edm {
  class BranchMapper;
  class DelayedReader;
  class Principal;
  class WrapperInterfaceBase;

  class ProductHolderBase {
  public:

    enum ResolveStatus { ProductFound, ProductNotFound, Ambiguous };

    ProductHolderBase();
    virtual ~ProductHolderBase();

    ProductHolderBase(ProductHolderBase const&) = delete; // Disallow copying and moving
    ProductHolderBase& operator=(ProductHolderBase const&) = delete; // Disallow copying and moving

    ProductData const& productData() const {
      return getProductData();
    }

    ProductData& productData() {
      return getProductData();
    }

    ProductData const* resolveProduct(ResolveStatus& resolveStatus, bool skipCurrentProcess) const {
      return resolveProduct_(resolveStatus, skipCurrentProcess);
    }

    void resetStatus () {
      resetStatus_();
    }

    void setProductDeleted () {
      setProductDeleted_();
    }

    void resetProductData() { resetProductData_(); }

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
    ProductProvenance* productProvenancePtr() const { return productProvenancePtr_(); }

    // Sets the the per event(lumi)(run) provenance.
    void setProductProvenance(ProductProvenance const& prov) const;

    // Retrieves a reference to the event independent provenance.
    ConstBranchDescription const& branchDescription() const {return branchDescription_();}

    // Retrieves a reference to the event independent provenance.
    bool singleProduct() const {return singleProduct_();}

    void setPrincipal(Principal* principal) { setPrincipal_(principal); }

    // Sets the pointer to the event independent provenance.
    void resetBranchDescription(boost::shared_ptr<ConstBranchDescription> bd) {resetBranchDescription_(bd);}

    // Retrieves a reference to the module label.
    std::string const& moduleLabel() const {return branchDescription().moduleLabel();}

    // Same as moduleLabel except in the case of an AliasProductHolder, in which
    // case it resolves the module which actually produces the product and returns
    // its module label
    std::string const& resolvedModuleLabel() const {return resolvedModuleLabel_();}

    // Retrieves a reference to the product instance name
    std::string const& productInstanceName() const {return branchDescription().productInstanceName();}

    // Retrieves a reference to the process name
    std::string const& processName() const {return branchDescription().processName();}

    // Retrieves pointer to a class containing both the event independent and the per even provenance.
    Provenance* provenance() const;

    // Initializes the event independent portion of the provenance, plus the process history ID, the product ID, and the mapper.
    void setProvenance(boost::shared_ptr<BranchMapper> mapper, ProcessHistoryID const& phid, ProductID const& pid) { setProvenance_(mapper, phid, pid); }

    // Initializes the process history ID.
    void setProcessHistoryID(ProcessHistoryID const& phid) { setProcessHistoryID_(phid); }

    // Write the product to the stream.
    void write(std::ostream& os) const;

    // Return the type of the product stored in this ProductHolder.
    // We are relying on the fact that Type instances are small, and
    // so we are free to copy them at will.
    TypeID productType() const;

    // Retrieves the product ID of the product.
    ProductID const& productID() const {return getProductData().prov_.productID();}

    // Puts the product and its per event(lumi)(run) provenance into the ProductHolder.
    void putProduct(WrapperOwningHolder const& edp, ProductProvenance const& productProvenance) {
      putProduct_(edp, productProvenance);
    }

    // Puts the product into the ProductHolder.
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

    void swap(ProductHolderBase& rhs) {swap_(rhs);}

    void throwProductDeletedException() const;

  private:
    virtual ProductData const& getProductData() const = 0;
    virtual ProductData& getProductData() = 0;
    virtual ProductData const* resolveProduct_(ResolveStatus& resolveStatus, bool skipCurrentProcess) const = 0;
    virtual void swap_(ProductHolderBase& rhs) = 0;
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
    virtual std::string const& resolvedModuleLabel_() const = 0;
    virtual void setProvenance_(boost::shared_ptr<BranchMapper> mapper, ProcessHistoryID const& phid, ProductID const& pid) = 0;
    virtual void setProcessHistoryID_(ProcessHistoryID const& phid) = 0;
    virtual ProductProvenance* productProvenancePtr_() const = 0;
    virtual void resetProductData_() = 0;
    virtual bool singleProduct_() const = 0;
    virtual void setPrincipal_(Principal* principal) = 0;
  };

  inline
  std::ostream&
  operator<<(std::ostream& os, ProductHolderBase const& phb) {
    phb.write(os);
    return os;
  }

  class InputProductHolder : public ProductHolderBase {
    public:
    explicit InputProductHolder(boost::shared_ptr<ConstBranchDescription> bd, Principal* principal) :
        ProductHolderBase(), productData_(bd), productIsUnavailable_(false),
        productHasBeenDeleted_(false), principal_(principal) {}
      virtual ~InputProductHolder();

      // The following is const because we can add an EDProduct to the
      // cache after creation of the ProductHolder, without changing the meaning
      // of the ProductHolder.
      void setProduct(WrapperOwningHolder const& prod) const;
      bool productIsUnavailable() const {return productIsUnavailable_;}
      void setProductUnavailable() const {productIsUnavailable_ = true;}

    private:
      virtual void swap_(ProductHolderBase& rhs) {
        InputProductHolder& other = dynamic_cast<InputProductHolder&>(rhs);
        edm::swap(productData_, other.productData_);
        std::swap(productIsUnavailable_, other.productIsUnavailable_);
      }
      virtual ProductData const* resolveProduct_(ResolveStatus& resolveStatus, bool skipCurrentProcess) const;
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
      virtual std::string const& resolvedModuleLabel_() const {return moduleLabel();}
      virtual void setProvenance_(boost::shared_ptr<BranchMapper> mapper, ProcessHistoryID const& phid, ProductID const& pid);
      virtual void setProcessHistoryID_(ProcessHistoryID const& phid);
      virtual ProductProvenance* productProvenancePtr_() const;
      virtual void resetProductData_();
      virtual bool singleProduct_() const;
      virtual void setPrincipal_(Principal* principal);

      ProductData productData_;
      mutable bool productIsUnavailable_;
      mutable bool productHasBeenDeleted_;
      Principal* principal_;
  };

  // Free swap function
  inline void swap(InputProductHolder& a, InputProductHolder& b) {
    a.swap(b);
  }

  class ProducedProductHolder : public ProductHolderBase {
    public:
    enum ProductStatus {
      Present = 0,
      NotRun = 3,
      NotCompleted = 4,
      NotPut = 5,
      UnscheduledNotRun = 6,
      ProductDeleted =7,
      Uninitialized = 0xff
    };
      ProducedProductHolder() : ProductHolderBase() {}
      virtual ~ProducedProductHolder();
      void producerStarted();
      void producerCompleted();
      ProductStatus& status() const {return status_();}
    private:
      virtual void putProduct_(WrapperOwningHolder const& edp, ProductProvenance const& productProvenance);
      virtual void putProduct_(WrapperOwningHolder const& edp) const;
      virtual void mergeProduct_(WrapperOwningHolder const& edp, ProductProvenance& productProvenance);
      virtual void mergeProduct_(WrapperOwningHolder const& edp) const;
      virtual bool putOrMergeProduct_() const;
      virtual void checkType_(WrapperOwningHolder const& prod) const {
        reallyCheckType(prod);
      }
      virtual ProductStatus& status_() const = 0;
      virtual bool productUnavailable_() const;
      virtual bool productWasDeleted_() const;
      virtual void setProductDeleted_();
      virtual ConstBranchDescription const& branchDescription_() const {return *productData().branchDescription();}
      virtual void resetBranchDescription_(boost::shared_ptr<ConstBranchDescription> bd) {productData().resetBranchDescription(bd);}
      virtual std::string const& resolvedModuleLabel_() const {return moduleLabel();}
      virtual void setProvenance_(boost::shared_ptr<BranchMapper> mapper, ProcessHistoryID const& phid, ProductID const& pid);
      virtual void setProcessHistoryID_(ProcessHistoryID const& phid);
      virtual ProductProvenance* productProvenancePtr_() const;
      virtual void resetProductData_();
      virtual bool singleProduct_() const;
      virtual void setPrincipal_(Principal* principal);
  };

  class ScheduledProductHolder : public ProducedProductHolder {
    public:
      explicit ScheduledProductHolder(boost::shared_ptr<ConstBranchDescription> bd) : ProducedProductHolder(), productData_(bd), theStatus_(NotRun) {}
      virtual ~ScheduledProductHolder();
    private:
      virtual void swap_(ProductHolderBase& rhs) {
        ScheduledProductHolder& other = dynamic_cast<ScheduledProductHolder&>(rhs);
        edm::swap(productData_, other.productData_);
        std::swap(theStatus_, other.theStatus_);
      }
      virtual ProductData const* resolveProduct_(ResolveStatus& resolveStatus, bool skipCurrentProcess) const;
      virtual void resetStatus_() {theStatus_ = NotRun;}
      virtual bool onDemand_() const {return false;}
      virtual ProductData const& getProductData() const {return productData_;}
      virtual ProductData& getProductData() {return productData_;}
      virtual ProductStatus& status_() const {return theStatus_;}

      ProductData productData_;
      mutable ProductStatus theStatus_;
  };

  // Free swap function
  inline void swap(ScheduledProductHolder& a, ScheduledProductHolder& b) {
    a.swap(b);
  }

  class UnscheduledProductHolder : public ProducedProductHolder {
    public:
      explicit UnscheduledProductHolder(boost::shared_ptr<ConstBranchDescription> bd, Principal* principal) :
        ProducedProductHolder(), productData_(bd), theStatus_(UnscheduledNotRun), principal_(principal) {}
      virtual ~UnscheduledProductHolder();
    private:
      virtual void swap_(ProductHolderBase& rhs) {
        UnscheduledProductHolder& other = dynamic_cast<UnscheduledProductHolder&>(rhs);
        edm::swap(productData_, other.productData_);
        std::swap(theStatus_, other.theStatus_);
      }
      virtual ProductData const* resolveProduct_(ResolveStatus& resolveStatus, bool skipCurrentProcess) const;
      virtual void resetStatus_() {theStatus_ = UnscheduledNotRun;}
      virtual bool onDemand_() const {return status() == UnscheduledNotRun;}
      virtual ProductData const& getProductData() const {return productData_;}
      virtual ProductData& getProductData() {return productData_;}
      virtual ProductStatus& status_() const {return theStatus_;}

      ProductData productData_;
      mutable ProductStatus theStatus_;
      Principal* principal_;
  };

  // Free swap function
  inline void swap(UnscheduledProductHolder& a, UnscheduledProductHolder& b) {
    a.swap(b);
  }

  class SourceProductHolder : public ProducedProductHolder {
    public:
      explicit SourceProductHolder(boost::shared_ptr<ConstBranchDescription> bd) : ProducedProductHolder(), productData_(bd), theStatus_(NotPut) {}
      virtual ~SourceProductHolder();
    private:
      virtual void swap_(ProductHolderBase& rhs) {
        SourceProductHolder& other = dynamic_cast<SourceProductHolder&>(rhs);
        edm::swap(productData_, other.productData_);
        std::swap(theStatus_, other.theStatus_);
      }
      virtual ProductData const* resolveProduct_(ResolveStatus& resolveStatus, bool skipCurrentProcess) const;
      virtual void resetStatus_() {theStatus_ = NotPut;}
      virtual bool onDemand_() const {return false;}
      virtual ProductData const& getProductData() const {return productData_;}
      virtual ProductData& getProductData() {return productData_;}
      virtual ProductStatus& status_() const {return theStatus_;}

      ProductData productData_;
      mutable ProductStatus theStatus_;
  };

  class AliasProductHolder : public ProductHolderBase {
    public:
      typedef ProducedProductHolder::ProductStatus ProductStatus;
      explicit AliasProductHolder(boost::shared_ptr<ConstBranchDescription> bd, ProducedProductHolder& realProduct) : ProductHolderBase(), realProduct_(realProduct), bd_(bd) {}
      virtual ~AliasProductHolder();
    private:
      virtual void swap_(ProductHolderBase& rhs) {
        AliasProductHolder& other = dynamic_cast<AliasProductHolder&>(rhs);
        realProduct_.swap(other.realProduct_);
        std::swap(bd_, other.bd_);
      }
      virtual ProductData const* resolveProduct_(ResolveStatus& resolveStatus, bool skipCurrentProcess) const {return realProduct_.resolveProduct(resolveStatus, skipCurrentProcess);}
      virtual bool onDemand_() const {return realProduct_.onDemand();}
      virtual ProductStatus& status_() const {return realProduct_.status();}
      virtual void resetStatus_() {realProduct_.resetStatus();}
      virtual bool productUnavailable_() const {return realProduct_.productUnavailable();}
      virtual bool productWasDeleted_() const {return realProduct_.productWasDeleted();}
      virtual void checkType_(WrapperOwningHolder const& prod) const {realProduct_.checkType(prod);}
      virtual ProductData const& getProductData() const {return realProduct_.productData();}
      virtual ProductData& getProductData() {return realProduct_.productData();}
      virtual void setProductDeleted_() {realProduct_.setProductDeleted();}
      virtual void putProduct_(WrapperOwningHolder const& edp, ProductProvenance const& productProvenance) {
        realProduct_.putProduct(edp, productProvenance);
      }
      virtual void putProduct_(WrapperOwningHolder const& edp) const {
        realProduct_.putProduct(edp);
      }
      virtual void mergeProduct_(WrapperOwningHolder const& edp, ProductProvenance& productProvenance) {
        realProduct_.mergeProduct(edp, productProvenance);
      }
      virtual void mergeProduct_(WrapperOwningHolder const& edp) const {
        realProduct_.mergeProduct(edp);
      }
      virtual bool putOrMergeProduct_() const {
        return realProduct_.putOrMergeProduct();
      }
      virtual ConstBranchDescription const& branchDescription_() const {return *bd_;}
      virtual void resetBranchDescription_(boost::shared_ptr<ConstBranchDescription> bd) {bd_ = bd;}
      virtual std::string const& resolvedModuleLabel_() const {return realProduct_.moduleLabel();}
      virtual void setProvenance_(boost::shared_ptr<BranchMapper> mapper, ProcessHistoryID const& phid, ProductID const& pid);
      virtual void setProcessHistoryID_(ProcessHistoryID const& phid);
      virtual ProductProvenance* productProvenancePtr_() const;
      virtual void resetProductData_();
      virtual bool singleProduct_() const;
      virtual void setPrincipal_(Principal* principal);

      ProducedProductHolder& realProduct_;
      boost::shared_ptr<ConstBranchDescription> bd_;
  };

  class NoProcessProductHolder : public ProductHolderBase {
    public:
      typedef ProducedProductHolder::ProductStatus ProductStatus;
      NoProcessProductHolder(std::vector<ProductHolderIndex> const&  matchingHolders,
                             std::vector<bool> const& ambiguous,
                             Principal* principal);
      virtual ~NoProcessProductHolder();
    private:
      virtual ProductData const& getProductData() const;
      virtual ProductData& getProductData();
      virtual ProductData const* resolveProduct_(ResolveStatus& resolveStatus, bool skipCurrentProcess) const;
      virtual void swap_(ProductHolderBase& rhs);
      virtual bool onDemand_() const;
      virtual bool productUnavailable_() const;
      virtual bool productWasDeleted_() const;
      virtual void putProduct_(WrapperOwningHolder const& edp, ProductProvenance const& productProvenance);
      virtual void putProduct_(WrapperOwningHolder const& edp) const;
      virtual void mergeProduct_(WrapperOwningHolder const&  edp, ProductProvenance& productProvenance);
      virtual void mergeProduct_(WrapperOwningHolder const& edp) const;
      virtual bool putOrMergeProduct_() const;
      virtual void checkType_(WrapperOwningHolder const& prod) const;
      virtual void resetStatus_();
      virtual void setProductDeleted_();
      virtual ConstBranchDescription const& branchDescription_() const;
      virtual void resetBranchDescription_(boost::shared_ptr<ConstBranchDescription> bd);
      virtual std::string const& resolvedModuleLabel_() const {return moduleLabel();}
      virtual void setProvenance_(boost::shared_ptr<BranchMapper> mapper, ProcessHistoryID const& phid, ProductID const& pid);
      virtual void setProcessHistoryID_(ProcessHistoryID const& phid);
      virtual ProductProvenance* productProvenancePtr_() const;
      virtual void resetProductData_();
      virtual bool singleProduct_() const;
      virtual void setPrincipal_(Principal* principal);

      std::vector<ProductHolderIndex> matchingHolders_;
      std::vector<bool> ambiguous_;
      Principal* principal_;
  };

  // Free swap function
  inline void swap(SourceProductHolder& a, SourceProductHolder& b) {
    a.swap(b);
  }
}

#endif
