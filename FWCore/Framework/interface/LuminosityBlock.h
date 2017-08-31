#ifndef FWCore_Framework_LuminosityBlock_h
#define FWCore_Framework_LuminosityBlock_h

// -*- C++ -*-
//
// Package:     Framework
// Class  :     LuminosityBlock
//
/**\class LuminosityBlock LuminosityBlock.h FWCore/Framework/interface/LuminosityBlock.h

Description: This is the primary interface for accessing per luminosity block EDProducts
and inserting new derived per luminosity block EDProducts.

For its usage, see "FWCore/Framework/interface/PrincipalGetAdapter.h"

*/
/*----------------------------------------------------------------------

----------------------------------------------------------------------*/

#include "DataFormats/Common/interface/Wrapper.h"
#include "FWCore/Common/interface/LuminosityBlockBase.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/PrincipalGetAdapter.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/ProductKindOfType.h"
#include "FWCore/Utilities/interface/LuminosityBlockIndex.h"
#include "FWCore/Utilities/interface/propagate_const.h"

#include <memory>
#include <string>
#include <typeinfo>
#include <vector>

namespace edm {
  class ModuleCallingContext;
  class ProducerBase;
  class SharedResourcesAcquirer;

  namespace stream {
    template< typename T> class ProducingModuleAdaptorBase;
  }


  class LuminosityBlock : public LuminosityBlockBase {
  public:
    LuminosityBlock(LuminosityBlockPrincipal const& lbp, ModuleDescription const& md,
                    ModuleCallingContext const*);
    ~LuminosityBlock() override;

    // AUX functions are defined in LuminosityBlockBase
    LuminosityBlockAuxiliary const& luminosityBlockAuxiliary() const override {return aux_;}

    /**\return Reusable index which can be used to separate data for different simultaneous LuminosityBlocks.
     */
    LuminosityBlockIndex index() const;

    /**If you are caching data from the LuminosityBlock, you should also keep
     this number.  If this number changes then you know that
     the data you have cached is invalid.
     The value of '0' will never be returned so you can use that to
     denote that you have not yet checked the value.
     */
    typedef unsigned long CacheIdentifier_t;
    CacheIdentifier_t
    cacheIdentifier() const;

    //Used in conjunction with EDGetToken
    void setConsumer(EDConsumerBase const* iConsumer);

    void setSharedResourcesAcquirer( SharedResourcesAcquirer* iResourceAcquirer);

    template <typename PROD>
    bool
    getByLabel(std::string const& label, Handle<PROD>& result) const;

    template <typename PROD>
    bool
    getByLabel(std::string const& label,
               std::string const& productInstanceName,
               Handle<PROD>& result) const;

    /// same as above, but using the InputTag class
    template <typename PROD>
    bool
    getByLabel(InputTag const& tag, Handle<PROD>& result) const;

    template<typename PROD>
    bool
    getByToken(EDGetToken token, Handle<PROD>& result) const;

    template<typename PROD>
    bool
    getByToken(EDGetTokenT<PROD> token, Handle<PROD>& result) const;


    template <typename PROD>
    void
    getManyByType(std::vector<Handle<PROD> >& results) const;

    Run const&
    getRun() const {
      return *run_;
    }

    ///Put a new product.
    template <typename PROD>
    void
    put(std::unique_ptr<PROD> product) {put<PROD>(std::move(product), std::string());}

    ///Put a new product with a 'product instance name'
    template <typename PROD>
    void
    put(std::unique_ptr<PROD> product, std::string const& productInstanceName);

    Provenance
    getProvenance(BranchID const& theID) const;

    void
    getAllStableProvenance(std::vector<StableProvenance const*>& provenances) const;

    ProcessHistoryID const& processHistoryID() const;

    ProcessHistory const&
    processHistory() const;

    ModuleCallingContext const* moduleCallingContext() const { return moduleCallingContext_; }

    void labelsForToken(EDGetToken const& iToken, ProductLabels& oLabels) const { provRecorder_.labelsForToken(iToken, oLabels); }

  private:
    LuminosityBlockPrincipal const&
    luminosityBlockPrincipal() const;

    // Override version from LuminosityBlockBase class
    BasicHandle getByLabelImpl(std::type_info const& iWrapperType, std::type_info const& iProductType, InputTag const& iTag) const override;

    typedef std::vector<std::pair<edm::propagate_const<std::unique_ptr<WrapperBase>>, BranchDescription const*>> ProductPtrVec;
    ProductPtrVec& putProducts() {return putProducts_;}
    ProductPtrVec const& putProducts() const {return putProducts_;}

    // commit_() is called to complete the transaction represented by
    // this PrincipalGetAdapter. The friendships required seems gross, but any
    // alternative is not great either.  Putting it into the
    // public interface is asking for trouble
    friend class InputSource;
    friend class RawInputSource;
    friend class ProducerBase;
    template<typename T> friend class stream::ProducingModuleAdaptorBase;


    void commit_(std::vector<edm::ProductResolverIndex> const& iShouldPut);

    PrincipalGetAdapter provRecorder_;
    ProductPtrVec putProducts_;
    LuminosityBlockAuxiliary const& aux_;
    std::shared_ptr<Run const> const run_;
    ModuleCallingContext const* moduleCallingContext_;

    static const std::string emptyString_;
  };

  template <typename PROD>
  void
  LuminosityBlock::put(std::unique_ptr<PROD> product, std::string const& productInstanceName) {
    if(product.get() == nullptr) {                // null pointer is illegal
      TypeID typeID(typeid(PROD));
      principal_get_adapter_detail::throwOnPutOfNullProduct("LuminosityBlock", typeID, productInstanceName);
    }

    // The following will call post_insert if T has such a function,
    // and do nothing if T has no such function.
    std::conditional_t<detail::has_postinsert<PROD>::value,
                       DoPostInsert<PROD>,
                       DoNotPostInsert<PROD>> maybe_inserter;
    maybe_inserter(product.get());

    BranchDescription const& desc =
      provRecorder_.getBranchDescription(TypeID(*product), productInstanceName);

    std::unique_ptr<Wrapper<PROD> > wp(new Wrapper<PROD>(std::move(product)));
    putProducts().emplace_back(std::move(wp), &desc);

    // product.release(); // The object has been copied into the Wrapper.
    // The old copy must be deleted, so we cannot release ownership.
  }

  template<typename PROD>
  bool
  LuminosityBlock::getByLabel(std::string const& label, Handle<PROD>& result) const {
    return getByLabel(label, emptyString_, result);
  }

  template<typename PROD>
  bool
  LuminosityBlock::getByLabel(std::string const& label,
                  std::string const& productInstanceName,
                  Handle<PROD>& result) const {
    if(!provRecorder_.checkIfComplete<PROD>()) {
      principal_get_adapter_detail::throwOnPrematureRead("Lumi", TypeID(typeid(PROD)), label, productInstanceName);
    }
    result.clear();
    BasicHandle bh = provRecorder_.getByLabel_(TypeID(typeid(PROD)), label, productInstanceName, emptyString_, moduleCallingContext_);
    convert_handle(std::move(bh), result);  // throws on conversion error
    if (result.failedToGet()) {
      return false;
    }
    return true;
  }

  /// same as above, but using the InputTag class
  template<typename PROD>
  bool
  LuminosityBlock::getByLabel(InputTag const& tag, Handle<PROD>& result) const {
    if(!provRecorder_.checkIfComplete<PROD>()) {
      principal_get_adapter_detail::throwOnPrematureRead("Lumi", TypeID(typeid(PROD)), tag.label(), tag.instance());
    }
    result.clear();
    BasicHandle bh = provRecorder_.getByLabel_(TypeID(typeid(PROD)), tag, moduleCallingContext_);
    convert_handle(std::move(bh), result);  // throws on conversion error
    if (result.failedToGet()) {
      return false;
    }
    return true;
  }

  template<typename PROD>
  bool
  LuminosityBlock::getByToken(EDGetToken token, Handle<PROD>& result) const {
    if(!provRecorder_.checkIfComplete<PROD>()) {
      principal_get_adapter_detail::throwOnPrematureRead("Lumi", TypeID(typeid(PROD)), token);
    }
    result.clear();
    BasicHandle bh = provRecorder_.getByToken_(TypeID(typeid(PROD)),PRODUCT_TYPE, token, moduleCallingContext_);
    convert_handle(std::move(bh), result);  // throws on conversion error
    if (result.failedToGet()) {
      return false;
    }
    return true;
  }

  template<typename PROD>
  bool
  LuminosityBlock::getByToken(EDGetTokenT<PROD> token, Handle<PROD>& result) const {
    if(!provRecorder_.checkIfComplete<PROD>()) {
      principal_get_adapter_detail::throwOnPrematureRead("Lumi", TypeID(typeid(PROD)), token);
    }
    result.clear();
    BasicHandle bh = provRecorder_.getByToken_(TypeID(typeid(PROD)),PRODUCT_TYPE, token, moduleCallingContext_);
    convert_handle(std::move(bh), result);  // throws on conversion error
    if (result.failedToGet()) {
      return false;
    }
    return true;
  }


  template<typename PROD>
  void
  LuminosityBlock::getManyByType(std::vector<Handle<PROD> >& results) const {
    if(!provRecorder_.checkIfComplete<PROD>()) {
      principal_get_adapter_detail::throwOnPrematureRead("Lumi", TypeID(typeid(PROD)));
    }
    return provRecorder_.getManyByType(results, moduleCallingContext_);
  }

}
#endif
