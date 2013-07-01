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
#include "DataFormats/Common/interface/WrapperOwningHolder.h"
#include "FWCore/Common/interface/LuminosityBlockBase.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/PrincipalGetAdapter.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/ProductKindOfType.h"
#include "FWCore/Utilities/interface/LuminosityBlockIndex.h"


#include "boost/shared_ptr.hpp"

#include <memory>
#include <set>
#include <string>
#include <typeinfo>
#include <vector>

namespace edm {
  class ProducerBase;

  class LuminosityBlock : public LuminosityBlockBase {
  public:
    LuminosityBlock(LuminosityBlockPrincipal& lbp, ModuleDescription const& md);
    ~LuminosityBlock();

    // AUX functions are defined in LuminosityBlockBase
    LuminosityBlockAuxiliary const& luminosityBlockAuxiliary() const {return aux_;}

    /**\return Reusable index which can be used to separate data for different simultaneous LuminosityBlocks.
     */
    LuminosityBlockIndex index() const;
    
    //Used in conjunction with EDGetToken
    void setConsumer(EDConsumerBase const* iConsumer);
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
    put(std::auto_ptr<PROD> product) {put<PROD>(product, std::string());}

    ///Put a new product with a 'product instance name'
    template <typename PROD>
    void
    put(std::auto_ptr<PROD> product, std::string const& productInstanceName);

    Provenance
    getProvenance(BranchID const& theID) const;

    void
    getAllProvenance(std::vector<Provenance const*>& provenances) const;

    ProcessHistoryID const& processHistoryID() const;

    ProcessHistory const&
    processHistory() const;

  private:
    LuminosityBlockPrincipal const&
    luminosityBlockPrincipal() const;

    LuminosityBlockPrincipal&
    luminosityBlockPrincipal();

    // Override version from LuminosityBlockBase class
    virtual BasicHandle getByLabelImpl(std::type_info const& iWrapperType, std::type_info const& iProductType, InputTag const& iTag) const;

    typedef std::vector<std::pair<WrapperOwningHolder, ConstBranchDescription const*> > ProductPtrVec;
    ProductPtrVec& putProducts() {return putProducts_;}
    ProductPtrVec const& putProducts() const {return putProducts_;}

    // commit_() is called to complete the transaction represented by
    // this PrincipalGetAdapter. The friendships required seems gross, but any
    // alternative is not great either.  Putting it into the
    // public interface is asking for trouble
    friend class InputSource;
    friend class DaqSource;
    friend class RawInputSource;
    friend class ProducerBase;

    void commit_();
    void addToGotBranchIDs(Provenance const& prov) const;

    PrincipalGetAdapter provRecorder_;
    ProductPtrVec putProducts_;
    LuminosityBlockAuxiliary const& aux_;
    boost::shared_ptr<Run const> const run_;
    typedef std::set<BranchID> BranchIDSet;
    mutable BranchIDSet gotBranchIDs_;

    static const std::string emptyString_;
  };

  template <typename PROD>
  void
  LuminosityBlock::put(std::auto_ptr<PROD> product, std::string const& productInstanceName) {
    if(product.get() == 0) {                // null pointer is illegal
      TypeID typeID(typeid(PROD));
      principal_get_adapter_detail::throwOnPutOfNullProduct("LuminosityBlock", typeID, productInstanceName);
    }

    // The following will call post_insert if T has such a function,
    // and do nothing if T has no such function.
    typename boost::mpl::if_c<detail::has_postinsert<PROD>::value,
      DoPostInsert<PROD>,
      DoNotPostInsert<PROD> >::type maybe_inserter;
    maybe_inserter(product.get());

    ConstBranchDescription const& desc =
      provRecorder_.getBranchDescription(TypeID(*product), productInstanceName);

    WrapperOwningHolder edp(new Wrapper<PROD>(product), Wrapper<PROD>::getInterface());
    putProducts().emplace_back(edp, &desc);

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
    BasicHandle bh = provRecorder_.getByLabel_(TypeID(typeid(PROD)), label, productInstanceName, emptyString_);
    convert_handle(bh, result);  // throws on conversion error
    if (bh.failedToGet()) {
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
    BasicHandle bh = provRecorder_.getByLabel_(TypeID(typeid(PROD)), tag);
    convert_handle(bh, result);  // throws on conversion error
    if (bh.failedToGet()) {
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
    BasicHandle bh = provRecorder_.getByToken_(TypeID(typeid(PROD)),PRODUCT_TYPE, token);
    convert_handle(bh, result);  // throws on conversion error
    if (bh.failedToGet()) {
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
    BasicHandle bh = provRecorder_.getByToken_(TypeID(typeid(PROD)),PRODUCT_TYPE, token);
    convert_handle(bh, result);  // throws on conversion error
    if (bh.failedToGet()) {
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
    return provRecorder_.getManyByType(results);
  }

}
#endif
