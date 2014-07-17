#ifndef FWCore_Common_LuminosityBlockBase_h
#define FWCore_Common_LuminosityBlockBase_h

// -*- C++ -*-
//
// Package:     FWCore/Common
// Class  :     LuminosityBlockBase
//
/**\class LuminosityBlockBase LuminosityBlockBase.h FWCore/Common/interface/LuminosityBlockBase.h

 Description: Base class for LuminosityBlocks in both the full and light framework

 Usage:
    One can use this class for code which needs to work in both the full and the
 light (i.e. FWLite) frameworks.  Data can be accessed using the same getByLabel
 interface which is available in the full framework.

*/
//
// Original Author:  Eric Vaandering
//         Created:  Tue Jan 12 15:31:00 CDT 2010
//

#if !defined(__CINT__) && !defined(__MAKECINT__)

#include "DataFormats/Common/interface/BasicHandle.h"
#include "DataFormats/Common/interface/ConvertHandle.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Provenance/interface/LuminosityBlockAuxiliary.h"
#include "DataFormats/Provenance/interface/LuminosityBlockID.h"
#include "DataFormats/Provenance/interface/RunID.h"
#include "FWCore/Utilities/interface/InputTag.h"

//#include "FWCore/Framework/interface/PrincipalGetAdapter.h"
//#include "FWCore/Framework/interface/Frameworkfwd.h"

namespace edm {

  class LuminosityBlockBase {
  public:
    LuminosityBlockBase();
    virtual ~LuminosityBlockBase();

    // AUX functions.
    LuminosityBlockNumber_t luminosityBlock() const {
      return luminosityBlockAuxiliary().luminosityBlock();
    }

    RunNumber_t run() const {
      return luminosityBlockAuxiliary().run();
    }

    LuminosityBlockID id() const {
      return luminosityBlockAuxiliary().id();
    }

    Timestamp const& beginTime() const {
      return luminosityBlockAuxiliary().beginTime();
    }
    Timestamp const& endTime() const {
      return luminosityBlockAuxiliary().endTime();
    }

    virtual edm::LuminosityBlockAuxiliary const& luminosityBlockAuxiliary() const = 0;

/*    template<typename PROD>
    bool
    get(SelectorBase const&, Handle<PROD>& result) const;

    template<typename PROD>
    bool
    getByLabel(std::string const& label, Handle<PROD>& result) const;

    template<typename PROD>
    bool
    getByLabel(std::string const& label,
               std::string const& productInstanceName,
               Handle<PROD>& result) const;
*/
    /// same as above, but using the InputTag class
    template<typename PROD>
    bool
    getByLabel(InputTag const& tag, Handle<PROD>& result) const;
/*
    template<typename PROD>
    void
    getMany(SelectorBase const&, std::vector<Handle<PROD> >& results) const;

    template<typename PROD>
    void
    getManyByType(std::vector<Handle<PROD> >& results) const;

    Run const&
    getRun() const {
      return *run_;
    }

    ///Put a new product.
    template<typename PROD>
    void
    put(std::auto_ptr<PROD> product) {put<PROD>(product, std::string());}

    ///Put a new product with a 'product instance name'
    template<typename PROD>
    void
    put(std::auto_ptr<PROD> product, std::string const& productInstanceName);

    Provenance
    getProvenance(BranchID const& theID) const;

    void
    getAllProvenance(std::vector<Provenance const*>& provenances) const;

    ProcessHistory const&
    processHistory() const;
*/
  private:
/*    LuminosityBlockPrincipal const&
    luminosityBlockPrincipal() const;

    LuminosityBlockPrincipal&
    luminosityBlockPrincipal();

    typedef std::vector<std::pair<EDProduct*, BranchDescription const*> > ProductPtrVec;
    ProductPtrVec& putProducts() {return putProducts_;}
    ProductPtrVec const& putProducts() const {return putProducts_;}

    // commit_() is called to complete the transaction represented by
    // this PrincipalGetAdapter. The friendships required seems gross, but any
    // alternative is not great either.  Putting it into the
    // public interface is asking for trouble
    friend class ConfigurableInputSource;
    friend class InputSource;
    friend class RawInputSource;
    friend class EDFilter;
    friend class EDProducer;

    void commit_();

    PrincipalGetAdapter provRecorder_;
    ProductPtrVec putProducts_;
    LuminosityBlockAuxiliary const& aux_;
    std::shared_ptr<Run const> const run_;
*/
    virtual BasicHandle getByLabelImpl(std::type_info const& iWrapperType, std::type_info const& iProductType, const InputTag& iTag) const = 0;

  };
/*
  template<typename PROD>
  void
  LuminosityBlock::put(std::auto_ptr<PROD> product, std::string const& productInstanceName) {
    if (product.get() == 0) {                // null pointer is illegal
      TypeID typeID(typeid(PROD));
      principal_get_adapter_detail::throwOnPutOfNullProduct("LuminosityBlock", typeID, productInstanceName);
    }

    // The following will call post_insert if T has such a function,
    // and do nothing if T has no such function.
    typename boost::mpl::if_c<detail::has_postinsert<PROD>::value,
      DoPostInsert<PROD>,
      DoNotPostInsert<PROD> >::type maybe_inserter;
    maybe_inserter(product.get());

    BranchDescription const& desc =
      provRecorder_.getBranchDescription(TypeID(*product), productInstanceName);

    Wrapper<PROD> *wp(new Wrapper<PROD>(product));
    WrapperHolder edp(wp, wp->getInterface());

    putProducts().push_back(std::make_pair(edp, &desc));

    // product.release(); // The object has been copied into the Wrapper.
    // The old copy must be deleted, so we cannot release ownership.
  }

  template<typename PROD>
  bool
  LuminosityBlock::get(SelectorBase const& sel, Handle<PROD>& result) const {
    return provRecorder_.get(sel,result);
  }

  template<typename PROD>
  bool
  LuminosityBlock::getByLabel(std::string const& label, Handle<PROD>& result) const {
    return provRecorder_.getByLabel(label,result);
  }

  template<typename PROD>
  bool
  LuminosityBlock::getByLabel(std::string const& label,
                  std::string const& productInstanceName,
                  Handle<PROD>& result) const {
    return provRecorder_.getByLabel(label,productInstanceName,result);
  }

  /// same as above, but using the InputTag class
  template<typename PROD>
  bool
  LuminosityBlock::getByLabel(InputTag const& tag, Handle<PROD>& result) const {
    return provRecorder_.getByLabel(tag,result);
  }

  template<typename PROD>
  void
  LuminosityBlock::getMany(SelectorBase const& sel, std::vector<Handle<PROD> >& results) const {
    return provRecorder_.getMany(sel,results);
  }

  template<typename PROD>
  void
  LuminosityBlock::getManyByType(std::vector<Handle<PROD> >& results) const {
    return provRecorder_.getManyByType(results);
  }
*/
#if !defined(__REFLEX__)
   template<class T>
   bool
   LuminosityBlockBase::getByLabel(const InputTag& tag, Handle<T>& result) const {
      result.clear();
      BasicHandle bh = this->getByLabelImpl(typeid(Wrapper<T>), typeid(T), tag);
      convert_handle(std::move(bh), result);  // throws on conversion error
      if (result.failedToGet()) {
         return false;
      }
      return true;
   }
#endif

}
#endif /*!defined(__CINT__) && !defined(__MAKECINT__)*/
#endif
