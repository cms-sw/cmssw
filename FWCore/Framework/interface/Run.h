#ifndef FWCore_Framework_Run_h
#define FWCore_Framework_Run_h

// -*- C++ -*-
//
// Package:     Framework
// Class  :     Run
//
/**\class Run Run.h FWCore/Framework/interface/Run.h

Description: This is the primary interface for accessing per run EDProducts and inserting new derived products.

For its usage, see "FWCore/Framework/interface/PrincipalGetAdapter.h"

*/
/*----------------------------------------------------------------------

----------------------------------------------------------------------*/

#include "DataFormats/Common/interface/WrapperOwningHolder.h"
#include "FWCore/Framework/interface/PrincipalGetAdapter.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Common/interface/RunBase.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/ProductKindOfType.h"

#include <memory>
#include <set>
#include <string>
#include <typeinfo>
#include <vector>

namespace edm {

  class Run : public RunBase {
  public:
    Run(RunPrincipal& rp, ModuleDescription const& md);
    ~Run();

    //Used in conjunction with EDGetToken
    void setConsumer(EDConsumerBase const* iConsumer) {
      provRecorder_.setConsumer(iConsumer);
    }
    

    typedef PrincipalGetAdapter Base;
    // AUX functions are defined in RunBase
    RunAuxiliary const& runAuxiliary() const {return aux_;}
    // AUX functions.
//     RunID const& id() const {return aux_.id();}
//     RunNumber_t run() const {return aux_.run();}
//     Timestamp const& beginTime() const {return aux_.beginTime();}
//     Timestamp const& endTime() const {return aux_.endTime();}

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

    // Return true if this Run has been subjected to a process with
    // the given processName, and false otherwise.
    // If true is returned, then ps is filled with the ParameterSets
    // (possibly more than one) used to configure the identified
    // process(es). Equivalent ParameterSets are compressed out of the
    // result.
    // This function is not yet implemented in full.
    //bool
    //getProcessParameterSet(std::string const& processName, std::vector<ParameterSet>& ps) const;

    ProcessHistoryID const& processHistoryID() const;

    ProcessHistory const&
    processHistory() const;

  private:
    RunPrincipal const&
    runPrincipal() const;

    RunPrincipal&
    runPrincipal();

    // Override version from RunBase class
    virtual BasicHandle getByLabelImpl(std::type_info const& iWrapperType, std::type_info const& iProductType, InputTag const& iTag) const;

    typedef std::vector<std::pair<WrapperOwningHolder, ConstBranchDescription const*> > ProductPtrVec;
    ProductPtrVec& putProducts() {return putProducts_;}
    ProductPtrVec const& putProducts() const {return putProducts_;}

    // commit_() is called to complete the transaction represented by
    // this PrincipalGetAdapter. The friendships required seems gross, but any
    // alternative is not great either.  Putting it into the
    // public interface is asking for trouble
    friend class DaqSource;
    friend class InputSource;
    friend class RawInputSource;
    friend class EDFilter;
    friend class EDProducer;

    void commit_();
    void addToGotBranchIDs(Provenance const& prov) const;

    PrincipalGetAdapter provRecorder_;
    ProductPtrVec putProducts_;
    RunAuxiliary const& aux_;
    typedef std::set<BranchID> BranchIDSet;
    mutable BranchIDSet gotBranchIDs_;

    static const std::string emptyString_;
  };

  template <typename PROD>
  void
  Run::put(std::auto_ptr<PROD> product, std::string const& productInstanceName) {
    if (product.get() == 0) {                // null pointer is illegal
      TypeID typeID(typeid(PROD));
      principal_get_adapter_detail::throwOnPutOfNullProduct("Run", typeID, productInstanceName);
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

  template <typename PROD>
  bool
  Run::getByLabel(std::string const& label, Handle<PROD>& result) const {
    return getByLabel(label, emptyString_, result);
  }

  template <typename PROD>
  bool
  Run::getByLabel(std::string const& label,
                  std::string const& productInstanceName,
                  Handle<PROD>& result) const {
    if(!provRecorder_.checkIfComplete<PROD>()) {
      principal_get_adapter_detail::throwOnPrematureRead("Run", TypeID(typeid(PROD)), label, productInstanceName);
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
  template <typename PROD>
  bool
  Run::getByLabel(InputTag const& tag, Handle<PROD>& result) const {
    if(!provRecorder_.checkIfComplete<PROD>()) {
      principal_get_adapter_detail::throwOnPrematureRead("Run", TypeID(typeid(PROD)), tag.label(), tag.instance());
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
  Run::getByToken(EDGetToken token, Handle<PROD>& result) const {
    if(!provRecorder_.checkIfComplete<PROD>()) {
      principal_get_adapter_detail::throwOnPrematureRead("Run", TypeID(typeid(PROD)), token);
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
  Run::getByToken(EDGetTokenT<PROD> token, Handle<PROD>& result) const {
    if(!provRecorder_.checkIfComplete<PROD>()) {
      principal_get_adapter_detail::throwOnPrematureRead("Run", TypeID(typeid(PROD)), token);
    }
    result.clear();
    BasicHandle bh = provRecorder_.getByToken_(TypeID(typeid(PROD)),PRODUCT_TYPE, token);
    convert_handle(bh, result);  // throws on conversion error
    if (bh.failedToGet()) {
      return false;
    }
    return true;
  }

  template <typename PROD>
  void
  Run::getManyByType(std::vector<Handle<PROD> >& results) const {
    if(!provRecorder_.checkIfComplete<PROD>()) {
      principal_get_adapter_detail::throwOnPrematureRead("Run", TypeID(typeid(PROD)));
    }
    return provRecorder_.getManyByType(results);
  }

}
#endif
