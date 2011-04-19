#ifndef Framework_Run_h
#define Framework_Run_h

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

#include "DataFormats/Common/interface/WrapperHolder.h"
#include "FWCore/Framework/interface/PrincipalGetAdapter.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Common/interface/RunBase.h"

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
    get(SelectorBase const&, Handle<PROD>& result) const;

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

    template <typename PROD>
    void
    getMany(SelectorBase const&, std::vector<Handle<PROD> >& results) const;

    template <typename PROD>
    bool
    getByType(Handle<PROD>& result) const;

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

    ProcessHistory const&
    processHistory() const;

  private:
    RunPrincipal const&
    runPrincipal() const;

    RunPrincipal&
    runPrincipal();

    // Override version from RunBase class
    virtual BasicHandle getByLabelImpl(WrapperInterfaceBase const* wrapperInterfaceBase, std::type_info const& iWrapperType, std::type_info const& iProductType, InputTag const& iTag) const;

    typedef std::vector<std::pair<WrapperHolder, ConstBranchDescription const*> > ProductPtrVec;
    ProductPtrVec& putProducts() {return putProducts_;}
    ProductPtrVec const& putProducts() const {return putProducts_;}

    // commit_() is called to complete the transaction represented by
    // this PrincipalGetAdapter. The friendships required seems gross, but any
    // alternative is not great either.  Putting it into the
    // public interface is asking for trouble
    friend class ConfigurableInputSource;
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

    WrapperInterfaceBase const* interface = Wrapper<PROD>::getInterface();
    boost::shared_ptr<void const> wp(new Wrapper<PROD>(product), WrapperHolder::EDProductDeleter(interface));
    WrapperHolder edp(wp, interface);
    putProducts().push_back(std::make_pair(edp, &desc));

    // product.release(); // The object has been copied into the Wrapper.
    // The old copy must be deleted, so we cannot release ownership.
  }

  template <typename PROD>
  bool
  Run::get(SelectorBase const& sel, Handle<PROD>& result) const {
    return provRecorder_.get(sel, result);
  }

  template <typename PROD>
  bool
  Run::getByLabel(std::string const& label, Handle<PROD>& result) const {
    return provRecorder_.getByLabel(label, result);
  }

  template <typename PROD>
  bool
  Run::getByLabel(std::string const& label,
                  std::string const& productInstanceName,
                  Handle<PROD>& result) const {
    return provRecorder_.getByLabel(label, productInstanceName, result);
  }

  /// same as above, but using the InputTag class
  template <typename PROD>
  bool
  Run::getByLabel(InputTag const& tag, Handle<PROD>& result) const {
    return provRecorder_.getByLabel(tag, result);
  }

  template <typename PROD>
  void
  Run::getMany(SelectorBase const& sel, std::vector<Handle<PROD> >& results) const {
    return provRecorder_.getMany(sel, results);
  }

  template <typename PROD>
  bool
  Run::getByType(Handle<PROD>& result) const {
    return provRecorder_.getByType(result);
  }

  template <typename PROD>
  void
  Run::getManyByType(std::vector<Handle<PROD> >& results) const {
    return provRecorder_.getManyByType(results);
  }

}
#endif
