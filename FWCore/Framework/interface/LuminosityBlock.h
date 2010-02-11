#ifndef Framework_LuminosityBlock_h
#define Framework_LuminosityBlock_h

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

#include "boost/shared_ptr.hpp"

#include "DataFormats/Provenance/interface/LuminosityBlockAuxiliary.h"
#include "DataFormats/Provenance/interface/LuminosityBlockID.h"
#include "DataFormats/Provenance/interface/RunID.h"

#include "FWCore/Framework/interface/PrincipalGetAdapter.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Common/interface/LuminosityBlockBase.h"

namespace edm {

  class LuminosityBlock : public LuminosityBlockBase {
  public:
    LuminosityBlock(LuminosityBlockPrincipal& lbp, ModuleDescription const& md);
    ~LuminosityBlock();

    // AUX functions are defined in LuminosityBlockBase
    LuminosityBlockAuxiliary const& luminosityBlockAuxiliary() const {return aux_;}

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

    ProcessHistory const&
    processHistory() const;

  private:
    LuminosityBlockPrincipal const&
    luminosityBlockPrincipal() const;

    LuminosityBlockPrincipal&
    luminosityBlockPrincipal();

    // Override version from LuminosityBlockBase class
    virtual BasicHandle getByLabelImpl(const std::type_info& iWrapperType, const std::type_info& iProductType, const InputTag& iTag) const;

    typedef std::vector<std::pair<EDProduct*, ConstBranchDescription const*> > ProductPtrVec;
    ProductPtrVec& putProducts() {return putProducts_;}
    ProductPtrVec const& putProducts() const {return putProducts_;}

    // commit_() is called to complete the transaction represented by
    // this PrincipalGetAdapter. The friendships required seems gross, but any
    // alternative is not great either.  Putting it into the
    // public interface is asking for trouble
    friend class ConfigurableInputSource;
    friend class InputSource;
    friend class DaqSource;
    friend class RawInputSource;
    friend class EDFilter;
    friend class EDProducer;

    void commit_();
    void addToGotBranchIDs(Provenance const& prov) const;

    PrincipalGetAdapter provRecorder_;
    ProductPtrVec putProducts_;
    LuminosityBlockAuxiliary const& aux_;
    boost::shared_ptr<Run const> const run_;
    typedef std::set<BranchID> BranchIDSet;
    mutable BranchIDSet gotBranchIDs_;
  };

  template <typename PROD>
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

    ConstBranchDescription const& desc =
      provRecorder_.getBranchDescription(TypeID(*product), productInstanceName);

    Wrapper<PROD> *wp(new Wrapper<PROD>(product));

    putProducts().push_back(std::make_pair(wp, &desc));

    // product.release(); // The object has been copied into the Wrapper.
    // The old copy must be deleted, so we cannot release ownership.
  }

  template <typename PROD>
  bool
  LuminosityBlock::get(SelectorBase const& sel, Handle<PROD>& result) const {
    return provRecorder_.get(sel,result);
  }

  template <typename PROD>
  bool
  LuminosityBlock::getByLabel(std::string const& label, Handle<PROD>& result) const {
    return provRecorder_.getByLabel(label,result);
  }

  template <typename PROD>
  bool
  LuminosityBlock::getByLabel(std::string const& label,
                  std::string const& productInstanceName,
                  Handle<PROD>& result) const {
    return provRecorder_.getByLabel(label,productInstanceName,result);
  }

  /// same as above, but using the InputTag class
  template <typename PROD>
  bool
  LuminosityBlock::getByLabel(InputTag const& tag, Handle<PROD>& result) const {
    return provRecorder_.getByLabel(tag,result);
  }

  template <typename PROD>
  void
  LuminosityBlock::getMany(SelectorBase const& sel, std::vector<Handle<PROD> >& results) const {
    return provRecorder_.getMany(sel,results);
  }

  template <typename PROD>
  bool
  LuminosityBlock::getByType(Handle<PROD>& result) const {
    return provRecorder_.getByType(result);
  }

  template <typename PROD>
  void
  LuminosityBlock::getManyByType(std::vector<Handle<PROD> >& results) const {
    return provRecorder_.getManyByType(results);
  }

}
#endif
