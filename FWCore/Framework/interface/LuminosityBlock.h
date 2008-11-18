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

For its usage, see "FWCore/Framework/interface/DataViewImpl.h"

*/
/*----------------------------------------------------------------------

$Id: LuminosityBlock.h,v 1.17 2008/05/12 18:14:07 wmtan Exp $

----------------------------------------------------------------------*/

#include "boost/shared_ptr.hpp"

#include "DataFormats/Provenance/interface/LuminosityBlockAuxiliary.h"
#include "DataFormats/Provenance/interface/LuminosityBlockID.h"
#include "DataFormats/Provenance/interface/RunID.h"

#include "FWCore/Framework/interface/DataViewImpl.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"

namespace edm {

  class LuminosityBlock : private DataViewImpl
  {
  public:
    LuminosityBlock(LuminosityBlockPrincipal& lbp, const ModuleDescription& md);
    ~LuminosityBlock() {}

    typedef DataViewImpl Base;
    // AUX functions.
    LuminosityBlockNumber_t luminosityBlock() const {return aux_.luminosityBlock();}

    RunNumber_t run() const {
      return aux_.run();
    }

    LuminosityBlockID id() const {
      return aux_.id();
    }

    Timestamp const& beginTime() const {return aux_.beginTime();}
    Timestamp const& endTime() const {return aux_.endTime();}

    using Base::get;
    using Base::getByLabel;
    using Base::getByType;
    using Base::getMany;
    using Base::getManyByType;
    using Base::me;
    using Base::processHistory;

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
    getAllProvenance(std::vector<Provenance const*> &provenances) const;

  private:
    LuminosityBlockPrincipal const&
    luminosityBlockPrincipal() const;

    LuminosityBlockPrincipal &
    luminosityBlockPrincipal();

    // commit_() is called to complete the transaction represented by
    // this DataViewImpl. The friendships required seems gross, but any
    // alternative is not great either.  Putting it into the
    // public interface is asking for trouble
    friend class ConfigurableInputSource;
    friend class InputSource;
    friend class DaqSource;
    friend class RawInputSource;
    friend class EDFilter;
    friend class EDProducer;

    void commit_();

    LuminosityBlockAuxiliary const& aux_;
    boost::shared_ptr<Run const> const run_;
  };

  template <typename PROD>
  void
  LuminosityBlock::put(std::auto_ptr<PROD> product, std::string const& productInstanceName)
  {
    if (product.get() == 0) {                // null pointer is illegal
      TypeID typeID(typeid(PROD));
      throw edm::Exception(edm::errors::NullPointerError)
        << "LuminosityBlock::put: A null auto_ptr was passed to 'put'.\n"
	<< "The pointer is of type " << typeID << ".\n"
	<< "The specified productInstanceName was '" << productInstanceName << "'.\n";
    }

    // The following will call post_insert if T has such a function,
    // and do nothing if T has no such function.
    typename boost::mpl::if_c<detail::has_postinsert<PROD>::value, 
      DoPostInsert<PROD>, 
      DoNotPostInsert<PROD> >::type maybe_inserter;
    maybe_inserter(product.get());

    ConstBranchDescription const& desc =
      getBranchDescription(TypeID(*product), productInstanceName);

    Wrapper<PROD> *wp(new Wrapper<PROD>(product));

    putProducts().push_back(std::make_pair(wp, &desc));

    // product.release(); // The object has been copied into the Wrapper.
    // The old copy must be deleted, so we cannot release ownership.
  }
  
}
#endif
