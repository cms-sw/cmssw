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

$Id: LuminosityBlock.h,v 1.1 2006/10/31 23:54:01 wmtan Exp $

----------------------------------------------------------------------*/

#include "DataFormats/Common/interface/BranchType.h"
#include "DataFormats/Common/interface/LuminosityBlockAux.h"
#include "DataFormats/Common/interface/LuminosityBlockID.h"
#include "DataFormats/Common/interface/Timestamp.h"

#include "FWCore/Framework/interface/DataViewImpl.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"

#include "FWCore/Utilities/interface/GCCPrerequisite.h"

namespace edm {

#if GCC_PREREQUISITE(3,4,4)
  class LuminosityBlock : private DataViewImpl
#else
  // Bug in gcc3.2.3 compiler forces public inheritance
  class LuminosityBlock : public DataViewImpl
#endif
  {
  public:
    LuminosityBlock(LuminosityBlockPrincipal& dbk, const ModuleDescription& md);
    ~LuminosityBlock(){}

    // AUX functions.
    LuminosityBlockID id() const {return aux_.id();}
    Timestamp time() const {return aux_.time();}

    ///Put a new product where the product is gotten using a 'product instance name'
    template <typename PROD>
    OrphanHandle<PROD>
    put(std::auto_ptr<PROD> product, std::string const& productInstanceName = std::string()) {
      return put_<PROD>(InLumi, product, productInstanceName);
    }

    ///Returns a RefProd to a product before that product has been placed into the DataViewImpl
    /// The RefProd (and any Ref's made from it) will no work properly until after the
    /// DataViewImpl has been committed (which happens after leaving the EDProducer::produce method)
    template <typename PROD>
    RefProd<PROD>
    getRefBeforePut(std::string const& productInstanceName = std::string()) {
      return getRefBeforePut_<PROD>(InLumi, productInstanceName);
    }

    using DataViewImpl::get;
    using DataViewImpl::getAllProvenance;
    using DataViewImpl::getByLabel;
    using DataViewImpl::getByType;
    using DataViewImpl::getMany;
    using DataViewImpl::getManyByType;
    using DataViewImpl::getProvenance;

  private:
    // commit_() is called to complete the transaction represented by
    // this DataViewImpl. The friendships required seems gross, but any
    // alternative is not great either.  Putting it into the
    // public interface is asking for trouble
    //friend class ConfigurableInputSource;
    //friend class RawInputSource;
    //friend class FilterWorker;
    //friend class ProducerWorker;

    LuminosityBlockAux const& aux_;
  };
}
#endif
