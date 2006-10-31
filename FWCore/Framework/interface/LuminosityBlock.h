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

$Id: LuminosityBlock.h,v 1.46 2006/10/30 23:07:53 wmtan Exp $

----------------------------------------------------------------------*/

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

    using DataViewImpl::get;
    using DataViewImpl::getAllProvenance;
    using DataViewImpl::getByLabel;
    using DataViewImpl::getByType;
    using DataViewImpl::getMany;
    using DataViewImpl::getManyByType;
    using DataViewImpl::getProvenance;
    using DataViewImpl::put;

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
