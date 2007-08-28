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

$Id: LuminosityBlock.h,v 1.12 2007/07/31 23:58:54 wmtan Exp $

----------------------------------------------------------------------*/

#include "boost/shared_ptr.hpp"

#include "DataFormats/Provenance/interface/LuminosityBlockAuxiliary.h"
#include "DataFormats/Provenance/interface/LuminosityBlockID.h"
#include "DataFormats/Provenance/interface/RunID.h"

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
    ~LuminosityBlock() {}

    // AUX functions.
    LuminosityBlockNumber_t luminosityBlock() const {return aux_.luminosityBlock();}

    RunNumber_t run() const {
      return aux_.run();
    }

    LuminosityBlockID id() const {
      return aux_.id();
    }

    using DataViewImpl::get;
    using DataViewImpl::getAllProvenance;
    using DataViewImpl::getByLabel;
    using DataViewImpl::getByType;
    using DataViewImpl::getMany;
    using DataViewImpl::getManyByType;
    using DataViewImpl::getProvenance;
    using DataViewImpl::getRefBeforePut;
    using DataViewImpl::me;
    using DataViewImpl::processHistory;
    using DataViewImpl::put;

    Run const&
    getRun() const {
      return *run_;
    }

  private:
    // commit_() is called to complete the transaction represented by
    // this DataViewImpl. The friendships required seems gross, but any
    // alternative is not great either.  Putting it into the
    // public interface is asking for trouble
    friend class ConfigurableInputSource;
    friend class InputSource;
    friend class DaqSource;
    friend class RawInputSource;
    friend class FilterWorker;
    friend class ProducerWorker;

    LuminosityBlockAuxiliary const& aux_;
    boost::shared_ptr<Run const> const run_;
  };
}
#endif
