#ifndef Framework_Run_h
#define Framework_Run_h

// -*- C++ -*-
//
// Package:     Framework
// Class  :     Run
// 
/**\class Run Run.h FWCore/Framework/interface/Run.h

Description: This is the primary interface for accessing per run EDProducts and inserting new derived products.

For its usage, see "FWCore/Framework/interface/DataViewImpl.h"

*/
/*----------------------------------------------------------------------

$Id: Run.h,v 1.7 2007/03/04 06:00:22 wmtan Exp $

----------------------------------------------------------------------*/

#include "DataFormats/Provenance/interface/RunAuxiliary.h"
#include "DataFormats/Provenance/interface/RunID.h"

#include "FWCore/Framework/interface/DataViewImpl.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"

#include "FWCore/Utilities/interface/GCCPrerequisite.h"

namespace edm {

#if GCC_PREREQUISITE(3,4,4)
  class Run : private DataViewImpl
#else
  // Bug in gcc3.2.3 compiler forces public inheritance
  class Run : public DataViewImpl
#endif
  {
  public:
    Run(RunPrincipal& dbk, const ModuleDescription& md);
    ~Run(){}

    // AUX functions.
    RunID const& id() const {return aux_.id();}
    RunNumber_t run() const {return aux_.run();}

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

  private:
    // commit_() is called to complete the transaction represented by
    // this DataViewImpl. The friendships required seems gross, but any
    // alternative is not great either.  Putting it into the
    // public interface is asking for trouble
    friend class ConfigurableInputSource;
    friend class InputSource;
    friend class RawInputSource;
    friend class FilterWorker;
    friend class ProducerWorker;

    RunAuxiliary const& aux_;
  };
}
#endif
