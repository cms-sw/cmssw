#ifndef Framework_LuminosityBlockPrincipal_h
#define Framework_LuminosityBlockPrincipal_h

/*----------------------------------------------------------------------
  
LuminosityBlockPrincipal: This is the class responsible for management of
per luminosity block EDProducts. It is not seen by reconstruction code;
such code sees the LuminosityBlock class, which is a proxy for LuminosityBlockPrincipal.

The major internal component of the LuminosityBlockPrincipal
is the DataBlock.

$Id: LuminosityBlockPrincipal.h,v 1.8 2007/01/12 18:02:55 wmtan Exp $

----------------------------------------------------------------------*/

#include "DataFormats/Common/interface/LuminosityBlockAux.h"
#include "FWCore/Framework/interface/DataBlockImpl.h"

#include "boost/shared_ptr.hpp"

namespace edm {
  class RunPrincipal;
  class LuminosityBlockPrincipal : private DataBlockImpl {
  typedef DataBlockImpl Base;
  public:
    LuminosityBlockPrincipal(LuminosityBlockID const& id,
	ProductRegistry const& reg,
        boost::shared_ptr<RunPrincipal const> rp,
        ProcessConfiguration const& pc,
	ProcessHistoryID const& hist = ProcessHistoryID(),
	boost::shared_ptr<DelayedReader> rtrv = boost::shared_ptr<DelayedReader>(new NoDelayedReader));

    LuminosityBlockPrincipal(LuminosityBlockID const& id,
	ProductRegistry const& reg,
        ProcessConfiguration const& pc,
	ProcessHistoryID const& hist = ProcessHistoryID(),
	boost::shared_ptr<DelayedReader> rtrv = boost::shared_ptr<DelayedReader>(new NoDelayedReader));

    ~LuminosityBlockPrincipal() {}

    RunPrincipal const& runPrincipal() const {
      return *runPrincipal_;
    }

    boost::shared_ptr<RunPrincipal const> const
    runPrincipalConstSharedPtr() const {
      return runPrincipal_;
    }

    LuminosityBlockID const& id() const {
      return aux().id();
    }

    LuminosityBlockAux const& aux() const {
      return aux_;
    }

    RunNumber_t const& runNumber() const {
      return aux().runID();
    }
    using Base::addGroup;
    using Base::addToProcessHistory;
    using Base::begin;
    using Base::end;
    using Base::getAllProvenance;
    using Base::getByLabel;
    using Base::get;
    using Base::getBySelector;
    using Base::getByType;
    using Base::getGroup;
    using Base::getIt;
    using Base::getMany;
    using Base::getManyByType;
    using Base::getProvenance;
    using Base::groupGetter;
    using Base::numEDProducts;
    using Base::processHistory;
    using Base::processHistoryID;
    using Base::prodGetter;
    using Base::productRegistry;
    using Base::put;
    using Base::size;
    using Base::store;

  private:
    virtual bool unscheduledFill(Group const&) const {return false;}
    virtual bool fillAndMatchSelector(Provenance &, SelectorBase const&) const {return false;}

    boost::shared_ptr<RunPrincipal const> const runPrincipal_;
    LuminosityBlockAux aux_;
  };
}
#endif

