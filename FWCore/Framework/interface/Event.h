#ifndef Framework_Event_h
#define Framework_Event_h

// -*- C++ -*-
//
// Package:     Framework
// Class  :     Event
// 
/**\class Event Event.h FWCore/Framework/interface/Event.h

Description: This is the primary interface for accessing EDProducts from a single collision and inserting new derived products.

For its usage, see "FWCore/Framework/interface/DataViewImpl.h"

*/
/*----------------------------------------------------------------------

$Id: Event.h,v 1.50 2006/11/30 15:37:06 paterno Exp $

----------------------------------------------------------------------*/

#include "boost/shared_ptr.hpp"

#include "DataFormats/Common/interface/EventAux.h"
#include "DataFormats/Common/interface/EventID.h"
#include "DataFormats/Common/interface/Timestamp.h"

#include "FWCore/Framework/interface/DataViewImpl.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"

#include "FWCore/Utilities/interface/GCCPrerequisite.h"

namespace edm {

#if GCC_PREREQUISITE(3,4,4)
  class Event : private DataViewImpl
#else
  // Bug in gcc3.2.3 compiler forces public inheritance
  class Event : public DataViewImpl
#endif
  {
  public:
    Event(EventPrincipal& dbk, const ModuleDescription& md);
    ~Event(){}

    // AUX functions.
    EventID id() const {return aux_.id();}
    Timestamp time() const {return aux_.time();}

    using DataViewImpl::get;
    using DataViewImpl::getAllProvenance;
    using DataViewImpl::getByLabel;
    using DataViewImpl::getByType;
    using DataViewImpl::getMany;
    using DataViewImpl::getManyByType;
    using DataViewImpl::getProvenance;
    using DataViewImpl::getRefBeforePut;
    using DataViewImpl::put;
    using DataViewImpl::size;

    LuminosityBlock const&
    getLuminosityBlock() const {
      return *luminosityBlock_;
    }

    Run const&
    getRun() const;

  private:
    // commit_() is called to complete the transaction represented by
    // this DataViewImpl. The friendships required seems gross, but any
    // alternative is not great either.  Putting it into the
    // public interface is asking for trouble
    friend class ConfigurableInputSource;
    friend class RawInputSource;
    friend class FilterWorker;
    friend class ProducerWorker;

    EventAux const& aux_;
    boost::shared_ptr<LuminosityBlock const> const luminosityBlock_;
  };
}
#endif
