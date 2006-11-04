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

$Id: Event.h,v 1.47 2006/10/31 23:54:01 wmtan Exp $

----------------------------------------------------------------------*/

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

    ///Put a new product where the product is gotten using a 'product instance name'
    template <typename PROD>
    OrphanHandle<PROD>
    put(std::auto_ptr<PROD> product, std::string const& productInstanceName = std::string()) {
      return put_<PROD>(InEvent, product, productInstanceName);
    }

    ///Returns a RefProd to a product before that product has been placed into the DataViewImpl
    /// The RefProd (and any Ref's made from it) will no work properly until after the
    /// DataViewImpl has been committed (which happens after leaving the EDProducer::produce method)
    template <typename PROD>
    RefProd<PROD>
    getRefBeforePut(std::string const& productInstanceName = std::string()) {
      return getRefBeforePut_<PROD>(InEvent, productInstanceName);
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
    friend class ConfigurableInputSource;
    friend class RawInputSource;
    friend class FilterWorker;
    friend class ProducerWorker;

    EventAux const& aux_;
  };
}
#endif
