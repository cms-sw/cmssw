#ifndef Framework_EPEventProvenanceFiller_h
#define Framework_EPEventProvenanceFiller_h
// -*- C++ -*-
//
// Package:     Framework
// Class  :     EPEventProvenanceFiller
// 
/**\class EPEventProvenanceFiller EPEventProvenanceFiller.h FWCore/Framework/interface/EPEventProvenanceFiller.h

 Description: Base class used internal to the Framework for filling Event based Provenance

 Usage:

    This class is used during 'unscheduled' execution for the case
    where a Selector uses the Provenance information to decide to
    'get' a datum from the Event. Under that case, the 'unscheduled'
    case must run the Producer who makes such a datum and then see if
    the provenance information is a match.  We use an abstract base
    class in order to avoid unnecessary physical coupling in the
    EventPrincipal class.

    This is the version used by the EventPrincipal.

*/

//
// Original Author:  Chris Jones
//         Created:  Wed Feb 15 14:54:43 IST 2006
// $Id: EPEventProvenanceFiller.h,v 1.2 2006/08/21 22:28:25 wmtan Exp $
//

// system include files

// user include files

#include "DataFormats/Common/interface/Provenance.h"
#include "FWCore/Framework/interface/EventProvenanceFiller.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/UnscheduledHandler.h"

// forward declarations
namespace edm 
{
  class EPEventProvenanceFiller : public EventProvenanceFiller
  {
  public:
    EPEventProvenanceFiller(boost::shared_ptr<edm::UnscheduledHandler> handler, 
			    edm::EventPrincipal* iEvent) : 
      handler_(handler), 
      event_(iEvent) 
    { }

    virtual bool fill(edm::Provenance& prov) 
    {
      if (handler_) handler_->tryToFill(prov, *event_);
      return true;
    }

  private:
    boost::shared_ptr<edm::UnscheduledHandler> handler_;
    edm::EventPrincipal* event_;
  };
}
#endif
