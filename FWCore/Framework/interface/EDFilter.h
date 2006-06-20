#ifndef Framework_EDFilter_h
#define Framework_EDFilter_h

/*----------------------------------------------------------------------
  
EDFilter: The base class of all "modules" used to control the flow of
processing in a processing path.
Filters can also insert products into the event.
These products should be informational products about the filter decision.

$Id: EDFilter.h,v 1.8 2006/04/20 22:33:21 wmtan Exp $

----------------------------------------------------------------------*/

#include "FWCore/Framework/interface/ProducerBase.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"

namespace edm {

  class EDFilter : public ProducerBase {
  public:
    typedef EDFilter ModuleType;
    
    EDFilter() : ProducerBase() {}
    virtual ~EDFilter();
    bool doFilter(Event& e, EventSetup const& c,
		  CurrentProcessingContext const* cpc);
    void doBeginJob(EventSetup const&) ;
    void doEndJob() ;

  private:    
    virtual bool filter(Event& e, EventSetup const& c) = 0;
    virtual void beginJob(EventSetup const&) ;
    virtual void endJob() ;

    CurrentProcessingContext const* current_context_;
  };
}

#endif
