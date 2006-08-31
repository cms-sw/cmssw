#ifndef Framework_EDFilter_h
#define Framework_EDFilter_h

/*----------------------------------------------------------------------
  
EDFilter: The base class of all "modules" used to control the flow of
processing in a processing path.
Filters can also insert products into the event.
These products should be informational products about the filter decision.

$Id: EDFilter.h,v 1.10 2006/06/21 19:03:12 paterno Exp $

----------------------------------------------------------------------*/

#include "FWCore/Framework/interface/ProducerBase.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"

namespace edm {

  class EDFilter : public ProducerBase {
  public:
    typedef EDFilter ModuleType;
    
    EDFilter() : ProducerBase() , current_context_(0) {}
    virtual ~EDFilter();
    bool doFilter(Event& e, EventSetup const& c,
		  CurrentProcessingContext const* cpc);
    void doBeginJob(EventSetup const&) ;
    void doEndJob() ;

  protected:
    // The returned pointer will be null unless the this is currently
    // executing its event loop function ('filter').
    CurrentProcessingContext const* currentContext() const;

  private:    
    virtual bool filter(Event& e, EventSetup const& c) = 0;
    virtual void beginJob(EventSetup const&) ;
    virtual void endJob() ;

    CurrentProcessingContext const* current_context_;
  };
}

#endif
