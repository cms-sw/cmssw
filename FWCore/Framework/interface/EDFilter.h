#ifndef FWCore_Framework_EDFilter_h
#define FWCore_Framework_EDFilter_h

/*----------------------------------------------------------------------
  
EDFilter: The base class of all "modules" used to control the flow of
processing in a processing path.
Filters can also insert products into the event.
These products should be informational products about the filter decision.

$Id: EDFilter.h,v 1.13 2006/11/03 17:57:51 wmtan Exp $

----------------------------------------------------------------------*/

#include "FWCore/Framework/interface/ProducerBase.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "DataFormats/Provenance/interface/ModuleDescription.h"

namespace edm {

  class EDFilter : public ProducerBase {
  public:
    friend class FilterWorker;
    typedef EDFilter ModuleType;
    
    EDFilter() : ProducerBase() , moduleDescription_(), current_context_(0) {}
    virtual ~EDFilter();
    bool doFilter(Event& e, EventSetup const& c,
		  CurrentProcessingContext const* cpc);
    void doBeginJob(EventSetup const&);
    void doEndJob();
    bool doBeginRun(Run & r, EventSetup const& c,
		   CurrentProcessingContext const* cpc);
    bool doEndRun(Run & r, EventSetup const& c,
		   CurrentProcessingContext const* cpc);
    bool doBeginLuminosityBlock(LuminosityBlock & lb, EventSetup const& c,
		   CurrentProcessingContext const* cpc);
    bool doEndLuminosityBlock(LuminosityBlock & lb, EventSetup const& c,
		   CurrentProcessingContext const* cpc);


  protected:
    // The returned pointer will be null unless the this is currently
    // executing its event loop function ('filter').
    CurrentProcessingContext const* currentContext() const;

  private:    
    virtual bool filter(Event&, EventSetup const&) = 0;
    virtual void beginJob(EventSetup const&){}
    virtual void endJob(){}
    virtual bool beginRun(Run &, EventSetup const&){return true;}
    virtual bool endRun(Run &, EventSetup const&){return true;}
    virtual bool beginLuminosityBlock(LuminosityBlock &, EventSetup const&){return true;}
    virtual bool endLuminosityBlock(LuminosityBlock &, EventSetup const&){return true;}

    void setModuleDescription(ModuleDescription const& md) {
      moduleDescription_ = md;
    }
    ModuleDescription moduleDescription_;
    CurrentProcessingContext const* current_context_;
  };
}

#endif
