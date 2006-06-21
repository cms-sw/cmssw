/*----------------------------------------------------------------------
  
$Id: EDFilter.cc,v 1.5 2006/06/20 23:13:27 paterno Exp $

----------------------------------------------------------------------*/

#include "FWCore/Framework/interface/EDFilter.h"
#include "FWCore/Framework/interface/CurrentProcessingContext.h"
#include "FWCore/Framework/src/CPCSentry.h"

namespace edm
{
  EDFilter::~EDFilter()
  { }

  bool
  EDFilter::doFilter(Event& e, EventSetup const& c,
		     CurrentProcessingContext const* cpc)
  {
    detail::CPCSentry sentry(current_context_, cpc);
    return this->filter(e, c);
  }

  void 
  EDFilter::doBeginJob(EventSetup const& es) 
  { 
    this->beginJob(es);
  }
   
  void EDFilter::doEndJob()
  { 
    this->endJob();
  }

  void 
  EDFilter::beginJob(EventSetup const&) 
  { }
   
  void 
  EDFilter::endJob()
  { }

  CurrentProcessingContext const*
  EDFilter::currentContext() const
  {
    return current_context_;
  }
  
   
}
  
