#ifndef EMEnrichingFilter_h
#define EMEnrichingFilter_h

/** \class EMEnrichingFilter
 *
 *  EMEnrichingFilter 
 *
 * \author J Lamb, UCSB
 * this is just the wrapper around the filtering algorithm
 * found in EMEnrichingFilterAlgo
 * 
 *
 ************************************************************/

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDFilter.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "GeneratorInterface/GenFilters/interface/EMEnrichingFilterAlgo.h"

class EMEnrichingFilter : public edm::EDFilter {
 public:
  explicit EMEnrichingFilter(const edm::ParameterSet&);
  ~EMEnrichingFilter();
  
  virtual bool filter(edm::Event&, const edm::EventSetup&);
  
 private:
  EMEnrichingFilterAlgo *EMEAlgo_;
  
};
#endif
