#ifndef doubleEMEnrichingFilter_h
#define doubleEMEnrichingFilter_h

/** \class doubleEMEnrichingFilter
 *
 *  doubleEMEnrichingFilter 
 *
 * \author R.Arcidiacono,C.Rovelli,R.Paramatti
 * this is just the wrapper around the filtering algorithm
 * found in doubleEMEnrichingFilterAlgo
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

#include "GeneratorInterface/GenFilters/interface/doubleEMEnrichingFilterAlgo.h"

class doubleEMEnrichingFilter : public edm::EDFilter {
 public:
  explicit doubleEMEnrichingFilter(const edm::ParameterSet&);
  ~doubleEMEnrichingFilter();
  
  virtual bool filter(edm::Event&, const edm::EventSetup&);
  
 private:
  doubleEMEnrichingFilterAlgo *doubleEMEAlgo_;
  
};
#endif
