#ifndef HeavyQuarkFromMPIFilter_h
#define HeavyQuarkFromMPIFilter_h

/** \class HeavyQuarkFromMPIFilter
 *
 *  HeavyQuarkFromMPIFilter 
 *
 * \author J Lamb, UCSB
 * this is just the wrapper around the filtering algorithm
 * found in HeavyQuarkFromMPIFilterAlgo
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

#include "GeneratorInterface/GenFilters/interface/HeavyQuarkFromMPIFilterAlgo.h"

class HeavyQuarkFromMPIFilter : public edm::EDFilter {
 public:
  explicit HeavyQuarkFromMPIFilter(const edm::ParameterSet&);
  ~HeavyQuarkFromMPIFilter();
  
  virtual bool filter(edm::Event&, const edm::EventSetup&);
  
 private:
  HeavyQuarkFromMPIFilterAlgo *HeavyQuarkFromMPIFilterAlgo_;
  
};
#endif
