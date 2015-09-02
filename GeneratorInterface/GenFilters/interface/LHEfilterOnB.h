#ifndef LHEfilterOnB_h
#define LHEfilterOnB_h

// system include files
#include <memory>
#include <iostream>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDFilter.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "SimDataFormats/GeneratorProducts/interface/LHEEventProduct.h"

//
// class declaration
//

class LHEfilterOnB : public edm::EDFilter {
 public:
  explicit LHEfilterOnB(const edm::ParameterSet&);
  ~LHEfilterOnB();
  
 private:
  virtual bool filter(edm::Event&, const edm::EventSetup&);
  virtual void endJob();
  
  // ----------member data ---------------------------
  
  edm::InputTag lhesrc_;
  int totalEvents_;                // counters
  int passedEvents_;

};
#endif
