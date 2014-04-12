#ifndef MCLongLivedParticles_h
#define MCLongLivedParticles_h
// -*- C++ -*-
//
// Package:    MCLongLivedParticles
// Class:      MCLongLivedParticles
// 
/* 

 Description: filter events based on the Pythia ProcessID and the Pt_hat

 Implementation: inherits from generic EDFilter
     
*/
//
// Original Author:  Filip Moortgat
//         Created:  Mon Sept 11 10:57:54 CET 2006
//
//


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDFilter.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"


//
// class decleration
//

class MCLongLivedParticles : public edm::EDFilter {
public:
  explicit MCLongLivedParticles(const edm::ParameterSet&);
  ~MCLongLivedParticles();
  

  virtual bool filter(edm::Event&, const edm::EventSetup&);
private:
  // ----------member data ---------------------------
  
  float theCut;
  edm::InputTag hepMCProductTag_;
};
#endif
