#ifndef MCParticlePairFilter_h
#define MCParticlePairFilter_h
// -*- C++ -*-
//
// Package:    MCParticlePairFilter
// Class:      MCParticlePairFilter
// 
/* 

 Description: filter events based on the Pythia particle information

 Implementation: inherits from generic EDFilter
     
*/
//
// Original Author:  Fabian Stoeckli
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

#include "FWCore/ParameterSet/interface/ParameterSet.h"


//
// class decleration
//
namespace edm {
  class HepMCProduct;
}

class MCParticlePairFilter : public edm::EDFilter {
   public:
      explicit MCParticlePairFilter(const edm::ParameterSet&);
      ~MCParticlePairFilter();


      virtual bool filter(edm::Event&, const edm::EventSetup&);
   private:
      // ----------memeber function----------------------
       int charge(const int& Id);

      // ----------member data ---------------------------
      
       edm::EDGetTokenT<edm::HepMCProduct> token_;
       std::vector<int> particleID1;
       std::vector<int> particleID2;
       std::vector<double> ptMin;
       std::vector<double> pMin;
       std::vector<double> etaMin;  
       std::vector<double> etaMax;
       std::vector<int> status;
       int particleCharge;
       double minInvMass;
       double maxInvMass;
       double minDeltaPhi;
       double maxDeltaPhi;
       double minDeltaR;
       double maxDeltaR;
       
};
#endif
