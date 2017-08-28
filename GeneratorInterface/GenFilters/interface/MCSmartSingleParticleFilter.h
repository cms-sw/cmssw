#ifndef MCSmartSingleParticleFilter_h
#define MCSmartSingleParticleFilter_h
// -*- C++ -*-
//
// Package:    MCSmartSingleParticleFilter
// Class:      MCSmartSingleParticleFilter
// 
/* 

 Description: filter events based on the Pythia particleID, the Pt and the production vertex

 Implementation: inherits from generic EDFilter
     
*/
//         Created:  J. Alcaraz, 04/07/2008
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
// class declaration
//
namespace edm {
  class HepMCProduct;
}

class MCSmartSingleParticleFilter : public edm::EDFilter {
   public:
      explicit MCSmartSingleParticleFilter(const edm::ParameterSet&);
      ~MCSmartSingleParticleFilter();


      virtual bool filter(edm::Event&, const edm::EventSetup&);
   private:
      // ----------memeber function----------------------

      // ----------member data ---------------------------
      
       edm::EDGetTokenT<edm::HepMCProduct> token_;
       std::vector<int> particleID;  
       std::vector<double> pMin;
       std::vector<double> ptMin;
       std::vector<double> etaMin;  
       std::vector<double> etaMax;
       std::vector<int> status;
       std::vector<double> decayRadiusMin;  
       std::vector<double> decayRadiusMax;
       std::vector<double> decayZMin;  
       std::vector<double> decayZMax;
       double betaBoost;
};
#endif
