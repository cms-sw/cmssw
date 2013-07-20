#ifndef MCSingleParticleFilter_h
#define MCSingleParticleFilter_h
// -*- C++ -*-
//
// Package:    MCSingleParticleFilter
// Class:      MCSingleParticleFilter
// 
/* 

 Description: filter events based on the Pythia particleID and the Pt_hat

 Implementation: inherits from generic EDFilter
     
*/
//
// Original Author:  Filip Moortgat
//         Created:  Mon Sept 11 10:57:54 CET 2006
// $Id: MCSingleParticleFilter.h,v 1.2 2010/07/21 04:23:24 wmtan Exp $
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

class MCSingleParticleFilter : public edm::EDFilter {
   public:
      explicit MCSingleParticleFilter(const edm::ParameterSet&);
      ~MCSingleParticleFilter();


      virtual bool filter(edm::Event&, const edm::EventSetup&);
   private:
      // ----------member data ---------------------------
      
       std::string label_;
       std::vector<int> particleID;  
       std::vector<double> ptMin;
       std::vector<double> etaMin;  
       std::vector<double> etaMax;
       std::vector<int> status;
};
#endif
