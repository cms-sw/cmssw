#ifndef MCProcessRangeFilter_h
#define MCProcessRangeFilter_h
// -*- C++ -*-
//
// Package:    MCProcessRangeFilter
// Class:      MCProcessRangeFilter
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

#include "FWCore/ParameterSet/interface/ParameterSet.h"


//
// class decleration
//
namespace edm {
  class HepMCProduct;
}

class MCProcessRangeFilter : public edm::EDFilter {
   public:
      explicit MCProcessRangeFilter(const edm::ParameterSet&);
      ~MCProcessRangeFilter();


      virtual bool filter(edm::Event&, const edm::EventSetup&);
   private:
      // ----------member data ---------------------------
      
       edm::EDGetTokenT<edm::HepMCProduct> token_;
       int minProcessID;
       int maxProcessID;  
       double pthatMin;
       double pthatMax;
};
#endif
