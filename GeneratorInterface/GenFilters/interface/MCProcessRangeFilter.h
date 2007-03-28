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
// $Id: MCProcessRangeFilter.h,v 1.1 2006/09/22 15:28:27 fmoortga Exp $
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


using namespace edm;
using namespace std;

//
// class decleration
//

class MCProcessRangeFilter : public edm::EDFilter {
   public:
      explicit MCProcessRangeFilter(const edm::ParameterSet&);
      ~MCProcessRangeFilter();


      virtual bool filter(Event&, const EventSetup&);
   private:
      // ----------member data ---------------------------
      
       std::string label_;      
       int minProcessID;
       int maxProcessID;  
       double pthatMin;
       double pthatMax;
};
#endif
