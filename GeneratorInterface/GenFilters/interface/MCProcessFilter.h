#ifndef MCPROCESSFILTER_h
#define MCPROCESSFILTER_h
// -*- C++ -*-
//
// Package:    MCProcessFilter
// Class:      MCProcessFilter
// 
/* 

 Description: filter events based on the Pythia ProcessID and the Pt_hat

 Implementation: inherits from generic EDFilter
     
*/
//
// Original Author:  Filip Moortgat
//         Created:  Mon Sept 11 10:57:54 CET 2006
// $Id: MCProcessFilter.h,v 1.2 2010/07/21 04:23:23 wmtan Exp $
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

class MCProcessFilter : public edm::EDFilter {
   public:
      explicit MCProcessFilter(const edm::ParameterSet&);
      ~MCProcessFilter();


      virtual bool filter(edm::Event&, const edm::EventSetup&);
   private:
      // ----------member data ---------------------------
      
       std::string label_;
       std::vector<int> processID;  
       std::vector<double> pthatMin;
       std::vector<double> pthatMax;  
};
#endif
