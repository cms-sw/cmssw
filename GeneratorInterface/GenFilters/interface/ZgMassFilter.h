#ifndef ZgMassFilter_h
#define ZgMassFilter_h
// -*- C++ -*-
//
// Package:    ZgMassFilter
// Class:      ZgMassFilter
// 
/* 

 Description: filter events based on the Pythia particle information

 Implementation: inherits from generic EDFilter
     
*/
//
// Original Author:  Alexey Ferapontov
//         Created:  Thu July 26 11:57:54 CDT 2012
// $Id: ZgMassFilter.h,v 1.1 2012/08/10 12:46:29 lenzip Exp $
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

class ZgMassFilter : public edm::EDFilter {
   public:
      explicit ZgMassFilter(const edm::ParameterSet&);
      ~ZgMassFilter();


      virtual bool filter(edm::Event&, const edm::EventSetup&);
   private:
      // ----------memeber function----------------------
       int charge(const int& Id);

      // ----------member data ---------------------------
      
       std::string label_;
       double minDileptonMass;
       double minZgMass;       
};
#endif
