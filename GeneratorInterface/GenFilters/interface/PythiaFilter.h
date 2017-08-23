#ifndef PYTHIAFILTER_h
#define PYTHIAFILTER_h
// -*- C++ -*-
//
// Package:    PythiaFilter
// Class:      PythiaFilter
// 
/**\class PythiaFilter PythiaFilter.cc IOMC/PythiaFilter/src/PythiaFilter.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Filip Moortgat
//         Created:  Mon Jan 23 14:57:54 CET 2006
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

class PythiaFilter : public edm::EDFilter {
   public:
      explicit PythiaFilter(const edm::ParameterSet&);
      ~PythiaFilter() override;


      bool filter(edm::Event&, const edm::EventSetup&) override;
   private:
      // ----------member data ---------------------------
      
       edm::EDGetTokenT<edm::HepMCProduct> token_;
       int particleID;
       double minpcut;
       double maxpcut;
       double minptcut;
       double maxptcut;
       double minetacut;
       double maxetacut;
       double minrapcut;
       double maxrapcut;
       double minphicut;
       double maxphicut;

       double rapidity;

       int status; 
       int motherID;   
       int processID;    

       double betaBoost;
};
#endif
