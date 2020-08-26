#ifndef PYTHIAFILTERMOTHERSISTER_h
#define PYTHIAFILTERMOTHERSISTER_h
// -*- C++ -*-
//
// Package:    PythiaFilterMotherSister
// Class:      PythiaFilterMotherSister
// 
/**\class PythiaFilterMotherSister PythiaFilterMotherSister.cc IOMC/PythiaFilterMotherSister/src/PythiaFilterMotherSister.cc

 Description: A filter to identify a particle with given id and kinematic && give mother id && given id of one among mother's daughters

 Implementation:
     <Notes on implementation>
*/
//
// 
//         
//
//


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/global/EDFilter.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

//
// class decleration
//
namespace edm {
  class HepMCProduct;
}

class PythiaFilterMotherSister : public edm::global::EDFilter<> {
   public:
      explicit PythiaFilterMotherSister(const edm::ParameterSet&);
      ~PythiaFilterMotherSister() override;


      bool filter(edm::StreamID, edm::Event&, const edm::EventSetup&) const override;
   private:
      // ----------member data ---------------------------
      
       const edm::EDGetTokenT<edm::HepMCProduct> token_;
       const int particleID;
       const double minpcut;
       const double maxpcut;
       const double minptcut;
       const double maxptcut;
       const double minetacut;
       const double maxetacut;
       const double minrapcut;
       const double maxrapcut;
       const double minphicut;
       const double maxphicut;

       //const int status; 
       const int motherID;   
       const int sisterID;   
       //const int processID;    

       const double betaBoost;
};
#endif
