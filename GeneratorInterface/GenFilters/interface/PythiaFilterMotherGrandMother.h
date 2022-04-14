#ifndef PYTHIAFILTERMOTHERGRANDMOTHER_h
#define PYTHIAFILTERMOTHERGRANDMOTHER_h
// -*- C++ -*-
//
// Package:    PythiaFilterMotherGrandMother
// Class:      PythiaFilterMotherGrandMother
// 
/**\class PythiaFilterMotherGrandMother PythiaFilterMotherGrandMother.cc IOMC/PythiaFilterMotherGrandMother/src/PythiaFilterMotherGrandMother.cc

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

class PythiaFilterMotherGrandMother : public edm::global::EDFilter<> {
   public:
      explicit PythiaFilterMotherGrandMother(const edm::ParameterSet&);
      ~PythiaFilterMotherGrandMother() override;


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
       std::vector<int> grandMotherIDs;
       const int motherID;   
       //const int processID;    

       const double betaBoost;
};
#endif
