#ifndef MCSingleParticleYPt_h
#define MCSingleParticleYPt_h
// -*- C++ -*-
//
// Package:    MCSingleParticleYPt
// Class:      MCSingleParticleYPt
//
/* 

 Description: filter events based on the Pythia particleID, Pt, Y and status. It is based on MCSingleParticleFilter.
              It will used to filter a b-hadron with the given kinematics, only one b-hadron is required to match.
 Implementation: inherits from generic EDFilter
     
*/
//
// Author: Alberto Sanchez-Hernandez
// Adapted on: August 2016
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

class MCSingleParticleYPt : public edm::EDFilter {
public:
  explicit MCSingleParticleYPt(const edm::ParameterSet&);
  ~MCSingleParticleYPt() override;
  bool filter(edm::Event&, const edm::EventSetup&) override;

private:
  // ----------member data ---------------------------

  int fVerbose;
  bool fchekantiparticle;
  edm::EDGetTokenT<edm::HepMCProduct> token_;
  std::vector<int> particleID;
  std::vector<double> ptMin;
  std::vector<double> rapMin;
  std::vector<double> rapMax;
  std::vector<int> status;
  double rapidity;
};
#endif
