#ifndef RecoMuon_L2MuonSeedGenerator_L2MuonSeedGenerator_H
#define RecoMuon_L2MuonSeedGenerator_L2MuonSeedGenerator_H

// -*- C++ -*-
//
// Package:    L2MuonSeedGenerator
// Class:      L2MuonSeedGenerator
// 
/**\class L2MuonSeedGenerator L2MuonSeedGenerator.cc src/L2MuonSeedGenerator/src/L2MuonSeedGenerator.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Adam A Everett
// $Id$
//
//


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"

#include "FWCore/Framework/interface/MakerMacros.h"

class TrajectorySeed;

namespace edm {class ParameterSet; class Event; class EventSetup;}

//
// class decleration
//

class L2MuonSeedGenerator : public edm::EDProducer {
 public:

  // Constructor
  explicit L2MuonSeedGenerator(const edm::ParameterSet&);

  // Destructor
  ~L2MuonSeedGenerator();
  
 private:
  virtual void beginJob(const edm::EventSetup&) ;
  virtual void produce(edm::Event&, const edm::EventSetup&);
  virtual void endJob() ;
  
  // ----------member data ---------------------------
  
  std::vector<TrajectorySeed> theSeeds;

  const double theL1MinPt;
  const double theL1MaxEta;
  const double theL1MinQuality;
  
};
#endif
