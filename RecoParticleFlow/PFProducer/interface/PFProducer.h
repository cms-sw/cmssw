#ifndef RecoParticleFlow_PFProducer_h_
#define RecoParticleFlow_PFProducer_h_

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/PFReco/interface/PFRecHit.h"

// #include "Geometry/CaloTopology/interface/CaloSubdetectorTopology.h"

// -*- C++ -*-
//
// Package:    PFProducer
// Class:      PFProducer
// 
/**\class PFProducer 

Description: <one line class summary>

Implementation:
<Notes on implementation>
*/
//
// Original Author:  Colin Bernet
//         Created:  Tue Jun 27 14:35:24 CEST 2006
// $Id$
//
//



class PFProducer : public edm::EDProducer {
 public:
  explicit PFProducer(const edm::ParameterSet&);
  ~PFProducer();

  
  virtual void produce(edm::Event&, const edm::EventSetup&);

 private:
  // ----------member data ---------------------------

};

#endif
