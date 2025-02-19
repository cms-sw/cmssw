#ifndef EgammaElectronProducers_SiStripElectronAssociator_h
#define EgammaElectronProducers_SiStripElectronAssociator_h
// -*- C++ -*-
//
// Package:     EgammaElectronProducers
// Class  :     SiStripElectronAssociator
// 
/**\class SiStripElectronAssociator SiStripElectronAssociator.h RecoEgamma/EgammaElectronProducers/interface/SiStripElectronAssociator.h

 Description: <one line class summary>

 Usage:
    <usage>

*/
//
// Original Author:  Jim Pivarski
//         Created:  Tue Aug  1 15:24:02 EDT 2006
//

// system include files
#include <memory>
#include <string>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

// forward declarations

//
// class decleration
//

class SiStripElectronAssociator : public edm::EDProducer {
 public:
  explicit SiStripElectronAssociator(const edm::ParameterSet&);
  ~SiStripElectronAssociator();
  
  
  virtual void produce(edm::Event&, const edm::EventSetup&);
 private:
  // ----------member data ---------------------------
  edm::InputTag siStripElectronProducer_;
  edm::InputTag siStripElectronCollection_;
  edm::InputTag trackProducer_;
  edm::InputTag trackCollection_;
  
  edm::InputTag electronsLabel_;
};

#endif // EgammaElectronProducers_SiStripElectronAssociator_h
