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
#include "FWCore/Framework/interface/stream/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

// forward declarations
#include "DataFormats/EgammaCandidates/interface/SiStripElectronFwd.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"

//
// class decleration
//

class SiStripElectronAssociator : public edm::stream::EDProducer<> {
 public:
  explicit SiStripElectronAssociator(const edm::ParameterSet&);
  ~SiStripElectronAssociator() override;
  
  
  void produce(edm::Event&, const edm::EventSetup&) override;
 private:
  // ----------member data ---------------------------
  edm::EDGetTokenT<reco::SiStripElectronCollection> siStripElectronCollection_;
  edm::EDGetTokenT<reco::TrackCollection> trackCollection_;
  
  edm::InputTag electronsLabel_;
};

#endif // EgammaElectronProducers_SiStripElectronAssociator_h
