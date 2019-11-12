// -*- C++ -*-
//
// Package:    HZZ4lFilter
// Class:      HZZ4lFilter
//
/**\class HZZ4lFilter HZZ4lFilter.cc IOMC/HZZ4lFilter/src/HZZ4lFilter.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Puljak Ivica
//         Created:  Wed Apr 18 12:52:31 CEST 2007
//
//

#include "GeneratorInterface/GenFilters/plugins/HZZ4lFilter.h"

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDFilter.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"

HZZ4lFilter::HZZ4lFilter(const edm::ParameterSet& iConfig)
    : token_(consumes<edm::HepMCProduct>(
          edm::InputTag(iConfig.getUntrackedParameter("moduleLabel", std::string("generator")), "unsmeared"))),
      minPtElectronMuon(iConfig.getUntrackedParameter("MinPtElectronMuon", 0.)),
      maxEtaElectronMuon(iConfig.getUntrackedParameter("MaxEtaElectronMuon", 10.)) {
  //now do what ever initialization is needed
}

HZZ4lFilter::~HZZ4lFilter() {
  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)
}

//
// member functions
//

// ------------ method called on each new Event  ------------
bool HZZ4lFilter::filter(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  using namespace edm;

  bool accepted = false;
  int nLeptons = 0;

  Handle<HepMCProduct> evt;
  iEvent.getByToken(token_, evt);

  const HepMC::GenEvent* myGenEvent = evt->GetEvent();

  for (HepMC::GenEvent::particle_const_iterator p = myGenEvent->particles_begin(); p != myGenEvent->particles_end();
       ++p) {
    if ((*p)->status() != 1)
      continue;

    if ((*p)->momentum().perp() > minPtElectronMuon && fabs((*p)->momentum().eta()) < maxEtaElectronMuon) {
      if (abs((*p)->pdg_id()) == 11 || abs((*p)->pdg_id()) == 13)
        nLeptons++;
    }
    if (nLeptons == 3) {
      accepted = true;
      break;
    }
  }

  delete myGenEvent;

  if (accepted) {
    return true;
  } else {
    return false;
  }
}

/*
// ------------ method called once each job just before starting event loop  ------------

// ------------ method called once each job just after ending the event loop  ------------
void 
HZZ4lFilter::endJob() {
}
*/
