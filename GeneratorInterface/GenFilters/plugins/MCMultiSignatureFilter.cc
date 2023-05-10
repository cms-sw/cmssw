// -*- C++ -*-
//
// Package:    MCMultiSignatureFilter
// Class:      MCMultiSignatureFilter
//
/*

 Description: select events satistify any or all of signatures

 Implementation: derived from MCMultiParticleFilter

*/

#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/global/EDFilter.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"

#include <cmath>
#include <cstdlib>
#include <vector>

//
// class declaration
//

class MCMultiSignatureFilter : public edm::global::EDFilter<> {
public:
  explicit MCMultiSignatureFilter(const edm::ParameterSet&);

private:
  bool filter(edm::StreamID, edm::Event&, const edm::EventSetup&) const override;

  // ----------member data ---------------------------

  const edm::EDGetTokenT<edm::HepMCProduct> token_;
  const bool selectAny_;               // if true select events that satistify any signature, otherwise all
  const std::vector<int> particleID_;  // particle IDs to look for
  std::vector<double> ptMin_;   // minimum Pt of particles
  std::vector<double> etaMax_;  // maximum fabs(eta) of particles
  std::vector<int> status_;     // status of particles
  std::vector<int> minN_;       // minimum number of particles
};

MCMultiSignatureFilter::MCMultiSignatureFilter(const edm::ParameterSet& iConfig)
    : token_(consumes<edm::HepMCProduct>(
          iConfig.getUntrackedParameter<edm::InputTag>("src", edm::InputTag("generator", "unsmeared")))),
      selectAny_(iConfig.getParameter<bool>("SelectAny")),
      particleID_(iConfig.getParameter<std::vector<int> >("ParticleID")),
      ptMin_(iConfig.getParameter<std::vector<double> >("PtMin")),
      etaMax_(iConfig.getParameter<std::vector<double> >("EtaMax")),
      status_(iConfig.getParameter<std::vector<int> >("Status")),
      minN_(iConfig.getParameter<std::vector<int> >("MinN"))
{
  if ( particleID_.size() != ptMin_.size() ||
       particleID_.size() != etaMax_.size() ||
       particleID_.size() != status_.size() ||
       particleID_.size() != minN_.size() )
    throw cms::Exception("BadConfig") << 
      "MCMultiSignatureFilter: parameter vectors must have the same length";
}

// ------------ method called to skim the data  ------------
bool MCMultiSignatureFilter::filter(edm::StreamID, edm::Event& iEvent, const edm::EventSetup&) const {
  edm::Handle<edm::HepMCProduct> evt;
  iEvent.getByToken(token_, evt);

  unsigned int nSignatures(particleID_.size());
  std::vector<int> searchResults(nSignatures, 0);

  const HepMC::GenEvent* genEvent = evt->GetEvent();

  for (HepMC::GenEvent::particle_const_iterator p = genEvent->particles_begin(); 
       p != genEvent->particles_end(); ++p) {
    auto genParticle = *p;
    for (unsigned int i = 0; i < nSignatures; ++i){
      if ( abs(genParticle->pdg_id()) != abs(particleID_[i]) or
	   genParticle->momentum().perp() < ptMin_[i] or
	   fabs(genParticle->momentum().eta()) > etaMax_[i] or
	   genParticle->status() != status_[i]) 
	continue;

      searchResults[i] += 1;
    }
  }
  
  bool passedAll(true);
  bool passedAny(false);
  for (unsigned int i = 0; i < nSignatures; ++i){
    bool passed = searchResults[i] >= minN_[i];
    passedAll = passedAll and passed;
    passedAny = passedAny or passed;
  }
  
  if (selectAny_)
    return passedAny;
  else
    return passedAll;
}

//define this as a plug-in
DEFINE_FWK_MODULE(MCMultiSignatureFilter);
