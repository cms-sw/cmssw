/* \class GenJetParticleSelector
*
*  Selects particles that are used as input for the GenJet collection.
*  Logic: select all stable particles, except for neutrinos that come from
*  W,Z and H decays, and except for invisible BSM particles.
*
*  Usage:
*
*  module genjetparticles = GenJetParticleSelector {
*        InputTag src = genParticles
*  }
*
* \author: Fedor Ratnikov, UMd by suggestion from Filip Moortgat
* $Id: GenJetParticleSelector.cc,v 1.1 2008/03/06 00:55:36 fedor Exp $
*/

#include "GenJetParticleSelector.h"

#include <memory>

#include "DataFormats/Common/interface/EDProduct.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "PhysicsTools/CandUtils/interface/pdgIdUtils.h"

namespace {
  inline bool isInvisibleBSMParticle (const reco::GenParticle& part) {
    long pdgId = abs(part.pdgId());
    return 
      pdgId == 1000022 || // ~chi_10
      pdgId == 2000012 || // ~nu_eR 
      pdgId == 2000014 || // ~nu_muR
      pdgId == 2000016 || // ~nu_tauR
      pdgId == 1000039 || // ~Gravitino
      pdgId == 5000039 || // Graviton*
      pdgId == 4000012 || // nu*_e0
      pdgId == 9900012 || // nu_Re
      pdgId == 9900014 || // nu_Rmu
      pdgId == 9900016 || // nu_Rtau
      pdgId == 39; // Graviton
  }

  inline bool isPromptNeutrino (const reco::GenParticle& part) {
    if (reco::isNeutrino (part)) {
      const reco::Candidate* mother = part.mother();
      if(mother) {
        int motherid = mother->pdgId();
        // pdg id from 23 to 39 are bosons
        if (abs(motherid)>=23 && abs(motherid)<=39) return true;
      }
    }
    return false;
  }
}


GenJetParticleSelector::GenJetParticleSelector (const edm::ParameterSet& ps) 
  : mSrc(ps.getParameter<edm::InputTag>( "src" ))
{
  produces <reco::GenParticleRefVector> ();
}

GenJetParticleSelector::~GenJetParticleSelector () {}

void GenJetParticleSelector::produce(edm::Event& e, const edm::EventSetup& c) {
  // get input
  edm::Handle<reco::GenParticleCollection> input; 
  e.getByLabel( mSrc, input);
  // make output
  std::auto_ptr<reco::GenParticleRefVector> output (new reco::GenParticleRefVector);
  for (size_t iPart = 0; iPart < input->size (); ++iPart) {
    const reco::GenParticle& part = (*input)[iPart];
    if (part.status () != 1) continue; // not stable
    if (isInvisibleBSMParticle (part)) continue; // invisible BSM
    if (isPromptNeutrino (part)) continue; // not prompt neutrino
    output->push_back (edm::Ref<reco::GenParticleCollection> (input, iPart));
  }
  e.put (output);
}

