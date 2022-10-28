// -*- C++ -*-
//
// Package:    HFFilter
// Class:      HFFilter
//
/**\class HFFilter HFFilter.cc PhysicsTools/HFFilter/src/HFFilter.cc

 Description: Filter to see if there are heavy flavor GenJets in this event

 Implementation:
     The implementation is simple, it loops over the GenJets and checks if any constituents
     have a pdg ID that matches a list. It also has a switch to count objects from a gluon parent,
     so the user can turn off counting gluon splitting.
*/
//
// Original Author:  "Salvatore Rappoccio"
//         Created:  Tue Apr  8 16:19:45 CDT 2008
//
//

// system include files
#include <memory>
#include <string>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/global/EDFilter.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/JetReco/interface/GenJet.h"

#include "PhysicsTools/JetMCUtils/interface/JetMCTag.h"

#include "DataFormats/HepMCCandidate/interface/GenParticle.h"

//
// class declaration
//

class HFFilter : public edm::global::EDFilter<> {
public:
  explicit HFFilter(const edm::ParameterSet&);

  bool filter(edm::StreamID, edm::Event&, const edm::EventSetup&) const override;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  // ----------member data ---------------------------
  edm::EDGetTokenT<std::vector<reco::GenJet>> genJetsCollToken_;  // Input GenJetsCollection
  double ptMin_;                                                  // Min pt
  double etaMax_;                                                 // Max abs(eta)
};
using namespace std;

//
// constructors and destructor
//
HFFilter::HFFilter(const edm::ParameterSet& iConfig) {
  genJetsCollToken_ = consumes<vector<reco::GenJet>>(iConfig.getParameter<edm::InputTag>("genJetsCollName"));
  ptMin_ = iConfig.getParameter<double>("ptMin");
  etaMax_ = iConfig.getParameter<double>("etaMax");
}

void HFFilter::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("genJetsCollName");
  desc.add<double>("ptMin");
  desc.add<double>("etaMax");

  descriptions.addDefault(desc);
}

//
// member functions
//

// Filter event based on whether there are heavy flavor GenJets in it that satisfy
// pt and eta cuts
bool HFFilter::filter(edm::StreamID, edm::Event& iEvent, const edm::EventSetup& iSetup) const {
  // Get the GenJetCollection
  using namespace edm;
  using namespace reco;
  Handle<std::vector<GenJet>> hGenJets;
  iEvent.getByToken(genJetsCollToken_, hGenJets);

  // Loop over the GenJetCollection
  vector<GenJet>::const_iterator ijet = hGenJets->begin();
  vector<GenJet>::const_iterator end = hGenJets->end();
  for (; ijet != end; ++ijet) {
    // Check to make sure the GenJet satisfies kinematic cuts. Ignore those that don't.
    if (ijet->pt() < ptMin_ || fabs(ijet->eta()) > etaMax_)
      continue;

    // Get the constituent particles
    vector<const GenParticle*> particles = ijet->getGenConstituents();

    // Loop over the constituent particles
    vector<const GenParticle*>::const_iterator genit = particles.begin();
    vector<const GenParticle*>::const_iterator genend = particles.end();
    for (; genit != genend; ++genit) {
      // See if any of them come from B or C hadrons
      const GenParticle& genitref = **genit;
      if (JetMCTagUtils::decayFromBHadron(genitref) || JetMCTagUtils::decayFromCHadron(genitref)) {
        return true;
      }
    }  // end loop over constituents
  }    // end loop over jets

  return false;
}

//define this as a plug-in
DEFINE_FWK_MODULE(HFFilter);
