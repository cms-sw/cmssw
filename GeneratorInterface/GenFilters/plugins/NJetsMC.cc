// -*- C++ -*-
//
// Package:   NJetsMC
// Class:     NJetsMC
//
/**\class NJetsMC NJetsMC.cc

 Description: Filter for DPS MC generation.

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  "Nathaniel Odell"
//         Created:  Thu Aug 12 09:24:46 CDT 2010
// then moved to more general N-jets purpose in GeneratorInterface/GenFilters
//

#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/global/EDFilter.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "DataFormats/JetReco/interface/GenJetCollection.h"
#include "DataFormats/JetReco/interface/GenJet.h"

#include <cstdint>

class NJetsMC : public edm::global::EDFilter<> {
public:
  explicit NJetsMC(const edm::ParameterSet&);

private:
  bool filter(edm::StreamID, edm::Event&, const edm::EventSetup&) const override;

  // ----------member data ---------------------------

  const edm::EDGetTokenT<reco::GenJetCollection> genToken_;
  const int njets_;
  const double minpt_;
};

NJetsMC::NJetsMC(const edm::ParameterSet& iConfig)
    : genToken_(consumes<reco::GenJetCollection>(iConfig.getUntrackedParameter<edm::InputTag>("GenTag"))),
      njets_(iConfig.getParameter<int32_t>("Njets")),
      minpt_(iConfig.getParameter<double>("MinPt")) {}

bool NJetsMC::filter(edm::StreamID, edm::Event& iEvent, const edm::EventSetup&) const {
  edm::Handle<reco::GenJetCollection> genJets;
  iEvent.getByToken(genToken_, genJets);

  int count = 0;
  bool result = false;

  for (reco::GenJetCollection::const_iterator iJet = genJets->begin(); iJet != genJets->end(); ++iJet) {
    reco::GenJet myJet = reco::GenJet(*iJet);

    if (myJet.pt() > minpt_)
      ++count;
  }

  if (count >= njets_)
    result = true;

  return result;
}
//define this as a plug-in
DEFINE_FWK_MODULE(NJetsMC);
