//
// $Id: PATMuonCleaner.cc,v 1.1 2008/03/06 09:23:10 llista Exp $
//

#include "PhysicsTools/PatAlgos/plugins/PATMuonCleaner.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/Candidate/interface/CandAssociation.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"

#include "TMath.h"

#include <vector>
#include <memory>

using pat::PATMuonCleaner;

PATMuonCleaner::PATMuonCleaner(const edm::ParameterSet & iConfig) :
  muonSrc_(iConfig.getParameter<edm::InputTag>( "muonSource" )),
  helper_(muonSrc_),
  selectionCfg_(iConfig.getParameter<edm::ParameterSet>("selection")),
  selector_(reco::modules::make<MuonSelector>(selectionCfg_))
{
  helper_.configure(iConfig);      // learn whether to save good, bad, all, ...
  helper_.registerProducts(*this); // issue the produces<>() commands
}


PATMuonCleaner::~PATMuonCleaner() {
}


void PATMuonCleaner::produce(edm::Event & iEvent, const edm::EventSetup & iSetup) {
  // start a new event
  helper_.newEvent(iEvent);
  
  for (size_t idx = 0, size = helper_.srcSize(); idx < size; ++idx) {
    // read the source muon
    const reco::Muon &srcMuon = helper_.srcAt(idx);    

    // clone the muon so we can modify it
    reco::Muon ourMuon = srcMuon; 

    // perform the selection
    if ( selector_.filter(idx,helper_.source()) ) continue; 

    // write the muon
    helper_.addItem(idx, ourMuon); 
  }

  // tell him that we're done.
  helper_.done(); // he does event.put by itself

}

void PATMuonCleaner::endJob()  { 
    edm::LogVerbatim("PATLayer0Summary|PATMuonCleaner") << "PATMuonCleaner end job. Input tag was " << muonSrc_.encode();
    helper_.endJob(); 
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(PATMuonCleaner);
