//
// $Id: PATMuonCleaner.cc,v 1.2 2008/03/12 16:13:27 gpetrucc Exp $
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
  isolator_(iConfig.exists("isolation") ? iConfig.getParameter<edm::ParameterSet>("isolation") : edm::ParameterSet() ),
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
  if (isolator_.enabled()) isolator_.beginEvent(iEvent);
  
  for (size_t idx = 0, size = helper_.srcSize(); idx < size; ++idx) {
    // read the source muon
    const reco::Muon &srcMuon = helper_.srcAt(idx);    

    // clone the muon so we can modify it
    reco::Muon ourMuon = srcMuon; 
    size_t selIdx = helper_.addItem(idx, ourMuon);

    // perform the selection
    if ( selector_.filter(idx,helper_.source()) ) {
        helper_.addMark(selIdx, pat::Flags::Selection::Bit0); // opaque, at the moment
    }

    // Add the muon

    // Isolation
    if (isolator_.enabled()) {
        uint32_t isolationWord = isolator_.test( helper_.source(), idx );
        helper_.addMark(selIdx, isolationWord);
    }

  }

  // tell him that we're done.
  helper_.done(); // he does event.put by itself
  if (isolator_.enabled()) isolator_.endEvent();
}

void PATMuonCleaner::endJob()  { 
    edm::LogVerbatim("PATLayer0Summary|PATMuonCleaner") << "PATMuonCleaner end job.\n" << 
            "Input tag was " << muonSrc_.encode() <<
            "\nIsolation information:\n" <<
            isolator_.printSummary() <<
            "\nCleaner summary information:\n" <<
            helper_.printSummary();

    helper_.endJob(); 
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(PATMuonCleaner);
