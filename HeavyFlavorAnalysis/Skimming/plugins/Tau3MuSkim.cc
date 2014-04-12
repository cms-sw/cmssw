// -*- C++ -*-
//
// Package:    Tau3MuSkim
// Class:      Tau3MuSkim
//
/**\class Tau3MuSkim Tau3MuSkim.cc HeavyFlavorAnalysis/Skimming/plugins/Tau3MuSkim.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author: Manuel Giffels <Manuel.Giffels@physik.rwth-aachen.de>
//         Created:  Mon Jul 23 10:19:11 CEST 2007
//
//
//

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"

#include "HeavyFlavorAnalysis/Skimming/interface/Tau3MuReco.h"
#include "HeavyFlavorAnalysis/Skimming/plugins/Tau3MuSkim.h"

Tau3MuSkim::Tau3MuSkim(const edm::ParameterSet& iConfig)
{
    m_Tau3MuReco = new Tau3MuReco(iConfig, consumesCollector());

    produces<reco::MuonCollection,edm::InEvent>("tau3MuCandidateMuons");
    produces<reco::TrackCollection,edm::InEvent>("tau3MuCandidateTracks");
}


Tau3MuSkim::~Tau3MuSkim()
{
    delete m_Tau3MuReco;
}


//
// member functions
//

// ------------ method called on each new Event  ------------
bool
Tau3MuSkim::filter(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
    std::auto_ptr<reco::MuonCollection> tau3MuCandidateMuons(new reco::MuonCollection);
    std::auto_ptr<reco::TrackCollection> tau3MuCandidateTracks(new reco::TrackCollection);

    bool accept = m_Tau3MuReco->doTau3MuReco(iEvent, iSetup, tau3MuCandidateMuons.get(), tau3MuCandidateTracks.get());

    iEvent.put(tau3MuCandidateMuons, "tau3MuCandidateMuons");
    iEvent.put(tau3MuCandidateTracks, "tau3MuCandidateTracks");

    return accept;
}

// ------------ method called once each job just before starting event loop  ------------
void
Tau3MuSkim::beginJob()
{
}

// ------------ method called once each job just after ending the event loop  ------------
void
Tau3MuSkim::endJob()
{
}

//define this as a plug-in
DEFINE_FWK_MODULE(Tau3MuSkim);
