#ifndef TAU3MURECO_H
#define TAU3MURECO_H

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include <DataFormats/TrackReco/interface/Track.h>
#include <DataFormats/MuonReco/interface/Muon.h>
#include <DataFormats/MuonReco/interface/MuonFwd.h>

class Tau3MuReco
{
 public:
    Tau3MuReco(const edm::ParameterSet& iConfig, edm::ConsumesCollector&& iC);
    ~Tau3MuReco();

    bool doTau3MuReco(const edm::Event& iEvent, const edm::EventSetup& iSetup, reco::MuonCollection* muonCollection, reco::TrackCollection* trackCollection);

 private:

    bool check4MuonTrack(const reco::Track& track); //compares track with reconstructed muons and return true if they are equal
    bool find3rdTrack(const edm::Event& iEvent, const edm::EventSetup& iSetup, const reco::TrackCollection& Tracks); //try to find a 3rd muon in tracks, if this was not detected as a muon
    bool findCorrectPairing(); //find the correct 3 muons, if more than 3 muons has been reconstructed
    double getInvariantMass(const reco::TrackCollection* tracks, const double MuonMass=0.106);
    double getDeltaR(const reco::Track& track1, const reco::Track& track2);
    bool removeIncorrectMuon(); //try to remove one muon, which seems to come not from a tau->3Mu decay

    const double m_kMatchingDeltaR;
    const double m_kMatchingPt;
    const double m_kTauMassCut;
    const double m_kTauMass;
    const double m_kMuonMass;

    const edm::EDGetTokenT<reco::MuonCollection> m_kMuonSourceToken;
    const edm::EDGetTokenT<reco::TrackCollection> m_kTrackSourceToken;

    reco::MuonCollection* m_MuonCollection;
    reco::TrackCollection* m_TrackCollection;
};

#endif
