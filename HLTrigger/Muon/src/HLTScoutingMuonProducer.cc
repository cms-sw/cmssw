// -*- C++ -*-
//
// Package:    HLTrigger/Muon
// Class:      HLTScoutingMuonProducer
//
/**\class HLTScoutingMuonProducer HLTScoutingMuonProducer.cc HLTrigger/Muon/src/HLTScoutingMuonProducer.cc

Description: Producer for ScoutingMuon

*/
//
// Original Author:  David G. Sheffield (Rutgers)
//         Created:  Fri, 31 Jul 2015
//
//

#include "HLTrigger/Muon/interface/HLTScoutingMuonProducer.h"

//
// constructors and destructor
//
HLTScoutingMuonProducer::HLTScoutingMuonProducer(const edm::ParameterSet& iConfig):
    ChargedCandidateCollection_(consumes<reco::RecoChargedCandidateCollection>
                                (iConfig.getParameter<edm::InputTag>("ChargedCandidates"))),
    TrackCollection_(consumes<reco::TrackCollection>
                     (iConfig.getParameter<edm::InputTag>("Tracks"))),
    EcalPFClusterIsoMap_(consumes<RecoChargedCandMap>(iConfig.getParameter<edm::InputTag>(
                                                          "EcalPFClusterIsoMap"))),
    HcalPFClusterIsoMap_(consumes<RecoChargedCandMap>(iConfig.getParameter<edm::InputTag>(
                                                          "HcalPFClusterIsoMap"))),
    TrackIsoMap_(consumes<edm::ValueMap<double>>(iConfig.getParameter<edm::InputTag>(
                                                     "TrackIsoMap"))),
    muonPtCut(iConfig.getParameter<double>("muonPtCut")),
    muonEtaCut(iConfig.getParameter<double>("muonEtaCut"))
{
    //register products
    produces<ScoutingMuonCollection>();
}

HLTScoutingMuonProducer::~HLTScoutingMuonProducer()
{ }

// ------------ method called to produce the data  ------------
void HLTScoutingMuonProducer::produce(edm::StreamID sid, edm::Event & iEvent,
                                      edm::EventSetup const & setup) const
{
    using namespace edm;

    // Get RecoChargedCandidate
    Handle<reco::RecoChargedCandidateCollection> ChargedCandidateCollection;
    if(!iEvent.getByToken(ChargedCandidateCollection_, ChargedCandidateCollection)){
        edm::LogError ("HLTScoutingMuonProducer")
            << "invalid collection: ChargedCandidateCollection" << "\n";
        return;
    }

    // Get Track
    Handle<reco::TrackCollection> TrackCollection;
    if(!iEvent.getByToken(TrackCollection_, TrackCollection)){
        edm::LogError ("HLTScoutingMuonProducer")
            << "invalid collection: TrackCollection" << "\n";
        return;
    }

    // Get EcalPFClusterIsoMap
    Handle<RecoChargedCandMap> EcalPFClusterIsoMap;
    if(!iEvent.getByToken(EcalPFClusterIsoMap_, EcalPFClusterIsoMap)){
        edm::LogError ("HLTScoutingMuonProducer")
            << "invalid collection: hltMuonEcalPFClusterIsoForMuons" << "\n";
        return;
    }

    // Get HcalPFClusterIsoMap
    Handle<RecoChargedCandMap> HcalPFClusterIsoMap;
    if(!iEvent.getByToken(HcalPFClusterIsoMap_, HcalPFClusterIsoMap)){
        edm::LogError ("HLTScoutingMuonProducer")
            << "invalid collection: hltMuonHcalPFClusterIsoForMuons" << "\n";
        return;
    }

    // Get TrackIsoMap
    Handle<ValueMap<double>> TrackIsoMap;
    if(!iEvent.getByToken(TrackIsoMap_, TrackIsoMap)){
        edm::LogError ("HLTScoutingMuonProducer")
            << "invalid collection: TrackIsoMap" << "\n";
        return;
    }

    // Produce muons
    std::auto_ptr<ScoutingMuonCollection> outMuons(new ScoutingMuonCollection());
    int index = 0;
    for (auto &muon : *ChargedCandidateCollection) {
        reco::RecoChargedCandidateRef muonRef = getRef(ChargedCandidateCollection, index);
        ++index;
        if (muonRef.isNull() || !muonRef.isAvailable())
            continue;

        reco::TrackRef track = muon.track();
        if (track.isNull() || !track.isAvailable())
            continue;

        if (muon.pt() < muonPtCut)
            continue;
        if (fabs(muon.eta()) > muonEtaCut)
            continue;

        outMuons->emplace_back(muon.pt(), muon.eta(), muon.phi(),  muon.mass(),
                               (*EcalPFClusterIsoMap)[muonRef], (*HcalPFClusterIsoMap)[muonRef],
                               (*TrackIsoMap)[muonRef], track->chi2(), track->ndof(),
                               track->charge(), track->dxy(), track->dz(),
                               track->hitPattern().numberOfValidMuonHits(),
                               track->hitPattern().numberOfValidPixelHits(),
                               0, // nMatchedStations
                               track->hitPattern().trackerLayersWithMeasurement(),
                               2); // Global muon
    }

    // Put output
    iEvent.put(outMuons);
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void HLTScoutingMuonProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;
    desc.add<edm::InputTag>("ChargedCandidates", edm::InputTag("hltL3MuonCandidates"));
    desc.add<edm::InputTag>("Tracks", edm::InputTag("hltL3Muons"));
    desc.add<edm::InputTag>("EcalPFClusterIsoMap", edm::InputTag("hltMuonEcalPFClusterIsoForMuons"));
    desc.add<edm::InputTag>("HcalPFClusterIsoMap", edm::InputTag("hltMuonHcalPFClusterIsoForMuons"));
    desc.add<edm::InputTag>("TrackIsoMap", edm::InputTag(
                                "hltMuonTkRelIsolationCut0p09Map:combinedRelativeIsoDeposits"));
    desc.add<double>("muonPtCut", 4.0);
    desc.add<double>("muonEtaCut", 2.4);
    descriptions.add("scoutingMuonProducer", desc);
}
