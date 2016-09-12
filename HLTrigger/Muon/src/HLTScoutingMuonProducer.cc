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
    vertexCollection_(consumes<reco::VertexCollection>(iConfig.getParameter<edm::InputTag>("vertexCollection"))),
    muonPtCut(iConfig.getParameter<double>("muonPtCut")),
    muonEtaCut(iConfig.getParameter<double>("muonEtaCut"))
{
    //register products
    produces<ScoutingMuonCollection>();
    produces<ScoutingVertexCollection>();
}

HLTScoutingMuonProducer::~HLTScoutingMuonProducer()
{ }

// ------------ method called to produce the data  ------------
void HLTScoutingMuonProducer::produce(edm::StreamID sid, edm::Event & iEvent,
                                      edm::EventSetup const & setup) const
{
    using namespace edm;

    std::unique_ptr<ScoutingMuonCollection> outMuons(new ScoutingMuonCollection());

    // Get RecoChargedCandidate
    Handle<reco::RecoChargedCandidateCollection> ChargedCandidateCollection;
    if(!iEvent.getByToken(ChargedCandidateCollection_, ChargedCandidateCollection)){
        iEvent.put(std::move(outMuons));
        return;
    }

    // Get Track
    Handle<reco::TrackCollection> TrackCollection;
    if(!iEvent.getByToken(TrackCollection_, TrackCollection)){
        iEvent.put(std::move(outMuons));
        return;
    }

    // Get EcalPFClusterIsoMap
    Handle<RecoChargedCandMap> EcalPFClusterIsoMap;
    iEvent.getByToken(EcalPFClusterIsoMap_, EcalPFClusterIsoMap);
    // if(!iEvent.getByToken(EcalPFClusterIsoMap_, EcalPFClusterIsoMap)){
      //        iEvent.put(std::move(outMuons));
      //  return;
      // }

    // Get HcalPFClusterIsoMap
    Handle<RecoChargedCandMap> HcalPFClusterIsoMap;
    iEvent.getByToken(HcalPFClusterIsoMap_, HcalPFClusterIsoMap);
    //    if(!iEvent.getByToken(HcalPFClusterIsoMap_, HcalPFClusterIsoMap)){
    //   iEvent.put(std::move(outMuons));
    //   return;
    // }

    // Get TrackIsoMap
    Handle<ValueMap<double>> TrackIsoMap;
    if(!iEvent.getByToken(TrackIsoMap_, TrackIsoMap)){
        iEvent.put(std::move(outMuons));
        return;
    }

    // Produce muons
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

	double ecalisopf=-99.0;
	if  ( !EcalPFClusterIsoMap.isValid() ) ecalisopf = -1.0 ;
	else ecalisopf = (*EcalPFClusterIsoMap)[muonRef]; 

	double hcalisopf=-99.0;
	if  ( !HcalPFClusterIsoMap.isValid() ) hcalisopf = -1.0 ;
	else hcalisopf = (*HcalPFClusterIsoMap)[muonRef]; 

        outMuons->emplace_back(muon.pt(), muon.eta(), muon.phi(),  muon.mass(),
			       //  (*EcalPFClusterIsoMap)[muonRef], (*HcalPFClusterIsoMap)[muonRef],
			       ecalisopf, hcalisopf,
                               (*TrackIsoMap)[muonRef], track->chi2(), track->ndof(),
                               track->charge(), track->dxy(), track->dz(),
                               track->hitPattern().numberOfValidMuonHits(),
                               track->hitPattern().numberOfValidPixelHits(),
                               0, // nMatchedStations
                               track->hitPattern().trackerLayersWithMeasurement(),
                               2, // Global muon
			       track->hitPattern().numberOfValidStripHits(),
			       track->qoverp(),
			       track->lambda(),
			       track->pt(),
			       track->vx(),
			       track->vy(),
			       track->vz(),
			       track->phi(),
			       track->eta(),
			       track->dxyError(),
			       track->dzError(),
			       track->qoverpError(),
			       track->lambdaError(),
			       track->phiError(),
			       track->dsz(),
			       track->dszError()
			       );
    }

    //get vertices
    std::unique_ptr<ScoutingVertexCollection> outVertices(new ScoutingVertexCollection());
    
    Handle<reco::VertexCollection> vertexCollection;
    if(!iEvent.getByToken(vertexCollection_, vertexCollection)){
      iEvent.put(std::move(outVertices));
      return;
    }
    //produce vertices (only if present; otherwise return an empty collection)
    for(auto &vtx : *vertexCollection){
      if ( !vtx.isValid() ) continue ;
      outVertices->emplace_back(
				vtx.x(), vtx.y(), vtx.z(), vtx.zError(), vtx.xError(), vtx.yError(), vtx.tracksSize(), vtx.chi2(), vtx.ndof()
				);
      
    }




    // Put output
    iEvent.put(std::move(outMuons));
    iEvent.put(std::move(outVertices));
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
    desc.add<edm::InputTag>("vertexCollection", edm::InputTag("hltPixelVertices"));
    desc.add<double>("muonPtCut", 4.0);
    desc.add<double>("muonEtaCut", 2.4);
    descriptions.add("hltScoutingMuonProducer", desc);
}
