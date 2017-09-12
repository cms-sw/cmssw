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
#include "DataFormats/Math/interface/deltaR.h"
#include "TMath.h"

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
    displacedvertexCollection_(consumes<reco::VertexCollection>(iConfig.getParameter<edm::InputTag>("displacedvertexCollection"))),
    muonPtCut(iConfig.getParameter<double>("muonPtCut")),
    muonEtaCut(iConfig.getParameter<double>("muonEtaCut")),
    minVtxProbCut(iConfig.getParameter<double>("minVtxProbCut"))
{
    //register products
    produces<ScoutingMuonCollection>();
    produces<ScoutingVertexCollection>("displacedVtx");
}

HLTScoutingMuonProducer::~HLTScoutingMuonProducer() = default;

// ------------ method called to produce the data  ------------
void HLTScoutingMuonProducer::produce(edm::StreamID sid, edm::Event & iEvent,
                                      edm::EventSetup const & setup) const
{
    using namespace edm;

    std::unique_ptr<ScoutingMuonCollection> outMuons(new ScoutingMuonCollection());
    std::unique_ptr<ScoutingVertexCollection> dispVertices(new ScoutingVertexCollection());

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

    // Get HcalPFClusterIsoMap
    Handle<RecoChargedCandMap> HcalPFClusterIsoMap;
    iEvent.getByToken(HcalPFClusterIsoMap_, HcalPFClusterIsoMap);

    // Get TrackIsoMap
    Handle<ValueMap<double>> TrackIsoMap;
    if(!iEvent.getByToken(TrackIsoMap_, TrackIsoMap)){
        iEvent.put(std::move(outMuons));
        return;
    }

    std::pair<reco::RecoChargedCandidate,reco::RecoChargedCandidate> ivtxMuPair;
    std::vector<std::pair<reco::RecoChargedCandidate,reco::RecoChargedCandidate> > vtxMuPair;
    
    //get displaced vertices
    Handle<reco::VertexCollection> displacedvertexCollection;
    if(iEvent.getByToken(displacedvertexCollection_, displacedvertexCollection)){
      
      for(auto &dispvtx : *displacedvertexCollection){
	if ( !dispvtx.isValid() ) continue ;
	float vtxProb = 0.0;
	if( (dispvtx.chi2()>=0.0) && (dispvtx.ndof()>0) ) vtxProb = TMath::Prob(dispvtx.chi2(), dispvtx.ndof() );
	if (vtxProb < minVtxProbCut) continue;
	
	// Get the 2 tracks associated to displaced vertex
	auto trackIt =  dispvtx.tracks_begin();
	reco::TrackRef vertextkRef1 =  (*trackIt).castTo<reco::TrackRef>() ;
	trackIt++;
	reco::TrackRef vertextkRef2 =  (*trackIt).castTo<reco::TrackRef>();
	
	// Get the muons associated with the tracks
	int iFoundRefs = 0;
	for (auto const & cand : *ChargedCandidateCollection) {
	  reco::TrackRef tkRef = cand.get<reco::TrackRef>();
	  if(tkRef == vertextkRef1) {ivtxMuPair.first= cand; iFoundRefs++ ;}
	  if(tkRef == vertextkRef2) {ivtxMuPair.second= cand; iFoundRefs++ ;}
	}
	if (iFoundRefs<2) continue;
	vtxMuPair.push_back(ivtxMuPair);
	
	dispVertices->emplace_back(
				   dispvtx.x(), dispvtx.y(), dispvtx.z(), 
				   dispvtx.zError(), dispvtx.xError(), 
				   dispvtx.yError(), dispvtx.tracksSize(), 
				   dispvtx.chi2(), dispvtx.ndof(), dispvtx.isValid()
				   );
	
      }
    }
    
    // Produce muons
    float minDR2=1e-06;
    int index = 0;
    for (auto &muon : *ChargedCandidateCollection) {
      reco::RecoChargedCandidateRef muonRef = getRef(ChargedCandidateCollection, index);
      std::vector<int> vtxInd;
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
      
      double ecalisopf=-1.0;
      if ( EcalPFClusterIsoMap.isValid()) { ecalisopf = (*EcalPFClusterIsoMap)[muonRef]; }
      
      double hcalisopf=-1.0;
      if ( HcalPFClusterIsoMap.isValid()) { hcalisopf = (*HcalPFClusterIsoMap)[muonRef]; }
      
      for (unsigned int i=0; i<vtxMuPair.size(); i++) {
	float dr2_1 = reco::deltaR2( ((vtxMuPair[i]).first),muon );
	float dr2_2 = reco::deltaR2( ((vtxMuPair[i]).second),muon );
	if ( (dr2_1<minDR2) || (dr2_2<minDR2) )  vtxInd.push_back(i) ;
      }
      
      outMuons->emplace_back(muon.pt(), muon.eta(), muon.phi(),  muon.mass(),
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
			     track->phi(),
			     track->eta(),
			     track->dxyError(),
			     track->dzError(),
			     track->qoverpError(),
			     track->lambdaError(),
			     track->phiError(),
			     track->dsz(),
			     track->dszError(),
			     vtxInd
			     );
      vtxInd.clear();
    }
    
    // Put output
    iEvent.put(std::move(outMuons));
    iEvent.put(std::move(dispVertices), "displacedVtx");
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
    desc.add<edm::InputTag>("displacedvertexCollection", edm::InputTag("hltDisplacedmumuVtxProducerDoubleMu3NoVtx"));
    desc.add<double>("muonPtCut", 4.0);
    desc.add<double>("muonEtaCut", 2.4);
    desc.add<double>("minVtxProbCut", 0.001);
    descriptions.add("hltScoutingMuonProducer", desc);
}
