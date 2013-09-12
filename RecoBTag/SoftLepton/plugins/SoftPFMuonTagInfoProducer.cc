#include "RecoBTag/SoftLepton/plugins/SoftPFMuonTagInfoProducer.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrackFwd.h"
#include "DataFormats/Math/interface/deltaR.h"
#include "DataFormats/Math/interface/Vector3D.h"
#include "DataFormats/Math/interface/Point3D.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "RecoParticleFlow/PFProducer/interface/Utils.h"
#include "DataFormats/JetReco/interface/PFJetCollection.h"
#include "DataFormats/JetReco/interface/Jet.h"
#include "DataFormats/PatCandidates/interface/Jet.h"
// Muons
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "DataFormats/MuonReco/interface/MuonSelectors.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateFwd.h"
#include "DataFormats/BTauReco/interface/SoftLeptonTagInfo.h"

// Transient Track and IP
#include "TrackingTools/TransientTrack/interface/TransientTrack.h"
#include "TrackingTools/TransientTrack/interface/TransientTrackBuilder.h"
#include "TrackingTools/Records/interface/TransientTrackRecord.h"
#include "TrackingTools/IPTools/interface/IPTools.h"
#include "DataFormats/GeometryVector/interface/GlobalVector.h"
#include <cmath>

SoftPFMuonTagInfoProducer::SoftPFMuonTagInfoProducer (const edm::ParameterSet& conf):
 PVerTag_(conf.getParameter<edm::InputTag>("primaryVertex") ),
 PFJet_  (conf.getParameter<edm::InputTag>("jets")  ),
 MuonId_ (conf.getParameter<int>          ("MuonId") )
{
	muonId=MuonId_;
	produces<reco::SoftLeptonTagInfoCollection>();
}

SoftPFMuonTagInfoProducer::~SoftPFMuonTagInfoProducer()
{

}

void SoftPFMuonTagInfoProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  	reco::SoftLeptonTagInfoCollection *MuonTI = new reco::SoftLeptonTagInfoCollection;		

	edm::ESHandle<TransientTrackBuilder> builder;
 	iSetup.get<TransientTrackRecord>().get("TransientTrackBuilder", builder);
 	transientTrackBuilder=builder.product();
 
 	edm::Handle<reco::VertexCollection> PVCollection;
 	iEvent.getByLabel(PVerTag_, PVCollection);
// 	if(!PVCollection.isValid() || PVCollection->empty()) return;
 	if(!PVCollection.isValid()) return;
 	if(!PVCollection->empty()){
		goodvertex = true;
		vertex=&PVCollection->front();
	}else goodvertex = false; 
	
 	std::vector<edm::RefToBase<reco::Jet> > jets;
 
 	edm::Handle<edm::View<reco::Jet> > inputJets;
 	iEvent.getByLabel(PFJet_, inputJets);
 	unsigned int size = inputJets->size();
 	jets.resize(size);
 	for (unsigned int i = 0; i < size; i++){
 		jets[i] = inputJets->refAt(i);
 		reco::PFCandidateCollection Muon;
 		const std::vector<reco::CandidatePtr> JetConst = jets[i]->getJetConstituents();
     		for (unsigned ic=0;ic<JetConst.size();++ic){
  	     		const reco::PFCandidate* pfc = dynamic_cast <const reco::PFCandidate*> (JetConst[ic].get());
			if(JetConst[ic].get()!=NULL && pfc==NULL) continue; 
 			if(pfc->particleId()==3){
 				if(!isMuonClean(iEvent,pfc))continue;
 				Muon.push_back(*(pfc));
       		}
	}
        reco::SoftLeptonTagInfo result_muons     = tagMuon( jets[i], Muon );
        MuonTI->push_back(result_muons);

  }
  std::auto_ptr<reco::SoftLeptonTagInfoCollection> MuonTagInfoCollection(MuonTI);
  iEvent.put(MuonTagInfoCollection);
}



reco::SoftLeptonTagInfo SoftPFMuonTagInfoProducer::tagMuon (
    const edm::RefToBase<reco::Jet> & jet,
    reco::PFCandidateCollection     & leptons
) {
  reco::SoftLeptonTagInfo info;
  info.setJetRef( jet );
  if(goodvertex){	
	  for(reco::PFCandidateCollection::const_iterator lepton = leptons.begin(); lepton != leptons.end(); ++lepton) {
			reco::SoftLeptonProperties properties=fillMuonProperties(*lepton, *jet);
    	const reco::Muon* muon=&*lepton->muonRef();
    	reco::TrackBaseRef trkRef(muon->globalTrack().isNonnull() ? muon->globalTrack() : muon->innerTrack());
    	info.insert(trkRef, properties );
		}
	}
  return info;
}


reco::SoftLeptonProperties SoftPFMuonTagInfoProducer::fillMuonProperties(const reco::PFCandidate &muon, const reco::Jet &jet) {
  reco::SoftLeptonProperties prop;
  // Calculate Soft Muon Tag Info, from SoftLepton.cc
  reco::TransientTrack transientTrack=transientTrackBuilder->build(muon.trackRef());
  prop.sip2d    = IPTools::signedTransverseImpactParameter(transientTrack, GlobalVector(jet.px(), jet.py(), jet.pz()), *vertex).second.significance();
  prop.sip3d    = IPTools::signedImpactParameter3D(transientTrack, GlobalVector(jet.px(), jet.py(), jet.pz()), *vertex).second.significance();
  prop.deltaR   = deltaR(jet, muon);
  prop.ptRel    = ( (jet.p4().Vect()-muon.p4().Vect()).Cross(muon.p4().Vect()) ).R() / jet.p4().Vect().R(); // | (Pj-Pu) X Pu | / | Pj |
  float mag = muon.p4().Vect().R()*jet.p4().Vect().R();
  float dot = muon.p4().Dot(jet.p4());
  prop.etaRel   = -log((mag - dot)/(mag + dot)) / 2.;
  prop.ratio    = muon.muonRef().get()->p() / jet.energy();
  prop.ratioRel = muon.muonRef().get()->p4().Dot(jet.p4()) / jet.p4().Vect().Mag2();
  return prop;
}


bool SoftPFMuonTagInfoProducer::isLooseMuon(const reco::Muon* muon) {
  return muon->isPFMuon() && (muon->isGlobalMuon() || muon->isTrackerMuon());
}
bool SoftPFMuonTagInfoProducer::isSoftMuon(const reco::Muon* muon) {
  return  muon::isGoodMuon(*muon, muon::TMOneStationTight)
    && muon->track()->hitPattern().trackerLayersWithMeasurement()       > 5
    && muon->innerTrack()->hitPattern().pixelLayersWithMeasurement()    > 1
    && muon->muonBestTrack()->normalizedChi2()                          < 1.8
    && muon->innerTrack()->dxy(vertex->position())                      < 3.
    && muon->innerTrack()->dz(vertex->position())                       < 30.
  ;
}
bool SoftPFMuonTagInfoProducer::isTightMuon(const reco::Muon* muon) {
  return  muon->isGlobalMuon()
    && muon->isPFMuon()
    && muon->muonBestTrack()->normalizedChi2()                          < 10.
    && (muon->globalTrack().isNonnull() ? muon->globalTrack()->hitPattern().numberOfValidMuonHits() : -1)        > 0
    && muon->numberOfMatchedStations()                                  > 1
    && fabs(muon->muonBestTrack()->dxy(vertex->position()))             < 0.2
    && fabs(muon->muonBestTrack()->dz(vertex->position()))              < 0.5
    && muon->innerTrack()->hitPattern().numberOfValidPixelHits()        > 0
    && muon->track()->hitPattern().trackerLayersWithMeasurement()       > 5
  ;
}


bool SoftPFMuonTagInfoProducer::isMuonClean(edm::Event& iEvent,const reco::PFCandidate* PFcandidate){
	const reco::Muon* muon=PFcandidate->muonRef().get();
 	if(muonId>=0 && !isLooseMuon(muon)) return false;
      	if(muonId>=1 && !isSoftMuon (muon)) return false;
      	if(muonId>=2 && !isTightMuon(muon)) return false;
	return true;
}


