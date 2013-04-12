#include "RecoBTag/SoftLepton/plugins/SoftPFLeptonTagInfoProducer.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectronFwd.h"
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
#include "RecoEgamma/EgammaTools/interface/ConversionTools.h"

SoftPFLeptonTagInfoProducer::SoftPFLeptonTagInfoProducer (const edm::ParameterSet& conf):
 PVerTag_(conf.getParameter<edm::InputTag>("primaryVertex") ),
 PFJet_  (conf.getParameter<edm::InputTag>("jets")  ),
 MuonId_ (conf.getParameter<int>          ("MuonId") ),
 SPFELabel_ (conf.getParameter<std::string>("SPFELabel")),
 SPFMLabel_ (conf.getParameter<std::string>("SPFMLabel")),
 SPFETILabel_ (conf.getParameter<std::string>("SPFELabel")),
 SPFMTILabel_ (conf.getParameter<std::string>("SPFMLabel"))
{
	muonId=MuonId_;
        produces<reco::SoftLeptonTagInfoCollection>(SPFETILabel_);
	produces<reco::SoftLeptonTagInfoCollection>(SPFMTILabel_);
}

SoftPFLeptonTagInfoProducer::~SoftPFLeptonTagInfoProducer()
{

}

void SoftPFLeptonTagInfoProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  	reco::SoftLeptonTagInfoCollection *ElecTI = new reco::SoftLeptonTagInfoCollection;
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
	
  edm::Handle<reco::BeamSpot> bsHandle;
  iEvent.getByLabel("offlineBeamSpot", bsHandle);
  const reco::BeamSpot &beamspot = *bsHandle.product();

  edm::Handle<reco::ConversionCollection> hConversions;
  iEvent.getByLabel("allConversions", hConversions);
 
 	std::vector<edm::RefToBase<reco::Jet> > jets;
 
 	edm::Handle<edm::View<reco::Jet> > inputJets;
 	iEvent.getByLabel(PFJet_, inputJets);
 	unsigned int size = inputJets->size();
 	jets.resize(size);
 	for (unsigned int i = 0; i < size; i++){
 		jets[i] = inputJets->refAt(i);
 		reco::PFCandidateCollection Elec;
 		reco::PFCandidateCollection Muon;
 		const std::vector<reco::CandidatePtr> JetConst = jets[i]->getJetConstituents();
     		for (unsigned ic=0;ic<JetConst.size();++ic){
       		const reco::PFCandidate* pfc = dynamic_cast <const reco::PFCandidate*> (JetConst[ic].get());
 
 			if(pfc->particleId()==2){
				bool matchConv=ConversionTools::hasMatchedConversion(*((*pfc).gsfElectronRef().get()),hConversions,beamspot.position());
 				if(!isElecClean(iEvent,pfc) || matchConv)continue;
 				reco::TransientTrack transientTrack=transientTrackBuilder->build(pfc->gsfTrackRef().get());
  				float sip3d= IPTools::signedImpactParameter3D(transientTrack, GlobalVector(jets[i]->px(), jets[i]->py(), jets[i]->pz()), *vertex).second.significance();
 				if(fabs(sip3d<200)) Elec.push_back(*(pfc));			
 			}
 			if(pfc->particleId()==3){
 				if(!isMuonClean(iEvent,pfc))continue;
 				Muon.push_back(*(pfc));
       		}
	}
//	std::cout<< "push_back the softleptontaginfo..." << std::endl;
        reco::SoftLeptonTagInfo result_electrons = tagElec( jets[i], Elec );
        reco::SoftLeptonTagInfo result_muons     = tagMuon( jets[i], Muon );
        ElecTI->push_back(result_electrons);
        MuonTI->push_back(result_muons);

  }
  std::auto_ptr<reco::SoftLeptonTagInfoCollection> ElecTagInfoCollection(ElecTI);
  std::auto_ptr<reco::SoftLeptonTagInfoCollection> MuonTagInfoCollection(MuonTI);
  iEvent.put(ElecTagInfoCollection,SPFETILabel_);
  iEvent.put(MuonTagInfoCollection,SPFMTILabel_);
}


reco::SoftLeptonTagInfo SoftPFLeptonTagInfoProducer::tagElec (
    const edm::RefToBase<reco::Jet> & jet,
    reco::PFCandidateCollection    & leptons
) {
  reco::SoftLeptonTagInfo info;
  info.setJetRef( jet );
  if(goodvertex){	
  	for(reco::PFCandidateCollection::const_iterator lepton = leptons.begin(); lepton != leptons.end(); ++lepton) {
    	reco::TrackBaseRef trkRef(lepton->gsfTrackRef());
    	reco::SoftLeptonProperties properties=fillElecProperties(*lepton, *jet);
    	info.insert(trkRef, properties );
		}	
	}
  return info;
}


reco::SoftLeptonTagInfo SoftPFLeptonTagInfoProducer::tagMuon (
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


reco::SoftLeptonProperties SoftPFLeptonTagInfoProducer::fillElecProperties(const reco::PFCandidate &elec, const reco::Jet &jet) {
  reco::SoftLeptonProperties prop;
  reco::TransientTrack transientTrack=transientTrackBuilder->build(elec.gsfTrackRef().get());
  prop.sip2d    = IPTools::signedTransverseImpactParameter(transientTrack, GlobalVector(jet.px(), jet.py(), jet.pz()), *vertex).second.significance();
  prop.sip3d    = IPTools::signedImpactParameter3D(transientTrack, GlobalVector(jet.px(), jet.py(), jet.pz()), *vertex).second.significance();
  prop.deltaR   = deltaR(jet, elec);
  prop.ptRel    = ( (jet.p4().Vect()-elec.gsfElectronRef().get()->p4().Vect()).Cross(elec.gsfElectronRef().get()->p4().Vect()) ).R() / jet.p4().Vect().R(); // | (Pj-Pu) X Pu | / | Pj |
  float mag = elec.gsfElectronRef().get()->p4().Vect().R()*jet.p4().Vect().R();
  float dot = elec.gsfElectronRef().get()->p4().Dot(jet.p4());
  prop.etaRel   = -log((mag - dot)/(mag + dot)) / 2.;
  prop.ratio    = elec.gsfElectronRef().get()->p() / jet.energy();
  prop.ratioRel = elec.gsfElectronRef().get()->p4().Dot(jet.p4()) / jet.p4().Vect().Mag2();
  return prop;
}



reco::SoftLeptonProperties SoftPFLeptonTagInfoProducer::fillMuonProperties(const reco::PFCandidate &muon, const reco::Jet &jet) {
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


bool SoftPFLeptonTagInfoProducer::isLooseMuon(const reco::Muon* muon) {
  return muon->isPFMuon() && (muon->isGlobalMuon() || muon->isTrackerMuon());
}
bool SoftPFLeptonTagInfoProducer::isSoftMuon(const reco::Muon* muon) {
  return  muon::isGoodMuon(*muon, muon::TMOneStationTight)
    && muon->track()->hitPattern().trackerLayersWithMeasurement()       > 5
    && muon->innerTrack()->hitPattern().pixelLayersWithMeasurement()    > 1
    && muon->muonBestTrack()->normalizedChi2()                          < 1.8
    && muon->innerTrack()->dxy(vertex->position())                      < 3.
    && muon->innerTrack()->dz(vertex->position())                       < 30.
  ;
}
bool SoftPFLeptonTagInfoProducer::isTightMuon(const reco::Muon* muon) {
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


bool SoftPFLeptonTagInfoProducer::isMuonClean(edm::Event& iEvent,const reco::PFCandidate* PFcandidate){
	const reco::Muon* muon=PFcandidate->muonRef().get();
 	if(muonId>=0 && !isLooseMuon(muon)) return false;
      	if(muonId>=1 && !isSoftMuon (muon)) return false;
      	if(muonId>=2 && !isTightMuon(muon)) return false;
	return true;
}

bool SoftPFLeptonTagInfoProducer::isElecClean(edm::Event& iEvent,const reco::PFCandidate* PFcandidate)
{
	const reco::HitPattern& hitPattern = PFcandidate->gsfTrackRef().get()->hitPattern();
        //check that the first hit is a pixel hit
        uint32_t hit = hitPattern.getHitPattern(0);
        bool hitCondition= !(hitPattern.validHitFilter(hit) && ( (hitPattern.pixelBarrelHitFilter(hit) && hitPattern.getLayer(hit) < 3) || hitPattern.pixelEndcapHitFilter(hit))); 
	if(hitCondition) return false;
   //     if(PFcandidate->mva_e_pi()<0.6)return false;

  return true;
}

