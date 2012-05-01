#include <algorithm>
#include <cmath>
#include <vector>

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/RecoCandidate/interface/RecoCandidate.h"
#include "DataFormats/RecoCandidate/interface/RecoChargedCandidate.h"

#include "TrackingTools/TransientTrack/interface/TransientTrackBuilder.h"
#include "TrackingTools/Records/interface/TransientTrackRecord.h"
 
#include "RecoVertex/KalmanVertexFit/interface/KalmanVertexFitter.h"
#include "RecoVertex/VertexPrimitives/interface/TransientVertex.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/HLTReco/interface/TriggerFilterObjectWithRefs.h"

#include "DataFormats/BeamSpot/interface/BeamSpot.h"

#include "DataFormats/Math/interface/deltaPhi.h"

#include "HLTmmkFilter.h"

using namespace edm;
using namespace reco;
using namespace std;
using namespace trigger;


// ----------------------------------------------------------------------
HLTmmkFilter::HLTmmkFilter(const edm::ParameterSet& iConfig):thirdTrackMass_(iConfig.getParameter<double>("ThirdTrackMass")),
                                                             maxEta_(iConfig.getParameter<double>("MaxEta")),
                                                             minPt_(iConfig.getParameter<double>("MinPt")),
                                                             minInvMass_(iConfig.getParameter<double>("MinInvMass")),
                                                             maxInvMass_(iConfig.getParameter<double>("MaxInvMass")),
                                                             maxNormalisedChi2_(iConfig.getParameter<double>("MaxNormalisedChi2")),
                                                             minLxySignificance_(iConfig.getParameter<double>("MinLxySignificance")),
                                                             minCosinePointingAngle_(iConfig.getParameter<double>("MinCosinePointingAngle")),
                                                             fastAccept_(iConfig.getParameter<bool>("FastAccept")),
							     saveTags_ (iConfig.getParameter<bool>("saveTags")),
							     beamSpotTag_ (iConfig.getParameter<edm::InputTag> ("BeamSpotTag")){

  muCandLabel_   = iConfig.getParameter<edm::InputTag>("MuCand");
  trkCandLabel_  = iConfig.getParameter<edm::InputTag>("TrackCand");
  
  produces<VertexCollection>();
  produces<CandidateCollection>();
  produces<trigger::TriggerFilterObjectWithRefs>();

}


// ----------------------------------------------------------------------
HLTmmkFilter::~HLTmmkFilter() {

}


// ----------------------------------------------------------------------
void HLTmmkFilter::beginJob() {

}


// ----------------------------------------------------------------------
void HLTmmkFilter::endJob() {
 	
}


// ----------------------------------------------------------------------
bool HLTmmkFilter::filter(edm::Event& iEvent, const edm::EventSetup& iSetup) {

  const double MuMass(0.106);
  const double MuMass2(MuMass*MuMass);
  
  const double thirdTrackMass2(thirdTrackMass_*thirdTrackMass_);
  
  // The filter object
  auto_ptr<TriggerFilterObjectWithRefs> filterobject (new TriggerFilterObjectWithRefs(path(),module()));
  
  auto_ptr<CandidateCollection> output(new CandidateCollection());    
  auto_ptr<VertexCollection> vertexCollection(new VertexCollection());
  
  //get the transient track builder:
  edm::ESHandle<TransientTrackBuilder> theB;
  iSetup.get<TransientTrackRecord>().get("TransientTrackBuilder",theB);

  //get the beamspot position
  edm::Handle<reco::BeamSpot> recoBeamSpotHandle;
  iEvent.getByLabel(beamSpotTag_,recoBeamSpotHandle);
  const reco::BeamSpot& vertexBeamSpot = *recoBeamSpotHandle;

  // Ref to Candidate object to be recorded in filter object
  RecoChargedCandidateRef refMu1;
  RecoChargedCandidateRef refMu2;
  RecoChargedCandidateRef refTrk;	
	
  // get hold of muon trks
  Handle<RecoChargedCandidateCollection> mucands;
  iEvent.getByLabel (muCandLabel_,mucands);

  // get track candidates around displaced muons
  Handle<RecoChargedCandidateCollection> trkcands;
  iEvent.getByLabel (trkCandLabel_,trkcands);
  
  if(saveTags_){
    filterobject->addCollectionTag(muCandLabel_);
    filterobject->addCollectionTag(trkCandLabel_);
  }
  
  double e1,e2,e3;
  Particle::LorentzVector p,p1,p2,p3;
  
  //TrackRefs to mu cands in trkcand
  vector<TrackRef> trkMuCands;
  
  //Already used mu tracks to avoid double counting candidates
  vector<bool> isUsedCand(trkcands->size(),false);
  
  int counter = 0;
  
  //run algorithm
  for (RecoChargedCandidateCollection::const_iterator mucand1=mucands->begin(), endCand1=mucands->end(); mucand1!=endCand1; ++mucand1) {
  
  	if ( mucands->size()<2) break;
  	if ( trkcands->size()<3) break;
  
  	TrackRef trk1 = mucand1->get<TrackRef>();
	LogDebug("HLTDisplacedMumukFilter") << " 1st muon: q*pt= " << trk1->charge()*trk1->pt() << ", eta= " << trk1->eta() << ", hits= " << trk1->numberOfValidHits();
  
  	// eta cut
	if (fabs(trk1->eta()) > maxEta_) continue;
	
	// Pt threshold cut
	if (trk1->pt() < minPt_) continue;

  	RecoChargedCandidateCollection::const_iterator mucand2 = mucand1; ++mucand2;
  	
  	for (RecoChargedCandidateCollection::const_iterator endCand2=mucands->end(); mucand2!=endCand2; ++mucand2) {
  
  		TrackRef trk2 = mucand2->get<TrackRef>();

		LogDebug("HLTDisplacedMumukFilter") << " 2nd muon: q*pt= " << trk2->charge()*trk2->pt() << ", eta= " << trk2->eta() << ", hits= " << trk2->numberOfValidHits();

		// eta cut
 		if (fabs(trk2->eta()) > maxEta_) continue;
	
		// Pt threshold cut
		if (trk2->pt() < minPt_) continue;

		RecoChargedCandidateCollection::const_iterator trkcand, endCandTrk;

		std::vector<bool>::iterator isUsedIter, endIsUsedCand;

		//get overlapping muon candidates
		for ( trkcand = trkcands->begin(), endCandTrk=trkcands->end(), isUsedIter = isUsedCand.begin(), endIsUsedCand = isUsedCand.end(); trkcand != endCandTrk && isUsedIter != endIsUsedCand; ++trkcand, ++isUsedIter) {
			TrackRef trk3 = trkcand->get<TrackRef>();
		
			//check for overlapping muon tracks and save TrackRefs
 			if (overlap(*mucand1,*trkcand)) {
 				trkMuCands.push_back(trk3);
				*isUsedIter = true;
 				continue;
 			}
 			else if (overlap(*mucand2,*trkcand)){
 				trkMuCands.push_back(trk3);
				*isUsedIter = true;
 				continue;
 			}
			
			if(trkMuCands.size()==2) break;
		}

		//Not all overlapping candidates found, skip event
		if (trkMuCands.size()!=2) continue;

		//combine muons with all tracks
  		for ( trkcand = trkcands->begin(), endCandTrk=trkcands->end(), isUsedIter = isUsedCand.begin(), endIsUsedCand = isUsedCand.end(); trkcand != endCandTrk && isUsedIter != endIsUsedCand; ++trkcand, ++isUsedIter) {
 
  			TrackRef trk3 = trkcand->get<TrackRef>();

			LogDebug("HLTDisplacedMumukFilter") << " 3rd track: q*pt= " << trk3->charge()*trk3->pt() << ", eta= " << trk3->eta() << ", hits= " << trk3->numberOfValidHits();
 
 			//skip overlapping muon candidates
 			if(trk3==trkMuCands.at(0) || trk3==trkMuCands.at(1)) continue;

			//skip already used tracks
			if(*isUsedIter) continue;
				
			// eta cut
			if (fabs(trk3->eta()) > maxEta_) continue;
	
			// Pt threshold cut
			if (trk3->pt() < minPt_) continue;

			// Combined system
			e1 = sqrt(trk1->momentum().Mag2()+MuMass2);
			e2 = sqrt(trk2->momentum().Mag2()+MuMass2);
			e3 = sqrt(trk3->momentum().Mag2()+thirdTrackMass2);
			
			p1 = Particle::LorentzVector(trk1->px(),trk1->py(),trk1->pz(),e1);
			p2 = Particle::LorentzVector(trk2->px(),trk2->py(),trk2->pz(),e2);
			p3 = Particle::LorentzVector(trk3->px(),trk3->py(),trk3->pz(),e3);
			
			p = p1+p2+p3;
			
 			//invariant mass cut
 			double invmass = abs(p.mass());
			
			LogDebug("HLTDisplacedMumukFilter") << " Invmass= " << invmass;
			
			if (invmass>maxInvMass_ || invmass<minInvMass_) continue;
			
			// do the vertex fit
			vector<TransientTrack> t_tks;
			t_tks.push_back((*theB).build(&trk1));
			t_tks.push_back((*theB).build(&trk2));
			t_tks.push_back((*theB).build(&trk3));
						
			if (t_tks.size()!=3) continue;
			
			KalmanVertexFitter kvf;
			TransientVertex tv = kvf.vertex(t_tks);
					
			if (!tv.isValid()) continue;
			
			Vertex vertex = tv;
			
			// get vertex position and error to calculate the decay length significance
			GlobalPoint secondaryVertex = tv.position();
			GlobalError err = tv.positionError();

			//calculate decay length  significance w.r.t. the beamspot
			GlobalPoint displacementFromBeamspot( -1*((vertexBeamSpot.x0() -secondaryVertex.x()) +  (secondaryVertex.z() - vertexBeamSpot.z0()) * vertexBeamSpot.dxdz()), -1*((vertexBeamSpot.y0() - secondaryVertex.y())+ (secondaryVertex.z() -vertexBeamSpot.z0()) * vertexBeamSpot.dydz()), 0);

			float lxy = displacementFromBeamspot.perp();
			float lxyerr = sqrt(err.rerr(displacementFromBeamspot));

			// get normalizes chi2
			float normChi2 = tv.normalisedChiSquared();
			
			//calculate the angle between the decay length and the mumu momentum
			Vertex::Point vperp(displacementFromBeamspot.x(),displacementFromBeamspot.y(),0.);
			math::XYZVector pperp(p.x(),p.y(),0.);

			float cosAlpha = vperp.Dot(pperp)/(vperp.R()*pperp.R());
			
			LogDebug("HLTDisplacedMumukFilter") << " vertex fit normalised chi2: " << normChi2 << ", Lxy significance: " << lxy/lxyerr << ", cosine pointing angle: " << cosAlpha;
			
			// put vertex in the event
			vertexCollection->push_back(vertex);
			
			if (normChi2 > maxNormalisedChi2_) continue;
			if (lxy/lxyerr < minLxySignificance_) continue;
			if(cosAlpha < minCosinePointingAngle_) continue;
			
			LogDebug("HLTDisplacedMumukFilter") << " Event passed!";
			
			//Add event
			++counter;
 			
 			//Get refs 
 			bool i1done = false;
			bool i2done = false;
			bool i3done = false;
			vector<RecoChargedCandidateRef> vref;
			filterobject->getObjects(TriggerMuon,vref);
			for (unsigned int i=0; i<vref.size(); i++) {
				RecoChargedCandidateRef candref =  RecoChargedCandidateRef(vref[i]);
				TrackRef trktmp = candref->get<TrackRef>();
				if (trktmp==trk1) {
					i1done = true;
				} else if (trktmp==trk2) {
					i2done = true;
				} else if (trktmp==trk3) {
				    i3done = true;
				}
				if (i1done && i2done && i3done) break;
			}
		
			if (!i1done) { 
				refMu1=RecoChargedCandidateRef( Ref<RecoChargedCandidateCollection> (mucands,distance(mucands->begin(), mucand1)));
				filterobject->addObject(TriggerMuon,refMu1);
			}
			if (!i2done) { 
				refMu2=RecoChargedCandidateRef( Ref<RecoChargedCandidateCollection> (mucands,distance(mucands->begin(),mucand2)));
				filterobject->addObject(TriggerMuon,refMu2);
			}
			if (!i3done) {
			    refTrk=RecoChargedCandidateRef( Ref<RecoChargedCandidateCollection> (trkcands,distance(trkcands->begin(),trkcand)));
				filterobject->addObject(TriggerTrack,refTrk);
			}
 			
 			if (fastAccept_) break;
  		}  
  		
  		trkMuCands.clear();
  	} 
  }

  // filter decision
  const bool accept (counter >= 1);
  
  LogDebug("HLTDisplacedMumukFilter") << " >>>>> Result of HLTDisplacedMumukFilter is "<< accept << ", number of muon pairs passing thresholds= " << counter; 
  
  // put filter object into the Event
  iEvent.put(filterobject);
  iEvent.put(vertexCollection);
  
  return accept;
}

// ----------------------------------------------------------------------
int HLTmmkFilter::overlap(const reco::Candidate &a, const reco::Candidate &b) {
  
  double eps(1.0e-5);

  double dpt = a.pt() - b.pt();
  dpt *= dpt;

  double dphi = deltaPhi(a.phi(), b.phi()); 
  dphi *= dphi; 

  double deta = a.eta() - b.eta(); 
  deta *= deta; 

  if ((dpt + dphi + deta) < eps) {
    return 1;
  } 

  return 0;

}

