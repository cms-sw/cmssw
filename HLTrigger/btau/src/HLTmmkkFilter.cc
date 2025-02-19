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
#include "TrackingTools/PatternTools/interface/TSCBLBuilderNoMaterial.h"
#include "TrackingTools/TrajectoryState/interface/FreeTrajectoryState.h" 
#include "TrackingTools/TrajectoryParametrization/interface/GlobalTrajectoryParameters.h"

#include "DataFormats/BeamSpot/interface/BeamSpot.h"

#include "DataFormats/Math/interface/deltaPhi.h"

#include "HLTmmkkFilter.h"

using namespace edm;
using namespace reco;
using namespace std;
using namespace trigger;


// ----------------------------------------------------------------------
HLTmmkkFilter::HLTmmkkFilter(const edm::ParameterSet& iConfig) : HLTFilter(iConfig),
  thirdTrackMass_(iConfig.getParameter<double>("ThirdTrackMass")),
  fourthTrackMass_(iConfig.getParameter<double>("FourthTrackMass")),
  maxEta_(iConfig.getParameter<double>("MaxEta")),
  minPt_(iConfig.getParameter<double>("MinPt")),
  minInvMass_(iConfig.getParameter<double>("MinInvMass")),
  maxInvMass_(iConfig.getParameter<double>("MaxInvMass")),
  maxNormalisedChi2_(iConfig.getParameter<double>("MaxNormalisedChi2")),
  minLxySignificance_(iConfig.getParameter<double>("MinLxySignificance")),
  minCosinePointingAngle_(iConfig.getParameter<double>("MinCosinePointingAngle")),
  minD0Significance_(iConfig.getParameter<double>("MinD0Significance")),
  fastAccept_(iConfig.getParameter<bool>("FastAccept")),
  beamSpotTag_ (iConfig.getParameter<edm::InputTag> ("BeamSpotTag"))
{
  muCandLabel_   = iConfig.getParameter<edm::InputTag>("MuCand");
  trkCandLabel_  = iConfig.getParameter<edm::InputTag>("TrackCand");
  
  produces<VertexCollection>();
  produces<CandidateCollection>();
}


// ----------------------------------------------------------------------
HLTmmkkFilter::~HLTmmkkFilter() {

}


// ----------------------------------------------------------------------
void HLTmmkkFilter::beginJob() {

}


// ----------------------------------------------------------------------
void HLTmmkkFilter::endJob() {
 	
}


// ----------------------------------------------------------------------
bool HLTmmkkFilter::hltFilter(edm::Event& iEvent, const edm::EventSetup& iSetup, trigger::TriggerFilterObjectWithRefs & filterproduct) {

  const double MuMass(0.106);
  const double MuMass2(MuMass*MuMass);
  
  const double thirdTrackMass2(thirdTrackMass_*thirdTrackMass_);
  const double fourthTrackMass2(fourthTrackMass_*fourthTrackMass_);
  
  auto_ptr<CandidateCollection> output(new CandidateCollection());    
  auto_ptr<VertexCollection> vertexCollection(new VertexCollection());
  
  //get the transient track builder:
  edm::ESHandle<TransientTrackBuilder> theB;
  iSetup.get<TransientTrackRecord>().get("TransientTrackBuilder",theB);

  //get the beamspot position
  edm::Handle<reco::BeamSpot> recoBeamSpotHandle;
  iEvent.getByLabel(beamSpotTag_,recoBeamSpotHandle);
  const reco::BeamSpot& vertexBeamSpot = *recoBeamSpotHandle;

  ESHandle<MagneticField> bFieldHandle;
  iSetup.get<IdealMagneticFieldRecord>().get(bFieldHandle);

  const MagneticField* magField = bFieldHandle.product();

  TSCBLBuilderNoMaterial blsBuilder;

  // Ref to Candidate object to be recorded in filter object
  RecoChargedCandidateRef refMu1;
  RecoChargedCandidateRef refMu2;
  RecoChargedCandidateRef refTrk;	
  RecoChargedCandidateRef refTrk2;	
	
  // get hold of muon trks
  Handle<RecoChargedCandidateCollection> mucands;
  iEvent.getByLabel (muCandLabel_,mucands);

  // get track candidates around displaced muons
  Handle<RecoChargedCandidateCollection> trkcands;
  iEvent.getByLabel (trkCandLabel_,trkcands);
  
  if (saveTags()) {
    filterproduct.addCollectionTag(muCandLabel_);
    filterproduct.addCollectionTag(trkCandLabel_);
  }
  
  double e1,e2,e3,e4;
  Particle::LorentzVector p,p1,p2,p3,p4;
  
  //TrackRefs to mu cands in trkcand
  vector<TrackRef> trkMuCands;
  
  //Already used mu tracks to avoid double counting candidates
  vector<bool> isUsedCand(trkcands->size(),false);
  
  int counter = 0;
  
  //run algorithm
  for (RecoChargedCandidateCollection::const_iterator mucand1=mucands->begin(), endCand1=mucands->end(); mucand1!=endCand1; ++mucand1) {
  
  	if ( mucands->size()<2) break;
  	if ( trkcands->size()<2) break;
  
  	TrackRef trk1 = mucand1->get<TrackRef>();
	LogDebug("HLTDisplacedMumukkFilter") << " 1st muon: q*pt= " << trk1->charge()*trk1->pt() << ", eta= " << trk1->eta() << ", hits= " << trk1->numberOfValidHits();
  
  	// eta cut
	if (fabs(trk1->eta()) > maxEta_) continue;
	
	// Pt threshold cut
	if (trk1->pt() < minPt_) continue;

  	RecoChargedCandidateCollection::const_iterator mucand2 = mucand1; ++mucand2;
  	
  	for (RecoChargedCandidateCollection::const_iterator endCand2=mucands->end(); mucand2!=endCand2; ++mucand2) {
  
  		TrackRef trk2 = mucand2->get<TrackRef>();

		LogDebug("HLTDisplacedMumukkFilter") << " 2nd muon: q*pt= " << trk2->charge()*trk2->pt() << ", eta= " << trk2->eta() << ", hits= " << trk2->numberOfValidHits();

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
		//if (trkMuCands.size()!=2) continue;

		//combine muons with all tracks
  		for ( trkcand = trkcands->begin(), endCandTrk=trkcands->end(), isUsedIter = isUsedCand.begin(), endIsUsedCand = isUsedCand.end(); trkcand != endCandTrk && isUsedIter != endIsUsedCand; ++trkcand, ++isUsedIter) {
 
  			TrackRef trk3 = trkcand->get<TrackRef>();

			LogDebug("HLTDisplacedMumukkFilter") << " 3rd track: q*pt= " << trk3->charge()*trk3->pt() << ", eta= " << trk3->eta() << ", hits= " << trk3->numberOfValidHits();
 
 			//skip overlapping muon candidates
			bool skip=false;
 			for (unsigned int itmc=0;itmc<trkMuCands.size();itmc++) if(trk3==trkMuCands.at(itmc)) skip=true;
			if(skip) continue;

			//skip already used tracks
			if(*isUsedIter) continue;
				
			// eta cut
			if (fabs(trk3->eta()) > maxEta_) continue;
	
			// Pt threshold cut
			if (trk3->pt() < minPt_) continue;

			RecoChargedCandidateCollection::const_iterator trkcand2;

			std::vector<bool>::iterator isUsedIter2;

                        for ( trkcand2 = trkcands->begin(), isUsedIter2 = isUsedCand.begin() ; trkcand2 != endCandTrk && isUsedIter2 != endIsUsedCand; ++trkcand2,++isUsedIter2) {

                          if (trkcand2==trkcand) continue;

                          TrackRef trk4 = trkcand2->get<TrackRef>();

                          LogDebug("HLTDisplacedMumukkFilter") << " 4th track: q*pt= " << trk4->charge()*trk4->pt() << ", eta= " << trk4->eta() << ", hits= " << trk4->numberOfValidHits();

                          //skip overlapping muon candidates                                                                                                       
			  bool skip2=false;
			  for (unsigned int itmc=0;itmc<trkMuCands.size();itmc++) if(trk4==trkMuCands.at(itmc)) skip2=true;
			  if(skip2) continue;

                          //skip already used tracks                                                                                                               
                          if(*isUsedIter2) continue;

                          // eta cut                                                                                                                               
                          if (fabs(trk4->eta()) > maxEta_) continue;

                          // Pt threshold cut                                                                                                                      
                          if (trk4->pt() < minPt_) continue;

			  // Combined system
			  e1 = sqrt(trk1->momentum().Mag2()+MuMass2);
			  e2 = sqrt(trk2->momentum().Mag2()+MuMass2);
			  e3 = sqrt(trk3->momentum().Mag2()+thirdTrackMass2);
			  e4 = sqrt(trk4->momentum().Mag2()+fourthTrackMass2);
			
			  p1 = Particle::LorentzVector(trk1->px(),trk1->py(),trk1->pz(),e1);
			  p2 = Particle::LorentzVector(trk2->px(),trk2->py(),trk2->pz(),e2);
			  p3 = Particle::LorentzVector(trk3->px(),trk3->py(),trk3->pz(),e3);
			  p4 = Particle::LorentzVector(trk4->px(),trk4->py(),trk4->pz(),e4);
			
			  p = p1+p2+p3+p4;
			  
			  //invariant mass cut
			  double invmass = abs(p.mass());
			  
			  LogDebug("HLTDisplacedMumukkFilter") << " Invmass= " << invmass;
			  
			  FreeTrajectoryState InitialFTS = initialFreeState(*trk3, magField);
			  TrajectoryStateClosestToBeamLine tscb( blsBuilder(InitialFTS, *recoBeamSpotHandle) );
			  double d0sig = tscb.transverseImpactParameter().significance();
			  
			  if (d0sig < minD0Significance_) continue;
			  
			  FreeTrajectoryState InitialFTS2 = initialFreeState(*trk4, magField);
			  TrajectoryStateClosestToBeamLine tscb2( blsBuilder(InitialFTS2, *recoBeamSpotHandle) );
			  d0sig = tscb2.transverseImpactParameter().value()/tscb2.transverseImpactParameter().significance();
			  
			  if (d0sig < minD0Significance_) continue;
			
			  if (invmass>maxInvMass_ || invmass<minInvMass_) continue;
			  
			  // do the vertex fit
			  vector<TransientTrack> t_tks;
			  t_tks.push_back((*theB).build(&trk1));
			  t_tks.push_back((*theB).build(&trk2));
			  t_tks.push_back((*theB).build(&trk3));
			  t_tks.push_back((*theB).build(&trk4));
			  
			  if (t_tks.size()!=4) continue;
			  
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
			  
			  LogDebug("HLTDisplacedMumukkFilter") << " vertex fit normalised chi2: " << normChi2 << ", Lxy significance: " << lxy/lxyerr << ", cosine pointing angle: " << cosAlpha;
			  
			  // put vertex in the event
			  vertexCollection->push_back(vertex);
			  
			  if (normChi2 > maxNormalisedChi2_) continue;
			  if (lxy/lxyerr < minLxySignificance_) continue;
			  if(cosAlpha < minCosinePointingAngle_) continue;
			  
			  LogDebug("HLTDisplacedMumukkFilter") << " Event passed!";
			  
			  //Add event
			  ++counter;
			  
			  //Get refs 
			  bool i1done = false;
			  bool i2done = false;
			  bool i3done = false;
			  bool i4done = false;
			  vector<RecoChargedCandidateRef> vref;
			  filterproduct.getObjects(TriggerMuon,vref);
			  for (unsigned int i=0; i<vref.size(); i++) {
			    RecoChargedCandidateRef candref =  RecoChargedCandidateRef(vref[i]);
			    TrackRef trktmp = candref->get<TrackRef>();
			    if (trktmp==trk1) {
			      i1done = true;
			    } else if (trktmp==trk2) {
			      i2done = true;
			    } else if (trktmp==trk3) {
				    i3done = true;
			    } else if (trktmp==trk4) {
			      i4done = true;
			    }
			    if (i1done && i2done && i3done && i4done) break;
			  }
			  
			  if (!i1done) { 
			    refMu1=RecoChargedCandidateRef( Ref<RecoChargedCandidateCollection> (mucands,distance(mucands->begin(), mucand1)));
			    filterproduct.addObject(TriggerMuon,refMu1);
			  }
			  if (!i2done) { 
			    refMu2=RecoChargedCandidateRef( Ref<RecoChargedCandidateCollection> (mucands,distance(mucands->begin(),mucand2)));
			    filterproduct.addObject(TriggerMuon,refMu2);
			  }
			  if (!i3done) {
			    refTrk=RecoChargedCandidateRef( Ref<RecoChargedCandidateCollection> (trkcands,distance(trkcands->begin(),trkcand)));
			    filterproduct.addObject(TriggerTrack,refTrk);
			  }
			  if (!i4done) {
			    refTrk2=RecoChargedCandidateRef( Ref<RecoChargedCandidateCollection> (trkcands,distance(trkcands->begin(),trkcand)));
			    filterproduct.addObject(TriggerTrack,refTrk2);
			  }
			  
			  if (fastAccept_) break;
			}
  		}  
  		
  		trkMuCands.clear();
  	} 
  }
  
  // filter decision
  const bool accept (counter >= 1);
  
  LogDebug("HLTDisplacedMumukkFilter") << " >>>>> Result of HLTDisplacedMumukkFilter is "<< accept << ", number of muon pairs passing thresholds= " << counter; 
  
  iEvent.put(vertexCollection);
  
  return accept;
}

// ----------------------------------------------------------------------
FreeTrajectoryState HLTmmkkFilter::initialFreeState( const reco::Track& tk,
						    const MagneticField* field)
{
  Basic3DVector<float> pos( tk.vertex());
  GlobalPoint gpos( pos);
  Basic3DVector<float> mom( tk.momentum());
  GlobalVector gmom( mom);
  GlobalTrajectoryParameters par( gpos, gmom, tk.charge(), field);
  CurvilinearTrajectoryError err( tk.covariance());
  return FreeTrajectoryState( par, err);
}

int HLTmmkkFilter::overlap(const reco::Candidate &a, const reco::Candidate &b) {
  
  double eps(1.44e-4);

  double dpt = a.pt() - b.pt();
  dpt *= dpt;

  double dphi = deltaPhi(a.phi(), b.phi()); 
  dphi *= dphi; 

  double deta = a.eta() - b.eta(); 
  deta *= deta; 

  if ((dphi + deta) < eps) {
    return 1;
  } 

  return 0;

}

