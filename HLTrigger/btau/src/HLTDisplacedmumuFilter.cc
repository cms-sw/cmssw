#include <iostream>

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/Candidate/interface/CandidateFwd.h"
#include "DataFormats/RecoCandidate/interface/RecoCandidate.h"
#include "DataFormats/RecoCandidate/interface/RecoChargedCandidate.h"
#include "DataFormats/RecoCandidate/interface/RecoChargedCandidateFwd.h"
#include "DataFormats/Candidate/interface/Candidate.h"

#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "TrackingTools/TransientTrack/interface/TransientTrackBuilder.h"
#include "TrackingTools/Records/interface/TransientTrackRecord.h"

#include "RecoVertex/KalmanVertexFit/interface/KalmanVertexFitter.h"
#include "RecoVertex/VertexPrimitives/interface/TransientVertex.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/Common/interface/RefToBase.h"
#include "DataFormats/HLTReco/interface/TriggerFilterObjectWithRefs.h"
#include "DataFormats/HLTReco/interface/TriggerRefsCollections.h"

#include "HLTDisplacedmumuFilter.h"

using namespace edm;
using namespace reco;
using namespace std; 
using namespace trigger; 

//
// constructors and destructor
//
HLTDisplacedmumuFilter::HLTDisplacedmumuFilter(const edm::ParameterSet& iConfig)
{
	//now do what ever initialization is needed
	minLxySignificance_ = iConfig.getParameter<double>("MinLxySignificance");       
	maxNormalisedChi2_ = iConfig.getParameter<double>("MaxNormalisedChi2");  
	minCosinePointingAngle_ = iConfig.getParameter<double>("MinCosinePointingAngle");
	src_ = iConfig.getParameter<edm::InputTag>("Src");
	maxEta_ = iConfig.getParameter<double>("MaxEta");
	minPt_ = iConfig.getParameter<double>("MinPt");
	minPtPair_ = iConfig.getParameter<double>("MinPtPair");
	minInvMass_ = iConfig.getParameter<double>("MinInvMass");
	maxInvMass_ = iConfig.getParameter<double>("MaxInvMass");
	chargeOpt_ = iConfig.getParameter<int>("ChargeOpt");
	fastAccept_ = iConfig.getParameter<bool>("FastAccept");
	// collections produced by the filter
	produces<VertexCollection>();
	produces<trigger::TriggerFilterObjectWithRefs>();
}


HLTDisplacedmumuFilter::~HLTDisplacedmumuFilter()
{

}


// ------------ method called once each job just before starting event loop  ------------
void HLTDisplacedmumuFilter::beginJob(const edm::EventSetup&)
{

}

// ------------ method called once each job just after ending the event loop  ------------
void HLTDisplacedmumuFilter::endJob() 
{
 	
}

// ------------ method called on each new Event  ------------
bool HLTDisplacedmumuFilter::filter(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
	double const MuMass = 0.106;
	double const MuMass2 = MuMass*MuMass;
	
	// The filter object  
	// All HLT filters must create and fill an HLT filter object,
	// recording any reconstructed physics objects satisfying (or not)
	// this HLT filter, and place it in the Event.
	auto_ptr<TriggerFilterObjectWithRefs> filterobject (new TriggerFilterObjectWithRefs(path(),module()));

	// Ref to Candidate object to be recorded in filter object
	RecoChargedCandidateRef ref1;
	RecoChargedCandidateRef ref2;
	
	// get hold of muon trks
	Handle<RecoChargedCandidateCollection> mucands;
	iEvent.getByLabel (src_,mucands);
	
	//get the transient track builder:
	edm::ESHandle<TransientTrackBuilder> theB;
	iSetup.get<TransientTrackRecord>().get("TransientTrackBuilder",theB);
	std::auto_ptr<VertexCollection> vertexCollection(new VertexCollection());

	// look at all mucands,  check cuts and add to filter object
	int n = 0;
	double e1,e2;
	Particle::LorentzVector p,p1,p2;
	
	RecoChargedCandidateCollection::const_iterator cand1;
	RecoChargedCandidateCollection::const_iterator cand2;
	for (cand1=mucands->begin(); cand1!=mucands->end(); cand1++) {
		TrackRef tk1 = cand1->get<TrackRef>();
		// eta cut
		LogDebug("HLTDisplacedMumuFilter") << " 1st muon in loop: q*pt= " << tk1->charge()*tk1->pt() << ", eta= " << tk1->eta() << ", hits= " << tk1->numberOfValidHits();
		
		if (fabs(tk1->eta())>maxEta_) continue;
	
		// Pt threshold cut
		double pt1 = tk1->pt();
		if (pt1 < minPt_) continue;
	
		cand2 = cand1; cand2++;
		for (; cand2!=mucands->end(); cand2++) {
			TrackRef tk2 = cand2->get<TrackRef>();
	
			// eta cut
			LogDebug("HLTMuonDimuonFilter") << " 2nd muon in loop: q*pt= " << tk2->charge()*tk2->pt() << ", eta= " << tk2->eta() << ", hits= " << tk2->numberOfValidHits() << ", d0= " << tk2->d0();
			if (fabs(tk2->eta())>maxEta_) continue;
	
			// Pt threshold cut
			double pt2 = tk2->pt();
			if (pt2 < minPt_) continue;
	
			if (chargeOpt_<0) {
					if (tk1->charge()*tk2->charge()>0) continue;
			} else if (chargeOpt_>0) {
					if (tk1->charge()*tk2->charge()<0) continue;
			}
	
			// Combined dimuon system
			e1 = sqrt(tk1->momentum().Mag2()+MuMass2);
			e2 = sqrt(tk2->momentum().Mag2()+MuMass2);
			p1 = Particle::LorentzVector(tk1->px(),tk1->py(),tk1->pz(),e1);
			p2 = Particle::LorentzVector(tk2->px(),tk2->py(),tk2->pz(),e2);
			p = p1+p2;
	
			double pt12 = p.pt();
			LogDebug("HLTDisplacedMumuFilter") << " ... 1-2 pt12= " << pt12;
			if (pt12<minPtPair_) continue;
	
			double invmass = abs(p.mass());
			LogDebug("HLTDisplacedMumuFilter") << " ... 1-2 invmass= " << invmass;
			if (invmass<minInvMass_) continue;
			if (invmass>maxInvMass_) continue;
			
			// do the vertex fit
			vector<TransientTrack> t_tks;
			TransientTrack ttkp1 = (*theB).build(&tk1);
			TransientTrack ttkp2 = (*theB).build(&tk2);
			t_tks.push_back(ttkp1);
			t_tks.push_back(ttkp2);
			
			if (t_tks.size()!=2) continue;
			
			KalmanVertexFitter kvf;
			TransientVertex tv = kvf.vertex(t_tks);
					
			if (!tv.isValid()) continue;
			
			Vertex vertex = tv;
					
			// get vertex position and error to calculate the decay length significance
			GlobalPoint v = tv.position();
			GlobalError err = tv.positionError();
			float lxy = v.perp();
			float lxyerr = sqrt(err.rerr(v));
			
			// get normalizes chi2
			float normChi2 = tv.normalisedChiSquared();
			
			//calculate the angle between the decay length and the mumu momentum
			Vertex::Point vperp(v.x(),v.y(),0.);
			math::XYZVector pperp(p.x(),p.y(),0.);
			
			float cosAlpha = vperp.Dot(pperp)/(vperp.R()*pperp.R());
			
			// put vertex in the event
			vertexCollection->push_back(vertex);
			LogDebug("HLTDisplacedMumuFilter") << " vertex fit normalised chi2: " << normChi2 << ", Lxy significance: " << lxy/lxyerr << ", cosine pointing angle: " << cosAlpha;
			if (cosAlpha < minCosinePointingAngle_) continue;
			if (normChi2 > maxNormalisedChi2_) continue;
			if (lxy/lxyerr < minLxySignificance_) continue;
					
			// Add this pair
			n++;
			LogDebug("HLTDisplacedMumuFilter") << " Track1 passing filter: pt= " << tk1->pt() << ", eta: " << tk1->eta();
			LogDebug("HLTDisplacedMumuFilter") << " Track2 passing filter: pt= " << tk2->pt() << ", eta: " << tk2->eta();
			LogDebug("HLTDisplacedMumuFilter") << " Invmass= " << invmass;
	
			bool i1done = false;
			bool i2done = false;
			vector<RecoChargedCandidateRef> vref;
			filterobject->getObjects(TriggerMuon,vref);
			for (unsigned int i=0; i<vref.size(); i++) {
				RecoChargedCandidateRef candref =  RecoChargedCandidateRef(vref[i]);
				TrackRef tktmp = candref->get<TrackRef>();
				if (tktmp==tk1) {
					i1done = true;
				} else if (tktmp==tk2) {
					i2done = true;
				}
				if (i1done && i2done) break;
			}
		
			if (!i1done) { 
				ref1=RecoChargedCandidateRef( Ref<RecoChargedCandidateCollection> 	(mucands,distance(mucands->begin(), cand1)));
				filterobject->addObject(TriggerMuon,ref1);
			}
			if (!i2done) { 
				ref2=RecoChargedCandidateRef( Ref<RecoChargedCandidateCollection> (mucands,distance(mucands->begin(),cand2 )));
				filterobject->addObject(TriggerMuon,ref2);
			}
	
			if (fastAccept_) break;
		}
	
	}

 	// filter decision
	const bool accept (n >= 1);

   	// put filter object into the Event
	iEvent.put(filterobject);
	iEvent.put(vertexCollection);
	LogDebug("HLTDisplacedMumuFilter") << " >>>>> Result of HLTDisplacedMumuFilter is "<< accept << ", number of muon pairs passing thresholds= " << n; 

	return accept;
}

