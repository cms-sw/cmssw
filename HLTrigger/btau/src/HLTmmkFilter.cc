#include "HLTrigger/btau/interface/HLTmmkFilter.h"

#include <iostream>
#include <cmath>

#include "FWCore/Framework/interface/ESHandle.h"

#include "DataFormats/Candidate/interface/CandidateFwd.h"
#include "DataFormats/RecoCandidate/interface/RecoCandidate.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/Candidate/interface/CompositeCandidate.h"

#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "TrackingTools/TransientTrack/interface/TransientTrackBuilder.h"
#include "TrackingTools/Records/interface/TransientTrackRecord.h"

#include "RecoVertex/KalmanVertexFit/interface/KalmanVertexFitter.h"
#include "RecoVertex/VertexPrimitives/interface/TransientVertex.h"
#include "RecoVertex/VertexTools/interface/VertexDistance3D.h"
#include "RecoVertex/VertexTools/interface/VertexDistanceXY.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/Common/interface/RefToBase.h"
#include "DataFormats/HLTReco/interface/HLTFilterObject.h"

#include "CommonTools/Statistics/interface/ChiSquaredProbability.h"
#include "PhysicsTools/CandUtils/interface/AddFourMomenta.h"
#include "PhysicsTools/HepMCCandAlgos/interface/MCCandMatcher.h"
#include "PhysicsTools/CandAlgos/interface/CandMatcher.h" 
#include "DataFormats/Candidate/interface/CandMatchMap.h"


#include "Math/PtEtaPhiM4D.h"

namespace math {
   typedef ROOT::Math::LorentzVector<ROOT::Math::PtEtaPhiM4D<double> > 
   PtEtaPhiMLorentzVectorD;
   typedef PtEtaPhiMLorentzVectorD PtEtaPhiMLorentzVector;
}

// ----------------------------------------------------------------------
#include <TLorentzVector.h>

using namespace edm;
using namespace reco;
using namespace std; 


// ----------------------------------------------------------------------
HLTmmkFilter::HLTmmkFilter(const edm::ParameterSet& iConfig) {
  fJpsiLabel   = iConfig.getParameter<edm::InputTag>("Src");
  fTrackLabel  = iConfig.getParameter<edm::InputTag>("Tracks");

  produces<VertexCollection>();
  produces<CandidateCollection>();
  produces<HLTFilterObjectWithRefs>();

}


// ----------------------------------------------------------------------
HLTmmkFilter::~HLTmmkFilter() {

}


// ----------------------------------------------------------------------
void HLTmmkFilter::beginJob(const edm::EventSetup&setup) {

}


// ----------------------------------------------------------------------
void HLTmmkFilter::endJob() {
 	
}


// ----------------------------------------------------------------------
bool HLTmmkFilter::filter(edm::Event& iEvent, const edm::EventSetup& iSetup) {

  // The filter object
  auto_ptr<CandidateCollection> output(new CandidateCollection());  
  auto_ptr<HLTFilterObjectWithRefs> filterproduct(new HLTFilterObjectWithRefs(path(),module()));
  
  auto_ptr<VertexCollection> vertexCollection(new VertexCollection());

  static int counter(0);
  ++counter;
  //  cout << "+++++++++ HLTmmkFilter> Event " << counter << endl;

  // -- get jpsi collection and retain muon candidate pointers
  Handle<HLTFilterObjectWithRefs> jpsiCollection;
  iEvent.getByLabel(fJpsiLabel, jpsiCollection);

  const Candidate *m0(0), *m1(0); 

  for (unsigned int i = 0; i < jpsiCollection->size(); ++i) {
    RefToBase<Candidate> jpsiRef = jpsiCollection->getParticleRef(i);
    const Candidate *cMuon;
    for (size_t i = 0; i < (jpsiRef.get())->numberOfDaughters(); ++ i ) {
      cMuon = (jpsiRef.get())->daughter(i);
      if (0 == i) m0 = cMuon;
      if (1 == i) m1 = cMuon;
    }
  }

  // -- get MY track CANDIDATES around jpsi
  edm::Handle<CandidateCollection> trackCollection;
  iEvent.getByLabel(fTrackLabel, trackCollection );  

  // -- Get muon indices. 
  unsigned int m0Idx(99999), m1Idx(99999);
  for (unsigned int i = 0; i < trackCollection->size(); ++i) {
    const Candidate &cTrack = (*trackCollection)[i]; 
    if (overlap(cTrack, *m0)) m0Idx = i; 
    if (overlap(cTrack, *m1)) m1Idx = i; 
  }

  if (m0Idx > 99998 || m1Idx > 99998) { 
    //    cout << "######### less than 2 muons found" << endl;
  }

  if (trackCollection->size() < 3) {
    //    cout << "######### no additional track found" << endl;
  }    

  // -- Combine all tracks with the two muons
  double minChi2(9999.);
  unsigned int minIndex(99999);
    
  Vertex vertex;
  CompositeCandidate *mmk = new CompositeCandidate();
  if ((m0Idx < 99999 && m1Idx < 99999)) {
    for (size_t i = 0; i < trackCollection->size(); ++i) {
      if (i == m0Idx) continue; 
      if (i == m1Idx) continue; 
      
      edm::ESHandle<TransientTrackBuilder> theB;
      iSetup.get<TransientTrackRecord>().get("TransientTrackBuilder", theB);
      
      vector<TransientTrack> t_tks;
      
      const Candidate &cMuon0 = (*trackCollection)[m0Idx]; 
      TrackRef trm0       = cMuon0.get<TrackRef>();
      TransientTrack ttm0 = (*theB).build(&trm0);
      t_tks.push_back(ttm0);
      
      const Candidate &cMuon1 = (*trackCollection)[m1Idx]; 
      TrackRef trm1       = cMuon1.get<TrackRef>();
      TransientTrack ttm1 = (*theB).build(&trm1);
      t_tks.push_back(ttm1);
      
      const Candidate &cTrack = (*trackCollection)[i]; 
      
      TrackRef trk = cTrack.get<TrackRef>();
      TransientTrack ttkp   = (*theB).build(&trk);
      t_tks.push_back(ttkp);
      
      KalmanVertexFitter kvf;
      TransientVertex tv = kvf.vertex(t_tks);
      if (tv.isValid()) {
	float normChi2 = tv.normalisedChiSquared();
	if (normChi2 < minChi2) {
	  minChi2 = normChi2;
	  minIndex = i; 
	  vertex = tv;

	  Candidate *c = cTrack.clone();
	  math::PtEtaPhiMLorentzVector p1 = math::PtEtaPhiMLorentzVector(c->p4());
	  p1.SetM(0.4937);
	  c->setP4(math::XYZTLorentzVector(p1));

	  mmk->addDaughter(cMuon0);
	  mmk->addDaughter(cMuon1);
	  mmk->addDaughter(*c);

	  AddFourMomenta addP4;
	  addP4.set(*mmk);

	}
      }
    }
  }

  if (minIndex < 9999) {
    const Candidate &cTrack = (*trackCollection)[minIndex]; 
    TrackRef trk = cTrack.get<TrackRef>();
    reco::Track tt(*trk);
  }


  // -- dummy decision
  if (minChi2 < 10) {

    //    cout << ">>>>>>>>>>>HLTmmkFilter> Accepting event" << endl;
    // -- Put vertex into event
    vertexCollection->push_back(vertex);
    iEvent.put(vertexCollection);

    // -- Put candidate collection into event
    output->push_back(mmk);
    iEvent.put(output);

// ?????????
//     // -- Put filter object into event
//     RefToBase<Candidate> candref = output->getParticleRef(static_cast<unsigned int>(0));
//     filterproduct->putParticle(candref);
//     iEvent.put(filterproduct);
    
    return true; 
  } else {
    return false;
  }

}


// ----------------------------------------------------------------------
double HLTmmkFilter::deltaPhi(double phi1, double phi2) {

  static double kPI = M_PI;
  static double kTWOPI = 2*M_PI;

  double x = phi1 - phi2; 
  while (x >= kPI) x -= kTWOPI;
  while (x < -kPI) x += kTWOPI; 

  return x; 
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

