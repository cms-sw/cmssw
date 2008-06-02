
#include "HLTrigger/btau/interface/HLTmumuGammaFilter.h"

#include <iostream>
#include <Math/VectorUtil.h>

#include "FWCore/Framework/interface/ESHandle.h"
#include "DataFormats/Common/interface/Handle.h"

#include "DataFormats/Common/interface/RefToBase.h"
#include "DataFormats/HLTReco/interface/HLTFilterObject.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/Candidate/interface/CandidateFwd.h"
#include "DataFormats/RecoCandidate/interface/RecoCandidate.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "PhysicsTools/Utilities/interface/DeltaR.h"
#include "DataFormats/Math/interface/LorentzVector.h"
#include "DataFormats/Math/interface/Vector3D.h"
#include "DataFormats/RecoCandidate/interface/RecoEcalCandidate.h"
#include "DataFormats/RecoCandidate/interface/RecoEcalCandidateFwd.h"

#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "TrackingTools/TransientTrack/interface/TransientTrackBuilder.h"
#include "TrackingTools/Records/interface/TransientTrackRecord.h"

#include "RecoVertex/KalmanVertexFit/interface/KalmanVertexFitter.h"
#include "RecoVertex/VertexPrimitives/interface/TransientVertex.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"


using namespace edm;
using namespace reco;
using namespace std; 

//
// constructors and destructor
//
HLTmumuGammaFilter::HLTmumuGammaFilter(const edm::ParameterSet& iConfig)
{
	//now do what ever initialization is needed

  CandSrc_ = iConfig.getParameter<edm::InputTag>("CandSrc");
  m_vertexSrc  = iConfig.getParameter<edm::InputTag>("TrigVertex");
  deltaRCut = iConfig.getParameter<double>("MaxDeltaR");
  ClusPtMin = iConfig.getParameter<double>("ClusterPtMin");
  minInvMass = iConfig.getParameter<double>("MinInvMass");
  maxInvMass = iConfig.getParameter<double>("MaxInvMass");


  //register your products
   produces<reco::HLTFilterObjectWithRefs>();
}


HLTmumuGammaFilter::~HLTmumuGammaFilter()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


// ------------ method called once each job just before starting event loop  ------------
void HLTmumuGammaFilter::beginJob(const edm::EventSetup&)
{

}

// ------------ method called once each job just after ending the event loop  ------------
void HLTmumuGammaFilter::endJob() 
{
 	
}

// ------------ method called on each new Event  ------------
bool HLTmumuGammaFilter::filter(edm::Event& iEvent, const edm::EventSetup& iSetup)
{

  // The filter object
  std::auto_ptr<reco::HLTFilterObjectWithRefs> filterproduct (new reco::HLTFilterObjectWithRefs(path(),module()));
  // Ref to Candidate object to be recorded in filter object
  edm::RefToBase<reco::Candidate> ref;

  bool accept = false;

  // get vertex collection
  Handle<VertexCollection> trigvertex;
  iEvent.getByLabel(m_vertexSrc, trigvertex);			

  // Get the recoEcalCandidates
   edm::Handle<reco::RecoEcalCandidateCollection> recoecalcands;
   iEvent.getByLabel(CandSrc_,recoecalcands);

   for (reco::RecoEcalCandidateCollection::const_iterator recoecalcand= recoecalcands->begin(); recoecalcand!=recoecalcands->end(); recoecalcand++) {
     /*cout<<" Reco Ecal candidate pT "<<recoecalcand->pt()
         << " eta "<<recoecalcand->eta()
         << "phi  "<<recoecalcand->phi()<<endl;
     */
     if(recoecalcand->pt()>ClusPtMin ){
       for(reco::VertexCollection::const_iterator vtx = trigvertex->begin(); vtx != trigvertex->end(); vtx++){
	 /*cout << " trig vtx position, x "<< vtx->x()
	   << " y "<<vtx->y() << " z " <<vtx->z()<<endl;*/
	 // get tracks
	 reco::TrackRefVector::const_iterator trk = vtx->tracks_begin();
	 //cout<<" trk ref1 pT "<< (*trk)->pt()<<endl;
	 //cout<<" trk ref2 pT "<< (*(trk+1))->pt()<<endl;
	 // match deltaR between trk and ecal cluster
	 double delr = ROOT::Math::VectorUtil::DeltaR(recoecalcand->p4(), (*trk)->momentum()+ (*(trk+1))->momentum());
	 //cout<< " delR trk cluster "<<delr<<endl;
	 if(delr < deltaRCut){
	   math::XYZVector p1= recoecalcand->momentum(), 
	     p2 = (*trk)->momentum(),
	     p3 = (*(trk+1))->momentum();
	   math::XYZTLorentzVector v1( p1.x(), p1.y(), p1.z(), p1.r() ), 
	     v2( p2.x(), p2.y(), p2.z(), p2.r() ),
	     v3(p3.x(), p3.y(), p3.z(), p3.r() );
	   double invMass = (v1 + v2 + v3).mass();
	   //cout << " mu mu gamma inv.mass "<<invMass<<endl;
	   if(minInvMass < invMass < maxInvMass){
	     accept = true;
	     ref=edm::RefToBase<reco::Candidate>(reco::RecoEcalCandidateRef(recoecalcands,distance(recoecalcands->begin(),recoecalcand)));
	     filterproduct->putParticle(ref);
	   }
	 } 
       }
     }
   }
   // put filter object into the Event
   iEvent.put(filterproduct);
   
   return accept;   
}

