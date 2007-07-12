#include <vector>
#include <memory>
#include <algorithm>

// Class header file
#include "Calibration/HcalIsolatedTrackReco/interface/IsolatedPixelTrackCandidateProducer.h"
#include "DataFormats/HcalIsolatedTrack/interface/IsolatedPixelTrackCandidate.h"
#include "DataFormats/HcalIsolatedTrack/interface/IsolatedPixelTrackCandidateFwd.h"
// Framework
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"
//
#include "DataFormats/Common/interface/TriggerResults.h"
// L1Extra
#include "DataFormats/L1Trigger/interface/L1EmParticle.h"
#include "DataFormats/L1Trigger/interface/L1JetParticle.h"
#include "DataFormats/L1Trigger/interface/L1ParticleMap.h"
// Math
#include "Math/GenVector/VectorUtil.h"
#include "Math/GenVector/PxPyPzE4D.h"


IsolatedPixelTrackCandidateProducer::IsolatedPixelTrackCandidateProducer(const edm::ParameterSet& config){
   
  l1eTauJetsSource_=config.getUntrackedParameter<edm::InputTag>("L1eTauJetsSource");
  pixelTracksSource_=config.getUntrackedParameter<edm::InputTag>("PixelTracksSource");
  pixelIsolationConeSize_=config.getParameter<double>("PixelIsolationConeSize");
  maxEta_=config.getParameter<double>("MaxEta");
  particleMapSource_=config.getUntrackedParameter<edm::InputTag>("ParticleMapSource");

  // Register the product
  produces< reco::IsolatedPixelTrackCandidateCollection >();

}

IsolatedPixelTrackCandidateProducer::~IsolatedPixelTrackCandidateProducer() {

}


void IsolatedPixelTrackCandidateProducer::produce(edm::Event& theEvent, const edm::EventSetup& theEventSetup) {

  using namespace edm;

  //  edm::LogInfo("IsolatedPixelTrackCandidateProducer") << "Producing event number: " << theEvent.id() << "\n";

  reco::IsolatedPixelTrackCandidateCollection * trackCollection=new reco::IsolatedPixelTrackCandidateCollection;

  edm::Handle<reco::TrackCollection> pixelTracks;
  theEvent.getByLabel(pixelTracksSource_,pixelTracks);

  edm::Handle<l1extra::L1JetParticleCollection> l1eTauJets;
  theEvent.getByLabel(l1eTauJetsSource_,l1eTauJets);

   // Get the successful L1 jet candidates.
  Handle< l1extra::L1ParticleMapCollection > mapColl ;
  theEvent.getByLabel( particleMapSource_, mapColl ) ;
  const l1extra::L1ParticleMap& singleJetMap = ( *mapColl )[ l1extra::L1ParticleMap::kSingleJet100 ];
  const l1extra::L1JetParticleVectorRef& triggeredJets = singleJetMap.jetParticles() ;

  // Loop over successful jets.
  int jetCounter = 0 ;
  double ptTriggered=0;
  double phiTriggered=-100;
  for( l1extra::L1JetParticleVectorRef::const_iterator jetItr = triggeredJets.begin() ; jetItr != triggeredJets.end() ; ++jetItr )
    {
      jetCounter++;
      //      edm::LogInfo("L1 seed jets:") << "Jet #" << jetCounter<< ": " << "   phi: "<<jetItr->get()->phi()<< "   pt: "<<jetItr->get()->pt();
      if (jetItr->get()->pt()>ptTriggered) 
	{
	  ptTriggered=jetItr->get()->pt();
	  phiTriggered=jetItr->get()->phi();
	}
    }
 
  double minPtTrack_=5;
  double drMaxL1Track_=0.5;
  
  //loop to select isolated tracks
  for (reco::TrackCollection::const_iterator track=pixelTracks->begin(); 
       track!=pixelTracks->end(); track++) {
    if(track->pt()<minPtTrack_) continue;

    if (fabs(track->eta())>maxEta_) continue;

    //selected tracks should match L1 taus
    for (l1extra::L1JetParticleCollection::const_iterator tj=l1eTauJets->begin(); tj!=l1eTauJets->end(); tj++) {
      
      //select taus not matched to triggered L1 jet
      double dPhi;
      if (fabs(tj->phi()-phiTriggered)>3.14159) dPhi=6.28318-fabs(tj->phi()-phiTriggered);
      else dPhi=fabs(tj->phi()-phiTriggered); 
      if (dPhi<1) continue;
      
      //select tracks matched to tau
      if(ROOT::Math::VectorUtil::DeltaR(track->momentum(),tj->momentum()) 
	 > drMaxL1Track_) continue;
      
      ///////////////////

      //calculate isolation
      double maxPt=0;
      double sumPt=0;
      for (reco::TrackCollection::const_iterator track2=pixelTracks->begin(); 
	   track2!=pixelTracks->end(); track2++) {
	if(track2!=track &&
	   ROOT::Math::VectorUtil::DeltaR(track->momentum(),track2->momentum())
	   <pixelIsolationConeSize_){
	  sumPt+=track2->pt();
	  if(track2->pt()>maxPt) maxPt=track2->pt();
	}
      }
      
      if(maxPt<5){
	reco::IsolatedPixelTrackCandidate newCandidate(reco::TrackRef(pixelTracks,track-pixelTracks->begin()), maxPt,sumPt);
	trackCollection->push_back(newCandidate);
      }

    } //loop over L1 tau
  }//loop over pixel tracks

  // put the product in the event
  std::auto_ptr< reco::IsolatedPixelTrackCandidateCollection > outCollection(trackCollection);
  theEvent.put(outCollection);


}
