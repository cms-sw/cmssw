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

#include "DataFormats/HLTReco/interface/HLTFilterObject.h"



#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutSetupFwd.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutRecord.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerObjectMapRecord.h"

#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerObjectMapFwd.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerObjectMap.h"


// Math
#include "Math/GenVector/VectorUtil.h"
#include "Math/GenVector/PxPyPzE4D.h"


IsolatedPixelTrackCandidateProducer::IsolatedPixelTrackCandidateProducer(const edm::ParameterSet& config){
   
  l1eTauJetsSource_=config.getUntrackedParameter<edm::InputTag>("L1eTauJetsSource");
  tauAssocCone_=config.getParameter<double>("tauAssociationCone"); 
  pixelTracksSource_=config.getUntrackedParameter<edm::InputTag>("PixelTracksSource");
  pixelIsolationConeSize_=config.getParameter<double>("PixelIsolationConeSize");
  maxEta_=config.getParameter<double>("MaxEta");
  hltGTseedlabel_=config.getUntrackedParameter<edm::InputTag>("L1GTSeedLabel");
  l1GtObjectMapSource_ = config.getUntrackedParameter<edm::InputTag> ("L1GtObjectMapSource");
//  particleMapSource_=config.getUntrackedParameter<edm::InputTag>("ParticleMapSource");

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

  edm::Handle<reco::HLTFilterObjectWithRefs> l1trigobj;
  theEvent.getByLabel(hltGTseedlabel_, l1trigobj);

///////////////////
  edm::Handle<L1GlobalTriggerObjectMapRecord> gtObjectMapRecord;
  theEvent.getByLabel(l1GtObjectMapSource_, gtObjectMapRecord);

  const std::vector<L1GlobalTriggerObjectMap>& objMapVec = gtObjectMapRecord->gtObjectMap();
///////////////////

//  std::cout<<"number of L1 triggered objects: "<<l1trigobj->size()<<std::endl;
   // Get the successful L1 jet candidates.

  double ptTriggered=-10;
  double etaTriggered=-100;
  double phiTriggered=-100;

/*
  for (std::vector<L1GlobalTriggerObjectMap>::const_iterator itMap = objMapVec.begin(); itMap != objMapVec.end(); ++itMap) 
	{
        // check if the map is needed (using algorithm bits)
        int algoBit = (*itMap).algoBitNumber();

        std::string algoName = (*itMap).algoName();
	bool res = (*itMap).algoGtlResult();
	
	int resu=-1;
	if (res) resu=1;
	else resu=0;
	
//	if (algoBit==35||algoBit==26) 
//	std::cout<<"Name: "<<algoName<<"  algo bit number: "<<algoBit<<"   result: "<<resu<<std::endl;
	
	}
  
*/

  for (unsigned int p=0; p<l1trigobj->size(); p++)
	{
	edm::RefToBase<reco::Candidate> l1jetref=l1trigobj->getParticleRef(p);
	if (l1jetref.get()->pt()>ptTriggered)
		{
		ptTriggered=l1jetref.get()->pt(); 
		phiTriggered=l1jetref.get()->phi();
		etaTriggered=l1jetref.get()->eta();
		}
	}


//  double ptTriggered=0;
//  double phiTriggered=-100;

  

  /*
  Handle< l1extra::L1ParticleMapCollection > mapColl ;
  theEvent.getByLabel( particleMapSource_, mapColl ) ;
  const l1extra::L1ParticleMap& singleJetMap = ( *mapColl )[ l1extra::L1ParticleMap::kSingleJet100 ];
  const l1extra::L1JetParticleVectorRef& triggeredJets = singleJetMap.jetParticles() ;

//  std::cout<<"EVENT Beg"<<std::endl;
  // Loop over successful jets.
  int jetCounter = 0 ;
  std::cout.precision(20);
  double ptTriggered=0;
  double phiTriggered=-100;
  for( l1extra::L1JetParticleVectorRef::const_iterator jetItr = triggeredJets.begin() ; jetItr != triggeredJets.end() ; ++jetItr )
    {
      jetCounter++;
      std::cout<< "Jet #" << jetCounter<< ": " << "   phi: "<<jetItr->get()->phi()<< "   pt: "<<jetItr->get()->pt()<<std::endl;
      if (jetItr->get()->pt()>ptTriggered) 
	{
	  ptTriggered=jetItr->get()->pt();
	  phiTriggered=jetItr->get()->phi();
	}
    }
  */
//  std::cout<<"phiTriggered: "<<phiTriggered<<"     etaTriggered: "<<etaTriggered<<"     ptTriggered: 
//"<<ptTriggered<<std::endl;

  double minPtTrack_=5;
  double drMaxL1Track_=tauAssocCone_;

  int ntr=0;
  
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
   
      if (dPhi<1) 
	{	
//	std::cout<<"SKIP"<<std::endl;
//	std::cout<<"phi value: "<<tj->phi()<<"   dPhi value: "<<dPhi<<std::endl;
	continue;
	}
      
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
//	std::cout<<"PUT<<<<phi value: "<<tj->phi()<<"   dPhi value: "<<dPhi<<std::endl;
//	std::cout<<"put track# "<<ntr<<std::endl;
	reco::IsolatedPixelTrackCandidate newCandidate(reco::TrackRef(pixelTracks,track-pixelTracks->begin()), maxPt,sumPt);
	trackCollection->push_back(newCandidate);
      	ntr++;
	}

    } //loop over L1 tau

  }//loop over pixel tracks

  // put the product in the event
  std::auto_ptr< reco::IsolatedPixelTrackCandidateCollection > outCollection(trackCollection);
  theEvent.put(outCollection);


}
