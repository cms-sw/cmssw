
/*
 *  See header file for a description of this class.
 *
 *  $Date: 2008/05/12 15:59:28 $
 *  $Revision: 1.1 $
 *  \author G. Mila - INFN Torino
 */

#include "DQMOffline/Muon/src/SegmentTrackAnalyzer.h"

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/MuonDetId/interface/MuonSubdetId.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"
#include "TrackingTools/TransientTrack/interface/TransientTrack.h"
#include "RecoMuon/TransientTrackingRecHit/interface/MuonTransientTrackingRecHit.h"
#include "DataFormats/DTRecHit/interface/DTRecSegment4DCollection.h"
#include "DataFormats/DTRecHit/interface/DTRecHitCollection.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <string>
using namespace std;
using namespace edm;



SegmentTrackAnalyzer::SegmentTrackAnalyzer(const edm::ParameterSet& pSet, MuonServiceProxy *theService):MuonAnalyzerBase(theService) {

  cout<<"[SegmentTrackAnalyzer] Constructor called!"<<endl;
  parameters = pSet;

  const ParameterSet SegmentsTrackAssociatorParameters = parameters.getParameter<ParameterSet>("SegmentsTrackAssociatorParameters");
  theSegmentsAssociator = new SegmentsTrackAssociator(SegmentsTrackAssociatorParameters);

}


SegmentTrackAnalyzer::~SegmentTrackAnalyzer() { }


void SegmentTrackAnalyzer::beginJob(edm::EventSetup const& iSetup,DQMStore * dbe) {

  metname = "segmTrackAnalyzer";
  LogTrace(metname)<<"[SegmentTrackAnalyzer] Parameters initialization";
  dbe->setCurrentFolder("Muons/SegmentTrackAnalyzer");
  
  // histograms initalization
  hitsNotUsed = dbe->book1D("HitsNotUsedForGlobalTracking", "HitsNotUsedForGlobalTracking", 50, -0.5, 49.5);
  TrackSegm = dbe->book2D("trackSegments", "trackSegments", 3, 0.5, 3.5, 50, -0.5, 49.5);
  TrackSegm->setBinLabel(1,"DT+CSC",1);
  TrackSegm->setBinLabel(1,"DT",2);
  TrackSegm->setBinLabel(1,"CSC",3);

}


void SegmentTrackAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup, const reco::Track& recoTrack){

  LogTrace(metname)<<"[SegmentTrackAnalyzer] Filling the histos";
  
  MuonTransientTrackingRecHit::MuonRecHitContainer TheSegments = theSegmentsAssociator->associate(iEvent, iSetup, recoTrack );
 
  LogTrace(metname)<<"[SegmentTrackAnalyzer] # of segments associated to the track: "<<(TheSegments).size();

  // hit counters
  int hitsFromDt=0;
  int hitsFromCsc=0;
  int hitsFromRpc=0;
  int hitsFromTk=0;
  int hitsFromTrack=0;
  int hitsFromSegmDt=0;
  int hitsFromSegmCsc=0;
  // segment counters
  int segmFromDt=0;
  int segmFromCsc=0;

  for (MuonTransientTrackingRecHit::MuonRecHitContainer::const_iterator segment=TheSegments.begin();
       segment!=TheSegments.end(); segment++) {
   
    DetId id = (*segment)->geographicalId();
    
    // hits from DT segments
    if (id.det() == DetId::Muon && id.subdetId() == MuonSubdetId::DT ) {
      const DTRecSegment4D *seg4D = dynamic_cast<const DTRecSegment4D*>((*segment)->hit());
      if((*seg4D).hasPhi()){
	hitsFromSegmDt+=(*seg4D).phiSegment()->specificRecHits().size();
	segmFromDt++;
      }
      if((*seg4D).hasZed()) {
	hitsFromSegmDt+=(*seg4D).zSegment()->specificRecHits().size();
	segmFromDt++;
      }
    }
    
    // hits from CSC segments
    if (id.det() == DetId::Muon && id.subdetId() == MuonSubdetId::CSC ) {
      hitsFromSegmCsc+=(*segment)->recHits().size();
      segmFromCsc++;
    }

  }


  // hits from track
  for(trackingRecHit_iterator recHit =  recoTrack.recHitsBegin(); recHit != recoTrack.recHitsEnd(); ++recHit){

    hitsFromTrack++;
     DetId id = (*recHit)->geographicalId();
     // hits from DT
     if (id.det() == DetId::Muon && id.subdetId() == MuonSubdetId::DT ) 
       hitsFromDt++;   
     // hits from CSC
      if (id.det() == DetId::Muon && id.subdetId() == MuonSubdetId::CSC ) 
       hitsFromCsc++;
     // hits from RPC
     if (id.det() == DetId::Muon && id.subdetId() == MuonSubdetId::RPC ) 
       hitsFromRpc++;
     // hits from Tracker
     if (id.det() == DetId::Tracker)
       hitsFromTk++;

  }

  // fill the histos
  hitsNotUsed->Fill(hitsFromSegmDt+hitsFromSegmCsc+hitsFromRpc+hitsFromTk-hitsFromTrack);
  TrackSegm->Fill(1,segmFromDt+segmFromCsc);
  TrackSegm->Fill(2,segmFromDt);
  TrackSegm->Fill(3,segmFromCsc);

} 

