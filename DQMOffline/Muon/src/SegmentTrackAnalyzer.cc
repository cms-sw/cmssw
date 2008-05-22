
/*
 *  See header file for a description of this class.
 *
 *  $Date: 2008/05/21 15:48:16 $
 *  $Revision: 1.6 $
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
  hitsNotUsedPercentual = dbe->book1D("HitsNotUsedForGlobalTrackingDvHitUsed", "HitsNotUsedForGlobalTrackingDvHitUsed", 100, 0, 1.);

  TrackSegm = dbe->book2D("trackSegments", "trackSegments", 3, 0.5, 3.5, 50, -0.5, 49.5);
  TrackSegm->setBinLabel(1,"DT+CSC",1);
  TrackSegm->setBinLabel(2,"DT",1);
  TrackSegm->setBinLabel(3,"CSC",1);
  
  hitStaProvenance = dbe->book1D("trackHitProvenance", "trackHitProvenance", 7, 0.5, 7.5);
  hitStaProvenance->setBinLabel(1,"DT");
  hitStaProvenance->setBinLabel(2,"CSC");
  hitStaProvenance->setBinLabel(3,"RPC");
  hitStaProvenance->setBinLabel(4,"DT+CSC");
  hitStaProvenance->setBinLabel(5,"DT+RPC");
  hitStaProvenance->setBinLabel(6,"CSC+RPC");
  hitStaProvenance->setBinLabel(7,"DT+CSC+RPC");

  int etaBin = parameters.getParameter<int>("etaBin");
  double etaMin = parameters.getParameter<double>("etaMin");
  double etaMax = parameters.getParameter<double>("etaMax");
  trackHitPercentualVsEta = dbe->book2D("trackHitDivtrackSegmHitVsEta", "trackHitDivtrackSegmHitVsEta", etaBin, etaMin, etaMax, 50, 0, 5);
  dtTrackHitPercentualVsEta = dbe->book2D("dtTrackHitDivtrackSegmHitVsEta", "dtTrackHitDivtrackSegmHitVsEta", etaBin, etaMin, etaMax, 20, 0, 2);
  cscTrackHitPercentualVsEta = dbe->book2D("cscTrackHitDivtrackSegmHitVsEta", "cscTrackHitDivtrackSegmHitVsEta", etaBin, etaMin, etaMax, 20, 0, 2);

  int phiBin = parameters.getParameter<int>("phiBin");
  double phiMin = parameters.getParameter<double>("phiMin");
  double phiMax = parameters.getParameter<double>("phiMax");
  trackHitPercentualVsPhi = dbe->book2D("trackHitDivtrackSegmHitVsPhi", "trackHitDivtrackSegmHitVsPhi", phiBin, phiMin, phiMax, 50, 0, 5);
  dtTrackHitPercentualVsPhi = dbe->book2D("dtTrackHitDivtrackSegmHitVsPhi", "dtTrackHitDivtrackSegmHitVsPhi", phiBin, phiMin, phiMax, 20, 0, 2);
  cscTrackHitPercentualVsPhi = dbe->book2D("cscTrackHitDivtrackSegmHitVsPhi", "cscTrackHitDivtrackSegmHitVsPhi", phiBin, phiMin, phiMax, 20, 0, 2);

  int ptBin = parameters.getParameter<int>("ptBin");
  double ptMin = parameters.getParameter<double>("ptMin");
  double ptMax = parameters.getParameter<double>("ptMax");
  trackHitPercentualVsPt = dbe->book2D("trackHitDivtrackSegmHitVsPt", "trackHitDivtrackSegmHitVsPt", ptBin, ptMin, ptMax, 50, 0, 5);
  dtTrackHitPercentualVsPt = dbe->book2D("dtTrackHitDivtrackSegmHitVsPt", "dtTrackHitDivtrackSegmHitVsPt", ptBin, ptMin, ptMax, 20, 0, 2);
  cscTrackHitPercentualVsPt = dbe->book2D("cscTrackHitDivtrackSegmHitVsPt", "cscTrackHitDivtrackSegmHitVsPt", ptBin, ptMin, ptMax, 20, 0, 2);


}


void SegmentTrackAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup, const reco::Track& recoTrack){

  LogTrace(metname)<<"[SegmentTrackAnalyzer] Filling the histos";
  
  MuonTransientTrackingRecHit::MuonRecHitContainer segments = theSegmentsAssociator->associate(iEvent, iSetup, recoTrack );
 
  LogTrace(metname)<<"[SegmentTrackAnalyzer] # of segments associated to the track: "<<(segments).size();

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

  for (MuonTransientTrackingRecHit::MuonRecHitContainer::const_iterator segment=segments.begin();
       segment!=segments.end(); segment++) {
   
    DetId id = (*segment)->geographicalId();
    
    // hits from DT segments
    if (id.det() == DetId::Muon && id.subdetId() == MuonSubdetId::DT ) {
      ++segmFromDt;
      const DTRecSegment4D *seg4D = dynamic_cast<const DTRecSegment4D*>((*segment)->hit());
      if((*seg4D).hasPhi())
	hitsFromSegmDt+=(*seg4D).phiSegment()->specificRecHits().size();
      if((*seg4D).hasZed())
	hitsFromSegmDt+=(*seg4D).zSegment()->specificRecHits().size();
      
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
  hitsNotUsedPercentual->Fill(double(hitsFromSegmDt+hitsFromSegmCsc+hitsFromRpc+hitsFromTk-hitsFromTrack)/hitsFromTrack);

  TrackSegm->Fill(1,segmFromDt+segmFromCsc);
  TrackSegm->Fill(2,segmFromDt);
  TrackSegm->Fill(3,segmFromCsc);

  if(hitsFromDt!=0 && hitsFromCsc==0 && hitsFromRpc==0) hitStaProvenance->Fill(1);
  if(hitsFromCsc!=0 && hitsFromDt==0 && hitsFromRpc==0) hitStaProvenance->Fill(2);
  if(hitsFromRpc!=0 && hitsFromDt==0 && hitsFromCsc==0) hitStaProvenance->Fill(3);
  if(hitsFromDt!=0 && hitsFromCsc!=0 && hitsFromRpc==0) hitStaProvenance->Fill(4);
  if(hitsFromDt!=0 && hitsFromRpc!=0 && hitsFromCsc==0) hitStaProvenance->Fill(5);
  if(hitsFromCsc!=0 && hitsFromRpc!=0 && hitsFromDt==0) hitStaProvenance->Fill(6);
  if(hitsFromDt!=0 && hitsFromCsc!=0 && hitsFromRpc!=0) hitStaProvenance->Fill(7);

  if(hitsFromSegmDt+hitsFromSegmCsc !=0){
    trackHitPercentualVsEta->Fill(recoTrack.eta(), double(hitsFromTrack)/(hitsFromSegmDt+hitsFromSegmCsc));
    trackHitPercentualVsPhi->Fill(recoTrack.phi(), double(hitsFromTrack)/(hitsFromSegmDt+hitsFromSegmCsc));
    trackHitPercentualVsPt->Fill(recoTrack.pt(), double(hitsFromTrack)/(hitsFromSegmDt+hitsFromSegmCsc));
  }

  if(hitsFromSegmDt!=0){
    dtTrackHitPercentualVsEta->Fill(recoTrack.eta(), double(hitsFromDt)/hitsFromSegmDt);
    dtTrackHitPercentualVsPhi->Fill(recoTrack.phi(), double(hitsFromDt)/hitsFromSegmDt);
    dtTrackHitPercentualVsPt->Fill(recoTrack.pt(), double(hitsFromDt)/hitsFromSegmDt);
  }

  if(hitsFromSegmCsc!=0){
    cscTrackHitPercentualVsEta->Fill(recoTrack.eta(), double(hitsFromCsc)/hitsFromSegmCsc);
    cscTrackHitPercentualVsPhi->Fill(recoTrack.phi(), double(hitsFromCsc)/hitsFromSegmCsc);
    cscTrackHitPercentualVsPt->Fill(recoTrack.pt(), double(hitsFromCsc)/hitsFromSegmCsc);
  }

} 

