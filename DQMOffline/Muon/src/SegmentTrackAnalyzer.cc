
/*
 *  See header file for a description of this class.
 *
 *  \author G. Mila - INFN Torino
 */

#include "DQMOffline/Muon/interface/SegmentTrackAnalyzer.h"

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/MuonDetId/interface/MuonSubdetId.h"
#include "RecoMuon/TransientTrackingRecHit/interface/MuonTransientTrackingRecHit.h"
#include "DataFormats/DTRecHit/interface/DTRecSegment4DCollection.h"
#include "DataFormats/DTRecHit/interface/DTRecHitCollection.h"
#include "DataFormats/SiStripDetId/interface/SiStripDetId.h"
#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"
#include "RecoMuon/TrackingTools/interface/MuonServiceProxy.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <string>

using namespace std;
using namespace edm;

//#define DEBUG

SegmentTrackAnalyzer::SegmentTrackAnalyzer(const edm::ParameterSet& pSet) {
  parameters = pSet;
  
  // MuonService
  theService = new MuonServiceProxy(parameters.getParameter<ParameterSet>("ServiceParameters"));
  theDbe = edm::Service<DQMStore>().operator->();
  
  // Read Data:
  theMuTrackCollectionLabel_ = consumes<reco::TrackCollection>(parameters.getParameter<edm::InputTag>("MuTrackCollection"));

  // SegmentTrackAssociatior Parameters
  const ParameterSet SegmentsTrackAssociatorParameters = parameters.getParameter<ParameterSet>("SegmentsTrackAssociatorParameters");

  edm::ConsumesCollector iC   = consumesCollector();
  theSegmentsAssociator = new SegmentsTrackAssociator(SegmentsTrackAssociatorParameters,iC);
  
  
}
void SegmentTrackAnalyzer::beginJob() {
  metname = "segmTrackAnalyzer";
  LogTrace(metname)<<"[SegmentTrackAnalyzer] Parameters initialization";
  
}
void SegmentTrackAnalyzer::beginRun(const edm::Run& iRun, const edm::EventSetup& iSetup){
  metname = "segmTrackAnalyzer";
  
  theDbe->cd();
  theDbe->setCurrentFolder("Muons/SegmentTrackAnalyzer");
  
  string trackCollection = parameters.getParameter<edm::InputTag>("MuTrackCollection").label() + parameters.getParameter<edm::InputTag>("MuTrackCollection").instance();

  // histograms initalization
  hitsNotUsed = theDbe->book1D("HitsNotUsedForGlobalTracking_"+trackCollection, "recHits not used for GLB    ["+trackCollection+"]", 50, -0.5, 49.5);
  hitsNotUsedPercentual = theDbe->book1D("HitsNotUsedForGlobalTrackingDvHitUsed_"+trackCollection, "(recHits_{notUsedForGLB}) / (recHits_{GLB})    ["+trackCollection+"]", 100, 0, 1.);

  TrackSegm = theDbe->book2D("trackSegments_"+trackCollection, "Number of segments associated to the track    ["+trackCollection+"]", 3, 0.5, 3.5, 8, 0, 8);
  TrackSegm->setBinLabel(1,"DT+CSC",1);
  TrackSegm->setBinLabel(2,"DT",1);
  TrackSegm->setBinLabel(3,"CSC",1);
  
  hitStaProvenance = theDbe->book1D("trackHitStaProvenance_"+trackCollection, "Number of recHits_{STAinTrack}    ["+trackCollection+"]", 7, 0.5, 7.5);
  hitStaProvenance->setBinLabel(1,"DT");
  hitStaProvenance->setBinLabel(2,"CSC");
  hitStaProvenance->setBinLabel(3,"RPC");
  hitStaProvenance->setBinLabel(4,"DT+CSC");
  hitStaProvenance->setBinLabel(5,"DT+RPC");
  hitStaProvenance->setBinLabel(6,"CSC+RPC");
  hitStaProvenance->setBinLabel(7,"DT+CSC+RPC");


  if(trackCollection!="standAloneMuons"){
    hitTkrProvenance = theDbe->book1D("trackHitTkrProvenance_"+trackCollection, "Number of recHits_{TKinTrack}    ["+trackCollection+"]", 6, 0.5, 6.5);
    hitTkrProvenance->setBinLabel(1,"PixBarrel");
    hitTkrProvenance->setBinLabel(2,"PixEndCap");
    hitTkrProvenance->setBinLabel(3,"TIB");
    hitTkrProvenance->setBinLabel(4,"TID");
    hitTkrProvenance->setBinLabel(5,"TOB");
    hitTkrProvenance->setBinLabel(6,"TEC");
  }

  int etaBin = parameters.getParameter<int>("etaBin");
  double etaMin = parameters.getParameter<double>("etaMin");
  double etaMax = parameters.getParameter<double>("etaMax");
  trackHitPercentualVsEta = theDbe->book2D("trackHitDivtrackSegmHitVsEta_"+trackCollection, "(recHits_{Track} / recHits_{associatedSegm}) vs #eta    [" +trackCollection+"]", etaBin, etaMin, etaMax, 20, 0, 1);
  dtTrackHitPercentualVsEta = theDbe->book2D("dtTrackHitDivtrackSegmHitVsEta_"+trackCollection, "(recHits_{DTinTrack} / recHits_{associatedSegm}) vs #eta    [" +trackCollection+"]", etaBin, etaMin, etaMax, 20, 0, 1);
  cscTrackHitPercentualVsEta = theDbe->book2D("cscTrackHitDivtrackSegmHitVsEta_"+trackCollection, "(recHits_{CSCinTrack} / recHits_{associatedSegm}) vs #eta    [" +trackCollection+"]", etaBin, etaMin, etaMax, 20, 0, 1);

  int phiBin = parameters.getParameter<int>("phiBin");
  double phiMin = parameters.getParameter<double>("phiMin");
  double phiMax = parameters.getParameter<double>("phiMax");
  trackHitPercentualVsPhi = theDbe->book2D("trackHitDivtrackSegmHitVsPhi_"+trackCollection, "(recHits_{Track} / recHits_{associatedSegm}) vs #phi    [" +trackCollection+"]", phiBin, phiMin, phiMax, 20, 0, 1);
  trackHitPercentualVsPhi->setAxisTitle("rad",2);
  dtTrackHitPercentualVsPhi = theDbe->book2D("dtTrackHitDivtrackSegmHitVsPhi_"+trackCollection, "(recHits_{DTinTrack} / recHits_{associatedSegm}) vs #phi    [" +trackCollection+"]", phiBin, phiMin, phiMax, 20, 0, 1);
  dtTrackHitPercentualVsPhi->setAxisTitle("rad",2);
  cscTrackHitPercentualVsPhi = theDbe->book2D("cscTrackHitDivtrackSegmHitVsPhi_"+trackCollection, "(recHits_{CSCinTrack} / recHits_{associatedSegm}) vs #phi    [" +trackCollection+"]", phiBin, phiMin, phiMax, 20, 0, 1);
  cscTrackHitPercentualVsPhi->setAxisTitle("rad",2);

  int ptBin = parameters.getParameter<int>("ptBin");
  double ptMin = parameters.getParameter<double>("ptMin");
  double ptMax = parameters.getParameter<double>("ptMax");
  trackHitPercentualVsPt = theDbe->book2D("trackHitDivtrackSegmHitVsPt_"+trackCollection, "(recHits_{Track} / recHits_{associatedSegm}) vs 1/p_{t}    [" +trackCollection+"]", ptBin, ptMin, ptMax, 20, 0, 1);
  trackHitPercentualVsPt->setAxisTitle("GeV",2);
  dtTrackHitPercentualVsPt = theDbe->book2D("dtTrackHitDivtrackSegmHitVsPt_"+trackCollection, "(recHits_{DTinTrack} / recHits_{associatedSegm}) vs 1/p_{t}    [" +trackCollection+"]", ptBin, ptMin, ptMax, 20, 0, 1);
  dtTrackHitPercentualVsPt->setAxisTitle("GeV",2);
  cscTrackHitPercentualVsPt = theDbe->book2D("cscTrackHitDivtrackSegmHitVsPt_"+trackCollection, "(recHits_{CSCinTrack} / recHits_{associatedSegm}) vs 1/p_{t}    [" +trackCollection+"]", ptBin, ptMin, ptMax, 20, 0, 1);
  cscTrackHitPercentualVsPt->setAxisTitle("GeV",2);

}


void SegmentTrackAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup){
  LogTrace(metname)<<"[SegmentTrackAnalyzer] Filling the histos";
  theService->update(iSetup);
  
  Handle<reco::TrackCollection> glbTracks;
  iEvent.getByToken(theMuTrackCollectionLabel_,glbTracks);
  
  for (reco::TrackCollection::const_iterator recoTrack = glbTracks->begin(); recoTrack!=glbTracks->end(); ++recoTrack){
    MuonTransientTrackingRecHit::MuonRecHitContainer segments = theSegmentsAssociator->associate(iEvent, iSetup, *recoTrack );
  
#ifdef DEBUG
    cout << "[SegmentTrackAnalyzer] # of segments associated to the track: "<<(segments).size() <<endl;
#endif
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
    for(trackingRecHit_iterator recHit =  recoTrack->recHitsBegin(); recHit != recoTrack->recHitsEnd(); ++recHit){

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
      if (id.det() == DetId::Tracker){
	hitsFromTk++;
	if(id.subdetId() == PixelSubdetector::PixelBarrel )
	  hitTkrProvenance->Fill(1);
	if(id.subdetId() == PixelSubdetector::PixelEndcap )
	  hitTkrProvenance->Fill(2);
	if(id.subdetId() == SiStripDetId::TIB )
	  hitTkrProvenance->Fill(3);
	if(id.subdetId() == SiStripDetId::TID )
	  hitTkrProvenance->Fill(4);
	if(id.subdetId() == SiStripDetId::TOB )
	  hitTkrProvenance->Fill(5);
	if(id.subdetId() == SiStripDetId::TEC )
	  hitTkrProvenance->Fill(6);
      }

    }

    // fill the histos
    hitsNotUsed->Fill(hitsFromSegmDt+hitsFromSegmCsc+hitsFromRpc+hitsFromTk-hitsFromTrack);
    hitsNotUsedPercentual->Fill(double(hitsFromSegmDt+hitsFromSegmCsc+hitsFromRpc+hitsFromTk-hitsFromTrack)/hitsFromTrack);

    if(hitsFromDt!=0 && hitsFromCsc!=0)
      TrackSegm->Fill(1,segmFromDt+segmFromCsc);
    if(hitsFromDt!=0 && hitsFromCsc==0)
      TrackSegm->Fill(2,segmFromDt);
    if(hitsFromDt==0 && hitsFromCsc!=0)
      TrackSegm->Fill(3,segmFromCsc);

    if(hitsFromDt!=0 && hitsFromCsc==0 && hitsFromRpc==0) hitStaProvenance->Fill(1);
    if(hitsFromCsc!=0 && hitsFromDt==0 && hitsFromRpc==0) hitStaProvenance->Fill(2);
    if(hitsFromRpc!=0 && hitsFromDt==0 && hitsFromCsc==0) hitStaProvenance->Fill(3);
    if(hitsFromDt!=0 && hitsFromCsc!=0 && hitsFromRpc==0) hitStaProvenance->Fill(4);
    if(hitsFromDt!=0 && hitsFromRpc!=0 && hitsFromCsc==0) hitStaProvenance->Fill(5);
    if(hitsFromCsc!=0 && hitsFromRpc!=0 && hitsFromDt==0) hitStaProvenance->Fill(6);
    if(hitsFromDt!=0 && hitsFromCsc!=0 && hitsFromRpc!=0) hitStaProvenance->Fill(7);

    if(hitsFromSegmDt+hitsFromSegmCsc !=0){
      trackHitPercentualVsEta->Fill(recoTrack->eta(), double(hitsFromDt+hitsFromCsc)/(hitsFromSegmDt+hitsFromSegmCsc));
      trackHitPercentualVsPhi->Fill(recoTrack->phi(), double(hitsFromDt+hitsFromCsc)/(hitsFromSegmDt+hitsFromSegmCsc));
      trackHitPercentualVsPt->Fill(recoTrack->pt(), double(hitsFromDt+hitsFromCsc)/(hitsFromSegmDt+hitsFromSegmCsc));
    }

    if(hitsFromSegmDt!=0){
      dtTrackHitPercentualVsEta->Fill(recoTrack->eta(), double(hitsFromDt)/hitsFromSegmDt);
      dtTrackHitPercentualVsPhi->Fill(recoTrack->phi(), double(hitsFromDt)/hitsFromSegmDt);
      dtTrackHitPercentualVsPt->Fill(recoTrack->pt(), double(hitsFromDt)/hitsFromSegmDt);
    }

    if(hitsFromSegmCsc!=0){
      cscTrackHitPercentualVsEta->Fill(recoTrack->eta(), double(hitsFromCsc)/hitsFromSegmCsc);
      cscTrackHitPercentualVsPhi->Fill(recoTrack->phi(), double(hitsFromCsc)/hitsFromSegmCsc);
      cscTrackHitPercentualVsPt->Fill(recoTrack->pt(), double(hitsFromCsc)/hitsFromSegmCsc);
    }
  }

} 

