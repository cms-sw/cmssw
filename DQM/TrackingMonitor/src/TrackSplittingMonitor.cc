/*
 *  See header file for a description of this class.
 *
 *  \author Suchandra Dutta , Giorgia Mila
 */

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"
#include "DataFormats/TrackCandidate/interface/TrackCandidateCollection.h" 
#include "TrackingTools/TransientTrack/interface/TransientTrackBuilder.h"
#include "TrackingTools/Records/interface/TransientTrackRecord.h"
#include "TrackingTools/TransientTrack/interface/TransientTrack.h"
#include "MagneticField/Engine/interface/MagneticField.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "DQMServices/Core/interface/DQMStore.h"
//#include "DQM/TrackingMonitor/interface/TrackAnalyzer.h"
#include "DQM/TrackingMonitor/interface/TrackSplittingMonitor.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h" 

#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"
#include "TrackingTools/Records/interface/TransientRecHitRecord.h" 
#include "TrackingTools/PatternTools/interface/TSCBLBuilderNoMaterial.h"
#include "TrackingTools/PatternTools/interface/TSCPBuilderNoMaterial.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include <string>

TrackSplittingMonitor::TrackSplittingMonitor(const edm::ParameterSet& iConfig) 
  : dqmStore_( edm::Service<DQMStore>().operator->() )
  , conf_( iConfig )
{

  splitTracks_ = conf_.getParameter< edm::InputTag >("splitTrackCollection");
  splitMuons_ = conf_.getParameter< edm::InputTag >("splitMuonCollection");
  splitTracksToken_ = consumes<std::vector<reco::Track> >(splitTracks_);
  splitMuonsToken_  = mayConsume<std::vector<reco::Muon> >(splitMuons_);
  plotMuons_ = conf_.getParameter<bool>("ifPlotMuons");

  // cuts
  pixelHitsPerLeg_ = conf_.getParameter<int>("pixelHitsPerLeg");
  totalHitsPerLeg_ = conf_.getParameter<int>("totalHitsPerLeg");
  d0Cut_ = conf_.getParameter<double>("d0Cut");
  dzCut_ = conf_.getParameter<double>("dzCut");
  ptCut_ = conf_.getParameter<double>("ptCut");
  norchiCut_ = conf_.getParameter<double>("norchiCut");
}

TrackSplittingMonitor::~TrackSplittingMonitor() { 
}

void TrackSplittingMonitor::bookHistograms(DQMStore::IBooker & ibooker,
					   edm::Run const & /* iRun */,
					   edm::EventSetup const & /* iSetup */)
  
{

  std::string MEFolderName = conf_.getParameter<std::string>("FolderName"); 
  ibooker.setCurrentFolder(MEFolderName);
  
  // bin declarations
  int    ddxyBin = conf_.getParameter<int>("ddxyBin");
  double ddxyMin = conf_.getParameter<double>("ddxyMin");
  double ddxyMax = conf_.getParameter<double>("ddxyMax");
  
  int    ddzBin = conf_.getParameter<int>("ddzBin");
  double ddzMin = conf_.getParameter<double>("ddzMin");
  double ddzMax = conf_.getParameter<double>("ddzMax");
  
  int    dphiBin = conf_.getParameter<int>("dphiBin");
  double dphiMin = conf_.getParameter<double>("dphiMin");
  double dphiMax = conf_.getParameter<double>("dphiMax");
  
  int    dthetaBin = conf_.getParameter<int>("dthetaBin");
  double dthetaMin = conf_.getParameter<double>("dthetaMin");
  double dthetaMax = conf_.getParameter<double>("dthetaMax");
  
  int    dptBin = conf_.getParameter<int>("dptBin");
  double dptMin = conf_.getParameter<double>("dptMin");
  double dptMax = conf_.getParameter<double>("dptMax");
  
  int    dcurvBin = conf_.getParameter<int>("dcurvBin");
  double dcurvMin = conf_.getParameter<double>("dcurvMin");
  double dcurvMax = conf_.getParameter<double>("dcurvMax");
  
  int    normBin = conf_.getParameter<int>("normBin");
  double normMin = conf_.getParameter<double>("normMin");
  double normMax = conf_.getParameter<double>("normMax");	
	
  // declare histogram
  ddxyAbsoluteResiduals_tracker_ = ibooker.book1D( "ddxyAbsoluteResiduals_tracker", "ddxyAbsoluteResiduals_tracker", ddxyBin, ddxyMin, ddxyMax );
  ddzAbsoluteResiduals_tracker_ = ibooker.book1D( "ddzAbsoluteResiduals_tracker", "ddzAbsoluteResiduals_tracker", ddzBin, ddzMin, ddzMax );
  dphiAbsoluteResiduals_tracker_ = ibooker.book1D( "dphiAbsoluteResiduals_tracker", "dphiAbsoluteResiduals_tracker", dphiBin, dphiMin, dphiMax );
  dthetaAbsoluteResiduals_tracker_ = ibooker.book1D( "dthetaAbsoluteResiduals_tracker", "dthetaAbsoluteResiduals_tracker", dthetaBin, dthetaMin, dthetaMax );
  dptAbsoluteResiduals_tracker_ = ibooker.book1D( "dptAbsoluteResiduals_tracker", "dptAbsoluteResiduals_tracker", dptBin, dptMin, dptMax );
  dcurvAbsoluteResiduals_tracker_ = ibooker.book1D( "dcurvAbsoluteResiduals_tracker", "dcurvAbsoluteResiduals_tracker", dcurvBin, dcurvMin, dcurvMax );
  
  ddxyNormalizedResiduals_tracker_ = ibooker.book1D( "ddxyNormalizedResiduals_tracker", "ddxyNormalizedResiduals_tracker", normBin, normMin, normMax );
  ddzNormalizedResiduals_tracker_ = ibooker.book1D( "ddzNormalizedResiduals_tracker", "ddzNormalizedResiduals_tracker", normBin, normMin, normMax );
  dphiNormalizedResiduals_tracker_ = ibooker.book1D( "dphiNormalizedResiduals_tracker", "dphiNormalizedResiduals_tracker", normBin, normMin, normMax );
  dthetaNormalizedResiduals_tracker_ = ibooker.book1D( "dthetaNormalizedResiduals_tracker", "dthetaNormalizedResiduals_tracker", normBin, normMin, normMax );
  dptNormalizedResiduals_tracker_ = ibooker.book1D( "dptNormalizedResiduals_tracker", "dptNormalizedResiduals_tracker", normBin, normMin, normMax );
  dcurvNormalizedResiduals_tracker_ = ibooker.book1D( "dcurvNormalizedResiduals_tracker", "dcurvNormalizedResiduals_tracker", normBin, normMin, normMax );
	
  if (plotMuons_){
    ddxyAbsoluteResiduals_global_ = ibooker.book1D( "ddxyAbsoluteResiduals_global", "ddxyAbsoluteResiduals_global", ddxyBin, ddxyMin, ddxyMax );
    ddzAbsoluteResiduals_global_ = ibooker.book1D( "ddzAbsoluteResiduals_global", "ddzAbsoluteResiduals_global", ddzBin, ddzMin, ddzMax );
    dphiAbsoluteResiduals_global_ = ibooker.book1D( "dphiAbsoluteResiduals_global", "dphiAbsoluteResiduals_global", dphiBin, dphiMin, dphiMax );
    dthetaAbsoluteResiduals_global_ = ibooker.book1D( "dthetaAbsoluteResiduals_global", "dthetaAbsoluteResiduals_global", dthetaBin, dthetaMin, dthetaMax );
    dptAbsoluteResiduals_global_ = ibooker.book1D( "dptAbsoluteResiduals_global", "dptAbsoluteResiduals_global", dptBin, dptMin, dptMax );
    dcurvAbsoluteResiduals_global_ = ibooker.book1D( "dcurvAbsoluteResiduals_global", "dcurvAbsoluteResiduals_global", dcurvBin, dcurvMin, dcurvMax );
    
    ddxyNormalizedResiduals_global_ = ibooker.book1D( "ddxyNormalizedResiduals_global", "ddxyNormalizedResiduals_global", normBin, normMin, normMax );
    ddzNormalizedResiduals_global_ = ibooker.book1D( "ddzNormalizedResiduals_global", "ddzNormalizedResiduals_global", normBin, normMin, normMax );
    dphiNormalizedResiduals_global_ = ibooker.book1D( "dphiNormalizedResiduals_global", "dphiNormalizedResiduals_global", normBin, normMin, normMax );
    dthetaNormalizedResiduals_global_ = ibooker.book1D( "dthetaNormalizedResiduals_global", "dthetaNormalizedResiduals_global", normBin, normMin, normMax );
    dptNormalizedResiduals_global_ = ibooker.book1D( "dptNormalizedResiduals_global", "dptNormalizedResiduals_global", normBin, normMin, normMax );
    dcurvNormalizedResiduals_global_ = ibooker.book1D( "dcurvNormalizedResiduals_global", "dcurvNormalizedResiduals_global", normBin, normMin, normMax );
  }
  
  ddxyAbsoluteResiduals_tracker_->setAxisTitle( "(#delta d_{xy})/#sqrt{2} [#mum]" );
  ddxyAbsoluteResiduals_tracker_->setAxisTitle( "(#delta d_{z})/#sqrt{2} [#mum]" );
  ddxyAbsoluteResiduals_tracker_->setAxisTitle( "(#delta #phi)/#sqrt{2} [mrad]" );
  ddxyAbsoluteResiduals_tracker_->setAxisTitle( "(#delta #theta)/#sqrt{2} [mrad]" );
  ddxyAbsoluteResiduals_tracker_->setAxisTitle( "(#delta pT)/#sqrt{2} [GeV]" );
  ddxyAbsoluteResiduals_tracker_->setAxisTitle( "(#delta (1/pT))/#sqrt{2} [GeV^{-1}]" );
  
  ddxyNormalizedResiduals_tracker_->setAxisTitle( "#delta d_{xy}/#sigma(d_{xy}" );
  ddxyNormalizedResiduals_tracker_->setAxisTitle( "#delta d_{z}/#sigma(d_{z})" );
  ddxyNormalizedResiduals_tracker_->setAxisTitle( "#delta #phi/#sigma(d_{#phi})" );
  ddxyNormalizedResiduals_tracker_->setAxisTitle( "#delta #theta/#sigma(d_{#theta})" );
  ddxyNormalizedResiduals_tracker_->setAxisTitle( "#delta p_{T}/#sigma(p_{T})" );
  ddxyNormalizedResiduals_tracker_->setAxisTitle( "#delta 1/p_{T}/#sigma(1/p_{T})" );
  
  if (plotMuons_){
    ddxyAbsoluteResiduals_global_->setAxisTitle( "(#delta d_{xy})/#sqrt{2} [#mum]" );
    ddxyAbsoluteResiduals_global_->setAxisTitle( "(#delta d_{z})/#sqrt{2} [#mum]" );
    ddxyAbsoluteResiduals_global_->setAxisTitle( "(#delta #phi)/#sqrt{2} [mrad]" );
    ddxyAbsoluteResiduals_global_->setAxisTitle( "(#delta #theta)/#sqrt{2} [mrad]" );
    ddxyAbsoluteResiduals_global_->setAxisTitle( "(#delta pT)/#sqrt{2} [GeV]" );
    ddxyAbsoluteResiduals_global_->setAxisTitle( "(#delta (1/pT))/#sqrt{2} [GeV^{-1}]" );
    
    ddxyNormalizedResiduals_global_->setAxisTitle( "#delta d_{xy}/#sigma(d_{xy}" );
    ddxyNormalizedResiduals_global_->setAxisTitle( "#delta d_{z}/#sigma(d_{z})" );
    ddxyNormalizedResiduals_global_->setAxisTitle( "#delta #phi/#sigma(d_{#phi})" );
    ddxyNormalizedResiduals_global_->setAxisTitle( "#delta #theta/#sigma(d_{#theta})" );
    ddxyNormalizedResiduals_global_->setAxisTitle( "#delta p_{T}/#sigma(p_{T})" );
    ddxyNormalizedResiduals_global_->setAxisTitle( "#delta 1/p_{T}/#sigma(1/p_{T})" );
  }

}


//
// -- Analyse
//
void TrackSplittingMonitor::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
	
  
  iSetup.get<IdealMagneticFieldRecord>().get(theMagField);   
  iSetup.get<TrackerDigiGeometryRecord>().get(theGeometry);
  iSetup.get<MuonGeometryRecord>().get(dtGeometry);
  iSetup.get<MuonGeometryRecord>().get(cscGeometry);
  iSetup.get<MuonGeometryRecord>().get(rpcGeometry);
  
  edm::Handle<std::vector<reco::Track> > splitTracks;
  iEvent.getByToken(splitTracksToken_, splitTracks);
  if (!splitTracks.isValid()) return;

  edm::Handle<std::vector<reco::Muon> > splitMuons;
  if (plotMuons_){
    iEvent.getByToken(splitMuonsToken_, splitMuons);
  }
  
  if (splitTracks->size() == 2){
    // check that there are 2 tracks in split track collection
    edm::LogInfo("TrackSplittingMonitor") << "Split Track size: " << splitTracks->size();
    
    // split tracks calculations
    reco::Track track1 = splitTracks->at(0);
    reco::Track track2 = splitTracks->at(1);
    
    
    // -------------------------- basic selection ---------------------------
    
    // hit counting
    // looping through the hits for track 1
    double nRechits1 =0;
    double nRechitinBPIX1 =0;
    for (trackingRecHit_iterator iHit = track1.recHitsBegin(); iHit != track1.recHitsEnd(); ++iHit) {
      if((*iHit)->isValid()) {       
	nRechits1++;
	int type =(*iHit)->geographicalId().subdetId();
	if(type==int(PixelSubdetector::PixelBarrel)){++nRechitinBPIX1;}
      }
    }    
    // looping through the hits for track 2
    double nRechits2 =0;
    double nRechitinBPIX2 =0;
    for (trackingRecHit_iterator iHit = track2.recHitsBegin(); iHit != track2.recHitsEnd(); ++iHit) {
      if((*iHit)->isValid()) {       
	nRechits2++;
	int type =(*iHit)->geographicalId().subdetId();
	if(type==int(PixelSubdetector::PixelBarrel)){++nRechitinBPIX2;}
      }
    }    
    
    // DCA of each track
    double d01 = track1.d0();
    double dz1 = track1.dz();
    double d02 = track2.d0();
    double dz2 = track2.dz();
    
    // pT of each track
    double pt1 = track1.pt();
    double pt2 = track2.pt();
    
    // chi2 of each track
    double norchi1 = track1.normalizedChi2();
    double norchi2 = track2.normalizedChi2();		
    
    // basic selection
    // pixel hits and total hits
    if ((nRechitinBPIX1 >= pixelHitsPerLeg_)&&(nRechitinBPIX1 >= pixelHitsPerLeg_)&&(nRechits1 >= totalHitsPerLeg_)&&(nRechits2 >= totalHitsPerLeg_)){
      // dca cut
      if ( ((fabs(d01) < d0Cut_))&&(fabs(d02) < d0Cut_)&&(fabs(dz2) < dzCut_)&&(fabs(dz2) < dzCut_) ){
	// pt cut
	if ( (pt1+pt2)/2 < ptCut_){
	  // chi2 cut
	  if ( (norchi1 < norchiCut_)&&(norchi2 < norchiCut_) ){
	    
	    // passed all cuts...
	    edm::LogInfo("TrackSplittingMonitor") << " Setected after all cuts ?";

	    double ddxyVal = d01 - d02;
	    double ddzVal = dz1 - dz2;						
	    double dphiVal = track1.phi() - track2.phi();
	    double dthetaVal = track1.theta() - track2.theta();						
	    double dptVal = pt1 - pt2;
	    double dcurvVal = (1/pt1) - (1/pt2);
	    
	    double d01ErrVal = track1.d0Error();
	    double d02ErrVal = track2.d0Error();
	    double dz1ErrVal = track1.dzError();
	    double dz2ErrVal = track2.dzError();
	    double phi1ErrVal = track1.phiError();
	    double phi2ErrVal = track2.phiError();
	    double theta1ErrVal = track1.thetaError();
	    double theta2ErrVal = track2.thetaError();
	    double pt1ErrVal = track1.ptError();
	    double pt2ErrVal = track2.ptError();
	    
	    ddxyAbsoluteResiduals_tracker_->Fill( 10000.0*ddxyVal/sqrt(2.0) );
	    ddxyAbsoluteResiduals_tracker_->Fill( 10000.0*ddzVal/sqrt(2.0) );
	    ddxyAbsoluteResiduals_tracker_->Fill( 1000.0*dphiVal/sqrt(2.0) );
	    ddxyAbsoluteResiduals_tracker_->Fill( 1000.0*dthetaVal/sqrt(2.0) );
	    ddxyAbsoluteResiduals_tracker_->Fill( dptVal/sqrt(2.0) );
	    ddxyAbsoluteResiduals_tracker_->Fill( dcurvVal/sqrt(2.0) );
	    
	    ddxyNormalizedResiduals_tracker_->Fill( ddxyVal/sqrt( d01ErrVal*d01ErrVal + d02ErrVal*d02ErrVal ) );
	    ddxyNormalizedResiduals_tracker_->Fill( ddzVal/sqrt( dz1ErrVal*dz1ErrVal + dz2ErrVal*dz2ErrVal ) );
	    ddxyNormalizedResiduals_tracker_->Fill( dphiVal/sqrt( phi1ErrVal*phi1ErrVal + phi2ErrVal*phi2ErrVal ) );
	    ddxyNormalizedResiduals_tracker_->Fill( dthetaVal/sqrt( theta1ErrVal*theta1ErrVal + theta2ErrVal*theta2ErrVal ) );
	    ddxyNormalizedResiduals_tracker_->Fill( dptVal/sqrt( pt1ErrVal*pt1ErrVal + pt2ErrVal*pt2ErrVal ) );
	    ddxyNormalizedResiduals_tracker_->Fill( dcurvVal/sqrt( pow(pt1ErrVal,2)/pow(pt1,4) + pow(pt2ErrVal,2)/pow(pt2,4) ) );
	    
	    // if do the same for split muons
	    if (plotMuons_ && splitMuons.isValid()){
	      
	      int gmCtr = 0; 
	      bool topGlobalMuonFlag = false;
	      bool bottomGlobalMuonFlag = false;
	      int topGlobalMuon = -1;
	      int bottomGlobalMuon = -1;
	      double topGlobalMuonNorchi2 = 1e10;
	      double bottomGlobalMuonNorchi2 = 1e10;
	      
	      // check if usable split global muons
	      for (std::vector<reco::Muon>::const_iterator gmI = splitMuons->begin(); gmI != splitMuons->end(); gmI++){
		if ( gmI->isTrackerMuon() && gmI->isStandAloneMuon() && gmI->isGlobalMuon() ){
		  
		  reco::TrackRef trackerTrackRef1( splitTracks, 0 );
		  reco::TrackRef trackerTrackRef2( splitTracks, 1 );
		  
		  if (gmI->innerTrack() == trackerTrackRef1){
		    if (gmI->globalTrack()->normalizedChi2() < topGlobalMuonNorchi2){
		      topGlobalMuonFlag = true;
		      topGlobalMuonNorchi2 = gmI->globalTrack()->normalizedChi2();
		      topGlobalMuon = gmCtr;
		    }
		  }
		  if (gmI->innerTrack() == trackerTrackRef2){
		    if (gmI->globalTrack()->normalizedChi2() < bottomGlobalMuonNorchi2){
		      bottomGlobalMuonFlag = true;
		      bottomGlobalMuonNorchi2 = gmI->globalTrack()->normalizedChi2();
		      bottomGlobalMuon = gmCtr;
		    }
		  }
		}
		gmCtr++;
	      } 
	      
	      if (bottomGlobalMuonFlag && topGlobalMuonFlag) {
		
		reco::Muon muonTop = splitMuons->at( topGlobalMuon );
		reco::Muon muonBottom = splitMuons->at( bottomGlobalMuon );                            
		
		reco::TrackRef glb1 = muonTop.globalTrack();
		reco::TrackRef glb2 = muonBottom.globalTrack();
		
		double ddxyValGlb = glb1->d0() - glb2->d0();
		double ddzValGlb = glb1->dz() - glb2->dz();						
		double dphiValGlb = glb1->phi() - glb2->phi();
		double dthetaValGlb = glb1->theta() - glb2->theta();						
		double dptValGlb = glb1->pt() - glb2->pt();
		double dcurvValGlb = (1/glb1->pt()) - (1/glb2->pt());
		
		double d01ErrValGlb = glb1->d0Error();
		double d02ErrValGlb = glb2->d0Error();
		double dz1ErrValGlb = glb1->dzError();
		double dz2ErrValGlb = glb2->dzError();
		double phi1ErrValGlb = glb1->phiError();
		double phi2ErrValGlb = glb2->phiError();
		double theta1ErrValGlb = glb1->thetaError();
		double theta2ErrValGlb = glb2->thetaError();
		double pt1ErrValGlb = glb1->ptError();
		double pt2ErrValGlb = glb2->ptError();
		
		ddxyAbsoluteResiduals_global_->Fill( 10000.0*ddxyValGlb/sqrt(2.0) );
		ddxyAbsoluteResiduals_global_->Fill( 10000.0*ddzValGlb/sqrt(2.0) );
		ddxyAbsoluteResiduals_global_->Fill( 1000.0*dphiValGlb/sqrt(2.0) );
		ddxyAbsoluteResiduals_global_->Fill( 1000.0*dthetaValGlb/sqrt(2.0) );
		ddxyAbsoluteResiduals_global_->Fill( dptValGlb/sqrt(2.0) );
		ddxyAbsoluteResiduals_global_->Fill( dcurvValGlb/sqrt(2.0) );
		
		ddxyNormalizedResiduals_global_->Fill( ddxyValGlb/sqrt( d01ErrValGlb*d01ErrValGlb + d02ErrValGlb*d02ErrValGlb ) );
		ddxyNormalizedResiduals_global_->Fill( ddzValGlb/sqrt( dz1ErrValGlb*dz1ErrValGlb + dz2ErrValGlb*dz2ErrValGlb ) );
		ddxyNormalizedResiduals_global_->Fill( dphiValGlb/sqrt( phi1ErrValGlb*phi1ErrValGlb + phi2ErrValGlb*phi2ErrValGlb ) );
		ddxyNormalizedResiduals_global_->Fill( dthetaValGlb/sqrt( theta1ErrValGlb*theta1ErrValGlb + theta2ErrValGlb*theta2ErrValGlb ) );
		ddxyNormalizedResiduals_global_->Fill( dptValGlb/sqrt( pt1ErrValGlb*pt1ErrValGlb + pt2ErrValGlb*pt2ErrValGlb ) );
		ddxyNormalizedResiduals_global_->Fill( dcurvValGlb/sqrt( pow(pt1ErrValGlb,2)/pow(pt1,4) + pow(pt2ErrValGlb,2)/pow(pt2,4) ) );
		
	      }
	      
	      
	    } // end of split muons loop
	  }
	}
      }
    }    
  }
}



DEFINE_FWK_MODULE(TrackSplittingMonitor);
