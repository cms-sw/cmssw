// -*- C++ -*-
//
// Package:    Phase2OuterTracker
// Class:      Phase2OuterTracker
//
/**\class Phase2OuterTracker OuterTrackerMonitorL1Track.cc DQM/Phase2OuterTracker/plugins/OuterTrackerMonitorL1Track.cc
 
 Description: [one line class summary]
 
 Implementation:
 [Notes on implementation]
 */
//
// Original Author:  Isis Marina Van Parijs
//         Created:  Mon, 16 Feb 2014 15:49:32 GMT
//

// system include files
#include <memory>
#include <vector>
#include <numeric>
#include <iostream>
#include <fstream>
#include <math.h>
#include "TMath.h"
#include "TNamed.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/Common/interface/DetSetVectorNew.h"
#include "DQM/SiStripCommon/interface/SiStripFolderOrganizer.h"
#include "DQM/Phase2OuterTracker/interface/OuterTrackerMonitorL1Track.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "DataFormats/L1TrackTrigger/interface/TTTypes.h"
#include "DataFormats/L1TrackTrigger/interface/TTTrack.h"
#include "DataFormats/L1TrackTrigger/interface/TTStub.h"
#include "DataFormats/L1TrackTrigger/interface/TTCluster.h"
#include "Geometry/TrackerGeometryBuilder/interface/PixelGeomDetUnit.h"
#include "Geometry/TrackerGeometryBuilder/interface/StackedTrackerGeometry.h"
#include "Geometry/Records/interface/StackedTrackerGeometryRecord.h"
#include "Geometry/CommonTopologies/interface/PixelTopology.h"



//
// constructors and destructor
//
OuterTrackerMonitorL1Track::OuterTrackerMonitorL1Track(const edm::ParameterSet& iConfig)
: dqmStore_(edm::Service<DQMStore>().operator->()), conf_(iConfig)
{
  topFolderName_ = conf_.getParameter<std::string>("TopFolderName");
}

OuterTrackerMonitorL1Track::~OuterTrackerMonitorL1Track()
{
  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)
}

//
// member functions
//

// ------------ method called for each event  ------------
void OuterTrackerMonitorL1Track::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  
  
  /// Track Trigger Tracks
  edm::Handle< std::vector< TTTrack< Ref_PixelDigi_ > > > PixelDigiTTTrackHandle;
  iEvent.getByLabel( "TTTracksFromPixelDigis", "Level1TTTracks",PixelDigiTTTrackHandle );
   

 
 unsigned int num3Stubs = 0;
 unsigned int num2Stubs = 0;
 unsigned int nTracks = 0; 
 
 // Go on only if there are TTTracks from PixelDigis
 if ( PixelDigiTTTrackHandle->size() > 0 )
 {
   /// Loop over TTTracks
    unsigned int tkCnt = 0;
    std::vector< TTTrack< Ref_PixelDigi_ > >::const_iterator iterTTTrack;
    
    for ( iterTTTrack = PixelDigiTTTrackHandle->begin();iterTTTrack != PixelDigiTTTrackHandle->end();++iterTTTrack )
    {
       /// Make the pointer
       edm::Ptr< TTTrack< Ref_PixelDigi_ > > tempTrackPtr( PixelDigiTTTrackHandle, tkCnt++ );
       nTracks++;
       
       /// Get everything is relevant
       unsigned int nStubs = tempTrackPtr->getStubRefs().size();
       unsigned int seedSector = tempTrackPtr->getSector();
       
       unsigned int seedWedge = tempTrackPtr->getWedge();
       //std::cout << "Sector = " << seedSector << " - Wedge = " << seedWedge << std::endl; 
       
       
       double trackPt = tempTrackPtr->getMomentum().perp();
       double trackPhi = tempTrackPtr->getMomentum().phi();
       double trackEta = tempTrackPtr->getMomentum().eta();
       //double trackTheta = tempTrackPtr->getMomentum().theta();
       double trackVtxZ0 = tempTrackPtr->getPOCA().z();
       double trackChi2 = tempTrackPtr->getChi2();
       double trackChi2R = tempTrackPtr->getChi2Red();

       
       L1Track_NStubs_PhiSector->Fill(seedSector,nStubs); 
       L1Track_NStubs_EtaWedge->Fill(seedWedge,nStubs);
       L1Track_PhiSector_L1Track_Phi->Fill( trackPhi, seedSector );
       L1Track_EtaWedge_L1Track_Eta->Fill( trackEta, seedWedge );
       L1Track_NStubs->Fill(nStubs); 
       
       
       if ( nStubs > 2 )
       {
           num3Stubs++;
	   
	   L1Track_3Stubs_Pt->Fill( trackPt );
	   L1Track_3Stubs_Eta->Fill( trackEta );
	   L1Track_3Stubs_Phi->Fill( trackPhi );
	   
	   
	   L1Track_3Stubs_VtxZ0->Fill( trackVtxZ0 );
	   L1Track_3Stubs_Chi2->Fill( trackChi2 );
	   L1Track_3Stubs_Chi2R->Fill( trackChi2R );
	   
	   L1Track_3Stubs_Chi2_NStubs->Fill( nStubs, trackChi2 );
	   L1Track_3Stubs_Chi2R_NStubs->Fill( nStubs, trackChi2R );

       }
       else
       {
           num2Stubs++;
	   
	   L1Track_2Stubs_Pt->Fill( trackPt );
	   L1Track_2Stubs_Eta->Fill( trackEta );
	   L1Track_2Stubs_Phi->Fill( trackPhi );
	   
	   
	   L1Track_2Stubs_VtxZ0->Fill( trackVtxZ0 );
	   L1Track_2Stubs_Chi2->Fill( trackChi2 );
	   L1Track_2Stubs_Chi2R->Fill( trackChi2R );
	   
	   L1Track_2Stubs_Chi2_NStubs->Fill( nStubs, trackChi2 );
	   L1Track_2Stubs_Chi2R_NStubs->Fill( nStubs, trackChi2R );
       
       }

     }

 } // end TTTracks from pixeldigis 
 
 L1Track_NTracks->Fill(nTracks); 
 L1Track_3Stubs_NTracks->Fill( num3Stubs );
 L1Track_2Stubs_NTracks->Fill( num2Stubs );

} // end of method


// ------------ method called once each job just before starting event loop  ------------
void
OuterTrackerMonitorL1Track::beginRun(const edm::Run& run, const edm::EventSetup& es)
{
  SiStripFolderOrganizer folder_organizer;
  folder_organizer.setSiStripFolderName(topFolderName_);
  
  
  
  folder_organizer.setSiStripFolder();	
	
  dqmStore_->setCurrentFolder(topFolderName_+"/L1Tracks/");
  
  //Phisector vs nb of stubs
  edm::ParameterSet psL1Track_NStubs_PhiSectorOrEtaWedge =  conf_.getParameter<edm::ParameterSet>("TH2_NStubs_PhiSectorOrEtaWedge");
  std::string HistoName = "L1Track_NStubs_PhiSector";
  L1Track_NStubs_PhiSector = dqmStore_->book2D(HistoName, HistoName,
  psL1Track_NStubs_PhiSectorOrEtaWedge.getParameter<int32_t>("Nbinsx"),
  psL1Track_NStubs_PhiSectorOrEtaWedge.getParameter<double>("xmin"),
  psL1Track_NStubs_PhiSectorOrEtaWedge.getParameter<double>("xmax"),
  psL1Track_NStubs_PhiSectorOrEtaWedge.getParameter<int32_t>("Nbinsy"),
  psL1Track_NStubs_PhiSectorOrEtaWedge.getParameter<double>("ymin"),
  psL1Track_NStubs_PhiSectorOrEtaWedge.getParameter<double>("ymax"));
  L1Track_NStubs_PhiSector->setAxisTitle("#phi sector of the L1 track", 1);
  L1Track_NStubs_PhiSector->setAxisTitle("# TTStubs", 2);
  
  
  //EtaWedge vs nb of stubs
  HistoName = "L1Track_NStubs_EtaWedge";
  L1Track_NStubs_EtaWedge = dqmStore_->book2D(HistoName, HistoName,
  psL1Track_NStubs_PhiSectorOrEtaWedge.getParameter<int32_t>("Nbinsx"),
  psL1Track_NStubs_PhiSectorOrEtaWedge.getParameter<double>("xmin"),
  psL1Track_NStubs_PhiSectorOrEtaWedge.getParameter<double>("xmax"),
  psL1Track_NStubs_PhiSectorOrEtaWedge.getParameter<int32_t>("Nbinsy"),
  psL1Track_NStubs_PhiSectorOrEtaWedge.getParameter<double>("ymin"),
  psL1Track_NStubs_PhiSectorOrEtaWedge.getParameter<double>("ymax"));
  L1Track_NStubs_EtaWedge->setAxisTitle("#eta wedge of the L1 track", 1);
  L1Track_NStubs_EtaWedge->setAxisTitle("# TTStubs", 2);;
  
  
  
  edm::ParameterSet psPhiSectorOrEtaWedge_PhiOrEta =  conf_.getParameter<edm::ParameterSet>("TH2_PhiSectorOrEtaWedge_PhiOrEta");
  HistoName = "L1Track_PhiSector_L1Track_Phi";
  L1Track_PhiSector_L1Track_Phi = dqmStore_->book2D(HistoName, HistoName,
  psPhiSectorOrEtaWedge_PhiOrEta.getParameter<int32_t>("Nbinsx"),
  psPhiSectorOrEtaWedge_PhiOrEta.getParameter<double>("xmin"),
  psPhiSectorOrEtaWedge_PhiOrEta.getParameter<double>("xmax"),
  psPhiSectorOrEtaWedge_PhiOrEta.getParameter<int32_t>("Nbinsy"),
  psPhiSectorOrEtaWedge_PhiOrEta.getParameter<double>("ymin"),
  psPhiSectorOrEtaWedge_PhiOrEta.getParameter<double>("ymax"));
  L1Track_PhiSector_L1Track_Phi->setAxisTitle("#phi sector of the L1 track", 2);
  L1Track_PhiSector_L1Track_Phi->setAxisTitle("#phi of the L1 track", 1);
  
  
  HistoName = "L1Track_EtaWedge_L1Track_Eta";
  L1Track_EtaWedge_L1Track_Eta = dqmStore_->book2D(HistoName, HistoName,
  psPhiSectorOrEtaWedge_PhiOrEta.getParameter<int32_t>("Nbinsx"),
  psPhiSectorOrEtaWedge_PhiOrEta.getParameter<double>("xmin"),
  psPhiSectorOrEtaWedge_PhiOrEta.getParameter<double>("xmax"),
  psPhiSectorOrEtaWedge_PhiOrEta.getParameter<int32_t>("Nbinsy"),
  psPhiSectorOrEtaWedge_PhiOrEta.getParameter<double>("ymin"),
  psPhiSectorOrEtaWedge_PhiOrEta.getParameter<double>("ymax"));
  L1Track_EtaWedge_L1Track_Eta->setAxisTitle("#eta wedge of the L1 track", 2);
  L1Track_EtaWedge_L1Track_Eta->setAxisTitle("#eta of the L1 track", 1);
  
  
  //Nb of stubs
  edm::ParameterSet psL1Track_NStubs =  conf_.getParameter<edm::ParameterSet>("TH1_NStubs");
  HistoName = "L1Track_NStubs";
  L1Track_NStubs = dqmStore_->book1D(HistoName, HistoName,
  psL1Track_NStubs.getParameter<int32_t>("Nbinsx"),
  psL1Track_NStubs.getParameter<double>("xmin"),
  psL1Track_NStubs.getParameter<double>("xmax"));
  L1Track_NStubs->setAxisTitle("# Stubs per L1 track", 1);
  L1Track_NStubs->setAxisTitle("# L1 tracks", 2);
  
  
  //Nb of tracks
  edm::ParameterSet psL1Track_NTracks =  conf_.getParameter<edm::ParameterSet>("TH1_NL1Tracks");
  HistoName = "L1Track_NTracks";
  L1Track_NTracks = dqmStore_->book1D(HistoName, HistoName,
  psL1Track_NTracks.getParameter<int32_t>("Nbinsx"),
  psL1Track_NTracks.getParameter<double>("xmin"),
  psL1Track_NTracks.getParameter<double>("xmax"));
  L1Track_NTracks->setAxisTitle("# L1 Tracks", 1);
  L1Track_NTracks->setAxisTitle("# events", 2);
  
  
  //start all 2stubs tracks
  
  dqmStore_->setCurrentFolder(topFolderName_+"/L1Tracks/2Stubs");
  
  	
  // Nb of L1Tracks
  HistoName = "L1Track_2Stubs_NTracks";
  L1Track_2Stubs_NTracks = dqmStore_->book1D(HistoName, HistoName,
  psL1Track_NTracks.getParameter<int32_t>("Nbinsx"),
  psL1Track_NTracks.getParameter<double>("xmin"),
  psL1Track_NTracks.getParameter<double>("xmax"));
  L1Track_2Stubs_NTracks->setAxisTitle("# L1Tracks from at most 2 TTStubs", 1);
  L1Track_2Stubs_NTracks->setAxisTitle("# events", 2);
  
  //Pt of the tracks
  edm::ParameterSet psL1Track_Pt =  conf_.getParameter<edm::ParameterSet>("TH1_L1Track_Pt");
  HistoName = "L1Track_2Stubs_Pt";
  L1Track_2Stubs_Pt = dqmStore_->book1D(HistoName, HistoName,
  psL1Track_Pt.getParameter<int32_t>("Nbinsx"),
  psL1Track_Pt.getParameter<double>("xmin"),
  psL1Track_Pt.getParameter<double>("xmax"));
  L1Track_2Stubs_Pt->setAxisTitle("p_{T} of L1Tracks from at most 2 TTStubs", 1);
  L1Track_2Stubs_Pt->setAxisTitle("# L1Tracks", 2);
  
  //Phi
  edm::ParameterSet psL1Track_Phi =  conf_.getParameter<edm::ParameterSet>("TH1_L1Track_Phi");
  HistoName = "L1Track_2Stubs_Phi";
  L1Track_2Stubs_Phi = dqmStore_->book1D(HistoName, HistoName,
  psL1Track_Phi.getParameter<int32_t>("Nbinsx"),
  psL1Track_Phi.getParameter<double>("xmin"),
  psL1Track_Phi.getParameter<double>("xmax"));
  L1Track_2Stubs_Phi->setAxisTitle("#phi of L1Tracks from at most 2 TTStubs", 1);
  L1Track_2Stubs_Phi->setAxisTitle("# L1Tracks", 2);
  
  
  //Eta
  edm::ParameterSet psL1Track_Eta =  conf_.getParameter<edm::ParameterSet>("TH1_L1Track_Eta");
  HistoName = "L1Track_2Stubs_Eta";
  L1Track_2Stubs_Eta = dqmStore_->book1D(HistoName, HistoName,
  psL1Track_Eta.getParameter<int32_t>("Nbinsx"),
  psL1Track_Eta.getParameter<double>("xmin"),
  psL1Track_Eta.getParameter<double>("xmax"));
  L1Track_2Stubs_Eta->setAxisTitle("#eta of L1Tracks from at most 2 TTStubs", 1);
  L1Track_2Stubs_Eta->setAxisTitle("# L1Tracks", 2);
  
  
  //VtxZ0
   edm::ParameterSet psL1Track_VtxZ0 =  conf_.getParameter<edm::ParameterSet>("TH1_L1Track_VtxZ0");
  HistoName = "L1Track_2Stubs_VtxZ0";
  L1Track_2Stubs_VtxZ0 = dqmStore_->book1D(HistoName, HistoName,
  psL1Track_VtxZ0.getParameter<int32_t>("Nbinsx"),
  psL1Track_VtxZ0.getParameter<double>("xmin"),
  psL1Track_VtxZ0.getParameter<double>("xmax"));
  L1Track_2Stubs_VtxZ0->setAxisTitle("VtxZ0 of L1Tracks from at most 2 TTStubs", 1);
  L1Track_2Stubs_VtxZ0->setAxisTitle("# L1Tracks", 2);
  
  
  //chi2
   edm::ParameterSet psL1Track_Chi2 =  conf_.getParameter<edm::ParameterSet>("TH1_L1Track_Chi2");
  HistoName = "L1Track_2Stubs_Chi2";
  L1Track_2Stubs_Chi2 = dqmStore_->book1D(HistoName, HistoName,
  psL1Track_Chi2.getParameter<int32_t>("Nbinsx"),
  psL1Track_Chi2.getParameter<double>("xmin"),
  psL1Track_Chi2.getParameter<double>("xmax"));
  L1Track_2Stubs_Chi2->setAxisTitle("#chi^{2} of L1Tracks from at most 2 TTStubs", 1);
  L1Track_2Stubs_Chi2->setAxisTitle("# L1Tracks", 2);
  
  //chi2Red
  edm::ParameterSet psL1Track_Chi2R =  conf_.getParameter<edm::ParameterSet>("TH1_L1Track_Chi2R");
  HistoName = "L1Track_2Stubs_Chi2R";
  L1Track_2Stubs_Chi2R = dqmStore_->book1D(HistoName, HistoName,
  psL1Track_Chi2R.getParameter<int32_t>("Nbinsx"),
  psL1Track_Chi2R.getParameter<double>("xmin"),
  psL1Track_Chi2R.getParameter<double>("xmax"));
  L1Track_2Stubs_Chi2R->setAxisTitle("#chi^{2}/dof of L1Tracks from at most 2 TTStubs", 1);
  L1Track_2Stubs_Chi2R->setAxisTitle("# L1Tracks", 2);
  
  edm::ParameterSet psL1Track_Chi2_NStubs =  conf_.getParameter<edm::ParameterSet>("TH2_L1Track_Chi2_NStubs");
  HistoName = "L1Track_2Stubs_Chi2_N";
  L1Track_2Stubs_Chi2_NStubs = dqmStore_->book2D(HistoName, HistoName,
  psL1Track_Chi2_NStubs.getParameter<int32_t>("Nbinsx"),
  psL1Track_Chi2_NStubs.getParameter<double>("xmin"),
  psL1Track_Chi2_NStubs.getParameter<double>("xmax"),
  psL1Track_Chi2_NStubs.getParameter<int32_t>("Nbinsy"),
  psL1Track_Chi2_NStubs.getParameter<double>("ymin"),
  psL1Track_Chi2_NStubs.getParameter<double>("ymax"));
  L1Track_2Stubs_Chi2_NStubs->setAxisTitle("# TTStubs", 1);
  L1Track_2Stubs_Chi2_NStubs->setAxisTitle("#chi^{2} of L1Tracks from at most 2 TTStubs", 2);
  
  edm::ParameterSet psL1Track_Chi2R_NStubs =  conf_.getParameter<edm::ParameterSet>("TH2_L1Track_Chi2R_NStubs");
  HistoName = "L1Track_2Stubs_Chi2R_NStubs";
  L1Track_2Stubs_Chi2R_NStubs = dqmStore_->book2D(HistoName, HistoName,
  psL1Track_Chi2R_NStubs.getParameter<int32_t>("Nbinsx"),
  psL1Track_Chi2R_NStubs.getParameter<double>("xmin"),
  psL1Track_NStubs.getParameter<double>("xmax"),
  psL1Track_Chi2R_NStubs.getParameter<int32_t>("Nbinsy"),
  psL1Track_Chi2R_NStubs.getParameter<double>("ymin"),
  psL1Track_Chi2R_NStubs.getParameter<double>("ymax"));
  L1Track_2Stubs_Chi2R_NStubs->setAxisTitle("# TTStubs", 1);
  L1Track_2Stubs_Chi2R_NStubs->setAxisTitle("#chi^2/dof of L1Tracks from at most 2 TTStubs", 2);
  
  
  
  //all tracks with more than 2 stubs
  dqmStore_->setCurrentFolder(topFolderName_+"/L1Tracks/3Stubs");
  
  // Nb of L1Tracks
  HistoName = "L1Track_3Stubs_NTracks";
  L1Track_3Stubs_NTracks = dqmStore_->book1D(HistoName, HistoName,
  psL1Track_NTracks.getParameter<int32_t>("Nbinsx"),
  psL1Track_NTracks.getParameter<double>("xmin"),
  psL1Track_NTracks.getParameter<double>("xmax"));
  L1Track_3Stubs_NTracks->setAxisTitle("# L1Tracks from at least 3 TTStubs", 1);
  L1Track_3Stubs_NTracks->setAxisTitle("# events", 2);
  
  //Pt of the tracks
  HistoName = "L1Track_3Stubs_Pt";
  L1Track_3Stubs_Pt = dqmStore_->book1D(HistoName, HistoName,
  psL1Track_Pt.getParameter<int32_t>("Nbinsx"),
  psL1Track_Pt.getParameter<double>("xmin"),
  psL1Track_Pt.getParameter<double>("xmax"));
  L1Track_3Stubs_Pt->setAxisTitle("p_{T} of L1Tracks from at least 3 TTStubs", 1);
  L1Track_3Stubs_Pt->setAxisTitle("# L1Tracks", 2);
  
  //Phi
  HistoName = "L1Track_3Stubs_Phi";
  L1Track_3Stubs_Phi = dqmStore_->book1D(HistoName, HistoName,
  psL1Track_Phi.getParameter<int32_t>("Nbinsx"),
  psL1Track_Phi.getParameter<double>("xmin"),
  psL1Track_Phi.getParameter<double>("xmax"));
  L1Track_3Stubs_Phi->setAxisTitle("#phi of L1Tracks from at least 3 TTStubs", 1);
  L1Track_3Stubs_Phi->setAxisTitle("# L1Tracks", 2);
  
  
  //Eta
  HistoName = "L1Track_3Stubs_Eta";
  L1Track_3Stubs_Eta = dqmStore_->book1D(HistoName, HistoName,
  psL1Track_Eta.getParameter<int32_t>("Nbinsx"),
  psL1Track_Eta.getParameter<double>("xmin"),
  psL1Track_Eta.getParameter<double>("xmax"));
  L1Track_3Stubs_Eta->setAxisTitle("#eta of L1Tracks from at least 3 TTStubs", 1);
  L1Track_3Stubs_Eta->setAxisTitle("# L1Tracks", 2);
  
  
  //VtxZ0
  HistoName = "L1Track_3Stubs_VtxZ0";
  L1Track_3Stubs_VtxZ0 = dqmStore_->book1D(HistoName, HistoName,
  psL1Track_VtxZ0.getParameter<int32_t>("Nbinsx"),
  psL1Track_VtxZ0.getParameter<double>("xmin"),
  psL1Track_VtxZ0.getParameter<double>("xmax"));
  L1Track_3Stubs_VtxZ0->setAxisTitle("VtxZ0 of L1Tracks from at least 3 TTStubs", 1);
  L1Track_3Stubs_VtxZ0->setAxisTitle("# L1Tracks", 2);
  
  
  //chi2
  HistoName = "L1Track_3Stubs_Chi2";
  L1Track_3Stubs_Chi2 = dqmStore_->book1D(HistoName, HistoName,
  psL1Track_Chi2.getParameter<int32_t>("Nbinsx"),
  psL1Track_Chi2.getParameter<double>("xmin"),
  psL1Track_Chi2.getParameter<double>("xmax"));
  L1Track_3Stubs_Chi2->setAxisTitle("#chi^{2} of L1Tracks from at least 3 TTStubs", 1);
  L1Track_3Stubs_Chi2->setAxisTitle("# L1Tracks", 2);
  
  //chi2Red
  HistoName = "L1Track_3Stubs_Chi2R";
  L1Track_3Stubs_Chi2R = dqmStore_->book1D(HistoName, HistoName,
  psL1Track_Chi2R.getParameter<int32_t>("Nbinsx"),
  psL1Track_Chi2R.getParameter<double>("xmin"),
  psL1Track_Chi2R.getParameter<double>("xmax"));
  L1Track_3Stubs_Chi2R->setAxisTitle("#chi^{2}/dof of L1Tracks from at least 3 TTStubs", 1);
  L1Track_3Stubs_Chi2R->setAxisTitle("# L1Tracks", 2);
  
  HistoName = "L1Track_3Stubs_Chi2_NStubs";
  L1Track_3Stubs_Chi2_NStubs = dqmStore_->book2D(HistoName, HistoName,
  psL1Track_Chi2_NStubs.getParameter<int32_t>("Nbinsx"),
  psL1Track_Chi2_NStubs.getParameter<double>("xmin"),
  psL1Track_Chi2_NStubs.getParameter<double>("xmax"),
  psL1Track_Chi2_NStubs.getParameter<int32_t>("Nbinsy"),
  psL1Track_Chi2_NStubs.getParameter<double>("ymin"),
  psL1Track_Chi2_NStubs.getParameter<double>("ymax"));
  L1Track_3Stubs_Chi2_NStubs->setAxisTitle("# TTStubs", 1);
  L1Track_3Stubs_Chi2_NStubs->setAxisTitle("#chi^{2} of L1Tracks from at least 3 TTStubs", 2);
  
  HistoName = "L1Track_3Stubs_Chi2R_NStubs";
  L1Track_3Stubs_Chi2R_NStubs = dqmStore_->book2D(HistoName, HistoName,
  psL1Track_Chi2R_NStubs.getParameter<int32_t>("Nbinsx"),
  psL1Track_Chi2R_NStubs.getParameter<double>("xmin"),
  psL1Track_Chi2R_NStubs.getParameter<double>("xmax"),
  psL1Track_Chi2R_NStubs.getParameter<int32_t>("Nbinsy"),
  psL1Track_Chi2R_NStubs.getParameter<double>("ymin"),
  psL1Track_Chi2R_NStubs.getParameter<double>("ymax"));
  L1Track_3Stubs_Chi2R_NStubs->setAxisTitle("# TTStubs", 1);
  L1Track_3Stubs_Chi2R_NStubs->setAxisTitle("#chi^{2}/dof of L1Tracks from at least 3 TTStubs", 2);
  
  
  
  
  
  
	
                                  
}//end of method

// ------------ method called once each job just after ending the event loop  ------------
void 
OuterTrackerMonitorL1Track::endJob(void) 
{
	
}

DEFINE_FWK_MODULE(OuterTrackerMonitorL1Track);
