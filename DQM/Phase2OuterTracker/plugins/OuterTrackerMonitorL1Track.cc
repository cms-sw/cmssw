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
       
       /// Get everything is relevant
       unsigned int nStubs = tempTrackPtr->getStubRefs().size();
       unsigned int seedSector = tempTrackPtr->getSector();
       
       unsigned int seedWedge = tempTrackPtr->getWedge();
       std::cout << "Sector = " << seedSector << " - Wedge = " << seedWedge << std::endl; 
       
       
       double trackPt = tempTrackPtr->getMomentum().perp();
       double trackPhi = tempTrackPtr->getMomentum().phi();
       double trackEta = tempTrackPtr->getMomentum().eta();
       //double trackTheta = tempTrackPtr->getMomentum().theta();
       double trackVtxZ0 = tempTrackPtr->getPOCA().z();
       double trackChi2 = tempTrackPtr->getChi2();
       double trackChi2R = tempTrackPtr->getChi2Red();

       
       L1Track_N_PhiSector->Fill(seedSector,nStubs); 
       L1Track_N_EtaWedge->Fill(seedWedge,nStubs);
       L1Track_PhiSector_Track_Phi->Fill( trackPhi, seedSector );
       L1Track_EtaWedge_Track_Eta->Fill( trackEta, seedWedge );
       
       
       if ( nStubs > 2 )
       {
           num3Stubs++;
	   
	   L1Track_Track_3Stubs_Pt->Fill( trackPt );
	   L1Track_Track_3Stubs_Eta->Fill( trackEta );
	   L1Track_Track_3Stubs_Phi->Fill( trackPhi );
	   
	   
	   L1Track_Track_3Stubs_VtxZ0->Fill( trackVtxZ0 );
	   L1Track_Track_3Stubs_Chi2->Fill( trackChi2 );
	   L1Track_Track_3Stubs_Chi2R->Fill( trackChi2R );
	   
	   L1Track_Track_3Stubs_Chi2_N->Fill( nStubs, trackChi2 );
	   L1Track_Track_3Stubs_Chi2R_N->Fill( nStubs, trackChi2R );

       }
       else
       {
           num2Stubs++;
	   
	   L1Track_Track_2Stubs_Pt->Fill( trackPt );
	   L1Track_Track_2Stubs_Eta->Fill( trackEta );
	   L1Track_Track_2Stubs_Phi->Fill( trackPhi );
	   
	   
	   L1Track_Track_2Stubs_VtxZ0->Fill( trackVtxZ0 );
	   L1Track_Track_2Stubs_Chi2->Fill( trackChi2 );
	   L1Track_Track_2Stubs_Chi2R->Fill( trackChi2R );
	   
	   L1Track_Track_2Stubs_Chi2_N->Fill( nStubs, trackChi2 );
	   L1Track_Track_2Stubs_Chi2R_N->Fill( nStubs, trackChi2R );
       
       }

     }

 } // end TTTracks from pixeldigis 
 
 L1Track_Track_3Stubs_N->Fill( num3Stubs );
 L1Track_Track_2Stubs_N->Fill( num2Stubs );

} // end of method


// ------------ method called once each job just before starting event loop  ------------
void
OuterTrackerMonitorL1Track::beginRun(const edm::Run& run, const edm::EventSetup& es)
{
  SiStripFolderOrganizer folder_organizer;
  folder_organizer.setSiStripFolderName(topFolderName_);
  
  //Phisector vs nb of stubs
  edm::ParameterSet psL1Track_N_PhiSector =  conf_.getParameter<edm::ParameterSet>("TH2L1Track_N_PhiSectorOrEtaWedge");
  std::string HistoName = "L1Track_N_PhiSector";
  L1Track_N_PhiSector = dqmStore_->book2D(HistoName, HistoName,
  psL1Track_N_PhiSector.getParameter<int32_t>("Nbinsx"),
  psL1Track_N_PhiSector.getParameter<double>("xmin"),
  psL1Track_N_PhiSector.getParameter<double>("xmax"),
  psL1Track_N_PhiSector.getParameter<int32_t>("Nbinsy"),
  psL1Track_N_PhiSector.getParameter<double>("ymin"),
  psL1Track_N_PhiSector.getParameter<double>("ymax"));
  L1Track_N_PhiSector->setAxisTitle("#phi sector", 1);
  L1Track_N_PhiSector->setAxisTitle("#stubs", 2);
  
  
  //EtaWedge vs nb of stubs
  edm::ParameterSet psL1Track_N_EtaWedge =  conf_.getParameter<edm::ParameterSet>("TH2L1Track_N_PhiSectorOrEtaWedge");
  HistoName = "L1Track_N_EtaWedge";
  L1Track_N_EtaWedge = dqmStore_->book2D(HistoName, HistoName,
  psL1Track_N_EtaWedge.getParameter<int32_t>("Nbinsx"),
  psL1Track_N_EtaWedge.getParameter<double>("xmin"),
  psL1Track_N_EtaWedge.getParameter<double>("xmax"),
  psL1Track_N_EtaWedge.getParameter<int32_t>("Nbinsy"),
  psL1Track_N_EtaWedge.getParameter<double>("ymin"),
  psL1Track_N_EtaWedge.getParameter<double>("ymax"));
  L1Track_N_EtaWedge->setAxisTitle("#eta wedge", 1);
  L1Track_N_EtaWedge->setAxisTitle("#stubs", 2);
  
  
  //Phisector vs nb of stubs
  edm::ParameterSet psL1Track_Phi =  conf_.getParameter<edm::ParameterSet>("TH2L1Track_PhiOrEta");
  HistoName = "L1Track_PhiSector_Track_Phi";
  L1Track_PhiSector_Track_Phi = dqmStore_->book2D(HistoName, HistoName,
  psL1Track_Phi.getParameter<int32_t>("Nbinsx"),
  psL1Track_Phi.getParameter<double>("xmin"),
  psL1Track_Phi.getParameter<double>("xmax"),
  psL1Track_Phi.getParameter<int32_t>("Nbinsy"),
  psL1Track_Phi.getParameter<double>("ymin"),
  psL1Track_Phi.getParameter<double>("ymax"));
  L1Track_PhiSector_Track_Phi->setAxisTitle("#phi sector", 2);
  L1Track_PhiSector_Track_Phi->setAxisTitle("#phi of the track", 1);
  
  
  //EtaWedge vs nb of stubs
  edm::ParameterSet psL1Track_Eta =  conf_.getParameter<edm::ParameterSet>("TH2L1Track_PhiOrEta");
  HistoName = "L1Track_EtaWedge_Track_Eta";
  L1Track_EtaWedge_Track_Eta = dqmStore_->book2D(HistoName, HistoName,
  psL1Track_Eta.getParameter<int32_t>("Nbinsx"),
  psL1Track_Eta.getParameter<double>("xmin"),
  psL1Track_Eta.getParameter<double>("xmax"),
  psL1Track_Eta.getParameter<int32_t>("Nbinsy"),
  psL1Track_Eta.getParameter<double>("ymin"),
  psL1Track_Eta.getParameter<double>("ymax"));
  L1Track_EtaWedge_Track_Eta->setAxisTitle("#eta wedge", 2);
  L1Track_EtaWedge_Track_Eta->setAxisTitle("#eta of the track", 1);
  
  folder_organizer.setSiStripFolder();	
	
  dqmStore_->setCurrentFolder(topFolderName_+"/L1Tracks/");
  dqmStore_->setCurrentFolder(topFolderName_+"/L1Tracks/2Stubs");	
  // Nb of L1Tracks
  edm::ParameterSet psL1Track_2Stubs_N =  conf_.getParameter<edm::ParameterSet>("TH1L1Track_N");
  HistoName = "L1Track_Track_2Stubs_N";
  L1Track_Track_2Stubs_N = dqmStore_->book1D(HistoName, HistoName,
  psL1Track_2Stubs_N.getParameter<int32_t>("Nbinsx"),
  psL1Track_2Stubs_N.getParameter<double>("xmin"),
  psL1Track_2Stubs_N.getParameter<double>("xmax"));
  L1Track_Track_2Stubs_N->setAxisTitle("# L1Tracks from at most 2 stubs", 1);
  L1Track_Track_2Stubs_N->setAxisTitle("# events", 2);
  
  //Pt of the tracks
  edm::ParameterSet psL1Track_2Stubs_Pt =  conf_.getParameter<edm::ParameterSet>("TH1L1Track_Pt");
  HistoName = "L1Track_Track_2Stubs_Pt";
  L1Track_Track_2Stubs_Pt = dqmStore_->book1D(HistoName, HistoName,
  psL1Track_2Stubs_Pt.getParameter<int32_t>("Nbinsx"),
  psL1Track_2Stubs_Pt.getParameter<double>("xmin"),
  psL1Track_2Stubs_N.getParameter<double>("xmax"));
  L1Track_Track_2Stubs_Pt->setAxisTitle("p_T of L1Tracks from at most 2 stubs", 1);
  L1Track_Track_2Stubs_Pt->setAxisTitle("# events", 2);
  
  //Phi
  edm::ParameterSet psL1Track_2Stubs_Phi =  conf_.getParameter<edm::ParameterSet>("TH1L1Track_Phi");
  HistoName = "L1Track_Track_2Stubs_Phi";
  L1Track_Track_2Stubs_Phi = dqmStore_->book1D(HistoName, HistoName,
  psL1Track_2Stubs_Phi.getParameter<int32_t>("Nbinsx"),
  psL1Track_2Stubs_Phi.getParameter<double>("xmin"),
  psL1Track_2Stubs_N.getParameter<double>("xmax"));
  L1Track_Track_2Stubs_Phi->setAxisTitle("#phi of L1Tracks from at most 2 stubs", 1);
  L1Track_Track_2Stubs_Phi->setAxisTitle("# events", 2);
  
  
  //Eta
  edm::ParameterSet psL1Track_2Stubs_Eta =  conf_.getParameter<edm::ParameterSet>("TH1L1Track_Eta");
  HistoName = "L1Track_Track_2Stubs_Eta";
  L1Track_Track_2Stubs_Eta = dqmStore_->book1D(HistoName, HistoName,
  psL1Track_2Stubs_Eta.getParameter<int32_t>("Nbinsx"),
  psL1Track_2Stubs_Eta.getParameter<double>("xmin"),
  psL1Track_2Stubs_N.getParameter<double>("xmax"));
  L1Track_Track_2Stubs_Eta->setAxisTitle("#eta of L1Tracks from at most 2 stubs", 1);
  L1Track_Track_2Stubs_Eta->setAxisTitle("# events", 2);
  
  
  //VtxZ0
   edm::ParameterSet psL1Track_2Stubs_VtxZ0 =  conf_.getParameter<edm::ParameterSet>("TH1L1Track_VtxZ0");
  HistoName = "L1Track_Track_2Stubs_VtxZ0";
  L1Track_Track_2Stubs_VtxZ0 = dqmStore_->book1D(HistoName, HistoName,
  psL1Track_2Stubs_VtxZ0.getParameter<int32_t>("Nbinsx"),
  psL1Track_2Stubs_VtxZ0.getParameter<double>("xmin"),
  psL1Track_2Stubs_N.getParameter<double>("xmax"));
  L1Track_Track_2Stubs_VtxZ0->setAxisTitle("VtxZ0 of L1Tracks from at most 2 stubs", 1);
  L1Track_Track_2Stubs_VtxZ0->setAxisTitle("# events", 2);
  
  
  //chi2
   edm::ParameterSet psL1Track_2Stubs_Chi2 =  conf_.getParameter<edm::ParameterSet>("TH1L1Track_Chi2");
  HistoName = "L1Track_Track_2Stubs_Chi2";
  L1Track_Track_2Stubs_Chi2 = dqmStore_->book1D(HistoName, HistoName,
  psL1Track_2Stubs_Chi2.getParameter<int32_t>("Nbinsx"),
  psL1Track_2Stubs_Chi2.getParameter<double>("xmin"),
  psL1Track_2Stubs_N.getParameter<double>("xmax"));
  L1Track_Track_2Stubs_Chi2->setAxisTitle("#chi^2 of L1Tracks from at most 2 stubs", 1);
  L1Track_Track_2Stubs_Chi2->setAxisTitle("# events", 2);
  
  //chi2Red
  edm::ParameterSet psL1Track_2Stubs_Chi2R =  conf_.getParameter<edm::ParameterSet>("TH1L1Track_Chi2R");
  HistoName = "L1Track_Track_2Stubs_Chi2R";
  L1Track_Track_2Stubs_Chi2R = dqmStore_->book1D(HistoName, HistoName,
  psL1Track_2Stubs_Chi2R.getParameter<int32_t>("Nbinsx"),
  psL1Track_2Stubs_Chi2R.getParameter<double>("xmin"),
  psL1Track_2Stubs_Chi2R.getParameter<double>("xmax"));
  L1Track_Track_2Stubs_Chi2R->setAxisTitle("#chi^2/dof of L1Tracks from at most 2 stubs", 1);
  L1Track_Track_2Stubs_Chi2R->setAxisTitle("# events", 2);
  
  edm::ParameterSet psL1Track_2Stubs_Chi2_N =  conf_.getParameter<edm::ParameterSet>("TH2L1Track_Chi2_N");
  HistoName = "L1Track_Track_2Stubs_Chi2_N";
  L1Track_Track_2Stubs_Chi2_N = dqmStore_->book2D(HistoName, HistoName,
  psL1Track_2Stubs_Chi2_N.getParameter<int32_t>("Nbinsx"),
  psL1Track_2Stubs_Chi2_N.getParameter<double>("xmin"),
  psL1Track_2Stubs_N.getParameter<double>("xmax"),
  psL1Track_2Stubs_Chi2_N.getParameter<int32_t>("Nbinsy"),
  psL1Track_2Stubs_Chi2_N.getParameter<double>("ymin"),
  psL1Track_2Stubs_Chi2_N.getParameter<double>("ymax"));
  L1Track_Track_2Stubs_Chi2_N->setAxisTitle("#chi^2 of L1Tracks from at most 2 stubs", 2);
  L1Track_Track_2Stubs_Chi2_N->setAxisTitle("#stubs", 1);
  
  edm::ParameterSet psL1Track_2Stubs_Chi2R_N =  conf_.getParameter<edm::ParameterSet>("TH2L1Track_Chi2R_N");
  HistoName = "L1Track_Track_2Stubs_Chi2R_N";
  L1Track_Track_2Stubs_Chi2R_N = dqmStore_->book2D(HistoName, HistoName,
  psL1Track_2Stubs_Chi2R_N.getParameter<int32_t>("Nbinsx"),
  psL1Track_2Stubs_Chi2R_N.getParameter<double>("xmin"),
  psL1Track_2Stubs_N.getParameter<double>("xmax"),
  psL1Track_2Stubs_Chi2R_N.getParameter<int32_t>("Nbinsy"),
  psL1Track_2Stubs_Chi2R_N.getParameter<double>("ymin"),
  psL1Track_2Stubs_Chi2R_N.getParameter<double>("ymax"));
  L1Track_Track_2Stubs_Chi2R_N->setAxisTitle("#chi^2/dof of L1Tracks from at most 2 stubs", 2);
  L1Track_Track_2Stubs_Chi2R_N->setAxisTitle("#stubs", 1);
  
  
  dqmStore_->setCurrentFolder(topFolderName_+"/L1Tracks/3Stubs");
  // Nb of L1Tracks
  edm::ParameterSet psL1Track_3Stubs_N =  conf_.getParameter<edm::ParameterSet>("TH1L1Track_N");
  HistoName = "L1Track_Track_3Stubs_N";
  L1Track_Track_3Stubs_N = dqmStore_->book1D(HistoName, HistoName,
  psL1Track_3Stubs_N.getParameter<int32_t>("Nbinsx"),
  psL1Track_3Stubs_N.getParameter<double>("xmin"),
  psL1Track_3Stubs_N.getParameter<double>("xmax"));
  L1Track_Track_3Stubs_N->setAxisTitle("# L1Tracks from at least 3 stubs", 1);
  L1Track_Track_3Stubs_N->setAxisTitle("# events", 2);
  
  //Pt of the tracks
  edm::ParameterSet psL1Track_3Stubs_Pt =  conf_.getParameter<edm::ParameterSet>("TH1L1Track_Pt");
  HistoName = "L1Track_Track_3Stubs_Pt";
  L1Track_Track_3Stubs_Pt = dqmStore_->book1D(HistoName, HistoName,
  psL1Track_3Stubs_Pt.getParameter<int32_t>("Nbinsx"),
  psL1Track_3Stubs_Pt.getParameter<double>("xmin"),
  psL1Track_3Stubs_N.getParameter<double>("xmax"));
  L1Track_Track_3Stubs_Pt->setAxisTitle("p_T of L1Tracks from at least 3 stubs", 1);
  L1Track_Track_3Stubs_Pt->setAxisTitle("# events", 2);
  
  
  //Phi of the tracks
  edm::ParameterSet psL1Track_3Stubs_Phi =  conf_.getParameter<edm::ParameterSet>("TH1L1Track_Phi");
  HistoName = "L1Track_Track_3Stubs_Phi";
  L1Track_Track_3Stubs_Phi = dqmStore_->book1D(HistoName, HistoName,
  psL1Track_3Stubs_Phi.getParameter<int32_t>("Nbinsx"),
  psL1Track_3Stubs_Phi.getParameter<double>("xmin"),
  psL1Track_3Stubs_N.getParameter<double>("xmax"));
  L1Track_Track_3Stubs_Phi->setAxisTitle("#phi of L1Tracks from at least 3 stubs", 1);
  L1Track_Track_3Stubs_Phi->setAxisTitle("# events", 2);
  
  
  
  //Eta of the tracks
  edm::ParameterSet psL1Track_3Stubs_Eta =  conf_.getParameter<edm::ParameterSet>("TH1L1Track_Eta");
  HistoName = "L1Track_Track_3Stubs_Eta";
  L1Track_Track_3Stubs_Eta = dqmStore_->book1D(HistoName, HistoName,
  psL1Track_3Stubs_Eta.getParameter<int32_t>("Nbinsx"),
  psL1Track_3Stubs_Eta.getParameter<double>("xmin"),
  psL1Track_3Stubs_N.getParameter<double>("xmax"));
  L1Track_Track_3Stubs_Eta->setAxisTitle("#eta of L1Tracks from at least 3 stubs", 1);
  L1Track_Track_3Stubs_Eta->setAxisTitle("# events", 2);
  
  
  
  //VtxZ0 of the tracks
  edm::ParameterSet psL1Track_3Stubs_VtxZ0 =  conf_.getParameter<edm::ParameterSet>("TH1L1Track_VtxZ0");
  HistoName = "L1Track_Track_3Stubs_VtxZ0";
  L1Track_Track_3Stubs_VtxZ0 = dqmStore_->book1D(HistoName, HistoName,
  psL1Track_3Stubs_VtxZ0.getParameter<int32_t>("Nbinsx"),
  psL1Track_3Stubs_VtxZ0.getParameter<double>("xmin"),
  psL1Track_3Stubs_N.getParameter<double>("xmax"));
  L1Track_Track_3Stubs_VtxZ0->setAxisTitle("VtxZ0 of L1Tracks from at least 3 stubs", 1);
  L1Track_Track_3Stubs_VtxZ0->setAxisTitle("# events", 2);
  
  //Chi2 of the tracks
  edm::ParameterSet psL1Track_3Stubs_Chi2 =  conf_.getParameter<edm::ParameterSet>("TH1L1Track_Chi2");
  HistoName = "L1Track_Track_3Stubs_Chi2";
  L1Track_Track_3Stubs_Chi2 = dqmStore_->book1D(HistoName, HistoName,
  psL1Track_3Stubs_Chi2.getParameter<int32_t>("Nbinsx"),
  psL1Track_3Stubs_Chi2.getParameter<double>("xmin"),
  psL1Track_3Stubs_N.getParameter<double>("xmax"));
  L1Track_Track_3Stubs_Chi2->setAxisTitle("#chi^2 of L1Tracks from at least 3 stubs", 1);
  L1Track_Track_3Stubs_Chi2->setAxisTitle("# events", 2);
  
 
  //Chi2R of the tracks
  edm::ParameterSet psL1Track_3Stubs_Chi2R =  conf_.getParameter<edm::ParameterSet>("TH1L1Track_Chi2R");
  HistoName = "L1Track_Track_3Stubs_Chi2R";
  L1Track_Track_3Stubs_Chi2R = dqmStore_->book1D(HistoName, HistoName,
  psL1Track_3Stubs_Chi2R.getParameter<int32_t>("Nbinsx"),
  psL1Track_3Stubs_Chi2R.getParameter<double>("xmin"),
  psL1Track_3Stubs_N.getParameter<double>("xmax"));
  L1Track_Track_3Stubs_Chi2R->setAxisTitle("#chi^2/dof of L1Tracks from at least 3 stubs", 1);
  L1Track_Track_3Stubs_Chi2R->setAxisTitle("# events", 2);
  
  //Chi2 of the tracks vs nb of stubs
  edm::ParameterSet psL1Track_3Stubs_Chi2_N =  conf_.getParameter<edm::ParameterSet>("TH2L1Track_Chi2_N");
  HistoName = "L1Track_Track_3Stubs_Chi2_N";
  L1Track_Track_3Stubs_Chi2_N = dqmStore_->book2D(HistoName, HistoName,
  psL1Track_3Stubs_Chi2_N.getParameter<int32_t>("Nbinsx"),
  psL1Track_3Stubs_Chi2_N.getParameter<double>("xmin"),
  psL1Track_3Stubs_N.getParameter<double>("xmax"),
  psL1Track_3Stubs_Chi2_N.getParameter<int32_t>("Nbinsy"),
  psL1Track_3Stubs_Chi2_N.getParameter<double>("ymin"),
  psL1Track_3Stubs_Chi2_N.getParameter<double>("ymax"));
  L1Track_Track_3Stubs_Chi2_N->setAxisTitle("#chi^2 of L1Tracks from at least 3 stubs", 2);
  L1Track_Track_3Stubs_Chi2_N->setAxisTitle("#stubs", 1);
  
  
  //chi2R of the tracks vs nb of stubs
  edm::ParameterSet psL1Track_3Stubs_Chi2R_N =  conf_.getParameter<edm::ParameterSet>("TH2L1Track_Chi2R_N");
  HistoName = "L1Track_Track_3Stubs_Chi2R_N";
  L1Track_Track_3Stubs_Chi2R_N = dqmStore_->book2D(HistoName, HistoName,
  psL1Track_3Stubs_Chi2R_N.getParameter<int32_t>("Nbinsx"),
  psL1Track_3Stubs_Chi2R_N.getParameter<double>("xmin"),
  psL1Track_3Stubs_N.getParameter<double>("xmax"),
  psL1Track_3Stubs_Chi2R_N.getParameter<int32_t>("Nbinsy"),
  psL1Track_3Stubs_Chi2R_N.getParameter<double>("ymin"),
  psL1Track_3Stubs_Chi2R_N.getParameter<double>("ymax"));
  L1Track_Track_3Stubs_Chi2R_N->setAxisTitle("#chi^2/dof of L1Tracks from at least 3 stubs", 2);
  L1Track_Track_3Stubs_Chi2R_N->setAxisTitle("#stubs", 1);
  
  
  
  
  
  
	
                                  
}//end of method

// ------------ method called once each job just after ending the event loop  ------------
void 
OuterTrackerMonitorL1Track::endJob(void) 
{
	
}

DEFINE_FWK_MODULE(OuterTrackerMonitorL1Track);
