// -*- C++ -*-
//
// Package:    Phase2OuterTracker
// Class:      Phase2OuterTracker
//
/**\class Phase2OuterTracker OuterTrackerMonitorCluster.cc DQM/Phase2OuterTracker/plugins/OuterTrackerMonitorCluster.cc
 
 Description: [one line class summary]
 
 Implementation:
 [Notes on implementation]
 */
//
// Original Author:  Isabelle Helena J De Bruyn
//         Created:  Mon, 10 Feb 2014 13:57:08 GMT
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
#include "DQM/Phase2OuterTracker/interface/OuterTrackerMonitorCluster.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "DataFormats/L1TrackTrigger/interface/TTTypes.h"
#include "DataFormats/L1TrackTrigger/interface/TTCluster.h"
#include "Geometry/TrackerGeometryBuilder/interface/PixelGeomDetUnit.h"
#include "Geometry/TrackerGeometryBuilder/interface/StackedTrackerGeometry.h"
#include "Geometry/Records/interface/StackedTrackerGeometryRecord.h"
#include "Geometry/CommonTopologies/interface/PixelTopology.h"

//
// constructors and destructor
//
OuterTrackerMonitorCluster::OuterTrackerMonitorCluster(const edm::ParameterSet& iConfig)
: dqmStore_(edm::Service<DQMStore>().operator->()), conf_(iConfig)
{
  topFolderName_ = conf_.getParameter<std::string>("TopFolderName");
  tagTTClusters_ = conf_.getParameter< edm::InputTag >("TTClusters");
}

OuterTrackerMonitorCluster::~OuterTrackerMonitorCluster()
{
  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)
}

//
// member functions
//

// ------------ method called for each event  ------------
void OuterTrackerMonitorCluster::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
	/// Track Trigger Clusters
	edm::Handle< edmNew::DetSetVector< TTCluster< Ref_PixelDigi_ > > > PixelDigiTTClusterHandle;
  iEvent.getByLabel( tagTTClusters_, PixelDigiTTClusterHandle );
  
  /// Geometry
  edm::ESHandle< StackedTrackerGeometry > StackedGeometryHandle;
  const StackedTrackerGeometry* theStackedGeometry;
  iSetup.get< StackedTrackerGeometryRecord >().get(StackedGeometryHandle);
  theStackedGeometry = StackedGeometryHandle.product();
  
  /// Loop over the input Clusters
  typename edmNew::DetSetVector< TTCluster< Ref_PixelDigi_ > >::const_iterator inputIter;
  typename edmNew::DetSet< TTCluster< Ref_PixelDigi_ > >::const_iterator contentIter;
  for ( inputIter = PixelDigiTTClusterHandle->begin();
       inputIter != PixelDigiTTClusterHandle->end();
       ++inputIter )
  {
    for(contentIter = inputIter->begin(); contentIter != inputIter->end(); ++contentIter)
    {
      edm::Ref< edmNew::DetSetVector< TTCluster< Ref_PixelDigi_ > >, TTCluster< Ref_PixelDigi_ > > tempCluRef = edmNew::makeRefTo( PixelDigiTTClusterHandle, contentIter );
      StackedTrackerDetId detIdClu( tempCluRef->getDetId() );
      unsigned int memberClu = tempCluRef->getStackMember();
      unsigned int widClu = tempCluRef->findWidth();
      
      GlobalPoint posClu  = theStackedGeometry->findAverageGlobalPosition( &(*tempCluRef) );
      
      double eta = posClu.eta();
      
      Cluster_W->Fill(widClu, memberClu);
      Cluster_Eta->Fill(eta);
      
      Cluster_RZ->Fill( posClu.z(), posClu.perp() );
      
      if ( detIdClu.isBarrel() )
      {
        
        if (memberClu == 0) Cluster_IMem_Barrel->Fill(detIdClu.iLayer());
        else Cluster_OMem_Barrel->Fill(detIdClu.iLayer());
        
        Cluster_Barrel_XY->Fill( posClu.x(), posClu.y() );
        Cluster_Barrel_XY_Zoom->Fill( posClu.x(), posClu.y() );
        
      }	// end if isBarrel()
      else if (detIdClu.isEndcap())
      {
        
        if (memberClu == 0)
        {
          Cluster_IMem_Endcap_Disc->Fill(detIdClu.iDisk());
          Cluster_IMem_Endcap_Ring->Fill(detIdClu.iRing());
        }
        else
        {
          Cluster_OMem_Endcap_Disc->Fill(detIdClu.iDisk());
          Cluster_OMem_Endcap_Ring->Fill(detIdClu.iRing());
        }
        
        if ( posClu.z() > 0 )
        {
          Cluster_Endcap_Fw_XY->Fill( posClu.x(), posClu.y() );
          Cluster_Endcap_Fw_RZ_Zoom->Fill( posClu.z(), posClu.perp() );
          if (memberClu == 0) Cluster_IMem_Endcap_Ring_Fw[detIdClu.iDisk()-1]->Fill(detIdClu.iRing());
          else Cluster_OMem_Endcap_Ring_Fw[detIdClu.iDisk()-1]->Fill(detIdClu.iRing());
        }
        else
        {
          Cluster_Endcap_Bw_XY->Fill( posClu.x(), posClu.y() );
          Cluster_Endcap_Bw_RZ_Zoom->Fill( posClu.z(), posClu.perp() );
          if (memberClu == 0) Cluster_IMem_Endcap_Ring_Bw[detIdClu.iDisk()-1]->Fill(detIdClu.iRing());
          else Cluster_OMem_Endcap_Ring_Bw[detIdClu.iDisk()-1]->Fill(detIdClu.iRing());
        }
        
      } // end if isEndcap()
    } // end loop contentIter
  } // end loop inputIter
} // end of method

// ------------ method called once each job just before starting event loop  ------------
void
OuterTrackerMonitorCluster::beginRun(const edm::Run& run, const edm::EventSetup& es)
{
  SiStripFolderOrganizer folder_organizer;
  folder_organizer.setSiStripFolderName(topFolderName_);
  folder_organizer.setSiStripFolder();	
  std::string HistoName;
  
  dqmStore_->setCurrentFolder(topFolderName_+"/Clusters/NClusters");
  
  // NClusters
  edm::ParameterSet psTTCluster_Barrel =  conf_.getParameter<edm::ParameterSet>("TH1TTCluster_Barrel");
  HistoName = "NClusters_IMem_Barrel";
  Cluster_IMem_Barrel = dqmStore_->book1D(HistoName, HistoName,
      psTTCluster_Barrel.getParameter<int32_t>("Nbinsx"),
      psTTCluster_Barrel.getParameter<double>("xmin"),
      psTTCluster_Barrel.getParameter<double>("xmax"));
  Cluster_IMem_Barrel->setAxisTitle("Barrel Layer", 1);
  Cluster_IMem_Barrel->setAxisTitle("# L1 Clusters", 2);
  
  HistoName = "NClusters_OMem_Barrel";
  Cluster_OMem_Barrel = dqmStore_->book1D(HistoName, HistoName,
      psTTCluster_Barrel.getParameter<int32_t>("Nbinsx"),
      psTTCluster_Barrel.getParameter<double>("xmin"),
      psTTCluster_Barrel.getParameter<double>("xmax"));
  Cluster_OMem_Barrel->setAxisTitle("Barrel Layer", 1);
  Cluster_OMem_Barrel->setAxisTitle("# L1 Clusters", 2);
  
  edm::ParameterSet psTTCluster_ECDisc =  conf_.getParameter<edm::ParameterSet>("TH1TTCluster_ECDiscs");
  HistoName = "NClusters_IMem_Endcap_Disc";
  Cluster_IMem_Endcap_Disc = dqmStore_->book1D(HistoName, HistoName,
      psTTCluster_ECDisc.getParameter<int32_t>("Nbinsx"),
      psTTCluster_ECDisc.getParameter<double>("xmin"),
      psTTCluster_ECDisc.getParameter<double>("xmax"));
  Cluster_IMem_Endcap_Disc->setAxisTitle("Endcap Disc", 1);
  Cluster_IMem_Endcap_Disc->setAxisTitle("# L1 Clusters", 2);
  
  HistoName = "NClusters_OMem_Endcap_Disc";
  Cluster_OMem_Endcap_Disc = dqmStore_->book1D(HistoName, HistoName,
      psTTCluster_ECDisc.getParameter<int32_t>("Nbinsx"),
      psTTCluster_ECDisc.getParameter<double>("xmin"),
      psTTCluster_ECDisc.getParameter<double>("xmax"));
  Cluster_OMem_Endcap_Disc->setAxisTitle("Endcap Disc", 1);
  Cluster_OMem_Endcap_Disc->setAxisTitle("# L1 Clusters", 2);
  
  edm::ParameterSet psTTCluster_ECRing =  conf_.getParameter<edm::ParameterSet>("TH1TTCluster_ECRings");  
  HistoName = "NClusters_IMem_Endcap_Ring";
  Cluster_IMem_Endcap_Ring = dqmStore_->book1D(HistoName, HistoName,
      psTTCluster_ECRing.getParameter<int32_t>("Nbinsx"),
      psTTCluster_ECRing.getParameter<double>("xmin"),
      psTTCluster_ECRing.getParameter<double>("xmax"));
  Cluster_IMem_Endcap_Ring->setAxisTitle("Endcap Ring", 1);
  Cluster_IMem_Endcap_Ring->setAxisTitle("# L1 Clusters", 2);
  
  HistoName = "NClusters_OMem_Endcap_Ring";
  Cluster_OMem_Endcap_Ring = dqmStore_->book1D(HistoName, HistoName,
      psTTCluster_ECRing.getParameter<int32_t>("Nbinsx"),
      psTTCluster_ECRing.getParameter<double>("xmin"),
      psTTCluster_ECRing.getParameter<double>("xmax"));
  Cluster_OMem_Endcap_Ring->setAxisTitle("Endcap Ring", 1);
  Cluster_OMem_Endcap_Ring->setAxisTitle("# L1 Clusters", 2);
  
  for(int i=0;i<5;i++){
    Char_t histo[200];
    sprintf(histo, "NClusters_IMem_Disc+%d", i+1);  
    Cluster_IMem_Endcap_Ring_Fw[i] = dqmStore_ ->book1D(histo, histo, 
        psTTCluster_ECRing.getParameter<int32_t>("Nbinsx"), 
        psTTCluster_ECRing.getParameter<double>("xmin"), 
        psTTCluster_ECRing.getParameter<double>("xmax")); 
    Cluster_IMem_Endcap_Ring_Fw[i]->setAxisTitle("Endcap Ring",1); 
    Cluster_IMem_Endcap_Ring_Fw[i]->setAxisTitle("# L1 Clusters ",2);
  }
  
  for(int i=0;i<5;i++){
    Char_t histo[200];
    sprintf(histo, "NClusters_IMem_Disc-%d", i+1);  
    Cluster_IMem_Endcap_Ring_Bw[i] = dqmStore_ ->book1D(histo, histo, 
        psTTCluster_ECRing.getParameter<int32_t>("Nbinsx"), 
        psTTCluster_ECRing.getParameter<double>("xmin"), 
        psTTCluster_ECRing.getParameter<double>("xmax")); 
    Cluster_IMem_Endcap_Ring_Bw[i]->setAxisTitle("Endcap Ring",1); 
    Cluster_IMem_Endcap_Ring_Bw[i]->setAxisTitle("# L1 Clusters ",2);
  }
  
  for(int i=0;i<5;i++){
    Char_t histo[200];
    sprintf(histo, "NClusters_OMem_Disc+%d", i+1);  
    Cluster_OMem_Endcap_Ring_Fw[i] = dqmStore_ ->book1D(histo, histo, 
        psTTCluster_ECRing.getParameter<int32_t>("Nbinsx"), 
        psTTCluster_ECRing.getParameter<double>("xmin"), 
        psTTCluster_ECRing.getParameter<double>("xmax")); 
    Cluster_OMem_Endcap_Ring_Fw[i]->setAxisTitle("Endcap Ring",1); 
    Cluster_OMem_Endcap_Ring_Fw[i]->setAxisTitle("# L1 Clusters ",2);
  }
  
  for(int i=0;i<5;i++){
    Char_t histo[200];
    sprintf(histo, "NClusters_OMem_Disc-%d", i+1);  
    Cluster_OMem_Endcap_Ring_Bw[i] = dqmStore_ ->book1D(histo, histo, 
        psTTCluster_ECRing.getParameter<int32_t>("Nbinsx"), 
        psTTCluster_ECRing.getParameter<double>("xmin"), 
        psTTCluster_ECRing.getParameter<double>("xmax")); 
    Cluster_OMem_Endcap_Ring_Bw[i]->setAxisTitle("Endcap Ring",1); 
    Cluster_OMem_Endcap_Ring_Bw[i]->setAxisTitle("# L1 Clusters ",2);
  }
  
  
  dqmStore_->setCurrentFolder(topFolderName_+"/Clusters");
        
  //Cluster Width
  edm::ParameterSet psTTClusterWidth =  conf_.getParameter<edm::ParameterSet>("TH2TTCluster_Width");
  HistoName = "Cluster_W";
  Cluster_W = dqmStore_->book2D(HistoName, HistoName,
      psTTClusterWidth.getParameter<int32_t>("Nbinsx"),
      psTTClusterWidth.getParameter<double>("xmin"),
      psTTClusterWidth.getParameter<double>("xmax"),
      psTTClusterWidth.getParameter<int32_t>("Nbinsy"),
      psTTClusterWidth.getParameter<double>("ymin"),
      psTTClusterWidth.getParameter<double>("ymax"));
  Cluster_W->setAxisTitle("L1 Cluster Width", 1);
  Cluster_W->setAxisTitle("Stack Member", 2);
  
  //Cluster eta distribution
  edm::ParameterSet psTTClusterEta = conf_.getParameter<edm::ParameterSet>("TH1TTCluster_Eta");
  HistoName = "Cluster_Eta";
  Cluster_Eta = dqmStore_->book1D(HistoName, HistoName, 
      psTTClusterEta.getParameter<int32_t>("Nbinsx"),
      psTTClusterEta.getParameter<double>("xmin"),
      psTTClusterEta.getParameter<double>("xmax"));
  Cluster_Eta->setAxisTitle("#eta", 1);
  Cluster_Eta->setAxisTitle("# L1 Clusters", 2);
  
  dqmStore_->setCurrentFolder(topFolderName_+"/Clusters/Position");
  
  //Position plots
  edm::ParameterSet psTTCluster_Barrel_XY =  conf_.getParameter<edm::ParameterSet>("TH2TTCluster_Position");
  HistoName = "Cluster_Barrel_XY";
  Cluster_Barrel_XY = dqmStore_->book2D(HistoName, HistoName,
      psTTCluster_Barrel_XY.getParameter<int32_t>("Nbinsx"),
      psTTCluster_Barrel_XY.getParameter<double>("xmin"),
      psTTCluster_Barrel_XY.getParameter<double>("xmax"),
      psTTCluster_Barrel_XY.getParameter<int32_t>("Nbinsy"),
      psTTCluster_Barrel_XY.getParameter<double>("ymin"),
      psTTCluster_Barrel_XY.getParameter<double>("ymax"));
  Cluster_Barrel_XY->setAxisTitle("L1 Cluster Barrel position x [cm]", 1);
  Cluster_Barrel_XY->setAxisTitle("L1 Cluster Barrel position y [cm]", 2);
  
  edm::ParameterSet psTTCluster_Barrel_XY_Zoom =  conf_.getParameter<edm::ParameterSet>("TH2TTCluster_Barrel_XY_Zoom");
  HistoName = "Cluster_Barrel_XY_Zoom";
  Cluster_Barrel_XY_Zoom = dqmStore_->book2D(HistoName, HistoName,
      psTTCluster_Barrel_XY_Zoom.getParameter<int32_t>("Nbinsx"),
      psTTCluster_Barrel_XY_Zoom.getParameter<double>("xmin"),
      psTTCluster_Barrel_XY_Zoom.getParameter<double>("xmax"),
      psTTCluster_Barrel_XY_Zoom.getParameter<int32_t>("Nbinsy"),
      psTTCluster_Barrel_XY_Zoom.getParameter<double>("ymin"),
      psTTCluster_Barrel_XY_Zoom.getParameter<double>("ymax"));
  Cluster_Barrel_XY_Zoom->setAxisTitle("L1 Cluster Barrel position x [cm]", 1);
  Cluster_Barrel_XY_Zoom->setAxisTitle("L1 Cluster Barrel position y [cm]", 2);
  
  edm::ParameterSet psTTCluster_Endcap_Fw_XY =  conf_.getParameter<edm::ParameterSet>("TH2TTCluster_Position");
  HistoName = "Cluster_Endcap_Fw_XY";
  Cluster_Endcap_Fw_XY = dqmStore_->book2D(HistoName, HistoName,
      psTTCluster_Endcap_Fw_XY.getParameter<int32_t>("Nbinsx"),
      psTTCluster_Endcap_Fw_XY.getParameter<double>("xmin"),
      psTTCluster_Endcap_Fw_XY.getParameter<double>("xmax"),
      psTTCluster_Endcap_Fw_XY.getParameter<int32_t>("Nbinsy"),
      psTTCluster_Endcap_Fw_XY.getParameter<double>("ymin"),
      psTTCluster_Endcap_Fw_XY.getParameter<double>("ymax"));
  Cluster_Endcap_Fw_XY->setAxisTitle("L1 Cluster Forward Endcap position x [cm]", 1);
  Cluster_Endcap_Fw_XY->setAxisTitle("L1 Cluster Forward Endcap position y [cm]", 2);
  
  edm::ParameterSet psTTCluster_Endcap_Bw_XY =  conf_.getParameter<edm::ParameterSet>("TH2TTCluster_Position");
  HistoName = "Cluster_Endcap_Bw_XY";
  Cluster_Endcap_Bw_XY = dqmStore_->book2D(HistoName, HistoName,
      psTTCluster_Endcap_Bw_XY.getParameter<int32_t>("Nbinsx"),
      psTTCluster_Endcap_Bw_XY.getParameter<double>("xmin"),
      psTTCluster_Endcap_Bw_XY.getParameter<double>("xmax"),
      psTTCluster_Endcap_Bw_XY.getParameter<int32_t>("Nbinsy"),
      psTTCluster_Endcap_Bw_XY.getParameter<double>("ymin"),
      psTTCluster_Endcap_Bw_XY.getParameter<double>("ymax"));
  Cluster_Endcap_Bw_XY->setAxisTitle("L1 Cluster Backward Endcap position x [cm]", 1);
  Cluster_Endcap_Bw_XY->setAxisTitle("L1 Cluster Backward Endcap position y [cm]", 2);
  
  //TTCluster #rho vs. z
  edm::ParameterSet psTTCluster_RZ =  conf_.getParameter<edm::ParameterSet>("TH2TTCluster_RZ");
  HistoName = "Cluster_RZ";
  Cluster_RZ = dqmStore_->book2D(HistoName, HistoName,
      psTTCluster_RZ.getParameter<int32_t>("Nbinsx"),
      psTTCluster_RZ.getParameter<double>("xmin"),
      psTTCluster_RZ.getParameter<double>("xmax"),
      psTTCluster_RZ.getParameter<int32_t>("Nbinsy"),
      psTTCluster_RZ.getParameter<double>("ymin"),
      psTTCluster_RZ.getParameter<double>("ymax"));
  Cluster_RZ->setAxisTitle("L1 Cluster position z [cm]", 1);
  Cluster_RZ->setAxisTitle("L1 Cluster position #rho [cm]", 2);
  
  //TTCluster Forward Endcap #rho vs. z
  edm::ParameterSet psTTCluster_Endcap_Fw_RZ_Zoom =  conf_.getParameter<edm::ParameterSet>("TH2TTCluster_Endcap_Fw_RZ_Zoom");
  HistoName = "Cluster_Endcap_Fw_RZ_Zoom";
  Cluster_Endcap_Fw_RZ_Zoom = dqmStore_->book2D(HistoName, HistoName,
      psTTCluster_Endcap_Fw_RZ_Zoom.getParameter<int32_t>("Nbinsx"),
      psTTCluster_Endcap_Fw_RZ_Zoom.getParameter<double>("xmin"),
      psTTCluster_Endcap_Fw_RZ_Zoom.getParameter<double>("xmax"),
      psTTCluster_Endcap_Fw_RZ_Zoom.getParameter<int32_t>("Nbinsy"),
      psTTCluster_Endcap_Fw_RZ_Zoom.getParameter<double>("ymin"),
      psTTCluster_Endcap_Fw_RZ_Zoom.getParameter<double>("ymax"));
  Cluster_Endcap_Fw_RZ_Zoom->setAxisTitle("L1 Cluster Forward Endcap position z [cm]", 1);
  Cluster_Endcap_Fw_RZ_Zoom->setAxisTitle("L1 Cluster Forward Endcap position #rho [cm]", 2);
  
  //TTCluster Backward Endcap #rho vs. z
  edm::ParameterSet psTTCluster_Endcap_Bw_RZ_Zoom =  conf_.getParameter<edm::ParameterSet>("TH2TTCluster_Endcap_Bw_RZ_Zoom");
  HistoName = "Cluster_Endcap_Bw_RZ_Zoom";
  Cluster_Endcap_Bw_RZ_Zoom = dqmStore_->book2D(HistoName, HistoName,
      psTTCluster_Endcap_Bw_RZ_Zoom.getParameter<int32_t>("Nbinsx"),
      psTTCluster_Endcap_Bw_RZ_Zoom.getParameter<double>("xmin"),
      psTTCluster_Endcap_Bw_RZ_Zoom.getParameter<double>("xmax"),
      psTTCluster_Endcap_Bw_RZ_Zoom.getParameter<int32_t>("Nbinsy"),
      psTTCluster_Endcap_Bw_RZ_Zoom.getParameter<double>("ymin"),
      psTTCluster_Endcap_Bw_RZ_Zoom.getParameter<double>("ymax"));
  Cluster_Endcap_Bw_RZ_Zoom->setAxisTitle("L1 Cluster Backward Endcap position z [cm]", 1);
  Cluster_Endcap_Bw_RZ_Zoom->setAxisTitle("L1 Cluster Backward Endcap position #rho [cm]", 2);
                                  
}//end of method

// ------------ method called once each job just after ending the event loop  ------------
void 
OuterTrackerMonitorCluster::endJob(void) 
{
	
}

DEFINE_FWK_MODULE(OuterTrackerMonitorCluster);
