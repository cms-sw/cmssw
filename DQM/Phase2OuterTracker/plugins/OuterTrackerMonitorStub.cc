// -*- C++ -*-
//
// Package:    Phase2OuterTracker
// Class:      Phase2OuterTracker
// 
/**\class Phase2OuterTracker OuterTrackerMonitorStub.cc DQM/Phase2OuterTracker/plugins/OuterTrackerMonitorStub.cc

 Description: [one line class summary]

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  Isis Marina Van Parijs
//         Created:  Fri, 24 Oct 2014 12:31:31 GMT
// $Id$
//
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

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"


#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/Common/interface/DetSetVectorNew.h"
#include "DQM/SiStripCommon/interface/SiStripFolderOrganizer.h"
#include "DQM/Phase2OuterTracker/interface/OuterTrackerMonitorStub.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "DataFormats/L1TrackTrigger/interface/TTTypes.h"
#include "DataFormats/L1TrackTrigger/interface/TTStub.h"
#include "Geometry/TrackerGeometryBuilder/interface/PixelGeomDetUnit.h"
#include "Geometry/TrackerGeometryBuilder/interface/StackedTrackerGeometry.h"
#include "Geometry/Records/interface/StackedTrackerGeometryRecord.h"
#include "Geometry/CommonTopologies/interface/PixelTopology.h"

//
// constructors and destructor
//
OuterTrackerMonitorStub::OuterTrackerMonitorStub(const edm::ParameterSet& iConfig)
: dqmStore_(edm::Service<DQMStore>().operator->()), conf_(iConfig)
{
   //now do what ever initialization is needed
   topFolderName_ = conf_.getParameter<std::string>("TopFolderName");
   tagTTStubs_ = conf_.getParameter< edm::InputTag >("TTStubs");
}


OuterTrackerMonitorStub::~OuterTrackerMonitorStub()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called for each event  ------------
void
OuterTrackerMonitorStub::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  //using namespace edm;
  
  /// Geometry handles etc
  edm::ESHandle< TrackerGeometry > GeometryHandle;
  edm::ESHandle< StackedTrackerGeometry > StackedGeometryHandle;
  const StackedTrackerGeometry* theStackedGeometry;
  StackedTrackerGeometry::StackContainerIterator StackedTrackerIterator;
  
  /// Geometry setup
  /// Set pointers to Geometry
  iSetup.get< TrackerDigiGeometryRecord >().get(GeometryHandle);
  /// Set pointers to Stacked Modules
  iSetup.get< StackedTrackerGeometryRecord >().get(StackedGeometryHandle);
  theStackedGeometry = StackedGeometryHandle.product(); /// Note this is different
                                                       /// from the "global" geometry
   
  /// Track Trigger Stubs
  edm::Handle< edmNew::DetSetVector< TTStub< Ref_PixelDigi_ > > > PixelDigiTTStubHandle;
  iEvent.getByLabel( tagTTStubs_, PixelDigiTTStubHandle );
   
   

  //loop over input Stubs
  typename edmNew::DetSetVector< TTStub< Ref_PixelDigi_ > >::const_iterator otherInputIter;
  typename edmNew::DetSet< TTStub< Ref_PixelDigi_ > >::const_iterator otherContentIter;
  for ( otherInputIter = PixelDigiTTStubHandle->begin();otherInputIter != PixelDigiTTStubHandle->end();++otherInputIter )
  {
   
    for ( otherContentIter = otherInputIter->begin();otherContentIter != otherInputIter->end();++otherContentIter )
    {
      //Make reference stub
      edm::Ref< edmNew::DetSetVector< TTStub< Ref_PixelDigi_ > >, TTStub< Ref_PixelDigi_ > > tempStubRef = edmNew::makeRefTo( PixelDigiTTStubHandle, otherContentIter );

      //define position stub 
      GlobalPoint posStub = theStackedGeometry->findGlobalPosition( &(*tempStubRef) );
      double eta = posStub.eta();

      // get det ID (place of the stub)
      StackedTrackerDetId detIdStub( tempStubRef->getDetId() );
          
      // Get trigger displacement/offset
      double displStub = tempStubRef->getTriggerDisplacement();
      double offsetStub = tempStubRef->getTriggerOffset();

      Stub_RZ->Fill( posStub.z(), posStub.perp() );
      Stub_Eta->Fill(eta);

      if ( detIdStub.isBarrel() ) //if the stub is in the barrel
      {
        Stub_Barrel->Fill(detIdStub.iLayer() ); 

        Stub_Barrel_XY->Fill( posStub.x(), posStub.y() );
        Stub_Barrel_XY_Zoom->Fill( posStub.x(), posStub.y() );
	     
        Stub_Barrel_W->Fill(detIdStub.iLayer(), displStub - offsetStub);
        Stub_Barrel_O->Fill(detIdStub.iLayer(), offsetStub);

      }
      else if ( detIdStub.isEndcap() )
      {
        int disk = detIdStub.iDisk();
        int ring = detIdStub.iRing();
        Stub_Endcap_Disc->Fill(disk);
        Stub_Endcap_Ring->Fill(ring);
        Stub_Endcap_Disc_W->Fill(disk, displStub - offsetStub);
        Stub_Endcap_Ring_W->Fill(ring, displStub - offsetStub);
        Stub_Endcap_Disc_O->Fill(disk, offsetStub);
        Stub_Endcap_Ring_O->Fill(ring, offsetStub);

        if ( posStub.z() > 0 )
        {
          Stub_Endcap_Fw_XY->Fill( posStub.x(), posStub.y() );
          Stub_Endcap_Fw_RZ_Zoom->Fill( posStub.z(), posStub.perp() );
          Stub_Endcap_Disc_Fw->Fill(disk);
          Stub_Endcap_Ring_Fw[disk-1]->Fill(ring);
          Stub_Endcap_Ring_W_Fw[disk-1]->Fill(ring, displStub - offsetStub);
          Stub_Endcap_Ring_O_Fw[disk-1]->Fill(ring, offsetStub);
        }
        else
        {
          Stub_Endcap_Bw_XY->Fill( posStub.x(), posStub.y() );
          Stub_Endcap_Bw_RZ_Zoom->Fill( posStub.z(), posStub.perp() );
          Stub_Endcap_Disc_Bw->Fill(disk);
          Stub_Endcap_Ring_Bw[disk-1]->Fill(ring);
          Stub_Endcap_Ring_W_Bw[disk-1]->Fill(ring, displStub - offsetStub);
          Stub_Endcap_Ring_O_Bw[disk-1]->Fill(ring, offsetStub);
        }
      }
    }
  }
}


// ------------ method called when starting to processes a run  ------------
void 
OuterTrackerMonitorStub::beginRun(edm::Run const&, edm::EventSetup const&)
{
   //Make subdivision in the rootfile
  SiStripFolderOrganizer folder_organizer;
  folder_organizer.setSiStripFolderName(topFolderName_);
  folder_organizer.setSiStripFolder();
  std::string HistoName;    

  dqmStore_->setCurrentFolder(topFolderName_+"/Stubs/Position");
  
  
  ////////////////////////////////////////
  ///// GLOBAL POSITION OF THE STUB //////
  ////////////////////////////////////////
  
  edm::ParameterSet psTTStub_Barrel_XY =  conf_.getParameter<edm::ParameterSet>("TH2TTStub_Position");
  HistoName = "Stub_Barrel_XY";
  //book the histogram
  Stub_Barrel_XY = dqmStore_->book2D(HistoName, HistoName,
      psTTStub_Barrel_XY.getParameter<int32_t>("Nbinsx"),
      psTTStub_Barrel_XY.getParameter<double>("xmin"),
      psTTStub_Barrel_XY.getParameter<double>("xmax"),
      psTTStub_Barrel_XY.getParameter<int32_t>("Nbinsy"),
      psTTStub_Barrel_XY.getParameter<double>("ymin"),
      psTTStub_Barrel_XY.getParameter<double>("ymax"));
  //set titles
  Stub_Barrel_XY->setAxisTitle("L1 Stub Barrel position x [cm]", 1);
  Stub_Barrel_XY->setAxisTitle("L1 Stub Barrel position y [cm]", 2);
  
  edm::ParameterSet psTTStub_Barrel_XY_Zoom =  conf_.getParameter<edm::ParameterSet>("TH2TTStub_Barrel_XY_Zoom");
  HistoName = "Stub_Barrel_XY_Zoom";
  //book the histogram
  Stub_Barrel_XY_Zoom = dqmStore_->book2D(HistoName, HistoName,
      psTTStub_Barrel_XY_Zoom.getParameter<int32_t>("Nbinsx"),
      psTTStub_Barrel_XY_Zoom.getParameter<double>("xmin"),
      psTTStub_Barrel_XY_Zoom.getParameter<double>("xmax"),
      psTTStub_Barrel_XY_Zoom.getParameter<int32_t>("Nbinsy"),
      psTTStub_Barrel_XY_Zoom.getParameter<double>("ymin"),
      psTTStub_Barrel_XY_Zoom.getParameter<double>("ymax"));
  //set titles
  Stub_Barrel_XY_Zoom->setAxisTitle("L1 Stub Barrel position x [cm]", 1);
  Stub_Barrel_XY_Zoom->setAxisTitle("L1 Stub Barrel position y [cm]", 2);
  
  
  edm::ParameterSet psTTStub_Endcap_Fw_XY =  conf_.getParameter<edm::ParameterSet>("TH2TTStub_Position");
  HistoName = "Stub_Endcap_Fw_XY";
  //book the histogram
  Stub_Endcap_Fw_XY = dqmStore_->book2D(HistoName, HistoName,
      psTTStub_Endcap_Fw_XY.getParameter<int32_t>("Nbinsx"),
      psTTStub_Endcap_Fw_XY.getParameter<double>("xmin"),
      psTTStub_Endcap_Fw_XY.getParameter<double>("xmax"),
      psTTStub_Endcap_Fw_XY.getParameter<int32_t>("Nbinsy"),
      psTTStub_Endcap_Fw_XY.getParameter<double>("ymin"),
      psTTStub_Endcap_Fw_XY.getParameter<double>("ymax"));
  //set titles
  Stub_Endcap_Fw_XY->setAxisTitle("L1 Stub Endcap position x [cm]", 1);
  Stub_Endcap_Fw_XY->setAxisTitle("L1 Stub Endcap position y [cm]", 2);
  
  
  edm::ParameterSet psTTStub_Endcap_Bw_XY =  conf_.getParameter<edm::ParameterSet>("TH2TTStub_Position");
  HistoName = "Stub_Endcap_Bw_XY";
  //book the histogram
  Stub_Endcap_Bw_XY = dqmStore_->book2D(HistoName, HistoName,
      psTTStub_Endcap_Bw_XY.getParameter<int32_t>("Nbinsx"),
      psTTStub_Endcap_Bw_XY.getParameter<double>("xmin"),
      psTTStub_Endcap_Bw_XY.getParameter<double>("xmax"),
      psTTStub_Endcap_Bw_XY.getParameter<int32_t>("Nbinsy"),
      psTTStub_Endcap_Bw_XY.getParameter<double>("ymin"),
      psTTStub_Endcap_Bw_XY.getParameter<double>("ymax"));
  //set titles
  Stub_Endcap_Bw_XY->setAxisTitle("L1 Stub Endcap position x [cm]", 1);
  Stub_Endcap_Bw_XY->setAxisTitle("L1 Stub Endcap position y [cm]", 2);
  
  //TTStub #rho vs. z
  edm::ParameterSet psTTStub_RZ =  conf_.getParameter<edm::ParameterSet>("TH2TTStub_RZ");
  HistoName = "Stub_RZ";
  //book the histogram
  Stub_RZ = dqmStore_->book2D(HistoName, HistoName,
      psTTStub_RZ.getParameter<int32_t>("Nbinsx"),
      psTTStub_RZ.getParameter<double>("xmin"),
      psTTStub_RZ.getParameter<double>("xmax"),
      psTTStub_RZ.getParameter<int32_t>("Nbinsy"),
      psTTStub_RZ.getParameter<double>("ymin"),
      psTTStub_RZ.getParameter<double>("ymax"));
  //set titles
  Stub_RZ->setAxisTitle("L1 Stub position z [cm]", 1);
  Stub_RZ->setAxisTitle("L1 Stub position #rho [cm]", 2);
  
  //TTStub Forward Endcap #rho vs. z
  edm::ParameterSet psTTStub_Endcap_Fw_RZ_Zoom =  conf_.getParameter<edm::ParameterSet>("TH2TTStub_Endcap_Fw_RZ_Zoom");
  HistoName = "Stub_Endcap_Fw_RZ_Zoom";
  //book the histogram
  Stub_Endcap_Fw_RZ_Zoom = dqmStore_->book2D(HistoName, HistoName,
      psTTStub_Endcap_Fw_RZ_Zoom.getParameter<int32_t>("Nbinsx"),
      psTTStub_Endcap_Fw_RZ_Zoom.getParameter<double>("xmin"),
      psTTStub_Endcap_Fw_RZ_Zoom.getParameter<double>("xmax"),
      psTTStub_Endcap_Fw_RZ_Zoom.getParameter<int32_t>("Nbinsy"),
      psTTStub_Endcap_Fw_RZ_Zoom.getParameter<double>("ymin"),
      psTTStub_Endcap_Fw_RZ_Zoom.getParameter<double>("ymax"));
  //set titles
  Stub_Endcap_Fw_RZ_Zoom->setAxisTitle("L1 Stub Endcap position z [cm]", 1);
  Stub_Endcap_Fw_RZ_Zoom->setAxisTitle("L1 Stub Endcap position #rho [cm]", 2);
  
  //TTStub Backward Endcap #rho vs. z
  edm::ParameterSet psTTStub_Endcap_Bw_RZ_Zoom =  conf_.getParameter<edm::ParameterSet>("TH2TTStub_Endcap_Bw_RZ_Zoom");
  HistoName = "Stub_Endcap_Bw_RZ_Zoom";
  //book the histogram
  Stub_Endcap_Bw_RZ_Zoom = dqmStore_->book2D(HistoName, HistoName,
      psTTStub_Endcap_Bw_RZ_Zoom.getParameter<int32_t>("Nbinsx"),
      psTTStub_Endcap_Bw_RZ_Zoom.getParameter<double>("xmin"),
      psTTStub_Endcap_Bw_RZ_Zoom.getParameter<double>("xmax"),
      psTTStub_Endcap_Bw_RZ_Zoom.getParameter<int32_t>("Nbinsy"),
      psTTStub_Endcap_Bw_RZ_Zoom.getParameter<double>("ymin"),
      psTTStub_Endcap_Bw_RZ_Zoom.getParameter<double>("ymax"));
  //set titles
  Stub_Endcap_Bw_RZ_Zoom->setAxisTitle("L1 Stub Endcap position z [cm]", 1);
  Stub_Endcap_Bw_RZ_Zoom->setAxisTitle("L1 Stub Endcap position #rho [cm]", 2);  

  dqmStore_->setCurrentFolder(topFolderName_+"/Stubs");
  
  //TTStub eta 
  edm::ParameterSet psTTStub_Eta =  conf_.getParameter<edm::ParameterSet>("TH1TTStub_Eta");
  HistoName = "Stub_Eta"; 
  Stub_Eta = dqmStore_ ->book1D(HistoName,HistoName, 
      psTTStub_Eta.getParameter<int32_t>("Nbinsx"), 
      psTTStub_Eta.getParameter<double>("xmin"), 
      psTTStub_Eta.getParameter<double>("xmax")); 
  //SetTitle
  Stub_Eta->setAxisTitle("#eta",1); 
  Stub_Eta->setAxisTitle("# L1 Stubs ",2);

  dqmStore_->setCurrentFolder(topFolderName_+"/Stubs/NStubs");
  
  //TTStub barrel stack
  edm::ParameterSet psTTStub_Barrel =  conf_.getParameter<edm::ParameterSet>("TH1TTStub_Layers");
  HistoName = "NStubs_Barrel"; 
  Stub_Barrel = dqmStore_ ->book1D(HistoName,HistoName, 
      psTTStub_Barrel.getParameter<int32_t>("Nbinsx"), 
      psTTStub_Barrel.getParameter<double>("xmin"), 
      psTTStub_Barrel.getParameter<double>("xmax")); 
  //SetTitle
  Stub_Barrel->setAxisTitle("Barrel Layer",1); 
  Stub_Barrel->setAxisTitle("# L1 Stubs ",2);
  
  //TTStub Endcap stack
  edm::ParameterSet psTTStub_ECDisc =  conf_.getParameter<edm::ParameterSet>("TH1TTStub_Disks");
  HistoName = "NStubs_Endcap_Disc"; 
  Stub_Endcap_Disc = dqmStore_ ->book1D(HistoName,HistoName, 
      psTTStub_ECDisc.getParameter<int32_t>("Nbinsx"), 
      psTTStub_ECDisc.getParameter<double>("xmin"), 
      psTTStub_ECDisc.getParameter<double>("xmax")); 
  //SetTitle
  Stub_Endcap_Disc->setAxisTitle("Endcap Disc",1); 
  Stub_Endcap_Disc->setAxisTitle("# L1 Stubs ",2);
  
  //TTStub Endcap stack
  HistoName = "NStubs_Endcap_Disc_Fw"; 
  Stub_Endcap_Disc_Fw = dqmStore_ ->book1D(HistoName,HistoName, 
      psTTStub_ECDisc.getParameter<int32_t>("Nbinsx"), 
      psTTStub_ECDisc.getParameter<double>("xmin"), 
      psTTStub_ECDisc.getParameter<double>("xmax")); 
  //SetTitle
  Stub_Endcap_Disc_Fw->setAxisTitle("Forward Endcap Disc",1); 
  Stub_Endcap_Disc_Fw->setAxisTitle("# L1 Stubs ",2);
  
  //TTStub Endcap stack
  HistoName = "NStubs_Endcap_Disc_Bw"; 
  Stub_Endcap_Disc_Bw = dqmStore_ ->book1D(HistoName,HistoName, 
      psTTStub_ECDisc.getParameter<int32_t>("Nbinsx"), 
      psTTStub_ECDisc.getParameter<double>("xmin"), 
      psTTStub_ECDisc.getParameter<double>("xmax")); 
  //SetTitle
  Stub_Endcap_Disc_Bw->setAxisTitle("Backward Endcap Disc",1); 
  Stub_Endcap_Disc_Bw->setAxisTitle("# L1 Stubs ",2);
  
  edm::ParameterSet psTTStub_ECRing =  conf_.getParameter<edm::ParameterSet>("TH1TTStub_Rings");
  
  HistoName = "NStubs_Endcap_Ring"; 
  Stub_Endcap_Ring = dqmStore_ ->book1D(HistoName,HistoName, 
      psTTStub_ECRing.getParameter<int32_t>("Nbinsx"), 
      psTTStub_ECRing.getParameter<double>("xmin"), 
      psTTStub_ECRing.getParameter<double>("xmax")); 
  //SetTitle
  Stub_Endcap_Ring->setAxisTitle("Endcap Ring",1); 
  Stub_Endcap_Ring->setAxisTitle("# L1 Stubs ",2);
  
  for(int i=0;i<5;i++){
    Char_t histo[200];
    sprintf(histo, "NStubs_Disc+%d", i+1);  
    //TTStub Endcap stack
    Stub_Endcap_Ring_Fw[i] = dqmStore_ ->book1D(histo, histo, 
        psTTStub_ECRing.getParameter<int32_t>("Nbinsx"), 
        psTTStub_ECRing.getParameter<double>("xmin"), 
        psTTStub_ECRing.getParameter<double>("xmax")); 
    //SetTitle
    Stub_Endcap_Ring_Fw[i]->setAxisTitle("Endcap Ring",1); 
    Stub_Endcap_Ring_Fw[i]->setAxisTitle("# L1 Stubs ",2);
  }
  
  for(int i=0;i<5;i++){
    Char_t histo[200];
    sprintf(histo, "NStubs_Disc-%d", i+1);  
    //TTStub Endcap stack
    Stub_Endcap_Ring_Bw[i] = dqmStore_ ->book1D(histo, histo, 
        psTTStub_ECRing.getParameter<int32_t>("Nbinsx"), 
        psTTStub_ECRing.getParameter<double>("xmin"), 
        psTTStub_ECRing.getParameter<double>("xmax")); 
    //SetTitle
    Stub_Endcap_Ring_Bw[i]->setAxisTitle("Endcap Ring",1); 
    Stub_Endcap_Ring_Bw[i]->setAxisTitle("# L1 Stubs ",2);
  }
  
  //TTStub displ/offset
  edm::ParameterSet psTTStub_Barrel_2D =  conf_.getParameter<edm::ParameterSet>("TH2TTStub_DisOf_Layer");
  edm::ParameterSet psTTStub_ECDisc_2D =  conf_.getParameter<edm::ParameterSet>("TH2TTStub_DisOf_Disk");
  edm::ParameterSet psTTStub_ECRing_2D =  conf_.getParameter<edm::ParameterSet>("TH2TTStub_DisOf_Ring");

  dqmStore_->setCurrentFolder(topFolderName_+"/Stubs/Width");
  
  HistoName = "Stub_Width_Barrel";
  //book the histogram
  Stub_Barrel_W = dqmStore_->book2D(HistoName, HistoName,
      psTTStub_Barrel_2D.getParameter<int32_t>("Nbinsx"),
      psTTStub_Barrel_2D.getParameter<double>("xmin"),
      psTTStub_Barrel_2D.getParameter<double>("xmax"),
      psTTStub_Barrel_2D.getParameter<int32_t>("Nbinsy"),
      psTTStub_Barrel_2D.getParameter<double>("ymin"),
      psTTStub_Barrel_2D.getParameter<double>("ymax"));
  //set titles
  Stub_Barrel_W->setAxisTitle("Barrel Layer",1); 
  Stub_Barrel_W->setAxisTitle("Displacement - Offset",2);
  
  HistoName = "Stub_Width_Endcap_Disc";
  //book the histogram
  Stub_Endcap_Disc_W = dqmStore_->book2D(HistoName, HistoName,
      psTTStub_ECDisc_2D.getParameter<int32_t>("Nbinsx"),
      psTTStub_ECDisc_2D.getParameter<double>("xmin"),
      psTTStub_ECDisc_2D.getParameter<double>("xmax"),
      psTTStub_ECDisc_2D.getParameter<int32_t>("Nbinsy"),
      psTTStub_ECDisc_2D.getParameter<double>("ymin"),
      psTTStub_ECDisc_2D.getParameter<double>("ymax"));
  //set titles
  Stub_Endcap_Disc_W->setAxisTitle("Endcap Disc",1); 
  Stub_Endcap_Disc_W->setAxisTitle("Displacement - Offset",2);
  
  HistoName = "Stub_Width_Endcap_Ring";
  //book the histogram
  Stub_Endcap_Ring_W = dqmStore_->book2D(HistoName, HistoName,
      psTTStub_ECRing_2D.getParameter<int32_t>("Nbinsx"),
      psTTStub_ECRing_2D.getParameter<double>("xmin"),
      psTTStub_ECRing_2D.getParameter<double>("xmax"),
      psTTStub_ECRing_2D.getParameter<int32_t>("Nbinsy"),
      psTTStub_ECRing_2D.getParameter<double>("ymin"),
      psTTStub_ECRing_2D.getParameter<double>("ymax"));
  //Set titles
  Stub_Endcap_Ring_W->setAxisTitle("Endcap Ring",1); 
  Stub_Endcap_Ring_W->setAxisTitle("Trigger Offset",2);
  
  for(int i=0;i<5;i++){
    Char_t histo[200];
    sprintf(histo, "Stub_Width_Disc+%d", i+1);
    //book the histograms
    Stub_Endcap_Ring_W_Fw[i] = dqmStore_->book2D(histo, histo,
        psTTStub_ECRing_2D.getParameter<int32_t>("Nbinsx"),
        psTTStub_ECRing_2D.getParameter<double>("xmin"),
        psTTStub_ECRing_2D.getParameter<double>("xmax"),
        psTTStub_ECRing_2D.getParameter<int32_t>("Nbinsy"),
        psTTStub_ECRing_2D.getParameter<double>("ymin"),
        psTTStub_ECRing_2D.getParameter<double>("ymax"));
    //set titles
    Stub_Endcap_Ring_W_Fw[i]->setAxisTitle("Endcap Ring",1); 
    Stub_Endcap_Ring_W_Fw[i]->setAxisTitle("Displacement - Offset",2);
  }
  
  for(int i=0;i<5;i++){
    Char_t histo[200];
    sprintf(histo, "Stub_Width_Disc-%d", i+1);
    //book the histograms
    Stub_Endcap_Ring_W_Bw[i] = dqmStore_->book2D(histo, histo,
        psTTStub_ECRing_2D.getParameter<int32_t>("Nbinsx"),
        psTTStub_ECRing_2D.getParameter<double>("xmin"),
        psTTStub_ECRing_2D.getParameter<double>("xmax"),
        psTTStub_ECRing_2D.getParameter<int32_t>("Nbinsy"),
        psTTStub_ECRing_2D.getParameter<double>("ymin"),
        psTTStub_ECRing_2D.getParameter<double>("ymax"));
    //set titles
    Stub_Endcap_Ring_W_Bw[i]->setAxisTitle("Endcap Ring",1); 
    Stub_Endcap_Ring_W_Bw[i]->setAxisTitle("Displacement - Offset",2);
  }

  dqmStore_->setCurrentFolder(topFolderName_+"/Stubs/Offset");
  
  HistoName = "Stub_Offset_Barrel";
  //book the histogram
  Stub_Barrel_O = dqmStore_->book2D(HistoName, HistoName,
      psTTStub_Barrel_2D.getParameter<int32_t>("Nbinsx"),
      psTTStub_Barrel_2D.getParameter<double>("xmin"),
      psTTStub_Barrel_2D.getParameter<double>("xmax"),
      psTTStub_Barrel_2D.getParameter<int32_t>("Nbinsy"),
      psTTStub_Barrel_2D.getParameter<double>("ymin"),
      psTTStub_Barrel_2D.getParameter<double>("ymax"));
  //set titles
  Stub_Barrel_O->setAxisTitle("Barrel Layer",1); 
  Stub_Barrel_O->setAxisTitle("Trigger Offset",2);
  
  HistoName = "Stub_Offset_Endcap_Disc";
  //book the histogram
  Stub_Endcap_Disc_O = dqmStore_->book2D(HistoName, HistoName,
      psTTStub_ECDisc_2D.getParameter<int32_t>("Nbinsx"),
      psTTStub_ECDisc_2D.getParameter<double>("xmin"),
      psTTStub_ECDisc_2D.getParameter<double>("xmax"),
      psTTStub_ECDisc_2D.getParameter<int32_t>("Nbinsy"),
      psTTStub_ECDisc_2D.getParameter<double>("ymin"),
      psTTStub_ECDisc_2D.getParameter<double>("ymax"));
  //Set titles
  Stub_Endcap_Disc_O->setAxisTitle("Endcap Disc",1); 
  Stub_Endcap_Disc_O->setAxisTitle("Trigger Offset",2);
  
  HistoName = "Stub_Offset_Endcap_Ring";
  //book the histogram
  Stub_Endcap_Ring_O = dqmStore_->book2D(HistoName, HistoName,
      psTTStub_ECRing_2D.getParameter<int32_t>("Nbinsx"),
      psTTStub_ECRing_2D.getParameter<double>("xmin"),
      psTTStub_ECRing_2D.getParameter<double>("xmax"),
      psTTStub_ECRing_2D.getParameter<int32_t>("Nbinsy"),
      psTTStub_ECRing_2D.getParameter<double>("ymin"),
      psTTStub_ECRing_2D.getParameter<double>("ymax"));
  //Set titles
  Stub_Endcap_Ring_O->setAxisTitle("Endcap Ring",1); 
  Stub_Endcap_Ring_O->setAxisTitle("Trigger Offset",2);
  
  for(int i=0;i<5;i++){
    Char_t histo[200];
    sprintf(histo, "Stub_Offset_Disc+%d", i+1);
    //book the histogram
    Stub_Endcap_Ring_O_Fw[i] = dqmStore_->book2D(histo, histo,
        psTTStub_ECRing_2D.getParameter<int32_t>("Nbinsx"),
        psTTStub_ECRing_2D.getParameter<double>("xmin"),
        psTTStub_ECRing_2D.getParameter<double>("xmax"),
        psTTStub_ECRing_2D.getParameter<int32_t>("Nbinsy"),
        psTTStub_ECRing_2D.getParameter<double>("ymin"),
        psTTStub_ECRing_2D.getParameter<double>("ymax"));
    //Set titles
    Stub_Endcap_Ring_O_Fw[i]->setAxisTitle("Endcap Ring",1); 
    Stub_Endcap_Ring_O_Fw[i]->setAxisTitle("Trigger Offset",2);
  }
  
  for(int i=0;i<5;i++){
    Char_t histo[200];
    sprintf(histo, "Stub_Offset_Disc-%d", i+1);
    //book the histogram
    Stub_Endcap_Ring_O_Bw[i] = dqmStore_->book2D(histo, histo,
        psTTStub_ECRing_2D.getParameter<int32_t>("Nbinsx"),
        psTTStub_ECRing_2D.getParameter<double>("xmin"),
        psTTStub_ECRing_2D.getParameter<double>("xmax"),
        psTTStub_ECRing_2D.getParameter<int32_t>("Nbinsy"),
        psTTStub_ECRing_2D.getParameter<double>("ymin"),
        psTTStub_ECRing_2D.getParameter<double>("ymax"));
    //Set titles
    Stub_Endcap_Ring_O_Bw[i]->setAxisTitle("Endcap Ring",1); 
    Stub_Endcap_Ring_O_Bw[i]->setAxisTitle("Trigger Offset",2);
  }
}

// ------------ method called once each job just after ending the event loop  ------------
void 
OuterTrackerMonitorStub::endJob() 
{
}


//define this as a plug-in
DEFINE_FWK_MODULE(OuterTrackerMonitorStub);
