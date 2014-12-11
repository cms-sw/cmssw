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
  iEvent.getByLabel( "TTStubsFromPixelDigis", "StubAccepted", PixelDigiTTStubHandle );
   
   

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
        Stub_Endcap->Fill(detIdStub.iDisk() );  
        Stub_Endcap_W->Fill(detIdStub.iDisk(), displStub - offsetStub);
        Stub_Endcap_O->Fill(detIdStub.iDisk(), offsetStub);

        if ( posStub.z() > 0 )
        {
          Stub_Endcap_Fw_XY->Fill( posStub.x(), posStub.y() );
          Stub_Endcap_Fw_RZ_Zoom->Fill( posStub.z(), posStub.perp() );
          Stub_Endcap_Fw->Fill(detIdStub.iDisk() );
        }
        else
        {
          Stub_Endcap_Bw_XY->Fill( posStub.x(), posStub.y() );
          Stub_Endcap_Bw_RZ_Zoom->Fill( posStub.z(), posStub.perp() );
          Stub_Endcap_Bw->Fill(detIdStub.iDisk() );
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

  dqmStore_->setCurrentFolder(topFolderName_+"/Stubs/");

  // Declaring histograms 
  std::string HistoName = "abc"; 
  
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
  Stub_Barrel_XY->setAxisTitle("TTStub Barrel position x ", 1);
  Stub_Barrel_XY->setAxisTitle("TTStub Barrel position y", 2);
  
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
  Stub_Barrel_XY_Zoom->setAxisTitle("TTStub Barrel position x ", 1);
  Stub_Barrel_XY_Zoom->setAxisTitle("TTStub Barrel position y", 2);
  
  
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
  Stub_Endcap_Fw_XY->setAxisTitle("TTStub Forward Endcap position x ", 1);
  Stub_Endcap_Fw_XY->setAxisTitle("TTStub Forward Endcap y", 2);
  
  
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
  Stub_Endcap_Bw_XY->setAxisTitle("TTStub Backward Endcap position x ", 1);
  Stub_Endcap_Bw_XY->setAxisTitle("TTStub Backward Endcap y", 2);
  
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
  Stub_RZ->setAxisTitle("TTStub z ", 1);
  Stub_RZ->setAxisTitle("TTStub #rho", 2);
  
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
  Stub_Endcap_Fw_RZ_Zoom->setAxisTitle("TTStub Forward Endcap z ", 1);
  Stub_Endcap_Fw_RZ_Zoom->setAxisTitle("TTStub Forward Endcap #rho", 2);
  
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
  Stub_Endcap_Bw_RZ_Zoom->setAxisTitle("TTStub Backward Endcap z ", 1);
  Stub_Endcap_Bw_RZ_Zoom->setAxisTitle("TTStub Backward Endcap #rho", 2);  
  
  //TTStub eta 
  edm::ParameterSet psTTStub_Eta =  conf_.getParameter<edm::ParameterSet>("TH1TTStub_Eta");
  HistoName = "Stub_Eta"; 
  Stub_Eta = dqmStore_ ->book1D(HistoName,HistoName, 
  psTTStub_Eta.getParameter<int32_t>("Nbinsx"), 
  psTTStub_Eta.getParameter<double>("xmin"), 
  psTTStub_Eta.getParameter<double>("xmax")); 
  //SetTitle
  Stub_Eta->setAxisTitle("TTStub eta",1); 
  Stub_Eta->setAxisTitle("# TTStubs ",2);
  
  //TTStub barrel stack
  edm::ParameterSet psTTStub_Barrel =  conf_.getParameter<edm::ParameterSet>("TH1TTStub_Stack");
  HistoName = "NStubs_Barrel"; 
  Stub_Barrel = dqmStore_ ->book1D(HistoName,HistoName, 
  psTTStub_Barrel.getParameter<int32_t>("Nbinsx"), 
  psTTStub_Barrel.getParameter<double>("xmin"), 
  psTTStub_Barrel.getParameter<double>("xmax")); 
  //SetTitle
  Stub_Barrel->setAxisTitle("Barrel Layer",1); 
  Stub_Barrel->setAxisTitle("# TTStubs ",2);
  
  //TTStub Endcap stack
  edm::ParameterSet psTTStub_Endcap =  conf_.getParameter<edm::ParameterSet>("TH1TTStub_Stack");
  HistoName = "NStubs_Endcap"; 
  Stub_Endcap = dqmStore_ ->book1D(HistoName,HistoName, 
  psTTStub_Endcap.getParameter<int32_t>("Nbinsx"), 
  psTTStub_Endcap.getParameter<double>("xmin"), 
  psTTStub_Endcap.getParameter<double>("xmax")); 
  //SetTitle
  Stub_Endcap->setAxisTitle("Endcap disk",1); 
  Stub_Endcap->setAxisTitle("# TTStubs ",2);
  
  //TTStub Endcap stack
  edm::ParameterSet psTTStub_Endcap_Fw =  conf_.getParameter<edm::ParameterSet>("TH1TTStub_Stack");
  HistoName = "NStubs_Endcap_Fw"; 
  Stub_Endcap_Fw = dqmStore_ ->book1D(HistoName,HistoName, 
  psTTStub_Endcap_Fw.getParameter<int32_t>("Nbinsx"), 
  psTTStub_Endcap_Fw.getParameter<double>("xmin"), 
  psTTStub_Endcap_Fw.getParameter<double>("xmax")); 
  //SetTitle
  Stub_Endcap_Fw->setAxisTitle("Forward Endcap disk",1); 
  Stub_Endcap_Fw->setAxisTitle("# TTStubs ",2);
  
  //TTStub Endcap stack
  edm::ParameterSet psTTStub_Endcap_Bw =  conf_.getParameter<edm::ParameterSet>("TH1TTStub_Stack");
  HistoName = "NStubs_Endcap_Bw"; 
  Stub_Endcap_Bw = dqmStore_ ->book1D(HistoName,HistoName, 
  psTTStub_Endcap_Bw.getParameter<int32_t>("Nbinsx"), 
  psTTStub_Endcap_Bw.getParameter<double>("xmin"), 
  psTTStub_Endcap_Bw.getParameter<double>("xmax")); 
  //SetTitle
  Stub_Endcap_Bw->setAxisTitle("Backward Endcap disk",1); 
  Stub_Endcap_Bw->setAxisTitle("# TTStubs ",2);
  
  //TTStub displ/offset
  edm::ParameterSet psTTStub_Barrel_W =  conf_.getParameter<edm::ParameterSet>("TH2TTStub_DisOf");
  HistoName = "Stub_Width_Barrel";
  //book the histogram
  Stub_Barrel_W = dqmStore_->book2D(HistoName, HistoName,
  psTTStub_Barrel_W.getParameter<int32_t>("Nbinsx"),
  psTTStub_Barrel_W.getParameter<double>("xmin"),
  psTTStub_Barrel_W.getParameter<double>("xmax"),
  psTTStub_Barrel_W.getParameter<int32_t>("Nbinsy"),
  psTTStub_Barrel_W.getParameter<double>("ymin"),
  psTTStub_Barrel_W.getParameter<double>("ymax"));
  //set titles
  Stub_Barrel_W->setAxisTitle("Layer",1); 
  Stub_Barrel_W->setAxisTitle("Displacement - Offset",2);
  
  edm::ParameterSet psTTStub_Barrel_O =  conf_.getParameter<edm::ParameterSet>("TH2TTStub_DisOf");
  HistoName = "Stub_Offset_Barrel";
  //book the histogram
  Stub_Barrel_O = dqmStore_->book2D(HistoName, HistoName,
  psTTStub_Barrel_O.getParameter<int32_t>("Nbinsx"),
  psTTStub_Barrel_O.getParameter<double>("xmin"),
  psTTStub_Barrel_O.getParameter<double>("xmax"),
  psTTStub_Barrel_O.getParameter<int32_t>("Nbinsy"),
  psTTStub_Barrel_O.getParameter<double>("ymin"),
  psTTStub_Barrel_O.getParameter<double>("ymax"));
  //set titles
  Stub_Barrel_O->setAxisTitle("Layer",1); 
  Stub_Barrel_O->setAxisTitle("Trigger Offset",2);
  
  edm::ParameterSet psTTStub_Endcap_W =  conf_.getParameter<edm::ParameterSet>("TH2TTStub_DisOf");
  HistoName = "Stub_Width_Endcap";
  //book the histogram
  Stub_Endcap_W = dqmStore_->book2D(HistoName, HistoName,
  psTTStub_Endcap_W.getParameter<int32_t>("Nbinsx"),
  psTTStub_Endcap_W.getParameter<double>("xmin"),
  psTTStub_Endcap_W.getParameter<double>("xmax"),
  psTTStub_Endcap_W.getParameter<int32_t>("Nbinsy"),
  psTTStub_Endcap_W.getParameter<double>("ymin"),
  psTTStub_Endcap_W.getParameter<double>("ymax"));
  //set titles
  Stub_Endcap_W->setAxisTitle("Layer",1); 
  Stub_Endcap_W->setAxisTitle("Displacement - Offset",2);
  
  edm::ParameterSet psTTStub_Endcap_O =  conf_.getParameter<edm::ParameterSet>("TH2TTStub_DisOf");
  HistoName = "Stub_Offset_Endcap";
  //book the histogram
  Stub_Endcap_O = dqmStore_->book2D(HistoName, HistoName,
  psTTStub_Endcap_O.getParameter<int32_t>("Nbinsx"),
  psTTStub_Endcap_O.getParameter<double>("xmin"),
  psTTStub_Endcap_O.getParameter<double>("xmax"),
  psTTStub_Endcap_O.getParameter<int32_t>("Nbinsy"),
  psTTStub_Endcap_O.getParameter<double>("ymin"),
  psTTStub_Endcap_O.getParameter<double>("ymax"));
  //Set titles
  Stub_Endcap_O->setAxisTitle("Layer",1); 
  Stub_Endcap_O->setAxisTitle("Trigger Offset",2);


}

// ------------ method called once each job just after ending the event loop  ------------
void 
OuterTrackerMonitorStub::endJob() 
{
}


//define this as a plug-in
DEFINE_FWK_MODULE(OuterTrackerMonitorStub);
