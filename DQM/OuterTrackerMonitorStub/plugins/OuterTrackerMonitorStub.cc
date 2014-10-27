// -*- C++ -*-
//
// Package:    OuterTrackerMonitorStub
// Class:      OuterTrackerMonitorStub
// 
/**\class OuterTrackerMonitorStub OuterTrackerMonitorStub.cc DQM/OuterTrackerMonitorStub/plugins/OuterTrackerMonitorStub.cc

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
#include "DQM/OuterTrackerMonitorStub/interface/OuterTrackerMonitorStub.h"
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
   
   /// Track Trigger
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
		
		// get det ID (place of the stub)
		StackedTrackerDetId detIdStub( tempStubRef->getDetId() );

		if ( detIdStub.isBarrel() ) //if the stub is in the barrel
		{
		
		   hStub_Barrel_XY->Fill( posStub.x(), posStub.y() );
		
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
  edm::ParameterSet psTTStub_Barrel_XY =  conf_.getParameter<edm::ParameterSet>("TH2TTStub_Barrel_XY");
  HistoName = "hStub_Barrel_XY";
  //book the histogram
  hStub_Barrel_XY = dqmStore_->book2D(HistoName, HistoName,
  psTTStub_Barrel_XY.getParameter<int32_t>("Nbinsx"),
  psTTStub_Barrel_XY.getParameter<double>("xmin"),
  psTTStub_Barrel_XY.getParameter<double>("xmax"),
  psTTStub_Barrel_XY.getParameter<int32_t>("Nbinsy"),
  psTTStub_Barrel_XY.getParameter<double>("ymin"),
  psTTStub_Barrel_XY.getParameter<double>("ymax"));
  //set titles
  hStub_Barrel_XY->setAxisTitle("TTStub Barrel position x ", 1);
  hStub_Barrel_XY->setAxisTitle("TTStub Barrel position y", 2);

}

// ------------ method called once each job just after ending the event loop  ------------
void 
OuterTrackerMonitorStub::endJob() 
{
}


//define this as a plug-in
DEFINE_FWK_MODULE(OuterTrackerMonitorStub);
