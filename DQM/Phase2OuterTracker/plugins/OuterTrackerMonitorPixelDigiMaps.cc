// -*- C++ -*-
//
// Package:    Phase2OuterTracker
// Class:      Phase2OuterTracker
//
/**\class Phase2OuterTracker OuterTrackerMonitorPixelDigiMaps.cc DQM/Phase2OuterTracker/plugins/OuterTrackerMonitorPixelDigiMaps.cc
 
 Description: [one line class summary]
 
 Implementation:
 [Notes on implementation]
 */
//
// Original Author:  Isis Van Parijs
//         Created:  Mon, 10 March 2015 13:57:08 GMT
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
#include "DQM/Phase2OuterTracker/interface/OuterTrackerMonitorPixelDigiMaps.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "Geometry/TrackerGeometryBuilder/interface/PixelGeomDetUnit.h"
//#include "Geometry/TrackerGeometryBuilder/interface/StackedTrackerGeometry.h"
#include "Geometry/Records/interface/StackedTrackerGeometryRecord.h"
#include "Geometry/CommonTopologies/interface/PixelTopology.h"
#include "DataFormats/SiPixelDigi/interface/PixelDigi.h"
#include "SimDataFormats/TrackerDigiSimLink/interface/PixelDigiSimLink.h"

//
// constructors and destructor
//
OuterTrackerMonitorPixelDigiMaps::OuterTrackerMonitorPixelDigiMaps(const edm::ParameterSet& iConfig)
: dqmStore_(edm::Service<DQMStore>().operator->()), conf_(iConfig)
{
  topFolderName_ = conf_.getParameter<std::string>("TopFolderName");
}

OuterTrackerMonitorPixelDigiMaps::~OuterTrackerMonitorPixelDigiMaps()
{
  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)
}

//
// member functions
//

// ------------ method called for each event  ------------
void OuterTrackerMonitorPixelDigiMaps::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
	// Geometry handles etc
	edm::ESHandle< TrackerGeometry > geometryHandle;
	const TrackerGeometry* theGeometry;
	
	/// Geometry setup
       /// Set pointers to Geometry
       iSetup.get<TrackerDigiGeometryRecord>().get(geometryHandle);
       theGeometry = &(*geometryHandle);
      
       /////////////////////
       // GET PIXEL DIGIS //
      /////////////////////
     edm::Handle< edm::DetSetVector< PixelDigi > > PixelDigiHandle;
     edm::Handle< edm::DetSetVector< PixelDigiSimLink > > PixelDigiSimLinkHandle;
     iEvent.getByLabel( "simSiPixelDigis", PixelDigiHandle );
     iEvent.getByLabel( "simSiPixelDigis", PixelDigiSimLinkHandle );
     
     //give errors     
     std::cerr<<PixelDigiHandle.id() << " ////// " << PixelDigiSimLinkHandle.id() <<std::endl;
     
     edm::DetSetVector<PixelDigi>::const_iterator detsIter;
     edm::DetSet<PixelDigi>::const_iterator hitsIter;
     
     
     /// Loop over detector elements identifying PixelDigis
     for ( detsIter = PixelDigiHandle->begin();  detsIter != PixelDigiHandle->end(); detsIter++ )
     {
     
           DetId tkId = detsIter->id;
	   
	   /// Loop over Digis in this specific detector element
	   for ( hitsIter = detsIter->data.begin(); hitsIter != detsIter->data.end(); hitsIter++ )
	   {
	   
	         /// Threshold (here it is NOT redundant)
		 if ( hitsIter->adc() <= 30 ) continue;
		 
		 /// Try to learn something from PixelDigi position
		 const GeomDetUnit* gDetUnit = theGeometry->idToDetUnit( tkId );
		 MeasurementPoint mp( hitsIter->row() + 0.5, hitsIter->column() + 0.5 );
		 GlobalPoint pdPos = gDetUnit->surface().toGlobal( gDetUnit->topology().localPosition( mp ) ) ;
		 
		 
		 PixelDigiMaps_RZ->Fill( pdPos.z(), pdPos.perp() );
	   
	   
	   
	   
	   }
     
     
     
     
     }

	
} // end of method


// ------------ method called once each job just before starting event loop  ------------
void
OuterTrackerMonitorPixelDigiMaps::beginRun(const edm::Run& run, const edm::EventSetup& es)
{
  SiStripFolderOrganizer folder_organizer;
	folder_organizer.setSiStripFolderName(topFolderName_);
	folder_organizer.setSiStripFolder();	
	
	dqmStore_->setCurrentFolder(topFolderName_+"/PixelDigiMaps/");
	
	
  TString HistoName;
  //TTPixelDigiMaps Backward Endcap #rho vs. z
  edm::ParameterSet psPixelDigiMaps_RZ =  conf_.getParameter<edm::ParameterSet>("TH2PixelDigiMaps_RZ");
  HistoName = "PixelDigiMaps_RZ";
  //book the histogram
  PixelDigiMaps_RZ = dqmStore_->book2D(HistoName, HistoName,
      psPixelDigiMaps_RZ.getParameter<int32_t>("Nbinsx"),
      psPixelDigiMaps_RZ.getParameter<double>("xmin"),
      psPixelDigiMaps_RZ.getParameter<double>("xmax"),
      psPixelDigiMaps_RZ.getParameter<int32_t>("Nbinsy"),
      psPixelDigiMaps_RZ.getParameter<double>("ymin"),
      psPixelDigiMaps_RZ.getParameter<double>("ymax"));
  //set titles
  PixelDigiMaps_RZ->setAxisTitle("PixelDigi measurement  z position", 1);
  PixelDigiMaps_RZ->setAxisTitle("PixelDigi measurement  #rho position", 2);
                                  
}//end of method

// ------------ method called once each job just after ending the event loop  ------------
void 
OuterTrackerMonitorPixelDigiMaps::endJob(void) 
{
	
}

DEFINE_FWK_MODULE(OuterTrackerMonitorPixelDigiMaps);
