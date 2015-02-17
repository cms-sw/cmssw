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
  tagTTTracks = iConfig.getParameter< edm::InputTag >("TTTracks");
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
  iEvent.getByLabel( tagTTTracks, PixelDigiTTTrackHandle );
  
  /// Geometry
 // edm::ESHandle< StackedTrackerGeometry > StackedGeometryHandle;
 // const StackedTrackerGeometry* theStackedGeometry;
 //  iSetup.get< StackedTrackerGeometryRecord >().get(StackedGeometryHandle);
 // theStackedGeometry = StackedGeometryHandle.product();
 
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
       
       
       
       if ( nStubs > 2 )
       {
           num3Stubs++;
       }
       else
       {
           num2Stubs++;
       
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
	folder_organizer.setSiStripFolder();	
	
	dqmStore_->setCurrentFolder(topFolderName_+"/L1Tracks/");
	
  // Nb of L1Tracks
  edm::ParameterSet psL1Track_3Stubs_N =  conf_.getParameter<edm::ParameterSet>("TH1L1Track_N");
   std::string HistoName = "L1Track_Track_3Stubs_N";
   L1Track_Track_3Stubs_N = dqmStore_->book1D(HistoName, HistoName,
      psL1Track_3Stubs_N.getParameter<int32_t>("Nbinsx"),
      psL1Track_3Stubs_N.getParameter<double>("xmin"),
      psL1Track_3Stubs_N.getParameter<double>("xmax"));
  L1Track_Track_3Stubs_N->setAxisTitle("# L1Tracks from at least 3 stubs", 1);
  L1Track_Track_3Stubs_N->setAxisTitle("# events", 2);
  
  edm::ParameterSet psL1Track_2Stubs_N =  conf_.getParameter<edm::ParameterSet>("TH1L1Track_N");
   std::string HistoName = "L1Track_Track_2Stubs_N";
   L1Track_Track_2Stubs_N = dqmStore_->book1D(HistoName, HistoName,
      psL1Track_2Stubs_N.getParameter<int32_t>("Nbinsx"),
      psL1Track_2Stubs_N.getParameter<double>("xmin"),
      psL1Track_2Stubs_N.getParameter<double>("xmax"));
  L1Track_Track_2Stubs_N->setAxisTitle("# L1Tracks from 2 stubs", 1);
  L1Track_Track_2Stubs_N->setAxisTitle("# events", 2);
	
                                  
}//end of method

// ------------ method called once each job just after ending the event loop  ------------
void 
OuterTrackerMonitorL1Track::endJob(void) 
{
	
}

DEFINE_FWK_MODULE(OuterTrackerMonitorL1Track);
