// -*- C++ -*-
//
// Package:    TestTrackerHierarchy
// Class:      TestTrackerHierarchy
// 
//
// Description: Module to test the Alignment software
//
//
// Original Author:  Frederic Ronga
//         Created:  March 16, 2006
//


// system include files
#include <sstream>
#include <string>
#include <TTree.h>
#include <TFile.h>
#include <TRotMatrix.h>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "Alignment/TrackerAlignment/interface/AlignableTracker.h"
#include "Alignment/TrackerAlignment/interface/AlignableTrackerBarrelLayer.h"
#include "Alignment/TrackerAlignment/interface/AlignableTrackerRod.h"
#include "Alignment/CommonAlignment/interface/AlignableObjectId.h"

#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"

//
//
// class declaration
//

class TestTrackerHierarchy : public edm::EDAnalyzer {
public:
  explicit TestTrackerHierarchy( const edm::ParameterSet& );
  ~TestTrackerHierarchy();
  
  
  virtual void analyze( const edm::Event&, const edm::EventSetup& );
private:
  // ----------member data ---------------------------
  void dumpAlignable( Alignable* alignable);

};

//
// constructors and destructor
//
TestTrackerHierarchy::TestTrackerHierarchy( const edm::ParameterSet& iConfig ) 
{ 

}


TestTrackerHierarchy::~TestTrackerHierarchy()
{ 
  
  
}


void
TestTrackerHierarchy::analyze( const edm::Event& iEvent, const edm::EventSetup& iSetup )
{

   
  edm::LogInfo("TrackerHierarchy") << "Starting!";

  //
  // Retrieve ideal geometry from event setup
  //
  edm::ESHandle<GeometricDet> gD;
  edm::ESHandle<TrackerGeometry> trackerGeometry;
  iSetup.get<TrackerDigiGeometryRecord>().get( trackerGeometry );
  iSetup.get<IdealGeometryRecord>().get( gD );

  AlignableTracker* theAlignableTracker = new AlignableTracker( &(*gD), &(*trackerGeometry) );

  // Now dump mother of each alignable
  dumpAlignable( theAlignableTracker );
  
  
  edm::LogInfo("TrackerAlignment") << "Done!";

}


//__________________________________________________________________________________________________
void TestTrackerHierarchy::dumpAlignable( Alignable* alignable )
{

  AlignableObjectId converter;


  std::ostringstream message;
  
  message << "I am a " << converter.typeToName( alignable->alignableObjectId() );

  if ( alignable->mother() )
	message << " and my mother is a "
			<< converter.typeToName( alignable->mother()->alignableObjectId() );
  else
	message << " and I have no mother :-/";

  edm::LogInfo("DumpAlignable") << message.str();

  if ( alignable->components().size() )
	{
	  std::vector<Alignable*> comp = alignable->components();
	  for ( std::vector<Alignable*>::iterator iter = comp.begin(); iter != comp.end(); iter++ )
		dumpAlignable( *iter );
	}

}

//define this as a plug-in
DEFINE_FWK_MODULE(TestTrackerHierarchy);
