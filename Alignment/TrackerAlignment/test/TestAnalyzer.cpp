// -*- C++ -*-
//
// Package:    TestAnalyzer
// Class:      TestAnalyzer
// 
//
// Description: Module to test the Alignment software
//
//
// Original Author:  Frederic Ronga
//         Created:  March 16, 2006
//


// system include files
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

#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"

//
//
// class declaration
//

class TestAnalyzer : public edm::EDAnalyzer {
public:
  explicit TestAnalyzer( const edm::ParameterSet& );
  ~TestAnalyzer();
  
  
  virtual void analyze( const edm::Event&, const edm::EventSetup& );
private:
  // ----------member data ---------------------------
  TTree* theTree;
  TFile* theFile;
  float x,y,z,phi,theta,length,thick,width;
  TRotMatrix* rot;

};

//
// constructors and destructor
//
TestAnalyzer::TestAnalyzer( const edm::ParameterSet& iConfig ) 
{ 

  // Open root file and define tree
  std::string fileName = iConfig.getUntrackedParameter<std::string>("fileName","test.root");
  theFile = new TFile( fileName.c_str(), "RECREATE" );
  theTree = new TTree( "theTree", "Detector units positions" );
  
  theTree->Branch("x",      &x,      "x/F"      );
  theTree->Branch("y",      &y,      "y/F"      );
  theTree->Branch("z",      &z,      "z/F"      );
  theTree->Branch("phi",    &phi,    "phi/F"    );
  theTree->Branch("theta",  &theta,  "theta/F"  );
  theTree->Branch("length", &length, "length/F" );
  theTree->Branch("width",  &width,  "width/F"  );
  theTree->Branch("thick",  &thick,  "thick/F"  );
  rot = 0;
  theTree->Branch("rot",    "TRotMatrix", &rot  );

}


TestAnalyzer::~TestAnalyzer()
{ 
  
  theTree->Write();
  theFile->Close();
  
}


void
TestAnalyzer::analyze( const edm::Event& iEvent, const edm::EventSetup& iSetup )
{

   
  edm::LogInfo("TrackerAlignment") << "Starting!";

  //
  // Retrieve tracker geometry from event setup
  //
  edm::ESHandle<TrackerGeometry> trackerGeometry;
  iSetup.get<TrackerDigiGeometryRecord>().get( trackerGeometry );

  // Now loop on detector units, and store position and orientation
  for ( std::vector<GeomDet*>::const_iterator iGeomDet = trackerGeometry->dets().begin();
		iGeomDet != trackerGeometry->dets().end(); iGeomDet++ )
	{
	  x      = (*iGeomDet)->position().x();
	  y      = (*iGeomDet)->position().y();
	  z      = (*iGeomDet)->position().z();
	  phi    = (*iGeomDet)->surface().normalVector().phi();
	  theta  = (*iGeomDet)->surface().normalVector().theta();
	  length = (*iGeomDet)->surface().bounds().length();
	  width  = (*iGeomDet)->surface().bounds().width();
	  thick  = (*iGeomDet)->surface().bounds().thickness();

	  double matrix[9] = { 
		(*iGeomDet)->rotation().xx(),
		(*iGeomDet)->rotation().xy(),
		(*iGeomDet)->rotation().xz(),
		(*iGeomDet)->rotation().yx(),
		(*iGeomDet)->rotation().yy(),
		(*iGeomDet)->rotation().yz(),
		(*iGeomDet)->rotation().zx(),
		(*iGeomDet)->rotation().zy(),
		(*iGeomDet)->rotation().zz()
	  };
	  rot = new TRotMatrix( "rot", "rot", matrix );

	  theTree->Fill();

	}
  
  edm::LogInfo("TrackerAlignment") << "Done!";

}

//define this as a plug-in
DEFINE_FWK_MODULE(TestAnalyzer)
