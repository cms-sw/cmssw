// -*- C++ -*-
//
// Package:    TestMisalign
// Class:      TestMisalign
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

#include "Alignment/MuonAlignment/interface/AlignableMuon.h"
#include "Alignment/MuonAlignment/interface/AlignableDTChamber.h"
#include "Alignment/MuonAlignment/interface/AlignableCSCChamber.h"

#include "Geometry/DTGeometryBuilder/src/DTGeometryBuilderFromDDD.h"
#include "Geometry/CSCGeometryBuilder/src/CSCGeometryBuilderFromDDD.h"

#include "Geometry/Records/interface/MuonGeometryRecord.h"


//
//
// class declaration
//

class TestMisalign : public edm::EDAnalyzer {
public:
  explicit TestMisalign( const edm::ParameterSet& );
  ~TestMisalign();
  
  
  virtual void analyze( const edm::Event&, const edm::EventSetup& );
private:

};

//
// constructors and destructor
//
TestMisalign::TestMisalign( const edm::ParameterSet& iConfig ) 
{ 


}


TestMisalign::~TestMisalign()
{ 
  
}


void
TestMisalign::analyze( const edm::Event& iEvent, const edm::EventSetup& iSetup )
{

   
  edm::LogInfo("MuonAlignment") << "Starting!";



  //
  // Retrieve ideal geometry from event setup
  //
  edm::ESHandle<DTGeometry> dtGeometry;
  edm::ESHandle<CSCGeometry> cscGeometry;
  iSetup.get<MuonGeometryRecord>().get( dtGeometry );
  iSetup.get<MuonGeometryRecord>().get( cscGeometry );

  AlignableMuon* theAlignableMuon = new AlignableMuon( &(*dtGeometry), &(*cscGeometry) );


  
  edm::LogInfo("MuonAlignment") << "Done!";

}

//define this as a plug-in
DEFINE_FWK_MODULE(TestMisalign)
