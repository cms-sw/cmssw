// -*- C++ -*-
//
// Package:    TestMuonHierarchy
// Class:      TestMuonHierarchy
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

#include "Alignment/MuonAlignment/interface/AlignableMuon.h"
#include "Alignment/CommonAlignment/interface/AlignableObjectId.h"

#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "Geometry/DTGeometryBuilder/src/DTGeometryBuilderFromDDD.h"
#include "Geometry/CSCGeometryBuilder/src/CSCGeometryBuilderFromDDD.h"

//
//
// class declaration
//

class TestMuonHierarchy : public edm::EDAnalyzer {
public:
  explicit TestMuonHierarchy( const edm::ParameterSet& );
  ~TestMuonHierarchy();
  
  
  virtual void analyze( const edm::Event&, const edm::EventSetup& );
private:
  // ----------member data ---------------------------
  void dumpAlignable( Alignable* alignable);

};

//
// constructors and destructor
//
TestMuonHierarchy::TestMuonHierarchy( const edm::ParameterSet& iConfig ) 
{ 

}


TestMuonHierarchy::~TestMuonHierarchy()
{ 
  
  
}


void
TestMuonHierarchy::analyze( const edm::Event& iEvent, const edm::EventSetup& iSetup )
{

   
  edm::LogInfo("MuonHierarchy") << "Starting!";

  //
  // Retrieve ideal geometry from event setup
  //
  edm::ESHandle<DTGeometry> dtGeometry;
  edm::ESHandle<CSCGeometry> cscGeometry;
  iSetup.get<MuonGeometryRecord>().get( dtGeometry );
  iSetup.get<MuonGeometryRecord>().get( cscGeometry );

  AlignableMuon* theAlignableMuon = new AlignableMuon( &(*dtGeometry), &(*cscGeometry) );

  // Now dump mother of each alignable
  dumpAlignable( theAlignableMuon );
  
  
  edm::LogInfo("MuonAlignment") << "Done!";

}


//__________________________________________________________________________________________________
void TestMuonHierarchy::dumpAlignable( Alignable* alignable )
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
DEFINE_FWK_MODULE(TestMuonHierarchy);
