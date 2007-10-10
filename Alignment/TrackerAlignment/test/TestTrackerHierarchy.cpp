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

// user include files
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "Alignment/TrackerAlignment/interface/AlignableTracker.h"
#include "Alignment/CommonAlignment/interface/AlignableObjectId.h"

//
//
// class declaration
//

class TestTrackerHierarchy : public edm::EDAnalyzer {
public:
  explicit TestTrackerHierarchy( const edm::ParameterSet& ) {}
  
  virtual void analyze( const edm::Event&, const edm::EventSetup& );
private:
  // ----------member data ---------------------------
  void dumpAlignable( const Alignable* );

};


void
TestTrackerHierarchy::analyze( const edm::Event&, const edm::EventSetup& )
{

  edm::LogInfo("TrackerHierarchy") << "Starting!";

  AlignableTracker* theAlignableTracker = new AlignableTracker;

  // Now dump mother of each alignable
  dumpAlignable( theAlignableTracker );
  
  
  edm::LogInfo("TrackerAlignment") << "Done!";

  delete theAlignableTracker;

}


//__________________________________________________________________________________________________
void TestTrackerHierarchy::dumpAlignable( const Alignable* alignable )
{

  static AlignableObjectId converter;

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
    const align::Alignables& comp = alignable->components();
    for ( align::Alignables::const_iterator iter = comp.begin(); iter != comp.end(); iter++ )
      dumpAlignable( *iter );
  }

}

//define this as a plug-in
DEFINE_FWK_MODULE(TestTrackerHierarchy);
