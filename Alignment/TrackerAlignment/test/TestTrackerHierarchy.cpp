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
#include <iomanip>

// user include files
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/EventSetup.h"	 
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"

#include "Alignment/TrackerAlignment/interface/AlignableTracker.h"
#include "Alignment/CommonAlignment/interface/AlignableObjectId.h"


static const int kLEAD_WIDTH = 40; // First field width

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
  void dumpAlignable( const Alignable*, unsigned int, unsigned int );
  void printInfo( const Alignable*, unsigned int );

  std::string leaders_, blank_, filled_;

};


void
TestTrackerHierarchy::analyze( const edm::Event&, const edm::EventSetup& setup )
{

  edm::LogInfo("TrackerHierarchy") << "Starting!";
  edm::ESHandle<TrackerGeometry> trackerGeometry;	 
  setup.get<TrackerDigiGeometryRecord>().get( trackerGeometry );
  std::auto_ptr<AlignableTracker> theAlignableTracker( new AlignableTracker(&(*trackerGeometry)) );

  leaders_ = "";
  blank_ = "   ";  // These two...
  filled_ = "|  "; // ... must have the same length

  // Now dump mother of each alignable
  //const Alignable* alignable = (&(*theAlignableTracker))->pixelHalfBarrels()[0];
  dumpAlignable( &(*theAlignableTracker), 1, 1 );
  
  
  edm::LogInfo("TrackerAlignment") << "Done!";

}


//__________________________________________________________________________________________________
// Recursive loop on alignable hierarchy
void TestTrackerHierarchy::dumpAlignable( const Alignable* alignable,
                                          unsigned int idau, unsigned int ndau )
{

  printInfo( alignable, idau );

  if ( ndau != idau ) leaders_ += filled_;
  else leaders_ += blank_;

  const align::Alignables& comps = alignable->components();
  if ( unsigned int ndau = comps.size() ) {
    unsigned int idau = 0;
    for ( align::Alignables::const_iterator iter = comps.begin(); iter != comps.end(); ++iter )
      dumpAlignable( *iter, ++idau, ndau );
  }

  leaders_ = leaders_.substr( 0, leaders_.length()-blank_.length() );

}


//__________________________________________________________________________________________________
// Do the actual printout
void TestTrackerHierarchy::printInfo( const Alignable* alignable,
                                      unsigned int idau )
{
  
  static AlignableObjectId converter;
  int width = kLEAD_WIDTH-leaders_.length();

  std::ostringstream name,pos,rot;

  name << converter.typeToName( alignable->alignableObjectId() ) << idau;

  // Position
  pos.setf(std::ios::fixed);
  pos << "(" << std::right 
      << std::setw(8)  << std::setprecision(4) << alignable->globalPosition().x() << ","
      << std::setw(8)  << std::setprecision(4) << alignable->globalPosition().y() << ","
      << std::setw(8)  << std::setprecision(4) << alignable->globalPosition().z() << ")";

  edm::LogVerbatim("DumpAlignable") 
    << leaders_ << "+-> "
    << std::setw(width) << std::left << name.str()
    << " | " << std::setw(3)  << std::left << alignable->components().size()
    << " | " << std::setw(11) << std::left << alignable->id()
    << " | " << pos.str();

}
//define this as a plug-in
DEFINE_FWK_MODULE(TestTrackerHierarchy);


