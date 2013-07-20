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
//         $Id: TestTrackerHierarchy.cpp,v 1.7 2013/01/07 19:44:30 wmtan Exp $


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

#include "CondFormats/Alignment/interface/AlignTransform.h"
#include "CondFormats/Alignment/interface/Alignments.h"


#include "Alignment/TrackerAlignment/interface/AlignableTracker.h"
#include "Alignment/CommonAlignment/interface/AlignableNavigator.h"
#include "Alignment/CommonAlignment/interface/AlignableObjectId.h"


static const int kLEAD_WIDTH = 40; // First field width

//
//
// class declaration
//

class TestTrackerHierarchy : public edm::EDAnalyzer {
public:
  explicit TestTrackerHierarchy( const edm::ParameterSet& pSet) 
    : dumpAlignments_(pSet.getUntrackedParameter<bool>("dumpAlignments")) {}
  
  virtual void analyze( const edm::Event&, const edm::EventSetup& );
private:
  // ----------member data ---------------------------
  void dumpAlignable( const Alignable*, unsigned int, unsigned int );
  void printInfo( const Alignable*, unsigned int );
  void dumpAlignments(const edm::EventSetup& setup, AlignableTracker *aliTracker) const;

  std::string leaders_, blank_, filled_;

  const bool dumpAlignments_;
};


void
TestTrackerHierarchy::analyze( const edm::Event&, const edm::EventSetup& setup )
{
  //Retrieve tracker topology from geometry
  edm::ESHandle<TrackerTopology> tTopoHandle;
  setup.get<IdealGeometryRecord>().get(tTopoHandle);
  const TrackerTopology* const tTopo = tTopoHandle.product();

  edm::LogInfo("TrackerHierarchy") << "Starting!";
  edm::ESHandle<TrackerGeometry> trackerGeometry;	 
  setup.get<TrackerDigiGeometryRecord>().get( trackerGeometry );
  AlignableTracker theAlignableTracker(&(*trackerGeometry), tTopo);

  leaders_ = "";
  blank_ = "   ";  // These two...
  filled_ = "|  "; // ... must have the same length

  // Now dump mother of each alignable
  //const Alignable* alignable = (&(*theAlignableTracker))->pixelHalfBarrels()[0];
  this->dumpAlignable(&theAlignableTracker, 1, 1);
  
  
  edm::LogInfo("TrackerHierarchy") << "Done!";

  if (dumpAlignments_) {
    this->dumpAlignments(setup, &theAlignableTracker);
  }

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
  int width = kLEAD_WIDTH-leaders_.length();

  std::ostringstream name,pos,rot;

  name << AlignableObjectId::idToString( alignable->alignableObjectId() ) << idau;

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

//__________________________________________________________________________________________________
void TestTrackerHierarchy::dumpAlignments(const edm::EventSetup& setup,
					  AlignableTracker *aliTracker) const
{
  edm::ESHandle<Alignments> alignments;
  setup.get<TrackerAlignmentRcd>().get(alignments);
  if (alignments->empty()) {
    edm::LogWarning("TrackerAlignment") << "@SUB=dumpAlignments"
					<< "No TrackerAlignmentRcd.";
  } else {
    AlignableNavigator navi(aliTracker);
    edm::LogInfo("TrackerAlignment") << "@SUB=dumpAlignments"
				     << "Start dumping alignments.";
    unsigned int nProblems = 0;
    for (std::vector<AlignTransform>::const_iterator iAlign = alignments->m_align.begin(),
	   iEnd = alignments->m_align.end(); iAlign != iEnd; ++iAlign) {
      const align::ID id = (*iAlign).rawId();
      const AlignTransform::Translation pos((*iAlign).translation());
      edm::LogVerbatim("DumpAlignable") << (*iAlign).rawId() << "  |  " << pos;

      AlignableDetOrUnitPtr aliPtr = navi.alignableFromDetId(id);
      if (!aliPtr.isNull()) {
	const Alignable::PositionType &aliPos = aliPtr->globalPosition();
	double dR = aliPos.perp() - pos.perp();
	double dRphi = (aliPos.phi() - pos.phi()) * pos.perp();
	double dZ = aliPos.z() - pos.z();
	if (dR*dR + dRphi*dRphi + dZ*dZ) { 
	  ++nProblems;
	  edm::LogWarning("Alignment") 
	    << "@SUB=analyze" << "Delta r,rphi,z: " << dR << " " << dRphi << " " << dZ
	    << "\nPos r,phi,z: " << pos.perp() << " " << pos.phi() << " " << pos.z();
	}
      } else {
	++nProblems;
	edm::LogWarning("Alignment") << "@SUB=dumpAlignments" << "No Alignable for Id " << id;
      }
    } // ending loop

    if (nProblems) {
      edm::LogWarning("TrackerAlignment") 
	<< "@SUB=dumpAlignments" << "Ending: " << nProblems << " Alignments with problems.";
    } else {
      edm::LogInfo("TrackerAlignment") << "@SUB=dumpAlignments" << "Ending without problem.";
    }
  }
}


//define this as a plug-in
DEFINE_FWK_MODULE(TestTrackerHierarchy);


