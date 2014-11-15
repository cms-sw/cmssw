// -*- C++ -*-
//
//
// Description: Module to test the Alignment software
//
//
// Original Author:  Frederic Ronga
//         Created:  March 16, 2006
//


// system include files
#include <TRotMatrix.h>

// user include files
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"


#include "CondFormats/Alignment/interface/Alignments.h"
#include "CondFormats/Alignment/interface/AlignmentErrorsExtended.h"
#include "CondFormats/AlignmentRecord/interface/TrackerAlignmentRcd.h"
#include "CondFormats/AlignmentRecord/interface/TrackerAlignmentErrorExtendedRcd.h"

//
//
// class declaration
//

class TestTrackerReader : public edm::EDAnalyzer {
public:
  explicit TestTrackerReader( const edm::ParameterSet& )
    : rot(0) {}
  
  
  virtual void analyze( const edm::Event&, const edm::EventSetup& );
private:
  // ----------member data ---------------------------
  float x,y,z,phi,theta,length,thick,width;
  TRotMatrix* rot;

};


void
TestTrackerReader::analyze( const edm::Event& iEvent, const edm::EventSetup& iSetup )
{

   
  edm::LogInfo("TrackerAlignment") << "Starting!";

  // Retrieve alignment[Error]s from DBase
  edm::ESHandle<Alignments> alignments;
  iSetup.get<TrackerAlignmentRcd>().get( alignments );
  edm::ESHandle<AlignmentErrorsExtended> alignmentErrors;
  iSetup.get<TrackerAlignmentErrorExtendedRcd>().get( alignmentErrors );

  edm::LogVerbatim("DumpAlignments")  << "\n----------------------\n";
  for ( std::vector<AlignTransform>::const_iterator it = alignments->m_align.begin();
		it != alignments->m_align.end(); it++ )
	{
	  CLHEP::HepRotation rot( (*it).rotation() );
	  align::RotationType rotation( rot.xx(), rot.xy(), rot.xz(),
					rot.yx(), rot.yy(), rot.yz(),
					rot.zx(), rot.zy(), rot.zz() );

	  edm::LogVerbatim("DumpAlignments") << (*it).rawId()
				<< "  " << (*it).translation().x()
				<< " " << (*it).translation().y()
				<< " " << (*it).translation().z()
				<< "  " << rotation.xx() << " " << rotation.xy() << " " << rotation.xz()
				<< " " << rotation.yx() << " " << rotation.yy() << " " << rotation.yz()
				<< " " << rotation.zx() << " " << rotation.zy() << " " << rotation.zz();

	}
  edm::LogVerbatim("DumpAlignments")  << "\n----------------------\n";
  edm::LogVerbatim("DumpAlignmentErrorsExtended")  << "\n----------------------\n";

  std::vector<AlignTransformErrorExtended> alignErrors = alignmentErrors->m_alignError;
  for ( std::vector<AlignTransformErrorExtended>::const_iterator it = alignErrors.begin();
		it != alignErrors.end(); it++ )
	{
	  edm::LogVerbatim("DumpAlignments") << (*it).rawId() << (*it).matrix();
	}
  edm::LogVerbatim("DumpAlignmentErrorsExtended")  << "\n----------------------\n";

  edm::LogInfo("TrackerAlignment") << "Done!";

}

//define this as a plug-in
DEFINE_FWK_MODULE(TestTrackerReader);
