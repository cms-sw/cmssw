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

#include "CondFormats/Alignment/interface/Alignments.h"
#include "CondFormats/Alignment/interface/AlignTransform.h"
#include "CondFormats/DataRecord/interface/TrackerAlignmentRcd.h"
#include "CondFormats/Alignment/interface/AlignmentErrors.h"
#include "CondFormats/Alignment/interface/AlignTransformError.h"
#include "CondFormats/DataRecord/interface/TrackerAlignmentErrorRcd.h"

//
//
// class declaration
//

class TestReader : public edm::EDAnalyzer {
public:
  explicit TestReader( const edm::ParameterSet& );
  ~TestReader();
  
  
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
TestReader::TestReader( const edm::ParameterSet& iConfig ) 
{ 
}


TestReader::~TestReader()
{ 
}


void
TestReader::analyze( const edm::Event& iEvent, const edm::EventSetup& iSetup )
{

   
  edm::LogInfo("TrackerAlignment") << "Starting!";

  // Retrieve alignment[Error]s from DBase
  edm::ESHandle<Alignments> alignments;
  iSetup.get<TrackerAlignmentRcd>().get( alignments );
  edm::ESHandle<AlignmentErrors> alignmentErrors;
  iSetup.get<TrackerAlignmentErrorRcd>().get( alignmentErrors );

  edm::LogVerbatim("DumpAlignments")  << "\n----------------------\n";
  for ( std::vector<AlignTransform>::const_iterator it = alignments->m_align.begin();
		it != alignments->m_align.end(); it++ )
	{
	  HepRotation fromAngles( (*it).eulerAngles()  );
	  Surface::RotationType rotation( fromAngles.xx(), fromAngles.xy(), fromAngles.xz(),
									  fromAngles.yx(), fromAngles.yy(), fromAngles.yz(),
									  fromAngles.zx(), fromAngles.zy(), fromAngles.zz() );

	  edm::LogVerbatim("DumpAlignments") << (*it).rawId()
				<< "  " << (*it).translation().x()
				<< " " << (*it).translation().y()
				<< " " << (*it).translation().z()
				<< "  " << rotation.xx() << " " << rotation.xy() << " " << rotation.xz()
				<< " " << rotation.yx() << " " << rotation.yy() << " " << rotation.yz()
				<< " " << rotation.zx() << " " << rotation.zy() << " " << rotation.zz();

	}
  edm::LogVerbatim("DumpAlignments")  << "\n----------------------\n";
  edm::LogVerbatim("DumpAlignmentErrors")  << "\n----------------------\n";

  std::vector<AlignTransformError> alignErrors = alignmentErrors->m_alignError;
  for ( std::vector<AlignTransformError>::const_iterator it = alignErrors.begin();
		it != alignErrors.end(); it++ )
	{
	  edm::LogVerbatim("DumpAlignments") << (*it).rawId() << (*it).matrix();
	}
  edm::LogVerbatim("DumpAlignmentErrors")  << "\n----------------------\n";

  edm::LogInfo("TrackerAlignment") << "Done!";

}

//define this as a plug-in
DEFINE_FWK_MODULE(TestReader);
