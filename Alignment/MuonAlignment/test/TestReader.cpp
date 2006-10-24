// -*- C++ -*-
//
//
// Description: Module to test the Alignment software
//
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
#include "CondFormats/DataRecord/interface/DTAlignmentRcd.h"
#include "CondFormats/DataRecord/interface/CSCAlignmentRcd.h"
#include "CondFormats/Alignment/interface/AlignmentErrors.h"
#include "CondFormats/Alignment/interface/AlignTransformError.h"
#include "CondFormats/DataRecord/interface/DTAlignmentErrorRcd.h"
#include "CondFormats/DataRecord/interface/CSCAlignmentErrorRcd.h"

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

   
  edm::LogInfo("MuonAlignment") << "Starting!";

  // Retrieve DT alignment[Error]s from DBase
  edm::ESHandle<Alignments> dtAlignments;
  iSetup.get<DTAlignmentRcd>().get( dtAlignments );
  edm::ESHandle<AlignmentErrors> dtAlignmentErrors;
  iSetup.get<DTAlignmentErrorRcd>().get( dtAlignmentErrors );

  for ( std::vector<AlignTransform>::const_iterator it = dtAlignments->m_align.begin();
		it != dtAlignments->m_align.end(); it++ )
	{
	  HepRotation fromAngles( (*it).eulerAngles()  );
	  Surface::RotationType rotation( fromAngles.xx(), fromAngles.xy(), fromAngles.xz(),
									  fromAngles.yx(), fromAngles.yy(), fromAngles.yz(),
									  fromAngles.zx(), fromAngles.zy(), fromAngles.zz() );

	  std::cout << (*it).rawId()
				<< "  " << (*it).translation().x()
				<< " " << (*it).translation().y()
				<< " " << (*it).translation().z()
				<< "  " << rotation.xx() << " " << rotation.xy() << " " << rotation.xz()
				<< " " << rotation.yx() << " " << rotation.yy() << " " << rotation.yz()
				<< " " << rotation.zx() << " " << rotation.zy() << " " << rotation.zz()
				<< std::endl;

	}
  std::cout << std::endl << "----------------------" << std::endl;

  for ( std::vector<AlignTransformError>::const_iterator it = dtAlignmentErrors->m_alignError.begin();
		it != dtAlignmentErrors->m_alignError.end(); it++ )
	{
	  HepSymMatrix error = (*it).matrix();
	  std::cout << (*it).rawId() << " ";
	  for ( int i=0; i<error.num_row(); i++ )
		for ( int j=0; j<=i; j++ ) 
		  std::cout << " " << error[i][j];
	  std::cout << std::endl;
	}



  // Retrieve CSC alignment[Error]s from DBase
  edm::ESHandle<Alignments> cscAlignments;
  iSetup.get<CSCAlignmentRcd>().get( cscAlignments );
  edm::ESHandle<AlignmentErrors> cscAlignmentErrors;
  iSetup.get<CSCAlignmentErrorRcd>().get( cscAlignmentErrors );

  for ( std::vector<AlignTransform>::const_iterator it = cscAlignments->m_align.begin();
		it != cscAlignments->m_align.end(); it++ )
	{
	  HepRotation fromAngles( (*it).eulerAngles()  );
	  Surface::RotationType rotation( fromAngles.xx(), fromAngles.xy(), fromAngles.xz(),
									  fromAngles.yx(), fromAngles.yy(), fromAngles.yz(),
									  fromAngles.zx(), fromAngles.zy(), fromAngles.zz() );

	  std::cout << (*it).rawId()
				<< "  " << (*it).translation().x()
				<< " " << (*it).translation().y()
				<< " " << (*it).translation().z()
				<< "  " << rotation.xx() << " " << rotation.xy() << " " << rotation.xz()
				<< " " << rotation.yx() << " " << rotation.yy() << " " << rotation.yz()
				<< " " << rotation.zx() << " " << rotation.zy() << " " << rotation.zz()
				<< std::endl;

	}
  std::cout << std::endl << "----------------------" << std::endl;

  for ( std::vector<AlignTransformError>::const_iterator it = cscAlignmentErrors->m_alignError.begin();
		it != cscAlignmentErrors->m_alignError.end(); it++ )
	{
	  HepSymMatrix error = (*it).matrix();
	  std::cout << (*it).rawId() << " ";
	  for ( int i=0; i<error.num_row(); i++ )
		for ( int j=0; j<=i; j++ ) 
		  std::cout << " " << error[i][j];
	  std::cout << std::endl;
	}


  edm::LogInfo("MuonAlignment") << "Done!";

}

//define this as a plug-in
DEFINE_FWK_MODULE(TestReader);
