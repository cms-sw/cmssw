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

#include "CondFormats/Alignment/interface/Alignments.h"
#include "CondFormats/Alignment/interface/AlignTransform.h"
#include "CondFormats/DataRecord/interface/MuonAlignmentRcd.h"
#include "CondFormats/Alignment/interface/AlignmentErrors.h"
#include "CondFormats/Alignment/interface/AlignTransformError.h"
#include "CondFormats/DataRecord/interface/MuonAlignmentErrorRcd.h"

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

  for ( std::vector<AlignTransform>::const_iterator it = alignments->m_align.begin();
		it != alignments->m_align.end(); it++ )
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

  for ( std::vector<AlignTransformError>::const_iterator it = alignmentErrors->m_alignError.begin();
		it != alignmentErrors->m_alignError.end(); it++ )
	{
	  HepSymMatrix error = (*it).matrix();
	  std::cout << (*it).rawId() << " ";
	  for ( int i=0; i<error.num_row(); i++ )
		for ( int j=0; j<=i; j++ ) 
		  std::cout << " " << error[i][j];
	  std::cout << std::endl;
	}


  edm::LogInfo("TrackerAlignment") << "Done!";

}

//define this as a plug-in
DEFINE_FWK_MODULE(TestReader)
