// -*- C++ -*-
//
//
// Description: Module to test the Alignment software
//
//


// system include files
#include <string>
#include <TTree.h>
#include <TRotMatrix.h>

// user include files
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"


#include "CondFormats/Alignment/interface/Alignments.h"
#include "CondFormats/Alignment/interface/AlignTransform.h"
#include "CondFormats/AlignmentRecord/interface/DTAlignmentRcd.h"
#include "CondFormats/AlignmentRecord/interface/CSCAlignmentRcd.h"
#include "CondFormats/Alignment/interface/AlignmentErrors.h"
#include "CondFormats/Alignment/interface/AlignTransformError.h"
#include "CondFormats/AlignmentRecord/interface/DTAlignmentErrorRcd.h"
#include "CondFormats/AlignmentRecord/interface/CSCAlignmentErrorRcd.h"

//
//
// class declaration
//

class TestMuonReader : public edm::EDAnalyzer {
public:
  explicit TestMuonReader( const edm::ParameterSet& );
  ~TestMuonReader();
  
  
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
TestMuonReader::TestMuonReader( const edm::ParameterSet& iConfig ) 
{ 
}


TestMuonReader::~TestMuonReader()
{ 
}


void
TestMuonReader::analyze( const edm::Event& iEvent, const edm::EventSetup& iSetup )
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
	  CLHEP::HepRotation rot( (*it).rotation() );
	  align::RotationType rotation( rot.xx(), rot.xy(), rot.xz(),
					rot.yx(), rot.yy(), rot.yz(),
					rot.zx(), rot.zy(), rot.zz() );

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
	  CLHEP::HepSymMatrix error = (*it).matrix();
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
	  CLHEP::HepRotation rot( (*it).rotation() );
	  align::RotationType rotation( rot.xx(), rot.xy(), rot.xz(),
					rot.yx(), rot.yy(), rot.yz(),
					rot.zx(), rot.zy(), rot.zz() );

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
	  CLHEP::HepSymMatrix error = (*it).matrix();
	  std::cout << (*it).rawId() << " ";
	  for ( int i=0; i<error.num_row(); i++ )
		for ( int j=0; j<=i; j++ ) 
		  std::cout << " " << error[i][j];
	  std::cout << std::endl;
	}


  edm::LogInfo("MuonAlignment") << "Done!";

}

//define this as a plug-in
DEFINE_FWK_MODULE(TestMuonReader);
