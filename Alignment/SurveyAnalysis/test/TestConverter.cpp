// -*- C++ -*-
//
// Package:    TestConverter
// Class:      TestConverter
// 
//
// Description: Module to test the SurveyConverter software
//
//
// Original Author:  Roberto Covarelli
//         Created:  March 16, 2006
//


// system include files
#include <string>
#include "TTree.h"
#include "TFile.h"
// #include "TRotMatrix.h"

// user include files
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "CondFormats/Alignment/interface/Alignments.h"
#include <boost/cstdint.hpp> 
#include "CondFormats/Alignment/interface/AlignmentErrorsExtended.h"
#include "CLHEP/Vector/RotationInterfaces.h" 
#include "CondFormats/AlignmentRecord/interface/TrackerAlignmentRcd.h"
#include "CondFormats/AlignmentRecord/interface/TrackerAlignmentErrorExtendedRcd.h"

#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "Geometry/Records/interface/TrackerTopologyRcd.h"

#include "Alignment/SurveyAnalysis/interface/SurveyDataReader.h"
//
//
// class declaration
//

class TestConverter : public edm::EDAnalyzer {

  typedef SurveyDataReader::MapType    MapType;
  typedef SurveyDataReader::PairType   PairType;
  typedef SurveyDataReader::MapTypeOr  MapTypeOr;
  typedef SurveyDataReader::PairTypeOr PairTypeOr;

public:
  explicit TestConverter( const edm::ParameterSet& );
  ~TestConverter();
  
  virtual void analyze( const edm::Event&, const edm::EventSetup& );
private:
  // ----------member data ---------------------------
  TTree* theTree;
  TFile* theFile;
  edm::ParameterSet theParameterSet;
  float dx_,dy_,dz_;
  float dtx_,dty_,dtz_;
  float dkx_,dky_,dkz_;
  float dnx_,dny_,dnz_;
  float errx_,erry_,errz_; 
  int Id_;
  // TRotMatrix* rot_;

  static const int NFILES = 2;
};

//
// constructors and destructor
//
TestConverter::TestConverter( const edm::ParameterSet& iConfig ) :
  theParameterSet( iConfig )
{ 
  
  // Open root file and define tree
  std::string fileName = theParameterSet.getUntrackedParameter<std::string>("fileName","test.root");
  theFile = new TFile( fileName.c_str(), "RECREATE" );
  theTree = new TTree( "theTree", "Detector units positions" );
  
  theTree->Branch("Id",     &Id_,     "Id/I"     );
  theTree->Branch("dx",     &dx_,     "dx/F"     );
  theTree->Branch("dy",     &dy_,     "dy/F"     );
  theTree->Branch("dz",     &dz_,     "dz/F"     );
  theTree->Branch("dtx",    &dtx_,    "dtx/F"    );
  theTree->Branch("dty",    &dty_,    "dty/F"    );
  theTree->Branch("dtz",    &dtz_,    "dtz/F"    );
  theTree->Branch("dkx",    &dkx_,    "dkx/F"    );
  theTree->Branch("dky",    &dky_,    "dky/F"    );
  theTree->Branch("dkz",    &dkz_,    "dkz/F"    );
  theTree->Branch("dnx",    &dnx_,    "dnx/F"    );
  theTree->Branch("dny",    &dny_,    "dny/F"    );
  theTree->Branch("dnz",    &dnz_,    "dnz/F"    ); 
  theTree->Branch("errx",   &errx_,   "errx/F"   );
  theTree->Branch("erry",   &erry_,   "erry/F"   );
  theTree->Branch("errz",   &errz_,   "errz/F"   );
}


TestConverter::~TestConverter()
{ 
  
  theTree->Write();
  theFile->Close();
  
}


void
TestConverter::analyze( const edm::Event& iEvent, const edm::EventSetup& iSetup )
{
  edm::LogInfo("TrackerAlignment") << "Starting!";

  //Retrieve tracker topology from geometry
  edm::ESHandle<TrackerTopology> tTopoHandle;
  iSetup.get<TrackerTopologyRcd>().get(tTopoHandle);
  const TrackerTopology* const tTopo = tTopoHandle.product();

  //
  // Read in the survey information from the text files
  // 
  edm::ParameterSet textFiles = theParameterSet.getParameter<edm::ParameterSet>( "textFileNames" );
  std::string textFileNames[NFILES]; 
  std::string fileType[NFILES];    
  textFileNames[0] = textFiles.getUntrackedParameter<std::string>("forTIB","NONE");  
  fileType[0] = "TIB";
  textFileNames[1] = textFiles.getUntrackedParameter<std::string>("forTID","NONE");
  fileType[1] = "TID";

  SurveyDataReader dataReader;
  for (int ii=0 ; ii<NFILES ;ii++) {
    if ( textFileNames[ii] == "NONE" )
      throw cms::Exception("BadConfig") << fileType[ii] << " input file not found in configuration";
    dataReader.readFile( textFileNames[ii], fileType[ii], tTopo );
  } 

  edm::LogInfo("TrackerAlignment") << "Files read";

  const MapTypeOr& theSurveyMap = dataReader.surveyMap();

  edm::LogInfo("TrackerAlignment") << "Map written";

  //
  // Retrieve tracker geometry from event setup
  //
  // edm::ESHandle<TrackerGeometry> trackerGeometry;
  // iSetup.get<TrackerDigiGeometryRecord>().get( trackerGeometry );

  // Retrieve alignment[Error]s from DBase
  edm::ESHandle<Alignments> alignments;
  iSetup.get<TrackerAlignmentRcd>().get( alignments );
  edm::ESHandle<AlignmentErrorsExtended> alignmentErrors;
  iSetup.get<TrackerAlignmentErrorExtendedRcd>().get( alignmentErrors );
  
  std::vector<AlignTransformErrorExtended> alignErrors = alignmentErrors->m_alignError;
  // int countDet = 0;

  // Now loop on detector units, and store difference position and orientation w.r.t. survey
  
   for ( std::vector<AlignTransform>::const_iterator iGeomDet = alignments->m_align.begin();
		iGeomDet != alignments->m_align.end(); iGeomDet++ )
	{
          
          /* if (countDet == 0) countDet = countDet + 3;
	  unsigned int comparisonVect[6] = {0,0,0,0,0,0};
	  
	  DetId * thisId = new DetId( (*iGeomDet).rawId() );
	  if (thisId->subdetId() == int(StripSubdetector::TIB)) {
	    
	    comparisonVect[0] = int(StripSubdetector::TIB);
	    comparisonVect[1] = tTopo->tibLayer(*thisId);  
            if (comparisonVect[1] < 3) {countDet--;} else {countDet = countDet - 3;}
	    std::vector<unsigned int> theString = tTopo->tibStringInfo(*thisId);
	    comparisonVect[2] = theString[0];
	    comparisonVect[3] = theString[1];
	    comparisonVect[4] = theString[2];
	    comparisonVect[5] = tTopo->tibModule(*thisId);
	    
	  } else if (thisId->subdetId() == int(StripSubdetector::TID)) {
	    
	    comparisonVect[0] = int(StripSubdetector::TID);
	    comparisonVect[1] = tTopo->tidSide(*thisId);
	    comparisonVect[2] = tTopo->tidWheel(*thisId);
	    comparisonVect[3] = tTopo->tidRing(*thisId); 
            if (comparisonVect[3] < 3) {countDet--;} else {countDet = countDet - 3;}
	    std::vector<unsigned int> theModule = tTopo->tidModule(thisId);
	    comparisonVect[4] = theModule[0];
	    comparisonVect[5] = theModule[1];
	    
	  }
	  
	  if (countDet == 0) { // Store only r-phi for double-sided modules
          */

	  for ( MapTypeOr::const_iterator it = theSurveyMap.begin(); it != theSurveyMap.end(); it++ ) {
	      const std::vector<int>& locPos = (it)->first;
	      const align::Scalars& align_params = (it)->second;
	      
	      /* if (locPos[0] == int(comparisonVect[0]) &&
		  locPos[2] == int(comparisonVect[1]) &&
		  locPos[3] == int(comparisonVect[2]) &&
		  locPos[4] == int(comparisonVect[3]) &&
		  locPos[5] == int(comparisonVect[4]) &&
		  locPos[6] == int(comparisonVect[5]) ) { */
                
              int thecomparison = (int)locPos[1] - (int)(*iGeomDet).rawId();
	      if (thecomparison == 0) {
                
		for ( std::vector<AlignTransformErrorExtended>::const_iterator it = alignErrors.begin();
		      it != alignErrors.end(); it++ ) {
		  
		  if ((*it).rawId() == (*iGeomDet).rawId()) {
		  
		    CLHEP::HepRotation fromAngles = (*iGeomDet).rotation() ;
		    align::RotationType rotation( fromAngles.xx(), fromAngles.xy(), fromAngles.xz(),
						  fromAngles.yx(), fromAngles.yy(), fromAngles.yz(),
						  fromAngles.zx(), fromAngles.zy(), fromAngles.zz() );
		    
		    Id_     = (*iGeomDet).rawId();    
		    dx_      = (*iGeomDet).translation().x() - align_params[15]; 
		    dy_      = (*iGeomDet).translation().y() - align_params[16];
		    dz_      = (*iGeomDet).translation().z() - align_params[17];
		    dtx_     = rotation.xx() - align_params[21];
		    dty_     = rotation.xy() - align_params[22];
		    dtz_     = rotation.xz() - align_params[23];
		    dkx_     = rotation.yx() - align_params[24];
		    dky_     = rotation.yy() - align_params[25];
		    dkz_     = rotation.yz() - align_params[26];
		    dnx_     = rotation.zx() - align_params[18];
		    dny_     = rotation.zy() - align_params[19];
		    dnz_     = rotation.zz() - align_params[20];
                    CLHEP::HepSymMatrix errMat = (*it).matrix();
                    errx_    = sqrt(errMat[0][0]); 
		    erry_    = sqrt(errMat[1][1]);
		    errz_    = sqrt(errMat[2][2]); 
		    
		    theTree->Fill();
		    
		    // if (dkx_ > 0.04) {
		    edm::LogInfo("TrackerAlignment") << "DetId = " << Id_ << "\n"
		     << "DetId decodified = " << locPos[0] << " " << locPos[2] << " " << locPos[3] << " " << locPos[4] << " " << locPos[5] << " " << locPos[6] << "\n"
		     << "X pos : TRACKER_MOVED = " << std::fixed << std::setprecision(2) << (*iGeomDet).translation().x() << " / SURVEY RICCARDO = " << align_params[15] << "\n"
		     << "Y pos : TRACKER_MOVED = " << std::fixed << std::setprecision(2) << (*iGeomDet).translation().y() << " / SURVEY RICCARDO = " << align_params[16] << "\n"
		     << "Z pos : TRACKER_MOVED = " << std::fixed << std::setprecision(2) << (*iGeomDet).translation().z() << " / SURVEY RICCARDO = " << align_params[17] << "\n"
		     << "SPATIAL DISTANCE = " << std::fixed << std::setprecision(3) << sqrt(pow(dx_,2)+pow(dy_,2)+pow(dz_,2)) << "\n"
		     << "Trans vect X : TRACKER_MOVED = " << std::fixed << std::setprecision(5) << rotation.xx() << " / SURVEY RICCARDO = " << align_params[21] << "\n"
		     << "Trans vect Y : TRACKER_MOVED = " << std::fixed << std::setprecision(5) << rotation.xy() << " / SURVEY RICCARDO = " << align_params[22] << "\n"
		     << "Trans vect Z : TRACKER_MOVED = " << std::fixed << std::setprecision(5) << rotation.xz() << " / SURVEY RICCARDO = " << align_params[23] << "\n" 
		     << "Long vect X : TRACKER_MOVED = " << std::fixed << std::setprecision(5) << rotation.yx() << " / SURVEY RICCARDO = " << align_params[24] << "\n"	
		     << "Long vect Y : TRACKER_MOVED = " << std::fixed << std::setprecision(5) << rotation.yy() << " / SURVEY RICCARDO = " << align_params[25] << "\n"  
		     << "Long vect Z : TRACKER_MOVED = " << std::fixed << std::setprecision(5) << rotation.yz() << " / SURVEY RICCARDO = " << align_params[26] << "\n"
		     << "Norm vect X : TRACKER_MOVED = " << std::fixed << std::setprecision(5) << rotation.zx() << " / SURVEY RICCARDO = " << align_params[18] << "\n"
		     << "Norm vect Y : TRACKER_MOVED = " << std::fixed << std::setprecision(5) << rotation.zy() << " / SURVEY RICCARDO = " << align_params[19] << "\n" 
		     << "Norm vect Z : TRACKER_MOVED = " << std::fixed << std::setprecision(5) << rotation.zz() << " / SURVEY RICCARDO = " << align_params[20]; 
		      // }
		  }
		}
	      }
	    }	  
	  } 
   //}
  edm::LogInfo("TrackerAlignment") << "Done!";

}

//define this as a plug-in
DEFINE_FWK_MODULE(TestConverter);
