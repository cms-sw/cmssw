// -*- C++ -*-
//
// Package:    TestIdealGeometry
// Class:      TestIdealGeometry
// 
//
// Description: Module to test the SurveyConverter software
//
//
// Original Author:  Roberto Covarelli
//         Created:  March 16, 2006
//


// system include files
#include <TTree.h>
#include <TFile.h>
// #include <TRotMatrix.h>

// user include files
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "Geometry/CommonDetUnit/interface/GeomDet.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
/* #include "CondFormats/Alignment/interface/Alignments.h"
#include "CondFormats/Alignment/interface/AlignTransform.h"
#include "CondFormats/AlignmentRecord/interface/TrackerAlignmentRcd.h"
#include "CondFormats/Alignment/interface/AlignmentErrors.h"
#include "CondFormats/Alignment/interface/AlignTransformError.h"
#include "CondFormats/AlignmentRecord/interface/TrackerAlignmentErrorRcd.h"*/

#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"

#include "Alignment/SurveyAnalysis/interface/SurveyDataReader.h"
//
//
// class declaration
//
using namespace std;

class TestIdealGeometry : public edm::EDAnalyzer {

  typedef SurveyDataReader::MapType    MapType;
  typedef SurveyDataReader::PairType   PairType;
  typedef SurveyDataReader::MapTypeOr  MapTypeOr;
  typedef SurveyDataReader::PairTypeOr PairTypeOr;

public:
  explicit TestIdealGeometry( const edm::ParameterSet& );
  ~TestIdealGeometry();
  
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
  int Id_;
  // TRotMatrix* rot_;
  static const int NFILES = 2;
};

//
// constructors and destructor
//
TestIdealGeometry::TestIdealGeometry( const edm::ParameterSet& iConfig ) :
  theParameterSet( iConfig )
{ 
  
  // Open root file and define tree
  std::string fileName = theParameterSet.getUntrackedParameter<std::string>("fileName","testideal.root");
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
}


TestIdealGeometry::~TestIdealGeometry()
{ 
  
  theTree->Write();
  theFile->Close();
  
}


void
TestIdealGeometry::analyze( const edm::Event& iEvent, const edm::EventSetup& iSetup )
{
  //Retrieve tracker topology from geometry
  edm::ESHandle<TrackerTopology> tTopoHandle;
  iSetup.get<IdealGeometryRecord>().get(tTopoHandle);
  const TrackerTopology* const tTopo = tTopoHandle.product();

  edm::LogInfo("TrackerAlignment") << "Starting!";

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
  edm::ESHandle<TrackerGeometry> trackerGeometry;
  iSetup.get<TrackerDigiGeometryRecord>().get( trackerGeometry );

  // Retrieve alignment[Error]s from DBase
  // edm::ESHandle<Alignments> alignments;
  // iSetup.get<TrackerAlignmentRcd>().get( alignments );
  // edm::ESHandle<AlignmentErrors> alignmentErrors;
  // iSetup.get<TrackerAlignmentErrorRcd>().get( alignmentErrors );

  int countDet = 0; 

  // Now loop on detector units, and store difference position and orientation w.r.t. survey
  for ( std::vector<GeomDet*>::const_iterator iGeomDet = trackerGeometry->dets().begin();
  		iGeomDet != trackerGeometry->dets().end(); iGeomDet++ )
    // for ( std::vector<AlignTransform>::const_iterator iGeomDet = alignments->m_align.begin();
    //	iGeomDet != alignments->m_align.end(); iGeomDet++ )
	{
          
	  if (countDet == 0) {  // Enter only once for double-sided dets

            countDet++;
	    unsigned int comparisonVect[6] = {0,0,0,0,0,0};
	    
	    if (((*iGeomDet)->geographicalId()).subdetId() == int(StripSubdetector::TIB)) {
	      
	      comparisonVect[0] = int(StripSubdetector::TIB);
	      
	      comparisonVect[1] = tTopo->tibLayer((*iGeomDet)->geographicalId());
              if (comparisonVect[1] < 3) countDet = countDet + 2;  
	      std::vector<unsigned int> theString = tTopo->tibStringInfo((*iGeomDet)->geographicalId());
	      comparisonVect[2] = theString[0];
	      comparisonVect[3] = theString[1];
	      comparisonVect[4] = theString[2];
	      comparisonVect[5] = tTopo->tibModule((*iGeomDet)->geographicalId());
	      
	    } else if (((*iGeomDet)->geographicalId()).subdetId() == int(StripSubdetector::TID)) {
	      
	      comparisonVect[0] = int(StripSubdetector::TID);
	      
	      comparisonVect[1] = tTopo->tidSide((*iGeomDet)->geographicalId());
	      comparisonVect[2] = tTopo->tidWheel((*iGeomDet)->geographicalId());
	      comparisonVect[3] = tTopo->tidRing((*iGeomDet)->geographicalId());
              if (comparisonVect[3] < 3) countDet = countDet + 2; 
	      std::vector<unsigned int> theModule = tTopo->tidModuleInfo((*iGeomDet)->geographicalId());
	      comparisonVect[4] = theModule[0];
	      comparisonVect[5] = theModule[1];
	      
	    }

	    for ( MapTypeOr::const_iterator it = theSurveyMap.begin(); it != theSurveyMap.end(); it++ ) {
	      std::vector<int> locPos = (it)->first;
	      align::Scalars align_params = (it)->second;
	      
	      if (locPos[0] == int(comparisonVect[0]) &&
		  locPos[1] == int(comparisonVect[1]) &&
		  locPos[2] == int(comparisonVect[2]) &&
		  locPos[3] == int(comparisonVect[3]) &&
		  locPos[4] == int(comparisonVect[4]) &&
		  locPos[5] == int(comparisonVect[5]) ) {
		
		Id_     = (*iGeomDet)->geographicalId().rawId();
		cout << "DetId = " << Id_ << " " << endl;
		cout << "DetId decodified = " << comparisonVect[0] << " " << comparisonVect[1] << " " << comparisonVect[2] << " " << comparisonVect[3] << " " << comparisonVect[4] << " " << comparisonVect[5] << endl;	      
		dx_      = (*iGeomDet)->position().x() - align_params[0];
		cout << "X pos : TRACKER_GEOM = " << std::fixed << std::setprecision(2) << (*iGeomDet)->position().x() << " / IDEAL RICCARDO = " << align_params[0] << endl; 
		dy_      = (*iGeomDet)->position().y() - align_params[1];
		cout << "Y pos : TRACKER_GEOM = " << std::fixed << std::setprecision(2) << (*iGeomDet)->position().y() << " / IDEAL RICCARDO = " << align_params[1] << endl;
		dz_      = (*iGeomDet)->position().z() - align_params[2];
		cout << "Z pos : TRACKER_GEOM = " << std::fixed << std::setprecision(2) << (*iGeomDet)->position().z() << " / IDEAL RICCARDO = " << align_params[2] << endl;
		dtx_     = (*iGeomDet)->rotation().xx() - align_params[6];
		cout << "Trans vect X : TRACKER_GEOM = " << std::fixed << std::setprecision(3) << (*iGeomDet)->rotation().xx() << " / IDEAL RICCARDO = " << align_params[6] << endl;
		dty_     = (*iGeomDet)->rotation().xy() - align_params[7];
		cout << "Trans vect Y : TRACKER_GEOM = " << std::fixed << std::setprecision(3) << (*iGeomDet)->rotation().xy() << " / IDEAL RICCARDO = " << align_params[7] << endl;
		dtz_     = (*iGeomDet)->rotation().xz() - align_params[8];
		cout << "Trans vect Z : TRACKER_GEOM = " << std::fixed << std::setprecision(3) << (*iGeomDet)->rotation().xz() << " / IDEAL RICCARDO = " << align_params[8] << endl; 	
		dkx_     = (*iGeomDet)->rotation().yx() - align_params[9];
		cout << "Long vect X : TRACKER_GEOM = " << std::fixed << std::setprecision(3) << (*iGeomDet)->rotation().yx() << " / IDEAL RICCARDO = " << align_params[9] << endl;
		dky_     = (*iGeomDet)->rotation().yy() - align_params[10];
		cout << "Long vect Y : TRACKER_GEOM = " << std::fixed << std::setprecision(3) << (*iGeomDet)->rotation().yy() << " / IDEAL RICCARDO = " << align_params[10] << endl;
		dkz_     = (*iGeomDet)->rotation().yz() - align_params[11];
		cout << "Long vect Z : TRACKER_GEOM = " << std::fixed << std::setprecision(3) << (*iGeomDet)->rotation().yz() << " / IDEAL RICCARDO = " << align_params[11] << endl;
		dnx_     = (*iGeomDet)->rotation().zx() - align_params[3];
		cout << "Norm vect X : TRACKER_GEOM = " << std::fixed << std::setprecision(3) << (*iGeomDet)->rotation().zx() << " / IDEAL RICCARDO = " << align_params[3] << endl;
		dny_     = (*iGeomDet)->rotation().zy() - align_params[4];
		cout << "Norm vect Y : TRACKER_GEOM = " << std::fixed << std::setprecision(3) << (*iGeomDet)->rotation().zy() << " / IDEAL RICCARDO = " << align_params[4] << endl;
		dnz_     = (*iGeomDet)->rotation().zz() - align_params[5];
		cout << "Norm vect Z : TRACKER_GEOM = " << std::fixed << std::setprecision(3) << (*iGeomDet)->rotation().zz() << " / IDEAL RICCARDO = " << align_params[5] << endl; 
		theTree->Fill();
	      }
	    }	  
	  } 
	  countDet--;
	}
  
  edm::LogInfo("TrackerAlignment") << "Done!";

}

//define this as a plug-in
DEFINE_FWK_MODULE(TestIdealGeometry);
