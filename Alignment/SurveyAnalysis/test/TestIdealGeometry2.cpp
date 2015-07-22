// -*- C++ -*-
//
// Package:    TestIdealGeometry2
// Class:      TestIdealGeometry2
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
#include "CondFormats/Alignment/interface/AlignmentErrorsExtended.h"
#include "CondFormats/AlignmentRecord/interface/TrackerAlignmentRcd.h"
#include "CondFormats/AlignmentRecord/interface/TrackerAlignmentErrorExtendedRcd.h"

#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "Geometry/Records/interface/TrackerTopologyRcd.h"

#include "Alignment/SurveyAnalysis/interface/SurveyDataReader.h"
//
//
// class declaration
//

class TestIdealGeometry2 : public edm::EDAnalyzer {

  typedef SurveyDataReader::MapType    MapType;
  typedef SurveyDataReader::PairType   PairType;
  typedef SurveyDataReader::MapTypeOr  MapTypeOr;
  typedef SurveyDataReader::PairTypeOr PairTypeOr;

public:
  explicit TestIdealGeometry2( const edm::ParameterSet& );
  ~TestIdealGeometry2();
  
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
TestIdealGeometry2::TestIdealGeometry2( const edm::ParameterSet& iConfig ) :
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


TestIdealGeometry2::~TestIdealGeometry2()
{ 
  
  theTree->Write();
  theFile->Close();
  
}


void
TestIdealGeometry2::analyze( const edm::Event& iEvent, const edm::EventSetup& iSetup )
{
  //Retrieve tracker topology from geometry
  edm::ESHandle<TrackerTopology> tTopoHandle;
  iSetup.get<TrackerTopologyRcd>().get(tTopoHandle);
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
  // edm::ESHandle<TrackerGeometry> trackerGeometry;
  // iSetup.get<TrackerDigiGeometryRecord>().get( trackerGeometry );

  // Retrieve alignment[Error]s from DBase
  edm::ESHandle<Alignments> alignments;
  iSetup.get<TrackerAlignmentRcd>().get( alignments );
  edm::ESHandle<AlignmentErrorsExtended> alignmentErrors;
  iSetup.get<TrackerAlignmentErrorExtendedRcd>().get( alignmentErrors );
  int countDet = 0;

  // Now loop on detector units, and store difference position and orientation w.r.t. survey
  
   for ( std::vector<AlignTransform>::const_iterator iGeomDet = alignments->m_align.begin();
		iGeomDet != alignments->m_align.end(); iGeomDet++ )
	{
          
          if (countDet == 0) countDet = countDet + 3;
	  unsigned int comparisonVect[6] = {0,0,0,0,0,0};
	  
	  DetId thisId( (*iGeomDet).rawId() );
	  if (thisId.subdetId() == int(StripSubdetector::TIB)) {
	    
	    comparisonVect[0] = int(StripSubdetector::TIB);
	    
	    comparisonVect[1] = tTopo->tibLayer(thisId);  
            if (comparisonVect[1] < 3) {countDet--;} else {countDet = countDet - 3;}
	    std::vector<unsigned int> theString = tTopo->tibStringInfo(thisId);
	    comparisonVect[2] = theString[0];
	    comparisonVect[3] = theString[1];
	    comparisonVect[4] = theString[2];
	    comparisonVect[5] = tTopo->tibModule(thisId);
	    
	  } else if (thisId.subdetId() == int(StripSubdetector::TID)) {
	    
	    comparisonVect[0] = int(StripSubdetector::TID);
	    
	    comparisonVect[1] = tTopo->tidSide(thisId);
	    comparisonVect[2] = tTopo->tidWheel(thisId);
	    comparisonVect[3] = tTopo->tidRing(thisId); 
            if (comparisonVect[3] < 3) {countDet--;} else {countDet = countDet - 3;}
	    std::vector<unsigned int> theModule = tTopo->tidModuleInfo(thisId);
	    comparisonVect[4] = theModule[0];
	    comparisonVect[5] = theModule[1];
	    
	  }
	  
	  if (countDet == 0) { // Store only r-phi for double-sided modules

	    for ( MapTypeOr::const_iterator it = theSurveyMap.begin(); it != theSurveyMap.end(); it++ ) {
	      const std::vector<int>& locPos = (it)->first;
	      const align::Scalars& align_params = (it)->second;
	      
	      if (locPos[0] == int(comparisonVect[0]) &&
		  locPos[1] == int(comparisonVect[1]) &&
		  locPos[2] == int(comparisonVect[2]) &&
		  locPos[3] == int(comparisonVect[3]) &&
		  locPos[4] == int(comparisonVect[4]) &&
		  locPos[5] == int(comparisonVect[5]) ) {
	      
		const CLHEP::HepRotation& rot = (*iGeomDet).rotation();
		align::RotationType rotation( rot.xx(), rot.xy(), rot.xz(),
					      rot.yx(), rot.yy(), rot.yz(),
					      rot.zx(), rot.zy(), rot.zz() );
		
		Id_     = (*iGeomDet).rawId();
		// cout << "DetId = " << Id_ << " " << endl;
		// cout << "DetId decodified = " << comparisonVect[0] << " " << comparisonVect[1] << " " << comparisonVect[2] << " " << comparisonVect[3] << " " << comparisonVect[4] << " " << comparisonVect[5] << endl;	      
		dx_      = (*iGeomDet).translation().x() - align_params[0];
		// cout << "X pos : TRACKER_ALIGN = " << std::fixed << std::setprecision(2) << (*iGeomDet).translation().x() << " / IDEAL RICCARDO = " << align_params[0] << endl; 
		dy_      = (*iGeomDet).translation().y() - align_params[1];
		// cout << "Y pos : TRACKER_ALIGN = " << std::fixed << std::setprecision(2) << (*iGeomDet).translation().y() << " / IDEAL RICCARDO = " << align_params[1] << endl;
		dz_      = (*iGeomDet).translation().z() - align_params[2];
		// cout << "Z pos : TRACKER_ALIGN = " << std::fixed << std::setprecision(2) << (*iGeomDet).translation().z() << " / IDEAL RICCARDO = " << align_params[2] << endl;
		// cout << "SPATIAL DISTANCE = " << std::fixed << std::setprecision(3) << sqrt(pow(dx_,2)+pow(dy_,2)+pow(dz_,2)) << endl;
		dtx_     = rotation.xx() - align_params[6];
		// cout << "Trans vect X : TRACKER_ALIGN = " << std::fixed << std::setprecision(3) << rotation.xx() << " / IDEAL RICCARDO = " << align_params[6] << endl;
		dty_     = rotation.xy() - align_params[7];
		// cout << "Trans vect Y : TRACKER_ALIGN = " << std::fixed << std::setprecision(3) << rotation.xy() << " / IDEAL RICCARDO = " << align_params[7] << endl;
		dtz_     = rotation.xz() - align_params[8];
		// cout << "Trans vect Z : TRACKER_ALIGN = " << std::fixed << std::setprecision(3) << rotation.xz() << " / IDEAL RICCARDO = " << align_params[8] << endl; 	
		dkx_     = rotation.yx() - align_params[9];
		// cout << "Long vect X : TRACKER_ALIGN = " << std::fixed << std::setprecision(3) << rotation.yx() << " / IDEAL RICCARDO = " << align_params[9] << endl;
		dky_     = rotation.yy() - align_params[10];
		// cout << "Long vect Y : TRACKER_ALIGN = " << std::fixed << std::setprecision(3) << rotation.yy() << " / IDEAL RICCARDO = " << align_params[10] << endl;
		dkz_     = rotation.yz() - align_params[11];
		// cout << "Long vect Z : TRACKER_ALIGN = " << std::fixed << std::setprecision(3) << rotation.yz() << " / IDEAL RICCARDO = " << align_params[11] << endl;
		dnx_     = rotation.zx() - align_params[3];
		// cout << "Norm vect X : TRACKER_ALIGN = " << std::fixed << std::setprecision(3) << rotation.zx() << " / IDEAL RICCARDO = " << align_params[3] << endl;
		dny_     = rotation.zy() - align_params[4];
		// cout << "Norm vect Y : TRACKER_ALIGN = " << std::fixed << std::setprecision(3) << rotation.zy() << " / IDEAL RICCARDO = " << align_params[4] << endl;
		dnz_     = rotation.zz() - align_params[5];
		// cout << "Norm vect Z : TRACKER_ALIGN = " << std::fixed << std::setprecision(3) << rotation.zz() << " / IDEAL RICCARDO = " << align_params[5] << endl; 
		theTree->Fill();
	      }
	    }
	  }	  
	} 
   
  edm::LogInfo("TrackerAlignment") << "Done!";

}

//define this as a plug-in
DEFINE_FWK_MODULE(TestIdealGeometry2);
