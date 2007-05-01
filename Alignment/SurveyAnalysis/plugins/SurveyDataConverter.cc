// System
#include <string>

// Framework
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

// Conditions database
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"

// Geometry
#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "DataFormats/MuonDetId/interface/CSCDetId.h"
#include "Geometry/DTGeometryBuilder/src/DTGeometryBuilderFromDDD.h"
#include "Geometry/CSCGeometryBuilder/src/CSCGeometryBuilderFromDDD.h"
#include "DataFormats/Math/interface/Vector3D.h"

// Alignment
#include "CondFormats/Alignment/interface/Alignments.h"
#include "CondFormats/Alignment/interface/AlignmentErrors.h"
#include "CondFormats/Alignment/interface/AlignmentSorter.h"
#include "Alignment/CommonAlignment/interface/AlignableObjectId.h"
#include "Alignment/MuonAlignment/interface/AlignableMuon.h"
#include "Alignment/MuonAlignment/interface/AlignableDTBarrel.h"
#include "Alignment/TrackerAlignment/interface/TrackerAlignment.h"

// Conditions database
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"

// Geometry
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeomBuilderFromGeometricDet.h"
#include "Geometry/TrackingGeometryAligner/interface/GeometryAligner.h"
#include "CondFormats/DataRecord/interface/TrackerAlignmentRcd.h"
#include "CondFormats/DataRecord/interface/TrackerAlignmentErrorRcd.h"
#include "Geometry/TrackerGeometryBuilder/interface/StripGeomDetUnit.h"
#include "Geometry/TrackerGeometryBuilder/interface/StripGeomDetType.h"
#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"
#include "DataFormats/SiStripDetId/interface/SiStripDetId.h"

#include "Alignment/SurveyAnalysis/interface/SurveyDataReader.h"
#include "Alignment/SurveyAnalysis/plugins/SurveyDataConverter.h"


//__________________________________________________________________________________________________
SurveyDataConverter::SurveyDataConverter(const edm::ParameterSet& iConfig) :
  theParameterSet( iConfig )
{  
}

//__________________________________________________________________________________________________
void SurveyDataConverter::analyze( const edm::Event& iEvent, const edm::EventSetup& iSetup )
{
  
  edm::LogInfo("SurveyDataConverter") << "Analyzer called";

  // Read in the information from the text files
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
    dataReader.readFile( textFileNames[ii], fileType[ii] );
  }

  // Get info and map
  const MapType mapIdToInfo = dataReader.detIdMap();
  std::cout << "DATA HAS BEEN READ INTO THE MAP" << std::endl;
  std::cout << "DATA HAS BEEN CONVERTED IN ALIGNABLE COORDINATES" << std::endl;  

  TrackerAlignment tr_align(iSetup);
  this->testapplyAllSurveyInfo(tr_align, mapIdToInfo);
  tr_align.saveToDB();
}

//___________________________________
//
void SurveyDataConverter::testapplyAllSurveyInfo( TrackerAlignment& tr_align, MapType map ){

	std::cout << "Test apply info: " << std::endl;
	std::string whichAPE = theParameterSet.getUntrackedParameter<std::string>("APEapplied","useFixedValue");
        double APEshift = theParameterSet.getParameter<double>( "APEshift" );
	for ( MapType::const_iterator it = map.begin(); it != map.end(); it++){
		int id = (it)->first;
		std::vector<float> align_params = (it)->second;

		std::vector<float> translations;
		translations.push_back(align_params[0]); 
		translations.push_back(align_params[1]); 
		translations.push_back(align_params[2]);

                std::vector<double> APEvector;
                if ( whichAPE == "useFullCorrectionSize" ) {
		  APEvector.push_back(align_params[0]); 
		  APEvector.push_back(align_params[1]); 
		  APEvector.push_back(align_params[2]);
                } else {
                  APEvector.push_back(APEshift); 
		  APEvector.push_back(APEshift); 
		  APEvector.push_back(APEshift);
		}

		RotationType rotation(align_params[3], align_params[4], align_params[5],
				      align_params[6], align_params[7], align_params[8],
				      align_params[9], align_params[10], align_params[11]);

	        tr_align.moveAlignableTIBTIDs(id, translations, rotation, APEvector);
	}
}

// Plug in to framework

#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(SurveyDataConverter);
