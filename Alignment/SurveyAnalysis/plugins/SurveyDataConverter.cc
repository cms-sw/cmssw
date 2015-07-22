
// Framework
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "Alignment/CommonAlignment/interface/AlignableModifier.h"
#include "Alignment/TrackerAlignment/interface/TrackerAlignment.h"
#include "Alignment/TrackerAlignment/interface/TrackerScenarioBuilder.h"
#include "Alignment/CommonAlignment/interface/Alignable.h" 

#include "Alignment/SurveyAnalysis/plugins/SurveyDataConverter.h"

#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "Geometry/Records/interface/TrackerTopologyRcd.h"

//__________________________________________________________________________________________________
SurveyDataConverter::SurveyDataConverter(const edm::ParameterSet& iConfig) :
  theParameterSet( iConfig )
{  
}

//__________________________________________________________________________________________________
void SurveyDataConverter::analyze( const edm::Event& iEvent, const edm::EventSetup& iSetup )
{
  //Retrieve tracker topology from geometry
  edm::ESHandle<TrackerTopology> tTopoHandle;
  iSetup.get<TrackerTopologyRcd>().get(tTopoHandle);
  const TrackerTopology* const tTopo = tTopoHandle.product();
  
  edm::LogInfo("SurveyDataConverter") << "Analyzer called";
  applyfineinfo = theParameterSet.getParameter<bool>("applyFineInfo");
  applycoarseinfo = theParameterSet.getParameter<bool>("applyCoarseInfo");
  adderrors = theParameterSet.getParameter<bool>("applyErrors");

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
    dataReader.readFile( textFileNames[ii], fileType[ii], tTopo );
  }

  // Get info and map
  const MapType& mapIdToInfo = dataReader.detIdMap();
  std::cout << "DATA HAS BEEN READ INTO THE MAP" << std::endl;
  std::cout << "DATA HAS BEEN CONVERTED IN ALIGNABLE COORDINATES" << std::endl;  

  TrackerAlignment tr_align( iSetup );
  if (applycoarseinfo) this->applyCoarseSurveyInfo(tr_align); 
  if (applyfineinfo) this->applyFineSurveyInfo(tr_align, mapIdToInfo);
  if (adderrors) this->applyAPEs(tr_align);
  tr_align.saveToDB();
}

//___________________________________
//
void SurveyDataConverter::applyFineSurveyInfo( TrackerAlignment& tr_align, const MapType& map ){

  std::cout << "Apply fine info: " << std::endl;
	
  for ( MapType::const_iterator it = map.begin(); it != map.end(); it++){

    const align::Scalars& align_params = (it)->second; 
      
    align::Scalars translations; 
    translations.push_back(align_params[0]);  
    translations.push_back(align_params[1]);  
    translations.push_back(align_params[2]); 

    align::RotationType bRotation(align_params[6], align_params[9], align_params[3],
                                  align_params[7], align_params[10], align_params[4],
                                  align_params[8], align_params[11], align_params[5]);

    align::RotationType fRotation(align_params[15], align_params[18], align_params[12],
                                  align_params[16], align_params[19], align_params[13],
                                  align_params[17], align_params[20], align_params[14]);

    // Use "false" for debugging only
    tr_align.moveAlignableTIBTIDs((it)->first, translations, bRotation, fRotation, true);
  }
}

//___________________________________
//
void SurveyDataConverter::applyCoarseSurveyInfo( TrackerAlignment& tr_align ){
        
  std::cout << "Apply coarse info: " << std::endl;
  MisalignScenario = theParameterSet.getParameter<edm::ParameterSet>( "MisalignmentScenario" );

  TrackerScenarioBuilder scenarioBuilder( tr_align.getAlignableTracker() );
  scenarioBuilder.applyScenario( MisalignScenario );
  
}

//___________________________________
//
void SurveyDataConverter::applyAPEs( TrackerAlignment& tr_align ) {
        
  std::cout << "Apply APEs: " << std::endl;
  // Neglect sensor-on-module mounting precision (10 um)
  // Irrelevant given other sizes ..
  std::vector<double> TIBerrors = theParameterSet.getParameter< std::vector<double> >("TIBerrors");
  std::vector<double> TOBerrors = theParameterSet.getParameter< std::vector<double> >("TOBerrors");
  std::vector<double> TIDerrors = theParameterSet.getParameter< std::vector<double> >("TIDerrors"); 
  std::vector<double> TECerrors = theParameterSet.getParameter< std::vector<double> >("TECerrors"); 
        
  if (TIBerrors.size() < 3 || TOBerrors.size() < 4 || TIDerrors.size() < 4 || TECerrors.size() < 4) {
    std::cout << "APE info not valid : please check test/run-converter.cfg" << std::endl;
    return;
  }
         
  AlignableModifier* theModifier = new AlignableModifier();
  AlignableTracker* theAlignableTracker = tr_align.getAlignableTracker() ; 
  align::Alignables::const_iterator iter;

  // TIB
  const align::Alignables& theTIBhb = theAlignableTracker->innerHalfBarrels();
  for (iter = theTIBhb.begin(); iter != theTIBhb.end(); ++iter ) 
    { theModifier->addAlignmentPositionErrorLocal( *iter, TIBerrors.at(0), 
                                                   TIBerrors.at(0), TIBerrors.at(0) ); }
  const align::Alignables& theTIBlayers = theAlignableTracker->innerBarrelLayers();
  for (iter = theTIBlayers.begin(); iter != theTIBlayers.end(); ++iter)
    { theModifier->addAlignmentPositionErrorLocal( *iter, TIBerrors.at(1), 
                                                   TIBerrors.at(1), TIBerrors.at(1) ); }
  const align::Alignables& theTIBgd = theAlignableTracker->innerBarrelGeomDets();
  for (iter = theTIBgd.begin(); iter != theTIBgd.end(); ++iter ) 
    { theModifier->addAlignmentPositionErrorLocal( *iter, TIBerrors.at(2), 
                                                   TIBerrors.at(2), TIBerrors.at(2) ); }

  // TOB
  const align::Alignables& theTOBhb = theAlignableTracker->outerHalfBarrels();
  for (iter = theTOBhb.begin(); iter != theTOBhb.end(); ++iter )
    { theModifier->addAlignmentPositionErrorLocal( *iter, TOBerrors.at(0), 
                                                   TOBerrors.at(0), TOBerrors.at(1) ); }
  const align::Alignables& theTOBrods = theAlignableTracker->outerBarrelRods();
  for (iter = theTOBrods.begin(); iter != theTOBrods.end(); ++iter ) 
    { theModifier->addAlignmentPositionErrorLocal( *iter, TOBerrors.at(2), 
                                                   TOBerrors.at(2), TOBerrors.at(2) ); }
  const align::Alignables& theTOBgd = theAlignableTracker->outerBarrelGeomDets();
  for (iter = theTOBgd.begin(); iter != theTOBgd.end(); ++iter )
    { theModifier->addAlignmentPositionErrorLocal( *iter, TOBerrors.at(3), 
                                                   TOBerrors.at(3), TOBerrors.at(3) ); }

  // TID
  const align::Alignables& theTIDs = theAlignableTracker->TIDs();
  for (iter = theTIDs.begin(); iter != theTIDs.end(); ++iter ) 
    { theModifier->addAlignmentPositionErrorLocal( *iter, TIDerrors.at(0), 
                                                   TIDerrors.at(0), TIDerrors.at(0) ); }
  const align::Alignables& theTIDdiscs = theAlignableTracker->TIDLayers();
  for (iter = theTIDdiscs.begin(); iter != theTIDdiscs.end(); ++iter )
    { theModifier->addAlignmentPositionErrorLocal( *iter, TIDerrors.at(1), 
                                                   TIDerrors.at(1), TIDerrors.at(1) ); }
  const align::Alignables& theTIDrings = theAlignableTracker->TIDRings();
  for (iter = theTIDrings.begin(); iter != theTIDrings.end(); ++iter )
    { theModifier->addAlignmentPositionErrorLocal( *iter, TIDerrors.at(2), 
                                                   TIDerrors.at(2), TIDerrors.at(2) ); } 
  const align::Alignables& theTIDgd = theAlignableTracker->TIDGeomDets();
  for (iter = theTIDgd.begin(); iter != theTIDgd.end(); ++iter )
    { theModifier->addAlignmentPositionErrorLocal( *iter, TIDerrors.at(3), 
                                                   TIDerrors.at(3), TIDerrors.at(3) ); } 

  // TEC
  const align::Alignables& theTECs = theAlignableTracker->endCaps();
  for (iter = theTECs.begin(); iter != theTECs.end(); ++iter ) 
    { theModifier->addAlignmentPositionErrorLocal( *iter, TECerrors.at(0), 
                                                   TECerrors.at(0), TECerrors.at(0) ); } 
  const align::Alignables& theTECdiscs = theAlignableTracker->endcapLayers();
  for (iter = theTECdiscs.begin(); iter != theTECdiscs.end(); ++iter )
    { theModifier->addAlignmentPositionErrorLocal( *iter, TECerrors.at(1), 
                                                   TECerrors.at(1), TECerrors.at(1) ); } 
  const align::Alignables& theTECpetals = theAlignableTracker->endcapPetals();
  for (iter = theTECpetals.begin(); iter != theTECpetals.end(); ++iter ) 
    { theModifier->addAlignmentPositionErrorLocal( *iter, TECerrors.at(2), 
                                                   TECerrors.at(2), TECerrors.at(2) ); }   
  const align::Alignables& theTECgd = theAlignableTracker->endcapGeomDets();
  for (iter = theTECgd.begin(); iter != theTECgd.end(); ++iter )
    { theModifier->addAlignmentPositionErrorLocal( *iter, TECerrors.at(3), 
                                                   TECerrors.at(3), TECerrors.at(3) ); }   
}

DEFINE_FWK_MODULE(SurveyDataConverter);

