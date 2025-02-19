//   COCOA class implementation file
//Id:  GlobalOptionMgr.cc
//CAT: ALIUtils
//
//   History: v1.0 
//   Pedro Arce
#include <fstream>

#include "Alignment/CocoaUtilities/interface/GlobalOptionMgr.h"
#include <iostream>
#include <iomanip>
#include "Alignment/CocoaUtilities/interface/ALIUtils.h"
#include "Alignment/CocoaUtilities/interface/ALIFileIn.h"
#include <cstdlib>

GlobalOptionMgr* GlobalOptionMgr::theInstance = 0;

GlobalOptionMgr* GlobalOptionMgr::getInstance()
{
  if(!theInstance) {
    theInstance = new GlobalOptionMgr;
  }

  return theInstance;
}


//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
void GlobalOptionMgr::setDefaultGlobalOptions()
{
  theGlobalOptions[ ALIstring("report_verbose") ] = 3; 
  theGlobalOptions[ ALIstring("debug_verbose") ] = 0;  
  //  theGlobalOptions[ ALIstring("sparse") ] = 0;  
  theGlobalOptions[ ALIstring("saveMatrices") ] = 1;  
  //  theGlobalOptions[ ALIstring("external_meas") ] = 0;  
  theGlobalOptions[ ALIstring("calcul_type") ] = 0;  
  theGlobalOptions[ ALIstring("length_value_dimension") ] = 0;  
  theGlobalOptions[ ALIstring("length_error_dimension") ] = 0;  
  theGlobalOptions[ ALIstring("angle_value_dimension") ] = 0;  
  theGlobalOptions[ ALIstring("angle_error_dimension") ] = 0;  
  theGlobalOptions[ ALIstring("output_length_value_dimension") ] = 0;  
  theGlobalOptions[ ALIstring("output_length_error_dimension") ] = 0;  
  theGlobalOptions[ ALIstring("output_angle_value_dimension") ] = 0;  
  theGlobalOptions[ ALIstring("output_angle_error_dimension") ] = 0;  
  theGlobalOptions[ ALIstring("checkExtraEntries") ] = 0;  
  theGlobalOptions[ ALIstring("cms_link") ] = 0;  
  theGlobalOptions[ ALIstring("cms_link_halfplanes") ] = 0;  
  theGlobalOptions[ ALIstring("cms_link_method") ] = 0;  
  theGlobalOptions[ ALIstring("range_studies") ] = 0;  
  theGlobalOptions[ ALIstring("histograms") ] = 0;  
  theGlobalOptions[ ALIstring("onlyDeriv") ] = 0; 
  theGlobalOptions[ ALIstring("onlyFirstPropagation") ] = 0;

  theGlobalOptions[ ALIstring("VisWriteVRML") ] = 0;  
  theGlobalOptions[ ALIstring("VisWriteIguana") ] = 0;  
  theGlobalOptions[ ALIstring("VisOnly") ] = 0;  
  theGlobalOptions[ ALIstring("VisWriteOptONames") ] = 1;  
  theGlobalOptions[ ALIstring("VisGlobalRotationX") ] = 0.;  
  theGlobalOptions[ ALIstring("VisGlobalRotationY") ] = 0.;  
  theGlobalOptions[ ALIstring("VisGlobalRotationZ") ] = 0.;  
  theGlobalOptions[ ALIstring("VisScale") ] = 1.;
  theGlobalOptions[ ALIstring("tiltmeter_meas_value_dimension") ] = 0; 
  theGlobalOptions[ ALIstring("distancemeter_meas_value_dimension") ] = 0; 
  theGlobalOptions[ ALIstring("dumpDateInFittedEntries") ] = 0;
  theGlobalOptions[ ALIstring("measurementErrorFromFile") ] = 0;

  theGlobalOptions[ ALIstring("maxNoFitIterations") ] = 50;
  theGlobalOptions[ ALIstring("fitQualityCut") ] = 0.1;
  theGlobalOptions[ ALIstring("relativeFitQualityCut") ] = 1.E-6;

  theGlobalOptions[ ALIstring("maxEvents") ] = 1.E6;

  //dimension factor to multiply the values in the files that give you the deviatin when traversing an ALMY. Files have numbers in microns, so it has to be 1 if 'length_value_dimension 2', 0.001 if 'length_value_dimension 1' (the same for angles)
  theGlobalOptions[ ALIstring("deviffValDimf") ] = 1.;
  theGlobalOptions[ ALIstring("deviffAngDimf") ] = 1.;
  theGlobalOptions[ ALIstring("rotateAroundLocal") ] = 1; 
  theGlobalOptions[ ALIstring("reportOutEntriesByShortName") ] = 0; 
  theGlobalOptions[ ALIstring("reportOutReadValue") ] = 1;
  theGlobalOptions[ ALIstring("reportOutReadSigma") ] = 1;
  theGlobalOptions[ ALIstring("reportOutReadQuality") ] = 1;
  theGlobalOptions[ ALIstring("maxDeviDerivative") ] = 1.E-6;

  theGlobalOptions[ ALIstring("stopAfter1stIteration") ] = 0;
  theGlobalOptions[ ALIstring("calParamInyfMatrix") ] = 0;
  theGlobalOptions[ ALIstring("writeXML") ] = 0;
  theGlobalOptions[ ALIstring("dumpInAllFrames") ] = 0;
  theGlobalOptions[ ALIstring("rootResults") ] = 0;
  theGlobalOptions[ ALIstring("writeDBAlign") ] = 0;
  theGlobalOptions[ ALIstring("writeDBOptAlign") ] = 0;
}

//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
ALIdouble GlobalOptionMgr::getGlobalOption( const ALIstring& sstr ) 
{
  ALIdouble val = 0.;
  //---------- Find Global Option by name
  std::map< ALIstring, ALIdouble, std::less<ALIstring> >::const_iterator msdcite = GlobalOptions().find( sstr ); 

  //---------- Dump Global Option found
  if( ALIUtils::debug >= 6) {
    std::cout << "Global Option " << (*msdcite).first << " = " << (*msdcite).second << std::endl;
  }

  if ( msdcite == GlobalOptions().end() ) {
    //---------- return 0 if GLobal Option not found
    std::cerr << " !!! FATAL ERROR: trying to get the value of an unknown Global Option : " << sstr << std::endl;
    abort();
  } else {
    //---------- return 1 if Global Option found
    //-std::cout << "SSparam" << (*msdcite).first << (*msdcite).second << "len" << OptOList().size() << std::endl;
    //----- set val to Global Option value
    val = (*msdcite).second;
  } 

  return val;

}

//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
ALIint GlobalOptionMgr::getGlobalOptionValue( const ALIstring& sstr, ALIdouble& val ) 
{
  //---------- Find Global Option by name
  std::map< ALIstring, ALIdouble, std::less<ALIstring> >::const_iterator msdcite = GlobalOptions().find( sstr ); 

  //---------- Dump Global Option found
  if( ALIUtils::debug >= 6) {
    std::cout << "Global Option " << (*msdcite).first << " = " << (*msdcite).second << std::endl;
  }

  if ( msdcite == GlobalOptions().end() ) {
    //---------- return 0 if GLobal Option not found
    return 0;
  } else {
    //---------- return 1 if Global Option found
    //-std::cout << "SSparam" << (*msdcite).first << (*msdcite).second << "len" << OptOList().size() << std::endl;
    //----- set val to Global Option value
    val = (*msdcite).second;
    return 1;
  } 

}


//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
void GlobalOptionMgr::setGlobalOption( const ALIstring gopt, const ALIdouble val, ALIFileIn& filein )
{

  if( !setGlobalOption( gopt, val, 0 ) ){
    filein.ErrorInLine();
    std::cerr << "!!! global option not found: " << gopt << std::endl;
    if ( ALIUtils::debug >= 3 ) {
      std::cout << "ALLOWED GLOBAL OPTIONS:" << std::endl;
      std::map< ALIstring, ALIdouble, std::less<ALIstring> >::iterator msdite;
      for ( msdite = theGlobalOptions.begin(); 
	    msdite != theGlobalOptions.end(); msdite++) {
	std::cout << (*msdite).first.c_str() << std::endl;
      }
    }
    exit(2);
  }

}

//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
bool GlobalOptionMgr::setGlobalOption( const ALIstring gopt, const ALIdouble val, bool bExit )
{
  //----- If global option exists: set it to value read
  if ( GlobalOptions().find( gopt ) != GlobalOptions().end() ){
    theGlobalOptions[ gopt ] = val;
    //------ Verbosity global options change static data
    if( gopt == "report_verbose") {
      ALIUtils::setReportVerbosity( ALIint(val) );
    }
    if( gopt == "debug_verbose" ) {
      ALIUtils::setDebugVerbosity( ALIint(val) );
    }
    
    return 1;
    //----- if global option does not exist: error
  } else {
    if( bExit ) {
      std::cerr << "!!! global option not found: " << gopt << std::endl;
      exit(2);
    }
    return 0;
  }
  
}
