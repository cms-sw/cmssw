// COCOA class implementation file
// Id:  Measurement.C
// CAT: Model
// ---------------------------------------------------------------------------
// History: v1.0 
// Authors:
//   Pedro Arce

#include "Alignment/CocoaModel/interface/MeasurementDistancemeter3dim.h"
#include "Alignment/CocoaModel/interface/LightRay.h"
#include "Alignment/CocoaModel/interface/OpticalObject.h"
#include <iostream>
#include <iomanip>
#include <cstdlib>
#ifdef COCOA_VIS
#include "Alignment/CocoaVisMgr/interface/ALIVRMLMgr.h"
#include "Alignment/IgCocoaFileWriter/interface/IgCocoaFileMgr.h"
#include "Alignment/IgCocoaFileWriter/interface/ALIVisLightPath.h"
#endif
#include "Alignment/CocoaUtilities/interface/GlobalOptionMgr.h"


//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
//@@ calculate the simulated value propagating the light ray through the OptO that take part in the Measurement
//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
void MeasurementDistancemeter3dim::calculateSimulatedValue( ALIbool firstTime ) 
{
 
  if( ALIUtils::debug >= 2) printStartCalculateSimulatedValue( this ); // important for Examples/FakeMeas

  //---------- Loop list of OptO that take part in measurement
  std::vector<OpticalObject*>::const_iterator vocite =  OptOList().begin();
  if( ALIUtils::debug >= 5) std::cout  << "OptOList size" <<OptOList().size() << std::endl;

  //----- Check that there are only two measurements that are 'distance_target' and 'distancemeter3dim'
  ALIbool right_objects = 0;
  if( OptOList().size() == 2 ) {
    if( (*vocite)->type() == "distance_target"
	&& ( (*(vocite+1))->type() == "distancemeter3dim" ) ) { 
      right_objects = 1;
    } 
  }
  if( !right_objects ) {
    std::cerr << "!!! ERROR in MeasurementDistancemeter3dim: " << name() << " There should only be two objects of type 'distance_target' and 'distancemeter3dim' " << std::endl;
 std::cerr	 << " 1st " << (*vocite)->name() << " 2nd " << (*vocite+1)->name()  << std::endl;
    std::cerr << " 1st " << (*vocite)->type() << " 2nd " << (*vocite+1)->type() << std::endl;

    DumpBadOrderOptOs();
    std::exception();
  }     

#ifdef COCOA_VIS
  ALIVisLightPath* vispath = 0;
  if( ALIUtils::getFirstTime() ) {
    GlobalOptionMgr* gomgr = GlobalOptionMgr::getInstance();
    if(gomgr->GlobalOptions()["VisWriteIguana"] > 1) {
      vispath = IgCocoaFileMgr::getInstance().newLightPath( name() );
    } 
  }
#endif

  ALIuint isec = 0;  //security variable to check OptOList().size()
  for( vocite = OptOList().begin(); vocite != OptOList().end(); vocite++) {
    if( ALIUtils::debug >= 2) std::cout << std::endl << "@@@@ LR:OBJECT " << (*vocite)->name() << std::endl;  
    isec ++;

    //---------- Get the behaviour of the object w.r.t the measurement (if it reflects the light, let it traverse it, ...)
    ALIstring behav = getMeasuringBehaviour(vocite);

    //---------- participate in measurement
    LightRay lightray;  //it is not used in this measurement type
    (*vocite)->participateInMeasurement( lightray, *this, behav);

#ifdef COCOA_VIS
    if( ALIUtils::getFirstTime() ) {
      GlobalOptionMgr* gomgr = GlobalOptionMgr::getInstance();
      if(gomgr->GlobalOptions()["VisWriteVRML"] > 1) {
	ALIVRMLMgr::getInstance().addLightPoint( lightray.point() );
	if(ALIUtils::debug >= 5)std::cout << "ALIVRMLMg  addLightPoint " << lightray.point()  << (*vocite)->name() << std::endl;
      }
      if(gomgr->GlobalOptions()["VisWriteIguana"] > 1) {
	vispath->addLightPoint( lightray.point(), *vocite );
      }
    }
#endif    

    if ( isec > OptOList().size() ) {
      std::cerr << "ERROR DE PROGRAMACION EN GetSimulatedValue" << std::endl;
      std::exception();
    }
  }
  
  if(ALIUtils::debug >= 5) std::cout << "end calculateSimulatedValue" <<std::endl;
  
}


//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
//@@ You input 2 numbers after 'DISTANCEMETER':
//@@  
//@@ set the conversion factor from mV to mm
//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
void MeasurementDistancemeter3dim::setConversionFactor( const std::vector<ALIstring>& wordlist ) 
{
  //--------- Check that the format is OK
  if(wordlist.size() == 1) return; 
  if( wordlist.size() != 4
    || !ALIUtils::IsNumber(wordlist[1]) || !ALIUtils::IsNumber(wordlist[3]) 
    || wordlist[2] != ALIstring("+-") ){  
    std::cerr << "!! Distancemeter3dim Measurement setConversionFactor: WRONG FORMAT "<<  std::endl 
	 << "It should be: DISTANCEMETER3DIM factor +- error " << std::endl 
	 << "It is: ";
    ALIUtils::dumpVS( wordlist, " ", std::cerr );
    std::exception();
  }
  theFactor = atof(wordlist[1].c_str());
  //------ correct by dimension of value of tiltmeter
  GlobalOptionMgr* gomgr = GlobalOptionMgr::getInstance();
  ALIint dimfac = ALIint( gomgr->GlobalOptions()[ ALIstring("distancemeter_meas_value_dimension") ] );
  if( dimfac == 0 ) {
    theFactor *= 1.;
  } else if( dimfac == 1 ) {
    theFactor *= 1.E-3;
  } else if( dimfac == 2 ) {
    theFactor *= 1.E-6;
  } else {
    std::cerr << " !!!EXITING: error in global option distancemeter3dim_meas_value_dimension, it can only take values 0,1,2, not " << dimfac;
    std::exception();
  }
  theFactorSigma = atof(wordlist[3].c_str());

}


//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
//@@ Value is given in mV: convert to mm 
//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
void MeasurementDistancemeter3dim::correctValueAndSigma()
{
  ALIdouble val = value()[0];
  ALIdouble sig = sigma()[0];
  if(ALIUtils::debug >= 4) std::cout << "MeasurementDistancemeter3dim::correctValueAndSigma: old value" << val << " +- " << sig << std::endl;

  //- std::cout << "FACTOR " << theFactor << "correct " << val << " "  << thePedestal << std::endl; 
  val *= theFactor; 
  //-------- Do not correct the sigma!!!!
  //-  sig *= theFactor; 
  if(ALIUtils::debug >= 4) std::cout << "MeasuremenDistancemeter3dim::correctValueAndSigma: new value " << val << " +- " << sig << std::endl;
  setValue( 0, val );
  setSigma( 0, sig );

}

