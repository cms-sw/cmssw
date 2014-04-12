// COCOA class implementation file
// Id:  Measurement.C
// CAT: Model
// ---------------------------------------------------------------------------
// History: v1.0 
// Authors:
//   Pedro Arce

#include "Alignment/CocoaModel/interface/MeasurementSensor2D.h"
#include "Alignment/CocoaModel/interface/LightRay.h"
#include "Alignment/CocoaModel/interface/OpticalObject.h"
#ifdef COCOA_VIS
#include "Alignment/CocoaVisMgr/interface/ALIVRMLMgr.h"
#include "Alignment/IgCocoaFileWriter/interface/IgCocoaFileMgr.h"
#include "Alignment/IgCocoaFileWriter/interface/ALIVisLightPath.h"
#endif

#include <iostream>
#include <iomanip>
#include <cstdlib>

//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
//@@ calculate the simulated value propagating the light ray through the OptO that take part in the Measurement
//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
void MeasurementSensor2D::calculateSimulatedValue( ALIbool firstTime ) 
{
 
  if( ALIUtils::debug >= 2) printStartCalculateSimulatedValue( this ); // important
  //---------- Create light ray
  LightRay* lightray = new LightRay;

  int isec = 0;  //security variable to check OptOList().size()

  //---------- Loop list of OptO that take part in measurement
  std::vector<OpticalObject*>::const_iterator vocite =  OptOList().begin();
  //-  if( ALIUtils::debug >= 5) std::cout  << "OptOList size" <<OptOList().size() << std::endl;

  //----- Check that first object is 'laser' or 'source'
  if( (*vocite)->type() != "laser" && (*vocite)->type() != "source" ) { 
    std::cerr << " first Optical object should be 'laser' or 'source'" << std::endl;
    DumpBadOrderOptOs();
    exit(1);
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

  //-  while( (*vocite) !=  *(OptOList().end()) ) {
  while( (vocite) !=  (OptOList().end()) ) {
    if( ALIUtils::debug >= 2) std::cout << std::endl << "@@@@ LR:OBJECT " << (*vocite)->name() << std::endl;  
    isec ++;

    //---------- Get the behaviour of the object w.r.t the measurement (if it reflects the light, let it traverse it, ...)
    ALIstring behav = getMeasuringBehaviour(vocite);

    //---------- Check that last object is a Sensor (that makes measuremnt and kill the lightray)
    if( lightray ) {
      (*vocite)->participateInMeasurement( *lightray, *this, behav );

#ifdef COCOA_VIS
      if( ALIUtils::getFirstTime() ) {
	GlobalOptionMgr* gomgr = GlobalOptionMgr::getInstance();
	if(gomgr->GlobalOptions()["VisWriteVRML"] > 1) {
	  ALIVRMLMgr::getInstance().addLightPoint( lightray->point() );
	  if(ALIUtils::debug >= 5)std::cout << "ALIVRMLMg  addLightPoint " << lightray->point() << (*vocite)->name() << std::endl;
	}
	if(gomgr->GlobalOptions()["VisWriteIguana"] > 1) {
	  vispath->addLightPoint( lightray->point(), (*vocite) );
	}
      }
#endif

    } else {
      std::cerr << "!! Last object is not Sensor 2D in measurement " << name() << std::endl;
      DumpBadOrderOptOs();
      exit(1);
    }

    vocite++;
    if ( isec > ALIint(OptOList().size()) ) {
      std::cerr << "ERROR DE PROGRAMACION EN GetSimulatedValue" << std::endl;
      exit(5);
    }
    //-    lightray.normalizeDirection();
  }

  delete lightray;
 
  if(ALIUtils::debug >= 9) std::cout << "end calculateSimulatedValue" <<std::endl;
  
}



//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
//@@ You input 8 numbers after 'TILMETER':
//@@  
//@@ set the conversion factor from mV to mrad and the pedestal 
//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
void MeasurementSensor2D::setConversionFactor( const std::vector<ALIstring>& wordlist ) 
{
  //--------- Check that the format is OK
  theDisplaceX = 0;
  theDisplaceY = 0;
  theMultiplyX = 1.;
  theMultiplyY = 1.;
  //-  std::cout << " 0 factors for correction X = " << theDisplaceX << " Y " << theDisplaceY << std::endl;
 
  if(wordlist.size() == 1) return; 
  if( (wordlist.size() != 3 && wordlist.size() != 5 )
      || !ALIUtils::IsNumber(wordlist[1]) || !ALIUtils::IsNumber(wordlist[2]) ) {
    std::cerr << "!! Sensor2D Measurement setConversionFactor: WRONG FORMAT "<<  std::endl 
	 << "It should be: SENSOR2D displace_X displace_Y " << std::endl 
	 << "It is: ";
    ALIUtils::dumpVS( wordlist, " ", std::cerr );
    exit(1);
  }
  theDisplaceX = atof(wordlist[1].c_str())* valueDimensionFactor();
  theDisplaceY = atof(wordlist[2].c_str())* valueDimensionFactor();
  //-  std::cout << " factors for correction X = " << theDisplaceX << " Y " << theDisplaceY << std::endl;

  if( wordlist.size() == 5 ) {
    theMultiplyX = atof(wordlist[3].c_str());
    theMultiplyY = atof(wordlist[4].c_str());
  } else {
    theMultiplyX = 1.;
    theMultiplyY = 1.;
  }  
}


//////////////////////////////////////////////////////////////////
void MeasurementSensor2D::correctValueAndSigma()
{
   //---------- Make  displacement
  ALIdouble val = value()[0];
  val += theDisplaceX;
  val *= theMultiplyX;
  //-  std::cout << " theDisplaceX " <<  theDisplaceX << " theMultiplyX " << theMultiplyX << std::endl;
  if(ALIUtils::debug >= 4) std::cout << "MeasurementSensor2D::correctValueAndSigma: " << " old value X " << value()[0] << " new " << val << std::endl;
  setValue( 0, val );

  val = value()[1];
  val += theDisplaceY;
  val *= theMultiplyY;
  if(ALIUtils::debug >= 4) std::cout << "MeasurementSensor2D::correctValueAndSigma: old value Y " << value()[1] << " new " << val << std::endl;
  setValue( 1, val );

}

