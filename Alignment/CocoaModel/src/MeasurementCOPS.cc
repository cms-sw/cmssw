// COCOA class implementation file
// Id:  Measurement.C
// CAT: Model
// ---------------------------------------------------------------------------
// History: v1.0 
// Authors:
//   Pedro Arce

#include "Alignment/CocoaModel/interface/MeasurementCOPS.h"
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
void MeasurementCOPS::calculateSimulatedValue( ALIbool firstTime ) 
{
 
  if( ALIUtils::debug >= 2) printStartCalculateSimulatedValue( this ); // important for Examples/FakeMeas

  //---------- Create light ray
  LightRay* lightray = new LightRay;

  //---------- Define types of OptO that may take part in the Measurement
  ALIuint isec = 0;  //security variable to check OptOList().size()

  //---------- Loop list of OptO that take part in measurement
  std::vector<OpticalObject*>::const_iterator vocite =  OptOList().begin();
  if( ALIUtils::debug >= 5) std::cout  << "OptOList size" <<OptOList().size() << std::endl;

  //----- Check that first object is 'Xlaser' 
  if( (*vocite)->type() != "Xlaser" ) { 
    std::cerr << "!!ERROR MeasurementCOPS: first Optical object should be 'Xlaser'" << std::endl;
    DumpBadOrderOptOs();
    exit(1);
  }     

  //---------- Check that last object is a COPS Sensor (that makes measuremnt and kill the lightray)
  if( ( *(OptOList().end() -1) )->type() != "COPS" ) { 
    std::cerr << "!!ERROR MeasurementCOPS: last Optical object should be 'COPS'" << std::endl;
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
  std::cout << (vocite == OptOList().end()) << " vocite " << (*vocite)->name() << std::endl;
  while( vocite != OptOList().end() ) {
    if( ALIUtils::debug >= -2) std::cout << std::endl << "@@@@ LR:OBJECT " << (*vocite)->name() << std::endl;  
    isec ++;

    //---------- Get the behaviour of the object w.r.t the measurement (if it reflects the light, let it traverse it, ...)
    ALIstring behav = getMeasuringBehaviour(vocite);

    if( lightray ) {
      (*vocite)->participateInMeasurement( *lightray, *this, behav );
#ifdef COCOA_VIS
      if( ALIUtils::getFirstTime() ) {
	GlobalOptionMgr* gomgr = GlobalOptionMgr::getInstance();
	if(gomgr->GlobalOptions()["VisWriteVRML"] > 1) {
	  ALIVRMLMgr::getInstance().addLightPoint( lightray->point());
	  //	  std::cout << "ALIVRMLMg  addLightPoint " << lightray->point() << (*vocite)->name() << std::endl;
	}
	if(gomgr->GlobalOptions()["VisWriteIguana"] > 1) {
	  vispath->addLightPoint( lightray->point(), *vocite  );
	}
      }
#endif
    } else {
      std::cerr << "!! Last object is not Sensor 2D in measurement " << name() << std::endl;
      DumpBadOrderOptOs();
      exit(1);
    }

    vocite++;
    if ( isec > OptOList().size() ) {
      std::cerr << "ERROR DE PROGRAMACION EN GetSimulatedValue" << std::endl;
      exit(5);
    }
  }

 
  if(ALIUtils::debug >= 9) std::cout << "end calculateSimulatedValue" <<std::endl;
  
  delete lightray;
}



//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
//@@ You input 8 numbers after 'TILMETER':
//@@  
//@@ set the conversion factor from mV to mrad and the pedestal 
//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
void MeasurementCOPS::setConversionFactor( const std::vector<ALIstring>& wordlist ) 
{
  //--------- Set it to 0 
  ALIuint ii;
  for( ii = 0; ii < dim(); ii++) {
    theDisplace[ii] = 0.; 
  }

  //--------- Check that the format is OK
  if(wordlist.size() == 1) return; 
  if( wordlist.size() != 3 
      || !ALIUtils::IsNumber(wordlist[1]) || !ALIUtils::IsNumber(wordlist[2])
      || !ALIUtils::IsNumber(wordlist[3]) || !ALIUtils::IsNumber(wordlist[4]) ) {
    std::cerr << "!! SensorCOPS Measurement setConversionFactor: WRONG FORMAT "<<  std::endl 
	 << "It should be: SENSOR2D displace_U displace_D displace_L displace_R " << std::endl 
	 << "It is: ";
    ALIUtils::dumpVS( wordlist, " ", std::cerr );
    exit(1);
  }

  for( ii = 0; ii < dim(); ii++) {
    theDisplace[ii] = atof(wordlist[ii+1].c_str())* valueDimensionFactor();
  }

}



void MeasurementCOPS::correctValueAndSigma()
{
   //---------- Make  displacement
  ALIuint ii;
  for( ii = 0; ii < dim(); ii++) { 
    ALIdouble val = value()[ii];
    val += theDisplace[ii];
    if(ALIUtils::debug >= 9) std::cout << "MeasurementCOPS::correctValueAndSigma: old value X " << value()[ii] << " new " << val << " +- " << std::endl;
    setValue( ii, val );
  }

}

