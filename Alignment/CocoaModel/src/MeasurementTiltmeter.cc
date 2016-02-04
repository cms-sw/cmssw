// COCOA class implementation file
// Id:  MeasurementTiltmeter.cc
// CAT: Model
// ---------------------------------------------------------------------------
// History: v1.0 
// Authors:
//   Pedro Arce

#include "Alignment/CocoaModel/interface/MeasurementTiltmeter.h"
#include "Alignment/CocoaModel/interface/LightRay.h"
#include "Alignment/CocoaModel/interface/OpticalObject.h"
#include "Alignment/CocoaUtilities/interface/GlobalOptionMgr.h"
#include <iostream>
#include <iomanip>
#include <cstdlib>

//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
//@@ calculate the simulated value propagating the light ray through the OptO that take part in the Measurement
//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
void MeasurementTiltmeter::calculateSimulatedValue( ALIbool firstTime ) 
{
 
  if( ALIUtils::debug >= 2) printStartCalculateSimulatedValue( this ); // important for Examples/FakeMeas

  //--------- Check there is only one OptO of type 'tiltmeter'
  std::vector<OpticalObject*>::const_iterator vocite =  OptOList().begin();
  if( OptOList().size() != 1 ||
      (*vocite)->type() == "distancemeter") {
    std::cerr << "!!! ERROR in MeasurementTiltmeter: " << name() << " There should only be one object of type 'tiltmeter' " << std::endl;
    DumpBadOrderOptOs();
    exit(1);
  }     
  
  //---------- Get the behaviour of the object w.r.t the measurement (if it reflects the light, let it traverse it, ...)
  ALIstring behav = getMeasuringBehaviour(vocite);
  
  //---------- participate in Measurement
  LightRay ll;
  (*vocite)->participateInMeasurement( ll, *this, behav );

  if(ALIUtils::debug >= 5) std::cout << "end calculateSimulatedValue" <<std::endl;
  
}



//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
//@@ You input 8 numbers after 'TILMETER':
//@@  
//@@ set the conversion factor from mV to mrad and the pedestal 
//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
void MeasurementTiltmeter::setConversionFactor( const std::vector<ALIstring>& wordlist ) 
{
  //--------- Check that the format is OK
  if(wordlist.size() == 1) return; 
  if( wordlist.size() != 7
    || !ALIUtils::IsNumber(wordlist[1]) || !ALIUtils::IsNumber(wordlist[3]) 
    || !ALIUtils::IsNumber(wordlist[4]) || !ALIUtils::IsNumber(wordlist[6])
    || wordlist[2] != ALIstring("+-")|| wordlist[5] != ALIstring("+-") ){  
    std::cerr << "!! Tiltmeter Measurement setConversionFactor: WRONG FORMAT "<<  std::endl 
	 << "It should be: TILTEMETER factor +- error constant_term +- error"
	 << (wordlist.size() != 7)
	 <<  !ALIUtils::IsNumber(wordlist[1])  << !ALIUtils::IsNumber(wordlist[3]) 
	 << !ALIUtils::IsNumber(wordlist[4]) << !ALIUtils::IsNumber(wordlist[6])
      //	 << (wordlist[2] != ALIstring("+-")) << (wordlist[5] != ALIstring("+-")) 
<<  std::endl 
	 << "It is: ";
    ALIUtils::dumpVS( wordlist, " ", std::cerr );
    exit(1);
  }
  theFactor = atof(wordlist[1].c_str());

  //------ correct by dimension of value of tiltmeter
  GlobalOptionMgr* gomgr = GlobalOptionMgr::getInstance();
  ALIint dimfac = ALIint( gomgr->GlobalOptions()[ ALIstring("tiltmeter_meas_value_dimension") ] );
  if( dimfac == 0 ) {
    theFactor *= 1.;
  } else if( dimfac == 1 ) {
    theFactor *= 1.E-3;
  } else if( dimfac == 2 ) {
    theFactor *= 1.E-6;
  } else {
    std::cerr << " !!!EXITING: error in global option tiltmeter_meas_value_dimension, it can only take values 0,1,2, not " << dimfac;
  }
  theFactorSigma = atof(wordlist[3].c_str());
  theConstantTerm = atof(wordlist[4].c_str()) * valueDimensionFactor();
  theConstantTermSigma = atof(wordlist[6].c_str()) * sigmaDimensionFactor();
  //-  std::cout << "correctVal  theConstantTerm" <<  theConstantTerm <<  valueDimensionFactor() << std::endl; 
  //----- Change value and sigma to dimensions used in SDF, because constant term and pedestal are in dimensions of SDF
  //-  thePedestal = atof(wordlist[7].c_str()) * valueDimensionFactor();
  //-thePedestalSigma = atof(wordlist[9].c_str()) * sigmaDimensionFactor();
  //  std::cout << "reading thePedestalSigma " << thePedestalSigma  << "= " << wordlist[9] << std::endl;
  //  TILTMETER 458.84 +- 1.58  0. +- 0. 1 +- 0
}


//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
//@@ Value is given in V: substract constant term, and convert to rad (taking into account the error in the constant term)
//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
void MeasurementTiltmeter::correctValueAndSigma()
{
  ALIdouble val = value()[0];
  ALIdouble sig = sigma()[0];
  if(ALIUtils::debug >= 4) std::cout << "MeasurementTiltmeter::correctValueAndSigma: old value" << val << " +- " << sig << std::endl;
   //---------- Substract pedestal
  val -= theConstantTerm;
  //-  sig = sqrt(sig*sig  + thePedestalSigma*thePedestalSigma );
  //-  std::cout << " sigma + pedestalSigma " << sig << " " << thePedestalSigma << std::endl;
  //-if( thePedestal != 0. ) {
    //-    sig += sqrt( sig*sig + val*val*thePedestalSigma*thePedestalSigma/thePedestal/thePedestal );
  //-}
  //---------- Add error in constant term
  sig = sqrt( sig*sig + theConstantTermSigma*theConstantTermSigma );
  //- std::cout << " sigma + costantTermSigma " << sig << " " << theConstantTermSigma << std::endl;

  //---------- Convert to rad 
  //- std::cout << "FACTOR " << theFactor << "correct " << val << " "  << thePedestal << std::endl; 
  val *= theFactor; 
  //-------- Do not correct the sigma!!!!
  //-  sig /= theFactor; 
  if(ALIUtils::debug >= 4) std::cout << "MeasurementTiltmeter::correctValueAndSigma: new value " << val << " +- " << sig << std::endl;
  setValue( 0, val );
  setSigma( 0, sig );

}

