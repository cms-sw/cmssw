//-------------------------------------------------
//
//   Class: DTBtiHit
//
//   Description: A class for hits in a drift cell
//
//
//   Author List:
//   C. Grandi
//   Modifications: 
//   S. Vanini
//   17/V/04  SV: tdrift in tdc units, phase is included!!
//   22/VI/04 SV: last trigger code update
//   05/II/07 SV: move setuptime to BtiCard
//--------------------------------------------------

//#include "Utilities/Configuration/interface/Architecture.h"

//-----------------------
// This Class's Header --
//-----------------------
#include "L1Trigger/DTBti/interface/DTBtiHit.h"

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include <DataFormats/DTDigi/interface/DTDigiCollection.h>
//#include "Utilities/UI/interface/SimpleConfigurable.h"
//#include "Utilities/Notification/interface/Singleton.h"
//#include "TestBeams/DTBXAnalysis/interface/DTBXCalibration.h"
//---------------
// C++ Headers --
//---------------

// Internal clock (time of a step: 12.5 ns or 16 tdc counts ==> 80 MHz)
const float DTBtiHit::_stepTime = 12.5;
const float DTBtiHit::_stepTimeTdc = 16.;
//const float DTBtiHit::_stepTime = 6.25;

//SV only for TestBeam version
//string DTBtiHit::t0envFlag = SimpleConfigurable<string>( " ",
//                                  "TestBeams:DTBXAnalysis:T0SetUpFlag" );

//----------------
// Constructors --
//----------------

DTBtiHit::DTBtiHit(const DTDigi* hitdigi, DTConfigBti* config) :
  _hitdigi(hitdigi), _config(config) {

  //SV tdcunits 11V04: time in tdc units! setup time too!
  _curTime = hitdigi->countsTDC(); //@@ only DT>0

/*
  // *** ATTENTION FOR RUNNING TESTBEAM DATA ***
  // SV 26/IX/03 if digi are corrected with wire t0s in DTBXDigitizer 
  // tdrift values must be corrected with  t0 mean instead because
  // trigger hardware don't see cable length !
  _curTime = hitdigi->countsTDC();

  if( t0envFlag != " " ){
    DTBXCalibration* calibration = Singleton<DTBXCalibration>::instance();
    //the following for digitization procedure.... see DTBXDigitizer
    int bmax = calibration->bitMax();
    if ( _curTime >= ( bmax + calibration->recMin() ) )
      _curTime -= bmax;

    //SV tdcunits 11V04: add t0 and trig to have raw time in tdcunits
    _curTime += (  calibration->t0( hitdigi->slayer(), hitdigi->layer(), hitdigi->wire() )
                 + calibration->tTrig() );

    //tdc clocks in 16 units
    int delay16 = int( ( calibration->t0mean(hitdigi->slayer()) +
                         calibration->tTrig() )
                       /_stepTimeTdc );

    //bti clocks of 16 tdc units
    _clockTime = (int)( (_curTime +
          _config->SetupTime())/_stepTimeTdc) - delay16;
}
   
*/
  //bti clocks of 16 tdc units : first clock is number 1!
  //_clockTime = (int)( (_curTime + _config->SetupTime()) / _stepTimeTdc ) + 1;
  _clockTime = (int)( _curTime  / _stepTimeTdc ) + 1;

}

DTBtiHit::DTBtiHit(int clockTime, DTConfigBti* config) :
  _config(config) {
  _clockTime = clockTime;
  _hitdigi = 0;
  _curTime = 4000;
}

DTBtiHit::DTBtiHit(const DTBtiHit& hit) :
  _hitdigi(hit._hitdigi), _config(hit._config), _curTime(hit._curTime), 
  _clockTime(hit._clockTime) {
}

//--------------
// Destructor --
//--------------
DTBtiHit::~DTBtiHit() {
}

//--------------
// Operations --
//--------------

DTBtiHit &
DTBtiHit::operator=(const DTBtiHit& hit){
  if(this != &hit){
    _hitdigi = hit._hitdigi;
    _config = hit._config;
    _curTime = hit._curTime;
    _clockTime = hit._clockTime;
  }
  return *this;
}
