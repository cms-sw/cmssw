//-------------------------------------------------
//
//   Class: DTBtiTrig
//
//   Description: BTI Trigger Data
//
//
//   Author List:
//   C. Grandi
//   Modifications: 
//
//
//--------------------------------------------------

//#include "Utilities/Configuration/interface/Architecture.h"

//-----------------------
// This Class's Header --
//-----------------------
#include "L1Trigger/DTBti/interface/DTBtiChip.h"
#include "L1Trigger/DTBti/interface/DTBtiTrig.h"

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include <DataFormats/DTDigi/interface/DTDigiCollection.h>
#include "L1Trigger/DTBti/interface/DTBtiChip.h"
//---------------
// C++ Headers --
//---------------

//----------------
// Constructors --
//----------------
DTBtiTrig::DTBtiTrig() {

  // reserve the appropriate amount of space for vectors
  _digi.reserve(4);
  clear();

}

DTBtiTrig::DTBtiTrig(DTBtiChip* tparent, int step) :
                                             _tparent(tparent)      {

  // reserve the appropriate amount of space for vectors
  _digi.reserve(4);
  clear();

  // data part of the trigger
  _data.setStep(step);
  _data.setParent(tparent->id());

}

DTBtiTrig::DTBtiTrig(DTBtiChip* tparent, 
                             int code, int K, int X, int step, int eq) : 
                             _tparent(tparent) {

  // reserve the appropriate amount of space for vectors
  _digi.reserve(4);

  // data part of the trigger
  _data.setStep(step);
  _data.setParent(tparent->id());
  _data.setCode(code);
  _data.setK(K);
  _data.setX(X);
  _data.setEq(eq);

}

DTBtiTrig::DTBtiTrig(DTBtiChip* tparent, 
                             int code, int K, int X, int step, int eq, int str, float* Keq) : 
                             _tparent(tparent) {

  // reserve the appropriate amount of space for vectors
  _digi.reserve(4);

  // data part of the trigger
  _data.setStep(step);
  _data.setParent(tparent->id());
  _data.setCode(code);
  _data.setK(K);
  _data.setX(X);
  _data.setEq(eq);
  _data.setStrobe(str);
  _data.setKeq(0,Keq[0]);
  _data.setKeq(1,Keq[1]);
  _data.setKeq(2,Keq[2]);
  _data.setKeq(3,Keq[3]);
  _data.setKeq(4,Keq[4]);
  _data.setKeq(5,Keq[5]);


}

DTBtiTrig::DTBtiTrig(DTBtiChip* parent, DTBtiTrigData data) :
                             _tparent(parent), _data(data) {

  // reserve the appropriate amount of space for vectors
  _digi.reserve(4);

}

//--------------
// Destructor --
//--------------
DTBtiTrig::~DTBtiTrig() {
}
