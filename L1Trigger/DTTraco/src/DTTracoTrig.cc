//-------------------------------------------------
//
//   Class: DTTracoTrig
//
//   Description: TRACO Trigger Data
//
//
//   Author List:
//   C. Grandi
//   Modifications: 
//   22/VI/04 SV: last trigger code update
//
//
//--------------------------------------------------

// #include "Utilities/Configuration/interface/Architecture.h"

//-----------------------
// This Class's Header --
//-----------------------
#include "L1Trigger/DTTraco/interface/DTTracoTrig.h"

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "L1Trigger/DTBti/interface/DTBtiTrigData.h"
#include "L1Trigger/DTTraco/interface/DTTracoChip.h"

//---------------
// C++ Headers --
//---------------

using namespace std;

//----------------
// Constructors --
//----------------
DTTracoTrig::DTTracoTrig() {

  // reserve the appropriate amount of space for vectors
  _btitrig.reserve(2);
  clear();

}

DTTracoTrig::DTTracoTrig(DTTracoChip* tparent, int step) : 
                                                   _tparent(tparent)     {

  // reserve the appropriate amount of space for vectors
  _btitrig.reserve(2);
  clear();

  // data part of the trigger
  _data.setStep(step);
  _data.setParent(tparent->id());

}

DTTracoTrig::DTTracoTrig(DTTracoChip* parent , 
                                 DTTracoTrigData data) : 
                                 _tparent(parent), _data(data) {

  // reserve the appropriate amount of space for vectors
  _btitrig.reserve(2);

}

//--------------
// Destructor --
//--------------
DTTracoTrig::~DTTracoTrig() {
}


bool
DTTracoTrig::operator == (const DTTracoTrig& tt) const {
  if(qdec()==7 && tt.qdec()==7)
    return true;

  if ( !(  isFirst()     == tt.isFirst())     ||
       !(  pvK()         == tt.pvK())         ||
       (  (fmod(double(pvCode()),8.)==0) ^ (fmod(double(tt.pvCode()),8.)==0) )  ||
       !(  pvCorr()      == tt.pvCorr())      ||
       !(  psiR()        == tt.psiR())        ||
       !(  DeltaPsiR()   == tt.DeltaPsiR())   ||
       !(  qdec()        == tt.qdec())        ||
       !(  data().pvIO() == tt.data().pvIO())
                                                        ){


    cout<<"fs:"<<isFirst() <<","<< tt.isFirst() <<endl;
    cout<<"pvCode:"<<pvCode()<<","<<tt.pvCode()<<endl;
    cout<<"pvK:"<< pvK() <<","<<tt.pvK()<<endl;
    cout<<"pvCorr:"<<pvCorr()<<","<<tt.pvCorr()<<endl;
    cout<<"psiR:"<<psiR()<<","<<tt.psiR()<<endl;
    cout<<"DeltaPsiR:"<<DeltaPsiR()<<","<<tt.DeltaPsiR()<<endl;
    cout<<"qdec:"<<qdec()<<","<<tt.qdec()<<endl;
    cout<<"data().pvIO:"<<data().pvIO()<<","<<tt.data().pvIO()<<endl;

     return false;
  }
  return true;
}
