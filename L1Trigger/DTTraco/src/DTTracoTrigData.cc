//-------------------------------------------------
//
//   Class: DTTracoTrigData
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

//-----------------------
// This Class's Header --
//-----------------------
#include "L1Trigger/DTTraco/interface/DTTracoTrigData.h"

//-------------------------------
// Collaborating Class Headers --
//-------------------------------

//---------------
// C++ Headers --
//---------------
#include <iostream>

using namespace std;

//----------------
// Constructors --
//----------------
DTTracoTrigData::DTTracoTrigData(DTTracoId tracoid, int step)
                                             : _tracoid(tracoid), _step(step) {
  clear();
}


//--------------
// Operations --
//--------------

int
DTTracoTrigData::qdec() const {
  // This is consistent with memo
  if (_codeIn==8 && _codeOut==8) return 6; // HH
  if (_codeIn==8 && _codeOut==0) return 2; // Hinner
  if (_codeIn==0 && _codeOut==8) return 3; // Houter
  if (_codeIn >0 && _codeOut==8) return 5; // LH
  if (_codeIn==8 && _codeOut> 0) return 5; // HL
  if (_codeIn> 0 && _codeOut> 0) return 4; // LL
  if (_codeIn> 0 && _codeOut==0) return 0; // Linner
  if (_codeIn==0 && _codeOut> 0) return 1; // Louter
  return 7;                                // null
}

void
DTTracoTrigData::print() const {
  cout << "TRACO Id=" << " ( " << _tracoid.wheel()   ;
  cout                << " , " << _tracoid.station() ;
  cout                << " , " << _tracoid.sector()  ;
  cout                << " # " << _tracoid.traco()   ;
  cout                << " ) " << " step: " << step();

  cout << dec << " code=" << code() << " K=" << K() << " X=" << X();
  cout << dec << " PVcode=" << pvCode() << " PVk=" << pvK() << " qdec=" << qdec();
  cout << " qdec=" << qdec();
  cout << hex << " psiR=" << psiR() << "  DeltaPsiR=" << DeltaPsiR() << dec << endl;
  if(isFirst())
    cout << " I trk"; 
  if(!isFirst())
    cout << " II trk"; 
  cout << " (";
  if(!pvCorr()) cout << "NOT ";
  cout << "correlated)" << endl;
}
