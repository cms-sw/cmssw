//using namespace std;
/*
 *  See header file for a description of this class.
 *
 *  $Date: 2007/12/07 15:13:20 $
 *  $Revision: 1.2 $
 *  \author Paolo Ronchese INFN Padova
 *
 */

//-----------------------
// This Class' Header --
//-----------------------
#include "CondTools/DT/interface/DTDBDataHandle.h"

//-------------------------------
// Collaborating Class Headers --
//-------------------------------


//---------------
// C++ Headers --
//---------------


//-------------------
// Initializations --
//-------------------


//----------------
// Constructors --
//----------------
DTDBDataHandle::DTDBDataHandle() {
}

//--------------
// Destructor --
//--------------
DTDBDataHandle::~DTDBDataHandle() {
}

//--------------
// Operations --
//--------------
int DTDBDataHandle::nearestInt( double d ) {

  if ( d > 0.0 ) d += 0.5;
  else           d -= 0.5;
  return static_cast<int>( d );

}


bool DTDBDataHandle::toBool( short s ) {

  union u_short_bool {
    short s_num;
    bool  b_num;
  };
  union u_short_bool dataBuffer;
  dataBuffer.s_num = s;
  return dataBuffer.b_num;

}


short DTDBDataHandle::toShort( bool b ) {

  union u_short_bool {
    short s_num;
    bool  b_num;
  };
  union u_short_bool dataBuffer;
  dataBuffer.s_num = 0;
  dataBuffer.b_num = b;
  return dataBuffer.s_num;

}




