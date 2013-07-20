//-------------------------------------------------
//
//   Class: L1MuDTAddressArray
//
//   Description: Array of relative Addresses
//
//
//   $Date: 2008/10/13 07:44:43 $
//   $Revision: 1.3 $
//
//   Author :
//   N. Neumeister            CERN EP
//
//--------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------

#include "L1Trigger/DTTrackFinder/src/L1MuDTAddressArray.h"

//---------------
// C++ Headers --
//---------------

#include <iostream>
#include <iomanip>
#include <vector>
#include <cassert>

//-------------------------------
// Collaborating Class Headers --
//-------------------------------

using namespace std;

// --------------------------------
//       class L1MuDTAddressArray
//---------------------------------

//----------------
// Constructors --
//----------------

L1MuDTAddressArray::L1MuDTAddressArray() {

  reset();

}


L1MuDTAddressArray::L1MuDTAddressArray(const L1MuDTAddressArray& addarray) {
  
  for ( int stat = 1; stat <= 4; stat++ ) {
    m_station[stat-1] = addarray.m_station[stat-1];
  }

}


//--------------
// Destructor --
//--------------

L1MuDTAddressArray::~L1MuDTAddressArray() {}


//--------------
// Operations --
//--------------

//
// assignment operator
//
L1MuDTAddressArray& L1MuDTAddressArray::operator=(const L1MuDTAddressArray& addarray) {

  if ( this != &addarray ) {
    for ( int stat = 1; stat <= 4; stat++ ) {
      m_station[stat-1] = addarray.m_station[stat-1];
    }
  }
  return *this;

}


//
//
//
bool L1MuDTAddressArray::operator==(const L1MuDTAddressArray& addarray) const {

  for ( int stat = 1; stat <= 4; stat++ ) {
    if ( m_station[stat-1] != addarray.m_station[stat-1] ) return false;
  }

  return true;

}


//
//
//
bool L1MuDTAddressArray::operator!=(const L1MuDTAddressArray& addarray) const {

  for ( int stat = 1; stat <= 4; stat++ ) {
    if ( m_station[stat-1] != addarray.m_station[stat-1] ) return true;
  }

  return false;

}


//
// reset AddressArray
//
void L1MuDTAddressArray::reset() {

  for ( int stat = 1; stat <= 4; stat++ ) {
    m_station[stat-1] = 15; 
  }

}


//
// set Address of a given station
//
void L1MuDTAddressArray::setStation(int stat, int adr) {

  //  assert( stat >  0 && stat <= 4 );
  //  assert( adr  >= 0 && adr  <= 15       );  
  m_station[stat-1] = adr;

}


//
// set Addresses of all four stations
//
void L1MuDTAddressArray::setStations(int adr1, int adr2, int adr3, int adr4) {

  setStation(1,adr1);
  setStation(2,adr2);
  setStation(3,adr3);
  setStation(4,adr4);

}


//
// get track address code (for eta track finder)
//
int L1MuDTAddressArray::trackAddressCode() const {

  int code = -1;
  
  int s1 = m_station[0];
  s1 = ( s1 == 15 ) ? 0 : ((s1/2)%2)+1;
  int s2 = m_station[1];
  s2 = ( s2 == 15 ) ? 0 : ((s2/2)%2)+1;
  int s3 = m_station[2];
  s3 = ( s3 == 15 ) ? 0 : ((s3/2)%2)+1;
  int s4 = m_station[3];
  s4 = ( s4 == 15 ) ? 0 : ((s4/2)%2)+1;

  //  0 ... empty track segment
  //  1 ... same wheel 
  //  2 ... next wheel 
  
  if ( s1 == 0 && s2 == 0 && s3 == 0 && s4 == 0 ) code =  0;
  if ( s1 == 0 && s2 == 0 && s3 == 2 && s4 == 1 ) code =  0;
  if ( s1 == 0 && s2 == 0 && s3 == 2 && s4 == 2 ) code =  0;
  if ( s1 == 0 && s2 == 2 && s3 == 0 && s4 == 1 ) code =  0;
  if ( s1 == 0 && s2 == 2 && s3 == 0 && s4 == 2 ) code =  0;
  if ( s1 == 0 && s2 == 2 && s3 == 1 && s4 == 0 ) code =  0;
  if ( s1 == 0 && s2 == 2 && s3 == 2 && s4 == 0 ) code =  0;   
  if ( s1 == 0 && s2 == 1 && s3 == 2 && s4 == 1 ) code =  0; 
  if ( s1 == 0 && s2 == 2 && s3 == 1 && s4 == 1 ) code =  0;
  if ( s1 == 0 && s2 == 2 && s3 == 1 && s4 == 2 ) code =  0;
  if ( s1 == 0 && s2 == 2 && s3 == 2 && s4 == 1 ) code =  0;
  if ( s1 == 0 && s2 == 2 && s3 == 2 && s4 == 2 ) code =  0;  
  if ( s1 == 1 && s2 == 0 && s3 == 2 && s4 == 1 ) code =  0; 
  if ( s1 == 1 && s2 == 2 && s3 == 0 && s4 == 1 ) code =  0;
  if ( s1 == 1 && s2 == 2 && s3 == 1 && s4 == 0 ) code =  0; 
  if ( s1 == 1 && s2 == 1 && s3 == 2 && s4 == 1 ) code =  0; 
  if ( s1 == 1 && s2 == 2 && s3 == 1 && s4 == 1 ) code =  0;
  if ( s1 == 1 && s2 == 2 && s3 == 1 && s4 == 2 ) code =  0;
  if ( s1 == 1 && s2 == 2 && s3 == 2 && s4 == 1 ) code =  0;    
  if ( s1 == 0 && s2 == 0 && s3 == 1 && s4 == 1 ) code =  1;
  if ( s1 == 0 && s2 == 0 && s3 == 1 && s4 == 2 ) code =  2;
  if ( s1 == 0 && s2 == 1 && s3 == 0 && s4 == 1 ) code =  3;
  if ( s1 == 0 && s2 == 1 && s3 == 0 && s4 == 2 ) code =  4;
  if ( s1 == 0 && s2 == 1 && s3 == 1 && s4 == 0 ) code =  5;
  if ( s1 == 0 && s2 == 1 && s3 == 1 && s4 == 1 ) code =  6;
  if ( s1 == 0 && s2 == 1 && s3 == 1 && s4 == 2 ) code =  7;
  if ( s1 == 0 && s2 == 1 && s3 == 2 && s4 == 0 ) code =  8;
  if ( s1 == 0 && s2 == 1 && s3 == 2 && s4 == 2 ) code =  8;
  if ( s1 == 1 && s2 == 0 && s3 == 0 && s4 == 1 ) code =  9;
  if ( s1 == 1 && s2 == 0 && s3 == 0 && s4 == 2 ) code = 10;
  if ( s1 == 1 && s2 == 0 && s3 == 1 && s4 == 0 ) code = 11;
  if ( s1 == 1 && s2 == 0 && s3 == 1 && s4 == 1 ) code = 12;
  if ( s1 == 1 && s2 == 0 && s3 == 1 && s4 == 2 ) code = 13;
  if ( s1 == 1 && s2 == 0 && s3 == 2 && s4 == 0 ) code = 14;
  if ( s1 == 1 && s2 == 0 && s3 == 2 && s4 == 2 ) code = 14;
  if ( s1 == 1 && s2 == 1 && s3 == 0 && s4 == 0 ) code = 15;
  if ( s1 == 1 && s2 == 1 && s3 == 0 && s4 == 1 ) code = 16;
  if ( s1 == 1 && s2 == 1 && s3 == 0 && s4 == 2 ) code = 17;
  if ( s1 == 1 && s2 == 1 && s3 == 1 && s4 == 0 ) code = 18;
  if ( s1 == 1 && s2 == 1 && s3 == 1 && s4 == 1 ) code = 19;
  if ( s1 == 1 && s2 == 1 && s3 == 1 && s4 == 2 ) code = 20;
  if ( s1 == 1 && s2 == 1 && s3 == 2 && s4 == 0 ) code = 21;
  if ( s1 == 1 && s2 == 1 && s3 == 2 && s4 == 2 ) code = 21;
  if ( s1 == 1 && s2 == 2 && s3 == 0 && s4 == 0 ) code = 22;
  if ( s1 == 1 && s2 == 2 && s3 == 0 && s4 == 2 ) code = 22;
  if ( s1 == 1 && s2 == 2 && s3 == 2 && s4 == 0 ) code = 22;
  if ( s1 == 1 && s2 == 2 && s3 == 2 && s4 == 2 ) code = 22;

  return code;

}


//
// get converted Addresses
//
L1MuDTAddressArray L1MuDTAddressArray::converted() const {

  unsigned short int adr1 = L1MuDTAddressArray::convert(m_station[0]); 
  unsigned short int adr2 = L1MuDTAddressArray::convert(m_station[1]);
  unsigned short int adr3 = L1MuDTAddressArray::convert(m_station[2]);
  unsigned short int adr4 = L1MuDTAddressArray::convert(m_station[3]);

  L1MuDTAddressArray newaddressarray; 
  newaddressarray.setStations(adr1,adr2,adr3,adr4);

  return newaddressarray;

}


//
//
//
ostream& operator<<(ostream& s, const L1MuDTAddressArray& adrarr ) {

  s.setf(ios::right,ios::adjustfield);
  for ( int stat = 1; stat <= 4; stat++ ) { 
    s << "stat " << stat << ": " << setw(2) << adrarr.station(stat) << "  ";
  }

  return s;

}


//
// convert address to corresponding VHDL address
//
unsigned short int L1MuDTAddressArray::convert(unsigned short int adr) {

  unsigned short int newaddress = 15;

  switch ( adr ) {
    case  0 : { newaddress =  8; break; }
    case  1 : { newaddress =  9; break; }
    case  2 : { newaddress =  0; break; }
    case  3 : { newaddress =  1; break; }
    case  4 : { newaddress = 10; break; }
    case  5 : { newaddress = 11; break; }
    case  6 : { newaddress =  2; break; }
    case  7 : { newaddress =  3; break; }
    case  8 : { newaddress = 12; break; }
    case  9 : { newaddress = 13; break; }
    case 10 : { newaddress =  4; break; }
    case 11 : { newaddress =  5; break; }
    case 15 : { newaddress = 15; break; }
    default:  { newaddress = 15; break; }

  }

   return newaddress;

}


// 
// is it a same wheel address?
//
bool L1MuDTAddressArray::sameWheel(unsigned short int adr) {

  //  if ( adr > 15 ) cerr << "L1MuDTAddressArray : Error wrong address " << adr << endl;
  return ( (adr/2)%2 == 0 );

}

    
//
// is it a next wheel address?
//
bool L1MuDTAddressArray::nextWheel(unsigned short int adr) {

  //  if ( adr > 15 ) cerr << "L1MuDTAddressArray : Error wrong address " << adr << endl;
  return ( (adr/2)%2 == 1 );

}
