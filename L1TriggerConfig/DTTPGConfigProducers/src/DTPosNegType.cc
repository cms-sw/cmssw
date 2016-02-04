/*
 *  See header file for a description of this class.
 *
 *  $Date: 2009/06/12 10:58:46 $
 *  $Revision: 1.1 $
 *  \author Paolo Ronchese INFN Padova
 *
 */

//-----------------------
// This Class' Header --
//-----------------------
#include "L1TriggerConfig/DTTPGConfigProducers/src/DTPosNegType.h"
//#include "CondTools/DT/interface/DTPosNeg.h"

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "DataFormats/MuonDetId/interface/DTChamberId.h"

//---------------
// C++ Headers --
//---------------
#include <iostream>

//-------------------
// Initializations --
//-------------------
bool DTPosNegType::initRequest = true;
std::map<int,int> DTPosNegType::geomMap;

//----------------
// Constructors --
//----------------
DTPosNegType::DTPosNegType() {
}

//--------------
// Destructor --
//--------------
DTPosNegType::~DTPosNegType() {
}

//--------------
// Operations --
//--------------
/*
DTPosNegCompare::operator()( const DTCCBId& idl,
                             const DTCCBId& idr ) {


}
*/

void DTPosNegType::dump() {

  if ( initRequest ) fillMap();
  std::map<int,int>::const_iterator iter = geomMap.begin();
  std::map<int,int>::const_iterator iend = geomMap.end();
  while ( iter != iend ) {
    const std::pair<int,int>& entry = *iter++;
    int cha = entry.first;
    int pnt = entry.second;
    int whe;
    int sec;
    int sta;
    decode( cha, whe, sec, sta );
    int p_n;
    int c_t;
    decode( pnt, p_n, c_t );
    if ( whe >= 0 ) std::cout << " ";
    std::cout << whe << " ";
    if ( sec < 10 ) std::cout << " ";
    std::cout << sec << " "
              << sta << " "
              << p_n << " "
              << c_t << std::endl;
    int pnc = getPN( whe, sec, sta );
    int ctc = getCT( whe, sec, sta );
    if ( ( pnc != p_n ) ||
         ( ctc != c_t ) ) std::cout << "****************** "
                                    << pnc << " " << ctc << std::endl;
  }
  return;

}


int DTPosNegType::getPN( int whe, int sec, int sta ) {
  int p_n;
  int c_t;
  decode( getData( whe, sec, sta ), p_n, c_t );
  return p_n;
}


int DTPosNegType::getPN( const DTChamberId& cha ) {
  return getPN( cha.wheel(),
                cha.sector(),
                cha.station() );
}


int DTPosNegType::getCT( int whe, int sec, int sta ) {
  int p_n;
  int c_t;
  decode( getData( whe, sec, sta ), p_n, c_t );
  return c_t;
}


int DTPosNegType::getCT( const DTChamberId& cha ) {
  return getCT( cha.wheel(),
                cha.sector(),
                cha.station() );
}


void DTPosNegType::fillMap() {
  //std::cout << "DTPosNeg::fillMap()" << std::endl;
  geomMap.clear();
//  DTChamberId().rawId() , 



// ---
  geomMap.insert( std::pair<int,int>( idCode( -2,  1, 1 ), pnCode(      1, 1 ) ) );
  geomMap.insert( std::pair<int,int>( idCode( -2,  1, 2 ),  pnCode(      2, 2 ) ) );
  geomMap.insert( std::pair<int,int>( idCode( -2,  1, 3 ),  pnCode(      0, 3 ) ) );
  geomMap.insert( std::pair<int,int>( idCode( -2,  1, 4 ),  pnCode(      2, 4 ) ) );
  geomMap.insert( std::pair<int,int>( idCode( -2,  2, 1 ),  pnCode(      1, 1 ) ) );
  geomMap.insert( std::pair<int,int>( idCode( -2,  2, 2 ),  pnCode(      2, 2 ) ) );
  geomMap.insert( std::pair<int,int>( idCode( -2,  2, 3 ),  pnCode(      0, 3 ) ) );
  geomMap.insert( std::pair<int,int>( idCode( -2,  2, 4 ),  pnCode(      2, 4 ) ) );
  geomMap.insert( std::pair<int,int>( idCode( -2,  3, 1 ),  pnCode(      1, 1 ) ) );
  geomMap.insert( std::pair<int,int>( idCode( -2,  3, 2 ),  pnCode(      2, 2 ) ) );
  geomMap.insert( std::pair<int,int>( idCode( -2,  3, 3 ),  pnCode(      0, 3 ) ) );
  geomMap.insert( std::pair<int,int>( idCode( -2,  3, 4 ),  pnCode(      2, 4 ) ) );
  geomMap.insert( std::pair<int,int>( idCode( -2,  4, 1 ),  pnCode(      1, 1 ) ) );
  geomMap.insert( std::pair<int,int>( idCode( -2,  4, 2 ),  pnCode(      2, 2 ) ) );
  geomMap.insert( std::pair<int,int>( idCode( -2,  4, 3 ),  pnCode(      0, 3 ) ) );
  geomMap.insert( std::pair<int,int>( idCode( -2,  4, 4 ),  pnCode(      0, 6 ) ) );
  geomMap.insert( std::pair<int,int>( idCode( -2,  5, 1 ),  pnCode(      1, 1 ) ) );
  geomMap.insert( std::pair<int,int>( idCode( -2,  5, 2 ),  pnCode(      2, 2 ) ) );
  geomMap.insert( std::pair<int,int>( idCode( -2,  5, 3 ),  pnCode(      0, 3 ) ) );
  geomMap.insert( std::pair<int,int>( idCode( -2,  5, 4 ),  pnCode(      1, 4 ) ) );
  geomMap.insert( std::pair<int,int>( idCode( -2,  6, 1 ),  pnCode(      1, 1 ) ) );
  geomMap.insert( std::pair<int,int>( idCode( -2,  6, 2 ),  pnCode(      2, 2 ) ) );
  geomMap.insert( std::pair<int,int>( idCode( -2,  6, 3 ),  pnCode(      0, 3 ) ) );
  geomMap.insert( std::pair<int,int>( idCode( -2,  6, 4 ),  pnCode(      1, 4 ) ) );
  geomMap.insert( std::pair<int,int>( idCode( -2,  7, 1 ),  pnCode(      1, 1 ) ) );
  geomMap.insert( std::pair<int,int>( idCode( -2,  7, 2 ),  pnCode(      2, 2 ) ) );
  geomMap.insert( std::pair<int,int>( idCode( -2,  7, 3 ),  pnCode(      0, 3 ) ) );
  geomMap.insert( std::pair<int,int>( idCode( -2,  7, 4 ),  pnCode(      1, 4 ) ) );
  geomMap.insert( std::pair<int,int>( idCode( -2,  8, 1 ),  pnCode(      1, 1 ) ) );
  geomMap.insert( std::pair<int,int>( idCode( -2,  8, 2 ),  pnCode(      2, 2 ) ) );
  geomMap.insert( std::pair<int,int>( idCode( -2,  8, 3 ),  pnCode(      0, 3 ) ) );
  geomMap.insert( std::pair<int,int>( idCode( -2,  8, 4 ),  pnCode(      1, 4 ) ) );
  geomMap.insert( std::pair<int,int>( idCode( -2,  9, 1 ),  pnCode(      1, 1 ) ) );
  geomMap.insert( std::pair<int,int>( idCode( -2,  9, 2 ),  pnCode(      2, 2 ) ) );
  geomMap.insert( std::pair<int,int>( idCode( -2,  9, 3 ),  pnCode(      0, 3 ) ) );
  geomMap.insert( std::pair<int,int>( idCode( -2,  9, 4 ),  pnCode(      0, 5 ) ) );
  geomMap.insert( std::pair<int,int>( idCode( -2, 10, 1 ),  pnCode(      1, 1 ) ) );
  geomMap.insert( std::pair<int,int>( idCode( -2, 10, 2 ),  pnCode(      2, 2 ) ) );
  geomMap.insert( std::pair<int,int>( idCode( -2, 10, 3 ),  pnCode(      0, 3 ) ) );
  geomMap.insert( std::pair<int,int>( idCode( -2, 10, 4 ),  pnCode(      2, 7 ) ) );
  geomMap.insert( std::pair<int,int>( idCode( -2, 11, 1 ),  pnCode(      1, 1 ) ) );
  geomMap.insert( std::pair<int,int>( idCode( -2, 11, 2 ),  pnCode(      2, 2 ) ) );
  geomMap.insert( std::pair<int,int>( idCode( -2, 11, 3 ),  pnCode(      0, 3 ) ) );
  geomMap.insert( std::pair<int,int>( idCode( -2, 11, 4 ),  pnCode(      0, 5 ) ) );
  geomMap.insert( std::pair<int,int>( idCode( -2, 12, 1 ),  pnCode(      1, 1 ) ) );
  geomMap.insert( std::pair<int,int>( idCode( -2, 12, 2 ),  pnCode(      2, 2 ) ) );
  geomMap.insert( std::pair<int,int>( idCode( -2, 12, 3 ),  pnCode(      0, 3 ) ) );
  geomMap.insert( std::pair<int,int>( idCode( -2, 12, 4 ),  pnCode(      2, 4 ) ) );
  geomMap.insert( std::pair<int,int>( idCode( -2, 13, 4 ),  pnCode(      0, 6 ) ) );
  geomMap.insert( std::pair<int,int>( idCode( -2, 14, 4 ),  pnCode(      1, 7 ) ) );
  geomMap.insert( std::pair<int,int>( idCode( -1,  1, 1 ),  pnCode(      1, 1 ) ) );
  geomMap.insert( std::pair<int,int>( idCode( -1,  1, 2 ),  pnCode(      2, 2 ) ) );
  geomMap.insert( std::pair<int,int>( idCode( -1,  1, 3 ),  pnCode(      0, 3 ) ) );
  geomMap.insert( std::pair<int,int>( idCode( -1,  1, 4 ),  pnCode(      2, 4 ) ) );
  geomMap.insert( std::pair<int,int>( idCode( -1,  2, 1 ),  pnCode(      1, 1 ) ) );
  geomMap.insert( std::pair<int,int>( idCode( -1,  2, 2 ),  pnCode(      2, 2 ) ) );
  geomMap.insert( std::pair<int,int>( idCode( -1,  2, 3 ),  pnCode(      0, 3 ) ) );
  geomMap.insert( std::pair<int,int>( idCode( -1,  2, 4 ),  pnCode(      2, 4 ) ) );
  geomMap.insert( std::pair<int,int>( idCode( -1,  3, 1 ),  pnCode(      1, 1 ) ) );
  geomMap.insert( std::pair<int,int>( idCode( -1,  3, 2 ),  pnCode(      2, 2 ) ) );
  geomMap.insert( std::pair<int,int>( idCode( -1,  3, 3 ),  pnCode(      0, 3 ) ) );
  geomMap.insert( std::pair<int,int>( idCode( -1,  3, 4 ),  pnCode(      2, 4 ) ) );
  geomMap.insert( std::pair<int,int>( idCode( -1,  4, 1 ),  pnCode(      1, 1 ) ) );
  geomMap.insert( std::pair<int,int>( idCode( -1,  4, 2 ),  pnCode(      2, 2 ) ) );
  geomMap.insert( std::pair<int,int>( idCode( -1,  4, 3 ),  pnCode(      0, 3 ) ) );
  geomMap.insert( std::pair<int,int>( idCode( -1,  4, 4 ),  pnCode(      0, 6 ) ) );
  geomMap.insert( std::pair<int,int>( idCode( -1,  5, 1 ),  pnCode(      1, 1 ) ) );
  geomMap.insert( std::pair<int,int>( idCode( -1,  5, 2 ),  pnCode(      2, 2 ) ) );
  geomMap.insert( std::pair<int,int>( idCode( -1,  5, 3 ),  pnCode(      0, 3 ) ) );
  geomMap.insert( std::pair<int,int>( idCode( -1,  5, 4 ),  pnCode(      1, 4 ) ) );
  geomMap.insert( std::pair<int,int>( idCode( -1,  6, 1 ),  pnCode(      1, 1 ) ) );
  geomMap.insert( std::pair<int,int>( idCode( -1,  6, 2 ),  pnCode(      2, 2 ) ) );
  geomMap.insert( std::pair<int,int>( idCode( -1,  6, 3 ),  pnCode(      0, 3 ) ) );
  geomMap.insert( std::pair<int,int>( idCode( -1,  6, 4 ),  pnCode(      1, 4 ) ) );
  geomMap.insert( std::pair<int,int>( idCode( -1,  7, 1 ),  pnCode(      1, 1 ) ) );
  geomMap.insert( std::pair<int,int>( idCode( -1,  7, 2 ),  pnCode(      2, 2 ) ) );
  geomMap.insert( std::pair<int,int>( idCode( -1,  7, 3 ),  pnCode(      0, 3 ) ) );
  geomMap.insert( std::pair<int,int>( idCode( -1,  7, 4 ),  pnCode(      1, 4 ) ) );
  geomMap.insert( std::pair<int,int>( idCode( -1,  8, 1 ),  pnCode(      1, 1 ) ) );
  geomMap.insert( std::pair<int,int>( idCode( -1,  8, 2 ),  pnCode(      2, 2 ) ) );
  geomMap.insert( std::pair<int,int>( idCode( -1,  8, 3 ),  pnCode(      0, 3 ) ) );
  geomMap.insert( std::pair<int,int>( idCode( -1,  8, 4 ),  pnCode(      1, 4 ) ) );
  geomMap.insert( std::pair<int,int>( idCode( -1,  9, 1 ),  pnCode(      1, 1 ) ) );
  geomMap.insert( std::pair<int,int>( idCode( -1,  9, 2 ),  pnCode(      2, 2 ) ) );
  geomMap.insert( std::pair<int,int>( idCode( -1,  9, 3 ),  pnCode(      0, 3 ) ) );
  geomMap.insert( std::pair<int,int>( idCode( -1,  9, 4 ),  pnCode(      0, 5 ) ) );
  geomMap.insert( std::pair<int,int>( idCode( -1, 10, 1 ),  pnCode(      1, 1 ) ) );
  geomMap.insert( std::pair<int,int>( idCode( -1, 10, 2 ),  pnCode(      2, 2 ) ) );
  geomMap.insert( std::pair<int,int>( idCode( -1, 10, 3 ),  pnCode(      0, 3 ) ) );
  geomMap.insert( std::pair<int,int>( idCode( -1, 10, 4 ),  pnCode(      2, 7 ) ) );
  geomMap.insert( std::pair<int,int>( idCode( -1, 11, 1 ),  pnCode(      1, 1 ) ) );
  geomMap.insert( std::pair<int,int>( idCode( -1, 11, 2 ),  pnCode(      2, 2 ) ) );
  geomMap.insert( std::pair<int,int>( idCode( -1, 11, 3 ),  pnCode(      0, 3 ) ) );
  geomMap.insert( std::pair<int,int>( idCode( -1, 11, 4 ),  pnCode(      0, 5 ) ) );
  geomMap.insert( std::pair<int,int>( idCode( -1, 12, 1 ),  pnCode(      1, 1 ) ) );
  geomMap.insert( std::pair<int,int>( idCode( -1, 12, 2 ),  pnCode(      2, 2 ) ) );
  geomMap.insert( std::pair<int,int>( idCode( -1, 12, 3 ),  pnCode(      0, 3 ) ) );
  geomMap.insert( std::pair<int,int>( idCode( -1, 12, 4 ),  pnCode(      2, 4 ) ) );
  geomMap.insert( std::pair<int,int>( idCode( -1, 13, 4 ),  pnCode(      0, 6 ) ) );
  geomMap.insert( std::pair<int,int>( idCode( -1, 14, 4 ),  pnCode(      1, 7 ) ) );
  geomMap.insert( std::pair<int,int>( idCode(  0,  1, 1 ),  pnCode(      1, 1 ) ) );
  geomMap.insert( std::pair<int,int>( idCode(  0,  1, 2 ),  pnCode(      2, 2 ) ) );
  geomMap.insert( std::pair<int,int>( idCode(  0,  1, 3 ),  pnCode(      0, 3 ) ) );
  geomMap.insert( std::pair<int,int>( idCode(  0,  1, 4 ),  pnCode(      2, 4 ) ) );
  geomMap.insert( std::pair<int,int>( idCode(  0,  2, 1 ),  pnCode(      2, 1 ) ) );
  geomMap.insert( std::pair<int,int>( idCode(  0,  2, 2 ),  pnCode(      1, 2 ) ) );
  geomMap.insert( std::pair<int,int>( idCode(  0,  2, 3 ),  pnCode(      0, 3 ) ) );
  geomMap.insert( std::pair<int,int>( idCode(  0,  2, 4 ),  pnCode(      1, 4 ) ) );
  geomMap.insert( std::pair<int,int>( idCode(  0,  3, 1 ),  pnCode(      2, 1 ) ) );
  geomMap.insert( std::pair<int,int>( idCode(  0,  3, 2 ),  pnCode(      1, 2 ) ) );
  geomMap.insert( std::pair<int,int>( idCode(  0,  3, 3 ),  pnCode(      0, 3 ) ) );
  geomMap.insert( std::pair<int,int>( idCode(  0,  3, 4 ),  pnCode(      1, 4 ) ) );
  geomMap.insert( std::pair<int,int>( idCode(  0,  4, 1 ),  pnCode(      1, 1 ) ) );
  geomMap.insert( std::pair<int,int>( idCode(  0,  4, 2 ),  pnCode(      2, 2 ) ) );
  geomMap.insert( std::pair<int,int>( idCode(  0,  4, 3 ),  pnCode(      0, 3 ) ) );
  geomMap.insert( std::pair<int,int>( idCode(  0,  4, 4 ),  pnCode(      0, 6 ) ) );
  geomMap.insert( std::pair<int,int>( idCode(  0,  5, 1 ),  pnCode(      1, 1 ) ) );
  geomMap.insert( std::pair<int,int>( idCode(  0,  5, 2 ),  pnCode(      2, 2 ) ) );
  geomMap.insert( std::pair<int,int>( idCode(  0,  5, 3 ),  pnCode(      0, 3 ) ) );
  geomMap.insert( std::pair<int,int>( idCode(  0,  5, 4 ),  pnCode(      1, 4 ) ) );
  geomMap.insert( std::pair<int,int>( idCode(  0,  6, 1 ),  pnCode(      2, 1 ) ) );
  geomMap.insert( std::pair<int,int>( idCode(  0,  6, 2 ),  pnCode(      1, 2 ) ) );
  geomMap.insert( std::pair<int,int>( idCode(  0,  6, 3 ),  pnCode(      0, 3 ) ) );
  geomMap.insert( std::pair<int,int>( idCode(  0,  6, 4 ),  pnCode(      2, 4 ) ) );
  geomMap.insert( std::pair<int,int>( idCode(  0,  7, 1 ),  pnCode(      2, 1 ) ) );
  geomMap.insert( std::pair<int,int>( idCode(  0,  7, 2 ),  pnCode(      1, 2 ) ) );
  geomMap.insert( std::pair<int,int>( idCode(  0,  7, 3 ),  pnCode(      0, 3 ) ) );
  geomMap.insert( std::pair<int,int>( idCode(  0,  7, 4 ),  pnCode(      2, 4 ) ) );
  geomMap.insert( std::pair<int,int>( idCode(  0,  8, 1 ),  pnCode(      1, 1 ) ) );
  geomMap.insert( std::pair<int,int>( idCode(  0,  8, 2 ),  pnCode(      2, 2 ) ) );
  geomMap.insert( std::pair<int,int>( idCode(  0,  8, 3 ),  pnCode(      0, 3 ) ) );
  geomMap.insert( std::pair<int,int>( idCode(  0,  8, 4 ),  pnCode(      1, 4 ) ) );
  geomMap.insert( std::pair<int,int>( idCode(  0,  9, 1 ),  pnCode(      1, 1 ) ) );
  geomMap.insert( std::pair<int,int>( idCode(  0,  9, 2 ),  pnCode(      2, 2 ) ) );
  geomMap.insert( std::pair<int,int>( idCode(  0,  9, 3 ),  pnCode(      0, 3 ) ) );
  geomMap.insert( std::pair<int,int>( idCode(  0,  9, 4 ),  pnCode(      0, 5 ) ) );
  geomMap.insert( std::pair<int,int>( idCode(  0, 10, 1 ),  pnCode(      2, 1 ) ) );
  geomMap.insert( std::pair<int,int>( idCode(  0, 10, 2 ),  pnCode(      1, 2 ) ) );
  geomMap.insert( std::pair<int,int>( idCode(  0, 10, 3 ),  pnCode(      0, 3 ) ) );
  geomMap.insert( std::pair<int,int>( idCode(  0, 10, 4 ),  pnCode(      2, 7 ) ) );
  geomMap.insert( std::pair<int,int>( idCode(  0, 11, 1 ),  pnCode(      2, 1 ) ) );
  geomMap.insert( std::pair<int,int>( idCode(  0, 11, 2 ),  pnCode(      1, 2 ) ) );
  geomMap.insert( std::pair<int,int>( idCode(  0, 11, 3 ),  pnCode(      0, 3 ) ) );
  geomMap.insert( std::pair<int,int>( idCode(  0, 11, 4 ),  pnCode(      0, 5 ) ) );
  geomMap.insert( std::pair<int,int>( idCode(  0, 12, 1 ),  pnCode(      1, 1 ) ) );
  geomMap.insert( std::pair<int,int>( idCode(  0, 12, 2 ),  pnCode(      2, 2 ) ) );
  geomMap.insert( std::pair<int,int>( idCode(  0, 12, 3 ),  pnCode(      0, 3 ) ) );
  geomMap.insert( std::pair<int,int>( idCode(  0, 12, 4 ),  pnCode(      2, 4 ) ) );
  geomMap.insert( std::pair<int,int>( idCode(  0, 13, 4 ),  pnCode(      0, 6 ) ) );
  geomMap.insert( std::pair<int,int>( idCode(  0, 14, 4 ),  pnCode(      1, 7 ) ) );
  geomMap.insert( std::pair<int,int>( idCode(  1,  1, 1 ),  pnCode(      2, 1 ) ) );
  geomMap.insert( std::pair<int,int>( idCode(  1,  1, 2 ),  pnCode(      1, 2 ) ) );
  geomMap.insert( std::pair<int,int>( idCode(  1,  1, 3 ),  pnCode(      0, 3 ) ) );
  geomMap.insert( std::pair<int,int>( idCode(  1,  1, 4 ),  pnCode(      1, 4 ) ) );
  geomMap.insert( std::pair<int,int>( idCode(  1,  2, 1 ),  pnCode(      2, 1 ) ) );
  geomMap.insert( std::pair<int,int>( idCode(  1,  2, 2 ),  pnCode(      1, 2 ) ) );
  geomMap.insert( std::pair<int,int>( idCode(  1,  2, 3 ),  pnCode(      0, 3 ) ) );
  geomMap.insert( std::pair<int,int>( idCode(  1,  2, 4 ),  pnCode(      1, 4 ) ) );
  geomMap.insert( std::pair<int,int>( idCode(  1,  3, 1 ),  pnCode(      2, 1 ) ) );
  geomMap.insert( std::pair<int,int>( idCode(  1,  3, 2 ),  pnCode(      1, 2 ) ) );
  geomMap.insert( std::pair<int,int>( idCode(  1,  3, 3 ),  pnCode(      0, 3 ) ) );
  geomMap.insert( std::pair<int,int>( idCode(  1,  3, 4 ),  pnCode(      1, 4 ) ) );
  geomMap.insert( std::pair<int,int>( idCode(  1,  4, 1 ),  pnCode(      2, 1 ) ) );
  geomMap.insert( std::pair<int,int>( idCode(  1,  4, 2 ),  pnCode(      1, 2 ) ) );
  geomMap.insert( std::pair<int,int>( idCode(  1,  4, 3 ),  pnCode(      0, 3 ) ) );
  geomMap.insert( std::pair<int,int>( idCode(  1,  4, 4 ),  pnCode(      0, 6 ) ) );
  geomMap.insert( std::pair<int,int>( idCode(  1,  5, 1 ),  pnCode(      2, 1 ) ) );
  geomMap.insert( std::pair<int,int>( idCode(  1,  5, 2 ),  pnCode(      1, 2 ) ) );
  geomMap.insert( std::pair<int,int>( idCode(  1,  5, 3 ),  pnCode(      0, 3 ) ) );
  geomMap.insert( std::pair<int,int>( idCode(  1,  5, 4 ),  pnCode(      2, 4 ) ) );
  geomMap.insert( std::pair<int,int>( idCode(  1,  6, 1 ),  pnCode(      2, 1 ) ) );
  geomMap.insert( std::pair<int,int>( idCode(  1,  6, 2 ),  pnCode(      1, 2 ) ) );
  geomMap.insert( std::pair<int,int>( idCode(  1,  6, 3 ),  pnCode(      0, 3 ) ) );
  geomMap.insert( std::pair<int,int>( idCode(  1,  6, 4 ),  pnCode(      2, 4 ) ) );
  geomMap.insert( std::pair<int,int>( idCode(  1,  7, 1 ),  pnCode(      2, 1 ) ) );
  geomMap.insert( std::pair<int,int>( idCode(  1,  7, 2 ),  pnCode(      1, 2 ) ) );
  geomMap.insert( std::pair<int,int>( idCode(  1,  7, 3 ),  pnCode(      0, 3 ) ) );
  geomMap.insert( std::pair<int,int>( idCode(  1,  7, 4 ),  pnCode(      2, 4 ) ) );
  geomMap.insert( std::pair<int,int>( idCode(  1,  8, 1 ),  pnCode(      2, 1 ) ) );
  geomMap.insert( std::pair<int,int>( idCode(  1,  8, 2 ),  pnCode(      1, 2 ) ) );
  geomMap.insert( std::pair<int,int>( idCode(  1,  8, 3 ),  pnCode(      0, 3 ) ) );
  geomMap.insert( std::pair<int,int>( idCode(  1,  8, 4 ),  pnCode(      1, 4 ) ) );
  geomMap.insert( std::pair<int,int>( idCode(  1,  9, 1 ),  pnCode(      2, 1 ) ) );
  geomMap.insert( std::pair<int,int>( idCode(  1,  9, 2 ),  pnCode(      1, 2 ) ) );
  geomMap.insert( std::pair<int,int>( idCode(  1,  9, 3 ),  pnCode(      0, 3 ) ) );
  geomMap.insert( std::pair<int,int>( idCode(  1,  9, 4 ),  pnCode(      0, 5 ) ) );
  geomMap.insert( std::pair<int,int>( idCode(  1, 10, 1 ),  pnCode(      2, 1 ) ) );
  geomMap.insert( std::pair<int,int>( idCode(  1, 10, 2 ),  pnCode(      1, 2 ) ) );
  geomMap.insert( std::pair<int,int>( idCode(  1, 10, 3 ),  pnCode(      0, 3 ) ) );
  geomMap.insert( std::pair<int,int>( idCode(  1, 10, 4 ),  pnCode(      2, 7 ) ) );
  geomMap.insert( std::pair<int,int>( idCode(  1, 11, 1 ),  pnCode(      2, 1 ) ) );
  geomMap.insert( std::pair<int,int>( idCode(  1, 11, 2 ),  pnCode(      1, 2 ) ) );
  geomMap.insert( std::pair<int,int>( idCode(  1, 11, 3 ),  pnCode(      0, 3 ) ) );
  geomMap.insert( std::pair<int,int>( idCode(  1, 11, 4 ),  pnCode(      0, 5 ) ) );
  geomMap.insert( std::pair<int,int>( idCode(  1, 12, 1 ),  pnCode(      2, 1 ) ) );
  geomMap.insert( std::pair<int,int>( idCode(  1, 12, 2 ),  pnCode(      1, 2 ) ) );
  geomMap.insert( std::pair<int,int>( idCode(  1, 12, 3 ),  pnCode(      0, 3 ) ) );
  geomMap.insert( std::pair<int,int>( idCode(  1, 12, 4 ),  pnCode(      2, 4 ) ) );
  geomMap.insert( std::pair<int,int>( idCode(  1, 13, 4 ),  pnCode(      0, 6 ) ) );
  geomMap.insert( std::pair<int,int>( idCode(  1, 14, 4 ),  pnCode(      1, 7 ) ) );
  geomMap.insert( std::pair<int,int>( idCode(  2,  1, 1 ),  pnCode(      2, 1 ) ) );
  geomMap.insert( std::pair<int,int>( idCode(  2,  1, 2 ),  pnCode(      1, 2 ) ) );
  geomMap.insert( std::pair<int,int>( idCode(  2,  1, 3 ),  pnCode(      0, 3 ) ) );
  geomMap.insert( std::pair<int,int>( idCode(  2,  1, 4 ),  pnCode(      1, 4 ) ) );
  geomMap.insert( std::pair<int,int>( idCode(  2,  2, 1 ),  pnCode(      2, 1 ) ) );
  geomMap.insert( std::pair<int,int>( idCode(  2,  2, 2 ),  pnCode(      1, 2 ) ) );
  geomMap.insert( std::pair<int,int>( idCode(  2,  2, 3 ),  pnCode(      0, 3 ) ) );
  geomMap.insert( std::pair<int,int>( idCode(  2,  2, 4 ),  pnCode(      1, 4 ) ) );
  geomMap.insert( std::pair<int,int>( idCode(  2,  3, 1 ),  pnCode(      2, 1 ) ) );
  geomMap.insert( std::pair<int,int>( idCode(  2,  3, 2 ),  pnCode(      1, 2 ) ) );
  geomMap.insert( std::pair<int,int>( idCode(  2,  3, 3 ),  pnCode(      0, 3 ) ) );
  geomMap.insert( std::pair<int,int>( idCode(  2,  3, 4 ),  pnCode(      1, 4 ) ) );
  geomMap.insert( std::pair<int,int>( idCode(  2,  4, 1 ),  pnCode(      2, 1 ) ) );
  geomMap.insert( std::pair<int,int>( idCode(  2,  4, 2 ),  pnCode(      1, 2 ) ) );
  geomMap.insert( std::pair<int,int>( idCode(  2,  4, 3 ),  pnCode(      0, 3 ) ) );
  geomMap.insert( std::pair<int,int>( idCode(  2,  4, 4 ),  pnCode(      0, 6 ) ) );
  geomMap.insert( std::pair<int,int>( idCode(  2,  5, 1 ),  pnCode(      2, 1 ) ) );
  geomMap.insert( std::pair<int,int>( idCode(  2,  5, 2 ),  pnCode(      1, 2 ) ) );
  geomMap.insert( std::pair<int,int>( idCode(  2,  5, 3 ),  pnCode(      0, 3 ) ) );
  geomMap.insert( std::pair<int,int>( idCode(  2,  5, 4 ),  pnCode(      2, 4 ) ) );
  geomMap.insert( std::pair<int,int>( idCode(  2,  6, 1 ),  pnCode(      2, 1 ) ) );
  geomMap.insert( std::pair<int,int>( idCode(  2,  6, 2 ),  pnCode(      1, 2 ) ) );
  geomMap.insert( std::pair<int,int>( idCode(  2,  6, 3 ),  pnCode(      0, 3 ) ) );
  geomMap.insert( std::pair<int,int>( idCode(  2,  6, 4 ),  pnCode(      2, 4 ) ) );
  geomMap.insert( std::pair<int,int>( idCode(  2,  7, 1 ),  pnCode(      2, 1 ) ) );
  geomMap.insert( std::pair<int,int>( idCode(  2,  7, 2 ),  pnCode(      1, 2 ) ) );
  geomMap.insert( std::pair<int,int>( idCode(  2,  7, 3 ),  pnCode(      0, 3 ) ) );
  geomMap.insert( std::pair<int,int>( idCode(  2,  7, 4 ),  pnCode(      2, 4 ) ) );
  geomMap.insert( std::pair<int,int>( idCode(  2,  8, 1 ),  pnCode(      2, 1 ) ) );
  geomMap.insert( std::pair<int,int>( idCode(  2,  8, 2 ),  pnCode(      1, 2 ) ) );
  geomMap.insert( std::pair<int,int>( idCode(  2,  8, 3 ),  pnCode(      0, 3 ) ) );
  geomMap.insert( std::pair<int,int>( idCode(  2,  8, 4 ),  pnCode(      2, 4 ) ) );
  geomMap.insert( std::pair<int,int>( idCode(  2,  9, 1 ),  pnCode(      2, 1 ) ) );
  geomMap.insert( std::pair<int,int>( idCode(  2,  9, 2 ),  pnCode(      1, 2 ) ) );
  geomMap.insert( std::pair<int,int>( idCode(  2,  9, 3 ),  pnCode(      0, 3 ) ) );
  geomMap.insert( std::pair<int,int>( idCode(  2,  9, 4 ),  pnCode(      0, 5 ) ) );
  geomMap.insert( std::pair<int,int>( idCode(  2, 10, 1 ),  pnCode(      2, 1 ) ) );
  geomMap.insert( std::pair<int,int>( idCode(  2, 10, 2 ),  pnCode(      1, 2 ) ) );
  geomMap.insert( std::pair<int,int>( idCode(  2, 10, 3 ),  pnCode(      0, 3 ) ) );
  geomMap.insert( std::pair<int,int>( idCode(  2, 10, 4 ),  pnCode(      2, 7 ) ) );
  geomMap.insert( std::pair<int,int>( idCode(  2, 11, 1 ),  pnCode(      2, 1 ) ) );
  geomMap.insert( std::pair<int,int>( idCode(  2, 11, 2 ),  pnCode(      1, 2 ) ) );
  geomMap.insert( std::pair<int,int>( idCode(  2, 11, 3 ),  pnCode(      0, 3 ) ) );
  geomMap.insert( std::pair<int,int>( idCode(  2, 11, 4 ),  pnCode(      0, 5 ) ) );
  geomMap.insert( std::pair<int,int>( idCode(  2, 12, 1 ),  pnCode(      2, 1 ) ) );
  geomMap.insert( std::pair<int,int>( idCode(  2, 12, 2 ),  pnCode(      1, 2 ) ) );
  geomMap.insert( std::pair<int,int>( idCode(  2, 12, 3 ),  pnCode(      0, 3 ) ) );
  geomMap.insert( std::pair<int,int>( idCode(  2, 12, 4 ),  pnCode(      1, 4 ) ) );
  geomMap.insert( std::pair<int,int>( idCode(  2, 13, 4 ),  pnCode(      0, 6 ) ) );
  geomMap.insert( std::pair<int,int>( idCode(  2, 14, 4 ),  pnCode(      1, 7 ) ) );
//
  initRequest = false;
}


int DTPosNegType::idCode( int whe, int sec, int sta ) {
  return ( ( ( ( whe + 3 ) * 100 ) + sec ) * 10 ) + sta;
}


int DTPosNegType::pnCode( int p, int t ) {
  return ( p * 1000 ) + t;
}


void DTPosNegType::decode( int code, int& whe, int& sec, int& sta ) {
  whe = ( code / 1000 ) - 3;
  sec = ( code /   10 ) % 100;
  sta =   code          % 10;
  return;
}


void DTPosNegType::decode( int code, int& p, int& t ) {
  p = code / 1000;
  t = code % 1000;
  return;
}

int DTPosNegType::getData( int whe, int sec, int sta ) {
  if ( initRequest ) fillMap();
  std::map<int,int>::const_iterator iter = geomMap.find( idCode( whe,                             sec,                             sta ) );
  std::map<int,int>::const_iterator iend = geomMap.end();
  if ( iter == iend ) return 999999;
  return iter->second;
}

