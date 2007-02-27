//-------------------------------------------------
//
//   Class: L1MuDTAssignmentUnit
//
//   Description: Assignment Unit
//
//
//   $Date: 2006/06/26 16:11:13 $
//   $Revision: 1.1 $
//
//   Author :
//   N. Neumeister            CERN EP
//   J. Troconiz              UAM Madrid
//
//--------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------

#include "L1Trigger/DTTrackFinder/src/L1MuDTAssignmentUnit.h"

//---------------
// C++ Headers --
//---------------

#include <iostream>
#include <cmath>
#include <cassert>

//-------------------------------
// Collaborating Class Headers --
//-------------------------------

#include "L1Trigger/DTTrackFinder/src/L1MuDTTFConfig.h"
#include "L1Trigger/DTTrackFinder/src/L1MuDTSectorProcessor.h"
#include "L1Trigger/DTTrackFinder/src/L1MuDTDataBuffer.h"
#include "L1Trigger/DTTrackFinder/src/L1MuDTTrackSegPhi.h"
#include "L1Trigger/DTTrackFinder/src/L1MuDTTrackSegLoc.h"
#include "L1Trigger/DTTrackFinder/src/L1MuDTTrackAssembler.h"
#include "L1Trigger/DTTrackFinder/src/L1MuDTTrackAssParam.h"
#include "L1Trigger/DTTrackFinder/src/L1MuDTPhiLut.h"
#include "L1Trigger/DTTrackFinder/src/L1MuDTPtaLut.h"
#include "L1Trigger/DTTrackFinder/interface/L1MuDTTrack.h"

using namespace std;

// --------------------------------
//       class L1MuDTAssignmentUnit
//---------------------------------

//----------------
// Constructors --
//----------------

L1MuDTAssignmentUnit::L1MuDTAssignmentUnit(L1MuDTSectorProcessor& sp, int id) : 
                m_sp(sp), m_id(id), 
                m_addArray(), m_TSphi(), m_ptAssMethod(UNDEF) {

  m_TSphi.reserve(4);  // a track candidate can consist of max 4 TS 
  reset();

  // read look-up tables for phi and pt-assignment
  readPhiLuts();
  readPtaLuts();

}


//--------------
// Destructor --
//--------------

L1MuDTAssignmentUnit::~L1MuDTAssignmentUnit() {

  if ( thePhiLUTs ) delete thePhiLUTs;
  thePhiLUTs = 0;
  if ( thePtaLUTs ) delete thePtaLUTs;
  thePtaLUTs = 0;

}


//--------------
// Operations --
//--------------

//
// run Assignment Unit
//
void L1MuDTAssignmentUnit::run() {

  // enable track candidate
  m_sp.track(m_id)->enable();

  // set track class
  TrackClass tc = m_sp.TA()->trackClass(m_id);
  m_sp.track(m_id)->setTC(tc);

  // get relative addresses of matching track segments
  m_addArray = m_sp.TA()->address(m_id);
  m_sp.track(m_id)->setAddresses(m_addArray);

  // get track segments (track segment router)
  TSR();
  m_sp.track(m_id)->setTSphi(m_TSphi);

  // set bunch-crossing (use first track segment)
  vector<const L1MuDTTrackSegPhi*>::const_iterator iter = m_TSphi.begin();
  int bx = (*iter)->bx();
  m_sp.track(m_id)->setBx(bx);

  // assign phi
  PhiAU();

  // assign pt and charge
  PtAU();
  
  // assign quality
  QuaAU();
  
  // special hack for overlap region
  for ( iter = m_TSphi.begin(); iter != m_TSphi.end(); iter++ ) {
    int wheel = abs((*iter)->wheel());
    if ( wheel == 3 && (*iter)->etaFlag() ) m_sp.track(m_id)->disable();
  }

}


//
// reset Assignment Unit
//
void L1MuDTAssignmentUnit::reset() {

  m_addArray.reset();
  m_TSphi.clear();
  m_ptAssMethod = UNDEF;

}


//
// assign phi with 8 bit precision
//
void L1MuDTAssignmentUnit::PhiAU() {

  // calculate phi at station 2 using 8 bits (precision = 2.5 degrees) 

  int sh_phi  = 12 - thePhiLUTs->getPrecision().first;
  int sh_phib = 10 - thePhiLUTs->getPrecision().second;

  const L1MuDTTrackSegPhi* second = getTSphi(2);  // track segment at station 2
  const L1MuDTTrackSegPhi* first  = getTSphi(1);  // track segment at station 1
  const L1MuDTTrackSegPhi* forth  = getTSphi(4);  // track segment at station 4

  int phi2 = 0;         // phi-value at station 2
  int sector = 0;

  if ( second ) {
    phi2 = second->phi() >> sh_phi;
    sector = second->sector();
  }
  else if ( second == 0 && first ) {
    int bend_angle = first->phib() >> sh_phib;
    phi2 = ( first->phi() >> sh_phi ) + thePhiLUTs->getDeltaPhi(0,bend_angle);
    sector = first->sector();
  }
  else if ( second == 0 && forth ) {
    int bend_angle = forth->phib() >> sh_phib;
    phi2 = ( forth->phi() >> sh_phi ) + thePhiLUTs->getDeltaPhi(1,bend_angle);
    sector = forth->sector();
  }

  // get sector center in 8 bit coding
  int sector_8 = convertSector(sector);

  // convert phi to 2.5 degree precision
  int phi_precision = 4096 >> sh_phi;
  const double k = (180.0/(2.5*M_PI*static_cast<float>(phi_precision)));
  double phi_f = static_cast<double>(phi2);
  int phi_8 = static_cast<int>(floor(phi_f*k));     

  int phi = (sector_8 + phi_8 + 144)%144;

  m_sp.track(m_id)->setPhi(phi);

}


//
// assign pt with 5 bit precision
//
void L1MuDTAssignmentUnit::PtAU() {

  // get pt-assignment method as function of track class and TS phib values
  m_ptAssMethod = getPtMethod();

  // get input address for look-up table
  int bend_angle = getPtAddress(m_ptAssMethod);

  // retrieve pt value from look-up table
  int lut_idx = m_ptAssMethod;
  int pt = thePtaLUTs->getPt(lut_idx,bend_angle );

  m_sp.track(m_id)->setPt(pt);

  // assign charge
  int chsign = getCharge(m_ptAssMethod);
  int charge = ( bend_angle >= 0 ) ? chsign : -1 * chsign;
  m_sp.track(m_id)->setCharge(charge);

}


//
// assign 3 bit quality code
//
void L1MuDTAssignmentUnit::QuaAU() {

  unsigned int quality = 0;
  
  const TrackClass tc = m_sp.TA()->trackClass(m_id);
  
  switch ( tc ) {
    case T1234 : { quality = 7; break; }
    case T123  : { quality = 6; break; }
    case T124  : { quality = 6; break; }
    case T134  : { quality = 5; break; }
    case T234  : { quality = 4; break; }
    case T12   : { quality = 3; break; }
    case T13   : { quality = 3; break; }
    case T14   : { quality = 3; break; } 
    case T23   : { quality = 2; break; }
    case T24   : { quality = 2; break; } 
    case T34   : { quality = 1; break; }
    default    : { quality = 0; break; }
  }

  m_sp.track(m_id)->setQuality(quality);

}


//
// Track Segment Router (TSR)
//
void L1MuDTAssignmentUnit::TSR() {

  // get the track segments from the data buffer 
  const L1MuDTTrackSegPhi* ts = 0;
  for ( int stat = 1; stat <= 4; stat++ ) {
    int adr = m_addArray.station(stat);
    if ( adr != 15 ) {
      ts = m_sp.data()->getTSphi(stat,adr);
      if ( ts != 0 ) m_TSphi.push_back( ts );
    }
  }

}


//
// get track segment from a given station
//
const L1MuDTTrackSegPhi* L1MuDTAssignmentUnit::getTSphi(int station) const {

  vector<const L1MuDTTrackSegPhi*>::const_iterator iter;
  for ( iter = m_TSphi.begin(); iter != m_TSphi.end(); iter++ ) {
    int stat = (*iter)->station();
    if ( station == stat ) {
      return (*iter);
      break;
    }
  }

  return 0;

}


//
// convert sector Id to a precision of 2.5 degrees using 8 bits (= sector center)
//
int L1MuDTAssignmentUnit::convertSector(int sector) {

  assert( sector >=0 && sector < 12 );
  const int sectorvalues[12] = {  0,  12,  24,  36, 48, 60, 72, 84, 
                                 96, 108, 120, 132 };

  return sectorvalues[sector];

}


//
// determine charge
//
int L1MuDTAssignmentUnit::getCharge(L1MuDTAssignmentUnit::PtAssMethod method) {

  int chargesign = 0;
  switch ( method ) {
    case PT12L  : { chargesign = -1; break; }
    case PT12H  : { chargesign = -1; break; }
    case PT13L  : { chargesign = -1; break; }
    case PT13H  : { chargesign = -1; break; }
    case PT14L  : { chargesign = -1; break; }
    case PT14H  : { chargesign = -1; break; }
    case PT23L  : { chargesign = -1; break; }
    case PT23H  : { chargesign = -1; break; }
    case PT24L  : { chargesign = -1; break; }
    case PT24H  : { chargesign = -1; break; }
    case PT34L  : { chargesign =  1; break; }
    case PT34H  : { chargesign =  1; break; }
    case PT12LO : { chargesign = -1; break; }
    case PT12HO : { chargesign = -1; break; }
    case PT13LO : { chargesign = -1; break; }
    case PT13HO : { chargesign = -1; break; }
    case PT14LO : { chargesign = -1; break; }
    case PT14HO : { chargesign = -1; break; }
    case PT23LO : { chargesign = -1; break; }
    case PT23HO : { chargesign = -1; break; }
    case PT24LO : { chargesign = -1; break; }
    case PT24HO : { chargesign = -1; break; }
    case PT34LO : { chargesign =  1; break; }
    case PT34HO : { chargesign =  1; break; }
    case PT15LO : { chargesign = -1; break; }
    case PT15HO : { chargesign = -1; break; }
    case PT25LO : { chargesign = -1; break; }
    case PT25HO : { chargesign = -1; break; }    
    case UNDEF  : { chargesign = 0; 
                    cerr << "AssignmentUnit::getCharge : undefined PtAssMethod!"
                         << endl;
                    break;
                  }
  }

  return chargesign;

}


//
// determine pt-assignment method
//
L1MuDTAssignmentUnit::PtAssMethod L1MuDTAssignmentUnit::getPtMethod() const {
   
  // determine which pt-assignment method should be used as a function 
  // of the track class and
  // of the phib values of the track segments making up this track candidate.

  // get bitmap of track candidate
  const bitset<4> s = m_sp.TA()->trackBitMap(m_id);
  
  int method = -1;
  
  if (  s.test(0) &&  s.test(3) ) method = 2; // stations 1 and 4
  if (  s.test(0) &&  s.test(2) ) method = 1; // stations 1 and 3
  if (  s.test(0) &&  s.test(1) ) method = 0; // stations 1 and 2
  if ( !s.test(0) &&  s.test(1) && s.test(3) ) method = 4; // stations 2 and 4
  if ( !s.test(0) &&  s.test(1) && s.test(2) ) method = 3; // stations 2 and 3
  if ( !s.test(0) && !s.test(1) && s.test(2) && s.test(3) ) method = 5; // stations 3 and 4

  if ( m_sp.ovl() ) {
    int adr = m_addArray.station(3);
    bool s5 = (adr == 15) ? false : ((adr/2)%2 == 1);    
    if (  s.test(0) &&  s.test(3) ) method = 8;  // stations 1 and 4
    if (  s.test(0) &&  s.test(2) &&  s5 ) method = 12; // stations 1 and 5
    if (  s.test(0) &&  s.test(2) && !s5 ) method = 7;  // stations 1 and 3
    if (  s.test(0) &&  s.test(1) ) method = 6;  // stations 1 and 2
    if ( !s.test(0) &&  s.test(1) && s.test(3) ) method = 10; // stations 2 and 4
    if ( !s.test(0) &&  s.test(1) && s.test(2) &&  s5 ) method = 13; // stations 2 and 5
    if ( !s.test(0) &&  s.test(1) && s.test(2) && !s5 ) method = 9;  // stations 2 and 3
    if ( !s.test(0) && !s.test(1) && s.test(2) &&  s.test(3) ) method = 11; // stations 3 and 4
  }

  int threshold = thePtaLUTs->getPtLutThreshold(method);
  
  // phib values of track segments from stations 1, 2 and 4
  int phib1 = ( getTSphi(1) != 0 ) ? getTSphi(1)->phib() : 0;
  int phib2 = ( getTSphi(2) != 0 ) ? getTSphi(2)->phib() : 0;
  int phib4 = ( getTSphi(4) != 0 ) ? getTSphi(4)->phib() : 0;

  L1MuDTAssignmentUnit::PtAssMethod pam = UNDEF;
  
  switch ( method ) {
    case 0 :  { pam = ( abs(phib1) <= threshold ) ? PT12H  : PT12L;  break; }
    case 1 :  { pam = ( abs(phib1) <= threshold ) ? PT13H  : PT13L;  break; }
    case 2 :  { pam = ( abs(phib1) <= threshold ) ? PT14H  : PT14L;  break; }
    case 3 :  { pam = ( abs(phib2) <= threshold ) ? PT23H  : PT23L;  break; }
    case 4 :  { pam = ( abs(phib2) <= threshold ) ? PT24H  : PT24L;  break; }
    case 5 :  { pam = ( abs(phib4) <= threshold ) ? PT34H  : PT34L;  break; }
    case 6 :  { pam = ( abs(phib1) <= threshold ) ? PT12HO : PT12LO; break; }
    case 7 :  { pam = ( abs(phib1) <= threshold ) ? PT13HO : PT13LO; break; }
    case 8 :  { pam = ( abs(phib1) <= threshold ) ? PT14HO : PT14LO; break; }
    case 9 :  { pam = ( abs(phib2) <= threshold ) ? PT23HO : PT23LO; break; }
    case 10 : { pam = ( abs(phib2) <= threshold ) ? PT24HO : PT24LO; break; }
    case 11 : { pam = ( abs(phib4) <= threshold ) ? PT34HO : PT34LO; break; }
    case 12 : { pam = ( abs(phib4) <= threshold ) ? PT15HO : PT15LO; break; }
    case 13 : { pam = ( abs(phib4) <= threshold ) ? PT25HO : PT25LO; break; }
    default : cout << "L1MuDTAssignmentUnit : Error in PT ass method evaluation" << endl;
  }
              
  return pam;

}


//
// calculate bend angle
//
int L1MuDTAssignmentUnit::getPtAddress(L1MuDTAssignmentUnit::PtAssMethod method) const {

  // calculate bend angle as difference of two azimuthal positions 

  int bendangle = 0;
  switch (method) {
    case PT12L  : { bendangle = phiDiff(1,2); break; }
    case PT12H  : { bendangle = phiDiff(1,2); break; }
    case PT13L  : { bendangle = phiDiff(1,3); break; }
    case PT13H  : { bendangle = phiDiff(1,3); break; }
    case PT14L  : { bendangle = phiDiff(1,4); break; }
    case PT14H  : { bendangle = phiDiff(1,4); break; }
    case PT23L  : { bendangle = phiDiff(2,3); break; }
    case PT23H  : { bendangle = phiDiff(2,3); break; }
    case PT24L  : { bendangle = phiDiff(2,4); break; }
    case PT24H  : { bendangle = phiDiff(2,4); break; }
    case PT34L  : { bendangle = phiDiff(4,3); break; }
    case PT34H  : { bendangle = phiDiff(4,3); break; }
    case PT12LO : { bendangle = phiDiff(1,2); break; }
    case PT12HO : { bendangle = phiDiff(1,2); break; }
    case PT13LO : { bendangle = phiDiff(1,3); break; }
    case PT13HO : { bendangle = phiDiff(1,3); break; }
    case PT14LO : { bendangle = phiDiff(1,4); break; }
    case PT14HO : { bendangle = phiDiff(1,4); break; }
    case PT23LO : { bendangle = phiDiff(2,3); break; }
    case PT23HO : { bendangle = phiDiff(2,3); break; }
    case PT24LO : { bendangle = phiDiff(2,4); break; }
    case PT24HO : { bendangle = phiDiff(2,4); break; }
    case PT34LO : { bendangle = phiDiff(4,3); break; }
    case PT34HO : { bendangle = phiDiff(4,3); break; }    
    case PT15LO : { bendangle = phiDiff(1,3); break; }
    case PT15HO : { bendangle = phiDiff(1,3); break; }
    case PT25LO : { bendangle = phiDiff(2,3); break; }
    case PT25HO : { bendangle = phiDiff(2,3); break; }        
    case UNDEF :  { bendangle = 0;
                    cerr << "AssignmentUnit::getPtAddress : undefined PtAssMethod" << endl;
                    break;
                  }
  }

  int signo = 1;
  if (bendangle < 0) signo=-1;
  bendangle = signo*bendangle;
  bendangle = bendangle%1024;
  if (bendangle > 511) bendangle=1023-bendangle;
  bendangle = signo*bendangle;

  return bendangle;

}


//
// build difference of two phi values
//
int L1MuDTAssignmentUnit::phiDiff(int stat1, int stat2) const {

  // calculate bit shift

  int sh_phi  = 12 - nbit_phi;

  // get 2 phi values and add offset (30 degrees ) for adjacent sector
  int sector1 = getTSphi(stat1)->sector();
  int sector2 = getTSphi(stat2)->sector();
  int phi1 = getTSphi(stat1)->phi() >> sh_phi;
  int phi2 = getTSphi(stat2)->phi() >> sh_phi; 

  // convert sector difference to values in the range -6 to +5

  int sectordiff = (sector2 - sector1)%12;
  if ( sectordiff >= 6 ) sectordiff -= 12;
  if ( sectordiff < -6 ) sectordiff += 12;
  
  assert( abs(sectordiff) <= 1 );
  
  int offset = (2144 >> sh_phi) * sectordiff;
  int bendangle = (phi2 - phi1 + offset);

  return bendangle;
    
}    
    

//
// read phi-assignment look-up tables
//
void  L1MuDTAssignmentUnit::readPhiLuts() {

  if ( thePhiLUTs == 0 ) thePhiLUTs = new L1MuDTPhiLut;

}


//
// read pt-assignment look-up tables
//
void  L1MuDTAssignmentUnit::readPtaLuts() {

  if ( thePtaLUTs == 0 ) {
    thePtaLUTs = new L1MuDTPtaLut;
    setPrecision();
  }
    
}


//
// overload output stream operator for PtAssMethod
//
ostream& operator<<( ostream& s, L1MuDTAssignmentUnit::PtAssMethod method) {

  switch (method) {
    case L1MuDTAssignmentUnit::PT12L  : { return s << "PT12L "; break; }
    case L1MuDTAssignmentUnit::PT12H  : { return s << "PT12H "; break; }
    case L1MuDTAssignmentUnit::PT13L  : { return s << "PT13L "; break; }
    case L1MuDTAssignmentUnit::PT13H  : { return s << "PT13H "; break; }
    case L1MuDTAssignmentUnit::PT14L  : { return s << "PT14L "; break; }
    case L1MuDTAssignmentUnit::PT14H  : { return s << "PT14H "; break; }
    case L1MuDTAssignmentUnit::PT23L  : { return s << "PT23L "; break; }
    case L1MuDTAssignmentUnit::PT23H  : { return s << "PT23H "; break; }
    case L1MuDTAssignmentUnit::PT24L  : { return s << "PT24L "; break; }
    case L1MuDTAssignmentUnit::PT24H  : { return s << "PT24H "; break; }
    case L1MuDTAssignmentUnit::PT34L  : { return s << "PT34L "; break; }
    case L1MuDTAssignmentUnit::PT34H  : { return s << "PT34H "; break; }
    case L1MuDTAssignmentUnit::PT12LO : { return s << "PT12LO "; break; }
    case L1MuDTAssignmentUnit::PT12HO : { return s << "PT12HO "; break; }    
    case L1MuDTAssignmentUnit::PT13LO : { return s << "PT13LO "; break; }
    case L1MuDTAssignmentUnit::PT13HO : { return s << "PT13HO "; break; }    
    case L1MuDTAssignmentUnit::PT14LO : { return s << "PT14LO "; break; }
    case L1MuDTAssignmentUnit::PT14HO : { return s << "PT14HO "; break; }    
    case L1MuDTAssignmentUnit::PT23LO : { return s << "PT23LO "; break; }
    case L1MuDTAssignmentUnit::PT23HO : { return s << "PT23HO "; break; }
    case L1MuDTAssignmentUnit::PT24LO : { return s << "PT24LO "; break; }
    case L1MuDTAssignmentUnit::PT24HO : { return s << "PT24HO "; break; }
    case L1MuDTAssignmentUnit::PT34LO : { return s << "PT34LO "; break; }
    case L1MuDTAssignmentUnit::PT34HO : { return s << "PT34HO "; break; }
    case L1MuDTAssignmentUnit::PT15LO : { return s << "PT15LO "; break; }
    case L1MuDTAssignmentUnit::PT15HO : { return s << "PT15HO "; break; }
    case L1MuDTAssignmentUnit::PT25LO : { return s << "PT25LO "; break; }
    case L1MuDTAssignmentUnit::PT25HO : { return s << "PT25HO "; break; }
    default :
      return s << "unknown pt-assignment method ";
  }

}


//
// set precision for pt-assignment of phi and phib
// default is 12 bits for phi and 10 bits for phib
//
void L1MuDTAssignmentUnit::setPrecision() {

  nbit_phi  = L1MuDTTFConfig::getNbitsPtaPhi();
  nbit_phib = L1MuDTTFConfig::getNbitsPtaPhib();

}


// static data members

L1MuDTPhiLut* L1MuDTAssignmentUnit::thePhiLUTs = 0;
L1MuDTPtaLut* L1MuDTAssignmentUnit::thePtaLUTs = 0;
unsigned short int L1MuDTAssignmentUnit::nbit_phi  = 12;
unsigned short int L1MuDTAssignmentUnit::nbit_phib = 10;
