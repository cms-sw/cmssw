//-------------------------------------------------
//
//   Class: L1MuBMAssignmentUnit
//
//   Description: Assignment Unit
//
//
//
//   Author :
//   N. Neumeister            CERN EP
//   J. Troconiz              UAM Madrid
//   G. Flouris               U. Ioannina
//--------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------

#include "L1MuBMAssignmentUnit.h"

//---------------
// C++ Headers --
//---------------

#include <iostream>
#include <cmath>
#include <cassert>

//-------------------------------
// Collaborating Class Headers --
//-------------------------------

#include "L1MuBMTFConfig.h"
#include "L1MuBMSectorProcessor.h"
#include "L1MuBMDataBuffer.h"
#include "L1MuBMTrackSegPhi.h"
#include "L1MuBMTrackSegLoc.h"
#include "L1MuBMTrackAssembler.h"
#include "L1MuBMTrackAssParam.h"

#include <iostream>
#include <iomanip>

using namespace std;

// --------------------------------
//       class L1MuBMAssignmentUnit
//---------------------------------

//----------------
// Constructors --
//----------------

L1MuBMAssignmentUnit::L1MuBMAssignmentUnit(L1MuBMSectorProcessor& sp, int id) :
                m_sp(sp), m_id(id),
                m_addArray(), m_TSphi(), m_ptAssMethod(NODEF) {

  m_TSphi.reserve(4);  // a track candidate can consist of max 4 TS
  reset();
  setPrecision();

}


//--------------
// Destructor --
//--------------

L1MuBMAssignmentUnit::~L1MuBMAssignmentUnit() {
}


//--------------
// Operations --
//--------------

//
// run Assignment Unit
//
void L1MuBMAssignmentUnit::run(const edm::EventSetup& c) {

  // enable track candidate
  m_sp.track(m_id)->enable();
  m_sp.tracK(m_id)->enable();

  // set track class
  TrackClass tc = m_sp.TA()->trackClass(m_id);
  m_sp.track(m_id)->setTC(tc);
  m_sp.tracK(m_id)->setTC(tc);

  // get relative addresses of matching track segments
  m_addArray = m_sp.TA()->address(m_id);
  m_sp.track(m_id)->setAddresses(m_addArray);
  m_sp.tracK(m_id)->setAddresses(m_addArray);

  // get track segments (track segment router)
  TSR();
  m_sp.track(m_id)->setTSphi(m_TSphi);
  m_sp.tracK(m_id)->setTSphi(m_TSphi);

  // set bunch-crossing (use first track segment)
  vector<const L1MuBMTrackSegPhi*>::const_iterator iter = m_TSphi.begin();
  int bx = (*iter)->bx();
  m_sp.track(m_id)->setBx(bx);
  m_sp.tracK(m_id)->setBx(bx);

  // assign phi
  PhiAU(c);

  // assign pt and charge
  PtAU(c);

  // assign quality
  QuaAU();

}


//
// reset Assignment Unit
//
void L1MuBMAssignmentUnit::reset() {

  m_addArray.reset();
  m_TSphi.clear();
  m_ptAssMethod = NODEF;

}


//
// assign phi with 8 bit precision
//
void L1MuBMAssignmentUnit::PhiAU(const edm::EventSetup& c) {

  const L1TMuonBarrelParamsRcd& bmtfParamsRcd = c.get<L1TMuonBarrelParamsRcd>();
  bmtfParamsRcd.get(bmtfParamsHandle);
  const L1TMuonBarrelParams& bmtfParams = *bmtfParamsHandle.product();
  thePhiLUTs =  new L1MuBMPhiLut(bmtfParams);  ///< phi-assignment look-up tables
  //thePhiLUTs->print();
  // calculate phi at station 2 using 8 bits (precision = 0.625 degrees)
  int sh_phi  = 12 - L1MuBMTFConfig::getNbitsPhiPhi();
  int sh_phib = 10 - L1MuBMTFConfig::getNbitsPhiPhib();

  const L1MuBMTrackSegPhi* second = getTSphi(2);  // track segment at station 2
  const L1MuBMTrackSegPhi* first  = getTSphi(1);  // track segment at station 1
  const L1MuBMTrackSegPhi* forth  = getTSphi(4);  // track segment at station 4

  int phi2 = 0;         // phi-value at station 2
  int sector = 0;
  if ( second ) {
    phi2 = second->phi() >> sh_phi;
    sector = second->sector();
  }
  else if ( second == 0 && first ) {
    phi2 = first->phi() >> sh_phi;
    sector = first->sector();
  }
  else if ( second == 0 && forth ) {
    phi2 = forth->phi() >> sh_phi;
    sector = forth->sector();
  }

  int sector0 = m_sp.id().sector();

  // convert sector difference to values in the range -6 to +5

  int sectordiff = (sector - sector0)%12;
  if ( sectordiff >= 6 ) sectordiff -= 12;
  if ( sectordiff < -6 ) sectordiff += 12;

  // convert phi to 0.625 degree precision
  int phi_precision = 4096 >> sh_phi;
  const double k = 57.2958/0.625/static_cast<float>(phi_precision);
  double phi_f = static_cast<double>(phi2);
  int phi_8 = static_cast<int>(floor(phi_f*k));

  if ( second == 0 && first ) {
    int bend_angle = (first->phib() >> sh_phib) << sh_phib;
    phi_8 = phi_8 + thePhiLUTs->getDeltaPhi(0,bend_angle);
    //phi_8 = phi_8 + getDeltaPhi(0, bend_angle, bmtfParams->phi_lut());
  }
  else if ( second == 0 && forth ) {

    int bend_angle = (forth->phib() >> sh_phib) << sh_phib;
    phi_8 = phi_8 + thePhiLUTs->getDeltaPhi(1,bend_angle);
    //phi_8 = phi_8 + getDeltaPhi(1, bend_angle, bmtfParams->phi_lut());
  }

  //If muon is found at the neighbour sector - second station
  //a shift is needed by 48
  phi_8 += sectordiff*48;

  int phi = phi_8 + 24;
  if (phi >  55) phi =  55;
  if (phi < -8) phi = -8;

  m_sp.track(m_id)->setPhi(phi); // Regional
  m_sp.tracK(m_id)->setPhi(phi);

delete thePhiLUTs;
}


//
// assign pt with 5 bit precision
//
void L1MuBMAssignmentUnit::PtAU(const edm::EventSetup& c) {

  const L1TMuonBarrelParamsRcd& bmtfParamsRcd = c.get<L1TMuonBarrelParamsRcd>();
  bmtfParamsRcd.get(bmtfParamsHandle);
  const L1TMuonBarrelParams& bmtfParams = *bmtfParamsHandle.product();
  thePtaLUTs =  new L1MuBMPtaLut(bmtfParams);   ///< pt-assignment look-up tables
  //thePtaLUTs->print();
  // get pt-assignment method as function of track class and TS phib values
  //m_ptAssMethod = getPtMethod(bmtfParams);
  m_ptAssMethod = getPtMethod();
  // get input address for look-up table
  int bend_angle = getPtAddress(m_ptAssMethod);
  int bend_carga = getPtAddress(m_ptAssMethod, 1);

  // retrieve pt value from look-up table
  int lut_idx = m_ptAssMethod;
  int pt = thePtaLUTs->getPt(lut_idx,bend_angle );
  //int pt = getPt(lut_idx, bend_angle, bmtfParams->pta_lut());
  m_sp.track(m_id)->setPt(pt);
  m_sp.tracK(m_id)->setPt(pt);

  // assign charge
  int chsign = getCharge(m_ptAssMethod);
  int charge = ( bend_carga >= 0 ) ? chsign : -1 * chsign;
  m_sp.track(m_id)->setCharge(charge);
  m_sp.tracK(m_id)->setCharge(charge);
  delete thePtaLUTs;

}


//
// assign 3 bit quality code
//
void L1MuBMAssignmentUnit::QuaAU() {

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
  m_sp.tracK(m_id)->setQuality(quality);

}


//
// Track Segment Router (TSR)
//
void L1MuBMAssignmentUnit::TSR() {

  // get the track segments from the data buffer
  const L1MuBMTrackSegPhi* ts = 0;
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
const L1MuBMTrackSegPhi* L1MuBMAssignmentUnit::getTSphi(int station) const {

  vector<const L1MuBMTrackSegPhi*>::const_iterator iter;
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
int L1MuBMAssignmentUnit::convertSector(int sector) {

  //  assert( sector >=0 && sector < 12 );
  const int sectorvalues[12] = {  0,  12,  24,  36, 48, 60, 72, 84,
                                 96, 108, 120, 132 };

  return sectorvalues[sector];

}


//
// determine charge
//
int L1MuBMAssignmentUnit::getCharge(PtAssMethod method) {

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

    case NODEF  : { chargesign = 0;
    //                    cerr << "AssignmentUnit::getCharge : undefined PtAssMethod!"
    //                         << endl;
                    break;
                  }
  }

  return chargesign;

}


//
// determine pt-assignment method
//
//PtAssMethod L1MuBMAssignmentUnit::getPtMethod(L1TMuonBarrelParams *l1tbmparams) const {
PtAssMethod L1MuBMAssignmentUnit::getPtMethod() const {

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
  int threshold = thePtaLUTs->getPtLutThreshold(method);

  // phib values of track segments from stations 1, 2 and 4
  int phib1 = ( getTSphi(1) != 0 ) ? getTSphi(1)->phib() : 0;
  int phib2 = ( getTSphi(2) != 0 ) ? getTSphi(2)->phib() : 0;
  int phib4 = ( getTSphi(4) != 0 ) ? getTSphi(4)->phib() : 0;

  PtAssMethod pam = NODEF;

  switch ( method ) {
    case 0 :  { pam = ( abs(phib1) < threshold ) ? PT12H  : PT12L;  break; }
    case 1 :  { pam = ( abs(phib1) < threshold ) ? PT13H  : PT13L;  break; }
    case 2 :  { pam = ( abs(phib1) < threshold ) ? PT14H  : PT14L;  break; }
    case 3 :  { pam = ( abs(phib2) < threshold ) ? PT23H  : PT23L;  break; }
    case 4 :  { pam = ( abs(phib2) < threshold ) ? PT24H  : PT24L;  break; }
    case 5 :  { pam = ( abs(phib4) < threshold ) ? PT34H  : PT34L;  break; }
    default : ;
      //cout << "L1MuBMAssignmentUnit : Error in PT ass method evaluation" << endl;
  }

  return pam;

}


//
// calculate bend angle
//
int L1MuBMAssignmentUnit::getPtAddress(PtAssMethod method, int bendcharge) const {

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
    case NODEF :  { bendangle = 0;
    //                    cerr << "AssignmentUnit::getPtAddress : undefined PtAssMethod" << endl;
                    break;
                  }
  }

  int signo = 1;
  bendangle = (bendangle+8192)%4096;
  if ( bendangle > 2047 ) bendangle -= 4096;
  if ( bendangle < 0 ) signo = -1;

  if (bendcharge) return signo;

  bendangle = (bendangle+2048)%1024;
  if ( bendangle > 511 ) bendangle -= 1024;

  return bendangle;

}


//
// build difference of two phi values
//
int L1MuBMAssignmentUnit::phiDiff(int stat1, int stat2) const {

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

  //  assert( abs(sectordiff) <= 1 );

  int offset = (2144 >> sh_phi) * sectordiff;
  int bendangle = (phi2 - phi1 + offset) << sh_phi;

  return bendangle;

}


//
// set precision for pt-assignment of phi and phib
// default is 12 bits for phi and 10 bits for phib
//
void L1MuBMAssignmentUnit::setPrecision() {

  nbit_phi  = L1MuBMTFConfig::getNbitsPtaPhi();
  nbit_phib = L1MuBMTFConfig::getNbitsPtaPhib();

}


// static data members

unsigned short int L1MuBMAssignmentUnit::nbit_phi  = 12;
unsigned short int L1MuBMAssignmentUnit::nbit_phib = 10;
