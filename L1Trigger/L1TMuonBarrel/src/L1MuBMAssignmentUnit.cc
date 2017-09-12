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
//   Modifications:
//   G. Flouris               U. Ioannina
//   G Karathanasis           U. Athens
//--------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------

#include "L1Trigger/L1TMuonBarrel/src/L1MuBMAssignmentUnit.h"

//---------------
// C++ Headers --
//---------------

#include <iostream>
#include <cmath>
#include <cassert>

//-------------------------------
// Collaborating Class Headers --
//-------------------------------

#include "L1Trigger/L1TMuonBarrel/src/L1MuBMTFConfig.h"
#include "L1Trigger/L1TMuonBarrel/src/L1MuBMSectorProcessor.h"
#include "L1Trigger/L1TMuonBarrel/src/L1MuBMDataBuffer.h"
#include "L1Trigger/L1TMuonBarrel/src/L1MuBMTrackAssembler.h"
#include "DataFormats/L1TMuon/interface/L1MuBMTrackSegPhi.h"
#include "DataFormats/L1TMuon/interface/BMTF/L1MuBMTrackSegLoc.h"
#include "DataFormats/L1TMuon/interface/BMTF/L1MuBMTrackAssParam.h"

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
                m_addArray(), m_TSphi(), m_ptAssMethod(L1MuBMLUTHandler::NODEF ) {

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
  m_ptAssMethod = L1MuBMLUTHandler::NODEF;

}


//
// assign phi with 8 bit precision
//
void L1MuBMAssignmentUnit::PhiAU(const edm::EventSetup& c) {

  const L1TMuonBarrelParamsRcd& bmtfParamsRcd = c.get<L1TMuonBarrelParamsRcd>();
  bmtfParamsRcd.get(bmtfParamsHandle);
  const L1TMuonBarrelParams& bmtfParams = *bmtfParamsHandle.product();
  thePhiLUTs =  new L1MuBMLUTHandler(bmtfParams);  ///< phi-assignment look-up tables
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
  else if ( second == nullptr && first ) {
    phi2 = first->phi() >> sh_phi;
    sector = first->sector();
  }
  else if ( second == nullptr && forth ) {
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
   int bit_div_phi=static_cast<int>(phi2)%4;
    if (bit_div_phi<0) bit_div_phi+=4;
   phi_f=phi_f-std::abs(bit_div_phi);
  int phi_8 = static_cast<int>(floor(phi_f*k));

  if ( second == nullptr && first ) {
    int bend_angle = (first->phib() >> sh_phib) << sh_phib;
    phi_8 = phi_8 + thePhiLUTs->getDeltaPhi(0,bend_angle);
    //phi_8 = phi_8 + getDeltaPhi(0, bend_angle, bmtfParams->phi_lut());
  }
  else if ( second == nullptr && forth ) {

    int bend_angle = (forth->phib() >> sh_phib) << sh_phib;
    phi_8 = phi_8 + thePhiLUTs->getDeltaPhi(1,bend_angle);
    //phi_8 = phi_8 + getDeltaPhi(1, bend_angle, bmtfParams->phi_lut());
  }

  //If muon is found at the neighbour sector - second station
  //a shift is needed by 48
  phi_8 += sectordiff*48;

  int phi = phi_8 + 24;
  // 78 phi bins (-8 to 69) correspond 30 degree sector plus
  // additional lower and higher bins for neighboring sectors.
  if (phi >  69) phi =  69;
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
  const L1TMuonBarrelParams& bmtfParams1 = *bmtfParamsHandle.product();
  const L1TMuonBarrelParamsAllPublic& bmtfParams = L1TMuonBarrelParamsAllPublic(bmtfParams1);
  thePtaLUTs =  new L1MuBMLUTHandler(bmtfParams);   ///< pt-assignment look-up tables
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

if(!bmtfParams.get_DisableNewAlgo()){

    if (Quality() < 4) {
    int ptj = pt;
    L1MuBMLUTHandler::PtAssMethod jj1 = getPt1Method(m_ptAssMethod);
    L1MuBMLUTHandler::PtAssMethod jj2 = getPt2Method(m_ptAssMethod);
    if (jj1 != L1MuBMLUTHandler::NODEF) {
      lut_idx = jj1;
      bend_angle = getPt1Address(m_ptAssMethod);
      if (abs(bend_angle) < 512) ptj = thePtaLUTs->getPt(lut_idx,bend_angle );
    }
    else if (jj2 != L1MuBMLUTHandler::NODEF) {
      lut_idx = jj2;
      bend_angle = getPt2Address(m_ptAssMethod);
      if (abs(bend_angle) < 512) ptj = thePtaLUTs->getPt(lut_idx,bend_angle );
    }
    if (ptj < pt) pt = ptj;
  }
}

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
// assign 4 bit quality code
//
void L1MuBMAssignmentUnit::QuaAU() {

  unsigned int quality = 0;

  const TrackClass tc = m_sp.TA()->trackClass(m_id);

  ///Two LSBs of BMTF Q = Nstations-1
  switch ( tc ) {
    case T1234 : { quality = 3; break; }
    case T123  : { quality = 2; break; }
    case T124  : { quality = 2; break; }
    case T134  : { quality = 2; break; }
    case T234  : { quality = 2; break; }
    case T12   : { quality = 1; break; }
    case T13   : { quality = 1; break; }
    case T14   : { quality = 1; break; }
    case T23   : { quality = 0; break; }
    case T24   : { quality = 0; break; }
    case T34   : { quality = 0; break; }
    default    : { quality = 0; break; }
  }

 ///Two MSB of BMTF Q = 11
 quality += 12;

  m_sp.track(m_id)->setQuality(quality);
  m_sp.tracK(m_id)->setQuality(quality);

}


//
// assign 3 bit quality code
//
unsigned int L1MuBMAssignmentUnit::Quality() {

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
    default    : { quality = 0; }
  }

  return quality;

}

//
// Track Segment Router (TSR)
//
void L1MuBMAssignmentUnit::TSR() {

  // get the track segments from the data buffer
  const L1MuBMTrackSegPhi* ts = nullptr;
  for ( int stat = 1; stat <= 4; stat++ ) {
    int adr = m_addArray.station(stat);
    if ( adr != 15 ) {
      ts = m_sp.data()->getTSphi(stat,adr);
      if ( ts != nullptr ) m_TSphi.push_back( ts );
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

  return nullptr;

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
int L1MuBMAssignmentUnit::getCharge(L1MuBMLUTHandler::PtAssMethod method) {

  int chargesign = 0;
  switch ( method ) {
    case L1MuBMLUTHandler::PT12L  : { chargesign = -1; break; }
    case L1MuBMLUTHandler::PT12H  : { chargesign = -1; break; }
    case L1MuBMLUTHandler::PT13L  : { chargesign = -1; break; }
    case L1MuBMLUTHandler::PT13H  : { chargesign = -1; break; }
    case L1MuBMLUTHandler::PT14L  : { chargesign = -1; break; }
    case L1MuBMLUTHandler::PT14H  : { chargesign = -1; break; }
    case L1MuBMLUTHandler::PT23L  : { chargesign = -1; break; }
    case L1MuBMLUTHandler::PT23H  : { chargesign = -1; break; }
    case L1MuBMLUTHandler::PT24L  : { chargesign = -1; break; }
    case L1MuBMLUTHandler::PT24H  : { chargesign = -1; break; }
    case L1MuBMLUTHandler::PT34L  : { chargesign =  1; break; }
    case L1MuBMLUTHandler::PT34H  : { chargesign =  1; break; }

    case L1MuBMLUTHandler::NODEF  : { chargesign = 0;
    //                    cerr << "AssignmentUnit::getCharge : undefined PtAssMethod!"
    //                         << endl;
                    break;
                  }
    default     : { chargesign = 0; }
  }

  return chargesign;

}


//
// determine pt-assignment method
//
//PtAssMethod L1MuBMAssignmentUnit::getPtMethod(L1TMuonBarrelParams *l1tbmparams) const {
L1MuBMLUTHandler::PtAssMethod L1MuBMAssignmentUnit::getPtMethod() const {

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
  int phib1 = ( getTSphi(1) != nullptr ) ? getTSphi(1)->phib() : 0;
  int phib2 = ( getTSphi(2) != nullptr ) ? getTSphi(2)->phib() : 0;
  int phib4 = ( getTSphi(4) != nullptr ) ? getTSphi(4)->phib() : 0;

  L1MuBMLUTHandler::PtAssMethod pam = L1MuBMLUTHandler::NODEF;

  switch ( method ) {
    case 0 :  { pam = ( abs(phib1) < threshold ) ? L1MuBMLUTHandler::PT12H  : L1MuBMLUTHandler::PT12L;  break; }
    case 1 :  { pam = ( abs(phib1) < threshold ) ? L1MuBMLUTHandler::PT13H  : L1MuBMLUTHandler::PT13L;  break; }
    case 2 :  { pam = ( abs(phib1) < threshold ) ? L1MuBMLUTHandler::PT14H  : L1MuBMLUTHandler::PT14L;  break; }
    case 3 :  { pam = ( abs(phib2) < threshold ) ? L1MuBMLUTHandler::PT23H  : L1MuBMLUTHandler::PT23L;  break; }
    case 4 :  { pam = ( abs(phib2) < threshold ) ? L1MuBMLUTHandler::PT24H  : L1MuBMLUTHandler::PT24L;  break; }
    case 5 :  { pam = ( abs(phib4) < threshold ) ? L1MuBMLUTHandler::PT34H  : L1MuBMLUTHandler::PT34L;  break; }
    default : ;
      //cout << "L1MuBMAssignmentUnit : Error in PT ass method evaluation" << endl;
  }

  return pam;

}


//
// calculate bend angle
//
int L1MuBMAssignmentUnit::getPtAddress(L1MuBMLUTHandler::PtAssMethod method, int bendcharge) const {

  // calculate bend angle as difference of two azimuthal positions

  int bendangle = 0;
  switch (method) {
    case L1MuBMLUTHandler::PT12L  : { bendangle = phiDiff(1,2); break; }
    case L1MuBMLUTHandler::PT12H  : { bendangle = phiDiff(1,2); break; }
    case L1MuBMLUTHandler::PT13L  : { bendangle = phiDiff(1,3); break; }
    case L1MuBMLUTHandler::PT13H  : { bendangle = phiDiff(1,3); break; }
    case L1MuBMLUTHandler::PT14L  : { bendangle = phiDiff(1,4); break; }
    case L1MuBMLUTHandler::PT14H  : { bendangle = phiDiff(1,4); break; }
    case L1MuBMLUTHandler::PT23L  : { bendangle = phiDiff(2,3); break; }
    case L1MuBMLUTHandler::PT23H  : { bendangle = phiDiff(2,3); break; }
    case L1MuBMLUTHandler::PT24L  : { bendangle = phiDiff(2,4); break; }
    case L1MuBMLUTHandler::PT24H  : { bendangle = phiDiff(2,4); break; }
    case L1MuBMLUTHandler::PT34L  : { bendangle = phiDiff(4,3); break; }
    case L1MuBMLUTHandler::PT34H  : { bendangle = phiDiff(4,3); break; }
    case L1MuBMLUTHandler::NODEF :  { bendangle = 0;
    //                    cerr << "AssignmentUnit::getPtAddress : undefined PtAssMethod" << endl;
                    break;
                  }
    default     : { bendangle = 0; }

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
// determine pt-assignment method
//
L1MuBMLUTHandler::PtAssMethod L1MuBMAssignmentUnit::getPt1Method(L1MuBMLUTHandler::PtAssMethod method) const {

  // quality values of track segments from stations 1, 2 and 4
  int qual1 = ( getTSphi(1) != nullptr ) ? getTSphi(1)->quality() : 0;
  int qual2 = ( getTSphi(2) != nullptr ) ? getTSphi(2)->quality() : 0;
  int qual4 = ( getTSphi(4) != nullptr ) ? getTSphi(4)->quality() : 0;

  L1MuBMLUTHandler::PtAssMethod pam = L1MuBMLUTHandler::NODEF;

  switch ( method ) {
    case L1MuBMLUTHandler::PT12H  : { if (qual1 > 3) pam = L1MuBMLUTHandler::PB12H;  break; }
    case L1MuBMLUTHandler::PT13H  : { if (qual1 > 3) pam = L1MuBMLUTHandler::PB13H;  break; }
    case L1MuBMLUTHandler::PT14H  : { if (qual1 > 3) pam = L1MuBMLUTHandler::PB14H;  break; }
    case L1MuBMLUTHandler::PT23H  : { if (qual2 > 3) pam = L1MuBMLUTHandler::PB23H;  break; }
    case L1MuBMLUTHandler::PT24H  : { if (qual2 > 3) pam = L1MuBMLUTHandler::PB24H;  break; }
    case L1MuBMLUTHandler::PT34H  : { if (qual4 > 3) pam = L1MuBMLUTHandler::PB34H;  break; }
    case L1MuBMLUTHandler::NODEF  : { pam = L1MuBMLUTHandler::NODEF; break;}
    default     : { pam = L1MuBMLUTHandler::NODEF; }
  }

  return pam;

}


//
// determine pt-assignment method
//
L1MuBMLUTHandler::PtAssMethod L1MuBMAssignmentUnit::getPt2Method(L1MuBMLUTHandler::PtAssMethod method) const {

  // quality values of track segments from stations 2 and 4
  int qual2 = ( getTSphi(2) != nullptr ) ? getTSphi(2)->quality() : 0;
  //  int qual4 = ( getTSphi(4) != 0 ) ? getTSphi(4)->quality() : 0;

  L1MuBMLUTHandler::PtAssMethod pam = L1MuBMLUTHandler::NODEF;

  switch ( method ) {
    case L1MuBMLUTHandler::PT12H  : { if (qual2 > 3) pam = L1MuBMLUTHandler::PB21H;  break; }
      //    case PT14H  : { if (qual4 > 3) pam = PB34H;  break; }
      //    case PT24H  : { if (qual4 > 3) pam = PB34H;  break; }
    //case PT12HO : { if (qual2 > 3) pam = PB21HO; break; }
      //    case PT14HO : { if (qual4 > 3) pam = PB34HO; break; }
      //    case PT24HO : { if (qual4 > 3) pam = PB34HO; break; }
    case L1MuBMLUTHandler::NODEF  : { pam = L1MuBMLUTHandler::NODEF; break; }
    default     : { pam = L1MuBMLUTHandler::NODEF; }
  }

  return pam;

}


//
// calculate bend angle
//
int L1MuBMAssignmentUnit::getPt1Address(L1MuBMLUTHandler::PtAssMethod method) const {

  // phib values of track segments from stations 1, 2 and 4
  int phib1 = ( getTSphi(1) != nullptr ) ? getTSphi(1)->phib() : -999;
  int phib2 = ( getTSphi(2) != nullptr ) ? getTSphi(2)->phib() : -999;
  int phib4 = ( getTSphi(4) != nullptr ) ? getTSphi(4)->phib() : -999;


  int bendangle = -999;
  switch (method) {
    case L1MuBMLUTHandler::PT12H  : { bendangle = phib1;  break; }
    case L1MuBMLUTHandler::PT13H  : { bendangle = phib1;  break; }
    case L1MuBMLUTHandler::PT14H  : { bendangle = phib1;  break; }
    case L1MuBMLUTHandler::PT23H  : { bendangle = phib2;  break; }
    case L1MuBMLUTHandler::PT24H  : { bendangle = phib2;  break; }
    case L1MuBMLUTHandler::PT34H  : { bendangle = phib4;  break; }
    case L1MuBMLUTHandler::NODEF  : { bendangle = -999; break; }
    default     : { bendangle = -999; }
  }

  return bendangle;

}


//
// calculate bend angle
//
int L1MuBMAssignmentUnit::getPt2Address(L1MuBMLUTHandler::PtAssMethod method) const {

  // phib values of track segments from stations 1, 2 and 4
  int phib2 = ( getTSphi(2) != nullptr ) ? getTSphi(2)->phib() : -999;
  int phib4 = ( getTSphi(4) != nullptr ) ? getTSphi(4)->phib() : -999;


  int bendangle = -999;
  switch (method) {
    case L1MuBMLUTHandler::PT12H  : { bendangle = phib2;  break; }
    case L1MuBMLUTHandler::PT14H  : { bendangle = phib4;  break; }
    case L1MuBMLUTHandler::PT24H  : { bendangle = phib4;  break; }
    //case L1MuBMLUTHandler::PT12HO : { bendangle = phib2;  break; }
    //case L1MuBMLUTHandler::PT14HO : { bendangle = phib4;  break; }
    //case L1MuBMLUTHandler::PT24HO : { bendangle = phib4;  break; }
    case L1MuBMLUTHandler::NODEF  : { bendangle = -999; break; }
    default     : { bendangle = -999; }
  }

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
