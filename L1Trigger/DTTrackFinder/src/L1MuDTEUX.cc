//-------------------------------------------------
//
//   Class: L1MuDTEUX
//
//   Description: Extrapolator
//
//
//   $Date: 2008/10/13 07:44:43 $
//   $Revision: 1.9 $
//
//   Author :
//   N. Neumeister            CERN EP
//
//--------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------

#include "L1Trigger/DTTrackFinder/src/L1MuDTEUX.h"

//---------------
// C++ Headers --
//---------------

#include <iostream>

//-------------------------------
// Collaborating Class Headers --
//-------------------------------

#include "L1Trigger/DTTrackFinder/src/L1MuDTTFConfig.h"
#include "CondFormats/L1TObjects/interface/L1MuDTExtParam.h"
#include "L1Trigger/DTTrackFinder/src/L1MuDTSectorProcessor.h"
#include "L1Trigger/DTTrackFinder/src/L1MuDTSEU.h"
#include "L1Trigger/DTTrackFinder/src/L1MuDTTrackSegPhi.h"
#include "CondFormats/L1TObjects/interface/L1MuDTExtLut.h"
#include "CondFormats/DataRecord/interface/L1MuDTExtLutRcd.h"
#include "CondFormats/L1TObjects/interface/L1MuDTTFParameters.h"
#include "CondFormats/DataRecord/interface/L1MuDTTFParametersRcd.h"

using namespace std;

// --------------------------------
//       class L1MuDTEUX
//---------------------------------

//----------------
// Constructors --
//----------------

L1MuDTEUX::L1MuDTEUX(const L1MuDTSectorProcessor& sp, const L1MuDTSEU& seu, int id) : 
    m_sp(sp), m_seu(seu), m_id(id), 
    m_result(false), m_quality(0), m_address(15),
    m_start(0), m_target(0) {

  setPrecision();

}


//--------------
// Destructor --
//--------------

L1MuDTEUX::~L1MuDTEUX() {}


//--------------
// Operations --
//--------------

//
// Equal operator
//
bool L1MuDTEUX::operator==(const L1MuDTEUX& eux) const {

  if ( m_id      != eux.id() )      return false;
  if ( m_result  != eux.result() )  return false;
  if ( m_quality != eux.quality() ) return false;
  if ( m_address != eux.address() ) return false;
  return true;

}


//
// run EUX
//
void L1MuDTEUX::run(const edm::EventSetup& c) {

  c.get< L1MuDTExtLutRcd >().get( theExtLUTs );
  c.get< L1MuDTTFParametersRcd >().get( pars );

  if ( L1MuDTTFConfig::Debug(4) ) cout << "Run EUX "  << m_id << endl;
  if ( L1MuDTTFConfig::Debug(4) ) cout << "start :  " << *m_start  << endl;
  if ( L1MuDTTFConfig::Debug(4) ) cout << "target : " << *m_target << endl;

  if ( m_start == 0 || m_target == 0 ) { 
    if ( L1MuDTTFConfig::Debug(4) ) cout << "Error: EUX has no data loaded" << endl;
    return;
  }

  // start sector
  int sector_st = m_start->sector();

  // target sector
  int sector_ta = m_target->sector();

  // get index of look-up table
  int lut_idx = m_seu.ext();
  if ( abs(m_target->wheel()) == 3 ) {

    switch ( m_seu.ext() ) {
      case EX13 : { lut_idx = EX15; break; }
      case EX23 : { lut_idx = EX25; break; }
      default   : { lut_idx = m_seu.ext(); break; }
    }

  }

  if ( L1MuDTTFConfig::Debug(5) ) cout << "EUX : using look-up table : "
                                       << static_cast<Extrapolation>(lut_idx)
                                       << endl;

  // Extrapolation TS quality filter
  int qcut = 0;
  if ( m_seu.ext() == EX12 ) qcut = pars->get_soc_qcut_st1(m_sp.id().wheel(), m_sp.id().sector());
  if ( m_seu.ext() == EX13 ) qcut = pars->get_soc_qcut_st1(m_sp.id().wheel(), m_sp.id().sector());
  if ( m_seu.ext() == EX14 ) qcut = pars->get_soc_qcut_st1(m_sp.id().wheel(), m_sp.id().sector());
  if ( m_seu.ext() == EX21 ) qcut = pars->get_soc_qcut_st2(m_sp.id().wheel(), m_sp.id().sector());
  if ( m_seu.ext() == EX23 ) qcut = pars->get_soc_qcut_st2(m_sp.id().wheel(), m_sp.id().sector());
  if ( m_seu.ext() == EX24 ) qcut = pars->get_soc_qcut_st2(m_sp.id().wheel(), m_sp.id().sector());
  if ( m_seu.ext() == EX34 ) qcut = pars->get_soc_qcut_st4(m_sp.id().wheel(), m_sp.id().sector());

  if ( m_start->quality() < qcut ) return;
  
  // calculate bit shift
  int sh_phi  = 12 - nbit_phi;
  int sh_phib = 10 - nbit_phib;

  int phi_target = m_target->phi() >> sh_phi;
  int phi_start  = m_start->phi()  >> sh_phi;
  int phib_start = (m_start->phib() >> sh_phib) << sh_phib;
  if ( phib_start < 0 ) phib_start += (1 << sh_phib) -1;

  // compute difference in phi
  int diff = phi_target - phi_start;

  // get low and high values from look-up table 
  // and add offset (30 degrees ) for extrapolation to adjacent sector 
  int offset = -2144 >> sh_phi;
  offset  *= sec_mod(sector_ta - sector_st);
  int low  = theExtLUTs->getLow(lut_idx,phib_start );
  int high = theExtLUTs->getHigh(lut_idx,phib_start );
  if ( low  < 0 ) low  += (1 << sh_phi) - 1;
  if ( high < 0 ) high += (1 << sh_phi) - 1;
  low  = (low  >> sh_phi) + offset;
  high = (high >> sh_phi) + offset;

  int phi_offset = phi_target - offset;
  if ( ( lut_idx == EX34 ) || ( lut_idx == EX21 ) )
    phi_offset = phi_start + offset;
  if ( phi_offset >= (1 << (nbit_phi-1)) -1 ) return;
  if ( phi_offset < -(1 << (nbit_phi-1)) +1 ) return;

  // is phi-difference within the extrapolation window?
  bool openlut = pars->get_soc_openlut_extr(m_sp.id().wheel(), m_sp.id().sector());
  if (( diff >= low && diff <= high ) || L1MuDTTFConfig::getopenLUTs() || openlut ) {
    m_result = true;
    int qual_st = m_start->quality();
    int qual_ta = m_target->quality();
    if ( m_seu.ext() == EX34 || m_seu.ext() == EX21 ) {
      m_quality = ( qual_st == 7 ) ? 0 : qual_st + 1;
    } 
    else {  
      m_quality = ( qual_ta == 7 ) ? 0 : qual_ta + 1;
    }
    m_address = m_id;
  }

  if ( L1MuDTTFConfig::Debug(5) ) cout << "diff : "   << low  << " "
                                       << diff << " " << high << " : "
                                       << m_result << " " << endl;

}


//
// load data into EUX
//
void L1MuDTEUX::load(const L1MuDTTrackSegPhi* start_ts, 
                     const L1MuDTTrackSegPhi* target_ts) {

  m_start  = start_ts;
  m_target = target_ts;

  // in case of EX34 and EX21 exchange start and target
  if ( ( m_seu.ext() == EX34 ) || ( m_seu.ext() == EX21 ) ) {
    m_start  = target_ts;
    m_target = start_ts;
  }

}


//
// reset this EUX
//

void L1MuDTEUX::reset() {
  
  m_result  = false;
  m_quality = 0;
  m_address = 15;

  m_start  = 0;
  m_target = 0;

}


//
// return pointer to start and target track segment
//
pair<const L1MuDTTrackSegPhi* ,const L1MuDTTrackSegPhi*> L1MuDTEUX::ts() const {

  return pair<const L1MuDTTrackSegPhi*, const L1MuDTTrackSegPhi*>(m_start,m_target);

}


//
// symmetric modulo function for sectors
// output values in the range -6 to +5
//
int L1MuDTEUX::sec_mod(int sector) const {

  int new_sector = sector%12;
  if ( new_sector >= 6 )
    new_sector = new_sector - 12;
  if ( new_sector < -6 )
    new_sector = new_sector + 12;
   
  return new_sector;

}


//
// set precision for phi and phib 
// default is 12 bits for phi and 10 bits for phib
//
void L1MuDTEUX::setPrecision() {

  nbit_phi  = L1MuDTTFConfig::getNbitsExtPhi();
  nbit_phib = L1MuDTTFConfig::getNbitsExtPhib();

  theExtFilter = L1MuDTTFConfig::getExtTSFilter();

}


// static data members

int L1MuDTEUX::theExtFilter = 1;
unsigned short int L1MuDTEUX::nbit_phi  = 12;
unsigned short int L1MuDTEUX::nbit_phib = 10;
