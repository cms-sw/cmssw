//-------------------------------------------------
//
//   Class: L1MuDTExtrapolationUnit
//
//   Description: Extrapolation Unit
//
//
//   $Date: 2008/11/28 10:30:51 $
//   $Revision: 1.5 $
//
//   Author :
//   N. Neumeister            CERN EP
//
//--------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------

#include "L1Trigger/DTTrackFinder/src/L1MuDTExtrapolationUnit.h"

//---------------
// C++ Headers --
//---------------

#include <iostream>
#include <bitset>
#include <cassert>

//-------------------------------
// Collaborating Class Headers --
//-------------------------------

#include "L1Trigger/DTTrackFinder/src/L1MuDTTFConfig.h"
#include "CondFormats/L1TObjects/interface/L1MuDTExtParam.h"
#include "L1Trigger/DTTrackFinder/src/L1MuDTSEU.h"
#include "L1Trigger/DTTrackFinder/src/L1MuDTEUX.h"
#include "L1Trigger/DTTrackFinder/src/L1MuDTERS.h"
#include "L1Trigger/DTTrackFinder/src/L1MuDTSectorProcessor.h"
#include "L1Trigger/DTTrackFinder/src/L1MuDTSecProcId.h"
#include "L1Trigger/DTTrackFinder/src/L1MuDTDataBuffer.h"
#include "L1Trigger/DTTrackFinder/src/L1MuDTTrackSegPhi.h"
#include "CondFormats/L1TObjects/interface/L1MuDTTFParameters.h"
#include "CondFormats/DataRecord/interface/L1MuDTTFParametersRcd.h"

using namespace std;

// --------------------------------
//       class L1MuDTExtrapolationUnit
//---------------------------------

//----------------
// Constructors --
//----------------

L1MuDTExtrapolationUnit::L1MuDTExtrapolationUnit(const L1MuDTSectorProcessor& sp) :
                    m_sp(sp), m_SEUs() {

  for ( int ext_idx = 0; ext_idx < MAX_EXT; ext_idx++ ) {
  
    Extrapolation ext = static_cast<Extrapolation>(ext_idx);   

    if ( ext == EX12 || ext == EX13 || ext == EX14 ||
         ext == EX21 || ext == EX23 || ext == EX24 || ext == EX34 ) {
         
      unsigned int maxaddr = 4;   
      
      if ( ext == EX12 || ext == EX13 || ext == EX14 ) maxaddr = 2;

      for ( unsigned int startAddress = 0; startAddress < maxaddr; startAddress++ ) {
        SEUId id = make_pair(ext, startAddress);
        m_SEUs[id] = new L1MuDTSEU(sp,ext,startAddress);
        if ( L1MuDTTFConfig::Debug(6) ) cout << "  creating SEU " << ext << " " << startAddress << endl;
      }  
    }
  }

}


//--------------
// Destructor --
//--------------

L1MuDTExtrapolationUnit::~L1MuDTExtrapolationUnit() {

  for ( SEUmap::iterator iter = m_SEUs.begin(); iter != m_SEUs.end(); iter++ ) {
    delete (*iter).second;
    (*iter).second = 0;
  }
  m_SEUs.clear();

}


//--------------
// Operations --
//--------------

//
// run Extrapolation Unit
//
void L1MuDTExtrapolationUnit::run(const edm::EventSetup& c) {

  c.get< L1MuDTTFParametersRcd >().get( pars );

  SEUmap::const_iterator iter;
  for ( iter = m_SEUs.begin(); iter != m_SEUs.end(); iter++ ) {

    pair<int,int> ext_pair = which_ext(((*iter).second)->ext());
    int start = ext_pair.first;

    const L1MuDTTrackSegPhi* ts = 0;

    //get start track segment
    ts = m_sp.data()->getTSphi(start, ((*iter).second)->tsId() );

    if ( ts != 0 && !ts->empty() ) {
      ((*iter).second)->load(ts);
      ((*iter).second)->run(c);
    }

  }

  //
  // use EX21 to cross-check EX12
  //
  bool run_21 = pars->get_soc_run_21(m_sp.id().wheel(), m_sp.id().sector());
  if ( L1MuDTTFConfig::getUseEX21() || run_21 ) {

    // search for EX12 + EX21 single extrapolation units
    for ( unsigned int startAdr = 0; startAdr < 2; startAdr++ ) {
    
      bitset<12> extab12 = this->getEXTable( EX12, startAdr );
      bitset<12> extab21 = this->getEXTable( EX21, startAdr );

      for ( int eux = 0; eux < 12; eux++ ) {
        if ( extab12.test(eux) && !extab21.test(eux) ) {
          reset(EX12,startAdr,eux); 
          if ( L1MuDTTFConfig::Debug(6) ) {
            SEUId seuid = make_pair(EX12, startAdr);
            L1MuDTSEU* SEU12 = m_SEUs[seuid];
            cout << "  EX12 - EX21 mismatch : "
                 << " EX12 : " << extab12 << " , " 
                 << " EX21 : " << extab21 << endl
                 << "  Cancel: " << SEU12->ext()
                 << " start addr = " << SEU12->tsId()
                 << " target addr = " << eux << endl;
          }  
        }
      }

    }
  }

}


//
// reset Extrapolation Unit
//
void L1MuDTExtrapolationUnit::reset() {

  SEUmap::const_iterator iter; 
  for ( iter = m_SEUs.begin(); iter != m_SEUs.end(); iter++ ) {
    ((*iter).second)->reset();
  }

}


//
// reset a single extrapolation
//
void L1MuDTExtrapolationUnit::reset(Extrapolation ext, unsigned int startAdr, unsigned int relAdr) {

  //  assert( startAdr >= 0 && startAdr <= 3 );
  //  assert( relAdr >= 0 && relAdr <= 12 );

  SEUId seuid = make_pair(ext, startAdr);
  SEUmap::const_iterator iter = m_SEUs.find(seuid);
  if ( iter != m_SEUs.end() ) ((*iter).second)->reset(relAdr);

}


//
// get extrapolation address from ERS
//
unsigned short int L1MuDTExtrapolationUnit::getAddress(Extrapolation ext, unsigned int startAdr, int id) const {

  // get extrapolation address from ERS 
  // startAdr = 0, 1  : own wheel
  // startAdr = 2, 3  : next wheel neighbour

  //  assert( startAdr >= 0 && startAdr <= 3 );
  //  assert( id == 0 || id == 1 );
  
  unsigned short int address = 15;
  
  SEUId seuid = make_pair(ext, startAdr);
  SEUmap::const_iterator iter = m_SEUs.find(seuid);
  if ( iter != m_SEUs.end() ) address = ((*iter).second)->ers()->address(id);

  return address;

}


//
// get extrapolation quality from ERS
//
unsigned short int L1MuDTExtrapolationUnit::getQuality(Extrapolation ext, unsigned int startAdr, int id) const {

  // get extrapolation quality from ERS 
  // startAdr = 0, 1  : own wheel
  // startAdr = 2, 3  : next wheel neighbour

  //  assert( startAdr >= 0 && startAdr <= 3 );
  //  assert( id == 0 || id == 1 );
  
  unsigned short int quality = 0;

  SEUId seuid = make_pair(ext, startAdr);
  SEUmap::const_iterator iter = m_SEUs.find(seuid);
  if ( iter != m_SEUs.end() ) quality = ((*iter).second)->ers()->quality(id);

  return quality;
  
}


//
// get Extrapolator table for a given SEU
//
const bitset<12>& L1MuDTExtrapolationUnit::getEXTable(Extrapolation ext, unsigned int startAdr) const {

  // startAdr = 0, 1  : own wheel
  // startAdr = 2, 3  : next wheel neighbour

  //  assert( startAdr >= 0 && startAdr <= 3 );
  
  SEUId seuid = make_pair(ext, startAdr);
  return m_SEUs[seuid]->exTable();

}


//
// get Quality Sorter table for a given SEU
//
const bitset<12>& L1MuDTExtrapolationUnit::getQSTable(Extrapolation ext, unsigned int startAdr) const {

  // startAdr = 0, 1  : own wheel
  // startAdr = 2, 3  : next wheel neighbour

  //  assert( startAdr >= 0 && startAdr <= 3 );
  
  SEUId seuid = make_pair(ext, startAdr);
  return m_SEUs[seuid]->qsTable();

}


//
// get number of successful extrapolations
//
int L1MuDTExtrapolationUnit::numberOfExt() const {

  int number = 0;
  SEUmap::const_iterator iter;
  for ( iter = m_SEUs.begin(); iter != m_SEUs.end(); iter++ ) {
    number += ((*iter).second)->numberOfExt();
  }
 
  return number;

}


//
// print all successful extrapolations
//
void L1MuDTExtrapolationUnit::print(int level) const {

  SEUmap::const_iterator iter_seu;

  if ( level == 0 ) {
    for ( iter_seu = m_SEUs.begin(); iter_seu != m_SEUs.end(); iter_seu++ ) {
      vector<L1MuDTEUX*> vec_eux = ((*iter_seu).second)->eux();
      vector<L1MuDTEUX*>::const_iterator iter_eux;
      for ( iter_eux  = vec_eux.begin(); 
            iter_eux != vec_eux.end(); iter_eux++ ) {
        if ( (*iter_eux)->result() ) {
          cout << ((*iter_seu).second)->ext() << " "
               << ((*iter_seu).second)->tsId() << " " 
               << (*iter_eux)->id() << endl; 
          cout << "start  : " << *(*iter_eux)->ts().first << endl;
          cout << "target : " << *(*iter_eux)->ts().second << endl;
          cout << "result : " << "quality = " << (*iter_eux)->quality() << '\t' 
                              << "address = " << (*iter_eux)->address() << endl;
        }
      }
    }
  }

  //
  // print all results from Extrapolator and Quality Sorter
  //
  if ( level == 1 ) {
    cout << "Results from Extrapolator and Quality Sorter of " << m_sp.id()
         << " : \n" << endl;

    cout << "             EXT            QSU      " << endl;
    cout << "  S E U   11            11           " << endl;
    cout << "          109876543210  109876543210 " << endl;
    cout << "-------------------------------------" << endl;     
    for ( iter_seu = m_SEUs.begin(); iter_seu != m_SEUs.end(); iter_seu++ ) {

      cout << ((*iter_seu).second)->ext() << "_ "
           << ((*iter_seu).second)->tsId() << ": "
           << ((*iter_seu).second)->exTable() << "  "
           << ((*iter_seu).second)->qsTable() << endl; 

    }
  
    cout << endl;
  }

}


// static 

//
// get station of start and target track segment for a given extrapolation
//
pair<int,int> L1MuDTExtrapolationUnit::which_ext(Extrapolation ext) {

  int source = 0;
  int target = 0;

  //  assert( ext >= 0 && ext < MAX_EXT );

  switch ( ext ) {
    case EX12 : { source = 1; target = 2; break; }
    case EX13 : { source = 1; target = 3; break; }
    case EX14 : { source = 1; target = 4; break; }
    case EX21 : { source = 1; target = 2; break; }
    case EX23 : { source = 2; target = 3; break; }
    case EX24 : { source = 2; target = 4; break; }
    case EX34 : { source = 3; target = 4; break; }
    case EX15 : { source = 1; target = 3; break; }
    case EX25 : { source = 2; target = 3; break; }
    default : { source = 1; target = 2; break; }
  }

  return pair<int,int>(source,target);

}
