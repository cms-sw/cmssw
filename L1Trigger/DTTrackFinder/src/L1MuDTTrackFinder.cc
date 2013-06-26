//-------------------------------------------------
//
//   Class: L1MuDTTrackFinder
//
//   Description: L1 barrel Muon Trigger Track Finder
//
//
//   $Date: 2008/05/09 15:02:00 $
//   $Revision: 1.10 $
//
//   Author :
//   N. Neumeister            CERN EP
//   J. Troconiz              UAM Madrid
//
//--------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------

#include "L1Trigger/DTTrackFinder/interface/L1MuDTTrackFinder.h"

//---------------
// C++ Headers --
//---------------

#include <iostream>

//-------------------------------
// Collaborating Class Headers --
//-------------------------------

#include <DataFormats/Common/interface/Handle.h>
#include <FWCore/Framework/interface/Event.h>
#include "DataFormats/L1DTTrackFinder/interface/L1MuDTChambPhContainer.h"
#include "DataFormats/L1DTTrackFinder/interface/L1MuDTTrackCand.h"
#include "L1Trigger/DTTrackFinder/src/L1MuDTTFConfig.h"
#include "L1Trigger/DTTrackFinder/src/L1MuDTSecProcId.h"
#include "L1Trigger/DTTrackFinder/src/L1MuDTSecProcMap.h"
#include "L1Trigger/DTTrackFinder/src/L1MuDTSectorProcessor.h"
#include "L1Trigger/DTTrackFinder/src/L1MuDTEtaProcessor.h"
#include "L1Trigger/DTTrackFinder/src/L1MuDTWedgeSorter.h"
#include "L1Trigger/DTTrackFinder/src/L1MuDTMuonSorter.h"
#include "L1Trigger/DTTrackFinder/interface/L1MuDTTrack.h"

using namespace std;

//---------------------------------
//       class L1MuDTTrackFinder
//---------------------------------


//----------------
// Constructors --
//----------------

L1MuDTTrackFinder::L1MuDTTrackFinder(const edm::ParameterSet & ps) {

  // set configuration parameters
  if ( m_config == 0 ) m_config = new L1MuDTTFConfig(ps);

  if ( L1MuDTTFConfig::Debug(1) ) cout << endl;
  if ( L1MuDTTFConfig::Debug(1) ) cout << "**** entering L1MuDTTrackFinder ****" << endl;
  if ( L1MuDTTFConfig::Debug(1) ) cout << endl;

  m_spmap = new L1MuDTSecProcMap();
  m_epvec.reserve(12);
  m_wsvec.reserve(12);
  m_ms = 0;

  _cache.reserve(4*17);
  _cache0.reserve(144*17);

}


//--------------
// Destructor --
//--------------

L1MuDTTrackFinder::~L1MuDTTrackFinder() {

  delete m_spmap;

  vector<L1MuDTEtaProcessor*>::iterator it_ep = m_epvec.begin();
  while ( it_ep != m_epvec.end() ) {
    delete (*it_ep);
    it_ep++;
  }
  m_epvec.clear();

  vector<L1MuDTWedgeSorter*>::iterator it_ws = m_wsvec.begin();
  while ( it_ws != m_wsvec.end() ) {
    delete (*it_ws);
    it_ws++;
  }
  m_wsvec.clear();

  delete m_ms;

  if ( m_config ) delete m_config;
  m_config = 0;

}


//--------------
// Operations --
//--------------

//
// setup MTTF configuration
//
void L1MuDTTrackFinder::setup() {

  // build the barrel Muon Trigger Track Finder

  if ( L1MuDTTFConfig::Debug(1) ) cout << endl;
  if ( L1MuDTTFConfig::Debug(1) ) cout << "**** L1MuDTTrackFinder building ****" << endl;
  if ( L1MuDTTFConfig::Debug(1) ) cout << endl;

  // create new sector processors
  for ( int wh = -3; wh <= 3; wh++ ) {
    if ( wh == 0 ) continue;
    for ( int sc = 0; sc < 12; sc++ ) {
      L1MuDTSecProcId tmpspid(wh,sc);
      L1MuDTSectorProcessor* sp = new L1MuDTSectorProcessor(*this,tmpspid);
      if ( L1MuDTTFConfig::Debug(2) ) cout << "creating " << tmpspid << endl;
      m_spmap->insert(tmpspid,sp);
    }
  }
 
  // create new eta processors and wedge sorters
  for ( int sc = 0; sc < 12; sc++ ) {
    L1MuDTEtaProcessor* ep = new L1MuDTEtaProcessor(*this,sc);
    if ( L1MuDTTFConfig::Debug(2) ) cout << "creating Eta Processor " << sc << endl;
    m_epvec.push_back(ep);
    L1MuDTWedgeSorter* ws = new L1MuDTWedgeSorter(*this,sc);
    if ( L1MuDTTFConfig::Debug(2) ) cout << "creating Wedge Sorter " << sc << endl;
    m_wsvec.push_back(ws);
  }

  // create new muon sorter
  if ( L1MuDTTFConfig::Debug(2) ) cout << "creating DT Muon Sorter " << endl;
  m_ms = new L1MuDTMuonSorter(*this);

}


//
// run MTTF
//
void L1MuDTTrackFinder::run(const edm::Event& e, const edm::EventSetup& c) {

  // run the barrel Muon Trigger Track Finder

  edm::Handle<L1MuDTChambPhContainer> dttrig;
  e.getByLabel(L1MuDTTFConfig::getDTDigiInputTag(),dttrig);
  if ( dttrig->getContainer()->size() == 0 ) return;

  if ( L1MuDTTFConfig::Debug(2) ) cout << endl;
  if ( L1MuDTTFConfig::Debug(2) ) cout << "**** L1MuDTTrackFinder processing ****" << endl;
  if ( L1MuDTTFConfig::Debug(2) ) cout << endl;

  int bx_min = L1MuDTTFConfig::getBxMin();
  int bx_max = L1MuDTTFConfig::getBxMax();

  for ( int bx = bx_min; bx <= bx_max; bx++ ) {

  if ( dttrig->bxEmpty(bx) ) continue;

  if ( L1MuDTTFConfig::Debug(2) ) cout << "L1MuDTTrackFinder processing bunch-crossing : " << bx << endl;

    // reset MTTF
    reset();

    // run sector processors
    L1MuDTSecProcMap::SPmap_iter it_sp = m_spmap->begin();
    while ( it_sp != m_spmap->end() ) {
      if ( L1MuDTTFConfig::Debug(2) ) cout << "running " 
                                           << (*it_sp).second->id() << endl;
      if ( (*it_sp).second ) (*it_sp).second->run(bx,e,c);
      if ( L1MuDTTFConfig::Debug(2) && (*it_sp).second ) (*it_sp).second->print();
      it_sp++;
    } 

    // run eta processors
    vector<L1MuDTEtaProcessor*>::iterator it_ep = m_epvec.begin();
    while ( it_ep != m_epvec.end() ) {
      if ( L1MuDTTFConfig::Debug(2) ) cout << "running Eta Processor "
                                       << (*it_ep)->id() << endl;
      if ( *it_ep ) (*it_ep)->run(bx,e,c);
      if ( L1MuDTTFConfig::Debug(2) && *it_ep ) (*it_ep)->print();
      it_ep++;
    }

    // read sector processors
    it_sp = m_spmap->begin();
    while ( it_sp != m_spmap->end() ) {
      if ( L1MuDTTFConfig::Debug(2) ) cout << "reading " 
                                           << (*it_sp).second->id() << endl;
      for ( int number = 0; number < 2; number++ ) {
        const L1MuDTTrack* cand = (*it_sp).second->tracK(number);
        if ( cand && !cand->empty() ) _cache0.push_back(L1MuDTTrackCand(cand->getDataWord(),cand->bx(),
                                              cand->spid().wheel(),cand->spid().sector(),number,cand->address(1),
                                              cand->address(2),cand->address(3),cand->address(4),cand->tc()));
      }
      it_sp++;
    } 

    // run wedge sorters
    vector<L1MuDTWedgeSorter*>::iterator it_ws = m_wsvec.begin();
    while ( it_ws != m_wsvec.end() ) {
      if ( L1MuDTTFConfig::Debug(2) ) cout << "running Wedge Sorter " 
                                           << (*it_ws)->id() << endl;
      if ( *it_ws ) (*it_ws)->run();
      if ( L1MuDTTFConfig::Debug(2) && *it_ws ) (*it_ws)->print();
      it_ws++;
    }

    // run muon sorter
    if ( L1MuDTTFConfig::Debug(2) ) cout << "running DT Muon Sorter" << endl;
    if ( m_ms ) m_ms->run();
    if ( L1MuDTTFConfig::Debug(2) && m_ms ) m_ms->print();

    // store found track candidates in container (cache)
    if ( m_ms->numberOfTracks() > 0 ) {
      const vector<const L1MuDTTrack*>&  mttf_cont = m_ms->tracks();
      vector<const L1MuDTTrack*>::const_iterator iter;
      for ( iter = mttf_cont.begin(); iter != mttf_cont.end(); iter++ ) {
        if ( *iter ) _cache.push_back(L1MuRegionalCand((*iter)->getDataWord(),(*iter)->bx()));
      }
    }
    
  }

}


//
// reset MTTF
//
void L1MuDTTrackFinder::reset() {

  L1MuDTSecProcMap::SPmap_iter it_sp = m_spmap->begin();
  while ( it_sp != m_spmap->end() ) {
    if ( (*it_sp).second ) (*it_sp).second->reset();
    it_sp++;
  }

  vector<L1MuDTEtaProcessor*>::iterator it_ep = m_epvec.begin();
  while ( it_ep != m_epvec.end() ) {
    if ( *it_ep ) (*it_ep)->reset();
    it_ep++;
  }
    
  vector<L1MuDTWedgeSorter*>::iterator it_ws = m_wsvec.begin();
  while ( it_ws != m_wsvec.end() ) {
    if ( *it_ws ) (*it_ws)->reset();
    it_ws++;
  }

  if ( m_ms ) m_ms->reset();

}


//
// return Sector Processor container 
//
const L1MuDTSectorProcessor* L1MuDTTrackFinder::sp(const L1MuDTSecProcId& id) const { 

  return m_spmap->sp(id); 

}


//
// return number of muon candidates found by the barrel MTTF
//
int L1MuDTTrackFinder::numberOfTracks() {

  return _cache.size();

}


L1MuDTTrackFinder::TFtracks_const_iter L1MuDTTrackFinder::begin() {

  return _cache.begin();

}


L1MuDTTrackFinder::TFtracks_const_iter L1MuDTTrackFinder::end() {

  return _cache.end();

}


void L1MuDTTrackFinder::clear() {

  _cache.clear();
  _cache0.clear();

}


//
// return number of muon candidates found by the barrel MTTF at a given bx
//
int L1MuDTTrackFinder::numberOfTracks(int bx) {

  int number = 0;
  for ( TFtracks_const_iter it  = _cache.begin(); it != _cache.end(); it++ ) {
    if ( (*it).bx() == bx ) number++;
  }
  
  return number;

}


// static data members

L1MuDTTFConfig* L1MuDTTrackFinder::m_config = 0;
