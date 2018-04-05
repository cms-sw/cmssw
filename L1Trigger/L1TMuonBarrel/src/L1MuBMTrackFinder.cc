//-------------------------------------------------
//
//   Class: L1MuBMTrackFinder
//
//   Description: L1 barrel Muon Trigger Track Finder
//
//
//
//   Author :
//   N. Neumeister            CERN EP
//   J. Troconiz              UAM Madrid
//   Modifications:
//   G. Flouris	              U.Ioannina
//   G. Karathanasis          U. Athens
//--------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------

#include "L1Trigger/L1TMuonBarrel/interface/L1MuBMTrackFinder.h"

//---------------
// C++ Headers --
//---------------

#include <iostream>
#include <string>

//-------------------------------
// Collaborating Class Headers --
//-------------------------------

#include <DataFormats/Common/interface/Handle.h>
#include <FWCore/Framework/interface/Event.h>

#include "L1Trigger/L1TMuonBarrel/src/L1MuBMTFConfig.h"
#include "L1Trigger/L1TMuonBarrel/src/L1MuBMSecProcMap.h"
#include "L1Trigger/L1TMuonBarrel/src/L1MuBMSectorProcessor.h"
#include "L1Trigger/L1TMuonBarrel/src/L1MuBMEtaProcessor.h"
#include "L1Trigger/L1TMuonBarrel/src/L1MuBMWedgeSorter.h"
#include "L1Trigger/L1TMuonBarrel/src/L1MuBMMuonSorter.h"

#include "DataFormats/L1TMuon/interface/BMTF/L1MuBMSecProcId.h"
#include "DataFormats/L1TMuon/interface/RegionalMuonCand.h"
#include "DataFormats/L1TMuon/interface/L1MuBMTrack.h"
#include "DataFormats/L1TMuon/interface/L1MuBMTrackSegPhi.h"
#include "DataFormats/L1TMuon/interface/L1MuBMTrackSegEta.h"

using namespace std;

//---------------------------------
//       class L1MuBMTrackFinder
//---------------------------------


//----------------
// Constructors --
//----------------
//:
L1MuBMTrackFinder::L1MuBMTrackFinder(const edm::ParameterSet & ps,edm::ConsumesCollector && iC):
_cache0(144,-9,8),_cache(36, -9, 8) {
  // set configuration parameters
  if ( m_config == nullptr ) m_config = new L1MuBMTFConfig(ps);

  if ( L1MuBMTFConfig::Debug(1) ) cout << endl;
  if ( L1MuBMTFConfig::Debug(1) ) cout << "**** entering L1MuBMTrackFinder ****" << endl;
  if ( L1MuBMTFConfig::Debug(1) ) cout << endl;

  m_spmap = new L1MuBMSecProcMap();
  m_epvec.reserve(12);
  m_wsvec.reserve(12);
  m_ms = nullptr;

  m_DTDigiToken = iC.consumes<L1MuDTChambPhContainer>(L1MuBMTFConfig::getBMDigiInputTag());
}


//--------------
// Destructor --
//--------------

L1MuBMTrackFinder::~L1MuBMTrackFinder() {

  delete m_spmap;

  vector<L1MuBMEtaProcessor*>::iterator it_ep = m_epvec.begin();
  while ( it_ep != m_epvec.end() ) {
    delete (*it_ep);
    it_ep++;
  }
  m_epvec.clear();

  vector<L1MuBMWedgeSorter*>::iterator it_ws = m_wsvec.begin();
  while ( it_ws != m_wsvec.end() ) {
    delete (*it_ws);
    it_ws++;
  }
  m_wsvec.clear();

  delete m_ms;

  if ( m_config ) delete m_config;
  m_config = nullptr;

}


//--------------
// Operations --
//--------------

//
// setup MTTF configuration
//
void L1MuBMTrackFinder::setup(edm::ConsumesCollector&& iC) {

  // build the barrel Muon Trigger Track Finder

  if ( L1MuBMTFConfig::Debug(1) ) cout << endl;
  if ( L1MuBMTFConfig::Debug(1) ) cout << "**** L1MuBMTrackFinder building ****" << endl;
  if ( L1MuBMTFConfig::Debug(1) ) cout << endl;

  // create new sector processors
  for ( int wh = -3; wh <= 3; wh++ ) {
    if ( wh == 0 ) continue;
    for ( int sc = 0; sc < 12; sc++ ) {
      L1MuBMSecProcId tmpspid(wh,sc);
      L1MuBMSectorProcessor* sp = new L1MuBMSectorProcessor(*this,tmpspid,std::move(iC));
      if ( L1MuBMTFConfig::Debug(2) ) cout << "creating " << tmpspid << endl;
      m_spmap->insert(tmpspid,sp);
    }
  }

  // create new eta processors and wedge sorters
  for ( int sc = 0; sc < 12; sc++ ) {
    L1MuBMEtaProcessor* ep = new L1MuBMEtaProcessor(*this,sc, std::move(iC));
    if ( L1MuBMTFConfig::Debug(2) ) cout << "creating Eta Processor " << sc << endl;
    m_epvec.push_back(ep);
    L1MuBMWedgeSorter* ws = new L1MuBMWedgeSorter(*this,sc);
    if ( L1MuBMTFConfig::Debug(2) ) cout << "creating Wedge Sorter " << sc << endl;
    m_wsvec.push_back(ws);
  }

  // create new muon sorter
  if ( L1MuBMTFConfig::Debug(2) ) cout << "creating BM Muon Sorter " << endl;
  m_ms = new L1MuBMMuonSorter(*this);

}


//
// run MTTF
//
void L1MuBMTrackFinder::run(const edm::Event& e, const edm::EventSetup& c) {

 m_config->setDefaultsES(c);
 int bx_min = L1MuBMTFConfig::getBxMin();
 int bx_max = L1MuBMTFConfig::getBxMax();

 //Resize the bx range according to the config file
  _cache0.setBXRange(bx_min,bx_max);
  _cache.setBXRange(bx_min,bx_max);



// run the barrel Muon Trigger Track Finder
  edm::Handle<L1MuDTChambPhContainer> dttrig;
  e.getByToken(m_DTDigiToken,dttrig);
  if ( dttrig->getContainer()->empty() ) return;

  if ( L1MuBMTFConfig::Debug(2) ) cout << endl;
  if ( L1MuBMTFConfig::Debug(2) ) cout << "**** L1MuBMTrackFinder processing ------****" << endl;
  if ( L1MuBMTFConfig::Debug(2) ) cout << endl;

  for ( int bx = bx_min; bx <= bx_max; bx++ ) {
  if ( dttrig->bxEmpty(bx) ) continue;

  if ( L1MuBMTFConfig::Debug(2) ) cout << "L1MuBMTrackFinder processing bunch-crossing : " << bx << endl;

    // reset MTTF
    reset();

    // run sector processors
    L1MuBMSecProcMap::SPmap_iter it_sp = m_spmap->begin();
    while ( it_sp != m_spmap->end() ) {
      if ( L1MuBMTFConfig::Debug(2) ) cout << "running "
                                           << (*it_sp).second->id() << endl;
      if ( (*it_sp).second ) (*it_sp).second->run(bx,e,c);
      if ( L1MuBMTFConfig::Debug(2) && (*it_sp).second ) (*it_sp).second->print();
      it_sp++;
    }

    // run eta processors
    vector<L1MuBMEtaProcessor*>::iterator it_ep = m_epvec.begin();
    while ( it_ep != m_epvec.end() ) {
      if ( L1MuBMTFConfig::Debug(2) ) cout << "running Eta Processor "
                                       << (*it_ep)->id() << endl;
      if ( *it_ep ) (*it_ep)->run(bx,e,c);
      if ( L1MuBMTFConfig::Debug(2) && *it_ep ) (*it_ep)->print();
      it_ep++;
    }

    // read sector processors
    it_sp = m_spmap->begin();
    while ( it_sp != m_spmap->end() ) {
      if ( L1MuBMTFConfig::Debug(2) ) cout << "reading "
                                           << (*it_sp).second->id() << endl;
      for ( int number = 0; number < 2; number++ ) {
        const L1MuBMTrack* cand = (*it_sp).second->tracK(number);

        if ( cand && !cand->empty() ) {

            l1t::RegionalMuonCand rmc;

            // max value in LUTs is 117
            if(cand->hwEta()>-117 || cand->hwEta()<117 )
                rmc.setHwEta(cand->hwEta());
            else
                rmc.setHwEta(-1000);

            rmc.setHwPt(cand->pt());
            int abs_add_1 = setAdd(1,cand->address(1));
            int abs_add_2 = setAdd(2,cand->address(2));
            int abs_add_3 = setAdd(3,cand->address(3));
            int abs_add_4 = setAdd(4,cand->address(4));

            rmc.setTrackSubAddress(l1t::RegionalMuonCand::kWheelSide, cand->spid().wheel() < 0); // this has to be set!
            rmc.setTrackSubAddress(l1t::RegionalMuonCand::kWheelNum, abs(cand->spid().wheel()) - 1); // this has to be set!
            rmc.setTrackSubAddress(l1t::RegionalMuonCand::kStat1, abs_add_1);
            rmc.setTrackSubAddress(l1t::RegionalMuonCand::kStat2, abs_add_2);
            rmc.setTrackSubAddress(l1t::RegionalMuonCand::kStat3, abs_add_3);
            rmc.setTrackSubAddress(l1t::RegionalMuonCand::kStat4, abs_add_4);
            rmc.setTrackSubAddress(l1t::RegionalMuonCand::kSegSelStat1, 0);
            rmc.setTrackSubAddress(l1t::RegionalMuonCand::kSegSelStat2, 0);
            rmc.setTrackSubAddress(l1t::RegionalMuonCand::kSegSelStat3, 0);
            rmc.setTrackSubAddress(l1t::RegionalMuonCand::kSegSelStat4, 0);
            rmc.setHwHF(cand->hwHF());

            rmc.setHwPhi(cand->hwPhi());
            rmc.setHwSign(cand->hwSign() == 1 ? 0 : 1 );
            rmc.setHwSignValid(cand->hwSignValid());
            rmc.setHwQual(cand->hwQual());
            rmc.setTFIdentifiers(cand->spid().sector(),l1t::tftype::bmtf);

            _cache0.push_back(cand->bx(), rmc);
            _cache2.insert(std::end(_cache2), std::begin(cand->getTSphi()), std::end(cand->getTSphi()));
            _cache3.insert(std::end(_cache3), std::begin(cand->getTSeta()), std::end(cand->getTSeta()));
        }
      }
      it_sp++;
    }

    // run wedge sorters
    vector<L1MuBMWedgeSorter*>::iterator it_ws = m_wsvec.begin();
    while ( it_ws != m_wsvec.end() ) {
      if ( L1MuBMTFConfig::Debug(2) ) cout << "running Wedge Sorter "
                                           << (*it_ws)->id() << endl;
      if ( *it_ws ) (*it_ws)->run();
      if ( L1MuBMTFConfig::Debug(2) && *it_ws ) (*it_ws)->print();

    // store found track candidates in container (cache)
    if ( (*it_ws)->anyMuonCands()  ) {
      const vector<const L1MuBMTrack*>&  mttf_cont = (*it_ws)->tracks();

      vector<const L1MuBMTrack*>::const_iterator iter;
      for ( iter = mttf_cont.begin(); iter != mttf_cont.end(); iter++ ) {
        if(!*iter) continue;
        l1t::RegionalMuonCand rmc;
        rmc.setHwPt((*iter)->hwPt());
        int abs_add_1 = setAdd(1,(*iter)->address(1));
        int abs_add_2 = setAdd(2,(*iter)->address(2));
        int abs_add_3 = setAdd(3,(*iter)->address(3));
        int abs_add_4 = setAdd(4,(*iter)->address(4));

        rmc.setTrackSubAddress(l1t::RegionalMuonCand::kWheelSide, (*iter)->spid().wheel() < 0); // this has to be set!
        rmc.setTrackSubAddress(l1t::RegionalMuonCand::kWheelNum, abs((*iter)->spid().wheel()) - 1); // this has to be set!
        rmc.setTrackSubAddress(l1t::RegionalMuonCand::kStat1, abs_add_1);
        rmc.setTrackSubAddress(l1t::RegionalMuonCand::kStat2, abs_add_2);
        rmc.setTrackSubAddress(l1t::RegionalMuonCand::kStat3, abs_add_3);
        rmc.setTrackSubAddress(l1t::RegionalMuonCand::kStat4, abs_add_4);
        rmc.setTrackSubAddress(l1t::RegionalMuonCand::kSegSelStat1, 0);
        rmc.setTrackSubAddress(l1t::RegionalMuonCand::kSegSelStat2, 0);
        rmc.setTrackSubAddress(l1t::RegionalMuonCand::kSegSelStat3, 0);
        rmc.setTrackSubAddress(l1t::RegionalMuonCand::kSegSelStat4, 0);
        rmc.setHwHF((*iter)->hwHF());

        rmc.setHwPhi((*iter)->hwPhi());
        if((*iter)->hwEta()>-117 || (*iter)->hwEta()<117 )
	  //  rmc.setHwEta(eta_map[(*iter)->hwEta()]);
           rmc.setHwEta((*iter)->hwEta());
        else
            rmc.setHwEta(-1000);
        rmc.setHwSign((*iter)->hwSign() == 1 ? 0 : 1);
        rmc.setHwSignValid((*iter)->hwSignValid());
        rmc.setHwQual((*iter)->hwQual());
        rmc.setTFIdentifiers((*iter)->spid().sector(),l1t::tftype::bmtf);

        if ( *iter ){
	  _cache.push_back((*iter)->bx(), rmc);
	  _cache1.push_back(**iter);
	}
     }
    }

          it_ws++;

    }//end wedge sorting

    /*    // run muon sorter
    if ( L1MuBMTFConfig::Debug(2) ) cout << "running BM Muon Sorter" << endl;
    if ( m_ms ) m_ms->run();
    if ( L1MuBMTFConfig::Debug(2) && m_ms ) m_ms->print();


    // store found track candidates in container (cache)
    if ( m_ms->numberOfTracks() > 0 ) {
      const vector<const L1MuBMTrack*>&  mttf_cont = m_ms->tracks();
      vector<const L1MuBMTrack*>::const_iterator iter;
      for ( iter = mttf_cont.begin(); iter != mttf_cont.end(); iter++ ) {

        l1t::RegionalMuonCand rmc;
        rmc.setHwPt((*iter)->hwPt());
        int abs_add_1 = setAdd(1,(*iter)->address(1));
        int abs_add_2 = setAdd(2,(*iter)->address(2));
        int abs_add_3 = setAdd(3,(*iter)->address(3));
        int abs_add_4 = setAdd(4,(*iter)->address(4));

        rmc.setTrackSubAddress(l1t::RegionalMuonCand::kWheelSide, (*iter)->spid().wheel() < 0); // this has to be set!
        rmc.setTrackSubAddress(l1t::RegionalMuonCand::kWheelNum, abs((*iter)->spid().wheel()) - 1); // this has to be set!
        rmc.setTrackSubAddress(l1t::RegionalMuonCand::kStat1, abs_add_1);
        rmc.setTrackSubAddress(l1t::RegionalMuonCand::kStat2, abs_add_2);
        rmc.setTrackSubAddress(l1t::RegionalMuonCand::kStat3, abs_add_3);
        rmc.setTrackSubAddress(l1t::RegionalMuonCand::kStat4, abs_add_4);
        rmc.setTrackSubAddress(l1t::RegionalMuonCand::kSegSelStat1, 0);
        rmc.setTrackSubAddress(l1t::RegionalMuonCand::kSegSelStat2, 0);
        rmc.setTrackSubAddress(l1t::RegionalMuonCand::kSegSelStat3, 0);
        rmc.setTrackSubAddress(l1t::RegionalMuonCand::kSegSelStat4, 0);


        rmc.setHwPhi((*iter)->hwPhi());
        if((*iter)->hwEta()>-33 || (*iter)->hwEta()<32 )
                rmc.setHwEta(eta_map[(*iter)->hwEta()]);
        else
            rmc.setHwEta(-1000);
        rmc.setHwSign((*iter)->hwSign() == 1 ? 0 : 1);
        rmc.setHwSignValid((*iter)->hwSignValid());
        rmc.setHwQual((*iter)->hwQual());
        rmc.setTFIdentifiers((*iter)->spid().sector(),l1t::tftype::bmtf);

        if ( *iter ){ _cache.push_back((*iter)->bx(), rmc);}
     }
    }
    */

  }//end of bx loop
}


//
// reset MTTF
//
void L1MuBMTrackFinder::reset() {

  L1MuBMSecProcMap::SPmap_iter it_sp = m_spmap->begin();
  while ( it_sp != m_spmap->end() ) {
    if ( (*it_sp).second ) (*it_sp).second->reset();
    it_sp++;
  }

  vector<L1MuBMEtaProcessor*>::iterator it_ep = m_epvec.begin();
  while ( it_ep != m_epvec.end() ) {
    if ( *it_ep ) (*it_ep)->reset();
    it_ep++;
  }

  vector<L1MuBMWedgeSorter*>::iterator it_ws = m_wsvec.begin();
  while ( it_ws != m_wsvec.end() ) {
    if ( *it_ws ) (*it_ws)->reset();
    it_ws++;
  }

  if ( m_ms ) m_ms->reset();

}


//
// return Sector Processor container
//
const L1MuBMSectorProcessor* L1MuBMTrackFinder::sp(const L1MuBMSecProcId& id) const {

  return m_spmap->sp(id);

}


//
// return number of muon candidates found by the barrel MTTF
//
int L1MuBMTrackFinder::numberOfTracks() {
  int num = 0;
  for (int bx = _cache.getFirstBX(); bx < _cache.getLastBX(); ++bx) {
    num += _cache.size(bx);
  }
  return num;
}


L1MuBMTrackFinder::TFtracks_const_iter L1MuBMTrackFinder::begin(int bx) {

  return _cache.begin(bx);

}


L1MuBMTrackFinder::TFtracks_const_iter L1MuBMTrackFinder::end(int bx) {

  return _cache.end(bx);

}


void L1MuBMTrackFinder::clear() {

  _cache.clear();
  _cache0.clear();
  _cache1.clear();
  _cache2.clear();
  _cache3.clear();
}


//
// return number of muon candidates found by the barrel MTTF at a given bx
//
int L1MuBMTrackFinder::numberOfTracks(int bx) {
  return _cache.size(0);
}


//
// Convert Relative to Absolute Track Addresses
//

int  L1MuBMTrackFinder::setAdd(int ust, int rel_add) {
  unsigned int uadd = rel_add;

  switch (uadd) {
    case  0:  { rel_add =   8; break; }
    case  1:  { rel_add =   9; break; }
    case  2:  { rel_add =   0; break; }
    case  3:  { rel_add =   1; break; }
    case  8:  { rel_add =  10; break; }
    case  9:  { rel_add =  11; break; }
    case  10: { rel_add =   2; break; }
    case  11: { rel_add =   3; break; }
    case  4:  { rel_add =  12; break; }
    case  5:  { rel_add =  13; break; }
    case  6:  { rel_add =   4; break; }
    case  7:  { rel_add =   5; break; }
    case 15:  { rel_add =  15; break; }
    default:  { rel_add =  15; break; }
  }

  if (ust!=1) return rel_add;

  switch (uadd) {
    case  0:  { rel_add =  2; break; }
    case  1:  { rel_add =  1; break; }
    case 15:  { rel_add =  3; break; }
    default:  { rel_add =  3; break; }
  }
  return rel_add;
}


// static data members

L1MuBMTFConfig* L1MuBMTrackFinder::m_config = nullptr;
