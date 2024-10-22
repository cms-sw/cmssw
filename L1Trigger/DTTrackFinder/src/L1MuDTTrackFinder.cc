//-------------------------------------------------
//
//   Class: L1MuDTTrackFinder
//
//   Description: L1 barrel Muon Trigger Track Finder
//
//
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
#include "L1Trigger/DTTrackFinder/interface/L1MuDTTFConfig.h"
#include "L1Trigger/DTTrackFinder/interface/L1MuDTSecProcId.h"
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

L1MuDTTrackFinder::L1MuDTTrackFinder(const edm::ParameterSet& ps, edm::ConsumesCollector&& iC) {
  // set configuration parameters
  m_config = std::make_unique<L1MuDTTFConfig>(ps);

  if (m_config->Debug(1))
    cout << endl;
  if (m_config->Debug(1))
    cout << "**** entering L1MuDTTrackFinder ****" << endl;
  if (m_config->Debug(1))
    cout << endl;

  m_spmap = std::make_unique<L1MuDTSecProcMap>();
  m_epvec.reserve(12);
  m_wsvec.reserve(12);

  _cache.reserve(4 * 17);
  _cache0.reserve(144 * 17);

  m_DTDigiToken = iC.consumes<L1MuDTChambPhContainer>(m_config->getDTDigiInputTag());
}

//--------------
// Destructor --
//--------------

L1MuDTTrackFinder::~L1MuDTTrackFinder() = default;

//--------------
// Operations --
//--------------

//
// setup MTTF configuration
//
void L1MuDTTrackFinder::setup(edm::ConsumesCollector&& iC) {
  // build the barrel Muon Trigger Track Finder

  if (m_config->Debug(1))
    cout << endl;
  if (m_config->Debug(1))
    cout << "**** L1MuDTTrackFinder building ****" << endl;
  if (m_config->Debug(1))
    cout << endl;

  // create new sector processors
  for (int wh = -3; wh <= 3; wh++) {
    if (wh == 0)
      continue;
    for (int sc = 0; sc < 12; sc++) {
      L1MuDTSecProcId tmpspid(wh, sc);
      auto sp = std::make_unique<L1MuDTSectorProcessor>(*this, tmpspid, iC);
      if (m_config->Debug(2))
        cout << "creating " << tmpspid << endl;
      m_spmap->insert(tmpspid, std::move(sp));
    }
  }

  // create new eta processors and wedge sorters
  for (int sc = 0; sc < 12; sc++) {
    auto ep = std::make_unique<L1MuDTEtaProcessor>(*this, sc, iC);
    if (m_config->Debug(2))
      cout << "creating Eta Processor " << sc << endl;
    m_epvec.push_back(std::move(ep));
    auto ws = std::make_unique<L1MuDTWedgeSorter>(*this, sc);
    if (m_config->Debug(2))
      cout << "creating Wedge Sorter " << sc << endl;
    m_wsvec.push_back(std::move(ws));
  }

  // create new muon sorter
  if (m_config->Debug(2))
    cout << "creating DT Muon Sorter " << endl;
  m_ms = std::make_unique<L1MuDTMuonSorter>(*this);
}

//
// run MTTF
//
void L1MuDTTrackFinder::run(const edm::Event& e, const edm::EventSetup& c) {
  // run the barrel Muon Trigger Track Finder

  edm::Handle<L1MuDTChambPhContainer> dttrig;
  e.getByToken(m_DTDigiToken, dttrig);
  if (dttrig->getContainer()->empty())
    return;

  if (m_config->Debug(2))
    cout << endl;
  if (m_config->Debug(2))
    cout << "**** L1MuDTTrackFinder processing ****" << endl;
  if (m_config->Debug(2))
    cout << endl;

  int bx_min = m_config->getBxMin();
  int bx_max = m_config->getBxMax();

  for (int bx = bx_min; bx <= bx_max; bx++) {
    if (dttrig->bxEmpty(bx))
      continue;

    if (m_config->Debug(2))
      cout << "L1MuDTTrackFinder processing bunch-crossing : " << bx << endl;

    // reset MTTF
    reset();

    // run sector processors
    for (auto& sp : *m_spmap) {
      if (m_config->Debug(2))
        cout << "running " << sp.second->id() << endl;
      if (sp.second)
        sp.second->run(bx, e, c);
      if (m_config->Debug(2) && sp.second)
        sp.second->print();
    }

    // run eta processors
    for (auto& ep : m_epvec) {
      if (m_config->Debug(2))
        cout << "running Eta Processor " << ep->id() << endl;
      if (ep)
        ep->run(bx, e, c);
      if (m_config->Debug(2) && ep)
        ep->print();
    }

    // read sector processors
    for (auto& sp : *m_spmap) {
      if (m_config->Debug(2))
        cout << "reading " << sp.second->id() << endl;
      for (int number = 0; number < 2; number++) {
        const L1MuDTTrack* cand = sp.second->tracK(number);
        if (cand && !cand->empty())
          _cache0.push_back(L1MuDTTrackCand(cand->getDataWord(),
                                            cand->bx(),
                                            cand->spid().wheel(),
                                            cand->spid().sector(),
                                            number,
                                            cand->address(1),
                                            cand->address(2),
                                            cand->address(3),
                                            cand->address(4),
                                            cand->tc()));
      }
    }

    // run wedge sorters
    for (auto& ws : m_wsvec) {
      if (m_config->Debug(2))
        cout << "running Wedge Sorter " << ws->id() << endl;
      if (ws)
        ws->run();
      if (m_config->Debug(2) && ws)
        ws->print();
    }

    // run muon sorter
    if (m_config->Debug(2))
      cout << "running DT Muon Sorter" << endl;
    if (m_ms)
      m_ms->run();
    if (m_config->Debug(2) && m_ms)
      m_ms->print();

    // store found track candidates in container (cache)
    if (m_ms->numberOfTracks() > 0) {
      for (auto const& mttf : m_ms->tracks()) {
        if (mttf)
          _cache.push_back(L1MuRegionalCand(mttf->getDataWord(), mttf->bx()));
      }
    }
  }
}

//
// reset MTTF
//
void L1MuDTTrackFinder::reset() {
  for (auto& sp : *m_spmap) {
    if (sp.second)
      sp.second->reset();
  }

  for (auto& ep : m_epvec) {
    if (ep)
      ep->reset();
  }

  for (auto& ws : m_wsvec) {
    if (ws)
      ws->reset();
  }

  if (m_ms)
    m_ms->reset();
}

//
// return Sector Processor container
//
const L1MuDTSectorProcessor* L1MuDTTrackFinder::sp(const L1MuDTSecProcId& id) const { return m_spmap->sp(id); }

//
// return number of muon candidates found by the barrel MTTF
//
int L1MuDTTrackFinder::numberOfTracks() { return _cache.size(); }

L1MuDTTrackFinder::TFtracks_const_iter L1MuDTTrackFinder::begin() { return _cache.begin(); }

L1MuDTTrackFinder::TFtracks_const_iter L1MuDTTrackFinder::end() { return _cache.end(); }

void L1MuDTTrackFinder::clear() {
  _cache.clear();
  _cache0.clear();
}

//
// return number of muon candidates found by the barrel MTTF at a given bx
//
int L1MuDTTrackFinder::numberOfTracks(int bx) {
  int number = 0;
  for (auto const& elem : _cache) {
    if (elem.bx() == bx)
      number++;
  }

  return number;
}
