//-------------------------------------------------
//
//   Class: L1MuDTTrack
//
//   Description: Muon Track Candidate
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

#include "L1Trigger/DTTrackFinder/interface/L1MuDTTrack.h"

//---------------
// C++ Headers --
//---------------

#include <iostream>
#include <iomanip>

//-------------------------------
// Collaborating Class Headers --
//-------------------------------

#include "L1Trigger/DTTrackFinder/interface/L1MuDTSecProcId.h"
#include "L1Trigger/DTTrackFinder/interface/L1MuDTTrackSegPhi.h"
#include "L1Trigger/DTTrackFinder/interface/L1MuDTTrackSegEta.h"
#include "CondFormats/L1TObjects/interface/L1MuTriggerPtScale.h"
#include "CondFormats/DataRecord/interface/L1MuTriggerPtScaleRcd.h"
#include "CondFormats/L1TObjects/interface/L1MuPacking.h"

using namespace std;

// --------------------------------
//       class L1MuDTTrack
//---------------------------------

//----------------
// Constructors --
//----------------
L1MuDTTrack::L1MuDTTrack()
    : L1MuRegionalCand(0, 0),
      m_spid(L1MuDTSecProcId()),
      m_name("L1MuDTTrack"),
      m_empty(true),
      m_tc(UNDEF),
      m_addArray(),
      m_tsphiList(),
      m_tsetaList() {
  m_tsphiList.reserve(4);
  m_tsetaList.reserve(3);

  setType(0);
  setChargeValid(true);
}

L1MuDTTrack::L1MuDTTrack(const L1MuDTSecProcId& spid)
    : L1MuRegionalCand(0, 0),
      m_spid(spid),
      m_name("L1MuDTTrack"),
      m_empty(true),
      m_tc(UNDEF),
      m_addArray(),
      m_tsphiList(),
      m_tsetaList() {
  m_tsphiList.reserve(4);
  m_tsetaList.reserve(3);

  setType(0);
  setChargeValid(true);
}

L1MuDTTrack::L1MuDTTrack(const L1MuDTTrack& id)
    : L1MuRegionalCand(id),
      m_spid(id.m_spid),
      m_name(id.m_name),
      m_empty(id.m_empty),
      m_tc(id.m_tc),
      m_addArray(id.m_addArray),
      m_tsphiList(id.m_tsphiList),
      m_tsetaList(id.m_tsetaList) {}

//--------------
// Destructor --
//--------------
L1MuDTTrack::~L1MuDTTrack() {}

//--------------
// Operations --
//--------------

//
// reset Muon Track Candidate
//
void L1MuDTTrack::reset() {
  L1MuRegionalCand::reset();
  m_empty = true;
  m_tc = UNDEF;
  m_addArray.reset();
  m_tsphiList.clear();
  m_tsetaList.clear();
}

//
// set (packed) eta-code of muon candidate
//
void L1MuDTTrack::setEta(int eta) {
  // eta is a signed integer [-32,31],
  // representing 64 bins in an interval [-1.2,+1.2]
  // first convert eta into an unsigned integer
  L1MuSignedPacking<6> pEta;
  setEtaPacked(pEta.packedFromIdx(eta));
}

//
// return start phi track segment
//
const L1MuDTTrackSegPhi& L1MuDTTrack::getStartTSphi() const { return m_tsphiList.front(); }

//
// return end phi track segment
//
const L1MuDTTrackSegPhi& L1MuDTTrack::getEndTSphi() const { return m_tsphiList.back(); }

//
// return start eta track segment
//
const L1MuDTTrackSegEta& L1MuDTTrack::getStartTSeta() const { return m_tsetaList.front(); }

//
// return end eta track segment
//
const L1MuDTTrackSegEta& L1MuDTTrack::getEndTSeta() const { return m_tsetaList.back(); }

//
// set phi track segments used to form the muon candidate
//
void L1MuDTTrack::setTSphi(const vector<const L1MuDTTrackSegPhi*>& tsList) {
  if (!tsList.empty()) {
    vector<const L1MuDTTrackSegPhi*>::const_iterator iter;
    for (iter = tsList.begin(); iter != tsList.end(); iter++) {
      if (*iter)
        m_tsphiList.push_back(**iter);
    }
  }
}

//
// set eta track segments used to form the muon candidate
//
void L1MuDTTrack::setTSeta(const vector<const L1MuDTTrackSegEta*>& tsList) {
  if (!tsList.empty()) {
    vector<const L1MuDTTrackSegEta*>::const_iterator iter;
    for (iter = tsList.begin(); iter != tsList.end(); iter++) {
      if (*iter)
        m_tsetaList.push_back(**iter);
    }
  }
}

//
// Assignment operator
//
L1MuDTTrack& L1MuDTTrack::operator=(const L1MuDTTrack& track) {
  if (this != &track) {
    this->setBx(track.bx());
    this->setDataWord(track.getDataWord());
    m_spid = track.m_spid;
    m_empty = track.m_empty;
    m_name = track.m_name;
    m_tc = track.m_tc;
    m_addArray = track.m_addArray;
    m_tsphiList = track.m_tsphiList;
    m_tsetaList = track.m_tsetaList;
  }
  return *this;
}

//
// Equal operator
//
bool L1MuDTTrack::operator==(const L1MuDTTrack& track) const {
  if (m_spid != track.m_spid)
    return false;
  if (m_empty != track.m_empty)
    return false;
  if (m_tc != track.m_tc)
    return false;
  if (bx() != track.bx())
    return false;
  if (phi() != track.phi())
    return false;
  if (eta() != track.eta())
    return false;
  if (fineEtaBit() != track.fineEtaBit())
    return false;
  if (pt() != track.pt())
    return false;
  if (charge() != track.charge())
    return false;
  if (quality() != track.quality())
    return false;
  if (m_addArray != track.m_addArray)
    return false;
  return true;
}

//
// Unequal operator
//
bool L1MuDTTrack::operator!=(const L1MuDTTrack& track) const {
  if (m_spid != track.m_spid)
    return true;
  if (m_empty != track.m_empty)
    return true;
  if (m_tc != track.m_tc)
    return true;
  if (bx() != track.bx())
    return true;
  if (phi() != track.phi())
    return true;
  if (eta() != track.eta())
    return true;
  if (fineEtaBit() != track.fineEtaBit())
    return true;
  if (pt() != track.pt())
    return true;
  if (charge() != track.charge())
    return true;
  if (quality() != track.quality())
    return true;
  if (m_addArray != track.m_addArray)
    return true;
  return false;
}

//
// print parameters of track candidate
//
void L1MuDTTrack::print() const {
  if (!empty()) {
    cout.setf(ios::showpoint);
    cout.setf(ios::right, ios::adjustfield);
    cout << setiosflags(ios::showpoint | ios::fixed);
    cout << "MUON : "
         << "pt = " << setw(2) << pt_packed() << "  "
         << "charge = " << setw(2) << charge_packed() << "  "
         << "eta = " << setw(2) << eta_packed() << " (" << setw(1) << finehalo_packed() << ")  "
         << "phi = " << setw(3) << phi_packed() << "  "
         << "quality = " << setw(1) << quality_packed() << '\t' << "class = " << tc() << "  "
         << "bx = " << setw(2) << bx() << endl;
    cout << "       found in " << m_spid << " with phi track segments :" << endl;
    vector<L1MuDTTrackSegPhi>::const_iterator iter;
    for (iter = m_tsphiList.begin(); iter != m_tsphiList.end(); iter++) {
      cout << "       " << (*iter) << endl;
    }
  }
}

//
// output stream operator for track candidate
//
ostream& operator<<(ostream& s, const L1MuDTTrack& id) {
  if (!id.empty()) {
    s << setiosflags(ios::showpoint | ios::fixed) << "pt = " << setw(2) << id.pt_packed() << "  "
      << "charge = " << setw(2) << id.charge_packed() << "  "
      << "eta = " << setw(2) << id.eta_packed() << " (" << setw(1) << id.finehalo_packed() << ")  "
      << "phi = " << setw(3) << id.phi_packed() << "  "
      << "quality = " << setw(1) << id.quality_packed() << '\t' << "bx = " << setw(2) << id.bx();
  }
  return s;
}
