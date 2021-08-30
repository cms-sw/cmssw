//-------------------------------------------------
//
//   Class: L1MuDTSectorProcessor
//
//   Description: Sector Processor
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

#include "L1Trigger/DTTrackFinder/src/L1MuDTSectorProcessor.h"

//---------------
// C++ Headers --
//---------------

#include <iostream>
#include <cmath>

//-------------------------------
// Collaborating Class Headers --
//-------------------------------

#include <FWCore/Framework/interface/Event.h>
#include "L1Trigger/DTTrackFinder/interface/L1MuDTTFConfig.h"
#include "L1Trigger/DTTrackFinder/interface/L1MuDTSecProcId.h"
#include "L1Trigger/DTTrackFinder/src/L1MuDTSectorReceiver.h"
#include "L1Trigger/DTTrackFinder/src/L1MuDTDataBuffer.h"
#include "L1Trigger/DTTrackFinder/src/L1MuDTExtrapolationUnit.h"
#include "L1Trigger/DTTrackFinder/src/L1MuDTTrackAssembler.h"
#include "L1Trigger/DTTrackFinder/src/L1MuDTAssignmentUnit.h"
#include "L1Trigger/DTTrackFinder/interface/L1MuDTTrackFinder.h"

using namespace std;

// --------------------------------
//       class L1MuDTSectorProcessor
//---------------------------------

//----------------
// Constructors --
//----------------

L1MuDTSectorProcessor::L1MuDTSectorProcessor(const L1MuDTTrackFinder& tf,
                                             const L1MuDTSecProcId& id,
                                             edm::ConsumesCollector iC)
    : m_tf(tf),
      m_spid(id),
      m_SectorReceiver(*this, iC),
      m_DataBuffer(*this),
      m_EU(*this, iC),
      m_TA(*this),
      m_AUs{{{*this, 0, iC}, {*this, 1, iC}}},
      m_TrackCands{{{m_spid}, {m_spid}}},
      m_TracKCands{{{m_spid}, {m_spid}}} {}

//--------------
// Destructor --
//--------------

L1MuDTSectorProcessor::~L1MuDTSectorProcessor() {}

//--------------
// Operations --
//--------------

//
// run Sector Processor
//
void L1MuDTSectorProcessor::run(int bx, const edm::Event& e, const edm::EventSetup& c) {
  // receive data and store them into the data buffer
  m_SectorReceiver.run(bx, e, c);

  // check content of data buffer
  if (m_tf.config()->Debug(4) && m_DataBuffer.numberTSphi() > 0) {
    cout << "Phi track segments received by " << m_spid << " : " << endl;
    m_DataBuffer.printTSphi();
  }

  // perform all extrapolations
  int n_ext = 0;  // number of successful extrapolations
  if (m_DataBuffer.numberTSphi() > 1) {
    m_EU.run(c);
    n_ext = m_EU.numberOfExt();
    if (m_tf.config()->Debug(3) && n_ext > 0) {
      cout << "Number of successful extrapolations : " << n_ext << endl;
      m_EU.print();
    }
  }

  // hardware debug (output from Extrapolator and Quality Sorter)
  // m_EU->print(1);

  // perform track assembling
  if (n_ext > 0) {
    m_TA.run();
    if (m_tf.config()->Debug(3))
      m_TA.print();
  }

  // assign pt, eta, phi and quality
  if (!m_TA.isEmpty(0))
    m_AUs[0].run(c);
  if (!m_TA.isEmpty(1))
    m_AUs[1].run(c);

  if (m_spid.wheel() == -1) {
    if (!m_TrackCands[0].empty() && m_TrackCands[0].address(2) > 3 && m_TrackCands[0].address(2) < 6)
      m_TrackCands[0].reset();
    if (!m_TrackCands[0].empty() && m_TrackCands[0].address(3) > 3 && m_TrackCands[0].address(3) < 6)
      m_TrackCands[0].reset();
    if (!m_TrackCands[0].empty() && m_TrackCands[0].address(4) > 3 && m_TrackCands[0].address(4) < 6)
      m_TrackCands[0].reset();

    if (!m_TracKCands[0].empty() && m_TracKCands[0].address(2) > 3 && m_TracKCands[0].address(2) < 6)
      m_TracKCands[0].reset();
    if (!m_TracKCands[0].empty() && m_TracKCands[0].address(3) > 3 && m_TracKCands[0].address(3) < 6)
      m_TracKCands[0].reset();
    if (!m_TracKCands[0].empty() && m_TracKCands[0].address(4) > 3 && m_TracKCands[0].address(4) < 6)
      m_TracKCands[0].reset();

    if (!m_TrackCands[1].empty() && m_TrackCands[1].address(2) > 3 && m_TrackCands[1].address(2) < 6)
      m_TrackCands[1].reset();
    if (!m_TrackCands[1].empty() && m_TrackCands[1].address(3) > 3 && m_TrackCands[1].address(3) < 6)
      m_TrackCands[1].reset();
    if (!m_TrackCands[1].empty() && m_TrackCands[1].address(4) > 3 && m_TrackCands[1].address(4) < 6)
      m_TrackCands[1].reset();

    if (!m_TracKCands[1].empty() && m_TracKCands[1].address(2) > 3 && m_TracKCands[1].address(2) < 6)
      m_TracKCands[1].reset();
    if (!m_TracKCands[1].empty() && m_TracKCands[1].address(3) > 3 && m_TracKCands[1].address(3) < 6)
      m_TracKCands[1].reset();
    if (!m_TracKCands[1].empty() && m_TracKCands[1].address(4) > 3 && m_TracKCands[1].address(4) < 6)
      m_TracKCands[1].reset();

    if (!m_TrackCands[0].empty() && m_TrackCands[0].address(2) > 7 && m_TrackCands[0].address(2) < 10)
      m_TrackCands[0].reset();
    if (!m_TrackCands[0].empty() && m_TrackCands[0].address(3) > 7 && m_TrackCands[0].address(3) < 10)
      m_TrackCands[0].reset();
    if (!m_TrackCands[0].empty() && m_TrackCands[0].address(4) > 7 && m_TrackCands[0].address(4) < 10)
      m_TrackCands[0].reset();

    if (!m_TracKCands[0].empty() && m_TracKCands[0].address(2) > 7 && m_TracKCands[0].address(2) < 10)
      m_TracKCands[0].reset();
    if (!m_TracKCands[0].empty() && m_TracKCands[0].address(3) > 7 && m_TracKCands[0].address(3) < 10)
      m_TracKCands[0].reset();
    if (!m_TracKCands[0].empty() && m_TracKCands[0].address(4) > 7 && m_TracKCands[0].address(4) < 10)
      m_TracKCands[0].reset();

    if (!m_TrackCands[1].empty() && m_TrackCands[1].address(2) > 7 && m_TrackCands[1].address(2) < 10)
      m_TrackCands[1].reset();
    if (!m_TrackCands[1].empty() && m_TrackCands[1].address(3) > 7 && m_TrackCands[1].address(3) < 10)
      m_TrackCands[1].reset();
    if (!m_TrackCands[1].empty() && m_TrackCands[1].address(4) > 7 && m_TrackCands[1].address(4) < 10)
      m_TrackCands[1].reset();

    if (!m_TracKCands[1].empty() && m_TracKCands[1].address(2) > 7 && m_TracKCands[1].address(2) < 10)
      m_TracKCands[1].reset();
    if (!m_TracKCands[1].empty() && m_TracKCands[1].address(3) > 7 && m_TracKCands[1].address(3) < 10)
      m_TracKCands[1].reset();
    if (!m_TracKCands[1].empty() && m_TracKCands[1].address(4) > 7 && m_TracKCands[1].address(4) < 10)
      m_TracKCands[1].reset();
  }
}

//
// reset Sector Processor
//
void L1MuDTSectorProcessor::reset() {
  m_SectorReceiver.reset();
  m_DataBuffer.reset();
  m_EU.reset();
  m_TA.reset();
  m_AUs[0].reset();
  m_AUs[1].reset();
  m_TrackCands[0].reset();
  m_TrackCands[1].reset();
  m_TracKCands[0].reset();
  m_TracKCands[1].reset();
}

//
// print candidates found in Sector Processor
//
void L1MuDTSectorProcessor::print() const {
  if (anyTrack()) {
    cout << "Muon candidates found in " << m_spid << " : " << endl;
    auto iter = m_TrackCands.cbegin();
    while (iter != m_TrackCands.end()) {
      iter->print();
      iter++;
    }
  }
}

//
// return pointer to nextWheel neighbour
//
const L1MuDTSectorProcessor* L1MuDTSectorProcessor::neighbour() const {
  int sector = m_spid.sector();
  int wheel = m_spid.wheel();

  // the neighbour is in the same wedge with the following definition:
  // current SP  -3  -2  -1  +1  +2  +3
  // neighbour   -2  -1  +1   0  +1  +2

  if (wheel == 1)
    return nullptr;
  wheel = (wheel == -1) ? 1 : (wheel / abs(wheel)) * (abs(wheel) - 1);

  const L1MuDTSecProcId id(wheel, sector);

  return m_tf.sp(id);
}

//
// are there any muon candidates?
//
bool L1MuDTSectorProcessor::anyTrack() const {
  if (!m_TrackCands[0].empty())
    return true;
  if (!m_TrackCands[1].empty())
    return true;

  return false;
}
