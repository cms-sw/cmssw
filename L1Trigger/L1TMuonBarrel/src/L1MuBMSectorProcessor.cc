//-------------------------------------------------
//
//   Class: L1MuBMSectorProcessor
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

#include "L1MuBMSectorProcessor.h"

//---------------
// C++ Headers --
//---------------

#include <iostream>
#include <cmath>

//-------------------------------
// Collaborating Class Headers --
//-------------------------------

#include <FWCore/Framework/interface/Event.h>
#include "L1Trigger/L1TMuonBarrel/interface/L1MuBMTFConfig.h"
#include "L1Trigger/L1TMuonBarrel/src/L1MuBMSectorReceiver.h"
#include "L1Trigger/L1TMuonBarrel/src/L1MuBMDataBuffer.h"
#include "L1Trigger/L1TMuonBarrel/src/L1MuBMExtrapolationUnit.h"
#include "L1Trigger/L1TMuonBarrel/src/L1MuBMTrackAssembler.h"
#include "L1Trigger/L1TMuonBarrel/src/L1MuBMAssignmentUnit.h"
#include "L1Trigger/L1TMuonBarrel/interface/L1MuBMTrackFinder.h"

#include "DataFormats/L1TMuon/interface/BMTF/L1MuBMSecProcId.h"
#include "DataFormats/L1TMuon/interface/L1MuBMTrack.h"

using namespace std;

// --------------------------------
//       class L1MuBMSectorProcessor
//---------------------------------

//----------------
// Constructors --
//----------------

L1MuBMSectorProcessor::L1MuBMSectorProcessor(const L1MuBMTrackFinder& tf,
                                             const L1MuBMSecProcId& id,
                                             edm::ConsumesCollector&& iC)
    :

      m_tf(tf),
      m_spid(id),
      m_SectorReceiver(*this, std::move(iC)),
      m_DataBuffer(*this),
      m_EU(*this, iC),
      m_TA(*this),
      m_bmtfParamsToken(iC.esConsumes()),
      m_AUs(),
      m_TrackCands(),
      m_TracKCands() {
  // 2 assignment units
  m_AUs.reserve(2);
  m_AUs.emplace_back(*this, 0);
  m_AUs.emplace_back(*this, 1);

  // now the 2 track candidates
  m_TrackCands.reserve(2);
  m_TrackCands.emplace_back(m_spid);
  m_TrackCands.push_back(m_spid);

  m_TracKCands.reserve(2);
  m_TracKCands.push_back(m_spid);
  m_TracKCands.push_back(m_spid);
}

//--------------
// Operations --
//--------------

//
// run Sector Processor
//
void L1MuBMSectorProcessor::run(int bx, const edm::Event& e, const edm::EventSetup& c) {
  // receive data and store them into the data buffer
  m_SectorReceiver.run(bx, e, c);

  // check content of data buffer
  if (config().Debug(4) && m_DataBuffer.numberTSphi() > 0) {
    cout << "Phi track segments received by " << m_spid << " : " << endl;
    m_DataBuffer.printTSphi();
  }

  // perform all extrapolations
  int n_ext = 0;  // number of successful extrapolations
  if (m_DataBuffer.numberTSphi() > 1) {
    m_EU.run(c);
    n_ext = m_EU.numberOfExt();
    if (config().Debug(3) && n_ext > 0) {
      //    if ( print_flag && n_ext > 0  ) {
      cout << "Number of successful extrapolations : " << n_ext << endl;
      m_EU.print();
    }
  }

  // hardware debug (output from Extrapolator and Quality Sorter)
  // m_EU.print(1);

  // perform track assembling
  if (n_ext > 0) {
    m_TA.run();
    if (config().Debug(3))
      m_TA.print();
  }

  L1TMuonBarrelParams const& bmtfParams = c.getData(m_bmtfParamsToken);

  // assign pt, eta, phi and quality
  if (!m_TA.isEmpty(0))
    m_AUs[0].run(bmtfParams);
  if (!m_TA.isEmpty(1))
    m_AUs[1].run(bmtfParams);

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
void L1MuBMSectorProcessor::reset() {
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
void L1MuBMSectorProcessor::print() const {
  if (anyTrack()) {
    cout << "Muon candidates found in " << m_spid << " : " << endl;
    for (auto const& t : m_TrackCands) {
      t.print();
    }
  }
}

//
// return pointer to nextWheel neighbour
//
const L1MuBMSectorProcessor* L1MuBMSectorProcessor::neighbour() const {
  int sector = m_spid.sector();
  int wheel = m_spid.wheel();

  // the neighbour is in the same wedge with the following definition:
  // current SP  -3  -2  -1  +1  +2  +3
  // neighbour   -2  -1  +1   0  +1  +2

  if (wheel == 1)
    return nullptr;
  wheel = (wheel == -1) ? 1 : (wheel / abs(wheel)) * (abs(wheel) - 1);

  const L1MuBMSecProcId id(wheel, sector);

  return m_tf.sp(id);
}

//
// are there any muon candidates?
//
bool L1MuBMSectorProcessor::anyTrack() const {
  if (!m_TrackCands[0].empty())
    return true;
  if (!m_TrackCands[1].empty())
    return true;

  return false;
}

const L1MuBMTFConfig& L1MuBMSectorProcessor::config() const { return m_tf.config(); }
