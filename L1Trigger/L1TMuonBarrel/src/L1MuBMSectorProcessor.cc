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
#include "L1Trigger/L1TMuonBarrel/src/L1MuBMTFConfig.h"
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
                                             edm::ConsumesCollector&& iC) :

      m_tf(tf), m_spid(id),
      m_SectorReceiver(new L1MuBMSectorReceiver(*this, std::move(iC))),
      m_DataBuffer(new L1MuBMDataBuffer(*this)),
      m_EU(new L1MuBMExtrapolationUnit(*this)),
      m_TA(new L1MuBMTrackAssembler(*this)),
      m_AUs(), m_TrackCands(), m_TracKCands() {

  // 2 assignment units
  m_AUs.reserve(2);
  m_AUs.push_back(new L1MuBMAssignmentUnit(*this,0));
  m_AUs.push_back(new L1MuBMAssignmentUnit(*this,1));

  // now the 2 track candidates
  m_TrackCands.reserve(2);
  m_TrackCands.push_back(new L1MuBMTrack(m_spid) );
  m_TrackCands.push_back(new L1MuBMTrack(m_spid) );

  m_TracKCands.reserve(2);
  m_TracKCands.push_back(new L1MuBMTrack(m_spid) );
  m_TracKCands.push_back(new L1MuBMTrack(m_spid) );

}


//--------------
// Destructor --
//--------------

L1MuBMSectorProcessor::~L1MuBMSectorProcessor() {

  delete m_SectorReceiver;
  delete m_DataBuffer;
  delete m_EU;
  delete m_TA;
  delete m_AUs[0];
  delete m_AUs[1];
  delete m_TrackCands[0];
  delete m_TrackCands[1];
  delete m_TracKCands[0];
  delete m_TracKCands[1];

}

//--------------
// Operations --
//--------------

//
// run Sector Processor
//
void L1MuBMSectorProcessor::run(int bx, const edm::Event& e, const edm::EventSetup& c) {

  // receive data and store them into the data buffer
  if ( m_SectorReceiver ) m_SectorReceiver->run(bx, e, c);

  // check content of data buffer
  if ( m_DataBuffer ) {
    if ( L1MuBMTFConfig::Debug(4) && m_DataBuffer->numberTSphi() > 0 ) {
      cout << "Phi track segments received by " << m_spid << " : " << endl;
      m_DataBuffer->printTSphi();
    }
  }

  // perform all extrapolations
  int n_ext = 0;	// number of successful extrapolations
  if ( m_EU && m_DataBuffer && m_DataBuffer->numberTSphi() > 1 ) {
    m_EU->run(c);
    n_ext = m_EU->numberOfExt();
    if ( L1MuBMTFConfig::Debug(3) && n_ext > 0  ) {
//    if ( print_flag && n_ext > 0  ) {
      cout << "Number of successful extrapolations : " << n_ext << endl;
      m_EU->print();
    }
  }

  // hardware debug (output from Extrapolator and Quality Sorter)
  // m_EU->print(1);

  // perform track assembling
  if ( m_TA &&  n_ext > 0 ) {
    m_TA->run();
    if ( L1MuBMTFConfig::Debug(3) ) m_TA->print();
  }

  // assign pt, eta, phi and quality
  if ( m_AUs[0] && !m_TA->isEmpty(0) ) m_AUs[0]->run(c);
  if ( m_AUs[1] && !m_TA->isEmpty(1) ) m_AUs[1]->run(c);

  if ( m_spid.wheel() == -1 ) {
    if ( m_TrackCands[0] && !m_TrackCands[0]->empty() && m_TrackCands[0]->address(2)>3 && m_TrackCands[0]->address(2)<6 ) m_TrackCands[0]->reset();
    if ( m_TrackCands[0] && !m_TrackCands[0]->empty() && m_TrackCands[0]->address(3)>3 && m_TrackCands[0]->address(3)<6 ) m_TrackCands[0]->reset();
    if ( m_TrackCands[0] && !m_TrackCands[0]->empty() && m_TrackCands[0]->address(4)>3 && m_TrackCands[0]->address(4)<6 ) m_TrackCands[0]->reset();

    if ( m_TracKCands[0] && !m_TracKCands[0]->empty() && m_TracKCands[0]->address(2)>3 && m_TracKCands[0]->address(2)<6 ) m_TracKCands[0]->reset();
    if ( m_TracKCands[0] && !m_TracKCands[0]->empty() && m_TracKCands[0]->address(3)>3 && m_TracKCands[0]->address(3)<6 ) m_TracKCands[0]->reset();
    if ( m_TracKCands[0] && !m_TracKCands[0]->empty() && m_TracKCands[0]->address(4)>3 && m_TracKCands[0]->address(4)<6 ) m_TracKCands[0]->reset();

    if ( m_TrackCands[1] && !m_TrackCands[1]->empty() && m_TrackCands[1]->address(2)>3 && m_TrackCands[1]->address(2)<6 ) m_TrackCands[1]->reset();
    if ( m_TrackCands[1] && !m_TrackCands[1]->empty() && m_TrackCands[1]->address(3)>3 && m_TrackCands[1]->address(3)<6 ) m_TrackCands[1]->reset();
    if ( m_TrackCands[1] && !m_TrackCands[1]->empty() && m_TrackCands[1]->address(4)>3 && m_TrackCands[1]->address(4)<6 ) m_TrackCands[1]->reset();

    if ( m_TracKCands[1] && !m_TracKCands[1]->empty() && m_TracKCands[1]->address(2)>3 && m_TracKCands[1]->address(2)<6 ) m_TracKCands[1]->reset();
    if ( m_TracKCands[1] && !m_TracKCands[1]->empty() && m_TracKCands[1]->address(3)>3 && m_TracKCands[1]->address(3)<6 ) m_TracKCands[1]->reset();
    if ( m_TracKCands[1] && !m_TracKCands[1]->empty() && m_TracKCands[1]->address(4)>3 && m_TracKCands[1]->address(4)<6 ) m_TracKCands[1]->reset();

    if ( m_TrackCands[0] && !m_TrackCands[0]->empty() && m_TrackCands[0]->address(2)>7 && m_TrackCands[0]->address(2)<10 ) m_TrackCands[0]->reset();
    if ( m_TrackCands[0] && !m_TrackCands[0]->empty() && m_TrackCands[0]->address(3)>7 && m_TrackCands[0]->address(3)<10 ) m_TrackCands[0]->reset();
    if ( m_TrackCands[0] && !m_TrackCands[0]->empty() && m_TrackCands[0]->address(4)>7 && m_TrackCands[0]->address(4)<10 ) m_TrackCands[0]->reset();

    if ( m_TracKCands[0] && !m_TracKCands[0]->empty() && m_TracKCands[0]->address(2)>7 && m_TracKCands[0]->address(2)<10 ) m_TracKCands[0]->reset();
    if ( m_TracKCands[0] && !m_TracKCands[0]->empty() && m_TracKCands[0]->address(3)>7 && m_TracKCands[0]->address(3)<10 ) m_TracKCands[0]->reset();
    if ( m_TracKCands[0] && !m_TracKCands[0]->empty() && m_TracKCands[0]->address(4)>7 && m_TracKCands[0]->address(4)<10 ) m_TracKCands[0]->reset();

    if ( m_TrackCands[1] && !m_TrackCands[1]->empty() && m_TrackCands[1]->address(2)>7 && m_TrackCands[1]->address(2)<10 ) m_TrackCands[1]->reset();
    if ( m_TrackCands[1] && !m_TrackCands[1]->empty() && m_TrackCands[1]->address(3)>7 && m_TrackCands[1]->address(3)<10 ) m_TrackCands[1]->reset();
    if ( m_TrackCands[1] && !m_TrackCands[1]->empty() && m_TrackCands[1]->address(4)>7 && m_TrackCands[1]->address(4)<10 ) m_TrackCands[1]->reset();

    if ( m_TracKCands[1] && !m_TracKCands[1]->empty() && m_TracKCands[1]->address(2)>7 && m_TracKCands[1]->address(2)<10 ) m_TracKCands[1]->reset();
    if ( m_TracKCands[1] && !m_TracKCands[1]->empty() && m_TracKCands[1]->address(3)>7 && m_TracKCands[1]->address(3)<10 ) m_TracKCands[1]->reset();
    if ( m_TracKCands[1] && !m_TracKCands[1]->empty() && m_TracKCands[1]->address(4)>7 && m_TracKCands[1]->address(4)<10 ) m_TracKCands[1]->reset();
  }

}


//
// reset Sector Processor
//
void L1MuBMSectorProcessor::reset() {

  if ( m_SectorReceiver ) m_SectorReceiver->reset();
  if ( m_DataBuffer ) m_DataBuffer->reset();
  if ( m_EU ) m_EU->reset();
  if ( m_TA ) m_TA->reset();
  if ( m_AUs[0] ) m_AUs[0]->reset();
  if ( m_AUs[1] ) m_AUs[1]->reset();
  if ( m_TrackCands[0] ) m_TrackCands[0]->reset();
  if ( m_TrackCands[1] ) m_TrackCands[1]->reset();
  if ( m_TracKCands[0] ) m_TracKCands[0]->reset();
  if ( m_TracKCands[1] ) m_TracKCands[1]->reset();

}


//
// print candidates found in Sector Processor
//
void L1MuBMSectorProcessor::print() const {

  if ( anyTrack() ) {
    cout << "Muon candidates found in " << m_spid << " : " << endl;
    vector<L1MuBMTrack*>::const_iterator iter = m_TrackCands.begin();
    while ( iter != m_TrackCands.end() ) {
      if ( *iter) (*iter)->print();
      iter++;
    }
  }

}


//
// return pointer to nextWheel neighbour
//
const L1MuBMSectorProcessor* L1MuBMSectorProcessor::neighbour() const {

  int sector = m_spid.sector();
  int wheel  = m_spid.wheel();

  // the neighbour is in the same wedge with the following definition:
  // current SP  -3  -2  -1  +1  +2  +3
  // neighbour   -2  -1  +1   0  +1  +2

  if ( wheel == 1) return nullptr;
  wheel = (wheel == -1) ? 1 : (wheel/abs(wheel)) * (abs(wheel)-1);

  const L1MuBMSecProcId id(wheel,sector);

  return m_tf.sp(id);

}


//
// are there any muon candidates?
//
bool L1MuBMSectorProcessor::anyTrack() const {

  if ( m_TrackCands[0] && !m_TrackCands[0]->empty() ) return true;
  if ( m_TrackCands[1] && !m_TrackCands[1]->empty() ) return true;

  return false;

}
