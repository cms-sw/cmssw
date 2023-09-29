//-------------------------------------------------
//
//   Class: L1MuDTTFConfig
//
//   Description: DTTrackFinder parameters for L1MuDTTrackFinder
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

#include "L1Trigger/DTTrackFinder/interface/L1MuDTTFConfig.h"

//---------------
// C++ Headers --
//---------------

#include <iostream>
#include <string>

//-------------------------------
// Collaborating Class Headers --
//-------------------------------

using namespace std;

// --------------------------------
//       class L1MuDTTFConfig
//---------------------------------

//----------------
// Constructors --
//----------------

L1MuDTTFConfig::L1MuDTTFConfig(const edm::ParameterSet& ps) { setDefaults(ps); }

//--------------
// Destructor --
//--------------
L1MuDTTFConfig::~L1MuDTTFConfig() {}

//--------------
// Operations --
//--------------

void L1MuDTTFConfig::setDefaults(const edm::ParameterSet& ps) {
  m_DTDigiInputTag = ps.getParameter<edm::InputTag>("DTDigi_Source");
  m_CSCTrSInputTag = ps.getParameter<edm::InputTag>("CSCStub_Source");

  m_debug = true;
  m_dbgLevel = ps.getUntrackedParameter<int>("Debug", 0);

  m_overlap = ps.getUntrackedParameter<bool>("Overlap", true);

  // set min and max bunch crossing
  m_BxMin = ps.getUntrackedParameter<int>("BX_min", -9);
  m_BxMax = ps.getUntrackedParameter<int>("BX_max", 7);
  s_BxMin = m_BxMin;
  s_BxMax = m_BxMax;

  // set Filter for Extrapolator
  m_extTSFilter = ps.getUntrackedParameter<int>("Extrapolation_Filter", 1);

  // set switch for open LUTs usage
  m_openLUTs = ps.getUntrackedParameter<bool>("Open_LUTs", false);

  // set switch for EX21 usage
  m_useEX21 = ps.getUntrackedParameter<bool>("Extrapolation_21", false);

  // set switch for eta track finder usage
  m_etaTF = ps.getUntrackedParameter<bool>("EtaTrackFinder", true);

  // set switch for etaFlag cancellation of CSC segments
  m_etacanc = ps.getUntrackedParameter<bool>("CSC_Eta_Cancellation", false);

  // set Filter for Out-of-time Track Segments
  m_TSOutOfTimeFilter = ps.getUntrackedParameter<bool>("OutOfTime_Filter", false);
  m_TSOutOfTimeWindow = ps.getUntrackedParameter<int>("OutOfTime_Filter_Window", 1);

  // set precision for extrapolation
  m_NbitsExtPhi = ps.getUntrackedParameter<int>("Extrapolation_nbits_Phi", 8);
  m_NbitsExtPhib = ps.getUntrackedParameter<int>("Extrapolation_nbits_PhiB", 8);

  // set precision for pt-assignment
  m_NbitsPtaPhi = ps.getUntrackedParameter<int>("PT_Assignment_nbits_Phi", 12);
  m_NbitsPtaPhib = ps.getUntrackedParameter<int>("PT_Assignment_nbits_PhiB", 10);

  // set precision for phi-assignment look-up tables
  m_NbitsPhiPhi = ps.getUntrackedParameter<int>("PHI_Assignment_nbits_Phi", 10);
  m_NbitsPhiPhib = ps.getUntrackedParameter<int>("PHI_Assignment_nbits_PhiB", 10);

  if (Debug(1))
    cout << endl;
  if (Debug(1))
    cout << "*******************************************" << endl;
  if (Debug(1))
    cout << "**** L1 barrel Track Finder settings : ****" << endl;
  if (Debug(1))
    cout << "*******************************************" << endl;
  if (Debug(1))
    cout << endl;

  if (Debug(1))
    cout << "L1 barrel Track Finder : DT Digi Source:  " << m_DTDigiInputTag << endl;
  if (Debug(1))
    cout << "L1 barrel Track Finder : CSC Stub Source: " << m_CSCTrSInputTag << endl;
  if (Debug(1))
    cout << endl;

  if (Debug(1))
    cout << "L1 barrel Track Finder : debug level: " << m_dbgLevel << endl;

  if (Debug(1) && m_overlap) {
    cout << "L1 barrel Track Finder : barrel-endcap overlap region : on" << endl;
  }
  if (Debug(1) && !m_overlap) {
    cout << "L1 barrel Track Finder : barrel-endcap overlap region : off" << endl;
  }

  if (Debug(1))
    cout << "L1 barrel Track Finder : minimal bunch-crossing : " << m_BxMin << endl;
  if (Debug(1))
    cout << "L1 barrel Track Finder : maximal bunch-crossing : " << m_BxMax << endl;

  if (Debug(1))
    cout << "L1 barrel Track Finder : Extrapolation Filter : " << m_extTSFilter << endl;

  if (Debug(1) && m_openLUTs) {
    cout << "L1 barrel Track Finder : use open LUTs : on" << endl;
  }
  if (Debug(1) && !m_openLUTs) {
    cout << "L1 barrel Track Finder : use open LUTs : off" << endl;
  }

  if (Debug(1) && m_useEX21) {
    cout << "L1 barrel Track Finder : use EX21 extrapolations : on" << endl;
  }
  if (Debug(1) && !m_useEX21) {
    cout << "L1 barrel Track Finder : use EX21 extrapolations : off" << endl;
  }

  if (Debug(1) && m_etaTF) {
    cout << "L1 barrel Track Finder : Eta Track Finder : on" << endl;
  }
  if (Debug(1) && !m_etaTF) {
    cout << "L1 barrel Track Finder : Eta Track Finder : off" << endl;
  }

  if (Debug(1) && m_etacanc) {
    cout << "L1 barrel Track Finder : CSC etaFlag cancellation : on" << endl;
  }
  if (Debug(1) && !m_etacanc) {
    cout << "L1 barrel Track Finder : CSC etaFlag cancellation : off" << endl;
  }

  if (Debug(1) && m_TSOutOfTimeFilter) {
    cout << "L1 barrel Track Finder : out-of-time TS filter : on" << endl;
    cout << "L1 barrel Track Finder : out-of-time TS filter window : " << m_TSOutOfTimeWindow << endl;
  }
  if (Debug(1) && !m_TSOutOfTimeFilter) {
    cout << "L1 barrel Track Finder : out-of-time TS filter : off" << endl;
  }

  if (Debug(1))
    cout << "L1 barrel Track Finder : # of bits used for phi  (extrapolation)  : " << m_NbitsExtPhi << endl;
  if (Debug(1))
    cout << "L1 barrel Track Finder : # of bits used for phib (extrapolation)  : " << m_NbitsExtPhib << endl;
  if (Debug(1))
    cout << "L1 barrel Track Finder : # of bits used for phi  (pt-assignment)  : " << m_NbitsPtaPhi << endl;
  if (Debug(1))
    cout << "L1 barrel Track Finder : # of bits used for phib (pt-assignment)  : " << m_NbitsPtaPhib << endl;
  if (Debug(1))
    cout << "L1 barrel Track Finder : # of bits used for phi  (phi-assignment) : " << m_NbitsPhiPhi << endl;
  if (Debug(1))
    cout << "L1 barrel Track Finder : # of bits used for phib (phi-assignment) : " << m_NbitsPhiPhib << endl;
}

std::atomic<bool> L1MuDTTFConfig::m_debug{false};
std::atomic<int> L1MuDTTFConfig::m_dbgLevel{-1};
std::atomic<int> L1MuDTTFConfig::s_BxMin{-9};
std::atomic<int> L1MuDTTFConfig::s_BxMax{-7};
