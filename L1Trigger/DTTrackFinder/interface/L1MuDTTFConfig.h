//-------------------------------------------------
//
/**  \class L1MuDTTFConfig
 *
 *   Configuration parameters for L1MuDTTrackFinder
 *
 *
 *
 *   N. Neumeister            CERN EP
 */
//
//--------------------------------------------------
#ifndef L1MUDT_TF_CONFIG_H
#define L1MUDT_TF_CONFIG_H

//---------------
// C++ Headers --
//---------------

#include <string>
#include <atomic>

//----------------------
// Base Class Headers --
//----------------------

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

#include <FWCore/ParameterSet/interface/ParameterSet.h>
#include <FWCore/Utilities/interface/InputTag.h>

//              ---------------------
//              -- Class Interface --
//              ---------------------

class L1MuDTTFConfig {
public:
  /// constructor
  L1MuDTTFConfig(const edm::ParameterSet& ps);

  /// destructor
  virtual ~L1MuDTTFConfig();

  edm::InputTag getDTDigiInputTag() const { return m_DTDigiInputTag; }
  edm::InputTag getCSCTrSInputTag() const { return m_CSCTrSInputTag; }

  static bool Debug() { return m_debug; }
  static bool Debug(int level) { return (m_debug && m_dbgLevel >= level); }

  static void setDebugLevel(int level) { m_dbgLevel = level; }
  static int getDebugLevel() { return m_dbgLevel; }
  static int getBxMinGlobally() { return s_BxMin; }
  static int getBxMaxGlobally() { return s_BxMax; }

  int getBxMin() const { return m_BxMin; }
  int getBxMax() const { return m_BxMax; }
  bool overlap() const { return m_overlap; }
  int getExtTSFilter() const { return m_extTSFilter; }
  bool getopenLUTs() const { return m_openLUTs; }
  bool getUseEX21() const { return m_useEX21; }
  bool getEtaTF() const { return m_etaTF; }
  bool getEtaCanc() const { return m_etacanc; }
  bool getTSOutOfTimeFilter() const { return m_TSOutOfTimeFilter; }
  int getTSOutOfTimeWindow() const { return m_TSOutOfTimeWindow; }
  int getNbitsExtPhi() const { return m_NbitsExtPhi; }
  int getNbitsExtPhib() const { return m_NbitsExtPhib; }
  int getNbitsPtaPhi() const { return m_NbitsPtaPhi; }
  int getNbitsPtaPhib() const { return m_NbitsPtaPhib; }
  int getNbitsPhiPhi() const { return m_NbitsPhiPhi; }
  int getNbitsPhiPhib() const { return m_NbitsPhiPhib; }

private:
  void setDefaults(const edm::ParameterSet&);

private:
  edm::InputTag m_DTDigiInputTag;
  edm::InputTag m_CSCTrSInputTag;

  static std::atomic<bool> m_debug;    // debug flag
  static std::atomic<int> m_dbgLevel;  // debug level
  static std::atomic<int> s_BxMin;
  static std::atomic<int> s_BxMax;

  bool m_overlap;  // barrel-endcap overlap region

  int m_BxMin;
  int m_BxMax;

  int m_extTSFilter;  // Extrapolation TS-Quality Filter

  bool m_openLUTs;  // use open LUTs

  bool m_useEX21;  // perform EX21 extrapolation (cross-check EX12)

  bool m_etaTF;  // use eta track finder

  bool m_etacanc;  // use etaFlag for CSC segment cancellation

  bool m_TSOutOfTimeFilter;  // perform out-of-time TS cancellation
  int m_TSOutOfTimeWindow;   // phi window size to be checked

  int m_NbitsExtPhi;  // precision for extrapolation
  int m_NbitsExtPhib;
  int m_NbitsPtaPhi;  // precision for pt-assignment
  int m_NbitsPtaPhib;
  int m_NbitsPhiPhi;  // precision for phi-assignment
  int m_NbitsPhiPhib;
};

#endif
