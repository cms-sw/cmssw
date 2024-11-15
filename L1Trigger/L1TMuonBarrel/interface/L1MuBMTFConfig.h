//-------------------------------------------------
//
/**  \class L1MuBMTFConfig
 *
 *   Configuration parameters for L1MuBMTrackFinder
 *
 *
 *
 *   N. Neumeister            CERN EP
 */
//
//--------------------------------------------------
#ifndef L1MUBM_TF_CONFIG_H
#define L1MUBM_TF_CONFIG_H

//---------------
// C++ Headers --
//---------------

#include <string>

//----------------------
// Base Class Headers --
//----------------------

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

#include <FWCore/ParameterSet/interface/ParameterSet.h>
#include <FWCore/Utilities/interface/InputTag.h>
#include "CondFormats/L1TObjects/interface/L1TMuonBarrelParams.h"

//              ---------------------
//              -- Class Interface --
//              ---------------------

class L1MuBMTFConfig {
public:
  /// constructor
  explicit L1MuBMTFConfig(const edm::ParameterSet& ps);

  void setDefaultsES(const L1TMuonBarrelParams&);

  edm::InputTag getBMDigiInputTag() const { return m_BMDigiInputTag; }
  edm::InputTag getBMThetaDigiInputTag() const { return m_BMThetaDigiInputTag; }

  bool Debug() const { return m_debug; }
  bool Debug(int level) const { return (m_debug && m_dbgLevel >= level); }

  void setDebugLevel(int level) { m_dbgLevel = level; }
  int getDebugLevel() const { return m_dbgLevel; }

  int getBxMin() const { return m_BxMin; }
  int getBxMax() const { return m_BxMax; }
  int getExtTSFilter() const { return m_extTSFilter; }
  bool getopenLUTs() const { return m_openLUTs; }
  bool getUseEX21() const { return m_useEX21; }
  bool getEtaTF() const { return m_etaTF; }
  bool getTSOutOfTimeFilter() const { return m_TSOutOfTimeFilter; }
  int getTSOutOfTimeWindow() const { return m_TSOutOfTimeWindow; }
  int getNbitsExtPhi() const { return m_NbitsExtPhi; }
  int getNbitsExtPhib() const { return m_NbitsExtPhib; }
  int getNbitsPtaPhi() const { return m_NbitsPtaPhi; }
  int getNbitsPtaPhib() const { return m_NbitsPtaPhib; }
  int getNbitsPhiPhi() const { return m_NbitsPhiPhi; }
  int getNbitsPhiPhib() const { return m_NbitsPhiPhib; }

private:
  void setDefaults(edm::ParameterSet const&);

private:
  edm::InputTag m_BMDigiInputTag;
  edm::InputTag m_BMThetaDigiInputTag;

  int m_dbgLevel = -1;  // debug level

  int m_BxMin = -9;
  int m_BxMax = 7;

  int m_extTSFilter = 1;  // Extrapolation TS-Quality Filter

  int m_TSOutOfTimeWindow = 1;  // phi window size to be checked

  int m_NbitsExtPhi = 8;  // precision for extrapolation
  int m_NbitsExtPhib = 8;
  int m_NbitsPtaPhi = 12;  // precision for pt-assignment
  int m_NbitsPtaPhib = 10;
  int m_NbitsPhiPhi = 10;  // precision for phi-assignment
  int m_NbitsPhiPhib = 10;

  bool m_debug = false;              // debug flag
  bool m_openLUTs = false;           // use open LUTs
  bool m_useEX21 = false;            // perform EX21 extrapolation (cross-check EX12)
  bool m_etaTF = true;               // use eta track finder
  bool m_TSOutOfTimeFilter = false;  // perform out-of-time TS cancellation
};

#endif
