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
#include <FWCore/Framework/interface/ESHandle.h>
#include "CondFormats/L1TObjects/interface/L1TMuonBarrelParams.h"
#include "FWCore/Framework/interface/Event.h"

//              ---------------------
//              -- Class Interface --
//              ---------------------

class L1MuBMTFConfig {

  public:

    /// constructor
    L1MuBMTFConfig(const edm::ParameterSet & ps);

    void setDefaultsES(const edm::EventSetup& c);

    /// destructor
    virtual ~L1MuBMTFConfig();

    static edm::InputTag getBMDigiInputTag() { return m_BMDigiInputTag; }
    static edm::InputTag getBMThetaDigiInputTag() { return m_BMThetaDigiInputTag; }

    static bool Debug() { return m_debug; }
    static bool Debug(int level) { return (m_debug && m_dbgLevel >= level); }

    static void setDebugLevel(int level) { m_dbgLevel = level; }
    static int  getDebugLevel() { return m_dbgLevel; }

    static int  getBxMin() { return m_BxMin; }
    static int  getBxMax() { return m_BxMax; }
    static int getExtTSFilter() { return m_extTSFilter; }
    static bool getopenLUTs() { return m_openLUTs; }
    static bool getUseEX21() { return m_useEX21; }
    static bool getEtaTF() { return m_etaTF; }
    static bool getTSOutOfTimeFilter() { return m_TSOutOfTimeFilter; }
    static int  getTSOutOfTimeWindow() { return m_TSOutOfTimeWindow; }
    static int getNbitsExtPhi() { return m_NbitsExtPhi; }
    static int getNbitsExtPhib() { return m_NbitsExtPhib; }
    static int getNbitsPtaPhi() { return m_NbitsPtaPhi; }
    static int getNbitsPtaPhib() { return m_NbitsPtaPhib; }
    static int getNbitsPhiPhi() { return m_NbitsPhiPhi; }
    static int getNbitsPhiPhib() { return m_NbitsPhiPhib; }

  private:

    void setDefaults();

    edm::ESHandle< L1TMuonBarrelParams > bmtfParamsHandle;

  private:

    const edm::ParameterSet* m_ps;
    const edm::EventSetup* m_es;

    static edm::InputTag m_BMDigiInputTag;
    static edm::InputTag m_BMThetaDigiInputTag;

    static bool   m_debug;             // debug flag
    static int    m_dbgLevel;          // debug level

    static int    m_BxMin;
    static int    m_BxMax;

    static int    m_extTSFilter;       // Extrapolation TS-Quality Filter

    static bool   m_openLUTs;          // use open LUTs

    static bool   m_useEX21;           // perform EX21 extrapolation (cross-check EX12)

    static bool   m_etaTF;             // use eta track finder

    static bool   m_TSOutOfTimeFilter; // perform out-of-time TS cancellation
    static int    m_TSOutOfTimeWindow; // phi window size to be checked

    static int    m_NbitsExtPhi;       // precision for extrapolation
    static int    m_NbitsExtPhib;
    static int    m_NbitsPtaPhi;       // precision for pt-assignment
    static int    m_NbitsPtaPhib;
    static int    m_NbitsPhiPhi;       // precision for phi-assignment
    static int    m_NbitsPhiPhib;

};

#endif
