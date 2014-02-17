//-------------------------------------------------
//
/** \class L1MuGMTConfig
 *  Configuration parameters for L1GlobalMuonTrigger.
*/
//
//   $Date: 2012/02/10 14:19:28 $
//   $Revision: 1.14 $
//
//   Author :
//   N. Neumeister             CERN EP
//
//   Migrated to CMSSW:
//   I. Mikulec
//
//--------------------------------------------------
#ifndef L1TriggerGlobalMuonTrigger_L1MuGMTConfig_h
#define L1TriggerGlobalMuonTrigger_L1MuGMTConfig_h

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
#include "FWCore/ParameterSet/interface/ParameterSet.h"

namespace edm {
  class InputTag;
}

class L1MuGMTRegCDLConfig;
class L1MuGMTRegMMConfigPhi;
class L1MuGMTRegMMConfigEta;
class L1MuGMTRegMMConfigPt;
class L1MuGMTRegMMConfigCharge;
class L1MuGMTRegMMConfigMIP;
class L1MuGMTRegMMConfigISO;
class L1MuGMTRegMMConfigSRK;
class L1MuGMTRegSortRankOffset;

class L1MuGMTEtaLUT;
class L1MuGMTLFCOUDeltaEtaLUT;
class L1MuGMTLFDeltaEtaLUT;
class L1MuGMTLFDisableHotLUT;
class L1MuGMTLFEtaConvLUT;
class L1MuGMTLFMatchQualLUT;
class L1MuGMTLFMergeRankCombineLUT;
class L1MuGMTLFMergeRankEtaPhiLUT;
class L1MuGMTLFMergeRankEtaQLUT;
class L1MuGMTLFMergeRankPtQLUT;
class L1MuGMTLFOvlEtaConvLUT;
class L1MuGMTLFPhiProEtaConvLUT;
class L1MuGMTLFPhiProLUT;
class L1MuGMTLFPtMixLUT;
class L1MuGMTLFSortRankCombineLUT;
class L1MuGMTLFSortRankEtaPhiLUT;
class L1MuGMTLFSortRankEtaQLUT;
class L1MuGMTLFSortRankPtQLUT;
class L1MuGMTMIAUEtaConvLUT;
class L1MuGMTMIAUEtaProLUT;
class L1MuGMTMIAUPhiPro1LUT;
class L1MuGMTMIAUPhiPro2LUT;
class L1MuGMTPhiLUT;

class L1MuGMTScales;
class L1MuTriggerScales;
class L1MuTriggerPtScale;
class L1MuGMTParameters;
class L1MuGMTChannelMask;

class L1CaloGeometry ;

//              ---------------------
//              -- Class Interface --
//              ---------------------


class L1MuGMTConfig {

  public:

    static const unsigned int MAXRPC = 8, MAXRPCbarrel = 4, MAXRPCendcap = 4, 
           MAXDTBX = 4, MAXCSC = 4 ;

    static const unsigned int MaxMuons = 4;

    /// constructor
    L1MuGMTConfig(const edm::ParameterSet& ps); 
     
    /// destructor 
    virtual ~L1MuGMTConfig();

    static edm::InputTag getDTInputTag()   { return m_DTInputTag; }
    static edm::InputTag getCSCInputTag()  { return m_CSCInputTag; }
    static edm::InputTag getRPCbInputTag() { return m_RPCbInputTag; }
    static edm::InputTag getRPCfInputTag() { return m_RPCfInputTag; }
    static edm::InputTag getMipIsoInputTag() { return m_MipIsoInputTag; }

    static bool Debug() { return m_debug; }
    static bool Debug(int level) { return ( m_debug && m_dbgLevel >= level ); }

    static void setDebugLevel(int level) { m_dbgLevel = level; }
    static int  getDebugLevel() { return m_dbgLevel; }

    static int getBxMin() { return m_BxMin; }
    static int getBxMax() { return m_BxMax; }

    static int getBxMinRo() { return m_BxMinRo; }
    static int getBxMaxRo() { return m_BxMaxRo; }

    static float getEtaWeightBarrel() { return m_EtaWeight_barrel; }
    static float getPhiWeightBarrel() { return m_PhiWeight_barrel; }
    static float getEtaPhiThresholdBarrel() { return m_EtaPhiThreshold_barrel; }
    static float getEtaWeightEndcap() { return m_EtaWeight_endcap; }
    static float getPhiWeightEndcap() { return m_PhiWeight_endcap; }
    static float getEtaPhiThresholdEndcap() { return m_EtaPhiThreshold_endcap; }
    static float getEtaWeightCOU() { return m_EtaWeight_COU; }
    static float getPhiWeightCOU() { return m_PhiWeight_COU; }
    static float getEtaPhiThresholdCOU() { return m_EtaPhiThreshold_COU; }

    static bool  getCaloTrigger() { return m_CaloTrigger; }
    static int   getIsolationCellSizeEta() { return m_IsolationCellSizeEta; }
    static int   getIsolationCellSizePhi() { return m_IsolationCellSizePhi; }

    /// require DT/CSC candidates to be confirmed by the RPC in the overlap region
    static bool  getDoOvlRpcAnd() { return m_DoOvlRpcAnd; }

    static bool getPropagatePhi() { return m_PropagatePhi; }
    
    static unsigned getVersionSortRankEtaQLUT() { return m_VersionSortRankEtaQLUT; }
    static unsigned getVersionLUTs() { return m_VersionLUTs; }

    // Register getters
    static L1MuGMTRegCDLConfig* getRegCDLConfig() { return m_RegCDLConfig; }
    static L1MuGMTRegMMConfigPhi* getRegMMConfigPhi() { return m_RegMMConfigPhi; }
    static L1MuGMTRegMMConfigEta* getRegMMConfigEta() { return m_RegMMConfigEta; }
    static L1MuGMTRegMMConfigPt* getRegMMConfigPt() { return m_RegMMConfigPt; }
    static L1MuGMTRegMMConfigCharge* getRegMMConfigCharge() { return m_RegMMConfigCharge; }
    static L1MuGMTRegMMConfigMIP* getRegMMConfigMIP() { return m_RegMMConfigMIP; }
    static L1MuGMTRegMMConfigISO* getRegMMConfigISO() { return m_RegMMConfigISO; }
    static L1MuGMTRegMMConfigSRK* getRegMMConfigSRK() { return m_RegMMConfigSRK; }
    static L1MuGMTRegSortRankOffset* getRegSortRankOffset() { return m_RegSortRankOffset; }

    // LUT getters
    static L1MuGMTEtaLUT* getEtaLUT() { return m_EtaLUT; }
    static L1MuGMTLFCOUDeltaEtaLUT* getLFCOUDeltaEtaLUT() { return m_LFCOUDeltaEtaLUT; }
    static L1MuGMTLFDeltaEtaLUT* getLFDeltaEtaLUT() { return m_LFDeltaEtaLUT; }
    static L1MuGMTLFDisableHotLUT* getLFDisableHotLUT() { return m_LFDisableHotLUT; }
    static L1MuGMTLFEtaConvLUT* getLFEtaConvLUT() { return m_LFEtaConvLUT; }
    static L1MuGMTLFMatchQualLUT* getLFMatchQualLUT() { return m_LFMatchQualLUT; }
    static L1MuGMTLFMergeRankCombineLUT* getLFMergeRankCombineLUT() { return m_LFMergeRankCombineLUT; }
    static L1MuGMTLFMergeRankEtaPhiLUT* getLFMergeRankEtaPhiLUT() { return m_LFMergeRankEtaPhiLUT; }
    static L1MuGMTLFMergeRankEtaQLUT* getLFMergeRankEtaQLUT() { return m_LFMergeRankEtaQLUT; }
    static L1MuGMTLFMergeRankPtQLUT* getLFMergeRankPtQLUT() { return m_LFMergeRankPtQLUT; }
    static L1MuGMTLFOvlEtaConvLUT* getLFOvlEtaConvLUT() { return m_LFOvlEtaConvLUT; }
    static L1MuGMTLFPhiProEtaConvLUT* getLFPhiProEtaConvLUT() { return m_LFPhiProEtaConvLUT; }
    static L1MuGMTLFPhiProLUT* getLFPhiProLUT() { return m_LFPhiProLUT; }
    static L1MuGMTLFPtMixLUT* getLFPtMixLUT() { return m_LFPtMixLUT; }
    static L1MuGMTLFSortRankCombineLUT* getLFSortRankCombineLUT() { return m_LFSortRankCombineLUT; }
    static L1MuGMTLFSortRankEtaPhiLUT* getLFSortRankEtaPhiLUT() { return m_LFSortRankEtaPhiLUT; }
    static L1MuGMTLFSortRankEtaQLUT* getLFSortRankEtaQLUT() { return m_LFSortRankEtaQLUT; }
    static L1MuGMTLFSortRankPtQLUT* getLFSortRankPtQLUT() { return m_LFSortRankPtQLUT; }
    static L1MuGMTMIAUEtaConvLUT* getMIAUEtaConvLUT() { return m_MIAUEtaConvLUT; }
    static L1MuGMTMIAUEtaProLUT* getMIAUEtaProLUT() { return m_MIAUEtaProLUT; }
    static L1MuGMTMIAUPhiPro1LUT* getMIAUPhiPro1LUT() { return m_MIAUPhiPro1LUT; }
    static L1MuGMTMIAUPhiPro2LUT* getMIAUPhiPro2LUT() { return m_MIAUPhiPro2LUT; }
    static L1MuGMTPhiLUT* getPhiLUT() { return m_PhiLUT; }

    void setGMTScales(const L1MuGMTScales* gmtscales) { m_GMTScales = gmtscales; }
    static const L1MuGMTScales* getGMTScales() { return m_GMTScales; }

    void setCaloGeom( const L1CaloGeometry* caloGeom ) { m_caloGeom = caloGeom ; }
    static const L1CaloGeometry* getCaloGeom() { return m_caloGeom ; }

    void setTriggerScales(const L1MuTriggerScales* trigscales) { m_TriggerScales = trigscales; }
    static const L1MuTriggerScales* getTriggerScales() { return m_TriggerScales; }

    void setTriggerPtScale(const L1MuTriggerPtScale* trigptscale) { m_TriggerPtScale = trigptscale; }
    static const L1MuTriggerPtScale* getTriggerPtScale() { return m_TriggerPtScale; }

    void setGMTParams(const L1MuGMTParameters* gmtparams) { m_GMTParams = gmtparams; }
    static const L1MuGMTParameters* getGMTParams() { return m_GMTParams; }
    
    void setGMTChanMask(const L1MuGMTChannelMask* gmtchanmask) { m_GMTChanMask = gmtchanmask; }
    static const L1MuGMTChannelMask* getGMTChanMask() { return m_GMTChanMask; }

    
    static const edm::ParameterSet* getParameterSet() { return m_ps; }
    
    void createLUTsRegs();
    void clearLUTsRegs();
    void dumpLUTs(std::string dir);
    void dumpRegs(std::string dir);

    void setDefaults();
  
  private:

    static const edm::ParameterSet* m_ps;
    static const L1MuGMTParameters* m_GMTParams;
    static const L1MuGMTChannelMask* m_GMTChanMask;

    static edm::InputTag m_DTInputTag;
    static edm::InputTag m_CSCInputTag;
    static edm::InputTag m_RPCbInputTag;
    static edm::InputTag m_RPCfInputTag;
    static edm::InputTag m_MipIsoInputTag;

    static bool m_debug;     // debug flag 
    static int  m_dbgLevel;  // debug level

    static int m_BxMin;
    static int m_BxMax;

    static int m_BxMinRo;
    static int m_BxMaxRo;

    static float m_EtaWeight_barrel;
    static float m_PhiWeight_barrel;
    static float m_EtaPhiThreshold_barrel;
    static float m_EtaWeight_endcap;
    static float m_PhiWeight_endcap;
    static float m_EtaPhiThreshold_endcap;
    static float m_EtaWeight_COU;
    static float m_PhiWeight_COU;
    static float m_EtaPhiThreshold_COU;
  
    static bool m_CaloTrigger;
    static int  m_IsolationCellSizeEta;
    static int  m_IsolationCellSizePhi;
    
    static bool m_DoOvlRpcAnd;

    static bool m_PropagatePhi;
    
    static unsigned m_VersionSortRankEtaQLUT;
    static unsigned m_VersionLUTs;
    
    // Register pointers
    static L1MuGMTRegCDLConfig* m_RegCDLConfig;
    static L1MuGMTRegMMConfigPhi* m_RegMMConfigPhi;
    static L1MuGMTRegMMConfigEta* m_RegMMConfigEta;
    static L1MuGMTRegMMConfigPt* m_RegMMConfigPt;
    static L1MuGMTRegMMConfigCharge* m_RegMMConfigCharge;
    static L1MuGMTRegMMConfigMIP* m_RegMMConfigMIP;
    static L1MuGMTRegMMConfigISO* m_RegMMConfigISO;
    static L1MuGMTRegMMConfigSRK* m_RegMMConfigSRK;
    static L1MuGMTRegSortRankOffset* m_RegSortRankOffset;

    // LUT pointers
    static L1MuGMTEtaLUT* m_EtaLUT;
    static L1MuGMTLFCOUDeltaEtaLUT* m_LFCOUDeltaEtaLUT;
    static L1MuGMTLFDeltaEtaLUT* m_LFDeltaEtaLUT;
    static L1MuGMTLFDisableHotLUT* m_LFDisableHotLUT;
    static L1MuGMTLFEtaConvLUT* m_LFEtaConvLUT;
    static L1MuGMTLFMatchQualLUT* m_LFMatchQualLUT;
    static L1MuGMTLFMergeRankCombineLUT* m_LFMergeRankCombineLUT;
    static L1MuGMTLFMergeRankEtaPhiLUT* m_LFMergeRankEtaPhiLUT;
    static L1MuGMTLFMergeRankEtaQLUT* m_LFMergeRankEtaQLUT;
    static L1MuGMTLFMergeRankPtQLUT* m_LFMergeRankPtQLUT;
    static L1MuGMTLFOvlEtaConvLUT* m_LFOvlEtaConvLUT;
    static L1MuGMTLFPhiProEtaConvLUT* m_LFPhiProEtaConvLUT;
    static L1MuGMTLFPhiProLUT* m_LFPhiProLUT;
    static L1MuGMTLFPtMixLUT* m_LFPtMixLUT;
    static L1MuGMTLFSortRankCombineLUT* m_LFSortRankCombineLUT;
    static L1MuGMTLFSortRankEtaPhiLUT* m_LFSortRankEtaPhiLUT;
    static L1MuGMTLFSortRankEtaQLUT* m_LFSortRankEtaQLUT;
    static L1MuGMTLFSortRankPtQLUT* m_LFSortRankPtQLUT;
    static L1MuGMTMIAUEtaConvLUT* m_MIAUEtaConvLUT;
    static L1MuGMTMIAUEtaProLUT* m_MIAUEtaProLUT;
    static L1MuGMTMIAUPhiPro1LUT* m_MIAUPhiPro1LUT;
    static L1MuGMTMIAUPhiPro2LUT* m_MIAUPhiPro2LUT;
    static L1MuGMTPhiLUT* m_PhiLUT;

    // scales pointers
    static const L1MuGMTScales* m_GMTScales;
    static const L1MuTriggerScales* m_TriggerScales;
    static const L1MuTriggerPtScale* m_TriggerPtScale;

    static const L1CaloGeometry* m_caloGeom ;
};

#endif


