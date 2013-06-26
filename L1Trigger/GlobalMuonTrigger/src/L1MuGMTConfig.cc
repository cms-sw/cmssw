//-------------------------------------------------
//
//   Class: L1MuGMTConfig
//
//   Description: Configuration parameters for L1GlobalMuonTrigger
//
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

//-----------------------
// This Class's Header --
//-----------------------

#include "L1Trigger/GlobalMuonTrigger/src/L1MuGMTConfig.h"

//---------------
// C++ Headers --
//---------------

#include <iostream>
#include <fstream>
#include <string>

//-------------------------------
// Collaborating Class Headers --
//-------------------------------

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "L1Trigger/GlobalMuonTrigger/src/L1MuGMTReg.h"
#include "L1Trigger/GlobalMuonTrigger/src/L1MuGMTEtaLUT.h"
#include "L1Trigger/GlobalMuonTrigger/src/L1MuGMTLFCOUDeltaEtaLUT.h"
#include "L1Trigger/GlobalMuonTrigger/src/L1MuGMTLFDeltaEtaLUT.h"
#include "L1Trigger/GlobalMuonTrigger/src/L1MuGMTLFDisableHotLUT.h"
#include "L1Trigger/GlobalMuonTrigger/src/L1MuGMTLFEtaConvLUT.h"
#include "L1Trigger/GlobalMuonTrigger/src/L1MuGMTLFMatchQualLUT.h"
#include "L1Trigger/GlobalMuonTrigger/src/L1MuGMTLFMergeRankCombineLUT.h"
#include "L1Trigger/GlobalMuonTrigger/src/L1MuGMTLFMergeRankEtaPhiLUT.h"
#include "L1Trigger/GlobalMuonTrigger/src/L1MuGMTLFMergeRankEtaQLUT.h"
#include "L1Trigger/GlobalMuonTrigger/src/L1MuGMTLFMergeRankPtQLUT.h"
#include "L1Trigger/GlobalMuonTrigger/src/L1MuGMTLFOvlEtaConvLUT.h"
#include "L1Trigger/GlobalMuonTrigger/src/L1MuGMTLFPhiProEtaConvLUT.h"
#include "L1Trigger/GlobalMuonTrigger/src/L1MuGMTLFPhiProLUT.h"
#include "L1Trigger/GlobalMuonTrigger/src/L1MuGMTLFPtMixLUT.h"
#include "L1Trigger/GlobalMuonTrigger/src/L1MuGMTLFSortRankCombineLUT.h"
#include "L1Trigger/GlobalMuonTrigger/src/L1MuGMTLFSortRankEtaPhiLUT.h"
#include "L1Trigger/GlobalMuonTrigger/src/L1MuGMTLFSortRankEtaQLUT.h"
#include "L1Trigger/GlobalMuonTrigger/src/L1MuGMTLFSortRankPtQLUT.h"
#include "L1Trigger/GlobalMuonTrigger/src/L1MuGMTMIAUEtaConvLUT.h"
#include "L1Trigger/GlobalMuonTrigger/src/L1MuGMTMIAUEtaProLUT.h"
#include "L1Trigger/GlobalMuonTrigger/src/L1MuGMTMIAUPhiPro1LUT.h"
#include "L1Trigger/GlobalMuonTrigger/src/L1MuGMTMIAUPhiPro2LUT.h"
#include "L1Trigger/GlobalMuonTrigger/src/L1MuGMTPhiLUT.h"

#include "CondFormats/L1TObjects/interface/L1MuGMTScales.h"
#include "CondFormats/L1TObjects/interface/L1MuGMTParameters.h"
#include "CondFormats/L1TObjects/interface/L1MuGMTChannelMask.h"

#include "CondFormats/L1TObjects/interface/L1CaloGeometry.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

// --------------------------------
//       class L1MuGMTConfig
//---------------------------------
using namespace std;

//----------------
// Constructors --
//----------------

L1MuGMTConfig::L1MuGMTConfig(const edm::ParameterSet& ps) {

  m_ps = &ps;

  m_DTInputTag   = m_ps->getParameter<edm::InputTag>("DTCandidates");
  m_CSCInputTag  = m_ps->getParameter<edm::InputTag>("CSCCandidates");
  m_RPCbInputTag = m_ps->getParameter<edm::InputTag>("RPCbCandidates");
  m_RPCfInputTag = m_ps->getParameter<edm::InputTag>("RPCfCandidates");
  m_MipIsoInputTag = m_ps->getParameter<edm::InputTag>("MipIsoData");

  m_debug = true;
  m_dbgLevel = m_ps->getUntrackedParameter<int>("Debug",0);

  // set min and max bunch crossing
  m_BxMin = m_ps->getParameter<int>("BX_min");
  m_BxMax = m_ps->getParameter<int>("BX_max");

  // set min and max bunch crossing for the readout
  m_BxMinRo = m_ps->getParameter<int>("BX_min_readout");
  m_BxMaxRo = m_ps->getParameter<int>("BX_max_readout");

}
  
//--------------
// Destructor --
//--------------
L1MuGMTConfig::~L1MuGMTConfig() {}


//--------------
// Operations --
//--------------


void L1MuGMTConfig::setDefaults() {
  
  // set weights for eta and phi
  m_EtaWeight_barrel = m_GMTParams->getEtaWeight_barrel();
  m_PhiWeight_barrel = m_GMTParams->getPhiWeight_barrel();
  m_EtaPhiThreshold_barrel = m_GMTParams->getEtaPhiThreshold_barrel();
  
  m_EtaWeight_endcap = m_GMTParams->getEtaWeight_endcap();
  m_PhiWeight_endcap = m_GMTParams->getPhiWeight_endcap();
  m_EtaPhiThreshold_endcap = m_GMTParams->getEtaPhiThreshold_endcap();
  
  m_EtaWeight_COU = m_GMTParams->getEtaWeight_COU();
  m_PhiWeight_COU = m_GMTParams->getPhiWeight_COU();
  m_EtaPhiThreshold_COU = m_GMTParams->getEtaPhiThreshold_COU();
  
  m_CaloTrigger = m_GMTParams->getCaloTrigger();
  m_IsolationCellSizeEta = m_GMTParams->getIsolationCellSizeEta();
  m_IsolationCellSizePhi = m_GMTParams->getIsolationCellSizePhi();
  
  m_DoOvlRpcAnd = m_GMTParams->getDoOvlRpcAnd();

  m_PropagatePhi = m_GMTParams->getPropagatePhi();
  
  m_VersionSortRankEtaQLUT = m_GMTParams->getVersionSortRankEtaQLUT();
  m_VersionLUTs = m_GMTParams->getVersionLUTs();

  if ( Debug(1) ) {
    stringstream stdss;
    stdss
        << endl
        << "*******************************************" << endl
        << "**** L1 Global Muon Trigger settings : ****" << endl
        << "*******************************************" << endl
        << endl

        << "L1 Global Muon Trigger : DTCandidates : " << m_DTInputTag << endl
        << "L1 Global Muon Trigger : CSCCandidates : " << m_CSCInputTag << endl
        << "L1 Global Muon Trigger : RPCbCandidates : " << m_RPCbInputTag << endl
        << "L1 Global Muon Trigger : RPCfCandidates : " << m_RPCfInputTag << endl
        << "L1 Global Muon Trigger : debug level : " << m_dbgLevel << endl
        << "L1 Global Muon Trigger : minimal bunch-crossing : " << m_BxMin << endl
        << "L1 Global Muon Trigger : maximal bunch-crossing : " << m_BxMax << endl
        << "L1 Global Muon Trigger : barrel eta weight : " << m_EtaWeight_barrel << endl
        << "L1 Global Muon Trigger : barrel phi weight : " << m_PhiWeight_barrel << endl
        << "L1 Global Muon Trigger : barrel eta-phi threshold : " << m_EtaPhiThreshold_barrel << endl
        << "L1 Global Muon Trigger : endcap eta weight : " << m_EtaWeight_endcap << endl
        << "L1 Global Muon Trigger : endcap phi weight : " << m_PhiWeight_endcap << endl
        << "L1 Global Muon Trigger : endcap eta-phi threshold : " << m_EtaPhiThreshold_endcap << endl
        << "L1 Global Muon Trigger : cancel out unit eta weight : " << m_EtaWeight_COU << endl
        << "L1 Global Muon Trigger : cancel out unit phi weight : " << m_PhiWeight_COU << endl
        << "L1 Global Muon Trigger : cancel out unit eta-phi threshold : " << m_EtaPhiThreshold_COU << endl
        << "L1 Global Muon Trigger : calorimeter trigger : " << m_CaloTrigger << endl
        << "L1 Global Muon Trigger : muon isolation cell size (eta) : " << m_IsolationCellSizeEta << endl
        << "L1 Global Muon Trigger : muon isolation cell size (phi) : " << m_IsolationCellSizePhi << endl
        << "L1 Global Muon Trigger : require confirmation by RPC in overlap region : " << m_DoOvlRpcAnd << endl
        << "L1 Global Muon Trigger : propagate phi to vertex : " << m_PropagatePhi << endl
        << "L1 Global Muon Trigger : version of low quality assignment LUT : " << m_VersionSortRankEtaQLUT << endl
        << "L1 Global Muon Trigger : general LUTs version : " << m_VersionLUTs << endl;
    edm::LogVerbatim("GMT_Config_info") << stdss.str();
  }
}

void L1MuGMTConfig::createLUTsRegs() {

  // create Registers
  m_RegCDLConfig = new L1MuGMTRegCDLConfig();
  m_RegMMConfigPhi = new L1MuGMTRegMMConfigPhi();
  m_RegMMConfigEta = new L1MuGMTRegMMConfigEta();
  m_RegMMConfigPt = new L1MuGMTRegMMConfigPt();
  m_RegMMConfigCharge = new L1MuGMTRegMMConfigCharge();
  m_RegMMConfigMIP = new L1MuGMTRegMMConfigMIP();
  m_RegMMConfigISO = new L1MuGMTRegMMConfigISO();
  m_RegMMConfigSRK = new L1MuGMTRegMMConfigSRK();
  m_RegSortRankOffset = new L1MuGMTRegSortRankOffset();

  // create LUTs
  m_EtaLUT = new L1MuGMTEtaLUT();
  m_LFCOUDeltaEtaLUT = new L1MuGMTLFCOUDeltaEtaLUT();
  m_LFDeltaEtaLUT = new L1MuGMTLFDeltaEtaLUT();
  m_LFDisableHotLUT = new L1MuGMTLFDisableHotLUT();
  m_LFEtaConvLUT = new L1MuGMTLFEtaConvLUT();
  m_LFMatchQualLUT = new L1MuGMTLFMatchQualLUT();
  m_LFMergeRankCombineLUT = new L1MuGMTLFMergeRankCombineLUT();
  m_LFMergeRankEtaPhiLUT = new L1MuGMTLFMergeRankEtaPhiLUT();
  m_LFMergeRankEtaQLUT = new L1MuGMTLFMergeRankEtaQLUT();
  m_LFMergeRankPtQLUT = new L1MuGMTLFMergeRankPtQLUT();
  m_LFOvlEtaConvLUT = new L1MuGMTLFOvlEtaConvLUT();
  m_LFPhiProEtaConvLUT = new L1MuGMTLFPhiProEtaConvLUT();
  m_LFPhiProLUT = new L1MuGMTLFPhiProLUT();
  m_LFPtMixLUT = new L1MuGMTLFPtMixLUT();
  m_LFSortRankCombineLUT = new L1MuGMTLFSortRankCombineLUT();
  m_LFSortRankEtaPhiLUT = new L1MuGMTLFSortRankEtaPhiLUT();
  m_LFSortRankEtaQLUT = new L1MuGMTLFSortRankEtaQLUT();
  m_LFSortRankPtQLUT = new L1MuGMTLFSortRankPtQLUT();
  m_MIAUEtaConvLUT = new L1MuGMTMIAUEtaConvLUT();
  m_MIAUEtaProLUT = new L1MuGMTMIAUEtaProLUT();
  m_MIAUPhiPro1LUT = new L1MuGMTMIAUPhiPro1LUT();
  m_MIAUPhiPro2LUT = new L1MuGMTMIAUPhiPro2LUT();
  m_PhiLUT = new L1MuGMTPhiLUT();

}

void L1MuGMTConfig::clearLUTsRegs() {
  // delete Registers
  delete m_RegCDLConfig;
  delete m_RegMMConfigPhi;
  delete m_RegMMConfigEta;
  delete m_RegMMConfigPt;
  delete m_RegMMConfigCharge;
  delete m_RegMMConfigMIP;
  delete m_RegMMConfigISO;
  delete m_RegMMConfigSRK;
  delete m_RegSortRankOffset;

  // delete LUTs
  delete m_EtaLUT;
  delete m_LFCOUDeltaEtaLUT;
  delete m_LFDeltaEtaLUT;
  delete m_LFDisableHotLUT;
  delete m_LFEtaConvLUT;
  delete m_LFMatchQualLUT;
  delete m_LFMergeRankCombineLUT;
  delete m_LFMergeRankEtaPhiLUT;
  delete m_LFMergeRankEtaQLUT;
  delete m_LFMergeRankPtQLUT;
  delete m_LFOvlEtaConvLUT;
  delete m_LFPhiProEtaConvLUT;
  delete m_LFPhiProLUT;
  delete m_LFPtMixLUT;
  delete m_LFSortRankCombineLUT;
  delete m_LFSortRankEtaPhiLUT;
  delete m_LFSortRankEtaQLUT;
  delete m_LFSortRankPtQLUT;
  delete m_MIAUEtaConvLUT;
  delete m_MIAUEtaProLUT;
  delete m_MIAUPhiPro1LUT;
  delete m_MIAUPhiPro2LUT;
  delete m_PhiLUT;
}

void L1MuGMTConfig::dumpLUTs(std::string dir) {
  std::vector<L1MuGMTLUT*> theLUTs;

  theLUTs.push_back( m_LFSortRankEtaQLUT );  
  theLUTs.push_back( m_LFSortRankPtQLUT );   
  theLUTs.push_back( m_LFSortRankEtaPhiLUT );
  theLUTs.push_back( m_LFSortRankCombineLUT );

  theLUTs.push_back( m_LFDisableHotLUT );

  theLUTs.push_back( m_LFMergeRankEtaQLUT );  
  theLUTs.push_back( m_LFMergeRankPtQLUT );   
  theLUTs.push_back( m_LFMergeRankEtaPhiLUT );
  theLUTs.push_back( m_LFMergeRankCombineLUT );

  theLUTs.push_back( m_LFDeltaEtaLUT );
  theLUTs.push_back( m_LFMatchQualLUT );
  theLUTs.push_back( m_LFOvlEtaConvLUT );
  theLUTs.push_back( m_LFCOUDeltaEtaLUT );

  theLUTs.push_back( m_LFEtaConvLUT );

  theLUTs.push_back( m_LFPtMixLUT );
  theLUTs.push_back( m_LFPhiProLUT );
  theLUTs.push_back( m_LFPhiProEtaConvLUT );

  theLUTs.push_back( m_MIAUEtaConvLUT );
  theLUTs.push_back( m_MIAUPhiPro1LUT );
  theLUTs.push_back( m_MIAUPhiPro2LUT );
  theLUTs.push_back( m_MIAUEtaProLUT );

  std::vector<L1MuGMTLUT*>::iterator it = theLUTs.begin();
  for (;it != theLUTs.end(); it++) {
    edm::LogVerbatim("GMT_LUTGen_info")
     << "**** Generating " << (*it)->Name() << " LUT ****" << endl
     << "saving" << endl;
    std::string fn = dir + "/" + (*it)->Name() + ".lut";
    (*it)->Save(fn.c_str());    
  }

  edm::LogVerbatim("GMT_LUTGen_info") 
      << "Successfully created all GMT look-up tables in directory './" << dir << "'" << endl << endl;

}

void L1MuGMTConfig::dumpRegs(std::string dir) {
  std::vector<L1MuGMTReg*> theRegs;

  theRegs.push_back( m_RegCDLConfig );
  theRegs.push_back( m_RegMMConfigPhi );
  theRegs.push_back( m_RegMMConfigEta );
  theRegs.push_back( m_RegMMConfigPt );
  theRegs.push_back( m_RegMMConfigCharge );
  theRegs.push_back( m_RegMMConfigSRK );
  theRegs.push_back( m_RegMMConfigMIP );
  theRegs.push_back( m_RegMMConfigISO );
  theRegs.push_back( m_RegSortRankOffset );


  ofstream of( (dir + "/LogicFPGARegs.cfg").c_str() );

  std::vector<L1MuGMTReg*>::iterator it = theRegs.begin();
  for (;it != theRegs.end(); it++) {

    for (unsigned int i=0; i<(*it)->getNumberOfInstances(); i++)
      of << (*it)->getName() << "[" << i << "] = " << (*it)->getValue(i) << endl;

  }

}

// static data members

const edm::ParameterSet* L1MuGMTConfig::m_ps=0;

edm::InputTag L1MuGMTConfig::m_DTInputTag = edm::InputTag();
edm::InputTag L1MuGMTConfig::m_CSCInputTag = edm::InputTag();
edm::InputTag L1MuGMTConfig::m_RPCbInputTag = edm::InputTag();
edm::InputTag L1MuGMTConfig::m_RPCfInputTag = edm::InputTag();
edm::InputTag L1MuGMTConfig::m_MipIsoInputTag = edm::InputTag();
int   L1MuGMTConfig::m_dbgLevel = 0;
bool  L1MuGMTConfig::m_debug = false;
int   L1MuGMTConfig::m_BxMin = -4;
int   L1MuGMTConfig::m_BxMax = 4;
int   L1MuGMTConfig::m_BxMinRo = -2;
int   L1MuGMTConfig::m_BxMaxRo = 2;
float L1MuGMTConfig::m_EtaWeight_barrel = 0.028;
float L1MuGMTConfig::m_PhiWeight_barrel = 1.0;
float L1MuGMTConfig::m_EtaPhiThreshold_barrel = 0.062;
float L1MuGMTConfig::m_EtaWeight_endcap = 0.13;
float L1MuGMTConfig::m_PhiWeight_endcap = 1.0;
float L1MuGMTConfig::m_EtaPhiThreshold_endcap = 0.062;
float L1MuGMTConfig::m_EtaWeight_COU = 0.316;
float L1MuGMTConfig::m_PhiWeight_COU = 1.0;
float L1MuGMTConfig::m_EtaPhiThreshold_COU = 0.127;
bool  L1MuGMTConfig::m_CaloTrigger = true;
int   L1MuGMTConfig::m_IsolationCellSizeEta = 2;
int   L1MuGMTConfig::m_IsolationCellSizePhi = 2;
bool  L1MuGMTConfig::m_DoOvlRpcAnd = false;

bool  L1MuGMTConfig::m_PropagatePhi = false;
unsigned L1MuGMTConfig::m_VersionSortRankEtaQLUT = 2;
unsigned L1MuGMTConfig::m_VersionLUTs = 0;

L1MuGMTRegCDLConfig* L1MuGMTConfig::m_RegCDLConfig=0;
L1MuGMTRegMMConfigPhi* L1MuGMTConfig::m_RegMMConfigPhi=0;
L1MuGMTRegMMConfigEta* L1MuGMTConfig::m_RegMMConfigEta=0;
L1MuGMTRegMMConfigPt* L1MuGMTConfig::m_RegMMConfigPt=0;
L1MuGMTRegMMConfigCharge* L1MuGMTConfig::m_RegMMConfigCharge=0;
L1MuGMTRegMMConfigMIP* L1MuGMTConfig::m_RegMMConfigMIP=0;
L1MuGMTRegMMConfigISO* L1MuGMTConfig::m_RegMMConfigISO=0;
L1MuGMTRegMMConfigSRK* L1MuGMTConfig::m_RegMMConfigSRK=0;
L1MuGMTRegSortRankOffset* L1MuGMTConfig::m_RegSortRankOffset=0;

L1MuGMTEtaLUT* L1MuGMTConfig::m_EtaLUT=0;
L1MuGMTLFCOUDeltaEtaLUT* L1MuGMTConfig::m_LFCOUDeltaEtaLUT=0;
L1MuGMTLFDeltaEtaLUT* L1MuGMTConfig::m_LFDeltaEtaLUT=0;
L1MuGMTLFDisableHotLUT* L1MuGMTConfig::m_LFDisableHotLUT=0;
L1MuGMTLFEtaConvLUT* L1MuGMTConfig::m_LFEtaConvLUT=0;
L1MuGMTLFMatchQualLUT* L1MuGMTConfig::m_LFMatchQualLUT=0;
L1MuGMTLFMergeRankCombineLUT* L1MuGMTConfig::m_LFMergeRankCombineLUT=0;
L1MuGMTLFMergeRankEtaPhiLUT* L1MuGMTConfig::m_LFMergeRankEtaPhiLUT=0;
L1MuGMTLFMergeRankEtaQLUT* L1MuGMTConfig::m_LFMergeRankEtaQLUT=0;
L1MuGMTLFMergeRankPtQLUT* L1MuGMTConfig::m_LFMergeRankPtQLUT=0;
L1MuGMTLFOvlEtaConvLUT* L1MuGMTConfig::m_LFOvlEtaConvLUT=0;
L1MuGMTLFPhiProEtaConvLUT* L1MuGMTConfig::m_LFPhiProEtaConvLUT=0;
L1MuGMTLFPhiProLUT* L1MuGMTConfig::m_LFPhiProLUT=0;
L1MuGMTLFPtMixLUT* L1MuGMTConfig::m_LFPtMixLUT=0;
L1MuGMTLFSortRankCombineLUT* L1MuGMTConfig::m_LFSortRankCombineLUT=0;
L1MuGMTLFSortRankEtaPhiLUT* L1MuGMTConfig::m_LFSortRankEtaPhiLUT=0;
L1MuGMTLFSortRankEtaQLUT* L1MuGMTConfig::m_LFSortRankEtaQLUT=0;
L1MuGMTLFSortRankPtQLUT* L1MuGMTConfig::m_LFSortRankPtQLUT=0;
L1MuGMTMIAUEtaConvLUT* L1MuGMTConfig::m_MIAUEtaConvLUT=0;
L1MuGMTMIAUEtaProLUT* L1MuGMTConfig::m_MIAUEtaProLUT=0;
L1MuGMTMIAUPhiPro1LUT* L1MuGMTConfig::m_MIAUPhiPro1LUT=0;
L1MuGMTMIAUPhiPro2LUT* L1MuGMTConfig::m_MIAUPhiPro2LUT=0;
L1MuGMTPhiLUT* L1MuGMTConfig::m_PhiLUT=0;

const L1MuGMTScales* L1MuGMTConfig::m_GMTScales=0;
const L1MuTriggerScales* L1MuGMTConfig::m_TriggerScales=0;
const L1MuTriggerPtScale* L1MuGMTConfig::m_TriggerPtScale=0;
const L1MuGMTParameters* L1MuGMTConfig::m_GMTParams=0;
const L1MuGMTChannelMask* L1MuGMTConfig::m_GMTChanMask=0;

const L1CaloGeometry* L1MuGMTConfig::m_caloGeom = 0 ;
