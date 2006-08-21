//-------------------------------------------------
//
//   Class: L1MuGMTConfig
//
//   Description: Configuration parameters for L1GlobalMuonTrigger
//
//
//   $Date: 2006/07/07 16:57:06 $
//   $Revision: 1.2 $
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
#include <sys/stat.h>
#include <sys/types.h>

//-------------------------------
// Collaborating Class Headers --
//-------------------------------

#include "FWCore/ParameterSet/interface/ParameterSet.h"
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
  setDefaults(); 

  bool writeLUTsAndRegs = m_ps->getUntrackedParameter<bool>("WriteLUTsAndRegs",false);
  if(writeLUTsAndRegs) {
    string dir = "gmtconfig";
  
    mkdir(dir.c_str(), S_ISUID|S_ISGID|S_ISVTX|S_IRUSR|S_IWUSR|S_IXUSR);

    dumpLUTs(dir);
    dumpRegs(dir);

  }

}
  
//--------------
// Destructor --
//--------------
L1MuGMTConfig::~L1MuGMTConfig() {}


//--------------
// Operations --
//--------------


void L1MuGMTConfig::setDefaults() {
  
  m_debug = true;
  m_dbgLevel = m_ps->getUntrackedParameter<int>("Debug",0);

  // set min and max bunch crossing
  m_BxMin = m_ps->getParameter<int>("BX_min");
  m_BxMax = m_ps->getParameter<int>("BX_max");

  // set min and max bunch crossing for the readout
  m_BxMinRo = m_ps->getParameter<int>("BX_min_readout");
  m_BxMaxRo = m_ps->getParameter<int>("BX_max_readout");

  // set weights for eta and phi
  m_EtaWeight_barrel = m_ps->getParameter<double>("EtaWeight_barrel");
  m_PhiWeight_barrel = m_ps->getParameter<double>("PhiWeight_barrel");
  m_EtaPhiThreshold_barrel = m_ps->getParameter<double>("EtaPhiThreshold_barrel");

  m_EtaWeight_endcap = m_ps->getParameter<double>("EtaWeight_endcap");
  m_PhiWeight_endcap = m_ps->getParameter<double>("PhiWeight_endcap");
  m_EtaPhiThreshold_endcap = m_ps->getParameter<double>("EtaPhiThreshold_endcap");

  m_EtaWeight_COU = m_ps->getParameter<double>("EtaWeight_COU");
  m_PhiWeight_COU = m_ps->getParameter<double>("PhiWeight_COU");
  m_EtaPhiThreshold_COU = m_ps->getParameter<double>("EtaPhiThreshold_COU");

  m_CaloTrigger = m_ps->getParameter<bool>("CaloTrigger");
  m_IsolationCellSizeEta = m_ps->getParameter<int>("IsolationCellSizeEta");
  m_IsolationCellSizePhi = m_ps->getParameter<int>("IsolationCellSizePhi");  

  m_DoOvlRpcAnd = m_ps->getParameter<bool>("DoOvlRpcAnd");  

  if ( Debug(1) ) {
    edm::LogVerbatim("GMT_Config_info")
        << endl
        << "*******************************************" << endl
        << "**** L1 Global Muon Trigger settings : ****" << endl
        << "*******************************************" << endl
        << endl

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
        << "L1 Global Muon Trigger : require confirmation by RPC in overlap region : " << m_DoOvlRpcAnd << endl;
  }

  m_PropagatePhi = m_ps->getParameter<bool>("PropagatePhi");  

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

void L1MuGMTConfig::dumpLUTs(string dir) {
  vector<L1MuGMTLUT*> theLUTs;

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

  vector<L1MuGMTLUT*>::iterator it = theLUTs.begin();
  for (;it != theLUTs.end(); it++) {
    edm::LogVerbatim("GMT_LUTGen_info")
     << "**** Generating " << (*it)->Name() << " LUT ****" << endl
     << "saving" << endl;
    string fn = dir + "/" + (*it)->Name() + ".lut";
    (*it)->Save(fn.c_str());    
  }

  edm::LogVerbatim("GMT_LUTGen_info") 
      << "Successfully created all GMT look-up tables in directory './" << dir << "'" << endl << endl;

}

void L1MuGMTConfig::dumpRegs(string dir) {
  vector<L1MuGMTReg*> theRegs;

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

  vector<L1MuGMTReg*>::iterator it = theRegs.begin();
  for (;it != theRegs.end(); it++) {

    for (unsigned int i=0; i<(*it)->getNumberOfInstances(); i++)
      of << (*it)->getName() << "[" << i << "] = " << (*it)->getValue(i) << endl;

  }

}

// static data members

const edm::ParameterSet* L1MuGMTConfig::m_ps=0;

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
