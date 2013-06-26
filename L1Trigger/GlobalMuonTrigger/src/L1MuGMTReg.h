//-------------------------------------------------
//
//   
/**  \class L1MuGMTReg
 *
 *   Description: A 16bit VME register
 *
 *   Used to configure the GMT. The register class represents 
 *   multiple instances of the register in the hardware (by default 2)
*/ 
//
//   $Date: 2009/12/18 21:21:57 $
//   $Revision: 1.5 $
//
//   Author :
//   H. Sakulin            HEPHY Vienna
//
//   Migrated to CMSSW:
//   I. Mikulec
//
//--------------------------------------------------
#ifndef L1TriggerGlobalMuonTrigger_L1MuGMTReg_h
#define L1TriggerGlobalMuonTrigger_L1MuGMTReg_h

//---------------
// C++ Headers --
//---------------

#include <string>
#include <vector>

//-------------------------------
// Collaborating Class Headers --
//-------------------------------

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "L1Trigger/GlobalMuonTrigger/src/L1MuGMTConfig.h"
#include "CondFormats/L1TObjects/interface/L1MuGMTParameters.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

//              ---------------------
//              -- Class Interface --
//              ---------------------


class L1MuGMTReg {

 public:  
  /// default constructor 
  L1MuGMTReg(int ninst = 2) : m_value(ninst, 0) { };

  /// destructor
  virtual ~L1MuGMTReg() {};

  /// get Value
  unsigned getValue(int idx) { return m_value[idx]; };

  /// get number on instances
  unsigned getNumberOfInstances() { return m_value.size(); }
  
  /// get Name
  virtual std::string getName() =0;
  
 protected:
  std::vector<unsigned> m_value;
};

//
/// \class L1MuGMTRegMMConfig 
//
/// GMT Register that implements enum of merge methods
//

class L1MuGMTRegMMConfig : public L1MuGMTReg {

 public: 
  enum MergeMethods { takeDTCSC, takeRPC, byRank, byMinPt, byCombi, Special };

  //// constructor
  L1MuGMTRegMMConfig(const std::string& param, MergeMethods def_brl, MergeMethods def_fwd) :
    m_param(param) { 
    m_default[0] = def_brl;
    m_default[1] = def_fwd;
    setMergeMethod(); 
  };

  //// destructor
  virtual ~L1MuGMTRegMMConfig() {};

  //// get Name
  virtual std::string getName() { return "MMConfig_" + m_param; };

  //// read the merge method from .orcarc
  void setMergeMethod() {
    static MergeMethods avlMethods[6] = { takeDTCSC, takeRPC, byRank, byMinPt, byCombi, Special };
    std::string mn[6] = { "takeDT", "takeRPC", "byRank", "byMinPt", "byCombi", "Special" };
    
    MergeMethods mm;
    std::string mm_str;

    mm = m_default[0];
    if(m_param=="Phi")         mm_str = L1MuGMTConfig::getGMTParams()->getMergeMethodPhiBrl();
    else if(m_param=="Eta")    mm_str = L1MuGMTConfig::getGMTParams()->getMergeMethodEtaBrl();
    else if(m_param=="Pt")     mm_str = L1MuGMTConfig::getGMTParams()->getMergeMethodPtBrl();
    else if(m_param=="Charge") mm_str = L1MuGMTConfig::getGMTParams()->getMergeMethodChargeBrl();
    for(int ii=0; ii<6; ii++) if(mm_str == mn[ii]) {mm = avlMethods[ii]; break;}
    m_value[0] = 1 << (5-(int) MergeMethods(mm));
    if ( L1MuGMTConfig::Debug(1) ) edm::LogVerbatim("GMT_Register_Info") << " "
                      << "MergeMethod" << m_param << "Brl"
                      << " is " << mm
                      << "( value " << m_value[0] << " )";
    
    mm = m_default[1];
    mn[0] = "takeCSC";
    if(m_param=="Phi")         mm_str = L1MuGMTConfig::getGMTParams()->getMergeMethodPhiFwd();
    else if(m_param=="Eta")    mm_str = L1MuGMTConfig::getGMTParams()->getMergeMethodEtaFwd();
    else if(m_param=="Pt")     mm_str = L1MuGMTConfig::getGMTParams()->getMergeMethodPtFwd();
    else if(m_param=="Charge") mm_str = L1MuGMTConfig::getGMTParams()->getMergeMethodChargeFwd();
    for(int ii=0; ii<6; ii++) if(mm_str == mn[ii]) {mm = avlMethods[ii]; break;}
    m_value[1] = 1 << (5-(int) MergeMethods(mm));
    if ( L1MuGMTConfig::Debug(1) ) edm::LogVerbatim("GMT_Register_Info") << " "
                      << "MergeMethod" << m_param << "Fwd"
                      << " is " << mm
                      << "( value " << m_value[1] << " )";
    
    
  };

 protected:
  std::string m_param;
  MergeMethods m_default[2];
} ;

//
/// \class L1MuGMTRegMMConfigPhi
//
/// GMT Merge Method Config Register Phi 
 
class L1MuGMTRegMMConfigPhi : public L1MuGMTRegMMConfig {
public:
  L1MuGMTRegMMConfigPhi() : L1MuGMTRegMMConfig("Phi", L1MuGMTRegMMConfig::takeDTCSC, L1MuGMTRegMMConfig::takeDTCSC ) {};
};

//
/// \class L1MuGMTRegMMConfigEta
//
/// GMT Merge Method Config Register Eta
 
class L1MuGMTRegMMConfigEta : public L1MuGMTRegMMConfig {
 public:
  L1MuGMTRegMMConfigEta() : L1MuGMTRegMMConfig("Eta", L1MuGMTRegMMConfig::Special, L1MuGMTRegMMConfig::Special ) {};
};

//
/// \class L1MuGMTRegMMConfigPt
//
/// GMT Merge Method Config Register Pt
 
class L1MuGMTRegMMConfigPt : public L1MuGMTRegMMConfig {
 public:
  L1MuGMTRegMMConfigPt() : L1MuGMTRegMMConfig("Pt", L1MuGMTRegMMConfig::byMinPt, L1MuGMTRegMMConfig::byMinPt ) {};
};

//
/// \class L1MuGMTRegMMConfigCharge
//
/// GMT Merge Method Config Register Charge
 
class L1MuGMTRegMMConfigCharge : public L1MuGMTRegMMConfig {
 public:
  L1MuGMTRegMMConfigCharge() : L1MuGMTRegMMConfig("Charge", L1MuGMTRegMMConfig::takeDTCSC, L1MuGMTRegMMConfig::takeDTCSC ) {};
};



//
/// \class L1MuGMTRegMMConfigMIPISO 
//
/// GMT Register that implements additional AND/OR flag
//

class L1MuGMTRegMMConfigMIPISO : public L1MuGMTRegMMConfig {

 public: 
  //// constructor
  L1MuGMTRegMMConfigMIPISO(const std::string& param, MergeMethods def_brl, MergeMethods def_fwd, bool def_and_brl, bool def_and_fwd) :
    L1MuGMTRegMMConfig(param, def_brl, def_fwd) { 

    bool doAND = false;
    
    if(m_param=="MIP")      doAND = L1MuGMTConfig::getGMTParams()->getMergeMethodMIPSpecialUseANDBrl();
    else if(m_param=="ISO") doAND = L1MuGMTConfig::getGMTParams()->getMergeMethodISOSpecialUseANDBrl();
    if(doAND) m_value[0] |= 64;
    if ( L1MuGMTConfig::Debug(1) ) edm::LogVerbatim("GMT_Register_Info") << " "
                      << "MergeMethod" << m_param  << "SpecialUseANDBrl"
                      << " is " << doAND;

    if(m_param=="MIP")      doAND = L1MuGMTConfig::getGMTParams()->getMergeMethodMIPSpecialUseANDFwd();
    else if(m_param=="ISO") doAND = L1MuGMTConfig::getGMTParams()->getMergeMethodISOSpecialUseANDFwd();
    if(doAND) m_value[1] |= 64;
    if ( L1MuGMTConfig::Debug(1) ) edm::LogVerbatim("GMT_Register_Info") << " "
                      << "MergeMethod" << m_param  << "SpecialUseANDFwd"
                      << " is " << doAND;

  };

  //// destructor
  virtual ~L1MuGMTRegMMConfigMIPISO() {};
} ;


//
/// \class L1MuGMTRegMMConfigMIP
//
/// GMT Merge Method Config Register MIP
 
class L1MuGMTRegMMConfigMIP : public L1MuGMTRegMMConfigMIPISO {
 public:
  L1MuGMTRegMMConfigMIP() : L1MuGMTRegMMConfigMIPISO("MIP", L1MuGMTRegMMConfig::Special, L1MuGMTRegMMConfig::Special, false, false ) {};
};

//
/// \class L1MuGMTRegMMConfigISO
//
/// GMT Merge Method Config Register ISO
 
class L1MuGMTRegMMConfigISO : public L1MuGMTRegMMConfigMIPISO {
 public:
  L1MuGMTRegMMConfigISO() : L1MuGMTRegMMConfigMIPISO("ISO", L1MuGMTRegMMConfig::Special, L1MuGMTRegMMConfig::Special, true, true ) {};
};

//
/// \class L1MuGMTRegMMConfigSRK
//
/// GMT Register that implements additional Halo Overwrites Matched bit
//

class L1MuGMTRegMMConfigSRK : public L1MuGMTRegMMConfig {

 public: 
  //// constructor
  L1MuGMTRegMMConfigSRK() : L1MuGMTRegMMConfig("SRK", L1MuGMTRegMMConfig::takeDTCSC, L1MuGMTRegMMConfig::takeDTCSC) { 

    bool haloOverwrites;
  
    haloOverwrites = L1MuGMTConfig::getGMTParams()->getHaloOverwritesMatchedBrl();
    if (haloOverwrites) m_value[0] |= 64;
    if ( L1MuGMTConfig::Debug(1) ) edm::LogVerbatim("GMT_Register_info") << " "
                      << "HaloOverwritesMatchedBrl"
                      << " is " << haloOverwrites;

    haloOverwrites = L1MuGMTConfig::getGMTParams()->getHaloOverwritesMatchedFwd();
    if (haloOverwrites) m_value[1] |= 64;
    if ( L1MuGMTConfig::Debug(1) ) edm::LogVerbatim("GMT_Register_info") << " "
                      << "HaloOverwritesMatchedFwd"
                      << " is " << haloOverwrites;

  };

  //// destructor
  virtual ~L1MuGMTRegMMConfigSRK() {};
} ;

//
/// \class L1MuGMTRegSortRankOffset
//
/// GMT Register that implements Rank offset for merged cands
//

class L1MuGMTRegSortRankOffset : public L1MuGMTReg {

 public: 
  //// constructor
  L1MuGMTRegSortRankOffset() {

    m_value[0] = L1MuGMTConfig::getGMTParams()->getSortRankOffsetBrl();
    if ( L1MuGMTConfig::Debug(1) ) 
       edm::LogVerbatim("GMT_Register_info") << " SortRankOffsetBrl is " << m_value[0];

    m_value[1] = L1MuGMTConfig::getGMTParams()->getSortRankOffsetFwd();
    if ( L1MuGMTConfig::Debug(1) ) 
       edm::LogVerbatim("GMT_Register_info") << " SortRankOffsetFwd is " << m_value[1];

  };

  //// get Name
  virtual std::string getName() { return "SortRankOffset"; };

  //// destructor
  virtual ~L1MuGMTRegSortRankOffset() {};
} ;

//
/// \class L1MuGMTRegCDLConfig 
//
/// GMT Register that implements Configuration of Cancel Decisison Logic 
//

class L1MuGMTRegCDLConfig : public L1MuGMTReg {

 public: 
  //// constructor
  L1MuGMTRegCDLConfig() : L1MuGMTReg(4) { 
    setCDLConfig(); 
  };

  //// destructor
  virtual ~L1MuGMTRegCDLConfig() {};

  //// get Name
  virtual std::string getName() { return std::string("CDLConfig"); };

  //// read the merge method from .orcarc
  void setCDLConfig() {

    m_value[0] = L1MuGMTConfig::getGMTParams()->getCDLConfigWordDTCSC();
    if ( L1MuGMTConfig::Debug(1) ) 
         edm::LogVerbatim("GMT_Register_info") << " CDLConfigWordDTCSC is " << m_value[0];
    
    m_value[1] = L1MuGMTConfig::getGMTParams()->getCDLConfigWordCSCDT();
    if ( L1MuGMTConfig::Debug(1) ) 
         edm::LogVerbatim("GMT_Register_info") << " CDLConfigWordCSCDT is " << m_value[1];
    
    m_value[2] = L1MuGMTConfig::getGMTParams()->getCDLConfigWordbRPCCSC();
    if ( L1MuGMTConfig::Debug(1) ) 
         edm::LogVerbatim("GMT_Register_info") << " CDLConfigWordbRPCCSC is " << m_value[2];
    
    m_value[3] = L1MuGMTConfig::getGMTParams()->getCDLConfigWordfRPCDT();
    if ( L1MuGMTConfig::Debug(1) ) 
         edm::LogVerbatim("GMT_Register_info") << " CDLConfigWordfRPCDT is " << m_value[3];
    
  };

 protected:
} ;


#endif












