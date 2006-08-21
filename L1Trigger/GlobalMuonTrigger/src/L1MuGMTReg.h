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
//   $Date: 2006/05/15 13:56:02 $
//   $Revision: 1.1 $
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

#include "FWCore/MessageLogger/interface/MessageLogger.h"

//              ---------------------
//              -- Class Interface --
//              ---------------------

using namespace std;

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
  virtual string getName() =0;
  
 protected:
  vector<unsigned> m_value;
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
  L1MuGMTRegMMConfig(const string& param, MergeMethods def_brl, MergeMethods def_fwd) :
    m_param(param) { 
    m_default[0] = def_brl;
    m_default[1] = def_fwd;
    setMergeMethod(); 
  };

  //// destructor
  virtual ~L1MuGMTRegMMConfig() {};

  //// get Name
  virtual string getName() { return "MMConfig_" + m_param; };

  //// read the merge method from .orcarc
  void setMergeMethod() {
    static MergeMethods avlMethods[6] = { takeDTCSC, takeRPC, byRank, byMinPt, byCombi, Special };
    string mn[6] = { "takeDT", "takeRPC", "byRank", "byMinPt", "byCombi", "Special" };

    for (int i=0; i<2; i++) {
      if (i==1) mn[0] = "takeCSC";
      //      string conf_name = "L1GlobalMuonTrigger:MergeMethod" + m_param + (i ? "Fwd" : "Brl");    
      //      ConfigurableEnum<MergeMethods,6> mm(m_default[i], avlMethods, mn, conf_name);
      MergeMethods mm = m_default[i];
      string conf_name = "MergeMethod" + m_param + (i ? "Fwd" : "Brl");    
      string mm_str = L1MuGMTConfig::getParameterSet()->getParameter<string> (conf_name);
      for(int ii=0; ii<6; ii++) if(mm_str == mn[ii]) {mm = avlMethods[ii]; break;}
      
      m_value[i] = 1 << (5-(int) MergeMethods(mm));

      if ( L1MuGMTConfig::Debug(1) ) edm::LogVerbatim("GMT_Register_Info") << " " << conf_name
				       //					  << " is " << mm.get() 
					  << " is " << mm
					  << "( value " << m_value[i] << " )" << endl;
    }
  };

 protected:
  string m_param;
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
  L1MuGMTRegMMConfigMIPISO(const string& param, MergeMethods def_brl, MergeMethods def_fwd, bool def_and_brl, bool def_and_fwd) :
    L1MuGMTRegMMConfig(param, def_brl, def_fwd) { 

    for (int i=0; i<2; i++) {
      //      string conf_name = "L1GlobalMuonTrigger:MergeMethod" + m_param  + "SpecialUseAND" + (i ? "Fwd" : "Brl");    
      //      bool doAND = SimpleConfigurable<bool> (i? def_and_fwd: def_and_brl, conf_name.c_str() );
      string conf_name = "MergeMethod" + m_param  + "SpecialUseAND" + (i ? "Fwd" : "Brl");    
      bool doAND = L1MuGMTConfig::getParameterSet()->getParameter<bool> (conf_name);

      if (doAND) m_value[i] |= 64;
      
      if ( L1MuGMTConfig::Debug(1) ) edm::LogVerbatim("GMT_Register_Info") << " " << conf_name
					  << " is " << doAND << endl;
    }
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

    for (int i=0; i<2; i++) {
      //      string conf_name = string("L1GlobalMuonTrigger:HaloOverwritesMatched") + (i ? "Fwd" : "Brl");    
      //      bool haloOverwrites = SimpleConfigurable<bool> (true, conf_name.c_str()); // edit default, here
      string conf_name = string("HaloOverwritesMatched") + (i ? "Fwd" : "Brl");    
      bool haloOverwrites = L1MuGMTConfig::getParameterSet()->getParameter<bool> (conf_name); // edit default, here

      if (haloOverwrites) m_value[i] |= 64;
      
      if ( L1MuGMTConfig::Debug(1) ) edm::LogVerbatim("GMT_Register_info") << " " << conf_name
					  << " is " << haloOverwrites << endl;
    }
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

    for (int i=0; i<2; i++) {
      //      string conf_name = string("L1GlobalMuonTrigger:SortRankOffset") + (i ? "Fwd" : "Brl");
      //      unsigned ofs = SimpleConfigurable<unsigned> (10, conf_name.c_str()); // edit default, here
      string conf_name = string("SortRankOffset") + (i ? "Fwd" : "Brl");
      unsigned ofs = L1MuGMTConfig::getParameterSet()->getParameter<unsigned> (conf_name);

      m_value[i] = ofs;
      
      if ( L1MuGMTConfig::Debug(1) ) edm::LogVerbatim("GMT_Register_info") << " " << conf_name 
					  << " is " << ofs << endl;
    }
  };

  //// get Name
  virtual string getName() { return "SortRankOffset"; };

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
  virtual string getName() { return string("CDLConfig"); };

  //// read the merge method from .orcarc
  void setCDLConfig() {
    static char* names[4] = { "DTCSC", "CSCDT", "bRPCCSC", "fRPCDT"};
    //    static unsigned defaults[4] = {2, 3, 16, 1}; // change defaults, here

    for (int i=0; i<4; i++) {
      //      string conf_name = string("L1GlobalMuonTrigger:CDLConfigWord") + names[i];
      //      unsigned cfgword = SimpleConfigurable<unsigned> (defaults[i], conf_name.c_str()); 
      string conf_name = string("CDLConfigWord") + names[i];
      unsigned cfgword = L1MuGMTConfig::getParameterSet()->getParameter<unsigned> (conf_name);

      m_value[i] = cfgword;
      
      if ( L1MuGMTConfig::Debug(1) ) edm::LogVerbatim("GMT_Register_info") << " " << conf_name 
					  << " is " << cfgword << endl;
    }    
  };

 protected:
} ;


#endif












