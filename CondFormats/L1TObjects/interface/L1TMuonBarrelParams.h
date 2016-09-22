///
/// \class L1TMuonBarrelParams
///
/// Description: Placeholder for BMTF parameters
///
///
/// \author: Giannis Flouris
///

#ifndef L1TBMTFParams_h
#define L1TBMTFParams_h

#include <memory>
#include <iostream>
#include <vector>

#include "CondFormats/Serialization/interface/Serializable.h"
#include "CondFormats/L1TObjects/interface/LUT.h"
#include "CondFormats/L1TObjects/interface/L1MuDTTFParameters.h"
#include "CondFormats/L1TObjects/interface/L1MuDTTFMasks.h"
#include "CondFormats/L1TObjects/interface/L1MuDTEtaPattern.h"


class L1TMuonBarrelParams {
public:
  L1TMuonBarrelParams();
  enum { Version = 1 };
  
  class Node {
  public:
    std::string type_;
    unsigned version_;
    l1t::LUT LUT_;
    std::vector<double> dparams_;
    std::vector<unsigned> uparams_;
    std::vector<int> iparams_;
    std::vector<std::string> sparams_;
    Node(){ type_="unspecified"; version_=0; }
    COND_SERIALIZABLE;
  };
  enum {
    CONFIG = 0,
    NUM_BMTF_PARAM_NODES=2
  };

  enum { PT_Assignment_nbits_Phi,
	 PT_Assignment_nbits_PhiB,
	 PHI_Assignment_nbits_Phi,
	 PHI_Assignment_nbits_PhiB,
	 Extrapolation_nbits_Phi,
	 Extrapolation_nbits_PhiB,
	 BX_min,
	 BX_max,
	 Extrapolation_Filter,
	 OutOfTime_Filter_Window,
	 OutOfTime_Filter,
	 Open_LUTs,
	 EtaTrackFinder,
	 Extrapolation_21, 
	 NUM_CONFIG_PARAMS};

  // after initial integration with downstream code, a small update will change:
  L1MuDTTFParameters  l1mudttfparams;
  L1MuDTTFMasks       l1mudttfmasks;
  // to this:
  //L1MuDTTFParameters & l1mudttfparams(){return l1mudttfparams_[0]; }
  //L1MuDTTFMasks &      l1mudttfmasks(){return l1mudttfmasks_[0]; }

  /// L1MuBMPtaLut
  typedef std::map<short, short, std::less<short> > LUT;
  ///Qual Pattern LUT
  typedef std::pair< short, short > LUTID;
  typedef std::pair< short, std::vector<short> > LUTCONT;
  typedef std::map< LUTID, LUTCONT > qpLUT;
  ///Eta Pattern LUT
  typedef std::map<short, L1MuDTEtaPattern, std::less<short> > etaLUT;
  
  class LUTParams{
  public:
    std::vector<LUT> pta_lut_;
    std::vector<LUT> phi_lut_;
    std::vector<int> pta_threshold_;
    qpLUT qp_lut_;
    etaLUT eta_lut_;

    // helper class for extrapolation look-up tables
    class extLUT {
    public:
      LUT low;
      LUT high;
      COND_SERIALIZABLE;
    };
    std::vector<extLUT> ext_lut_;
    LUTParams() : pta_lut_(0), phi_lut_(0), pta_threshold_(6), ext_lut_(0){  }
    COND_SERIALIZABLE;
  };
  
  


  std::string AssLUTPath() const  { return pnodes_[CONFIG].sparams_.size() > 0 ? pnodes_[CONFIG].sparams_[0] : ""; }
  void setAssLUTPath        (std::string path) { pnodes_[CONFIG].sparams_.push_back(path); }
  
  void setpta_lut(std::vector<LUT> ptalut) { lutparams_.pta_lut_ = ptalut; };
  std::vector<LUT> pta_lut() const {return lutparams_.pta_lut_; };
  void setpta_threshold(std::vector<int> ptathresh) { lutparams_.pta_threshold_ = ptathresh;  };
  std::vector<int> pta_threshold() const { return lutparams_.pta_threshold_;  };
  
  void setphi_lut(std::vector<LUT> philut) { lutparams_.phi_lut_ = philut; };
  std::vector<LUT> phi_lut() const {return lutparams_.phi_lut_; };
  
  void setext_lut(std::vector<LUTParams::extLUT> extlut) { lutparams_.ext_lut_ = extlut; };
  std::vector<LUTParams::extLUT> ext_lut() const {return lutparams_.ext_lut_; };
  
  void setqp_lut(qpLUT qplut) { lutparams_.qp_lut_ = qplut; };
  qpLUT qp_lut() const {return lutparams_.qp_lut_; };
  
  void seteta_lut(etaLUT eta_lut) { lutparams_.eta_lut_ = eta_lut; };
  etaLUT eta_lut() const {return lutparams_.eta_lut_; };
  
  
  void set_PT_Assignment_nbits_Phi(int par1) {pnodes_[CONFIG].iparams_[PT_Assignment_nbits_Phi] = par1;}
  void set_PT_Assignment_nbits_PhiB(int par1) {pnodes_[CONFIG].iparams_[PT_Assignment_nbits_PhiB] = par1;}
  void set_PHI_Assignment_nbits_Phi(int par1) {pnodes_[CONFIG].iparams_[PHI_Assignment_nbits_Phi] = par1;}
  void set_PHI_Assignment_nbits_PhiB(int par1) {pnodes_[CONFIG].iparams_[PHI_Assignment_nbits_PhiB] = par1;}
  void set_Extrapolation_nbits_Phi(int par1) {pnodes_[CONFIG].iparams_[Extrapolation_nbits_Phi] = par1;}
  void set_Extrapolation_nbits_PhiB(int par1) {pnodes_[CONFIG].iparams_[Extrapolation_nbits_PhiB] = par1;}
  void set_BX_min(int par1) {pnodes_[CONFIG].iparams_[BX_min] = par1;}
  void set_BX_max(int par1) {pnodes_[CONFIG].iparams_[BX_max] = par1;}
  void set_Extrapolation_Filter(int par1) {pnodes_[CONFIG].iparams_[Extrapolation_Filter] = par1;}
  void set_OutOfTime_Filter_Window(int par1) {pnodes_[CONFIG].iparams_[OutOfTime_Filter_Window] = par1;}
  void set_OutOfTime_Filter(bool par1) {pnodes_[CONFIG].iparams_[OutOfTime_Filter] = par1;}
  void set_Open_LUTs(bool par1) {pnodes_[CONFIG].iparams_[Open_LUTs] = par1;}
  void set_EtaTrackFinder(bool par1) {pnodes_[CONFIG].iparams_[EtaTrackFinder] = par1;}
  void set_Extrapolation_21(bool par1) {pnodes_[CONFIG].iparams_[Extrapolation_21] = par1;}
  

  int get_PT_Assignment_nbits_Phi() const{return pnodes_[CONFIG].iparams_[PT_Assignment_nbits_Phi];}
  int get_PT_Assignment_nbits_PhiB() const {return pnodes_[CONFIG].iparams_[PT_Assignment_nbits_PhiB];}
  int get_PHI_Assignment_nbits_Phi() const {return pnodes_[CONFIG].iparams_[PHI_Assignment_nbits_Phi];}
  int get_PHI_Assignment_nbits_PhiB() const {return pnodes_[CONFIG].iparams_[PHI_Assignment_nbits_PhiB];}
  int get_Extrapolation_nbits_Phi() const {return pnodes_[CONFIG].iparams_[Extrapolation_nbits_Phi];}
  int get_Extrapolation_nbits_PhiB() const {return pnodes_[CONFIG].iparams_[Extrapolation_nbits_PhiB];}
  int get_BX_min() const {return pnodes_[CONFIG].iparams_[BX_min] ;}
  int get_BX_max() const {return pnodes_[CONFIG].iparams_[BX_max];}
  int get_Extrapolation_Filter() const {return pnodes_[CONFIG].iparams_[Extrapolation_Filter];}
  int get_OutOfTime_Filter_Window() const {return pnodes_[CONFIG].iparams_[OutOfTime_Filter_Window] ;}
  
  bool get_OutOfTime_Filter() const {return pnodes_[CONFIG].iparams_[OutOfTime_Filter];}
  bool get_Open_LUTs() const {return pnodes_[CONFIG].iparams_[Open_LUTs] ;}
  bool get_EtaTrackFinder() const {return pnodes_[CONFIG].iparams_[EtaTrackFinder] ;}
  bool get_Extrapolation_21() const {return pnodes_[CONFIG].iparams_[Extrapolation_21] ;}
  
  ~L1TMuonBarrelParams() {}

  // FW version
  unsigned fwVersion() const { return fwVersion_; }
  void setFwVersion(unsigned fwVersion) { fwVersion_ = fwVersion; }

  // print parameters to stream:
  void print(std::ostream&) const;
  friend std::ostream& operator<<(std::ostream& o, const L1TMuonBarrelParams & p) { p.print(o); return o; }
 private:
  unsigned version_;
  unsigned fwVersion_;
  
  std::vector<Node> pnodes_;
  // std::vector here is just so we can use "blob" in DB and evade max size limitations...
  std::vector<L1MuDTTFParameters> l1mudttfparams_;
  std::vector<L1MuDTTFMasks>      l1mudttfmasks_;  
  LUTParams lutparams_;
  
  COND_SERIALIZABLE;
};
#endif
