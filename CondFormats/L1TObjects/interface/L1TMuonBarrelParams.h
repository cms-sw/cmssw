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
#include "L1Trigger/L1TMuonBarrel/interface/L1MuBMTrack.h"

class L1TMuonBarrelParams {

public:
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
      pta = 0,
	  NUM_BMTFPARAMNODES=1
  };


    /// L1MuBMPtaLut
    typedef std::map<short, short, std::less<short> > LUT;

    class LUTParams{
    public:
      std::vector<LUT> pta_lut_;
      std::vector<LUT> phi_lut_;
      std::vector<int> pta_threshold_;


      LUTParams() : pta_lut_(0), phi_lut_(0), pta_threshold_(6)
      { pta_lut_.reserve(12); pta_threshold_.reserve(6); phi_lut_.reserve(2); }

      COND_SERIALIZABLE;
    };
  std::string AssLUTPath() const  { return pnodes_[pta].sparams_.size() > 0 ? pnodes_[pta].sparams_[0] : ""; }
  void setAssLUTPath        (std::string path) { pnodes_[pta].sparams_.push_back(path); }

  void setpta_lut(std::vector<LUT> ptalut) { lutparams_.pta_lut_ = ptalut; };
  std::vector<LUT> pta_lut() const {return lutparams_.pta_lut_; };
  void setpta_threshold(std::vector<int> ptathresh) { lutparams_.pta_threshold_ = ptathresh;  };
  std::vector<int> pta_threshold() const { return lutparams_.pta_threshold_;  };

  void setphi_lut(std::vector<LUT> philut) { lutparams_.phi_lut_ = philut; };
  std::vector<LUT> phi_lut() const {return lutparams_.phi_lut_; };


    class ConfigParams{
    public:
        int PT_Assignment_nbits_Phi;
        int PT_Assignment_nbits_PhiB;
        int PHI_Assignment_nbits_Phi;
        int PHI_Assignment_nbits_PhiB;
        int Extrapolation_nbits_Phi;
        int Extrapolation_nbits_PhiB;
        int BX_min;
        int BX_max;
        int Extrapolation_Filter;
        int OutOfTime_Filter_Window;
        bool OutOfTime_Filter;
        bool Open_LUTs;
        bool EtaTrackFinder;
        bool Extrapolation_21;


      ConfigParams() {  }

      COND_SERIALIZABLE;
    };

    void set_PT_Assignment_nbits_Phi(int par1) {conparams_.PT_Assignment_nbits_Phi = par1;}
    void set_PT_Assignment_nbits_PhiB(int par1) {conparams_.PT_Assignment_nbits_PhiB = par1;}
    void set_PHI_Assignment_nbits_Phi(int par1) {conparams_.PHI_Assignment_nbits_Phi = par1;}
    void set_PHI_Assignment_nbits_PhiB(int par1) {conparams_.PHI_Assignment_nbits_PhiB = par1;}
    void set_Extrapolation_nbits_Phi(int par1) {conparams_.Extrapolation_nbits_Phi = par1;}
    void set_Extrapolation_nbits_PhiB(int par1) {conparams_.Extrapolation_nbits_PhiB = par1;}
    void set_BX_min(int par1) {conparams_.BX_min = par1;}
    void set_BX_max(int par1) {conparams_.BX_max = par1;}
    void set_Extrapolation_Filter(int par1) {conparams_.Extrapolation_Filter = par1;}
    void set_OutOfTime_Filter_Window(int par1) {conparams_.OutOfTime_Filter_Window = par1;}
    void set_OutOfTime_Filter(bool par1) {conparams_.OutOfTime_Filter = par1;}
    void set_Open_LUTs(bool par1) {conparams_.Open_LUTs = par1;}
    void set_EtaTrackFinder(bool par1) {conparams_.EtaTrackFinder = par1;}
    void set_Extrapolation_21(bool par1) {conparams_.Extrapolation_21 = par1;}


    int get_PT_Assignment_nbits_Phi() const{return conparams_.PT_Assignment_nbits_Phi;}
    int get_PT_Assignment_nbits_PhiB() const {return conparams_.PT_Assignment_nbits_PhiB;}
    int get_PHI_Assignment_nbits_Phi() const {return conparams_.PHI_Assignment_nbits_Phi;}
    int get_PHI_Assignment_nbits_PhiB() const {return conparams_.PHI_Assignment_nbits_PhiB;}
    int get_Extrapolation_nbits_Phi() const {return conparams_.Extrapolation_nbits_Phi;}
    int get_Extrapolation_nbits_PhiB() const {return conparams_.Extrapolation_nbits_PhiB;}
    int get_BX_min() const {return conparams_.BX_min ;}
    int get_BX_max() const {return conparams_.BX_max;}
    int get_Extrapolation_Filter() const {return conparams_.Extrapolation_Filter;}
    int get_OutOfTime_Filter_Window() const {return conparams_.OutOfTime_Filter_Window ;}

    bool get_OutOfTime_Filter() const {return conparams_.OutOfTime_Filter;}
    bool get_Open_LUTs() const {return conparams_.Open_LUTs ;}
    bool get_EtaTrackFinder() const {return conparams_.EtaTrackFinder ;}
    bool get_Extrapolation_21() const {return conparams_.Extrapolation_21 ;}


  L1TMuonBarrelParams() { version_=Version; pnodes_.resize(NUM_BMTFPARAMNODES); }
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

  LUTParams lutparams_;
  ConfigParams conparams_;

  COND_SERIALIZABLE;
};
#endif
