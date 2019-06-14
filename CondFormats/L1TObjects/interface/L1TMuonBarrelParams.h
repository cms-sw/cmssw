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
    Node() {
      type_ = "unspecified";
      version_ = 0;
    }
    COND_SERIALIZABLE;
  };
  enum { CONFIG = 0, NUM_BMTF_PARAM_NODES = 2 };

  enum {
    PT_Assignment_nbits_Phi,
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
    DisableNewAlgo,
    NUM_CONFIG_PARAMS
  };

  // after initial integration with downstream code, a small update will change:
  L1MuDTTFParameters l1mudttfparams;
  L1MuDTTFMasks l1mudttfmasks;
  // to this:
  //L1MuDTTFParameters & l1mudttfparams(){return l1mudttfparams_[0]; }
  //L1MuDTTFMasks &      l1mudttfmasks(){return l1mudttfmasks_[0]; }

  /// L1MuBMPtaLut
  typedef std::map<short, short, std::less<short> > LUT;
  ///Qual Pattern LUT
  typedef std::pair<short, short> LUTID;
  typedef std::pair<short, std::vector<short> > LUTCONT;
  typedef std::map<LUTID, LUTCONT> qpLUT;
  ///Eta Pattern LUT
  typedef std::map<short, L1MuDTEtaPattern, std::less<short> > etaLUT;

  class LUTParams {
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
    LUTParams() : pta_lut_(0), phi_lut_(0), pta_threshold_(6), ext_lut_(0) {}
    COND_SERIALIZABLE;
  };

  ~L1TMuonBarrelParams() {}

public:
  unsigned version_;
  unsigned fwVersion_;

  std::vector<Node> pnodes_;
  // std::vector here is just so we can use "blob" in DB and evade max size limitations...
  std::vector<L1MuDTTFParameters> l1mudttfparams_;
  std::vector<L1MuDTTFMasks> l1mudttfmasks_;
  LUTParams lutparams_;

  COND_SERIALIZABLE;
};
#endif
