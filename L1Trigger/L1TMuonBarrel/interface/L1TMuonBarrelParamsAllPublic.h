#ifndef L1TMuonBarrelParamsAllPublic_h
#define L1TMuonBarrelParamsAllPublic_h

#include <cassert>

#include "CondFormats/L1TObjects/interface/L1TMuonBarrelParams.h"

class L1TMuonBarrelParamsAllPublic : public L1TMuonBarrelParams {
public:
  L1TMuonBarrelParamsAllPublic(){};
  ~L1TMuonBarrelParamsAllPublic(){};

  L1TMuonBarrelParamsAllPublic(const L1TMuonBarrelParams& a) {
    assert(sizeof(L1TMuonBarrelParamsAllPublic) == sizeof(L1TMuonBarrelParams));
    //const void * pa = &a;
    //const L1TMuonBarrelParamsAllPublic * py
    *this = *static_cast<const L1TMuonBarrelParamsAllPublic*>(&a);
    //this = py;
    //delete (L1TMuonBarrelParams*)pa;
  };

  std::string AssLUTPath() const { return !pnodes_[CONFIG].sparams_.empty() ? pnodes_[CONFIG].sparams_[0] : ""; };
  void setAssLUTPath(std::string path) { pnodes_[CONFIG].sparams_.push_back(path); };

  void setpta_lut(std::vector<LUT> ptalut) { lutparams_.pta_lut_ = ptalut; };
  std::vector<LUT> pta_lut() const { return lutparams_.pta_lut_; };
  void setpta_threshold(std::vector<int> ptathresh) { lutparams_.pta_threshold_ = ptathresh; };
  std::vector<int> pta_threshold() const { return lutparams_.pta_threshold_; };

  void setphi_lut(std::vector<LUT> philut) { lutparams_.phi_lut_ = philut; };
  std::vector<LUT> phi_lut() const { return lutparams_.phi_lut_; };

  void setext_lut(std::vector<LUTParams::extLUT> extlut) { lutparams_.ext_lut_ = extlut; };
  std::vector<LUTParams::extLUT> ext_lut() const { return lutparams_.ext_lut_; };

  void setqp_lut(qpLUT qplut) { lutparams_.qp_lut_ = qplut; };
  qpLUT qp_lut() const { return lutparams_.qp_lut_; };

  void seteta_lut(etaLUT eta_lut) { lutparams_.eta_lut_ = eta_lut; };
  etaLUT eta_lut() const { return lutparams_.eta_lut_; };

  void set_PT_Assignment_nbits_Phi(int par1) { pnodes_[CONFIG].iparams_[PT_Assignment_nbits_Phi] = par1; };
  void set_PT_Assignment_nbits_PhiB(int par1) { pnodes_[CONFIG].iparams_[PT_Assignment_nbits_PhiB] = par1; };
  void set_PHI_Assignment_nbits_Phi(int par1) { pnodes_[CONFIG].iparams_[PHI_Assignment_nbits_Phi] = par1; };
  void set_PHI_Assignment_nbits_PhiB(int par1) { pnodes_[CONFIG].iparams_[PHI_Assignment_nbits_PhiB] = par1; };
  void set_Extrapolation_nbits_Phi(int par1) { pnodes_[CONFIG].iparams_[Extrapolation_nbits_Phi] = par1; };
  void set_Extrapolation_nbits_PhiB(int par1) { pnodes_[CONFIG].iparams_[Extrapolation_nbits_PhiB] = par1; };
  void set_BX_min(int par1) { pnodes_[CONFIG].iparams_[BX_min] = par1; };
  void set_BX_max(int par1) { pnodes_[CONFIG].iparams_[BX_max] = par1; };
  void set_Extrapolation_Filter(int par1) { pnodes_[CONFIG].iparams_[Extrapolation_Filter] = par1; };
  void set_OutOfTime_Filter_Window(int par1) { pnodes_[CONFIG].iparams_[OutOfTime_Filter_Window] = par1; };
  void set_OutOfTime_Filter(bool par1) { pnodes_[CONFIG].iparams_[OutOfTime_Filter] = par1; };
  void set_Open_LUTs(bool par1) { pnodes_[CONFIG].iparams_[Open_LUTs] = par1; };
  void set_EtaTrackFinder(bool par1) { pnodes_[CONFIG].iparams_[EtaTrackFinder] = par1; };
  void set_Extrapolation_21(bool par1) { pnodes_[CONFIG].iparams_[Extrapolation_21] = par1; };
  void set_DisableNewAlgo(bool par1) { pnodes_[CONFIG].iparams_[DisableNewAlgo] = par1; };

  int get_PT_Assignment_nbits_Phi() const { return pnodes_[CONFIG].iparams_[PT_Assignment_nbits_Phi]; };
  int get_PT_Assignment_nbits_PhiB() const { return pnodes_[CONFIG].iparams_[PT_Assignment_nbits_PhiB]; };
  int get_PHI_Assignment_nbits_Phi() const { return pnodes_[CONFIG].iparams_[PHI_Assignment_nbits_Phi]; };
  int get_PHI_Assignment_nbits_PhiB() const { return pnodes_[CONFIG].iparams_[PHI_Assignment_nbits_PhiB]; };
  int get_Extrapolation_nbits_Phi() const { return pnodes_[CONFIG].iparams_[Extrapolation_nbits_Phi]; };
  int get_Extrapolation_nbits_PhiB() const { return pnodes_[CONFIG].iparams_[Extrapolation_nbits_PhiB]; };
  int get_BX_min() const { return pnodes_[CONFIG].iparams_[BX_min]; };
  int get_BX_max() const { return pnodes_[CONFIG].iparams_[BX_max]; };
  int get_Extrapolation_Filter() const { return pnodes_[CONFIG].iparams_[Extrapolation_Filter]; };
  int get_OutOfTime_Filter_Window() const { return pnodes_[CONFIG].iparams_[OutOfTime_Filter_Window]; };

  bool get_OutOfTime_Filter() const { return pnodes_[CONFIG].iparams_[OutOfTime_Filter]; };
  bool get_Open_LUTs() const { return pnodes_[CONFIG].iparams_[Open_LUTs]; };
  bool get_EtaTrackFinder() const { return pnodes_[CONFIG].iparams_[EtaTrackFinder]; };
  bool get_Extrapolation_21() const { return pnodes_[CONFIG].iparams_[Extrapolation_21]; };
  bool get_DisableNewAlgo() const { return pnodes_[CONFIG].iparams_[DisableNewAlgo]; };

  unsigned fwVersion() const { return fwVersion_; };
  void setFwVersion(unsigned fwVersion) { fwVersion_ = fwVersion; };
};

//const L1TMuonBarrelParamsAllPublic& cast_to_L1TMuonBarrelParamsAllPublic(const L1TMuonBarrelParams& a);
const L1TMuonBarrelParams& cast_to_L1TMuonBarrelParams(const L1TMuonBarrelParamsAllPublic& a);

#endif
