#ifndef ELECTRONSELECTIONSPARAMETERS_H
#define ELECTRONSELECTIONSPARAMETERS_H

#include <vector>

namespace HWWFunctions {

  //
  // 2012 cut based WP
  //

  enum wp2012_tightness {
      VETO,
      LOOSE,
      MEDIUM,
      TIGHT
  };

  //
  // Data required for V02 of 
  // Branson/Sani electron ID
  //
  // Settings provided
  //
  //--------------------------------
  enum cic_tightness {
      CIC_VERYLOOSE,
      CIC_LOOSE,
      CIC_MEDIUM,
      CIC_TIGHT,
      CIC_SUPERTIGHT,
      CIC_HYPERTIGHT1,
      CIC_HYPERTIGHT2,
      CIC_HYPERTIGHT3,
      CIC_HYPERTIGHT4
  };
  //--------------------------------

  //
  // Data required for VBTF ID
  // optimised for us by N. Rompotis/C. Seez
  //
  // Settings provided
  //
  //--------------------------------
  enum vbtf_tightness {
      VBTF_35X_95,
      VBTF_35X_90,
      VBTF_35X_85,
      VBTF_35X_80,
      VBTF_35X_70,
      VBTF_35X_60,
      VBTF_35Xr2_70,
      VBTF_35Xr2_60,
      VBTF_80_NOHOEEND,
      VBTF_85_NOHOEEND,
      VBTF_85,
      VBTF_70_NOHOEEND,
      VBTF_90_HLT,
      VBTF_90_HLT_CALOIDT_TRKIDVL,
      VBTF_95_NOHOEEND
  };
  //--------------------------------

  //
  // Data required for CAND ID
  //
  // Settings provided
  //
  //--------------------------------
  enum cand_tightness {
      CAND_01,
      CAND_02
  };
  //--------------------------------

  void eidGetWP2012(const wp2012_tightness tightness, std::vector<double> &cutdeta, std::vector<double> &cutdphi, std::vector<double> &cuthoe, std::vector<double> &cutsee, std::vector<double> &cutooemoop, std::vector<double> &cutd0vtx, std::vector<double> &cutdzvtx, std::vector<bool> &cutvtxfit, std::vector<int> &cutmhit, std::vector<double> &cutrelisohighpt, std::vector<double> &cutrelisolowpt);

  void eidGetVBTF(const vbtf_tightness tightness, std::vector<double> &cutdeta, std::vector<double> &cutdphi, std::vector<double> &cuthoe, std::vector<double> &cutsee, std::vector<double> &cutreliso);

  void eidAssign(std::vector<double> &cutarr, double cutvals[], unsigned int size);
  void eidAssign(std::vector<int> &cutarr, int cutvals[], unsigned int size);
  void eidAssign(std::vector<bool> &cutarr, bool cutvals[], unsigned int size);
}
#endif
