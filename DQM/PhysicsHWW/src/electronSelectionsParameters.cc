#include <iostream>
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DQM/PhysicsHWW/interface/electronSelectionsParameters.h"

namespace HWWFunctions {

  void eidGetWP2012(const wp2012_tightness tightness, std::vector<double> &cutdeta, std::vector<double> &cutdphi, std::vector<double> &cuthoe, std::vector<double> &cutsee, std::vector<double> &cutooemoop, std::vector<double> &cutd0vtx, std::vector<double> &cutdzvtx, std::vector<bool> &cutvtxfit, std::vector<int> &cutmhit, std::vector<double> &cutrelisohighpt, std::vector<double> &cutrelisolowpt)
  {

      switch (tightness) {
          case VETO:
              {
                  double dEtaIn_tmp[2]        = {0.007, 0.010};
                  double dPhiIn_tmp[2]        = {0.800, 0.700};
                  double sigmaIEtaIEta_tmp[2] = {0.010, 0.030};
                  double hoe_tmp[2]           = {0.150, 999.9};
                  double ooemoop_tmp[2]       = {999.9, 999.9};
                  double d0Vtx_tmp[2]         = {0.040, 0.040};
                  double dzVtx_tmp[2]         = {0.200, 0.200};
                  bool vtxFit_tmp[2]          = {false, false};
                  int mHits_tmp[2]            = {999, 999};
                  double isoHi_tmp[2]         = {0.150, 0.150};
                  double isoLo_tmp[2]         = {0.150, 0.150};
                  eidAssign(cutdeta,          dEtaIn_tmp, 2);
                  eidAssign(cutdphi,          dPhiIn_tmp, 2);
                  eidAssign(cutsee,           sigmaIEtaIEta_tmp, 2);
                  eidAssign(cuthoe,           hoe_tmp, 2);
                  eidAssign(cutooemoop,       ooemoop_tmp, 2);
                  eidAssign(cutd0vtx,         d0Vtx_tmp, 2);
                  eidAssign(cutdzvtx,         dzVtx_tmp, 2);
                  eidAssign(cutvtxfit,        vtxFit_tmp, 2);
                  eidAssign(cutmhit,          mHits_tmp, 2);
                  eidAssign(cutrelisohighpt,  isoHi_tmp, 2);
                  eidAssign(cutrelisolowpt,   isoLo_tmp, 2);
                  return;
              }
          case LOOSE:
              {
                  double dEtaIn_tmp[2]        = {0.007, 0.009};
                  double dPhiIn_tmp[2]        = {0.150, 0.100};
                  double sigmaIEtaIEta_tmp[2] = {0.010, 0.030};
                  double hoe_tmp[2]           = {0.120, 0.100};
                  double ooemoop_tmp[2]       = {0.050, 0.050};
                  double d0Vtx_tmp[2]         = {0.020, 0.020};
                  double dzVtx_tmp[2]         = {0.200, 0.200};
                  bool vtxFit_tmp[2]          = {true, true};
                  int mHits_tmp[2]            = {1, 1};
                  double isoHi_tmp[2]         = {0.150, 0.150};
                  double isoLo_tmp[2]         = {0.150, 0.100};
                  eidAssign(cutdeta,          dEtaIn_tmp, 2);
                  eidAssign(cutdphi,          dPhiIn_tmp, 2);
                  eidAssign(cutsee,           sigmaIEtaIEta_tmp, 2);
                  eidAssign(cuthoe,           hoe_tmp, 2);
                  eidAssign(cutooemoop,       ooemoop_tmp, 2);
                  eidAssign(cutd0vtx,         d0Vtx_tmp, 2);
                  eidAssign(cutdzvtx,         dzVtx_tmp, 2);
                  eidAssign(cutvtxfit,        vtxFit_tmp, 2);
                  eidAssign(cutmhit,          mHits_tmp, 2);
                  eidAssign(cutrelisohighpt,  isoHi_tmp, 2);
                  eidAssign(cutrelisolowpt,   isoLo_tmp, 2);
                  return;
              }
          case MEDIUM:
              {
                  double dEtaIn_tmp[2]        = {0.004, 0.007};
                  double dPhiIn_tmp[2]        = {0.060, 0.030};
                  double sigmaIEtaIEta_tmp[2] = {0.010, 0.030};
                  double hoe_tmp[2]           = {0.120, 0.100};
                  double ooemoop_tmp[2]       = {0.050, 0.050};
                  double d0Vtx_tmp[2]         = {0.020, 0.020};
                  double dzVtx_tmp[2]         = {0.100, 0.100};
                  bool vtxFit_tmp[2]          = {true, true};
                  int mHits_tmp[2]            = {1, 1};
                  double isoHi_tmp[2]         = {0.150, 0.150};
                  double isoLo_tmp[2]         = {0.150, 0.100};
                  eidAssign(cutdeta,          dEtaIn_tmp, 2);
                  eidAssign(cutdphi,          dPhiIn_tmp, 2);
                  eidAssign(cutsee,           sigmaIEtaIEta_tmp, 2);
                  eidAssign(cuthoe,           hoe_tmp, 2);
                  eidAssign(cutooemoop,       ooemoop_tmp, 2);
                  eidAssign(cutd0vtx,         d0Vtx_tmp, 2);
                  eidAssign(cutdzvtx,         dzVtx_tmp, 2);
                  eidAssign(cutvtxfit,        vtxFit_tmp, 2);
                  eidAssign(cutmhit,          mHits_tmp, 2);
                  eidAssign(cutrelisohighpt,  isoHi_tmp, 2);
                  eidAssign(cutrelisolowpt,   isoLo_tmp, 2);
                  return;
              }
          case TIGHT:
              {
                  double dEtaIn_tmp[2]        = {0.004, 0.005};
                  double dPhiIn_tmp[2]        = {0.030, 0.020};
                  double sigmaIEtaIEta_tmp[2] = {0.010, 0.030};
                  double hoe_tmp[2]           = {0.120, 0.100};
                  double ooemoop_tmp[2]       = {0.050, 0.050};
                  double d0Vtx_tmp[2]         = {0.020, 0.020};
                  double dzVtx_tmp[2]         = {0.100, 0.100};
                  bool vtxFit_tmp[2]          = {true, true};
                  int mHits_tmp[2]            = {0, 0};
                  double isoHi_tmp[2]         = {0.100, 0.100};
                  double isoLo_tmp[2]         = {0.100, 0.070};
                  eidAssign(cutdeta,          dEtaIn_tmp, 2);
                  eidAssign(cutdphi,          dPhiIn_tmp, 2);
                  eidAssign(cutsee,           sigmaIEtaIEta_tmp, 2);
                  eidAssign(cuthoe,           hoe_tmp, 2);
                  eidAssign(cutooemoop,       ooemoop_tmp, 2);
                  eidAssign(cutd0vtx,         d0Vtx_tmp, 2);
                  eidAssign(cutdzvtx,         dzVtx_tmp, 2);
                  eidAssign(cutvtxfit,        vtxFit_tmp, 2);
                  eidAssign(cutmhit,          mHits_tmp, 2);
                  eidAssign(cutrelisohighpt,  isoHi_tmp, 2);
                  eidAssign(cutrelisolowpt,   isoLo_tmp, 2);
                  return;
              }

          default:
              edm::LogError("InvalidInput") << "[eidGetWP2012] ERROR! Invalid tightness level";

      }

      return;

  }

  void eidGetVBTF(const vbtf_tightness tightness, std::vector<double> &cutdeta, std::vector<double> &cutdphi, std::vector<double> &cuthoe, std::vector<double> &cutsee, std::vector<double> &cutreliso)
  {

      switch (tightness) {
          case VBTF_35X_95:
              {
                  double isoThresholds_tmp[2]                 = {0.15,     0.1};
                  double sigmaIEtaIEtaThresholds_tmp[2]       = {0.01,    0.03};
                  double dPhiInThresholds_tmp[2]              = {0.8,     0.7};
                  double dEtaInThresholds_tmp[2]              = {0.007,   0.01};
                  double hoeThresholds_tmp[2]                 = {0.5,     0.07};
                  eidAssign(cutreliso, isoThresholds_tmp, 2);
                  eidAssign(cutdeta, dEtaInThresholds_tmp, 2);
                  eidAssign(cutdphi, dPhiInThresholds_tmp, 2);
                  eidAssign(cuthoe, hoeThresholds_tmp, 2);
                  eidAssign(cutsee, sigmaIEtaIEtaThresholds_tmp, 2);
                  return;
              }

          case VBTF_35X_90:
              {
                  double isoThresholds_tmp[2]                 = {0.1,     0.07};
                  double sigmaIEtaIEtaThresholds_tmp[2]       = {0.01,    0.03};
                  double dPhiInThresholds_tmp[2]              = {0.8,     0.7};
                  double dEtaInThresholds_tmp[2]              = {0.007,   0.009};
                  double hoeThresholds_tmp[2]                 = {0.12,     0.05};
                  eidAssign(cutreliso, isoThresholds_tmp, 2);
                  eidAssign(cutdeta, dEtaInThresholds_tmp, 2);
                  eidAssign(cutdphi, dPhiInThresholds_tmp, 2);
                  eidAssign(cuthoe, hoeThresholds_tmp, 2);
                  eidAssign(cutsee, sigmaIEtaIEtaThresholds_tmp, 2);
                  return;
              }   

          case VBTF_35X_85:
              {
                  double isoThresholds_tmp[2]                 = {0.09,     0.06};
                  double sigmaIEtaIEtaThresholds_tmp[2]       = {0.01,    0.03};
                  double dPhiInThresholds_tmp[2]              = {0.06,     0.04};
                  double dEtaInThresholds_tmp[2]              = {0.006,   0.007};            
                  double hoeThresholds_tmp[2]                 = {0.04,     0.025};
                  eidAssign(cutreliso, isoThresholds_tmp, 2);
                  eidAssign(cutdeta, dEtaInThresholds_tmp, 2);
                  eidAssign(cutdphi, dPhiInThresholds_tmp, 2);
                  eidAssign(cuthoe, hoeThresholds_tmp, 2);
                  eidAssign(cutsee, sigmaIEtaIEtaThresholds_tmp, 2);
                  return;
              }

          case VBTF_35X_80:
              {
                  double isoThresholds_tmp[2]                 = {0.07,     0.06};
                  double sigmaIEtaIEtaThresholds_tmp[2]       = {0.01,    0.03};
                  double dPhiInThresholds_tmp[2]              = {0.06,     0.03};
                  double dEtaInThresholds_tmp[2]              = {0.004,   0.007};            
                  double hoeThresholds_tmp[2]                 = {0.04,     0.025};
                  eidAssign(cutreliso, isoThresholds_tmp, 2);
                  eidAssign(cutdeta, dEtaInThresholds_tmp, 2);
                  eidAssign(cutdphi, dPhiInThresholds_tmp, 2);
                  eidAssign(cuthoe, hoeThresholds_tmp, 2);
                  eidAssign(cutsee, sigmaIEtaIEtaThresholds_tmp, 2);
                  return;
              }

          case VBTF_35X_70:
              {
                  double isoThresholds_tmp[2]                 = {0.05,     0.04};
                  double sigmaIEtaIEtaThresholds_tmp[2]       = {0.01,    0.03};
                  double dPhiInThresholds_tmp[2]              = {0.03,     0.02};
                  double dEtaInThresholds_tmp[2]              = {0.003,   0.005};
                  double hoeThresholds_tmp[2]                 = {0.025,     0.012};
                  eidAssign(cutreliso, isoThresholds_tmp, 2);
                  eidAssign(cutdeta, dEtaInThresholds_tmp, 2);
                  eidAssign(cutdphi, dPhiInThresholds_tmp, 2);
                  eidAssign(cuthoe, hoeThresholds_tmp, 2);
                  eidAssign(cutsee, sigmaIEtaIEtaThresholds_tmp, 2);
                  return;
              }

          case VBTF_35X_60:
              {
                  double isoThresholds_tmp[2]                 = {0.04,     0.03};
                  double sigmaIEtaIEtaThresholds_tmp[2]       = {0.01,    0.03};
                  double dPhiInThresholds_tmp[2]              = {0.02,     0.02};
                  double dEtaInThresholds_tmp[2]              = {0.0025,   0.003};
                  double hoeThresholds_tmp[2]                 = {0.025,     0.009};
                  eidAssign(cutreliso, isoThresholds_tmp, 2);
                  eidAssign(cutdeta, dEtaInThresholds_tmp, 2);
                  eidAssign(cutdphi, dPhiInThresholds_tmp, 2);
                  eidAssign(cuthoe, hoeThresholds_tmp, 2);
                  eidAssign(cutsee, sigmaIEtaIEtaThresholds_tmp, 2);
                  return;
              }

          case VBTF_35Xr2_70:
              {
                  double isoThresholds_tmp[2]                 = {0.04,     0.03};
                  double sigmaIEtaIEtaThresholds_tmp[2]       = {0.01,    0.03};
                  double dPhiInThresholds_tmp[2]              = {0.03,     0.02};
                  double dEtaInThresholds_tmp[2]              = {0.004,   0.005};
                  double hoeThresholds_tmp[2]                 = {0.025,     0.025};
                  eidAssign(cutreliso, isoThresholds_tmp, 2);
                  eidAssign(cutdeta, dEtaInThresholds_tmp, 2);
                  eidAssign(cutdphi, dPhiInThresholds_tmp, 2);
                  eidAssign(cuthoe, hoeThresholds_tmp, 2);
                  eidAssign(cutsee, sigmaIEtaIEtaThresholds_tmp, 2);
                  return;
              }

          case VBTF_35Xr2_60:
              {
                  double isoThresholds_tmp[2]                 = {0.03,     0.02};
                  double sigmaIEtaIEtaThresholds_tmp[2]       = {0.01,    0.03};
                  double dPhiInThresholds_tmp[2]              = {0.025,     0.02};
                  double dEtaInThresholds_tmp[2]              = {0.004,   0.005};
                  double hoeThresholds_tmp[2]                 = {0.025,     0.025};
                  eidAssign(cutreliso, isoThresholds_tmp, 2);
                  eidAssign(cutdeta, dEtaInThresholds_tmp, 2);
                  eidAssign(cutdphi, dPhiInThresholds_tmp, 2);
                  eidAssign(cuthoe, hoeThresholds_tmp, 2);
                  eidAssign(cutsee, sigmaIEtaIEtaThresholds_tmp, 2);
                  return;
              }

          case VBTF_80_NOHOEEND:
              {
                  double isoThresholds_tmp[2]                 = {0.07,     0.06};
                  double sigmaIEtaIEtaThresholds_tmp[2]       = {0.01,    0.03};
                  double dPhiInThresholds_tmp[2]              = {0.06,     0.03};
                  double dEtaInThresholds_tmp[2]              = {0.004,   0.007};            
                  double hoeThresholds_tmp[2]                 = {0.04,     9999.};
                  eidAssign(cutreliso, isoThresholds_tmp, 2);
                  eidAssign(cutdeta, dEtaInThresholds_tmp, 2);
                  eidAssign(cutdphi, dPhiInThresholds_tmp, 2);
                  eidAssign(cuthoe, hoeThresholds_tmp, 2);
                  eidAssign(cutsee, sigmaIEtaIEtaThresholds_tmp, 2);
                  return;
              }

          case VBTF_85_NOHOEEND:
              {
                  double isoThresholds_tmp[2]                 = { 0.09  ,  0.06  };
                  double sigmaIEtaIEtaThresholds_tmp[2]       = { 0.01  ,  0.03  };
                  double dPhiInThresholds_tmp[2]              = { 0.06  ,  0.04  };
                  double dEtaInThresholds_tmp[2]              = { 0.006 , 0.007  };
                  double hoeThresholds_tmp[2]                 = { 0.04  , 9999.  };
                  eidAssign(cutreliso, isoThresholds_tmp, 2);
                  eidAssign(cutdeta, dEtaInThresholds_tmp, 2);
                  eidAssign(cutdphi, dPhiInThresholds_tmp, 2);
                  eidAssign(cuthoe, hoeThresholds_tmp, 2);
                  eidAssign(cutsee, sigmaIEtaIEtaThresholds_tmp, 2);
                  return;
              }

          case VBTF_85:
              {
                  double isoThresholds_tmp[2]                 = { 0.09  ,  0.06  };
                  double sigmaIEtaIEtaThresholds_tmp[2]       = { 0.01  ,  0.03  };
                  double dPhiInThresholds_tmp[2]              = { 0.06  ,  0.04  };
                  double dEtaInThresholds_tmp[2]              = { 0.006 , 0.007  };
                  double hoeThresholds_tmp[2]                 = { 0.04  , 0.025  };
                  eidAssign(cutreliso, isoThresholds_tmp, 2);
                  eidAssign(cutdeta, dEtaInThresholds_tmp, 2);
                  eidAssign(cutdphi, dPhiInThresholds_tmp, 2);
                  eidAssign(cuthoe, hoeThresholds_tmp, 2);
                  eidAssign(cutsee, sigmaIEtaIEtaThresholds_tmp, 2);
                  return;
              }

          case VBTF_70_NOHOEEND:
              {
                  double isoThresholds_tmp[2]                 = {0.04,     0.03};
                  double sigmaIEtaIEtaThresholds_tmp[2]       = {0.01,    0.03};
                  double dPhiInThresholds_tmp[2]              = {0.03,     0.02};
                  double dEtaInThresholds_tmp[2]              = {0.004,   0.005};
                  double hoeThresholds_tmp[2]                 = {0.025,     9999.};
                  eidAssign(cutreliso, isoThresholds_tmp, 2);
                  eidAssign(cutdeta, dEtaInThresholds_tmp, 2);
                  eidAssign(cutdphi, dPhiInThresholds_tmp, 2);
                  eidAssign(cuthoe, hoeThresholds_tmp, 2);
                  eidAssign(cutsee, sigmaIEtaIEtaThresholds_tmp, 2);
                  return;
              }
          case VBTF_90_HLT:
              {
                  double isoThresholds_tmp[2]                 = {0.1,     0.07};
                  double sigmaIEtaIEtaThresholds_tmp[2]       = {0.01,    0.03};
                  double dPhiInThresholds_tmp[2]              = {0.15,    0.10};
                  double dEtaInThresholds_tmp[2]              = {0.007,   0.009};
                  double hoeThresholds_tmp[2]                 = {0.12,    0.10};
                  eidAssign(cutreliso, isoThresholds_tmp, 2);
                  eidAssign(cutdeta, dEtaInThresholds_tmp, 2);
                  eidAssign(cutdphi, dPhiInThresholds_tmp, 2);
                  eidAssign(cuthoe, hoeThresholds_tmp, 2);
                  eidAssign(cutsee, sigmaIEtaIEtaThresholds_tmp, 2);
                  return;
              }   

          case VBTF_90_HLT_CALOIDT_TRKIDVL:
              {
                  double isoThresholds_tmp[2]                 = {0.1,     0.07};
                  double sigmaIEtaIEtaThresholds_tmp[2]       = {0.01,    0.03};
                  double dPhiInThresholds_tmp[2]              = {0.15,    0.10};
                  double dEtaInThresholds_tmp[2]              = {0.007,   0.009};
                  double hoeThresholds_tmp[2]                 = {0.10,    0.075};
                  eidAssign(cutreliso, isoThresholds_tmp, 2);
                  eidAssign(cutdeta, dEtaInThresholds_tmp, 2);
                  eidAssign(cutdphi, dPhiInThresholds_tmp, 2);
                  eidAssign(cuthoe, hoeThresholds_tmp, 2);
                  eidAssign(cutsee, sigmaIEtaIEtaThresholds_tmp, 2);
                  return;
              }   

          case VBTF_95_NOHOEEND:
              {
                  double isoThresholds_tmp[2]                 = {0.15,    0.10};
                  double sigmaIEtaIEtaThresholds_tmp[2]       = {0.01,    0.03};
                  double dPhiInThresholds_tmp[2]              = {0.80,    0.70};
                  double dEtaInThresholds_tmp[2]              = {0.007,   0.01};
                  double hoeThresholds_tmp[2]                 = {0.15,    999.};
                  eidAssign(cutreliso, isoThresholds_tmp, 2);
                  eidAssign(cutdeta, dEtaInThresholds_tmp, 2);
                  eidAssign(cutdphi, dPhiInThresholds_tmp, 2);
                  eidAssign(cuthoe, hoeThresholds_tmp, 2);
                  eidAssign(cutsee, sigmaIEtaIEtaThresholds_tmp, 2);
                  return;
              }   

          default:
              edm::LogError("InvalidInput") << "[eidGetVBTF] ERROR! Invalid tightness level";
      }

      return;
  }

  void eidAssign(std::vector<double> &cutarr, double cutvals[], unsigned int size)
  {
      cutarr.clear();
      for (unsigned int i = 0; i < size; ++i) {
          cutarr.push_back(cutvals[i]);
      }
  }


  void eidAssign(std::vector<bool> &cutarr, bool cutvals[], unsigned int size)
  {
      cutarr.clear();
      for (unsigned int i = 0; i < size; ++i) {
          cutarr.push_back(cutvals[i]);
      }
  }

  void eidAssign(std::vector<int> &cutarr, int cutvals[], unsigned int size)
  {
      cutarr.clear();
      for (unsigned int i = 0; i < size; ++i) {
          cutarr.push_back(cutvals[i]);
      }
  }

}
