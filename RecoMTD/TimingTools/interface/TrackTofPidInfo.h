#ifndef RecoMTD_TimingTools_TrackTofPidInfo_h
#define RecoMTD_TimingTools_TrackTofPidInfo_h

#include "RecoMTD/TimingTools/interface/TrackSegments.h"

namespace mtd {

  struct TrackTofPidInfo {
    float tmtd;
    float tmtderror;
    float pathlength;

    float betaerror;

    float dt;
    float dterror;
    float dterror2;
    float dtchi2;

    float dt_best;
    float dterror_best;
    float dtchi2_best;

    float gammasq_pi;
    float beta_pi;
    float dt_pi;
    float sigma_dt_pi;

    float gammasq_k;
    float beta_k;
    float dt_k;
    float sigma_dt_k;

    float gammasq_p;
    float beta_p;
    float dt_p;
    float sigma_dt_p;

    float prob_pi;
    float prob_k;
    float prob_p;
  };

  enum class TofCalc { kCost = 1, kSegm = 2, kMixd = 3 };
  enum class SigmaTofCalc { kCost = 1, kSegm = 2, kMixd = 3 };

  const TrackTofPidInfo computeTrackTofPidInfo(float magp2,
                                               float length,
                                               TrackSegments trs,
                                               float t_mtd,
                                               float t_mtderr,
                                               float t_vtx,
                                               float t_vtx_err,
                                               bool addPIDError = true,
                                               TofCalc choice = TofCalc::kCost,
                                               SigmaTofCalc sigma_choice = SigmaTofCalc::kCost);

}  // namespace mtd

#endif
