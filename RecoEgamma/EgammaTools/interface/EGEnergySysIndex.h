#ifndef RecoEgamma_EgammaTools_EGEnergySysIndex_h
#define RecoEgamma_EgammaTools_EGEnergySysIndex_h

//Currently the Electron and Photon energy calibrators return a big vector of calculated
//energy and systematic values
//this class allows the elements to be indexed easier
//ultimately should become a class

#include <array>

class EGEnergySysIndex {
public:
  enum Index {
    kScaleStatUp=0,
    kScaleStatDown,
    kScaleSystUp,
    kScaleSystDown,
    kScaleGainUp,
    kScaleGainDown,
    kSmearRhoUp,
    kSmearRhoDown,
    kSmearPhiUp,
    kSmearPhiDown,
    kScaleUp,
    kScaleDown,
    kSmearUp,
    kSmearDown,
    kScaleValue,
    kSmearValue,
    kSmearNrSigma,
    kEcalPreCorr,
    kEcalErrPreCorr,
    kEcalPostCorr,
    kEcalErrPostCorr,
    kEcalTrkPreCorr,
    kEcalTrkErrPreCorr,
    kEcalTrkPostCorr,
    kEcalTrkErrPostCorr
  };
  static constexpr size_t kNrSysErrs=kEcalTrkErrPostCorr+1; 

  static const std::string& name(size_t index){return names_[index];}

private:
  static const std::array<std::string,kNrSysErrs> names_;


   
};

#endif
