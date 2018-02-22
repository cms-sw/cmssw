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
   
};
