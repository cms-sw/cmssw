#ifndef RecoLocalCalo_HcalRecAlgos_PulseShapeFunctor_h
#define RecoLocalCalo_HcalRecAlgos_PulseShapeFunctor_h

#include "CalibCalorimetry/HcalAlgos/interface/HcalPulseShapes.h"
#include "RecoLocalCalo/HcalRecAlgos/interface/HcalConstants.h"

namespace FitterFuncs {

  class PulseShapeFunctor {
  public:
    PulseShapeFunctor(const HcalPulseShapes::Shape &pulse,
                      bool iPedestalConstraint,
                      bool iTimeConstraint,
                      bool iAddPulseJitter,
                      double iPulseJitter,
                      double iTimeMean,
                      double iPedMean,
                      unsigned int nSamplesToFit);
    ~PulseShapeFunctor();

    void EvalPulse(const float *pars);
    double EvalPulseM2(const double *pars, const unsigned nPar);

    void setDefaultcntNANinfit() { cntNANinfit = 0; }
    int getcntNANinfit() { return cntNANinfit; }

    void setpsFitx(double *x) {
      for (int i = 0; i < hcal::constants::maxSamples; ++i)
        psFit_x[i] = x[i];
    }
    void setpsFity(double *y) {
      for (int i = 0; i < hcal::constants::maxSamples; ++i)
        psFit_y[i] = y[i];
    }
    void setpsFiterry(double *erry) {
      for (int i = 0; i < hcal::constants::maxSamples; ++i)
        psFit_erry[i] = erry[i];
    }
    void setpsFiterry2(double *erry2) {
      for (int i = 0; i < hcal::constants::maxSamples; ++i)
        psFit_erry2[i] = erry2[i];
    }
    void setpsFitslew(double *slew) {
      for (int i = 0; i < hcal::constants::maxSamples; ++i) {
        psFit_slew[i] = slew[i];
      }
    }
    double getSiPMDarkCurrent(double darkCurrent, double fcByPE, double lambda);
    void setinvertpedSig2(double x) { invertpedSig2_ = x; }
    void setinverttimeSig2(double x) { inverttimeSig2_ = x; }

    inline void singlePulseShapeFuncMahi(const float *x) { return EvalPulse(x); }
    inline double singlePulseShapeFunc(const double *x) { return EvalPulseM2(x, 3); }
    inline double doublePulseShapeFunc(const double *x) { return EvalPulseM2(x, 5); }
    inline double triplePulseShapeFunc(const double *x) { return EvalPulseM2(x, 7); }

    void getPulseShape(std::array<double, hcal::constants::maxSamples> &fillPulseShape) {
      fillPulseShape = pulse_shape_;
    }

    // getters
    inline std::vector<float> const &acc25nsVec() const { return acc25nsVec_; }
    inline std::vector<float> const &diff25nsItvlVec() const { return diff25nsItvlVec_; }
    inline std::vector<float> const &accVarLenIdxZEROVec() const { return accVarLenIdxZEROVec_; }
    inline std::vector<float> const &diffVarItvlIdxZEROVec() const { return diffVarItvlIdxZEROVec_; }
    inline std::vector<float> const &accVarLenIdxMinusOneVec() const { return accVarLenIdxMinusOneVec_; }
    inline std::vector<float> const &diffVarItvlIdxMinusOneVec() const { return diffVarItvlIdxMinusOneVec_; }

  private:
    std::array<float, hcal::constants::maxPSshapeBin> pulse_hist;

    int cntNANinfit;
    std::vector<float> acc25nsVec_, diff25nsItvlVec_;
    std::vector<float> accVarLenIdxZEROVec_, diffVarItvlIdxZEROVec_;
    std::vector<float> accVarLenIdxMinusOneVec_, diffVarItvlIdxMinusOneVec_;

    void funcShape(std::array<double, hcal::constants::maxSamples> &ntmpbin,
                   const double pulseTime,
                   const double pulseHeight,
                   const double slew,
                   bool scalePulse);
    double psFit_x[hcal::constants::maxSamples], psFit_y[hcal::constants::maxSamples],
        psFit_erry[hcal::constants::maxSamples], psFit_erry2[hcal::constants::maxSamples],
        psFit_slew[hcal::constants::maxSamples];

    unsigned nSamplesToFit_;
    bool pedestalConstraint_;
    bool timeConstraint_;
    bool addPulseJitter_;
    bool unConstrainedFit_;
    double pulseJitter_;
    double timeMean_;
    double timeSig_;
    double pedMean_;
    double timeShift_;

    double inverttimeSig2_;
    double invertpedSig2_;
    std::array<double, hcal::constants::maxSamples> pulse_shape_;
    std::array<double, hcal::constants::maxSamples> pulse_shape_sum_;
  };

}  // namespace FitterFuncs

#endif  // PulseShapeFunctor_h
