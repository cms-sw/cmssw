#ifndef RecoLocalCalo_EcalRecAlgos_EcalUncalibRecHitRecAnalFitAlgo_HH
#define RecoLocalCalo_EcalRecAlgos_EcalUncalibRecHitRecAnalFitAlgo_HH

/** \class EcalUncalibRecHitRecAnalFitAlgo
  *  Template used to compute amplitude, pedestal, time jitter, chi2 of a pulse
  *  using an analytical fit
  *
  *  \author A. Palma, Sh. Rahatlou Roma1
  */

#include "CLHEP/Matrix/Matrix.h"
#include "CLHEP/Matrix/SymMatrix.h"
#include "RecoLocalCalo/EcalRecAlgos/interface/EcalUncalibRecHitRecAbsAlgo.h"
#include <vector>
#include <string>

#include "TROOT.h"
#include "TGraph.h"
#include "TF1.h"

#include "TMinuitMinimizer.h"

template <class C>
class EcalUncalibRecHitRecAnalFitAlgo : public EcalUncalibRecHitRecAbsAlgo<C> {
private:
  double pulseShapeFunction(double* var, double* par) {
    double x = var[0];
    double ampl = par[0];
    double tp = par[1];
    double alpha = par[2];
    double t0 = par[3];

    double f = pow((x - t0) / tp, alpha) * exp(-alpha * (x - tp - t0) / tp);
    return ampl * f;
  };

  double pedestalFunction(double* var, double* par) {
    double ped = par[0];
    return ped;
  };

public:
  EcalUncalibRecHitRecAnalFitAlgo() {
    //In order to make fitting ROOT histograms thread safe
    // one must call this undocumented function
    TMinuitMinimizer::UseStaticMinuit(false);
  }

  // destructor
  ~EcalUncalibRecHitRecAnalFitAlgo() override{};

  /// Compute parameters
  EcalUncalibratedRecHit makeRecHit(const C& dataFrame,
                                    const double* pedestals,
                                    const double* gainRatios,
                                    const EcalWeightSet::EcalWeightMatrix** weights,
                                    const EcalWeightSet::EcalChi2WeightMatrix** chi2Matrix) override {
    double amplitude_(-1.), pedestal_(-1.), jitter_(-1.), chi2_(-1.);

    // Get time samples
    //HepMatrix frame(C::MAXSAMPLES, 1);
    double frame[C::MAXSAMPLES];
    //    int gainId0 = dataFrame.sample(0).gainId();
    int gainId0 = 1;
    int iGainSwitch = 0;
    double maxsample(-1);
    int imax(-1);
    bool isSaturated = false;
    uint32_t flag = 0;
    for (int iSample = 0; iSample < C::MAXSAMPLES; iSample++) {
      int gainId = dataFrame.sample(iSample).gainId();
      if (dataFrame.isSaturated()) {
        gainId = 3;
        isSaturated = true;
      }

      if (gainId != gainId0)
        ++iGainSwitch;
      if (!iGainSwitch)
        frame[iSample] = double(dataFrame.sample(iSample).adc());
      else
        frame[iSample] =
            double(((double)(dataFrame.sample(iSample).adc()) - pedestals[gainId - 1]) * gainRatios[gainId - 1]);

      if (frame[iSample] > maxsample) {
        maxsample = frame[iSample];
        imax = iSample;
      }
    }

    // Compute parameters
    //std::cout << "EcalUncalibRecHitRecAnalFitAlgo::makeRecHit() not yey implemented. returning dummy rechit" << std::endl;

    // prepare TGraph for analytic fit
    double xarray[10] = {0., 1., 2., 3., 4., 5., 6., 7., 8., 9.};
    TGraph graph(10, xarray, frame);

    // fit functions
    TF1 pulseShape = TF1("pulseShape",
                         "[0]*pow((x - [3])/[1],[2])*exp(-[2]*(x - [1] - [3])/[1])",
                         imax - 1.,
                         imax + 3.,
                         TF1::EAddToList::kNo);
    TF1 pedestal = TF1("pedestal", "[0]", 0., 2., TF1::EAddToList::kNo);

    //pulseShape parameters
    // Amplitude
    double FIT_A = (double)maxsample;  //Amplitude
    pulseShape.SetParameter(0, FIT_A);
    pulseShape.SetParName(0, "Amplitude");
    // T peak
    double FIT_Tp = (double)imax;  //T peak
    pulseShape.SetParameter(1, FIT_Tp);
    pulseShape.SetParName(1, "t_{P}");
    // Alpha
    double FIT_ALFA = 1.5;  //Alpha
    pulseShape.SetParameter(2, FIT_ALFA);
    pulseShape.SetParName(2, "\\alpha");
    // T off
    double FIT_To = 3.;  //T off
    pulseShape.SetParameter(3, FIT_To);
    pulseShape.SetParName(3, "t_{0}");

    // pedestal
    pedestal.SetParameter(0, frame[0]);
    pedestal.SetParName(0, "Pedestal");

    int result = graph.Fit(&pulseShape, "QRMN SERIAL");

    if (0 == result) {
      double amplitude_value = pulseShape.GetParameter(0);

      graph.Fit(&pedestal, "QRLN SERIAL");
      double pedestal_value = pedestal.GetParameter(0);

      if (!iGainSwitch)
        amplitude_ = amplitude_value - pedestal_value;
      else
        amplitude_ = amplitude_value;

      pedestal_ = pedestal_value;
      jitter_ = pulseShape.GetParameter(3);
      chi2_ = 1.;  // successful fit
      if (isSaturated)
        flag = EcalUncalibratedRecHit::kSaturated;
      /*
      std::cout << "separate fits\nA: " <<  amplitude_value << ", Ped: " << pedestal_value
                << ", t0: " << jitter_ << ", tp: " << pulseShape.GetParameter(1)
                << ", alpha: " << pulseShape.GetParameter(2)
                << std::endl;
      */
    }

    return EcalUncalibratedRecHit(dataFrame.id(), amplitude_, pedestal_, jitter_ - 6, chi2_, flag);
  }
};
#endif
