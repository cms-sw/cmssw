#ifndef PhysicsTools_Utilities_interface_LumiReWeighting_h
#define PhysicsTools_Utilities_interface_LumiReWeighting_h

/**
  \class    LumiReWeighting LumiReWeighting.h "PhysicsTools/Utilities/interface/LumiReWeighting.h"
  \brief    Class to provide lumi weighting for analyzers to weight "flat-to-N" MC samples to data

  This class will trivially take two histograms:
  1. The generated "flat-to-N" distributions from a given processing
  2. A histogram generated from the "estimatePileup" macro here:

  https://twiki.cern.ch/twiki/bin/view/CMS/LumiCalc#How_to_use_script_estimatePileup

  \author Salvatore Rappoccio
*/

#include "TH1.h"
#include "TFile.h"
#include <cmath>
#include <string>

#include <vector>
#include "FWCore/Utilities/interface/InputTag.h"

namespace reweight {

  // add a class to shift the mean of a poisson-like luminosity distribution by an arbitrary amount.
  // Only valid for small (<1.5) shifts about the 2011 lumi distribution for now, because shifts are non-linear
  // Each PoissonMeanShifter does one shift, so defining multiples can give you an arbitrary collection

  class PoissonMeanShifter {
  public:
    PoissonMeanShifter(){};

    PoissonMeanShifter(float Shift) {
      // these are the polynomial or exponential coefficients for each bin of a 25-bin sequence that
      // convert the Distribution of the 2011 luminosity to something with a lower or higher peak luminosity.
      // The distributions aren't quite poisson because they model luminosity decreasing during a fill. This implies that
      // they do get wider as the mean increases, so the weights are not linear with increasing mean.

      constexpr double p0_minus[20] = {1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.};
      constexpr double p1_minus[20] = {-0.677786, -0.619614, -0.49465, -0.357963, -0.238359, -0.110002, 0.0348629,
                                       0.191263,  0.347648,  0.516615, 0.679646,  0.836673,  0.97764,   1.135,
                                       1.29922,   1.42467,   1.55901,  1.61762,   1.67275,   1.96008};
      constexpr double p2_minus[20] = {0.526164,   0.251816,   0.11049,   0.026917, -0.0464692, -0.087022, -0.0931581,
                                       -0.0714295, -0.0331772, 0.0347473, 0.108658, 0.193048,   0.272314,  0.376357,
                                       0.4964,     0.58854,    0.684959,  0.731063, 0.760044,   1.02386};

      constexpr double p1_expoM[5] = {1.63363e-03, 6.79290e-04, 3.69900e-04, 2.24349e-04, 9.87156e-06};

      constexpr double p2_expoM[5] = {2.64692, 3.26585, 3.53229, 4.18035, 5.64027};

      constexpr double p0_plus[20] = {1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.};
      constexpr double p1_plus[20] = {-0.739059, -0.594445, -0.477276, -0.359707, -0.233573, -0.103458, 0.0373401,
                                      0.176571,  0.337617,  0.499074,  0.675126,  0.840522,  1.00917,   1.15847,
                                      1.23816,   1.44271,   1.52982,   1.46385,   1.5802,    0.988689};
      constexpr double p2_plus[20] = {0.208068,    0.130033,   0.0850356,  0.0448344,  0.000749832,
                                      -0.0331347,  -0.0653281, -0.0746009, -0.0800667, -0.0527636,
                                      -0.00402649, 0.103338,   0.261261,   0.491084,   0.857966,
                                      1.19495,     1.75071,    2.65559,    3.35433,    5.48835};

      constexpr double p1_expoP[5] = {1.42463e-01, 4.18966e-02, 1.12697e-01, 1.66197e-01, 1.50768e-01};

      constexpr double p2_expoP[5] = {1.98758, 2.27217, 2.26799, 2.38455, 2.52428};

      // initialize weights based on desired Shift

      for (int ibin = 0; ibin < 20; ibin++) {
        if (Shift < .0) {
          Pweight_[ibin] = p0_minus[ibin] + p1_minus[ibin] * Shift + p2_minus[ibin] * Shift * Shift;
        } else {
          Pweight_[ibin] = p0_plus[ibin] + p1_plus[ibin] * Shift + p2_plus[ibin] * Shift * Shift;
        }
      }

      // last few bins fit better to an exponential...

      for (int ibin = 20; ibin < 25; ibin++) {
        if (Shift < 0.) {
          Pweight_[ibin] = p1_expoM[ibin - 20] * exp(p2_expoM[ibin - 20] * Shift);
        } else {
          Pweight_[ibin] = p1_expoP[ibin - 20] * exp(p2_expoP[ibin - 20] * Shift);
        }
      }
    };

    double ShiftWeight(int ibin) {
      if (ibin < 25 && ibin >= 0) {
        return Pweight_[ibin];
      } else {
        return 0;
      }
    };

    double ShiftWeight(float pvnum) { return ShiftWeight(int(pvnum)); };

  private:
    double Pweight_[25];
  };

}  // namespace reweight

namespace edm {
  class EventBase;
  class LumiReWeighting {
  public:
    LumiReWeighting(std::string generatedFile,
                    std::string dataFile,
                    std::string GenHistName = "pileup",
                    std::string DataHistName = "pileup",
                    const edm::InputTag& PileupSumInfoInputTag = edm::InputTag("addPileupInfo"));

    LumiReWeighting(const std::vector<float>& MC_distr,
                    const std::vector<float>& Lumi_distr,
                    const edm::InputTag& PileupSumInfoInputTag = edm::InputTag("addPileupInfo"));

    LumiReWeighting(){};

    double weight(int npv);

    double weight(float npv);

    double weight(const edm::EventBase& e);

    double weightOOT(const edm::EventBase& e);

  protected:
    std::string generatedFileName_;
    std::string dataFileName_;
    std::string GenHistName_;
    std::string DataHistName_;
    edm::InputTag pileupSumInfoTag_;
    std::shared_ptr<TFile> generatedFile_;
    std::shared_ptr<TFile> dataFile_;
    std::shared_ptr<TH1> weights_;

    //keep copies of normalized distributions:

    std::shared_ptr<TH1> MC_distr_;
    std::shared_ptr<TH1> Data_distr_;

    int OldLumiSection_;
    bool FirstWarning_;
  };
}  // namespace edm

#endif
