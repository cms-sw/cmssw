#include <iostream>
#include <cmath>
#include <climits>
#include "RecoLocalCalo/HcalRecAlgos/interface/PedestalSub.h"

using namespace std;

PedestalSub::PedestalSub() {}

PedestalSub::~PedestalSub() {}

void PedestalSub::calculate(const std::vector<double>& inputCharge,
                            const std::vector<double>& inputPedestal,
                            const std::vector<double>& inputNoise,
                            std::vector<double>& corrCharge,
                            int soi,
                            int nTS) const {
  double bseCorr = PedestalSub::getCorrection(inputCharge, inputPedestal, inputNoise, soi, nTS);
  for (auto i = 0; i < nTS; i++) {
    corrCharge.push_back(inputCharge[i] - inputPedestal[i] - bseCorr);
  }
}

double PedestalSub::getCorrection(const std::vector<double>& inputCharge,
                                  const std::vector<double>& inputPedestal,
                                  const std::vector<double>& inputNoise,
                                  int soi,
                                  int nTS) const {
  double baseline = 0;

  for (auto i = 0; i < nTS; i++) {
    if (i == soi || i == (soi + 1))
      continue;
    if ((inputCharge[i] - inputPedestal[i]) < 3 * inputNoise[i]) {
      baseline += (inputCharge[i] - inputPedestal[i]);
    } else {
      baseline += 3 * inputNoise[i];
    }
  }
  baseline /= (nTS - 2);
  return baseline;
}
