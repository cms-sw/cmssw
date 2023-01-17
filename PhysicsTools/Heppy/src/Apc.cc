#include "PhysicsTools/Heppy/interface/Apc.h"
#include "DataFormats/Math/interface/deltaPhi.h"
#include "DataFormats/Math/interface/deltaR.h"

#include <cmath>
#include <numeric>
#include <vector>
#include <algorithm>
#include <cstdlib>

namespace heppy {

  double Apc::getApcJetMetMin(const std::vector<double>& et,
                              const std::vector<double>& px,
                              const std::vector<double>& py,
                              const double metx,
                              const double mety) {
    if (et.empty())
      return -1.;

    // Momentum sums in transverse plane
    const double ht = accumulate(et.begin(), et.end(), 0.);

    // jets are pt-sorted
    double apcjetmetmin(0.);

    std::vector<double> apcjetvector;
    std::vector<double> apcjetmetvector;
    for (size_t j = 0; j < et.size(); j++) {
      apcjetvector.push_back(0.);
      apcjetmetvector.push_back(0.);
      double jet_phi_j = atan2(py[j], px[j]);
      for (size_t i = 0; i < et.size(); i++) {
        double jet_phi_i = atan2(py[i], px[i]);
        double dphi_jet = fabs(deltaPhi(jet_phi_i, jet_phi_j));
        double met_phi = atan2(mety, metx);
        double dphimet = fabs(deltaPhi(jet_phi_i, met_phi));

        apcjetvector.back() += et[i] * cos(dphi_jet / 2.0);
        apcjetmetvector.back() += et[i] * cos(dphi_jet / 2.0) * sin(dphimet / 2.0);
      }
    }
    if (!apcjetvector.empty() && !apcjetmetvector.empty()) {
      apcjetmetmin = *min_element(apcjetmetvector.begin(), apcjetmetvector.end());
    }

    if (ht != 0)
      return apcjetmetmin / ht;
    else
      return -1.;
  }
}  // namespace heppy
