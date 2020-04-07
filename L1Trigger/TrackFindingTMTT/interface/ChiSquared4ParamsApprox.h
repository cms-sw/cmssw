#ifndef L1Trigger_TrackFindingTMTT_ChiSquared4ParamsApprox_h
#define L1Trigger_TrackFindingTMTT_ChiSquared4ParamsApprox_h

#include "L1Trigger/TrackFindingTMTT/interface/L1ChiSquared.h"

namespace tmtt {

  class ChiSquared4ParamsApprox : public L1ChiSquared {
  public:
    ChiSquared4ParamsApprox(const Settings* settings, const uint nPar);

    ~ChiSquared4ParamsApprox() {}

  protected:
    std::vector<double> seed(const L1track3D& l1track3D);
    std::vector<double> residuals(std::vector<double> x);
    Matrix<double> D(std::vector<double> x);
    Matrix<double> Vinv();
    std::map<std::string, double> convertParams(std::vector<double> x);

  private:
    std::vector<double> mapToVec(std::map<std::string, double> x);
    std::map<std::string, double> vecToMap(std::vector<double> x);
  };

}  // namespace tmtt

#endif
