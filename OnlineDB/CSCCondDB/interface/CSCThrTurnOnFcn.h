#ifndef CSCCalibration_CSCThrTurnOnFcn_h
#define CSCCalibration_CSCThrTurnOnFcn_h

/** \class CSCThrTurnOnFcn
 *
 * Model functional form for fitting AFEB turn-on threshold
 * information from Muon Endcap CSC's. This version
 * is for ROOT Minuit2. Based on CSCPulseHeightFcn  as an example.
 *
 */

#include "Minuit2/FCNBase.h"
#include <vector>
#include <RVersion.h>
#if ROOT_VERSION_CODE >= ROOT_VERSION(6, 33, 1)
#include <span>
#endif

class CSCThrTurnOnFcn : public ROOT::Minuit2::FCNBase {
private:
  ///  data
  std::vector<float> xdata;
  std::vector<float> ydata;
  std::vector<float> ery;
  float norm;

public:
  ///Cache the current data, x and y
  void setData(const std::vector<float>& x, const std::vector<float>& y) {
    for (unsigned int i = 0; i < x.size(); i++) {
      xdata.push_back(x[i]);
      ydata.push_back(y[i]);
    }
  };

  /// Set the errors
  void setErrors(const std::vector<float>& er) {
    for (unsigned int i = 0; i < er.size(); i++)
      ery.push_back(er[i]);
  };

  /// Set the norm (if needed)
  void setNorm(float n) { norm = n; };

  /// Provide the chi-squared function for the given data
#if ROOT_VERSION_CODE >= ROOT_VERSION(6, 33, 1)
  double operator()(std::span<const double>) const override;
#else
  double operator()(const std::vector<double>&) const override;
#endif

  ///@@ What?
  double Up() const override { return 1.; }
};

#endif
