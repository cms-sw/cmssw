#ifndef CalibCalorimetry_HcalAlgos_AbsElectronicODERHS_h_
#define CalibCalorimetry_HcalAlgos_AbsElectronicODERHS_h_

#include <vector>
#include <climits>
#include <algorithm>

#include "CalibCalorimetry/HcalAlgos/interface/AbsODERHS.h"
#include "CondFormats/HcalObjects/interface/HcalInterpolatedPulse.h"

//
// Modeling of electronic circuits always involves an input pulse that
// determines the circuit output. This class adds an input pulse to
// AbsODERHS and establishes a uniform interface to circuit parameters.
//
class AbsElectronicODERHS : public AbsODERHS {
public:
  static const unsigned invalidNode = UINT_MAX - 1U;

  inline AbsElectronicODERHS() : initialized_(false), allSet_(false) {}

  inline explicit AbsElectronicODERHS(const HcalInterpolatedPulse& pulse) : inputPulse_(pulse) {}

  inline ~AbsElectronicODERHS() override {}

  inline const HcalInterpolatedPulse& inputPulse() const { return inputPulse_; }

  inline HcalInterpolatedPulse& inputPulse() { return inputPulse_; }

  template <class Pulse>
  inline void setInputPulse(const Pulse& pulse) {
    inputPulse_ = pulse;
  }

  // The following methods must be overriden by derived classes.
  // Total number of nodes included in the simulation:
  virtual unsigned numberOfNodes() const = 0;

  // The node which counts as "output" (preamp output in case
  // of QIE8, which is not necessarily the node which accumulates
  // the charge):
  virtual unsigned outputNode() const = 0;

  // The node which counts as "control". If this method returns
  // "invalidNode" then there is no such node in the circuit.
  virtual unsigned controlNode() const { return invalidNode; }

  // The number of simulation parameters:
  virtual unsigned nParameters() const = 0;

  // Check if all parameters have been set
  inline bool allParametersSet() const {
    // Raise "allSet_" flag if all parameters have been set
    if (!allSet_) {
      const unsigned nExpected = this->nParameters();
      if (nExpected) {
        if (paramMask_.size() != nExpected)
          return false;
        unsigned count = 0;
        const unsigned char* mask = &paramMask_[0];
        for (unsigned i = 0; i < nExpected; ++i)
          count += mask[i];
        allSet_ = count == nExpected;
      } else
        allSet_ = true;
    }
    return allSet_;
  }

  inline void setParameter(const unsigned which, const double value) {
    if (!initialized_)
      initialize();
    paramMask_.at(which) = 1;
    params_[which] = value;
  }

  inline double getParameter(const unsigned which) const {
    if (!paramMask_.at(which))
      throw cms::Exception(
          "In AbsElectronicODERHS::getParameter: no such parameter or "
          "parameter value is not established yet");
    return params_[which];
  }

  inline const std::vector<double>& getAllParameters() const {
    if (!allParametersSet())
      throw cms::Exception(
          "In AbsElectronicODERHS::getAllParameters: "
          "some parameter values were not established yet");
    return params_;
  }

  inline void setLeadingParameters(const double* values, const unsigned len) {
    if (len) {
      assert(values);
      if (!initialized_)
        initialize();
      const unsigned sz = params_.size();
      const unsigned imax = std::min(sz, len);
      for (unsigned i = 0; i < imax; ++i) {
        params_[i] = values[i];
        paramMask_[i] = 1;
      }
    }
  }

  inline void setLeadingParameters(const std::vector<double>& values) {
    if (!values.empty())
      setLeadingParameters(&values[0], values.size());
  }

protected:
  HcalInterpolatedPulse inputPulse_;
  std::vector<double> params_;

private:
  std::vector<unsigned char> paramMask_;
  bool initialized_;
  mutable bool allSet_;

  inline void initialize() {
    const unsigned nExpected = this->nParameters();
    if (nExpected) {
      params_.resize(nExpected);
      paramMask_.resize(nExpected);
      for (unsigned i = 0; i < nExpected; ++i)
        paramMask_[i] = 0;
    }
    initialized_ = true;
  }
};

#endif  // CalibCalorimetry_HcalAlgos_AbsElectronicODERHS_h_
