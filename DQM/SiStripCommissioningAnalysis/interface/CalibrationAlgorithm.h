#ifndef DQM_SiStripCommissioningAnalysis_CalibrationAlgorithm_H
#define DQM_SiStripCommissioningAnalysis_CalibrationAlgorithm_H

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DQM/SiStripCommissioningAnalysis/interface/CommissioningAlgorithm.h"
#include <vector>
#include <map>

class CalibrationAnalysis;
class TH1;
class TF1;

/**
   @class CalibrationAlgorithm
   @author C. Delaere
   @brief Algorithm for calibration runs
*/

class CalibrationAlgorithm : public CommissioningAlgorithm {
public:
  CalibrationAlgorithm(const edm::ParameterSet& pset, CalibrationAnalysis* const);
  ~CalibrationAlgorithm() override { ; }

  inline const Histo& histo(int& i) { return histo_[i]; }

private:
  CalibrationAlgorithm() { ; }

  void extract(const std::vector<TH1*>&) override;

  void analyse() override;

  void correctDistribution(TH1*) const;

  float baseLine(TF1*);
  float turnOn(TF1*, const float&);
  float decayTime(TF1*);

private:
  /** pulse shape*/
  std::vector<Histo> histo_;
  std::vector<int> stripId_;
  std::vector<int> calChan_;
  std::vector<int> apvId_;

  /** analysis object */
  CalibrationAnalysis* cal_;
};

#endif  // DQM_SiStripCommissioningAnalysis_CalibrationAlgorithm_H
