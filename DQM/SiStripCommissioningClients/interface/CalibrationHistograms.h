#ifndef DQM_SiStripCommissioningClients_CalibrationHistograms_H
#define DQM_SiStripCommissioningClients_CalibrationHistograms_H

#include "DQM/SiStripCommissioningClients/interface/CommissioningHistograms.h"

class DQMStore;

class CalibrationHistograms : virtual public CommissioningHistograms {
public:
  CalibrationHistograms(const edm::ParameterSet& pset, DQMStore*, const sistrip::RunType& task = sistrip::CALIBRATION);
  ~CalibrationHistograms() override;

  void histoAnalysis(bool debug) override;

  void printAnalyses() override;  // override

  void save(std::string& filename, uint32_t run_number = 0, std::string partitionName = "");

private:
  // Needed for the calibration-scan analysis
  float targetRiseTime_;
  float targetDecayTime_;
  bool tuneSimultaneously_;
};

#endif  // DQM_SiStripCommissioningClients_CalibrationHistograms_H
