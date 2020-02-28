#ifndef DQM_SiStripCommissioningAnalysis_CalibrationScanAlgorithm_H
#define DQM_SiStripCommissioningAnalysis_CalibrationScanAlgorithm_H

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DQM/SiStripCommissioningAnalysis/interface/CommissioningAlgorithm.h"
#include <vector>
#include <map>
#include "TGraph.h"

class CalibrationScanAnalysis;
class TH1;
class TF1;

/**
   @class CalibrationScanAlgorithm
   @author C. Delaere
   @brief Algorithm for calibration runs
*/

class CalibrationScanAlgorithm : public CommissioningAlgorithm {
public:
  CalibrationScanAlgorithm(const edm::ParameterSet &pset, CalibrationScanAnalysis *const);
  ~CalibrationScanAlgorithm() override { ; }

  inline const Histo &histo(std::string &key, int &i) { return histo_[key][i]; }

  void tuneIndependently(const int &, const float &, const float &);
  void tuneSimultaneously(const int &, const float &, const float &);
  void fillTunedObservables(const int &);

private:
  CalibrationScanAlgorithm() { ; }

  void extract(const std::vector<TH1 *> &) override;

  void analyse() override;

  void correctDistribution(TH1 *, const bool &) const;
  float baseLine(TF1 *);
  float turnOn(TF1 *, const float &);
  float decayTime(TF1 *);

  /** pulse shape*/
  std::map<std::string, std::vector<Histo> > histo_;

  /** analysis object */
  CalibrationScanAnalysis *cal_;

  /** values of the scanned isha and vfs **/
  std::vector<int> scanned_isha_;
  std::vector<int> scanned_vfs_;
};

#endif  // DQM_SiStripCommissioningAnalysis_CalibrationScanAlgorithm_H
