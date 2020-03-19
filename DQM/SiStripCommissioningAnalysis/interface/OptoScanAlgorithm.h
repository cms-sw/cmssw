#ifndef DQM_SiStripCommissioningAnalysis_OptoScanAlgorithm_H
#define DQM_SiStripCommissioningAnalysis_OptoScanAlgorithm_H

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DQM/SiStripCommissioningAnalysis/interface/CommissioningAlgorithm.h"
#include <vector>
#include <cstdint>

class OptoScanAnalysis;
class TProfile;
class TH1;

/** 
   @class OptoScanAnalysis
   @author M. Wingham, R.Bainbridge
   @brief Histogram-based analysis for opto bias/gain scan.
*/
class OptoScanAlgorithm : public CommissioningAlgorithm {
public:
  OptoScanAlgorithm(const edm::ParameterSet& pset, OptoScanAnalysis* const);

  ~OptoScanAlgorithm() override { ; }

  /** Histogram pointer and title. */
  Histo histo(const uint16_t& gain, const uint16_t& digital_level) const;

private:
  OptoScanAlgorithm() { ; }

  /** Extracts and organises histograms. */
  void extract(const std::vector<TH1*>&) override;

  /** Performs histogram anaysis. */
  void analyse() override;

private:
  /** Pointers and titles for histograms. */
  std::vector<std::vector<Histo> > histos_;

  /** Analysis parameters */
  float targetGain_;
};

#endif  // DQM_SiStripCommissioningAnalysis_OptoScanAlgorithm_H
