#ifndef DQM_SiStripCommissioningAnalysis_ApvTimingAlgorithm_H
#define DQM_SiStripCommissioningAnalysis_ApvTimingAlgorithm_H

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DQM/SiStripCommissioningAnalysis/interface/CommissioningAlgorithm.h"
#include <vector>

class ApvTimingAnalysis;
class TH1;

/**
   @class ApvTimingAlgorithm
   @author M. Wingham, R.Bainbridge
   @brief Analysis for timing run using APV tick marks.
*/
class ApvTimingAlgorithm : public CommissioningAlgorithm {
public:
  ApvTimingAlgorithm(const edm::ParameterSet& pset, ApvTimingAnalysis* const);

  ~ApvTimingAlgorithm() override { ; }

  /** Container of histogram pointer and title. */
  inline const Histo& histo() const;

private:
  /** Private constructor. */
  ApvTimingAlgorithm() { ; }

  /** Extracts and organises histograms. */
  void extract(const std::vector<TH1*>&) override;

  /** Performs histogram anaysis. */
  void analyse() override;

private:
  /** Container of histogram pointer and title. */
  Histo histo_;
};

// ---------- Inline methods ----------

const ApvTimingAlgorithm::Histo& ApvTimingAlgorithm::histo() const { return histo_; }

#endif  // DQM_SiStripCommissioningAnalysis_ApvTimingAlgorithm_H
