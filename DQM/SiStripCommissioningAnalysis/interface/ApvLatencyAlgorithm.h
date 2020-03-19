#ifndef DQM_SiStripCommissioningAnalysis_ApvLatencyAlgorithm_H
#define DQM_SiStripCommissioningAnalysis_ApvLatencyAlgorithm_H

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DQM/SiStripCommissioningAnalysis/interface/CommissioningAlgorithm.h"
#include <vector>

class ApvLatencyAnalysis;
class TH1;

/** 
   @class ApvLatencyAlgorithm
   @author M. Wingham, R.Bainbridge
   @brief Algorithm for APV latency scan.
*/
class ApvLatencyAlgorithm : public CommissioningAlgorithm {
public:
  ApvLatencyAlgorithm(const edm::ParameterSet& pset, ApvLatencyAnalysis* const);

  ~ApvLatencyAlgorithm() override { ; }

  inline const Histo& histo() const;

private:
  ApvLatencyAlgorithm() { ; }

  void extract(const std::vector<TH1*>&) override;

  void analyse() override;

private:
  /** APV latency histo */
  Histo histo_;
};

const ApvLatencyAlgorithm::Histo& ApvLatencyAlgorithm::histo() const { return histo_; }

#endif  // DQM_SiStripCommissioningAnalysis_ApvLatencyAlgorithm_H
