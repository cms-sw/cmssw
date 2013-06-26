#ifndef DQM_SiStripCommissioningAnalysis_VpspScanAlgorithm_H
#define DQM_SiStripCommissioningAnalysis_VpspScanAlgorithm_H

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DQM/SiStripCommissioningAnalysis/interface/CommissioningAlgorithm.h"
#include <vector>

class VpspScanAnalysis;
class TH1;

/** 
   @class VpspScanAlgorithm
   @author M. Wingham, R.Bainbridge
   @brief Histogram-based analysis for VPSP scan.
*/
class VpspScanAlgorithm : public CommissioningAlgorithm {
  
 public:

  VpspScanAlgorithm( const edm::ParameterSet & pset, VpspScanAnalysis* const );

  virtual ~VpspScanAlgorithm() {;}

  /** Histogram pointer and title. */
  const Histo& histo( const uint16_t& apv ) const;
  
 private:

  VpspScanAlgorithm() {;}

  /** Extracts and organises histograms. */
  void extract( const std::vector<TH1*>& );

  /** Performs histogram anaysis. */
  void analyse();
  
 private:
  
  /** Pointers and titles for histograms. */
  std::vector<Histo> histos_;
  
};

#endif // DQM_SiStripCommissioningAnalysis_VpspScanAlgorithm_H

