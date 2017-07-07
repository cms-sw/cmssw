#ifndef DQM_SiStripCommissioningAnalysis_SamplingAlgorithm_H
#define DQM_SiStripCommissioningAnalysis_SamplingAlgorithm_H

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DQM/SiStripCommissioningAnalysis/interface/CommissioningAlgorithm.h"
#include <vector>

class SamplingAnalysis;
class TProfile;
class TF1;

/**
   @class SamplingAlgorithm
   @author C. Delaere
   @brief Algorithm for latency run
*/

class SamplingAlgorithm : public CommissioningAlgorithm {
  
 public:
  
  SamplingAlgorithm( const edm::ParameterSet & pset, SamplingAnalysis* const, uint32_t latencyCode = 0 );
  
  ~SamplingAlgorithm() override {;}
  
  inline const Histo& histo() const;
  
 private:
  
  SamplingAlgorithm() {;}

  void extract( const std::vector<TH1*>& ) override;
  
  void analyse() override;
  
  void pruneProfile( TProfile* profile ) const;
  
  void correctBinning( TProfile* prof ) const;
  
  void correctProfile( TProfile* profile, float SoNcut=3. ) const;
  
 private:
  
  /** pulse shape*/
  Histo histo_;
  
  /** Fitter in peak and deconvolution mode */
  TF1* deconv_fitter_;
  TF1* peak_fitterA_;
  TF1* peak_fitterB_;

  /** latency code for fine delay scans */
  uint32_t latencyCode_;

  /** SamplingAnalysis object */
  SamplingAnalysis* samp_;
  
};

#endif // DQM_SiStripCommissioningAnalysis_SamplingAlgorithm_H

