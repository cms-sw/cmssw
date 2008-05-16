#ifndef DQM_SiStripCommissioningAnalysis_SamplingAlgorithm_H
#define DQM_SiStripCommissioningAnalysis_SamplingAlgorithm_H

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
  
  SamplingAlgorithm( SamplingAnalysis* const );
  
  virtual ~SamplingAlgorithm() {;}
  
  inline const Histo& histo() const;
  
 private:
  
  SamplingAlgorithm() {;}

  virtual void extract( const std::vector<TH1*>& );
  
  void analyse();
  
  void pruneProfile( TProfile* profile ) const;
  
  void correctBinning( TProfile* prof ) const;
  
  void correctProfile( TProfile* profile, float SoNcut=3. ) const;
  
 private:
  
  /** pulse shape*/
  Histo histo_;
  
  /** Fitter in peak and deconvolution mode */
  TF1* deconv_fitter_;
  
  TF1* peak_fitter_;

  SamplingAnalysis* samp_;
  
};

#endif // DQM_SiStripCommissioningAnalysis_SamplingAlgorithm_H

