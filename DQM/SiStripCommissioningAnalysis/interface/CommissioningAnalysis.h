#ifndef DQM_SiStripCommissioningAnalysis_CommissioningAnalysis_H
#define DQM_SiStripCommissioningAnalysis_CommissioningAnalysis_H

/**
   @file : DQM/SiStripCommissioningAnalysis/interface/CommissioningAnalysis.h
   @class : CommissioningAnalysis
   
   @brief Abstract base for derived classes that provide analysis of commissioning histograms.
*/

class CommissioningAnalysis {

 public:
  
  /** Constructor */
  CommissioningAnalysis() {;}
  /** Destructor */
  virtual ~CommissioningAnalysis() {;}
  
};

#endif // DQM_SiStripCommissioningAnalysis_CommissioningAnalysis_H

