#ifndef DQM_SiStripCommissioningAnalysis_CommissioningAnalysis_H
#define DQM_SiStripCommissioningAnalysis_CommissioningAnalysis_H

#include <sstream>

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
  
  /** Abstract base class for holding various parameter values that
      are extracted from the appropriate histogram analysis. */
  class Monitorables  {
  public:
    Monitorables() {;}
    virtual ~Monitorables() {;}
    virtual void print( std::stringstream& ) {;}
  };
  
};

#endif // DQM_SiStripCommissioningAnalysis_CommissioningAnalysis_H

