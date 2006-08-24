#ifndef DQM_SiStripCommissioningAnalysis_CommissioningAnalysis_H
#define DQM_SiStripCommissioningAnalysis_CommissioningAnalysis_H

#include <sstream>

/**
   @class CommissioningAnalysis
   @author M.Wingham, R.Bainbridge 

   @brief Base for derived classes that provide analysis of
   commissioning histograms.
*/

class CommissioningAnalysis {

 public:
  
  CommissioningAnalysis() {;}
  virtual ~CommissioningAnalysis() {;}
  
  /** Base class for holding various parameter values that are
      extracted from the appropriate histogram analysis. */
  class Monitorables  {
  public:
    Monitorables() {;}
    virtual ~Monitorables() {;}
    virtual void print( std::stringstream& ) {;}
  };
  
};

#endif // DQM_SiStripCommissioningAnalysis_CommissioningAnalysis_H

