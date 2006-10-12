#ifndef DQM_SiStripCommissioningSummary_PedestalsSummary_H
#define DQM_SiStripCommissioningSummary_PedestalsSummary_H

#include "DQM/SiStripCommissioningSummary/interface/CommissioningSummary.h"

/**
   @file : DQM/SiStripCommissioningSummary/interface/PedestalsSummary.h
   @class : CommissioningSummary
   @author: M.Wingham

   @brief : Class for SST pedestals commissioning summaries. Inherits from 
   CommissioningSummary.h. Extra histogram formatting is added in the format() method.
*/

class PedestalsSummary : public CommissioningSummary {

 public:

  /** Constructor */
  PedestalsSummary(sistrip::View);

  /** Destructor */
  ~PedestalsSummary();

 private:
  
  /** Add extra formatting to the summary_ and histogram_ histos owned by the CommissioningSummary.h base.*/
  void format();

};

#endif // DQM_SiStripCommissioningSummary_PedestalsSummary_H
