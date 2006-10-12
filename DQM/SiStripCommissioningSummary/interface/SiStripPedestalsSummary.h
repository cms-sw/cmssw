#ifndef DQM_SiStripCommissioningSummary_SiStripPedestalsSummary_H
#define DQM_SiStripCommissioningSummary_SiStripPedestalsSummary_H

#include "DQM/SiStripCommissioningSummary/interface/SiStripSummary.h"

/**
   @file : DQM/SiStripCommissioningSummary/interface/SiStripPedestalsSummary.h
   @class : SiStripSummary
   @author: M.Wingham

   @brief : Class for SST pedestals commissioning summaries. Inherits from 
   SiStripSummary.h. Extra histogram formatting is added in the format() method.
*/

class SiStripPedestalsSummary : public SiStripSummary {

 public:

  /** Constructor */
  SiStripPedestalsSummary(sistrip::View);

  /** Destructor */
  ~SiStripPedestalsSummary();

 private:
  
  /** Add extra formatting to the summary_ and histogram_ histos owned by the SiStripSummary.h base.*/
  void format();

};

#endif // DQM_SiStripCommissioningSummary_SiStripPedestalsSummary_H
