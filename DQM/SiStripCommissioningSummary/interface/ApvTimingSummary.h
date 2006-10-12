#ifndef DQM_SiStripCommissioningSummary_ApvTimingSummary_H
#define DQM_SiStripCommissioningSummary_ApvTimingSummary_H

#include "DQM/SiStripCommissioningSummary/interface/CommissioningSummary.h"
#include "DQM/SiStripCommon/interface/SiStripHistoNamingScheme.h"

/** */
class ApvTimingSummary : public CommissioningSummary {
  
 public:
  
  /** Constructor */
  ApvTimingSummary(sistrip::View);
  
  ApvTimingSummary( const SummaryFactory::Histo&,
		    const std::string& directory );
  
  /** Destructor */
  virtual ~ApvTimingSummary();
  
 private:
  
};

#endif // DQM_SiStripCommissioningSummary_ApvTimingSummary_H
