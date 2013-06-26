#ifndef DQM_SiStripCommissioningSummary_SummaryGeneratorReadoutView_H
#define DQM_SiStripCommissioningSummary_SummaryGeneratorReadoutView_H

#include "DQM/SiStripCommissioningSummary/interface/SummaryGenerator.h"

/**
   @class SummaryGeneratorReadoutView
   @author M.Wingham, R.Bainbridge
   @brief Fills "summary histograms" in FED or "readout" view.
*/

class SummaryGeneratorReadoutView : public SummaryGenerator {

 public:

  SummaryGeneratorReadoutView();

  virtual ~SummaryGeneratorReadoutView() {;}

  /** */
  void fill( const std::string& directory_level,
	     const sistrip::Granularity&,
	     const uint32_t& key, 
	     const float& value, 
	     const float& error );
  
};

#endif // DQM_SiStripCommissioningSummary_SummaryGeneratorReadoutView_H
