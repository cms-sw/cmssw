#ifndef DQM_SiStripCommissioningSummary_SummaryGeneratorControlView_H
#define DQM_SiStripCommissioningSummary_SummaryGeneratorControlView_H

#include "DQM/SiStripCommissioningSummary/interface/SummaryGenerator.h"

/**
   @class SummaryGeneratorControlView
   @author M.Wingham, R.Bainbridge
   @brief Fills "summary histograms" in FEC or "control" view.
*/

class SummaryGeneratorControlView : public SummaryGenerator {
public:
  SummaryGeneratorControlView();

  ~SummaryGeneratorControlView() override { ; }

  /** */
  void fill(const std::string& directory_level,
            const sistrip::Granularity&,
            const uint32_t& key,
            const float& value,
            const float& error) override;
};

#endif  // DQM_SiStripCommissioningSummary_SummaryGeneratorControlView_H
