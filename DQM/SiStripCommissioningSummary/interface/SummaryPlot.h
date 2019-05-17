#ifndef DQM_SiStripCommissioningSummary_SummaryPlot_H
#define DQM_SiStripCommissioningSummary_SummaryPlot_H

#include "DataFormats/SiStripCommon/interface/SiStripConstants.h"
#include <iostream>
#include <sstream>
#include <string>

class SummaryPlot;

/** Provides debug information. */
std::ostream& operator<<(std::ostream&, const SummaryPlot&);

/** 
    @class SummaryPlot
    @author R.Bainbridge, P.Kalavase
    @brief Class holding info that defines a summary plot.
*/
class SummaryPlot {
public:
  // ---------- Con(de)structors ----------

  /** */
  SummaryPlot(const std::string& monitorable,
              const std::string& presentation,
              const std::string& granularity,
              const std::string& level);

  /** */
  SummaryPlot(const SummaryPlot&);

  /** */
  SummaryPlot();

  /** */
  ~SummaryPlot() { ; }

  // ---------- Access to member data ----------

  /** */
  inline const sistrip::Monitorable& monitorable() const;

  /** */
  inline const sistrip::Presentation& presentation() const;

  /** */
  inline const sistrip::View& view() const;

  /** */
  inline const sistrip::Granularity& granularity() const;

  /** */
  inline const std::string& level() const;

  // ---------- Utility methods ----------

  /** */
  inline const bool& isValid() const;

  /** */
  void reset();

  /** */
  void print(std::stringstream&) const;

private:
  // ---------- Private methods ----------

  /** */
  void check();

  // ---------- Private member data ----------

  sistrip::Monitorable mon_;

  sistrip::Presentation pres_;

  sistrip::View view_;

  sistrip::Granularity gran_;

  std::string level_;

  bool isValid_;
};

// ---------- Inline methods ----------

const sistrip::Monitorable& SummaryPlot::monitorable() const { return mon_; }
const sistrip::Presentation& SummaryPlot::presentation() const { return pres_; }
const sistrip::View& SummaryPlot::view() const { return view_; }
const sistrip::Granularity& SummaryPlot::granularity() const { return gran_; }
const std::string& SummaryPlot::level() const { return level_; }
const bool& SummaryPlot::isValid() const { return isValid_; }

#endif  // DQM_SiStripCommissioningSummary_SummaryPlot_H
