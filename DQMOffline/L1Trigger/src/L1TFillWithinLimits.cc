#include "DQMOffline/L1Trigger/interface/L1TFillWithinLimits.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

namespace dqmoffline {
  namespace l1t {

    /**
 * Fills a given
 * @param
 */
    void fillWithinLimits(MonitorElement* mon, double value, double weight) {
      double min(mon->getAxisMin(1));
      double max(mon->getAxisMax(1));

      double fillValue = getFillValueWithinLimits(value, min, max);
      mon->Fill(fillValue, weight);
    }
    void fill2DWithinLimits(MonitorElement* mon, double valueX, double valueY, double weight) {
      double minX(mon->getAxisMin(1));
      double minY(mon->getAxisMin(2));

      double maxX(mon->getAxisMax(1));
      double maxY(mon->getAxisMax(2));

      double fillValueX = getFillValueWithinLimits(valueX, minX, maxX);
      double fillValueY = getFillValueWithinLimits(valueY, minY, maxY);
      mon->Fill(fillValueX, fillValueY, weight);
    }

    double getFillValueWithinLimits(double value, double min, double max) {
      if (value < min)
        return min;

      // histograms are [min, max), hence fill with a slightly smaller value
      if (value > max)
        return max - 1e-6 * max;

      return value;
    }

  }  // namespace l1t
}  // namespace dqmoffline
