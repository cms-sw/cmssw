#ifndef DQMOFFLINE_L1TRIGGER_L1TFILLWITHINLIMITS_H
#define DQMOFFLINE_L1TRIGGER_L1TFILLWITHINLIMITS_H

#include "DQMServices/Core/interface/DQMStore.h"

namespace dqmoffline {
  namespace l1t {

    typedef dqm::reco::MonitorElement MonitorElement;

    /**
 * Fills a given MonitorElement within the boundaries of the underlying histogram.
 * This means that underflow is filled into the first bin and overflow is filled into the last bin.
 * @param pointer to the DQM MonitorElement
 * @param fill value
 * @param optional weight
 */
    void fillWithinLimits(MonitorElement* mon, double value, double weight = 1.);
    /**
 * Fills a given MonitorElement within the boundaries of the underlying histogram.
 * This means that underflow is filled into the first bin and overflow is filled into the last bin.
 * @param pointer to the DQM MonitorElement
 * @param fill value for X
 * @param fill value for Y
 * @param optional weight X
 * @param optional weight Y
 */
    void fill2DWithinLimits(MonitorElement* mon, double valueX, double valueY, double weight = 1.);

    double getFillValueWithinLimits(double value, double min, double max);
  }  // namespace l1t
}  // namespace dqmoffline

#endif
