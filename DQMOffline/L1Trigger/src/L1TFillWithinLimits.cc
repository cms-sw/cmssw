#include "DQMOffline/L1Trigger/interface/L1TFillWithinLimits.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

namespace dqmoffline {
namespace l1t {

/**
 * Fills a given
 * @param
 */
void fillWithinLimits(MonitorElement* mon, double value, double weight)
{
  TH1 * hist = mon->getTH1F();
  double min(hist->GetXaxis()->GetXmin());
  double max(hist->GetXaxis()->GetXmax());

  double fillValue = getFillValueWithinLimits(value, min, max);
  mon->Fill(fillValue, weight);

}
void fill2DWithinLimits(MonitorElement* mon, double valueX, double valueY, double weight)
{
  TH1 * hist = mon->getTH2F();
  double minX(hist->GetXaxis()->GetXmin());
  double minY(hist->GetYaxis()->GetXmin());

  double maxX(hist->GetXaxis()->GetXmax());
  double maxY(hist->GetYaxis()->GetXmax());

  double fillValueX = getFillValueWithinLimits(valueX, minX, maxX);
  double fillValueY = getFillValueWithinLimits(valueY, minY, maxY);
  mon->Fill(fillValueX, fillValueY, weight);

}

double getFillValueWithinLimits(double value, double min, double max)
{
  if (value < min)
    return min;

  // histograms are [min, max), hence fill with a slightly smaller value
  if (value > max)
    return max - 1e-6 * max;

  return value;
}

}
}
