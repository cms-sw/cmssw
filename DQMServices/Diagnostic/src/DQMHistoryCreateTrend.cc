#include "DQMServices/Diagnostic/interface/DQMHistoryCreateTrend.h"

using namespace std;

void DQMHistoryCreateTrend::operator()(const DQMHistoryTrendsConfig & trend)
{
  if( trend.firstRun != trend.lastRun ) {
    inspector_->createTrend(trend.item, trend.canvasName, trend.logY, trend.conditions, trend.Labels,
                            trend.firstRun, trend.lastRun, trend.useYrange, trend.yMin, trend.yMax);
  }
  else if( trend.nRuns != 0 ) {
    inspector_->createTrendLastRuns(trend.item, trend.canvasName, trend.logY, trend.conditions, trend.Labels,
                                    trend.nRuns, trend.useYrange, trend.yMin, trend.yMax);
  }
  else {
    cout << "WARNING: both first("<<trend.firstRun<<")==last("<<trend.lastRun<<") and nRuns("
         <<trend.nRuns<<")==0. No trends will be created for " << trend.item << endl;
  }
}
