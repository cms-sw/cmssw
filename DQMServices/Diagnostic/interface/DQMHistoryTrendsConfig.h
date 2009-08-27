#ifndef DQMHISTORYTRENDSCONFIG_H
#define DQMHISTORYTRENDSCONFIG_H

#include <string>

using namespace std;

/**
 * Holds the configuration of all the requested trends. <br>
 * Since it is a very small class used only to store and pass back
 * configuration date, all its data memeber are public.
 */
class DQMHistoryTrendsConfig
{
public:
  /// Used to pass firstRuns, lastRun and eventually nRuns
  DQMHistoryTrendsConfig(const string & inputItem, const string & inputCanvasName, const int inputLogY,
                         const string & inputConditions, const std::string& inputLabels, const int inputFirstRun, const int inputLastRun, const int inputNruns = 0,
                         const double & yMinIn = 999999, const double & yMaxIn = -999999) :
    item(inputItem), canvasName(inputCanvasName), logY(inputLogY),
    conditions(inputConditions), Labels(inputLabels), firstRun(inputFirstRun), lastRun(inputLastRun), nRuns(inputNruns),
    yMin(yMinIn), yMax(yMaxIn)
  {
    // Do not use the range if the defaults were changed
    useYrange = 0;
    if ( yMin != 999999 && yMax != -999999) {
      useYrange = 3;
    } else if ( yMin != 999999) {
      useYrange = 1;
    } else if( yMax != -999999 ) {
      useYrange = 2;
    }
  }

  /// Used to pass only nRuns
  DQMHistoryTrendsConfig(const string & inputItem, const string & inputCanvasName, const int inputLogY,
                         const string & inputConditions, const std::string& inputLabels, const int inputNruns,
                         const double & yMinIn = 999999, const double & yMaxIn = -999999) :
    item(inputItem), canvasName(inputCanvasName), logY(inputLogY),
    conditions(inputConditions), Labels(inputLabels), firstRun(0), lastRun(0), nRuns(inputNruns),
    yMin(yMinIn), yMax(yMaxIn)
  {
    useYrange = 0;
    if ( yMin != 999999 && yMax != -999999) {
      useYrange = 3;
    } else if ( yMin != 999999) {
      useYrange = 1;
    } else if( yMax != -999999 ) {
      useYrange = 2;
    }
  }

  // All public data members
  string item;
  string canvasName;
  int logY;
  string conditions;
  std::string Labels;
  int firstRun;
  int lastRun;
  int nRuns;
  double yMin, yMax;
  int useYrange;
};

#endif
