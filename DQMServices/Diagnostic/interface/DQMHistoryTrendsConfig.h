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
                         const string & inputConditions, const int inputFirstRun, const int inputLastRun, const int inputNruns = 0) :
    item(inputItem), canvasName(inputCanvasName), logY(inputLogY),
    conditions(inputConditions), firstRun(inputFirstRun), lastRun(inputLastRun), nRuns(inputNruns) {}

  /// Used to pass only nRuns
  DQMHistoryTrendsConfig(const string & inputItem, const string & inputCanvasName, const int inputLogY,
                         const string & inputConditions, const int inputNruns) :
    item(inputItem), canvasName(inputCanvasName), logY(inputLogY),
    conditions(inputConditions), firstRun(0), lastRun(0), nRuns(inputNruns) {}

  // All public data members
  string item;
  string canvasName;
  int logY;
  string conditions;
  int firstRun;
  int lastRun;
  int nRuns;
};

#endif
