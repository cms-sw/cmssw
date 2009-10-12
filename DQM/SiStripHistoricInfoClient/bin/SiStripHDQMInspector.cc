#include "DQMServices/Diagnostic/interface/HDQMInspector.h"
#include "DQM/SiStripHistoricInfoClient/test/HDQMInspectorConfigSiStrip.h"
#include "DQMServices/Diagnostic/interface/DQMHistoryTrendsConfig.h"
#include "DQMServices/Diagnostic/interface/DQMHistoryCreateTrend.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <string>
#include <vector>

using namespace std;

/// Simple helper function for building the string for TIB,TOB,TEC,TID items
string multiItems(const vector<string> & subDetectors, const string & item)
{
  string multiItems;
  if( !subDetectors.empty() ) {
    // Fill the first and then the others, prepending the rest with a ","
    vector<string>::const_iterator subDet = subDetectors.begin();
    multiItems.append(*subDet+"@"+item);
    ++subDet;
    for( ; subDet != subDetectors.end(); ++subDet ) {
      multiItems.append(","+*subDet+"@"+item);
    }
  }
  return multiItems;
}

/**
 * Extraction of the summary information using DQMServices/Diagnostic/test/HDQMInspector. <br>
 * The sqlite database should have been filled using the new SiPixelHistoryDQMService.   
 */
void runTrackingInspector( const string &tagName, const string & Password, const int Start, const int End, const int nRuns )
{
  // IMPORTANT SETTINGS:
  string condition = "369098752@Summary_TotalNumberOfClusters_OffTrack@entries > 10000";
  string blackList = "109468";

  // string siStripTracker = "268435456";
  vector<string> subDetectors;
  subDetectors.push_back("369098752"); // TIB (check)
  subDetectors.push_back("436207616"); // TOB (check)
  subDetectors.push_back("402653184"); // TID (check)
  subDetectors.push_back("469762048"); // TEC (check)
  // -------------------

  HDQMInspectorConfigSiStrip siStripConfig;
  vector<string> ItemsForIntegration;
  ItemsForIntegration.push_back("TotalNumberOfClusters_OnTrack_entries");
  siStripConfig.computeIntegralList(ItemsForIntegration);
  DQMHistoryCreateTrend makeTrend(&siStripConfig);

  // Database and output configuration
  makeTrend.setDB("oracle://cms_orcoff_prep/CMS_DQM_31X_OFFLINE",tagName,"cms_dqm_31x_offline", Password,"");
  makeTrend.setDebug(1);
  makeTrend.setDoStat(1);
  makeTrend.setBlackList(blackList);

  // Definition of trends
  typedef DQMHistoryTrendsConfig Trend;
  vector<Trend> config;
  config.push_back(Trend( multiItems(subDetectors, "Summary_TotalNumberOfClusters_OnTrack@mean"), "OnTrackClusters.gif", 0,
                          condition+"&& 369098752@Summary_TotalNumberOfClusters_OnTrack@mean > 0", Start, End, nRuns ));
  config.push_back(Trend( multiItems(subDetectors, "Summary_TotalNumberOfClusters_OnTrack@entries"), "OnTrackClusters_entries.gif", 0,
                          condition+"&& 369098752@Summary_TotalNumberOfClusters_OnTrack@entries > 0", Start, End, nRuns ));
  config.push_back(Trend( multiItems(subDetectors, "Summary_TotalNumberOfClusters_OffTrack@mean"), "TotalNumberOfClusters_OffTrack.gif", 0,
                          condition+"&& 369098752@Summary_TotalNumberOfClusters_OffTrack@mean > 0", Start, End, nRuns ));
  config.push_back(Trend( multiItems(subDetectors, "Summary_TotalNumberOfClusters_OffTrack@entries"), "TotalNumberOfClusters_OffTrack_entries.gif", 0,
                          condition+"&& 369098752@Summary_TotalNumberOfClusters_OffTrack@entries > 0", Start, End, nRuns ));
  config.push_back(Trend( multiItems(subDetectors, "Summary_ClusterChargeCorr_OnTrack@landauPeak"), "ClusterChargeCorr_OnTrack_landau.gif", 0,
                          condition+"&& 369098752@Summary_ClusterChargeCorr_OnTrack@entries > 10000", Start, End, nRuns ));
  config.push_back(Trend( multiItems(subDetectors, "Summary_ClusterCharge_OffTrack@landauPeak"), "ClusterCharge_OffTrack_landau.gif", 0,
                          condition+"&& 369098752@Summary_ClusterCharge_OffTrack@entries > 10000", Start, End, nRuns ));
  config.push_back(Trend( multiItems(subDetectors, "Summary_ClusterNoise_OnTrack@gaussMean"), "ClusterNoise_OnTrack_gauss.gif", 0,
                          condition+"&& 369098752@Summary_ClusterNoise_OnTrack@entries > 10000", Start, End, nRuns ));
  config.push_back(Trend( multiItems(subDetectors, "Summary_ClusterNoise_OffTrack@gaussMean"), "ClusterNoise_OffTrack_gauss.gif", 0,
                          condition+"&& 369098752@Summary_ClusterNoise_OffTrack@entries > 10000", Start, End, nRuns ));
  config.push_back(Trend( multiItems(subDetectors, "Summary_ClusterStoNCorr_OnTrack@mean"), "ClusterStoNCorr_OnTrack.gif", 0,
                          condition+"&& 369098752@Summary_ClusterStoNCorr_OnTrack@mean > 0", Start, End, nRuns ));
  config.push_back(Trend( multiItems(subDetectors, "Summary_ClusterStoNCorr_OnTrack@landauPeak"), "ClusterStoNCorr_OnTrack_landau.gif", 0,
                          condition+"&& 369098752@Summary_ClusterStoNCorr_OnTrack@entries > 10000", Start, End, nRuns ));
  config.push_back(Trend( multiItems(subDetectors, "Summary_ClusterStoN_OffTrack@landauPeak"), "ClusterStoN_OffTrack_landau.gif", 0,
                          condition+"&& 369098752@Summary_ClusterStoN_OffTrack@entries > 10000", Start, End, nRuns ));
  config.push_back(Trend( multiItems(subDetectors, "Summary_ClusterWidth_OnTrack@mean"), "ClusterWidth_OnTrack.gif", 0,
                          condition+"&& 369098752@Summary_ClusterWidth_OnTrack@mean > 0", Start, End, nRuns ));

  // FED errors entries
  config.push_back(Trend( multiItems(subDetectors, "nFEDErrors@entries"), "nFEDErrors.gif", 0,
                          condition, Start, End, nRuns ));
  config.push_back(Trend( multiItems(subDetectors, "nBadActiveChannelStatusBits@entries"), "nBadActiveChannelStatusBits.gif", 0,
                          condition, Start, End, nRuns ));
  config.push_back(Trend( multiItems(subDetectors, "nBadChannelStatusBits@entries"), "nBadChannelStatusBits.gif", 0,
                          condition, Start, End, nRuns ));
  config.push_back(Trend( multiItems(subDetectors, "nAPVAddressError@entries"), "nAPVAddressError.gif", 0,
                          condition, Start, End, nRuns ));
  config.push_back(Trend( multiItems(subDetectors, "nUnlocked@entries"), "nUnlocked.gif", 0,
                          condition, Start, End, nRuns ));
  config.push_back(Trend( multiItems(subDetectors, "nOutOfSync@entries"), "nOutOfSync.gif", 0,
                          condition, Start, End, nRuns ));

  // FED errors means
  config.push_back(Trend( multiItems(subDetectors, "nFEDErrors@mean"), "nFEDErrors.gif", 0,
                          condition, Start, End, nRuns ));
  config.push_back(Trend( multiItems(subDetectors, "nBadActiveChannelStatusBits@mean"), "nBadActiveChannelStatusBits.gif", 0,
                          condition, Start, End, nRuns ));
  config.push_back(Trend( multiItems(subDetectors, "nBadChannelStatusBits@mean"), "nBadChannelStatusBits.gif", 0,
                          condition, Start, End, nRuns ));
  config.push_back(Trend( multiItems(subDetectors, "nAPVAddressError@mean"), "nAPVAddressError.gif", 0,
                          condition, Start, End, nRuns ));
  config.push_back(Trend( multiItems(subDetectors, "nUnlocked@mean"), "nUnlocked.gif", 0,
                          condition, Start, End, nRuns ));
  config.push_back(Trend( multiItems(subDetectors, "nOutOfSync@mean"), "nOutOfSync.gif", 0,
                          condition, Start, End, nRuns ));


  // Creation of trends
  for_each(config.begin(), config.end(), makeTrend);

  // Close the output file
  makeTrend.closeFile();
}

void SiStripHDQMInspector( const string & tagName, const string & password, const int start, const int end )
{
  runTrackingInspector(tagName, password, start, end, 0);
}

void SiStripHDQMInspector( const string & tagName, const string & password, const int nRuns )
{
  runTrackingInspector(tagName, password, 0, 0, nRuns);
}

int main (int argc, char* argv[])
{
  if (argc != 4 && argc != 5) {
    std::cerr << "Usage: " << argv[0] << " [TagName] [Password] [NRuns] " << std::endl;
    std::cerr << "Or:    " << argv[0] << " [TagName] [Password] [FirstRun] [LastRun] " << std::endl;
    return 1;
  }

  if (argc == 4) {
    std::cout << "Creating trends for NRuns = " << argv[3] << " for tag: " << argv[1] << std::endl;
    SiStripHDQMInspector( argv[1], argv[2], atoi(argv[3]) );
  } else if(argc == 5) {
    std::cout << "Creating trends for range:  " << argv[3] << " " << argv[4] << " for tag: " << argv[1] << std::endl;
    SiStripHDQMInspector( argv[1], argv[2], atoi(argv[3]), atoi(argv[4]) );
  }

  return 0;
}
