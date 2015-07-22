#include "DQMServices/Diagnostic/interface/HDQMInspector.h"
#include "DQM/SiStripHistoricInfoClient/interface/HDQMInspectorConfigSiStrip.h"
#include "DQMServices/Diagnostic/interface/DQMHistoryTrendsConfig.h"
#include "DQMServices/Diagnostic/interface/DQMHistoryCreateTrend.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <string>
#include <vector>

#include "FWCore/FWLite/interface/FWLiteEnabler.h"
#include <TROOT.h>
#include <TFile.h>
#include <TSystem.h>

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
void runTrackingInspector( const string &dbName, const string &tagName, const string & Password, const string & whiteListFile,
			   const int Start, const int End, const int nRuns )
{
  // IMPORTANT SETTINGS:
  string condition = "";
  // string condition = "369098752@Summary_TotalNumberOfClusters_OffTrack@entries > 10000 || 436207616@Summary_TotalNumberOfClusters_OffTrack@entries > 10000 || 402653184@Summary_TotalNumberOfClusters_OffTrack@entries > 10000 || 469762048@Summary_TotalNumberOfClusters_OffTrack@entries > 10000";
  string blackList = "109468";

  // string whiteList = readListFromFile(whiteListFile);

  string siStripTracker = "268435456";
  vector<string> subDetectors;
  subDetectors.push_back("369098752"); // TIB (check)
  subDetectors.push_back("436207616"); // TOB (check)
  // subDetectors.push_back("402653184"); // TID (check)
  subDetectors.push_back("402661376"); // TID- (check)
  subDetectors.push_back("402669568"); // TID+ (check)
  // subDetectors.push_back("469762048"); // TEC (check)
  subDetectors.push_back("470024192"); // TEC- (check)
  subDetectors.push_back("470286336"); // TEC+ (check)
  // -------------------

  HDQMInspectorConfigSiStrip siStripConfig;
  vector<string> ItemsForIntegration;
  ItemsForIntegration.push_back("TotalNumberOfClusters_OnTrack_entries");
  siStripConfig.computeIntegralList(ItemsForIntegration);
  DQMHistoryCreateTrend makeTrend(&siStripConfig);

  // Database and output configuration
  // makeTrend.setDB("oracle://cms_orcoff_prep/CMS_DQM_31X_OFFLINE",tagName,"cms_dqm_31x_offline", Password,"");
  // makeTrend.setDB(dbName,tagName,"cms_dqm_31x_offline", Password,"");
  makeTrend.setDB(dbName,tagName,"/afs/cern.ch/cms/DB/conddb");
  // makeTrend.setDB(dbName,tagName,"CMS_COND_GENERAL_R", Password,"");
  makeTrend.setDebug(0);
  makeTrend.setDoStat(1);
  makeTrend.setSkip99s(true);
  makeTrend.setBlackList(blackList);
  // makeTrend.setWhiteList(whiteList);
  makeTrend.setWhiteListFromFile(whiteListFile);

  // Definition of trends
  typedef DQMHistoryTrendsConfig Trend;
  vector<Trend> config;
  config.push_back(Trend( multiItems(subDetectors, "Summary_TotalNumberOfClusters_OnTrack@mean"), "OnTrackClusters_mean.gif", 0,
                          condition+"", "", Start, End, nRuns, 0 ));
  config.push_back(Trend( multiItems(subDetectors, "Summary_TotalNumberOfClusters_OnTrack@entries"), "OnTrackClusters_entries.gif", 0,
                          condition+"&& 369098752@Summary_TotalNumberOfClusters_OnTrack@entries > 0", "", Start, End, nRuns, 0 ));
  config.push_back(Trend( multiItems(subDetectors, "Summary_TotalNumberOfClusters_OffTrack@mean"), "TotalNumberOfClusters_OffTrack_mean.gif", 0,
                          condition+"&& 369098752@Summary_TotalNumberOfClusters_OffTrack@mean > 0", "", Start, End, nRuns, 0, 1000 ));
  config.push_back(Trend( multiItems(subDetectors, "Summary_TotalNumberOfClusters_OffTrack@entries"), "TotalNumberOfClusters_OffTrack_entries.gif", 0,
                          condition+"&& 369098752@Summary_TotalNumberOfClusters_OffTrack@entries > 0", "", Start, End, nRuns, 0 ));
  config.push_back(Trend( multiItems(subDetectors, "Summary_ClusterChargeCorr_OnTrack@landauPeak"), "ClusterChargeCorr_OnTrack_landau.gif", 0,
                          condition+"&& 369098752@Summary_ClusterChargeCorr_OnTrack@entries > 10000", "", Start, End, nRuns, 0 ));
  config.push_back(Trend( multiItems(subDetectors, "Summary_ClusterCharge_OffTrack@mean"), "ClusterCharge_OffTrack_mean.gif", 0,
                          condition+"&& 369098752@Summary_ClusterCharge_OffTrack@entries > 10000", "", Start, End, nRuns, 0 ));
  config.push_back(Trend( multiItems(subDetectors, "Summary_ClusterCharge_OffTrack@landauPeak"), "ClusterCharge_OffTrack_landau.gif", 0,
                          condition+"&& 369098752@Summary_ClusterCharge_OffTrack@entries > 10000", "", Start, End, nRuns, 0 ));
  config.push_back(Trend( multiItems(subDetectors, "Summary_ClusterNoise_OnTrack@gaussMean"), "ClusterNoise_OnTrack_gauss.gif", 0,
                          condition+"&& 369098752@Summary_ClusterNoise_OnTrack@entries > 10000", "", Start, End, nRuns, 0 ));
  config.push_back(Trend( multiItems(subDetectors, "Summary_ClusterNoise_OffTrack@gaussMean"), "ClusterNoise_OffTrack_gauss.gif", 0,
                          condition+"&& 369098752@Summary_ClusterNoise_OffTrack@entries > 10000", "", Start, End, nRuns, 0, 8 ));
  config.push_back(Trend( multiItems(subDetectors, "Summary_ClusterStoNCorr_OnTrack@landauPeak"), "ClusterStoNCorr_OnTrack_landauPeak.gif", 0,
                          condition+"&& 369098752@Summary_ClusterStoNCorr_OnTrack@landauPeak > 0", "", Start, End, nRuns, 0 ));
  config.push_back(Trend( multiItems(subDetectors, "Summary_ClusterStoNCorr_OnTrack@mean"), "ClusterStoNCorr_OnTrack_mean.gif", 0,
                          condition+"&& 369098752@Summary_ClusterStoNCorr_OnTrack@mean > 0", "", Start, End, nRuns, 0 ));
  config.push_back(Trend( multiItems(subDetectors, "Summary_ClusterStoN_OffTrack@landauPeak"), "ClusterStoN_OffTrack_landauPeak.gif", 0,
                          condition+"&& 369098752@Summary_ClusterStoN_OffTrack@entries > 10000", "", Start, End, nRuns, 0, 60 ));
  config.push_back(Trend( multiItems(subDetectors, "Summary_ClusterStoN_OffTrack@mean"), "ClusterStoN_OffTrack_mean.gif", 0,
                          condition+"&& 369098752@Summary_ClusterStoN_OffTrack@entries > 10000", "", Start, End, nRuns, 0, 80 ));
  config.push_back(Trend( multiItems(subDetectors, "Summary_ClusterWidth_OnTrack@mean"), "ClusterWidth_OnTrack_mean.gif", 0,
                          condition+"&& 369098752@Summary_ClusterWidth_OnTrack@mean > 0", "", Start, End, nRuns, 0 ));
  config.push_back(Trend( multiItems(subDetectors, "Summary_ClusterWidth_OffTrack@mean"), "ClusterWidth_OffTrack_mean.gif", 0,
                          condition+"", "", Start, End, nRuns, 0, 7 ));
  config.push_back(Trend( multiItems(subDetectors, "TotalNumberOfClusterProfile@yMean"), "TotalNumberOfClusterProfile_ymean.gif", 0,
                          condition+"", "", Start, End, nRuns, 0, 15 ));

  // FED errors entries
  config.push_back(Trend( siStripTracker+"@nFEDErrors@entries", "nFEDErrors.gif", 0,
                          condition, "", Start, End, nRuns ));
  config.push_back(Trend( siStripTracker+"@nBadActiveChannelStatusBits@entries", "nBadActiveChannelStatusBits.gif", 0,
                          condition, "", Start, End, nRuns ));
  config.push_back(Trend( siStripTracker+"@nBadChannelStatusBits@entries", "nBadChannelStatusBits.gif", 0,
                          condition, "", Start, End, nRuns ));
  config.push_back(Trend( siStripTracker+"@nAPVAddressError@entries", "nAPVAddressError.gif", 0,
                          condition, "", Start, End, nRuns ));
  config.push_back(Trend( siStripTracker+"@nUnlocked@entries", "nUnlocked.gif", 0,
                          condition, "", Start, End, nRuns ));
  config.push_back(Trend( siStripTracker+"@nOutOfSync@entries", "nOutOfSync.gif", 0,
                          condition, "", Start, End, nRuns ));

  // FED errors means
  config.push_back(Trend( siStripTracker+"@nFEDErrors@mean", "nFEDErrors_mean.gif", 0,
                          condition, "", Start, End, nRuns ));
  config.push_back(Trend( siStripTracker+"@nBadActiveChannelStatusBits@mean", "nBadActiveChannelStatusBits_mean.gif", 0,
                          condition, "", Start, End, nRuns ));
  config.push_back(Trend( siStripTracker+"@nBadChannelStatusBits@mean", "nBadChannelStatusBits_mean.gif", 0,
                          condition, "", Start, End, nRuns ));
  config.push_back(Trend( siStripTracker+"@nAPVAddressError@mean", "nAPVAddressError_mean.gif", 0,
                          condition, "", Start, End, nRuns ));
  config.push_back(Trend( siStripTracker+"@nUnlocked@mean", "nUnlocked_mean.gif", 0,
                          condition, "", Start, End, nRuns ));
  config.push_back(Trend( siStripTracker+"@nOutOfSync@mean", "nOutOfSync_mean.gif", 0,
                          condition, "", Start, End, nRuns ));

  // Creation of trends
  for_each(config.begin(), config.end(), makeTrend);

  // Close the output file
  makeTrend.closeFile();
}

void SiStripHDQMInspector( const string & dbName, const string & tagName, const string & password, const std::string & whiteListFile,
			   const int start, const int end )
{
  runTrackingInspector(dbName, tagName, password, whiteListFile, start, end, 0);
}

void SiStripHDQMInspector( const string & dbName, const string & tagName, const string & password, const std::string & whiteListFile,
			   const int nRuns )
{
  runTrackingInspector(dbName, tagName, password, whiteListFile, 0, 0, nRuns);
}

int main (int argc, char* argv[])
{
  gSystem->Load( "libFWCoreFWLite" );
  FWLiteEnabler::enable();

  if (argc != 6 && argc != 7) {
    std::cerr << "Usage: " << argv[0] << " [Database] [TagName] [Password] [WhiteListFile] [NRuns] " << std::endl;
    std::cerr << "Or:    " << argv[0] << " [Database] [TagName] [Password] [WhiteListFile] [FirstRun] [LastRun] " << std::endl;
    return 1;
  }

  if (argc == 6) {
    std::cout << "Creating trends for NRuns = " << argv[5] << " for tag: " << argv[2] << std::endl;
    SiStripHDQMInspector( argv[1], argv[2], argv[3], argv[4], atoi(argv[5]) );
  } else if(argc == 7) {
    std::cout << "Creating trends for range:  " << argv[5] << " " << argv[6] << " for tag: " << argv[2] << std::endl;
    SiStripHDQMInspector( argv[1], argv[2], argv[3], argv[4], atoi(argv[5]), atoi(argv[6]) );
  }

  return 0;
}
