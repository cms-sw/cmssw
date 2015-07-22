#include "DQMServices/Diagnostic/interface/HDQMInspector.h"
#include "DQM/SiStripHistoricInfoClient/interface/HDQMInspectorConfigTracking.h"
#include "DQMServices/Diagnostic/interface/DQMHistoryTrendsConfig.h"
#include "DQMServices/Diagnostic/interface/DQMHistoryCreateTrend.h"
#include <string>

#include "FWCore/FWLite/interface/FWLiteEnabler.h"
#include <TROOT.h>
#include <TFile.h>
#include <TSystem.h>

using namespace std;

/**
 * Extraction of the summary information using DQMServices/Diagnostic/test/HDQMInspector. <br>
 * The sqlite database should have been filled using the new SiPixelHistoryDQMService.   
 */
void runTrackingInspector( const string & dbName, const string &tagName, const string & Password, const string & whiteListFile,
			   const int Start, const int End, const int nRuns )
{
  // IMPORTANT SETTINGS:
  string siStripTracker = "268435456";
  // string condition = siStripTracker+"@Chi2oNDF_CKFTk@entries > 100";
  string condition = siStripTracker+"@Chi2oNDF_GenTk@entries > 100";  // Use for collision data
  //string condition = "";
  string blackList = "109468";
  // -------------------

  HDQMInspectorConfigTracking trackingConfig;
  // Select quantities you want the integral of
  vector<string> ItemsForIntegration;
  //ItemsForIntegration.push_back("Chi2oNDF_CKFTk_entries");
  //ItemsForIntegration.push_back("NumberOfTracks_CKFTk_entries");
  //ItemsForIntegration.push_back("Chi2oNDF_RSTk_entries");
  //ItemsForIntegration.push_back("NumberOfTracks_RSTk_entries");
  //ItemsForIntegration.push_back("Chi2oNDF_CosmicTk_entries");
  //ItemsForIntegration.push_back("NumberOfTracks_CosmicTk_entries");
  ItemsForIntegration.push_back("Chi2oNDF_GenTk_entries");
  ItemsForIntegration.push_back("NumberOfTracks_GenTk_entries");
  trackingConfig.computeIntegralList(ItemsForIntegration);
  // Create the functor
  DQMHistoryCreateTrend makeTrend(&trackingConfig);

  // Database and output configuration
  makeTrend.setDB(dbName,tagName,"/afs/cern.ch/cms/DB/conddb");
  makeTrend.setDebug(1);
  makeTrend.setDoStat(1);
  makeTrend.setSkip99s(true);
  makeTrend.setBlackList(blackList);
  makeTrend.setWhiteListFromFile(whiteListFile);

  // Definition of trends
  typedef DQMHistoryTrendsConfig Trend;
  vector<Trend> config;


  // Something you might want for collisions
  config.push_back(Trend( siStripTracker+"@Chi2oNDF_GenTk@mean", "Chi2oNDF_GenTk_mean.gif", 0, condition, "", Start, End, nRuns, 0, 50 ));
  config.push_back(Trend( siStripTracker+"@NumberOfTracks_GenTk@mean", "NumberOfTracks_GenTk_mean.gif", 0, condition, "", Start, End, nRuns ));
  //config.push_back(Trend( siStripTracker+"@NumberOfRecHitsPerTrack_GenTk@mean", "NumberOfRecHitsPerTrack_GenTk_mean.gif", 0, condition, "", Start, End, nRuns ));
  config.push_back(Trend( siStripTracker+"@TrackPt_ImpactPoint_GenTk@mean", "TrackPt_GenTk_mean.gif", 0, condition, "", Start, End, nRuns, 0, 200 ));
  config.push_back(Trend( siStripTracker+"@TrackPx_ImpactPoint_GenTk@mean", "TrackPx_GenTk_mean.gif", 0, condition, "", Start, End, nRuns, -20, 20 ));
  config.push_back(Trend( siStripTracker+"@TrackPy_ImpactPoint_GenTk@mean", "TrackPy_GenTk_mean.gif", 0, condition, "", Start, End, nRuns, -100, 100 ));
  config.push_back(Trend( siStripTracker+"@TrackPz_ImpactPoint_GenTk@mean", "TrackPz_GenTk_mean.gif", 0, condition, "", Start, End, nRuns, -20, 20 ));
  config.push_back(Trend( siStripTracker+"@TrackPhi_ImpactPoint_GenTk@mean", "TrackPhi_GenTk_mean.gif", 0, condition, "", Start, End, nRuns ));
  config.push_back(Trend( siStripTracker+"@TrackEta_ImpactPoint_GenTk@mean", "TrackEta_GenTk_mean.gif", 0, condition, "", Start, End, nRuns ));
  config.push_back(Trend( siStripTracker+"@DistanceOfClosestApproach_GenTk@mean", "DistanceOfClosestApproach_GenTk_mean.gif", 0, condition, "", Start, End, nRuns, -15, 15 ));
  // Integral
  config.push_back(Trend( siStripTracker+"@Chi2oNDF_GenTk@entries", "Chi2oNDF_GenTk_entries.gif", 0, condition, "", Start, End, nRuns ));
  config.push_back(Trend( siStripTracker+"@NumberOfTracks_GenTk@entries", "NumberOfTracks_GenTk_entries.gif", 0, condition, "", Start, End, nRuns ));


/*
  config.push_back(Trend( siStripTracker+"@Chi2oNDF_GenTk@mean", "Chi2oNDF_GenTk_mean.gif", 0, condition, "", Start, End, nRuns, 0, 50 ));
  config.push_back(Trend( siStripTracker+"@NumberOfTracks_GenTk@mean", "NumberOfTracks_GenTk_mean.gif", 0, condition, "", Start, End, nRuns ));
  config.push_back(Trend( siStripTracker+"@NumberOfRecHitsPerTrack_GenTk@mean", "NumberOfRecHitsPerTrack_GenTk_mean.gif", 0, condition, "", Start, End, nRuns ));
  config.push_back(Trend( siStripTracker+"@TrackPt_GenTk@mean", "TrackPt_GenTk_mean.gif", 0, condition, "", Start, End, nRuns, 0, 200 ));
  config.push_back(Trend( siStripTracker+"@TrackPx_GenTk@mean", "TrackPx_GenTk_mean.gif", 0, condition, "", Start, End, nRuns, -20, 20 ));
  config.push_back(Trend( siStripTracker+"@TrackPy_GenTk@mean", "TrackPy_GenTk_mean.gif", 0, condition, "", Start, End, nRuns, -100, 100 ));
  config.push_back(Trend( siStripTracker+"@TrackPz_GenTk@mean", "TrackPz_GenTk_mean.gif", 0, condition, "", Start, End, nRuns, -20, 20 ));
  config.push_back(Trend( siStripTracker+"@TrackPhi_GenTk@mean", "TrackPhi_GenTk_mean.gif", 0, condition, "", Start, End, nRuns ));
  config.push_back(Trend( siStripTracker+"@TrackEta_GenTk@mean", "TrackEta_GenTk_mean.gif", 0, condition, "", Start, End, nRuns ));
  config.push_back(Trend( siStripTracker+"@DistanceOfClosestApproach_GenTk@mean", "DistanceOfClosestApproach_GenTk_mean.gif", 0, condition, "", Start, End, nRuns, -15, 15 ));
  // Integral
  config.push_back(Trend( siStripTracker+"@Chi2oNDF_GenTk@entries", "Chi2oNDF_GenTk_entries.gif", 0, condition, "", Start, End, nRuns ));
  config.push_back(Trend( siStripTracker+"@NumberOfTracks_GenTk@entries", "NumberOfTracks_GenTk_entries.gif", 0, condition, "", Start, End, nRuns ));


  config.push_back(Trend( siStripTracker+"@Chi2oNDF_CKFTk@mean,"
                         +siStripTracker+"@Chi2oNDF_RSTk@mean,"
                         +siStripTracker+"@Chi2oNDF_CosmicTk@mean",
                         "Chi2oNDF_mean.gif", 0, condition, "CKFTk,RSTk,CosmicTk", Start, End, nRuns, 0, 50 ));
  config.push_back(Trend( siStripTracker+"@Chi2oNDF_CKFTk@mean", "Chi2oNDF_CKFTk_mean.gif", 0, condition, "", Start, End, nRuns, 0, 50 ));
  config.push_back(Trend( siStripTracker+"@Chi2oNDF_RSTk@mean", "Chi2oNDF_RSTk_mean.gif", 0, condition, "", Start, End, nRuns, 0, 50 ));
  config.push_back(Trend( siStripTracker+"@Chi2oNDF_CosmicTk@mean", "Chi2oNDF_CosmicTk_mean.gif", 0, condition, "", Start, End, nRuns, 0, 50 ));

  config.push_back(Trend( siStripTracker+"@NumberOfTracks_CKFTk@mean,"
                         +siStripTracker+"@NumberOfTracks_RSTk@mean,"
                         +siStripTracker+"@NumberOfTracks_CosmicTk@mean",
                         "NumberOfTracks_mean.gif", 0, condition, "CKFTk,RSTk,CosmicTk", Start, End, nRuns ));
  config.push_back(Trend( siStripTracker+"@NumberOfTracks_CKFTk@mean", "NumberOfTracks_CKFTk_mean.gif", 0, condition, "", Start, End, nRuns ));
  config.push_back(Trend( siStripTracker+"@NumberOfTracks_RSTk@mean", "NumberOfTracks_RSTk_mean.gif", 0, condition, "", Start, End, nRuns ));
  config.push_back(Trend( siStripTracker+"@NumberOfTracks_CosmicTk@mean", "NumberOfTracks_CosmicTk_mean.gif", 0, condition, "", Start, End, nRuns ));

  config.push_back(Trend( siStripTracker+"@NumberOfRecHitsPerTrack_CKFTk@mean,"
                         +siStripTracker+"@NumberOfRecHitsPerTrack_RSTk@mean,"
                         +siStripTracker+"@NumberOfRecHitsPerTrack_CosmicTk@mean",
                         "NumberOfRecHitsPerTrack_mean.gif", 0, condition, "CKFTk,RSTk,CosmicTk", Start, End, nRuns ));
  config.push_back(Trend( siStripTracker+"@NumberOfRecHitsPerTrack_CKFTk@mean", "NumberOfRecHitsPerTrack_CKFTk_mean.gif", 0, condition, "", Start, End, nRuns ));
  config.push_back(Trend( siStripTracker+"@NumberOfRecHitsPerTrack_RSTk@mean", "NumberOfRecHitsPerTrack_RSTk_mean.gif", 0, condition, "", Start, End, nRuns ));
  config.push_back(Trend( siStripTracker+"@NumberOfRecHitsPerTrack_CosmicTk@mean", "NumberOfRecHitsPerTrack_CosmicTk_mean.gif", 0, condition, "", Start, End, nRuns ));

  config.push_back(Trend( siStripTracker+"@TrackPt_CKFTk@mean,"
                         +siStripTracker+"@TrackPt_RSTk@mean,"
                         +siStripTracker+"@TrackPt_CosmicTk@mean",
                         "TrackPt_mean.gif", 0, condition, "CKFTk,RSTk,CosmicTk", Start, End, nRuns, 0, 200 ));
  config.push_back(Trend( siStripTracker+"@TrackPt_CKFTk@mean", "TrackPt_CKFTk_mean.gif", 0, condition, "", Start, End, nRuns, 0, 200 ));
  config.push_back(Trend( siStripTracker+"@TrackPt_RSTk@mean", "TrackPt_RSTk_mean.gif", 0, condition, "", Start, End, nRuns, 0, 200 ));
  config.push_back(Trend( siStripTracker+"@TrackPt_CosmicTk@mean", "TrackPt_CosmicTk_mean.gif", 0, condition, "", Start, End, nRuns, 0, 200 ));

  config.push_back(Trend( siStripTracker+"@TrackPx_CKFTk@mean,"
                         +siStripTracker+"@TrackPx_RSTk@mean,"
                         +siStripTracker+"@TrackPx_CosmicTk@mean",
                         "TrackPx_mean.gif", 0, condition, "CKFTk,RSTk,CosmicTk", Start, End, nRuns, -20, 20 ));
  config.push_back(Trend( siStripTracker+"@TrackPx_CKFTk@mean", "TrackPx_CKFTk_mean.gif", 0, condition, "", Start, End, nRuns, -20, 20 ));
  config.push_back(Trend( siStripTracker+"@TrackPx_RSTk@mean", "TrackPx_RSTk_mean.gif", 0, condition, "", Start, End, nRuns, -20, 20 ));
  config.push_back(Trend( siStripTracker+"@TrackPx_CosmicTk@mean", "TrackPx_CosmicTk_mean.gif", 0, condition, "", Start, End, nRuns, -20, 20 ));

  config.push_back(Trend( siStripTracker+"@TrackPy_CKFTk@mean,"
                         +siStripTracker+"@TrackPy_RSTk@mean,"
                         +siStripTracker+"@TrackPy_CosmicTk@mean",
                         "TrackPy_mean.gif", 0, condition, "CKFTk,RSTk,CosmicTk", Start, End, nRuns, -100, 100 ));
  config.push_back(Trend( siStripTracker+"@TrackPy_CKFTk@mean", "TrackPy_CKFTk_mean.gif", 0, condition, "", Start, End, nRuns, -100, 100 ));
  config.push_back(Trend( siStripTracker+"@TrackPy_RSTk@mean", "TrackPy_RSTk_mean.gif", 0, condition, "", Start, End, nRuns, -100, 100 ));
  config.push_back(Trend( siStripTracker+"@TrackPy_CosmicTk@mean", "TrackPy_CosmicTk_mean.gif", 0, condition, "", Start, End, nRuns, -100, 100 ));

  config.push_back(Trend( siStripTracker+"@TrackPz_CKFTk@mean,"
                         +siStripTracker+"@TrackPz_RSTk@mean,"
                         +siStripTracker+"@TrackPz_CosmicTk@mean",
                         "TrackPz_mean.gif", 0, condition, "CKFTk,RSTk,CosmicTk", Start, End, nRuns, -20, 20 ));
  config.push_back(Trend( siStripTracker+"@TrackPz_CKFTk@mean", "TrackPz_CKFTk_mean.gif", 0, condition, "", Start, End, nRuns, -20, 20 ));
  config.push_back(Trend( siStripTracker+"@TrackPz_RSTk@mean", "TrackPz_RSTk_mean.gif", 0, condition, "", Start, End, nRuns, -20, 20 ));
  config.push_back(Trend( siStripTracker+"@TrackPz_CosmicTk@mean", "TrackPz_CosmicTk_mean.gif", 0, condition, "", Start, End, nRuns, -20, 20 ));

  config.push_back(Trend( siStripTracker+"@TrackPhi_CKFTk@mean,"
                         +siStripTracker+"@TrackPhi_RSTk@mean,"
                         +siStripTracker+"@TrackPhi_CosmicTk@mean",
                         "TrackPhi_mean.gif", 0, condition, "CKFTk,RSTk,CosmicTk", Start, End, nRuns ));
  config.push_back(Trend( siStripTracker+"@TrackPhi_CKFTk@mean", "TrackPhi_CKFTk_mean.gif", 0, condition, "", Start, End, nRuns ));
  config.push_back(Trend( siStripTracker+"@TrackPhi_RSTk@mean", "TrackPhi_RSTk_mean.gif", 0, condition, "", Start, End, nRuns ));
  config.push_back(Trend( siStripTracker+"@TrackPhi_CosmicTk@mean", "TrackPhi_CosmicTk_mean.gif", 0, condition, "", Start, End, nRuns ));

  config.push_back(Trend( siStripTracker+"@TrackEta_CKFTk@mean,"
                         +siStripTracker+"@TrackEta_RSTk@mean,"
                         +siStripTracker+"@TrackEta_CosmicTk@mean",
                         "TrackEta_mean.gif", 0, condition, "CKFTk,RSTk,CosmicTk", Start, End, nRuns ));
  config.push_back(Trend( siStripTracker+"@TrackEta_CKFTk@mean", "TrackEta_CKFTk_mean.gif", 0, condition, "", Start, End, nRuns ));
  config.push_back(Trend( siStripTracker+"@TrackEta_RSTk@mean", "TrackEta_RSTk_mean.gif", 0, condition, "", Start, End, nRuns ));
  config.push_back(Trend( siStripTracker+"@TrackEta_CosmicTk@mean", "TrackEta_CosmicTk_mean.gif", 0, condition, "", Start, End, nRuns ));

  config.push_back(Trend( siStripTracker+"@DistanceOfClosestApproach_CKFTk@mean,"
                         +siStripTracker+"@DistanceOfClosestApproach_RSTk@mean,"
                         +siStripTracker+"@DistanceOfClosestApproach_CosmicTk@mean",
                         "DistanceOfClosestApproach_mean.gif", 0, condition, "CKFTk,RSTk,CosmicTk", Start, End, nRuns, -15, 15 ));
  config.push_back(Trend( siStripTracker+"@DistanceOfClosestApproach_CKFTk@mean", "DistanceOfClosestApproach_CKFTk_mean.gif", 0, condition, "", Start, End, nRuns, -15, 15 ));
  config.push_back(Trend( siStripTracker+"@DistanceOfClosestApproach_RSTk@mean", "DistanceOfClosestApproach_RSTk_mean.gif", 0, condition, "", Start, End, nRuns, -15, 15 ));
  config.push_back(Trend( siStripTracker+"@DistanceOfClosestApproach_CosmicTk@mean", "DistanceOfClosestApproach_CosmicTk_mean.gif", 0, condition, "", Start, End, nRuns, -15, 15 ));

  // Integral
  config.push_back(Trend( siStripTracker+"@Chi2oNDF_CKFTk@entries,"
                         +siStripTracker+"@Chi2oNDF_RSTk@entries,"
                         +siStripTracker+"@Chi2oNDF_CosmicTk@entries",
                         "Chi2oNDF_entries.gif", 0, condition, "CKFTk,RSTk,CosmicTk", Start, End, nRuns ));
  config.push_back(Trend( siStripTracker+"@Chi2oNDF_CKFTk@entries", "Chi2oNDF_CKFTk_entries.gif", 0, condition, "", Start, End, nRuns ));
  config.push_back(Trend( siStripTracker+"@Chi2oNDF_RSTk@entries", "Chi2oNDF_RSTk_entries.gif", 0, condition, "", Start, End, nRuns ));
  config.push_back(Trend( siStripTracker+"@Chi2oNDF_CosmicTk@entries", "Chi2oNDF_CosmicTk_entries.gif", 0, condition, "", Start, End, nRuns ));

  config.push_back(Trend( siStripTracker+"@NumberOfTracks_CKFTk@entries,"
                         +siStripTracker+"@NumberOfTracks_RSTk@entries,"
                         +siStripTracker+"@NumberOfTracks_CosmicTk@entries",
                         "NumberOfTracks_entries.gif", 0, condition, "CKFTk,RSTk,CosmicTk", Start, End, nRuns ));
  config.push_back(Trend( siStripTracker+"@NumberOfTracks_CKFTk@entries", "NumberOfTracks_CKFTk_entries.gif", 0, condition, "", Start, End, nRuns ));
  config.push_back(Trend( siStripTracker+"@NumberOfTracks_RSTk@entries", "NumberOfTracks_RSTk_entries.gif", 0, condition, "", Start, End, nRuns ));
  config.push_back(Trend( siStripTracker+"@NumberOfTracks_CosmicTk@entries", "NumberOfTracks_CosmicTk_entries.gif", 0, condition, "", Start, End, nRuns ));
  // config.push_back(Trend( siStripTracker+"@NumberOfTracks_CKFTk@entries", "NumberOfTracks_CKFTk_entries.gif", 0, condition, "", Start, End, nRuns ));
  // config.push_back(Trend( siStripTracker+"@NumberOfRecHitsPerTrack_CKFTk@entries", "NumberOfRecHitsPerTrack_CKFTk_entries.gif", 0, condition, "", Start, End, nRuns ));
*/


  // Creation of trends
  for_each(config.begin(), config.end(), makeTrend);

  // Close the output file
  makeTrend.closeFile();
}

void TrackingHDQMInspector( const string & dbName, const string & tagName, const string & password, const string & whiteListFile,
			    const int start, const int end )
{
  runTrackingInspector(dbName, tagName, password, whiteListFile, start, end, 0);
}

void TrackingHDQMInspector( const string & dbName, const string & tagName, const string & password, const string & whiteListFile,
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
    TrackingHDQMInspector( argv[1], argv[2], argv[3], argv[4], atoi(argv[5]) );
  } else if(argc == 7) {
    std::cout << "Creating trends for range:  " << argv[5] << " " << argv[6] << " for tag: " << argv[1] << std::endl;
    TrackingHDQMInspector( argv[1], argv[2], argv[3], argv[4], atoi(argv[5]), atoi(argv[6]) );
  }

  return 0;
}
