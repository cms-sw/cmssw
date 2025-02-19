#include "DQMServices/Diagnostic/interface/HDQMInspector.h"
#include "DQM/SiPixelHistoricInfoClient/interface/HDQMInspectorConfigSiPixel.h"
#include "DQMServices/Diagnostic/interface/DQMHistoryTrendsConfig.h"
#include "DQMServices/Diagnostic/interface/DQMHistoryCreateTrend.h"
#include <string>
#include <vector>
#include <memory>
#include <algorithm>

#include "FWCore/FWLite/interface/AutoLibraryLoader.h"
#include <TSystem.h>

using namespace std;

string const Condition = "0@SUMOFF_nclusters_OffTrack@yMean > 0";
string const BlackList = "";
// string const WhiteList = ""; // set a runs & range such as: "123,321,456,108000-109000";

/**
 * Extraction of the summary information using DQMServices/Diagnostic/test/HDQMInspector. <br>
 * The sqlite database should have been filled using the new SiPixelHistoryDQMService.   
 */
void runSiPixelInspector( const string & dbName, const string &tagName, const string & Password, const string & whiteListFile,
			  const int Start, const int End, const int nRuns )
{
  HDQMInspectorConfigSiPixel PixelConfig;
  DQMHistoryCreateTrend makeTrend(&PixelConfig);

  // Database and output configuration
  makeTrend.setDB(dbName,tagName,"/afs/cern.ch/cms/DB/conddb");
  makeTrend.setDebug(0);
  makeTrend.setDoStat(1);
  makeTrend.setSkip99s(true);
  makeTrend.setBlackList(BlackList);
  // makeTrend.setWhiteList(WhiteList);
  //makeTrend.setSeparator("@@#@@");  // TO change the seperator
  makeTrend.setWhiteListFromFile(whiteListFile);

  // Definition of trends
  typedef DQMHistoryTrendsConfig Trend;
  vector<Trend> config;
  config.push_back(Trend( "1@SUMOFF_adc@yMean", "adc_yMean_Barrel.gif", 0, Condition, "", Start, End, nRuns ));
  config.push_back(Trend( "6@SUMOFF_adc@yMean", "adc_yMean_Endcap.gif", 0, Condition, "", Start, End, nRuns ));
  config.push_back(Trend( "0@SUMOFF_charge_OffTrack@yMean", "charge_OffTrack_yMean.gif", 0, Condition, "", Start, End, nRuns ));
  config.push_back(Trend( "1@SUMOFF_charge_OnTrack@yMean", "charge_OnTrack_yMean_Barrel.gif", 0, Condition, "", Start, End, nRuns ));
  config.push_back(Trend( "6@SUMOFF_charge_OnTrack@yMean", "charge_OnTrack_yMean_Endcap.gif", 0, Condition, "", Start, End, nRuns ));
  config.push_back(Trend( "0@SUMOFF_nRecHits@yMean", "nRecHits_yMean.gif", 0, Condition, "", Start, End, nRuns ));
  config.push_back(Trend( "0@SUMOFF_nclusters_OffTrack@yMean", "nclusters_OffTrack_yMean.gif", 0, Condition, "", Start, End, nRuns ));
  config.push_back(Trend( "0@SUMOFF_nclusters_OnTrack@yMean", "nclusters_OnTrack_yMean.gif", 0, Condition, "", Start, End, nRuns ));
  config.push_back(Trend( "0@SUMOFF_ndigis@yMean", "ndigis_yMean.gif", 0, Condition, "", Start, End, nRuns ));
  config.push_back(Trend( "0@SUMOFF_size_OffTrack@yMean", "size_OffTrack_yMean.gif", 0, Condition, "", Start, End, nRuns ));
  config.push_back(Trend( "0@SUMOFF_size_OnTrack@yMean", "size_OnTrack_yMean.gif", 0, Condition, "", Start, End, nRuns ));
  config.push_back(Trend( "0@ntracks_generalTracks@NTracksPixOverAll", "NTracksPixOverAll.gif", 0, Condition, "", Start, End, nRuns ));
  config.push_back(Trend( "0@ntracks_generalTracks@NTracksFPixOverBPix", "NTracksFPixOverBPix.gif", 0, Condition, "", Start, End, nRuns ));

  config.push_back(Trend( "0@bigEventRate@yMean", "bigEventRate_yMean.gif", 0, Condition, "", Start, End, nRuns ));
  config.push_back(Trend( "0@bigFpixClusterEventRate@yMean", "bigFpixClusterEventRate_yMean.gif", 0, Condition, "", Start, End, nRuns ));
  config.push_back(Trend( "0@pixEventRate@yMean", "pixEventRate_yMean.gif", 0, Condition, "", Start, End, nRuns ));

  // Creation of trends
  for_each(config.begin(), config.end(), makeTrend);

  // Close the output file
  makeTrend.closeFile();
}

/// Simple method to create the trends. The actual operations are performed in runSiPixelInspector.
void SiPixelHDQMInspector( const string & dbName, const string &tagName, const string & password, const std::string & whiteListFile,
			   const int start, const int end )
{
  runSiPixelInspector( dbName, tagName, password, whiteListFile, start, end, 0 );
}

/// Simple method to create the trends. The actual operations are performed in runSiPixelInspector.
void SiPixelHDQMInspector( const string & dbName, const string & tagName, const string & password, const std::string & whiteListFile,
			   const int nRuns )
{
  runSiPixelInspector( dbName, tagName, password, whiteListFile, 0, 0, nRuns );
}

int main (int argc, char* argv[])
{
  gSystem->Load( "libFWCoreFWLite" );
  AutoLibraryLoader::enable();

  if (argc != 6 && argc != 7) {
    cerr << "Usage: " << argv[0] << " [Database] [TagName] [Password] [WhiteListFile] [NRuns] " << endl;
    cerr << "Or:    " << argv[0] << " [Database] [TagName] [Password] [WhiteListFile] [FirstRun] [LastRun] " << endl;
    return 1;
  }

  if (argc == 6) {
    cout << "Creating trends for NRuns = " << argv[5] << " for tag: " << argv[2] << endl;
    SiPixelHDQMInspector( argv[1], argv[2], argv[3], argv[4], atoi(argv[5]) );
  } else if(argc == 7) {
    cout << "Creating trends for range:  " << argv[5] << " " << argv[6] << " for tag: " << argv[2] << endl;
    SiPixelHDQMInspector( argv[1], argv[2], argv[3], argv[4], atoi(argv[5]), atoi(argv[6]) );
  }

  return 0;
}
