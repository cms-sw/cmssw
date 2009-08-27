#include "DQMServices/Diagnostic/interface/HDQMInspector.h"
#include "DQM/SiPixelHistoricInfoClient/interface/HDQMInspectorConfigSiPixel.h"
#include "DQMServices/Diagnostic/interface/DQMHistoryTrendsConfig.h"
#include "DQMServices/Diagnostic/interface/DQMHistoryCreateTrend.h"
#include <string>
#include <vector>
#include <memory>
#include <algorithm>

using namespace std;

string const Condition = "0@SUMOFF_nclusters_OffTrack@yMean > 0";
string const BlackList = "";

/**
 * Extraction of the summary information using DQMServices/Diagnostic/test/HDQMInspector. <br>
 * The sqlite database should have been filled using the new SiPixelHistoryDQMService.   
 */
void runSiPixelInspector( const string &tagName, const string & Password, const int Start, const int End, const int nRuns )
{
  HDQMInspectorConfigSiPixel PixelConfig;
  DQMHistoryCreateTrend makeTrend(&PixelConfig);

  // Database and output configuration
  makeTrend.setDB("oracle://cms_orcoff_prep/CMS_DQM_31X_OFFLINE",tagName,"cms_dqm_31x_offline", Password,"");
  makeTrend.setDebug(0);
  makeTrend.setDoStat(1);
  makeTrend.setSkip99s(true);
  makeTrend.setBlackList(BlackList);

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
  config.push_back(Trend( "0@ntracks_rsWithMaterialTracksP5@NTracksPixOverAll", "NTracksPixOverAll.gif", 0, Condition, "", Start, End, nRuns ));
  config.push_back(Trend( "0@ntracks_rsWithMaterialTracksP5@NTracksFPixOverBPix", "NTracksFPixOverBPix.gif", 0, Condition, "", Start, End, nRuns ));

  // Creation of trends
  for_each(config.begin(), config.end(), makeTrend);

  // Close the output file
  makeTrend.closeFile();
}

/// Simple method to create the trends. The actual operations are performed in runSiPixelInspector.
void SiPixelHDQMInspector( const string &tagName, const string & password, const int start, const int end )
{
  runSiPixelInspector( tagName, password, start, end, 0 );
}

/// Simple method to create the trends. The actual operations are performed in runSiPixelInspector.
void SiPixelHDQMInspector( const string & tagName, const string & password, const int nRuns )
{
  runSiPixelInspector( tagName, password, 0, 0, nRuns );
}

int main (int argc, char* argv[])
{
  if (argc != 4 && argc != 5) {
    cerr << "Usage: " << argv[0] << " [TagName] [Password] [NRuns] " << endl;
    cerr << "Or:    " << argv[0] << " [TagName] [Password] [FirstRun] [LastRun] " << endl;
    return 1;
  }

  if (argc == 4) {
    cout << "Creating trends for NRuns = " << argv[3] << " for tag: " << argv[1] << endl;
    SiPixelHDQMInspector( argv[1], argv[2], atoi(argv[3]) );
  } else if(argc == 5) {
    cout << "Creating trends for range:  " << argv[3] << " " << argv[4] << " for tag: " << argv[1] << endl;
    SiPixelHDQMInspector( argv[1], argv[2], atoi(argv[3]), atoi(argv[4]) );
  }

  return 0;
}
