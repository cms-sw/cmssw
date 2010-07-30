#include "DQMServices/Diagnostic/interface/HDQMInspector.h"
#include "DQM/SiStripHistoricInfoClient/interface/HDQMInspectorConfigTracking.h"
#include "DQMServices/Diagnostic/interface/DQMHistoryTrendsConfig.h"
#include "DQMServices/Diagnostic/interface/DQMHistoryCreateTrend.h"
#include <string>
#include <fstream>
#include <boost/algorithm/string.hpp>

#include "FWCore/FWLite/interface/AutoLibraryLoader.h"
#include <TROOT.h>
#include <TFile.h>
#include <TSystem.h>

using namespace std;

/**
 * Extraction of the summary information using DQMServices/Diagnostic/test/HDQMInspector. <br>
 * The sqlite database should have been filled using the new TrackingHistoryDQMService.   
 */
void runTrackingInspector( const string & dbName, const string &tagName, const string & Password, const string & whiteListFile,
			   const string & selectedTrends, const int Start, const int End )
{
  // IMPORTANT SETTINGS:
  // string siStripTracker = "268435456";
  // string condition = siStripTracker+"@Chi2oNDF_GenTk@entries > 100";  // Use for collision data
  string condition = "";  // Use for collision data
  string blackList = "";
  // -------------------

  HDQMInspectorConfigTracking trackingConfig;
  // Select quantities you want the integral of
  vector<string> ItemsForIntegration;
  ItemsForIntegration.push_back("Chi2oNDF_GenTk_entries");
  ItemsForIntegration.push_back("NumberOfTracks_GenTk_entries");
  trackingConfig.computeIntegralList(ItemsForIntegration);
  // Create the functor
  DQMHistoryCreateTrend makeTrend(&trackingConfig);

  // Database and output configuration
  makeTrend.setDB(dbName,tagName,"cms_dqm_31x_offline", Password,"");
  makeTrend.setDebug(1);
  makeTrend.setDoStat(1);
  makeTrend.setSkip99s(true);
  makeTrend.setBlackList(blackList);
  makeTrend.setWhiteListFromFile(whiteListFile);

  // Definition of trends
  typedef DQMHistoryTrendsConfig Trend;
  vector<Trend> config;



  std::string trendsFileName(selectedTrends);
  std::ifstream trendsFile;
  trendsFile.open(trendsFileName.c_str());
  if( !trendsFile ) {
    std::cout << "Error: trends configuration file: " << trendsFileName << " not found" << std::endl;
    exit(1);
  }
  std::string configLine;
  while( !trendsFile.eof() ) {
    std::string line;
    getline(trendsFile, line);
    if( line != "" ) {
      std::vector<std::string> strs;
      boost::split(strs, line, boost::is_any_of(" "));
      if( strs.size() == 4 ) {
	int nums[3] = {0,0,0};
	for( int i=0; i<3; ++i ) {
	  stringstream ss;
	  ss << strs[i+1];
	  ss >> nums[i];
	}
	config.push_back(Trend( strs[0], strs[0]+".gif", nums[0], condition, "", nums[1], nums[2] ));
      }
      else {
	std::cout << "Warning: trend configuration line: " << line << " is not formatted correctly. It will be skipped." << std::endl;
      }
    }
  }

  // Creation of trends
  for_each(config.begin(), config.end(), makeTrend);

  // Close the output file
  makeTrend.closeFile();
}

int main (int argc, char* argv[])
{
  // load framework libraries
  gSystem->Load( "libFWCoreFWLite" );
  AutoLibraryLoader::enable();

  if (argc != 8) {
    std::cerr << argv[0] << " [Database] [TagName] [Password] [WhiteListFile] [SelectedTrends] [FirstRun] [LastRun] " << std::endl;
    return 1;
  }

  std::cout << "Creating trends for range:  " << argv[6] << " " << argv[7] << " for tag: " << argv[1] << std::endl;
  runTrackingInspector( argv[1], argv[2], argv[3], argv[4], argv[5], atoi(argv[6]), atoi(argv[7]) );

  return 0;
}
