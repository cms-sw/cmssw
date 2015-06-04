#include "DQMServices/Diagnostic/interface/HDQMInspector.h"
#include "DQM/SiStripHistoricInfoClient/interface/HDQMInspectorConfigTracking.h"
#include "DQMServices/Diagnostic/interface/DQMHistoryTrendsConfig.h"
#include "DQMServices/Diagnostic/interface/DQMHistoryCreateTrend.h"
#include <string>
#include <fstream>
#include <map>
#include <boost/algorithm/string.hpp>
#include <vector>
#include <sstream>
#include <fstream>
#include <iostream>
#include "FWCore/FWLite/interface/FWLiteEnabler.h"
#include <TROOT.h>
#include <TFile.h>
#include <TSystem.h>

using namespace std;

/**
 * Extraction of the summary information using DQMServices/Diagnostic/test/HDQMInspector. <br>
 * The sqlite database should have been filled using the new TrackingHistoryDQMService.   
 */



//struct for holding the data for super imposed plots
struct plotData
{
  plotData(const std::string & data)
  {
    std::vector<std::string> strs;
    boost::split(strs,data, boost::is_any_of(" "));
    names = strs[0];
    fill(strs[1], logY);
    fill(strs[2], firstRun);
    fill(strs[3], lastRun);
    fill(strs[5], minY);
    fill(strs[6], maxY);
    //fill(strs[8], condition);
    //fill(strs[7], runsQuality);
    
  }
 
plotData()
{}

  template<typename T>
  void fill(const std::string & value, T & var)
  {
    std::stringstream ss;
    ss << value;
    ss >> var;
  }

  std::string names;
  int firstRun;
  int lastRun;
  double minY;
  double maxY;
  int logY;
  string condition;
 // int runsQuality;
  	
};


std::string concatNames(const string & line)
{
  std::vector<std::string> nameVector;
  std::string concatValues="";
  boost::split(nameVector,line, boost::is_any_of(",@"));
  int time=0;	
  for (unsigned int i=0;i<nameVector.size();i++)
  {
    if (time==1)
    {	
      concatValues+=","+nameVector[i];
      time++;
    }
    else if (time==2)
    {
      concatValues+=nameVector[i];
      time=0;
    }
    else
    {
      time++;
    }
	
		
  }
  return concatValues;
}

void runInspector( const string & dbName, const string &tagName, const string & Password, const string & whiteListFile,const string & selectedTrends, const int Start, const int End , const string &CondList = "")
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
  makeTrend.setDB(dbName,tagName,"/afs/cern.ch/cms/DB/conddb");
  // makeTrend.setDB(dbName,tagName,"cms_dqm_31x_offline", Password,"");
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

  //Creating condition string
  if ( CondList != "" ){
    std::ifstream condFile;
    condFile.open(CondList.c_str());
    if ( condFile ){
      //Read the file
      bool first = true;
      while (!condFile.eof()){
        std::string line;
        getline(condFile, line);
        if ( line != ""){
          if (!first)
            condition+= " && ";
          else 
            first = false;
          condition+= line;
        }
      }
      cout << condition << endl; 
    }
    else
      std::cout << "Warning: File " << CondList << " not found : conditions will not be used" << std::endl; 
  }

  
  
  std::string configLine;

  typedef std::map<std::string,plotData> trendsMap;
  //map for holding the super imposed plots
  std::map<std::string,plotData> superImposedtrendsMap;
  //vector for holding the normal plots
  std::vector<plotData> trendsVector;
  //iterator for Map
  typedef std::map<std::string,plotData>::iterator trendsMapIter;


  while( !trendsFile.eof() ) 
  {
    std::string line;
    getline(trendsFile, line);
    if( line != "" ) 
    {
      std::vector<std::string> strs;
      boost::split(strs, line, boost::is_any_of(" "));
	
		
     if( strs.size() == 8 ) 
     {
       //Tracker name			
       std::string index=strs[4];	
       plotData plot(line);	
       if (index=="0")
         trendsVector.push_back(plot);
       else
       {
         pair<trendsMapIter, bool> insertResult = superImposedtrendsMap.insert(std::make_pair(index,plot));
         if(!insertResult.second) 
         {
           std::string newName=strs[0];
           superImposedtrendsMap[index].names += "," + newName;
         }
       }		
     }//if
     else 
     {
       std::cout << "Warning: trend configuration line: " << line << " is not formatted correctly. It will be skipped." << std::endl;
     }//else
	
    } //if
  } //while

  //creating super imposed plots
  for(map<std::string,plotData>::const_iterator it = superImposedtrendsMap.begin(); it != superImposedtrendsMap.end(); ++it)
  {
    plotData plot=it->second;
    if (plot.maxY<plot.minY)
    {
      plot.minY=0;		
      plot.maxY=100;
    }
    //pushing back the superimposed plots into the vector 
    config.push_back(Trend(plot.names, plot.names+".gif",plot.logY, condition, plot.names, plot.firstRun,plot.lastRun,0,plot.minY,plot.maxY));
  }  
  //creating normal plots 
  for (unsigned int i=0; i<trendsVector.size();i++)
  {
    plotData plot=trendsVector[i];
    printf("New Trend:\n%s %d %d %d %d %f %f\n",plot.names.c_str(),plot.logY, plot.firstRun,plot.lastRun,0,plot.minY,plot.maxY);
    config.push_back(Trend(plot.names, plot.names+".gif",plot.logY, condition, plot.names, plot.firstRun,plot.lastRun,0,plot.minY,plot.maxY));
  }    

  cout << "Test Web: Trends created. " << endl;
  // Creation of trend
  for_each(config.begin(), config.end(), makeTrend);
  cout << "Test Web: Trends maded" << endl;
  // Close the output file
  makeTrend.closeFile();
}



int main (int argc, char* argv[])
{
  // load framework libraries
  gSystem->Load( "libFWCoreFWLite" );
  FWLiteEnabler::enable();

  if (argc < 9) {
    std::cerr << argv[0] << " [Database] [TagName] [Password] [WhiteListFile] [SelectedTrends] [FirstRun] [LastRun] [CondList]" << std::endl;
  return 1;
  }

  std::cout << "Creating trends for range:  " << argv[6] << " " << argv[7] << " for tag: " << argv[1] << std::endl;
  runInspector( argv[1], argv[2], argv[3], argv[4], argv[5], atoi(argv[6]), atoi(argv[7]), argv[8] );
  return 0;
}
