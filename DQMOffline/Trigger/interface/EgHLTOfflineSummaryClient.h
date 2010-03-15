#ifndef DQMOFFLINE_TRIGGER_EGHLTOFFLINESUMMARYCLIENT
#define DQMOFFLINE_TRIGGER_EGHLTOFFLINESUMMARYCLIENT

// -*- C++ -*-
//
// Package:    EgammaHLTOfflineSummaryClient
// Class:      EgammaHLTOffline
// 
/*
 Description: This module makes the summary histogram of the E/g HLT offline

 Notes:
   this takes the results of the quality tests and produces a module summarising each one. There are two summary histograms, one with each E/g trigger which is either green or red and one eta/phi bad/good region

*/
//
// Original Author:  Sam Harper
//         Created:  March 2009
// 
//
//

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include <vector>
#include <string>

class DQMStore;
class MonitorElement;


class EgHLTOfflineSummaryClient : public edm::EDAnalyzer {

public:
  struct SumHistBinData {
    std::string name;
    std::vector<std::string> qTestPatterns;
  };

 private:
  DQMStore* dbe_; //dbe seems to be the standard name for this, I dont know why. We of course dont own it
  std::string dirName_;
  std::string egHLTSumHistName_;

  std::vector<std::string> eleHLTFilterNames_;//names of the filters monitored using electrons to make plots for
  std::vector<std::string> phoHLTFilterNames_;//names of the filters monitored using photons to make plots for
  std::vector<std::string> egHLTFiltersToMon_;//names of the filters to include in summary histogram

  std::vector<std::string> eleHLTFilterNamesForSumBit_; //names of the filters to include in the summary bit
  std::vector<std::string> phoHLTFilterNamesForSumBit_; //names of the filters to include in the summary bit
  

  //the name of the bin label and the regex pattern to search for the quality tests to pass
  std::vector<SumHistBinData> egHLTSumHistXBins_; 
  std::vector<SumHistBinData> eleQTestsForSumBit_;
  std::vector<SumHistBinData> phoQTestsForSumBit_;

  bool runClientEndLumiBlock_;
  bool runClientEndRun_;
  bool runClientEndJob_;
  
  std::vector<std::string> egHLTFiltersToMonPaths_;
  bool usePathNames_;
  
  bool filterInactiveTriggers_;
  bool isSetup_;
  std::string hltTag_;
  

  //disabling copying/assignment (in theory this is copyable but lets not just in case)
  EgHLTOfflineSummaryClient(const EgHLTOfflineSummaryClient& rhs){}
  EgHLTOfflineSummaryClient& operator=(const EgHLTOfflineSummaryClient& rhs){return *this;}

 public:
  explicit EgHLTOfflineSummaryClient(const edm::ParameterSet& );
  virtual ~EgHLTOfflineSummaryClient();
  
  
  virtual void beginJob();
  virtual void analyze(const edm::Event&, const edm::EventSetup&); //dummy
  virtual void endJob();
  virtual void beginRun(const edm::Run& run, const edm::EventSetup& c);
  virtual void endRun(const edm::Run& run, const edm::EventSetup& c);
  
  
  virtual void beginLuminosityBlock(const edm::LuminosityBlock& lumiSeg,const edm::EventSetup& context){}
  // DQM Client Diagnostic
  virtual void endLuminosityBlock(const edm::LuminosityBlock& lumiSeg,const edm::EventSetup& c);



private:
  void runClient_(); //master function which runs the client  

  int getQTestBinData_(const edm::ParameterSet&);

  //takes a vector of strings of the form stringA:stringB and splits them into pairs containing stringA stringB
  void splitStringsToPairs_(const std::vector<std::string>& stringsToSplit,std::vector<std::pair<std::string,std::string> >& splitStrings);

  MonitorElement* getEgHLTSumHist_(); //makes our histogram
  //gets a list of filters we are monitoring
  //the reason we pass in ele and photon triggers seperately and then combine rather than passsing in a combined
  //list is to be able to share the declearation with the rest of the E/g HLT DQM Offline modules
  void getEgHLTFiltersToMon_(std::vector<std::string>& filterNames)const;



  //gets the quality tests for the filter matching pattern, if any of them fail it returns a 0, otherwise a 1
  //it does not care if the tests exist and in this situation will return a 1 (default to good)
  int getQTestResults_(const std::string& filterName,const std::vector<std::string>& pattern)const;
 
  static void fillQTestData_(const edm::ParameterSet& iConfig,std::vector<SumHistBinData>& qTests,const std::string& label);
};

#endif
