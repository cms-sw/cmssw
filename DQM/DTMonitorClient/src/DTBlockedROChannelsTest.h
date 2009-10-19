#ifndef DTMonitorModule_DTBlockedROChannelsTest_H
#define DTMonitorModule_DTBlockedROChannelsTest_H

/** \class DTBlockedROChannelsTest
 * *
 *  DQM Client to Summarize LS by LS the status of the Read-Out channels.
 *
 *  $Date: 2009/06/10 10:03:41 $
 *  $Revision: 1.1 $
 *  \author G. Cerminara - University and INFN Torino
 *   
 */
#include <FWCore/Framework/interface/EDAnalyzer.h>
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include <FWCore/Framework/interface/Event.h>
#include "FWCore/Framework/interface/ESHandle.h"
#include <FWCore/Framework/interface/EventSetup.h>
#include <FWCore/Framework/interface/LuminosityBlock.h>
#include "DataFormats/MuonDetId/interface/DTChamberId.h"

class DQMStore;
class MonitorElement;
class DTReadOutMapping;
class DTTimeEvolutionHisto;

class DTBlockedROChannelsTest: public edm::EDAnalyzer{

public:

  /// Constructor
  DTBlockedROChannelsTest(const edm::ParameterSet& ps);

 /// Destructor
 ~DTBlockedROChannelsTest();

protected:

  /// BeginJob
  void beginJob(const edm::EventSetup& c);

  /// BeginRun
  void beginRun(const edm::Run& run, const edm::EventSetup& c);
 
  /// Analyze
  void analyze(const edm::Event& e, const edm::EventSetup& c);

  
  void endRun(edm::Run const& run, edm::EventSetup const& eSetup);
  /// Endjob
  void endJob();

  void beginLuminosityBlock(edm::LuminosityBlock const& lumiSeg, edm::EventSetup const& context) ;

  /// DQM Client Diagnostic
  void endLuminosityBlock(edm::LuminosityBlock const& lumiSeg, edm::EventSetup const& c);

private:
  int readOutToGeometry(int dduId, int rosNumber, int& wheel, int& sector);

private:

  //Number of onUpdates
  int nupdates;

  // prescale on the # of LS to update the test
  int prescaleFactor;

  int nevents;
  int neventsPrev;
  unsigned int nLumiSegs;
  unsigned int prevNLumiSegs;
  double prevTotalPerc;

  int run;


  DQMStore* dbe;
  edm::ESHandle<DTReadOutMapping> mapping;
  

  // Monitor Elements
  std::map<int, MonitorElement*> wheelHitos;
  MonitorElement *summaryHisto;

  bool offlineMode;

  std::map<int, double> resultsPerLumi;
  DTTimeEvolutionHisto* hSystFractionVsLS;


  class DTRobBinsMap {
  public:
    DTRobBinsMap(const int fed, const int ros, const DQMStore* dbe);

    DTRobBinsMap();


    ~DTRobBinsMap();
    
    // add a rob to the set of robs
    void addRobBin(int robBin);
    
    bool robChanged(int robBin);

    double getChamberPercentage();

    void readNewValues();
    
  private:
    int getValueRobBin(int robBin) const;

    std::map<int, int> robsAndValues;

    const MonitorElement* meROS;

    std::string hName;
    const DQMStore* theDbe;
  };

  std::map<DTChamberId, DTRobBinsMap> chamberMap;

 };

#endif
