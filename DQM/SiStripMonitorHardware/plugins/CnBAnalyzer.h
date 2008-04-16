// CMSSW Framework and related headers
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"  
#include "FWCore/Framework/interface/EventSetup.h"

// Data Formats
#include "DataFormats/Common/interface/Handle.h"

// Condition Formats
#include "CondFormats/SiStripObjects/interface/SiStripFedCabling.h"
#include "CondFormats/DataRecord/interface/SiStripFedCablingRcd.h"

// DQM headers as well as the service header
//#include"DQMServices/Core/interface/DaqMonitorBEInterface.h"
#include"DQMServices/Core/interface/MonitorElement.h"
#include"FWCore/ServiceRegistry/interface/Service.h"

#include <iostream>
#include <vector>
#include <map>
#include <sstream>
#include <string>

// BinCounter 
#include "DQM/SiStripMonitorHardware/interface/BinCounters.h"

using namespace std;
class DQMStore;

class CnBAnalyzer : public edm::EDAnalyzer {
 public:
  CnBAnalyzer(const edm::ParameterSet&);
  ~CnBAnalyzer();

 private:
  
  void beginJob(const edm::EventSetup&);
  void analyze(const edm::Event&, const edm::EventSetup&);
  void endJob();


  // A data structure to record
  // the found FEDs
  // It is set to true as soon as its plots are created
  std::map<uint16_t, bool> foundFeds_;

  // back-end interface
  //  DaqMonitorBEInterface* dbe;
  DQMStore* const dqm( std::string method = "" ) const;

  DQMStore* dqm_;

  // ME for % of FEs in synch globally over event number (Mersi Plot 2)
  MonitorElement* AddConstPerEvent;
  MonitorElement* ApvAddConstPerEvent;
  MonitorElement* ApvAddConstPerEvent1;
  MonitorElement* ApvAddConstPerEvent2;
  MonitorElement* NoLock;
  MonitorElement* BadHead;
  MonitorElement* NoSynch;

  // Set to 0 for buffer and non zero for FRL - SLINK readout - compensates for additional DQA headers, etc.
  // (K. Hahn request)
  int swapOn_;

  // Number for event info for plots

  // Name of output file
  std::string outputFileName_;
  std::string outputFileDir_;

  // vector of addresses to get median value for "golden address" which should match the apve address
  // TODO: add median calculation in wrong apv addresses
  // vector<uint16_t> feMedianAddr;
  // uint16_t medianAddr;
  // vector<vector<uint16_t> > feMajorAddress;

  // Avaliable FEDid vector (updated for each event)
  std::vector<uint16_t> fedIds_;

  // Functions to book histograms in the output
  void createRootFedHistograms();
  void createDetailedFedHistograms( const uint16_t& fed_id );

  // The first and last valid FedID for the Tracker
  std::pair<int,int> fedIdBoundaries_;
  int totalNumberOfFeds_;

  // Generic Monitor Elements
  MonitorElement* fedGenericErrors_;
  MonitorElement* fedFreeze_;
  MonitorElement* fedBx_;

  // Trend plots
  MonitorElement* totalChannels_;
  MonitorElement* faultyChannels_;

  // Monitor Elements per FED
  std::map<int, MonitorElement* > feOverFlow_;
  std::map<int, MonitorElement* > feAPVAddr_;
  std::map<int, MonitorElement* > chanErrUnlock_;
  std::map<int, MonitorElement* > chanErrOOS_;
  std::map<int, MonitorElement* > badApv_;

};


