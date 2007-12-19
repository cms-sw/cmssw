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
#include"DQMServices/Core/interface/DaqMonitorBEInterface.h"
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
  std::map<uint16_t, bool> foundFeds;

  // event counter
  int eventCounter;
  vector<BinCounters*> bc; //indexes APV Error BinCounters with FedId #

  // percentage stuff
  vector<double> apvPrct;
  vector<double>::iterator pi;

  // Nick Plot 1 - 2D - FED vs. Evt No. filling on every OOS
  MonitorElement * oosFedEvent;
       	
  // back-end interface
  DaqMonitorBEInterface * dbe;

  // vector for APV error and accomanying binCounters
  vector<MonitorElement*> ApveErr;   // Indexes APV Error Histograms with FedId #
  vector<BinCounters*> ApveErrCount; // Indexes APV Error BinCounters with FedId #

  // vector for fe majority apv error checking
  vector<MonitorElement*> FeMajApvErr;   // Indexes APV Error Histograms with FedId #
  vector<BinCounters*> FeMajApvErrCount; // Indexes APV Error BinCounters with FedId #

  // vector to hold the FE Synch Out Packet Values
  vector<vector<unsigned long> > FsopLong;
  vector<uint16_t> FsopShort; 
	
  // vectors for FEFPGA APVErrorB<APV0> status bits
  vector<vector<vector<MonitorElement*> > > FiberStatusBits;  // Indexes Histograms with FedId # per FE FPGA
  vector<vector<vector<BinCounters*> > > FiberStatusBitCount; // Indexes BinCounters with FedId # per FEFPGA

  // fiber wrong header error histograms
  vector<vector<MonitorElement*> > FiberWHApv; // Indexes Histograms with FedId # per Fiber
  vector<MonitorElement*> FeWHApv;  // Indexes APV Error Histograms with FedId # per FPGA
  vector<MonitorElement*> FeLKErr;  // Indexes LK Error Histograms with FedId # per FPGA
  vector<MonitorElement*> FeSYErr;  // Indexes SY Error Histograms with FedId # per FPGA
  vector<MonitorElement*> FeRWHErr; // Indexes RAW wrong header Error Histograms with FedId # per FPGA

  // ME for the preliminary check of APV address accross feds (Mersi Plot 1)
  // for now write addreses to the histo and check to see that its a flat line 
  MonitorElement *  AddCheck0;

  // ME for % of FEs in synch globally over event number (Mersi Plot 2)
  MonitorElement * AddConstPerEvent;
  MonitorElement * ApvAddConstPerEvent;
  MonitorElement * ApvAddConstPerEvent1;
  MonitorElement * ApvAddConstPerEvent2;
  MonitorElement * NoLock;
  MonitorElement * BadHead;
  MonitorElement * NoSynch;

  int oos;
  int nolock;
  int goodFe;	
  double prct;

  // ME Cumulative number of address errors per FED 
  MonitorElement * CumNumber;
  MonitorElement * CumNumber1;
  MonitorElement * CumNumber2; // Lock per fed
  MonitorElement * CumNumber3; // Sych per fed
  MonitorElement * CumNumber4; // Raw header error per fed
	
  // Set to 0 for buffer and non zero for FRL - SLINK readout - compensates for additional DQA headers, etc.
  // (K. Hahn request)
	
  int swapOn_;

  int garb_;

  // Number for event info for plots - 23 indicates simulation
  int runNumber_;

  // Histogram presentation variables	
	
  int percent_; // Gives us percent readout
  int N; // the modulo parameter

  // ApveError % 
  float apveErrorPercent;
	
  // APVerrorB<APV0> for FE 8 %
  float fe8apverrorBapv0Percent;

  // Name of output file
  string fileName_;

  // Nick's functioxb

  MonitorElement * goodAPVsPerEvent_;
  int APVProblemCounter_;

  // for Steve
  // for the Out of Synch Per Fed Per Event
  vector<MonitorElement*> OosPerFed;

  // vector of addresses to get median value for "golden address" which should match the apve address
  vector<uint16_t> feMedianAddr;
  uint16_t medianAddr; 	

  vector<vector<int> > WHError;
  vector<vector<int> > LKError;
  vector<vector<int> > SYError;
  vector<vector<int> > RWHError;

  vector<vector<uint16_t> > feMajorAddress;
  int feEnabledCount;
  int feEnable;

  int badApvCounter;
  int goodApvCounter;

  int fedCounter;

  std::vector<uint16_t> fedIds_;
  vector<vector<MonitorElement*> > errors;

  bool firstEvent_;

  void histoNaming( const std::vector<uint16_t>& fed_ids, const int& runNumber );

  void createRootFedHistograms( const int& runNumber );
  void createDetailedFedHistograms( uint16_t fed_id, const int& runNumber );

  // The first and last valid FedID for the Tracker
  std::pair<int,int> fedIdBoundaries_;
  int totalNumberOfFeds_;

  // Generic Monitor Elements
  MonitorElement* fedGenericErrors_;
  MonitorElement* fedFreeze_;
  MonitorElement* fedBx_;

  // Monitor Elements per FED
  std::map<int, MonitorElement* > feOverFlow_;
  std::map<int, MonitorElement* > feAPVAddr_;
  std::map<int, MonitorElement* > chanErrUnlock_;
  std::map<int, MonitorElement* > chanErrOOS_;
  std::map<int, MonitorElement* > badApv_;
  
};
