#ifndef EcalBarrelMonitorClient_H
#define EcalBarrelMonitorClient_H

/*
 * \file EcalBarrelMonitorClient.h
 *
 * $Date: 2008/04/08 15:06:21 $
 * $Revision: 1.103 $
 * \author G. Della Ricca
 * \author F. Cossutti
 *
*/

#include <string>
#include <vector>
#include <map>

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include <DQM/EcalBarrelMonitorClient/interface/EBClient.h>

#include <DQM/EcalBarrelMonitorClient/interface/EBSummaryClient.h>

#include "EventFilter/Utilities/interface/ModuleWeb.h"

#include "xgi/Input.h"
#include "xgi/Output.h"

#include "TROOT.h"
#include "TH1.h"

class DQMOldReceiver;
class DQMStore;
class RunIOV;
class MonRunIOV;

class EcalBarrelMonitorClient: public edm::EDAnalyzer, public evf::ModuleWeb{

public:

/// Constructor
EcalBarrelMonitorClient(const edm::ParameterSet & ps);

/// Destructor
~EcalBarrelMonitorClient();

// Initialize
void initialize(const edm::ParameterSet & ps);

/// Analyze
void analyze(void);
void analyze(const edm::Event & e, const edm::EventSetup & c);

/// BeginJob
void beginJob(const edm::EventSetup & c);

/// EndJob
void endJob(void);

/// BeginRun
void beginRun(void);
void beginRun(const edm::Run & r, const edm::EventSetup & c);

/// EndRun
void endRun(void);
void endRun(const edm::Run & r, const edm::EventSetup & c);

/// BeginLumiBlock
void beginLuminosityBlock(const edm::LuminosityBlock & l, const edm::EventSetup & c);

/// EndLumiBlock
void endLuminosityBlock(const edm::LuminosityBlock & l, const edm::EventSetup & c);

/// Setup
void setup(void);

/// Cleanup
void cleanup(void);

/// HtmlOutput
void htmlOutput(bool current=false);

/// XDAQ web page
void defaultWebPage(xgi::Input *in, xgi::Output *out);
void publish(xdata::InfoSpace *){};

/// BeginRunDB
void beginRunDb(void);

/// WriteDB
void writeDb(void);

/// EndRunDB
void endRunDb(void);

inline int getEvtPerJob() { return( ievt_ ); }
inline int getEvtPerRun() { return( jevt_ ); }
inline int getEvt( void ) { return( evt_ ); }
inline int getRun( void ) { return( run_ ); }

inline const char* getRunType( void ) { return( runType_ == -1 ? "UNKNOWN" : runTypes_[runType_].c_str() ); }

inline std::vector<std::string> getRunTypes( void ) { return( runTypes_ ); }

inline const std::vector<EBClient*>  getClients() { return( clients_ ); }
inline const std::vector<std::string> getClientsNames() { return( clientsNames_ ); }

inline RunIOV getRunIOV() { return( runiov_ ); }
inline MonRunIOV getMonIOV() { return( moniov_ ); }
inline const TH1F* getEntryHisto() { return( h_ ); }

private:

int ievt_;
int jevt_;

bool cloneME_;

bool verbose_;
bool debug_;

bool enableMonitorDaemon_;

bool enableCleanup_;

std::string clientName_;

std::string hostName_;
int         hostPort_;

std::string inputFile_;
 
std::string dbName_;
std::string dbHostName_;
int         dbHostPort_;
std::string dbUserName_;
std::string dbPassword_;

std::string maskFile_;

bool mergeRuns_;
 
RunIOV runiov_;
MonRunIOV moniov_;

bool enableSubRunDb_;
bool enableSubRunHtml_;
int subrun_;
 
time_t current_time_;
time_t last_time_db_;
time_t last_time_html_;
time_t dbRefreshTime_;
time_t htmlRefreshTime_;
 
std::string baseHtmlDir_;

std::vector<int> superModules_;

std::vector<std::string> enabledClients_;

std::multimap<EBClient*,int> clientsRuns_; 
std::vector<std::string> runTypes_;
std::vector<EBClient*> clients_; 
std::vector<std::string> clientsNames_; 
std::map<std::string,int> clientsStatus_;

EBSummaryClient* summaryClient_;

DQMOldReceiver* mui_;
DQMStore* dqmStore_;

std::string prefixME_;
 
bool enableUpdate_;
 
std::string location_;

int runType_;
int evtType_;

std::string status_;

int run_;
int evt_;
 
bool begin_run_;
bool end_run_;
 
bool forced_status_;
 
bool forced_update_;

int last_run_;
 
TH1F* h_;

};

#endif
