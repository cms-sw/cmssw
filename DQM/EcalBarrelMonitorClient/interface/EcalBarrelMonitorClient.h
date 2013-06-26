#ifndef EcalBarrelMonitorClient_H
#define EcalBarrelMonitorClient_H

/*
 * \file EcalBarrelMonitorClient.h
 *
 * $Date: 2012/04/27 13:45:58 $
 * $Revision: 1.128 $
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

#include "DQM/EcalBarrelMonitorClient/interface/EBClient.h"

#include "DQM/EcalBarrelMonitorClient/interface/EBSummaryClient.h"

#include "TROOT.h"
#include "TH1.h"

class DQMStore;
#ifdef WITH_ECAL_COND_DB
class RunIOV;
class MonRunIOV;
#endif

class EcalBarrelMonitorClient: public edm::EDAnalyzer{

friend class EcalBarrelMonitorXdaqClient;

public:

/// Constructor
EcalBarrelMonitorClient(const edm::ParameterSet & ps);

/// Destructor
virtual ~EcalBarrelMonitorClient();

/// Analyze
void analyze(void);
void analyze(const edm::Event & e, const edm::EventSetup & c);

/// BeginJob
void beginJob(void);

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

/// Reset
void reset(void);

/// Setup
void setup(void);

/// Cleanup
void cleanup(void);

/// SoftReset
void softReset(bool flag);

/// BeginRunDB
void beginRunDb(void);

/// WriteDB
void writeDb(void);

/// EndRunDB
void endRunDb(void);

inline const char* getRunType( void ) { return( runType_ == -1 ? "UNKNOWN" : runTypes_[runType_].c_str() ); }

private:

int ievt_;
int jevt_;

bool cloneME_;

bool verbose_;
bool debug_;

int prescaleFactor_;

bool enableCleanup_;

std::string inputFile_;
 
std::string dbName_;
std::string dbHostName_;
int         dbHostPort_;
std::string dbUserName_;
std::string dbPassword_;

std::string dbTagName_;

std::string resetFile_;

bool mergeRuns_;

#ifdef WITH_ECAL_COND_DB 
RunIOV runiov_;
MonRunIOV moniov_;
#endif

int subrun_;
 
time_t current_time_;

time_t last_time_update_;
time_t last_time_reset_;

time_t updateTime_;
time_t dbUpdateTime_;
 
std::vector<int> superModules_;

std::vector<std::string> enabledClients_;

std::multimap<EBClient*,int> clientsRuns_; 
std::vector<std::string> runTypes_;
std::vector<EBClient*> clients_; 
std::vector<std::string> clientsNames_; 
std::map<std::string,int> clientsStatus_;

EBSummaryClient* summaryClient_;

DQMStore* dqmStore_;

std::string prefixME_;

 bool produceReports_;
 
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
