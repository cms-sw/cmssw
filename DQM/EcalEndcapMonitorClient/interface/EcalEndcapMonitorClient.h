#ifndef EcalEndcapMonitorClient_H
#define EcalEndcapMonitorClient_H

/*
 * \file EcalEndcapMonitorClient.h
 *
 * $Date: 2012/04/27 13:46:05 $
 * $Revision: 1.64 $
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

#include "DQM/EcalEndcapMonitorClient/interface/EEClient.h"

#include "DQM/EcalEndcapMonitorClient/interface/EESummaryClient.h"

#include "TROOT.h"
#include "TH1.h"

class DQMStore;
#ifdef WITH_ECAL_COND_DB
class RunIOV;
class MonRunIOV;
#endif

class EcalEndcapMonitorClient: public edm::EDAnalyzer{

friend class EcalEndcapMonitorXdaqClient;

public:

/// Constructor
EcalEndcapMonitorClient(const edm::ParameterSet & ps);

/// Destructor
virtual ~EcalEndcapMonitorClient();

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

inline const char* getRunType( void )  { return( runType_ == -1 ? "UNKNOWN" : runTypes_[runType_].c_str() ); }

private:

int ievt_;
int jevt_;

bool cloneME_;

bool verbose_;
bool debug_;

int prescaleFactor_;

bool enableCleanup_;

std::string prefixME_;

 bool produceReports_;

std::string inputFile_;
std::string referenceFile_;
 
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

std::multimap<EEClient*,int> clientsRuns_;
std::vector<std::string> runTypes_;
std::vector<EEClient*> clients_; 
std::vector<std::string> clientsNames_; 
std::map<std::string,int> clientsStatus_;

EESummaryClient* summaryClient_;

DQMStore* dqmStore_;
 
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
