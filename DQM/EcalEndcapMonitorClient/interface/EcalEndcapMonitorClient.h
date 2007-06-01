#ifndef EcalEndcapMonitorClient_H
#define EcalEndcapMonitorClient_H

/*
 * \file EcalEndcapMonitorClient.h
 *
 * $Date: 2007/06/01 19:15:24 $
 * $Revision: 1.7 $
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

#include "DQMServices/Core/interface/MonitorUserInterface.h"
#include "OnlineDB/EcalCondDB/interface/RunIOV.h"
#include "OnlineDB/EcalCondDB/interface/MonRunIOV.h"

#include <DQM/EcalEndcapMonitorClient/interface/EEClient.h>

#include <DQM/EcalEndcapMonitorClient/interface/EESummaryClient.h>

#include "TROOT.h"
#include "TH1.h"

class EcalEndcapMonitorClient: public edm::EDAnalyzer{

public:

/// Constructor
EcalEndcapMonitorClient(const edm::ParameterSet & ps);
EcalEndcapMonitorClient(const edm::ParameterSet & ps, MonitorUserInterface* mui);

/// Destructor
~EcalEndcapMonitorClient();

/// Subscribe/Unsubscribe to Monitoring Elements
void subscribe(void);
void subscribeNew(void);
void unsubscribe(void);

/// softReset
void softReset(void);

// Initialize
void initialize(const edm::ParameterSet & ps);

/// Analyze
void analyze(void);
void analyze(const edm::Event & e, const edm::EventSetup & c);

/// BeginJob
void beginJob(void);
void beginJob(const edm::EventSetup & c){ this->beginJob(); }

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

/// BeginRunDB
void beginRunDb(void);

/// WriteDB
void writeDb(void);

/// EndRunDB
void endRunDb(void);

inline int                      getEvtPerJob()      { return( ievt_ ); }
inline int                      getEvtPerRun()      { return( jevt_ ); }
inline int                      getEvt( void )      { return( evt_ ); }
inline int                      getRun( void )      { return( run_ ); }
inline string                   getRunType( void )  { return( runtype_ == -1 ? "UNKNOWN" : runTypes_[runtype_] ); }
inline vector<string>           getRunTypes( void ) { return( runTypes_ ); }
inline const vector<EEClient*>  getClients()        { return( clients_ ); }
inline const vector<string>     getClientNames()    { return( clientNames_ ); }
inline RunIOV                   getRunIOV()         { return( runiov_ ); }
inline MonRunIOV                getMonIOV()         { return( moniov_ ); }
inline const TH1F*              getEntryHisto()     { return( h_ ); }

private:

int ievt_;
int jevt_;

bool collateSources_;
bool cloneME_;
bool enableQT_;

bool enableTCC_;
bool enableCluster_;
 
bool verbose_;

bool enableMonitorDaemon_;

string clientName_;

string prefixME_;

string hostName_;
int    hostPort_;

bool enableServer_;
int  serverPort_;
 
string inputFile_;
string referenceFile_;
string outputFile_;
 
string dbName_;
string dbHostName_;
int    dbHostPort_;
string dbUserName_;
string dbPassword_;

string maskFile_;

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

string baseHtmlDir_;

vector<int> superModules_;

typedef multimap<EEClient*,int> EECIMMap; 
EECIMMap chb_;
vector<string> runTypes_;
vector<EEClient*> clients_; 
vector<string> clientNames_; 

EESummaryClient* summaryClient_;

MonitorUserInterface* mui_;
 
bool enableStateMachine_;
 
string location_;
int    runtype_;
string status_;
int run_;
int evt_;
 
bool begin_run_;
bool end_run_;
 
bool forced_status_;
 
bool forced_update_;

bool enableExit_;
 
int last_update_;

int last_run_;
 
int last_jevt_;
 
int unknowns_;
 
CollateMonitorElement* me_h_;

TH1F* h_;

};

#endif
