#ifndef EBSummaryClient_H
#define EBSummaryClient_H

/*
 * \file EBSummaryClient.h
 *
 * $Date: 2007/12/18 13:58:03 $
 * $Revision: 1.24 $
 * \author G. Della Ricca
 *
*/

#include <vector>
#include <string>
#include <fstream>

#include "TROOT.h"
#include "TProfile2D.h"
#include "TH1F.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DQM/EcalBarrelMonitorClient/interface/EBClient.h"

class MonitorElement;
class MonitorUserInterface;
class DaqMonitorBEInterface;
class EcalCondDBInterface;
class RunIOV;
class MonRunIOV;

class EBSummaryClient : public EBClient {

public:

/// Constructor
EBSummaryClient(const edm::ParameterSet& ps);

/// Destructor
virtual ~EBSummaryClient();

/// softReset
void softReset(void);

/// Analyze
void analyze(void);

/// BeginJob
void beginJob(MonitorUserInterface* mui);

/// EndJob
void endJob(void);

/// BeginRun
void beginRun(void);

/// EndRun
void endRun(void);

/// Setup
void setup(void);

/// Cleanup
void cleanup(void);

/// HtmlOutput
void htmlOutput(int run, std::string htmlDir, std::string htmlName);

/// WriteDB
bool writeDb(EcalCondDBInterface* econn, RunIOV* runiov, MonRunIOV* moniov);

/// Get Functions
inline int getEvtPerJob() { return ievt_; }
inline int getEvtPerRun() { return jevt_; }

/// Set Clients
inline void setFriends(std::vector<EBClient*> clients) { clients_ = clients; }

private:

void writeMap( std::ofstream& hf, std::string mapname );

int ievt_;
int jevt_;

bool cloneME_;

bool verbose_;

bool enableMonitorDaemon_;

bool enableCleanup_;

std::string prefixME_;

std::vector<int> superModules_;

std::vector<EBClient*> clients_;

MonitorUserInterface* mui_;
DaqMonitorBEInterface* dbe_;

MonitorElement* meIntegrity_;
MonitorElement* meIntegrityErr_;
MonitorElement* meOccupancy_;
MonitorElement* meOccupancy1D_;
MonitorElement* mePedestalOnline_;
MonitorElement* mePedestalOnlineErr_;
MonitorElement* meLaserL1_;
MonitorElement* meLaserL1Err_;
MonitorElement* meLaserL1PN_;
MonitorElement* meLaserL1PNErr_;
MonitorElement* mePedestal_;
MonitorElement* mePedestalErr_;
MonitorElement* mePedestalPN_;
MonitorElement* mePedestalPNErr_;
MonitorElement* meTestPulse_;
MonitorElement* meTestPulseErr_;
MonitorElement* meTestPulsePN_;
MonitorElement* meTestPulsePNErr_;

MonitorElement* meCosmic_;
MonitorElement* meTiming_;
MonitorElement* meTriggerTowerEt_;
MonitorElement* meTriggerTowerEmulError_;

MonitorElement* meGlobalSummary_;

};

#endif
