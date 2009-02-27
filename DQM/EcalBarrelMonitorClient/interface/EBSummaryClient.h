#ifndef EBSummaryClient_H
#define EBSummaryClient_H

/*
 * \file EBSummaryClient.h
 *
 * $Date: 2009/02/27 12:31:29 $
 * $Revision: 1.42 $
 * \author G. Della Ricca
 *
*/

#include <vector>
#include <string>
#include <fstream>

#include "TROOT.h"
#include "TProfile2D.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DQM/EcalBarrelMonitorClient/interface/EBClient.h"

class MonitorElement;
class DQMStore;
class EcalCondDBInterface;
class RunIOV;
class MonRunIOV;

class EBSummaryClient : public EBClient {

public:

/// Constructor
EBSummaryClient(const edm::ParameterSet& ps);

/// Destructor
virtual ~EBSummaryClient();

/// Analyze
void analyze(void);

/// BeginJob
void beginJob(DQMStore* dqmStore);

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

/// SoftReset
void softReset(bool flag);

/// WriteDB
bool writeDb(EcalCondDBInterface* econn, RunIOV* runiov, MonRunIOV* moniov, bool& status, bool flag);

/// Get Functions
inline int getEvtPerJob() { return ievt_; }
inline int getEvtPerRun() { return jevt_; }

/// Set Clients
inline void setFriends(std::vector<EBClient*> clients) { clients_ = clients; }

private:

int ievt_;
int jevt_;

bool cloneME_;

bool verbose_;
bool debug_;

std::string prefixME_;

bool enableCleanup_;

std::vector<int> superModules_;

std::vector<EBClient*> clients_;

DQMStore* dqmStore_;

MonitorElement* meIntegrity_;
MonitorElement* meIntegrityErr_;
MonitorElement* meStatusFlags_;
MonitorElement* meStatusFlagsErr_;
MonitorElement* meOccupancy_;
MonitorElement* meOccupancy1D_;
MonitorElement* mePedestalOnline_;
MonitorElement* mePedestalOnlineErr_;
MonitorElement* mePedestalOnlineMean_;
MonitorElement* mePedestalOnlineRMS_;
MonitorElement* mePedestalOnlineRMSMap_;
MonitorElement* meLaserL1_;
MonitorElement* meLaserL1Err_;
MonitorElement* meLaserL1AmplOverPN_;
MonitorElement* meLaserL1Timing_;
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
MonitorElement* meTestPulseAmplG01_;
MonitorElement* meTestPulseAmplG06_;
MonitorElement* meTestPulseAmplG12_;

MonitorElement* meCosmic_;
MonitorElement* meTiming_;
MonitorElement* meTriggerTowerEt_;
MonitorElement* meTriggerTowerEmulError_;
MonitorElement* meTriggerTowerTiming_;

MonitorElement* meGlobalSummary_;

TProfile2D* hpot01_[36];
TProfile2D* httt01_[36];

};

#endif
