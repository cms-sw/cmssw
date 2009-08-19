#ifndef EBSummaryClient_H
#define EBSummaryClient_H

/*
 * \file EBSummaryClient.h
 *
 * $Date: 2009/08/02 15:46:36 $
 * $Revision: 1.49 $
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
std::vector<int> laserWavelengths_;
std::vector<int> MGPAGains_;
std::vector<int> MGPAGainsPN_;

std::vector<EBClient*> clients_;

DQMStore* dqmStore_;

MonitorElement* meIntegrity_;
MonitorElement* meIntegrityPN_;
MonitorElement* meIntegrityErr_;
MonitorElement* meStatusFlags_;
MonitorElement* meStatusFlagsErr_;
MonitorElement* meOccupancy_;
MonitorElement* meOccupancyPN_;
MonitorElement* meOccupancy1D_;
MonitorElement* mePedestalOnline_;
MonitorElement* mePedestalOnlineErr_;
MonitorElement* mePedestalOnlineMean_;
MonitorElement* mePedestalOnlineRMS_;
MonitorElement* mePedestalOnlineRMSMap_;
MonitorElement* meLaserL1_;
MonitorElement* meLaserL1Err_;
MonitorElement* meLaserL1Ampl_;
MonitorElement* meLaserL1Timing_;
MonitorElement* meLaserL1AmplOverPN_;
MonitorElement* meLaserL1PN_;
MonitorElement* meLaserL1PNErr_;
MonitorElement* meLaserL2_;
MonitorElement* meLaserL2Err_;
MonitorElement* meLaserL2Ampl_;
MonitorElement* meLaserL2Timing_;
MonitorElement* meLaserL2AmplOverPN_;
MonitorElement* meLaserL2PN_;
MonitorElement* meLaserL2PNErr_;
MonitorElement* meLaserL3_;
MonitorElement* meLaserL3Err_;
MonitorElement* meLaserL3Ampl_;
MonitorElement* meLaserL3Timing_;
MonitorElement* meLaserL3AmplOverPN_;
MonitorElement* meLaserL3PN_;
MonitorElement* meLaserL3PNErr_;
MonitorElement* meLaserL4_;
MonitorElement* meLaserL4Err_;
MonitorElement* meLaserL4Ampl_;
MonitorElement* meLaserL4Timing_;
MonitorElement* meLaserL4AmplOverPN_;
MonitorElement* meLaserL4PN_;
MonitorElement* meLaserL4PNErr_;
MonitorElement* mePedestalG01_;
MonitorElement* mePedestalG06_;
MonitorElement* mePedestalG12_;
MonitorElement* mePedestalPNG01_;
MonitorElement* mePedestalPNG16_;
MonitorElement* meTestPulseG01_;
MonitorElement* meTestPulseG06_;
MonitorElement* meTestPulseG12_;
MonitorElement* meTestPulsePNG01_;
MonitorElement* meTestPulsePNG16_;
MonitorElement* meTestPulseAmplG01_;
MonitorElement* meTestPulseAmplG06_;
MonitorElement* meTestPulseAmplG12_;

MonitorElement* meCosmic_;
MonitorElement* meTiming_;
MonitorElement* meTriggerTowerEt_;
MonitorElement* meTriggerTowerEmulError_;
MonitorElement* meTriggerTowerTiming_;
MonitorElement* meTriggerTowerNonSingleTiming_;

MonitorElement* meGlobalSummary_;

TProfile2D* hpot01_[36];
TProfile2D* httt01_[36];

};

#endif
