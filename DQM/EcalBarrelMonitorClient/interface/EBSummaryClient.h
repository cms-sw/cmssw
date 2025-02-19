#ifndef EBSummaryClient_H
#define EBSummaryClient_H

/*
 * \file EBSummaryClient.h
 *
 * $Date: 2012/06/11 22:57:15 $
 * $Revision: 1.62 $
 * \author G. Della Ricca
 *
*/

#include <vector>
#include <string>

#include "TROOT.h"
#include "TProfile2D.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DQM/EcalBarrelMonitorClient/interface/EBClient.h"

class MonitorElement;
class DQMStore;
#ifdef WITH_ECAL_COND_DB
class EcalCondDBInterface;
class RunIOV;
class MonRunIOV;
#endif

class EBSummaryClient : public EBClient {

public:

/// Constructor
EBSummaryClient(const edm::ParameterSet& ps);

/// Destructor
virtual ~EBSummaryClient();

/// Analyze
void analyze(void);

/// BeginJob
void beginJob(void);

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

#ifdef WITH_ECAL_COND_DB
/// WriteDB
bool writeDb(EcalCondDBInterface* econn, RunIOV* runiov, MonRunIOV* moniov, bool& status);
#endif

/// Get Functions
inline int getEvtPerJob() { return ievt_; }
inline int getEvtPerRun() { return jevt_; }

/// Set Clients
inline void setFriends(std::vector<EBClient*> clients) { clients_ = clients; }

private:

int ievt_;
int jevt_;

float synchErrorThreshold_;

bool cloneME_;

bool verbose_;
bool debug_;

std::string prefixME_;

 std::string subfolder_;

bool enableCleanup_;

 bool produceReports_;

 bool reducedReports_;

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

MonitorElement* meRecHitEnergy_;
MonitorElement* meTiming_;
MonitorElement* meTimingMean1D_;
MonitorElement* meTimingRMS1D_;
MonitorElement* meTimingMean_;
MonitorElement* meTimingRMS_;
MonitorElement* meTriggerTowerEt_;
MonitorElement* meTriggerTowerEmulError_;
MonitorElement* meTriggerTowerTiming_;
MonitorElement* meTriggerTowerNonSingleTiming_;

MonitorElement* meGlobalSummary_;

 MonitorElement* meSummaryErr_;

TProfile2D* hot01_[36];
TProfile2D* hpot01_[36];
TProfile2D* httt01_[36];
TProfile2D* htmt01_[36];
TH1F* norm01_, *synch01_;

 int timingNHitThreshold_;

};

#endif
