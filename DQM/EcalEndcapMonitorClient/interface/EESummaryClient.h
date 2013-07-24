#ifndef EESummaryClient_H
#define EESummaryClient_H

/*
 * \file EESummaryClient.h
 *
 * $Date: 2012/06/11 22:57:16 $
 * $Revision: 1.55 $
 * \author G. Della Ricca
 *
*/

#include <vector>
#include <string>

#include "TROOT.h"
#include "TProfile2D.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DQM/EcalEndcapMonitorClient/interface/EEClient.h"

class MonitorElement;
class DQMStore;
#ifdef WITH_ECAL_COND_DB
class EcalCondDBInterface;
class RunIOV;
class MonRunIOV;
#endif

class EESummaryClient : public EEClient {

public:

/// Constructor
EESummaryClient(const edm::ParameterSet& ps);

/// Destructor
virtual ~EESummaryClient();

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
inline void setFriends(const std::vector<EEClient*> &clients) { clients_ = clients; }

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
std::vector<int> ledWavelengths_;
std::vector<int> MGPAGains_;
std::vector<int> MGPAGainsPN_;

std::vector<EEClient*> clients_;

DQMStore* dqmStore_;

MonitorElement* meIntegrity_[2];
MonitorElement* meIntegrityPN_;
MonitorElement* meIntegrityErr_;
MonitorElement* meStatusFlags_[2];
MonitorElement* meStatusFlagsErr_;
MonitorElement* meOccupancy_[2];
MonitorElement* meOccupancyPN_;
MonitorElement* meOccupancy1D_;
MonitorElement* mePedestalOnline_[2];
MonitorElement* mePedestalOnlineErr_;
MonitorElement* mePedestalOnlineMean_;
MonitorElement* mePedestalOnlineRMS_;
MonitorElement* mePedestalOnlineRMSMap_[2];
MonitorElement* meLaserL1_[2];
MonitorElement* meLaserL1Err_;
MonitorElement* meLaserL1PN_;
MonitorElement* meLaserL1PNErr_;
MonitorElement* meLaserL1Ampl_; 
MonitorElement* meLaserL1Timing_;
MonitorElement* meLaserL1AmplOverPN_;
MonitorElement* meLaserL2_[2];
MonitorElement* meLaserL2Err_;
MonitorElement* meLaserL2PN_;
MonitorElement* meLaserL2PNErr_;
MonitorElement* meLaserL2Ampl_; 
MonitorElement* meLaserL2Timing_;
MonitorElement* meLaserL2AmplOverPN_;
MonitorElement* meLaserL3_[2];
MonitorElement* meLaserL3Err_;
MonitorElement* meLaserL3PN_;
MonitorElement* meLaserL3PNErr_;
MonitorElement* meLaserL3Ampl_; 
MonitorElement* meLaserL3Timing_;
MonitorElement* meLaserL3AmplOverPN_;
MonitorElement* meLaserL4_[2];
MonitorElement* meLaserL4Err_;
MonitorElement* meLaserL4PN_;
MonitorElement* meLaserL4PNErr_;
MonitorElement* meLaserL4Ampl_; 
MonitorElement* meLaserL4Timing_;
MonitorElement* meLaserL4AmplOverPN_;
MonitorElement* meLedL1_[2];
MonitorElement* meLedL1Err_;
MonitorElement* meLedL1PN_;
MonitorElement* meLedL1PNErr_;
MonitorElement* meLedL1Ampl_;
MonitorElement* meLedL1Timing_;
MonitorElement* meLedL1AmplOverPN_;
MonitorElement* meLedL2_[2];
MonitorElement* meLedL2Err_;
MonitorElement* meLedL2PN_;
MonitorElement* meLedL2PNErr_;
MonitorElement* meLedL2Ampl_;
MonitorElement* meLedL2Timing_;
MonitorElement* meLedL2AmplOverPN_;
MonitorElement* mePedestalG01_[2];
MonitorElement* mePedestalG06_[2];
MonitorElement* mePedestalG12_[2];
MonitorElement* mePedestalPNG01_;
MonitorElement* mePedestalPNG16_;
MonitorElement* meTestPulseG01_[2];
MonitorElement* meTestPulseG06_[2];
MonitorElement* meTestPulseG12_[2];
MonitorElement* meTestPulsePNG01_;
MonitorElement* meTestPulsePNG16_;
MonitorElement* meTestPulseAmplG01_;
MonitorElement* meTestPulseAmplG06_;
MonitorElement* meTestPulseAmplG12_;

MonitorElement* meRecHitEnergy_[2];
MonitorElement* meTiming_[2];
MonitorElement* meTimingMean1D_[2];
MonitorElement* meTimingRMS1D_[2];
MonitorElement* meTimingMean_;
MonitorElement* meTimingRMS_;
MonitorElement* meTriggerTowerEt_[2];
MonitorElement* meTriggerTowerEtSpectrum_[2];
MonitorElement* meTriggerTowerEmulError_[2];
MonitorElement* meTriggerTowerTiming_[2];
MonitorElement* meTriggerTowerNonSingleTiming_[2];

MonitorElement* meGlobalSummary_[2];

 MonitorElement* meSummaryErr_;

TProfile2D* hot01_[18];
TProfile2D* hpot01_[18];
TProfile2D* httt01_[18];
TProfile2D* htmt01_[18];
TH1F* norm01_, *synch01_;

 float timingNHitThreshold_;

};

#endif
