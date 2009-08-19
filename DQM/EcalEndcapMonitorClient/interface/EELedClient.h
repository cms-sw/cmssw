#ifndef EELedClient_H
#define EELedClient_H

/*
 * \file EELedClient.h
 *
 * $Date: 2009/06/29 13:27:49 $
 * $Revision: 1.31 $
 * \author G. Della Ricca
 *
*/

#include <vector>
#include <string>

#include "TROOT.h"
#include "TProfile2D.h"
#include "TH1F.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DQM/EcalEndcapMonitorClient/interface/EEClient.h"

class MonitorElement;
class DQMStore;
class EcalCondDBInterface;
class RunIOV;
class MonRunIOV;

class EELedClient : public EEClient {

friend class EESummaryClient;

public:

/// Constructor
EELedClient(const edm::ParameterSet& ps);

/// Destructor
virtual ~EELedClient();

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

private:

int ievt_;
int jevt_;

bool cloneME_;

bool verbose_;
bool debug_;

std::string prefixME_;

bool enableCleanup_;

std::vector<int> superModules_;
std::vector<int> ledWavelengths_;

DQMStore* dqmStore_;

TProfile2D* h01_[18];
TProfile2D* h02_[18];
TProfile2D* h03_[18];
TProfile2D* h04_[18];

TProfile2D* h09_[18];
TProfile2D* h10_[18];

TProfile2D* hs01_[18];
TProfile2D* hs02_[18];

MonitorElement* meg01_[18];
MonitorElement* meg02_[18];

MonitorElement* meg05_[18];
MonitorElement* meg06_[18];

MonitorElement* meg09_[18];
MonitorElement* meg10_[18];

MonitorElement* mea01_[18];
MonitorElement* mea02_[18];

MonitorElement* met01_[18];
MonitorElement* met02_[18];

MonitorElement* metav01_[18];
MonitorElement* metav02_[18];

MonitorElement* metrms01_[18];
MonitorElement* metrms02_[18];

MonitorElement* meaopn01_[18];
MonitorElement* meaopn02_[18];

MonitorElement* mepnprms01_[18];
MonitorElement* mepnprms02_[18];

MonitorElement* mepnprms05_[18];
MonitorElement* mepnprms06_[18];

MonitorElement* me_hs01_[18];
MonitorElement* me_hs02_[18];

TProfile* i01_[18];
TProfile* i02_[18];

TProfile* i05_[18];
TProfile* i06_[18];

TProfile* i09_[18];
TProfile* i10_[18];

TProfile* i13_[18];
TProfile* i14_[18];

// Quality check on crystals

float percentVariation_;
float amplitudeThreshold_; 
float rmsThresholdRelative_;

// Quality check on PNs

float amplitudeThresholdPnG01_;
float amplitudeThresholdPnG16_;
float pedPnExpectedMean_[2];
float pedPnDiscrepancyMean_[2];
float pedPnRMSThreshold_[2];

};

#endif
