#ifndef EBLaserClient_H
#define EBLaserClient_H

/*
 * \file EBLaserClient.h
 *
 * $Date: 2010/02/14 20:56:22 $
 * $Revision: 1.92 $
 * \author G. Della Ricca
 *
*/

#include <vector>
#include <string>

#include "TROOT.h"
#include "TProfile2D.h"
#include "TH1F.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DQM/EcalBarrelMonitorClient/interface/EBClient.h"

class MonitorElement;
class DQMStore;
#ifdef WITH_ECAL_COND_DB
class EcalCondDBInterface;
class RunIOV;
class MonRunIOV;
#endif

class EBLaserClient : public EBClient {

friend class EBSummaryClient;

public:

/// Constructor
EBLaserClient(const edm::ParameterSet& ps);

/// Destructor
virtual ~EBLaserClient();

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

DQMStore* dqmStore_;

TProfile2D* h01_[36];
TProfile2D* h02_[36];
TProfile2D* h03_[36];
TProfile2D* h04_[36];
TProfile2D* h05_[36];
TProfile2D* h06_[36];
TProfile2D* h07_[36];
TProfile2D* h08_[36];

TProfile2D* h09_[36];
TProfile2D* h10_[36];
TProfile2D* h11_[36];
TProfile2D* h12_[36];

TProfile2D* hs01_[36];
TProfile2D* hs02_[36];
TProfile2D* hs03_[36];
TProfile2D* hs04_[36];

MonitorElement* meg01_[36];
MonitorElement* meg02_[36];
MonitorElement* meg03_[36];
MonitorElement* meg04_[36];

MonitorElement* meg05_[36];
MonitorElement* meg06_[36];
MonitorElement* meg07_[36];
MonitorElement* meg08_[36];
MonitorElement* meg09_[36];
MonitorElement* meg10_[36];
MonitorElement* meg11_[36];
MonitorElement* meg12_[36];

MonitorElement* mea01_[36];
MonitorElement* mea02_[36];
MonitorElement* mea03_[36];
MonitorElement* mea04_[36];

MonitorElement* met01_[36];
MonitorElement* met02_[36];
MonitorElement* met03_[36];
MonitorElement* met04_[36];

MonitorElement* metav01_[36];
MonitorElement* metav02_[36];
MonitorElement* metav03_[36];
MonitorElement* metav04_[36];

MonitorElement* metrms01_[36];
MonitorElement* metrms02_[36];
MonitorElement* metrms03_[36];
MonitorElement* metrms04_[36];

MonitorElement* meaopn01_[36];
MonitorElement* meaopn02_[36];
MonitorElement* meaopn03_[36];
MonitorElement* meaopn04_[36];

MonitorElement* mepnprms01_[36];
MonitorElement* mepnprms02_[36];
MonitorElement* mepnprms03_[36];
MonitorElement* mepnprms04_[36];
MonitorElement* mepnprms05_[36];
MonitorElement* mepnprms06_[36];
MonitorElement* mepnprms07_[36];
MonitorElement* mepnprms08_[36];

MonitorElement* me_hs01_[36];
MonitorElement* me_hs02_[36];
MonitorElement* me_hs03_[36];
MonitorElement* me_hs04_[36];
MonitorElement* me_hs05_[36];
MonitorElement* me_hs06_[36];
MonitorElement* me_hs07_[36];
MonitorElement* me_hs08_[36];

TProfile* i01_[36];
TProfile* i02_[36];
TProfile* i03_[36];
TProfile* i04_[36];
TProfile* i05_[36];
TProfile* i06_[36];
TProfile* i07_[36];
TProfile* i08_[36];
TProfile* i09_[36];
TProfile* i10_[36];
TProfile* i11_[36];
TProfile* i12_[36];
TProfile* i13_[36];
TProfile* i14_[36];
TProfile* i15_[36];
TProfile* i16_[36];

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
