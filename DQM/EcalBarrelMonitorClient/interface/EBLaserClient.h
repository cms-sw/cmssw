#ifndef EBLaserClient_H
#define EBLaserClient_H

/*
 * \file EBLaserClient.h
 *
 * $Date: 2008/03/15 14:50:54 $
 * $Revision: 1.74 $
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
class EcalCondDBInterface;
class RunIOV;
class MonRunIOV;

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
void beginJob(DQMStore* mui);

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
void htmlOutput(int run, std::string& htmlDir, std::string& htmlName);

/// WriteDB
bool writeDb(EcalCondDBInterface* econn, RunIOV* runiov, MonRunIOV* moniov);

/// Get Functions
inline int getEvtPerJob() { return ievt_; }
inline int getEvtPerRun() { return jevt_; }

private:

int ievt_;
int jevt_;

bool cloneME_;

bool debug_;

bool enableCleanup_;

std::vector<int> superModules_;

DQMStore* dbe_;

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

TProfile2D* h13_[36];
TProfile2D* h14_[36];
TProfile2D* h15_[36];
TProfile2D* h16_[36];
TProfile2D* h17_[36];
TProfile2D* h18_[36];
TProfile2D* h19_[36];
TProfile2D* h20_[36];

TProfile2D* h21_[36];
TProfile2D* h22_[36];
TProfile2D* h23_[36];
TProfile2D* h24_[36];

TProfile2D* hs01_[36];
TProfile2D* hs02_[36];
TProfile2D* hs03_[36];
TProfile2D* hs04_[36];

TProfile2D* hs05_[36];
TProfile2D* hs06_[36];
TProfile2D* hs07_[36];
TProfile2D* hs08_[36];

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
MonitorElement* mea05_[36];
MonitorElement* mea06_[36];
MonitorElement* mea07_[36];
MonitorElement* mea08_[36];

MonitorElement* met01_[36];
MonitorElement* met02_[36];
MonitorElement* met03_[36];
MonitorElement* met04_[36];
MonitorElement* met05_[36];
MonitorElement* met06_[36];
MonitorElement* met07_[36];
MonitorElement* met08_[36];

MonitorElement* metav01_[36];
MonitorElement* metav02_[36];
MonitorElement* metav03_[36];
MonitorElement* metav04_[36];
MonitorElement* metav05_[36];
MonitorElement* metav06_[36];
MonitorElement* metav07_[36];
MonitorElement* metav08_[36];

MonitorElement* metrms01_[36];
MonitorElement* metrms02_[36];
MonitorElement* metrms03_[36];
MonitorElement* metrms04_[36];
MonitorElement* metrms05_[36];
MonitorElement* metrms06_[36];
MonitorElement* metrms07_[36];
MonitorElement* metrms08_[36];

MonitorElement* meaopn01_[36];
MonitorElement* meaopn02_[36];
MonitorElement* meaopn03_[36];
MonitorElement* meaopn04_[36];
MonitorElement* meaopn05_[36];
MonitorElement* meaopn06_[36];
MonitorElement* meaopn07_[36];
MonitorElement* meaopn08_[36];

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

// Quality check on PNs

float amplitudeThresholdPnG01_;
float amplitudeThresholdPnG16_;
float pedPnExpectedMean_[2];
float pedPnDiscrepancyMean_[2];
float pedPnRMSThreshold_[2];

};

#endif
