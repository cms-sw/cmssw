#ifndef EELaserClient_H
#define EELaserClient_H

/*
 * \file EELaserClient.h
 *
 * $Date: 2007/08/09 14:36:54 $
 * $Revision: 1.4 $
 * \author G. Della Ricca
 *
*/

#include <vector>
#include <string>

#include "TROOT.h"
#include "TProfile2D.h"
#include "TH1F.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "OnlineDB/EcalCondDB/interface/EcalCondDBInterface.h"
#include "OnlineDB/EcalCondDB/interface/MonRunIOV.h"

#include "DQMServices/Core/interface/MonitorElement.h"
#include "DQMServices/Core/interface/MonitorUserInterface.h"
#include "DQMServices/Core/interface/DaqMonitorBEInterface.h"
#include "DQMServices/Core/interface/CollateMonitorElement.h"

#include "DQM/EcalEndcapMonitorClient/interface/EEClient.h"

class EELaserClient : public EEClient {

friend class EESummaryClient;

public:

/// Constructor
EELaserClient(const edm::ParameterSet& ps);

/// Destructor
virtual ~EELaserClient();

/// Subscribe/Unsubscribe to Monitoring Elements
void subscribe(void);
void subscribeNew(void);
void unsubscribe(void);

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
void htmlOutput(int run, string htmlDir, string htmlName);

/// WriteDB
bool writeDb(EcalCondDBInterface* econn, RunIOV* runiov, MonRunIOV* moniov);

/// Get Functions
inline int getEvtPerJob() { return ievt_; }
inline int getEvtPerRun() { return jevt_; }

private:

int ievt_;
int jevt_;

bool collateSources_;
bool cloneME_;
bool enableQT_;

bool verbose_;

bool enableMonitorDaemon_;

string prefixME_;

vector<int> superModules_;

MonitorUserInterface* mui_;
DaqMonitorBEInterface* dbe_;

CollateMonitorElement* me_h01_[18];
CollateMonitorElement* me_h02_[18];
CollateMonitorElement* me_h03_[18];
CollateMonitorElement* me_h04_[18];
CollateMonitorElement* me_h05_[18];
CollateMonitorElement* me_h06_[18];
CollateMonitorElement* me_h07_[18];
CollateMonitorElement* me_h08_[18];

CollateMonitorElement* me_h09_[18];
CollateMonitorElement* me_h10_[18];
CollateMonitorElement* me_h11_[18];
CollateMonitorElement* me_h12_[18];

CollateMonitorElement* me_h13_[18];
CollateMonitorElement* me_h14_[18];
CollateMonitorElement* me_h15_[18];
CollateMonitorElement* me_h16_[18];
CollateMonitorElement* me_h17_[18];
CollateMonitorElement* me_h18_[18];
CollateMonitorElement* me_h19_[18];
CollateMonitorElement* me_h20_[18];

CollateMonitorElement* me_h21_[18];
CollateMonitorElement* me_h22_[18];
CollateMonitorElement* me_h23_[18];
CollateMonitorElement* me_h24_[18];

CollateMonitorElement* me_hs01_[18];
CollateMonitorElement* me_hs02_[18];
CollateMonitorElement* me_hs03_[18];
CollateMonitorElement* me_hs04_[18];

CollateMonitorElement* me_hs05_[18];
CollateMonitorElement* me_hs06_[18];
CollateMonitorElement* me_hs07_[18];
CollateMonitorElement* me_hs08_[18];

TProfile2D* h01_[18];
TProfile2D* h02_[18];
TProfile2D* h03_[18];
TProfile2D* h04_[18];
TProfile2D* h05_[18];
TProfile2D* h06_[18];
TProfile2D* h07_[18];
TProfile2D* h08_[18];

TProfile2D* h09_[18];
TProfile2D* h10_[18];
TProfile2D* h11_[18];
TProfile2D* h12_[18];

TProfile2D* h13_[18];
TProfile2D* h14_[18];
TProfile2D* h15_[18];
TProfile2D* h16_[18];
TProfile2D* h17_[18];
TProfile2D* h18_[18];
TProfile2D* h19_[18];
TProfile2D* h20_[18];

TProfile2D* h21_[18];
TProfile2D* h22_[18];
TProfile2D* h23_[18];
TProfile2D* h24_[18];

MEContentsProf2DWithinRangeROOT* qth01_[18];
MEContentsProf2DWithinRangeROOT* qth02_[18];
MEContentsProf2DWithinRangeROOT* qth03_[18];
MEContentsProf2DWithinRangeROOT* qth04_[18];
MEContentsProf2DWithinRangeROOT* qth05_[18];
MEContentsProf2DWithinRangeROOT* qth06_[18];
MEContentsProf2DWithinRangeROOT* qth07_[18];
MEContentsProf2DWithinRangeROOT* qth08_[18];

MEContentsProf2DWithinRangeROOT* qth09_[18];
MEContentsProf2DWithinRangeROOT* qth10_[18];
MEContentsProf2DWithinRangeROOT* qth11_[18];
MEContentsProf2DWithinRangeROOT* qth12_[18];
MEContentsProf2DWithinRangeROOT* qth13_[18];
MEContentsProf2DWithinRangeROOT* qth14_[18];
MEContentsProf2DWithinRangeROOT* qth15_[18];
MEContentsProf2DWithinRangeROOT* qth16_[18];
MEContentsProf2DWithinRangeROOT* qth17_[18];
MEContentsProf2DWithinRangeROOT* qth18_[18];
MEContentsProf2DWithinRangeROOT* qth19_[18];
MEContentsProf2DWithinRangeROOT* qth20_[18];
MEContentsProf2DWithinRangeROOT* qth21_[18];
MEContentsProf2DWithinRangeROOT* qth22_[18];
MEContentsProf2DWithinRangeROOT* qth23_[18];
MEContentsProf2DWithinRangeROOT* qth24_[18];

TProfile2D* hs01_[18];
TProfile2D* hs02_[18];
TProfile2D* hs03_[18];
TProfile2D* hs04_[18];

TProfile2D* hs05_[18];
TProfile2D* hs06_[18];
TProfile2D* hs07_[18];
TProfile2D* hs08_[18];

MonitorElement* meg01_[18];
MonitorElement* meg02_[18];
MonitorElement* meg03_[18];
MonitorElement* meg04_[18];

MonitorElement* meg05_[18];
MonitorElement* meg06_[18];
MonitorElement* meg07_[18];
MonitorElement* meg08_[18];
MonitorElement* meg09_[18];
MonitorElement* meg10_[18];
MonitorElement* meg11_[18];
MonitorElement* meg12_[18];

MonitorElement* mea01_[18];
MonitorElement* mea02_[18];
MonitorElement* mea03_[18];
MonitorElement* mea04_[18];
MonitorElement* mea05_[18];
MonitorElement* mea06_[18];
MonitorElement* mea07_[18];
MonitorElement* mea08_[18];

MonitorElement* met01_[18];
MonitorElement* met02_[18];
MonitorElement* met03_[18];
MonitorElement* met04_[18];
MonitorElement* met05_[18];
MonitorElement* met06_[18];
MonitorElement* met07_[18];
MonitorElement* met08_[18];

MonitorElement* metav01_[18];
MonitorElement* metav02_[18];
MonitorElement* metav03_[18];
MonitorElement* metav04_[18];
MonitorElement* metav05_[18];
MonitorElement* metav06_[18];
MonitorElement* metav07_[18];
MonitorElement* metav08_[18];

MonitorElement* metrms01_[18];
MonitorElement* metrms02_[18];
MonitorElement* metrms03_[18];
MonitorElement* metrms04_[18];
MonitorElement* metrms05_[18];
MonitorElement* metrms06_[18];
MonitorElement* metrms07_[18];
MonitorElement* metrms08_[18];

MonitorElement* meaopn01_[18];
MonitorElement* meaopn02_[18];
MonitorElement* meaopn03_[18];
MonitorElement* meaopn04_[18];
MonitorElement* meaopn05_[18];
MonitorElement* meaopn06_[18];
MonitorElement* meaopn07_[18];
MonitorElement* meaopn08_[18];

MonitorElement* mepnprms01_[18];
MonitorElement* mepnprms02_[18];
MonitorElement* mepnprms03_[18];
MonitorElement* mepnprms04_[18];
MonitorElement* mepnprms05_[18];
MonitorElement* mepnprms06_[18];
MonitorElement* mepnprms07_[18];
MonitorElement* mepnprms08_[18];

CollateMonitorElement* me_i01_[18];
CollateMonitorElement* me_i02_[18];
CollateMonitorElement* me_i03_[18];
CollateMonitorElement* me_i04_[18];
CollateMonitorElement* me_i05_[18];
CollateMonitorElement* me_i06_[18];
CollateMonitorElement* me_i07_[18];
CollateMonitorElement* me_i08_[18];
CollateMonitorElement* me_i09_[18];
CollateMonitorElement* me_i10_[18];
CollateMonitorElement* me_i11_[18];
CollateMonitorElement* me_i12_[18];
CollateMonitorElement* me_i13_[18];
CollateMonitorElement* me_i14_[18];
CollateMonitorElement* me_i15_[18];
CollateMonitorElement* me_i16_[18];

TProfile2D* i01_[18];
TProfile2D* i02_[18];
TProfile2D* i03_[18];
TProfile2D* i04_[18];
TProfile2D* i05_[18];
TProfile2D* i06_[18];
TProfile2D* i07_[18];
TProfile2D* i08_[18];
TProfile2D* i09_[18];
TProfile2D* i10_[18];
TProfile2D* i11_[18];
TProfile2D* i12_[18];
TProfile2D* i13_[18];
TProfile2D* i14_[18];
TProfile2D* i15_[18];
TProfile2D* i16_[18];

// Quality check on crystals

float percentVariation_;

// Quality check on PNs

float amplitudeThresholdPnG01_;
float amplitudeThresholdPnG16_;
float pedPnExpectedMean_[2];
float pedPnDiscrepancyMean_[2];
float pedPnRMSThreshold_[2];

MEContentsTH2FWithinRangeROOT* qtg01_[36];
MEContentsTH2FWithinRangeROOT* qtg02_[36];
MEContentsTH2FWithinRangeROOT* qtg03_[36];
MEContentsTH2FWithinRangeROOT* qtg04_[36];

MEContentsTH2FWithinRangeROOT* qtg05_[36];
MEContentsTH2FWithinRangeROOT* qtg06_[36];
MEContentsTH2FWithinRangeROOT* qtg07_[36];
MEContentsTH2FWithinRangeROOT* qtg08_[36];
MEContentsTH2FWithinRangeROOT* qtg09_[36];
MEContentsTH2FWithinRangeROOT* qtg10_[36];
MEContentsTH2FWithinRangeROOT* qtg11_[36];
MEContentsTH2FWithinRangeROOT* qtg12_[36];

};

#endif
