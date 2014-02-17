#ifndef EETimingClient_H
#define EETimingClient_H

/*
 * \file EETimingClient.h
 *
 * $Date: 2012/04/27 13:46:04 $
 * $Revision: 1.35 $
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

#include "FWCore/Framework/interface/ESHandle.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"

#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"

class MonitorElement;
class DQMStore;
#ifdef WITH_ECAL_COND_DB
class EcalCondDBInterface;
class RunIOV;
class MonRunIOV;
#endif

class EETimingClient : public EEClient {

friend class EESummaryClient;

public:

/// Constructor
EETimingClient(const edm::ParameterSet& ps);

/// Destructor
virtual ~EETimingClient();

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

DQMStore* dqmStore_;

MonitorElement* meh01_[18];
MonitorElement* meh02_[18];

TProfile2D* h01_[18];
TH2F* h02_[18];

MonitorElement* meg01_[18];

MonitorElement* mea01_[18];

MonitorElement* mep01_[18];

MonitorElement* mer01_[18];

 MonitorElement* meTimeSummaryMapProjEta_[2];
 MonitorElement* meTimeSummaryMapProjPhi_[2];

// Quality check on crystals, one per each gain

 float expectedMean_;
 float meanThreshold_;
 float rmsThreshold_;

 int nHitThreshold_;
};

#endif
