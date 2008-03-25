#ifndef EBSummaryClient_H
#define EBSummaryClient_H

/*
 * \file EBSummaryClient.h
 *
 * $Date: 2007/07/18 09:37:55 $
 * $Revision: 1.11 $
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

#include "OnlineDB/EcalCondDB/interface/EcalCondDBInterface.h"
#include "OnlineDB/EcalCondDB/interface/MonRunIOV.h"

#include "DQMServices/Core/interface/MonitorElement.h"
#include "DQMServices/Core/interface/MonitorUserInterface.h"
#include "DQMServices/Core/interface/CollateMonitorElement.h"

#include "DQM/EcalBarrelMonitorClient/interface/EBClient.h"

class EBSummaryClient : public EBClient {

public:

/// Constructor
EBSummaryClient(const edm::ParameterSet& ps);

/// Destructor
virtual ~EBSummaryClient();

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

/// Set Clients
inline void setFriends(vector<EBClient*> clients) { clients_ = clients; }

private:

void writeMap( std::ofstream& hf, std::string mapname );

int ievt_;
int jevt_;

bool collateSources_;
bool cloneME_;
bool enableQT_;

bool verbose_;

bool enableMonitorDaemon_;

string prefixME_;

vector<int> superModules_;

vector<EBClient*> clients_;

MonitorUserInterface* mui_;

MonitorElement* meIntegrity_;
MonitorElement* meOccupancy_;
MonitorElement* mePedestalOnline_;
MonitorElement* meLaserL1_;
MonitorElement* meLaserL1PN_;
MonitorElement* mePedestal_;
MonitorElement* mePedestalPN_;
MonitorElement* meTestPulse_;
MonitorElement* meTestPulsePN_;
MonitorElement* meGlobalSummary_;

MEContentsTH2FWithinRangeROOT* qtg01_;
MEContentsTH2FWithinRangeROOT* qtg02_;
MEContentsTH2FWithinRangeROOT* qtg03_;
MEContentsTH2FWithinRangeROOT* qtg04_;
MEContentsTH2FWithinRangeROOT* qtg04PN_;
MEContentsTH2FWithinRangeROOT* qtg05_;
MEContentsTH2FWithinRangeROOT* qtg05PN_;
MEContentsTH2FWithinRangeROOT* qtg06_;
MEContentsTH2FWithinRangeROOT* qtg06PN_;
MEContentsTH2FWithinRangeROOT* qtg07_;

};

#endif
