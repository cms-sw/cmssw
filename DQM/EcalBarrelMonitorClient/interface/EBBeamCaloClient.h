#ifndef EBBeamCaloClient_H
#define EBBeamCaloClient_H

/*
 * \file EBBeamCaloClient.h
 *
 * $Date: 2008/04/07 07:24:31 $
 * $Revision: 1.35 $
 * \author G. Della Ricca
 * \author A. Ghezzi
 *
*/

#include <vector>
#include <string>

#include "TROOT.h"
#include "TProfile2D.h"
#include "TProfile.h"
#include "TH1F.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DQM/EcalBarrelMonitorClient/interface/EBClient.h"

class MonitorElement;
class DQMStore;
class EcalCondDBInterface;
class RunIOV;
class MonRunIOV;

class EBBeamCaloClient : public EBClient {

public:

/// Constructor
EBBeamCaloClient(const edm::ParameterSet& ps);

/// Destructor
virtual ~EBBeamCaloClient();

/// Analyze
void analyze(void);

/// BeginJob
void beginJob(DQMStore* dbe);

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

template<class T> void AdjustRange(T obj);

/// Get Functions
inline int getEvtPerJob() { return ievt_; }
inline int getEvtPerRun() { return jevt_; }

private:

const static int cryInArray_ = 9;

int ievt_;
int jevt_;

bool cloneME_;

bool verbose_;
bool debug_;

bool enableCleanup_;

std::vector<int> superModules_;

DQMStore* dbe_;

//specific task me
std::vector<int> checkedSteps_;
float prescaling_;
//specific task me

TH1F* hBGains_[cryInArray_];
TProfile* hBpulse_[cryInArray_];

TH1F* hBEne1_;

//TH1F* hBGainsMoving_[cryInArray_];
//TH1F* hBEne1Moving_;

TH1F* hBAllNeededCry_;

TH1F* hBNumReadCry_;

TH1F* hBE3x3_;

TH1F* hBE3x3Moving_;

TH2F* hBCryOnBeam_;

TH2F* hBMaxEneCry_;

TH1F* hBReadCryErrors_;

TProfile* hBE1vsCry_;

TProfile* hBE3x3vsCry_;

TH1F* hBEntriesvsCry_;

TH1F* hBcryDone_;

TH2F* hBBeamCentered_;

TH1F* hbTBmoving_;

TH1F* hbE1MaxCry_;

TH1F* hbDesync_;

TProfile* pBCriInBeamEvents_;

MonitorElement* meEBBCaloRedGreen_;
MonitorElement* meEBBCaloRedGreenReadCry_;
MonitorElement* meEBBCaloRedGreenSteps_;

// quality check parameters

int minEvtNum_;

float aveEne1_;
float E1Th_;
float aveEne3x3_;
float E3x3Th_;
float RMSEne3x3_;
float ReadCryErrThr_;

};

#endif
