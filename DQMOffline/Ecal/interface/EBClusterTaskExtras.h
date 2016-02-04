#ifndef EBClusterTaskExtras_H
#define EBClusterTaskExtras_H

/*
 * \file EBClusterTaskExtras.h
 *
 * $Date: 2009/12/14 21:14:06 $
 * $Revision: 1.5 $
 * \author G. Della Ricca
 *
 */

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#define EBCLUSTERTASKEXTRAS_DQMOFFLINE

class MonitorElement;
class DQMStore;

class EBClusterTaskExtras: public edm::EDAnalyzer{

 public:

/// Constructor
EBClusterTaskExtras(const edm::ParameterSet& ps);

/// Destructor
virtual ~EBClusterTaskExtras();

protected:

/// Analyze
void analyze(const edm::Event& e, const edm::EventSetup& c);

/// BeginJob
void beginJob(void);

/// EndJob
void endJob(void);

/// BeginRun
void beginRun(const edm::Run & r, const edm::EventSetup & c);

/// EndRun
void endRun(const edm::Run & r, const edm::EventSetup & c);

/// Reset
void reset(void);

/// Setup
void setup(void);

/// Cleanup
void cleanup(void);

// Determine Trigger
std::vector<bool> determineTriggers(const edm::Event&, const edm::EventSetup& eventSetup);
bool isExclusiveTrigger(int l1Trigger, std::vector<bool>& l1Triggers);
bool doMonitorElement(std::string meName);

enum L1Triggers {
   CSC_TRIGGER	= 0,
   DT_TRIGGER	= 1,
   ECAL_TRIGGER	= 2,
   HCAL_TRIGGER	= 3,
   RPC_TRIGGER	= 4
};

private:

int ievt_;

DQMStore* dqmStore_;

std::string prefixME_;

bool enableCleanup_;

bool mergeRuns_;

edm::InputTag SuperClusterCollection_;
edm::InputTag EcalRecHitCollection_;
edm::InputTag l1GMTReadoutRecTag_;
edm::InputTag l1GTReadoutRecTag_;
std::vector<std::string> meList_;

#ifndef EBCLUSTERTASKEXTRAS_DQMOFFLINE
MonitorElement* meSCSizCrystal_;
MonitorElement* meSCSizBC_;
MonitorElement* meSCSizPhi_;

MonitorElement* meSCSeedEne_;
MonitorElement* meSCEne2_;
MonitorElement* meSCEneLow_;
MonitorElement* meSCEneHigh_;
MonitorElement* meSCEneSingleCrystal_;

MonitorElement* meSCSeedMapOccTT_;
MonitorElement* meSCSeedMapOccHighEne_;
MonitorElement* meSCSeedMapOccSingleCrystal_;

MonitorElement* meSCSeedTime_;
MonitorElement* meSCSeedMapTimeTT_;
MonitorElement* meSCSeedMapTimeMod_;
MonitorElement* meSCSeedTimeVsPhi_;
MonitorElement* meSCSeedTimeVsAmp_;
MonitorElement* meSCSeedTimeEBM_;
MonitorElement* meSCSeedTimeEBP_;
MonitorElement* meSCSeedTimeEBMTop_;
MonitorElement* meSCSeedTimeEBPTop_;
MonitorElement* meSCSeedTimeEBMBot_;
MonitorElement* meSCSeedTimeEBPBot_;
MonitorElement* meSCSeedTimePerFed_[36];

MonitorElement* meSCSeedMapOccTrg_[5];
MonitorElement* meSCSeedMapOccTrgExcl_[5];
MonitorElement* meSCSeedMapTimeTrgMod_[5];
#endif

MonitorElement* meSCSizCrystalVsEne_;

MonitorElement* meSCSeedMapOcc_;
MonitorElement* meSCSeedMapOccHighEneTT_;

MonitorElement* meSCSeedMapOccTrgTT_[5];
MonitorElement* meSCSeedMapOccTrgExclTT_[5];

MonitorElement* meSCSeedMapTimeTrgTT_[5];
MonitorElement* meSCSeedTimeTrg_[5];

MonitorElement* meTrg_;
MonitorElement* meTrgExcl_;

bool init_;

};

#endif
