#ifndef EBCosmicTask_H
#define EBCosmicTask_H

/*
 * \file EBCosmicTask.h
 *
 * \author G. Della Ricca
 *
*/

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/EcalRawData/interface/EcalRawDataCollections.h"
#include "DataFormats/EcalRecHit/interface/EcalUncalibratedRecHit.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"

class MonitorElement;
class DQMStore;

class EBCosmicTask: public edm::EDAnalyzer{

public:

/// Constructor
EBCosmicTask(const edm::ParameterSet& ps);

/// Destructor
virtual ~EBCosmicTask();

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

private:

int ievt_;

DQMStore* dqmStore_;

std::string prefixME_;

bool enableCleanup_;

bool mergeRuns_;

edm::EDGetTokenT<EcalRawDataCollection> EcalRawDataCollection_;
edm::EDGetTokenT<EcalUncalibratedRecHitCollection> EcalUncalibratedRecHitCollection_;
edm::EDGetTokenT<EcalRecHitCollection> EcalRecHitCollection_;

MonitorElement* meCutMap_[36];

MonitorElement* meSelMap_[36];

MonitorElement* meSpectrum_[2][36];

double threshold_;
double minJitter_;
double maxJitter_;

bool init_;

};

#endif
