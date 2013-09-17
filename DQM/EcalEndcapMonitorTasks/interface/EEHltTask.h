#ifndef EEHltTask_H
#define EEHltTask_H

/*
 * \file EEHltTask.h
 *
 * \author G. Della Ricca
 *
*/

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"
#include "DataFormats/EcalDetId/interface/EcalElectronicsId.h"
#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "DataFormats/EcalDetId/interface/EcalDetIdCollections.h"

#include "Geometry/EcalMapping/interface/EcalElectronicsMapping.h"

class MonitorElement;
class DQMStore;

class EEHltTask: public edm::EDAnalyzer{

public:

/// Constructor
EEHltTask(const edm::ParameterSet& ps);

/// Destructor
virtual ~EEHltTask();

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

void initGeometry( const edm::EventSetup& setup );

EcalSubdetector subDet( const EEDetId& id ) { return( id.subdet() ); }

EcalSubdetector subDet( const EcalElectronicsId& id ) { return( id.subdet() ); }

int iSM( const EEDetId& id );

int iSM( const EcalElectronicsId& id );

private:

int ievt_;

DQMStore* dqmStore_;

std::string prefixME_;
std::string folderName_;

bool enableCleanup_;

bool mergeRuns_;

edm::EDGetTokenT<EEDetIdCollection> EEDetIdCollection0_;
edm::EDGetTokenT<EEDetIdCollection> EEDetIdCollection1_;
edm::EDGetTokenT<EEDetIdCollection> EEDetIdCollection2_;
edm::EDGetTokenT<EEDetIdCollection> EEDetIdCollection3_;
edm::EDGetTokenT<EEDetIdCollection> EEDetIdCollection4_;
edm::EDGetTokenT<EcalElectronicsIdCollection> EcalElectronicsIdCollection1_;
edm::EDGetTokenT<EcalElectronicsIdCollection> EcalElectronicsIdCollection2_;
edm::EDGetTokenT<EcalElectronicsIdCollection> EcalElectronicsIdCollection3_;
edm::EDGetTokenT<EcalElectronicsIdCollection> EcalElectronicsIdCollection4_;
edm::EDGetTokenT<EcalElectronicsIdCollection> EcalElectronicsIdCollection5_;
edm::EDGetTokenT<EcalElectronicsIdCollection> EcalElectronicsIdCollection6_;
edm::EDGetTokenT<FEDRawDataCollection> FEDRawDataCollection_;

MonitorElement* meEEFedsOccupancy_;
MonitorElement* meEEFedsSizeErrors_;
MonitorElement* meEEFedsIntegrityErrors_;

bool init_;
bool initGeometry_;

const EcalElectronicsMapping* map;

};

#endif
