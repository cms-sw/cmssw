#ifndef EBHltTask_H
#define EBHltTask_H

/*
 * \file EBHltTask.h
 *
 * \author G. Della Ricca
 *
*/

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDetId/interface/EcalElectronicsId.h"
#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"

#include "Geometry/EcalMapping/interface/EcalElectronicsMapping.h"

class MonitorElement;
class DQMStore;

class EBHltTask: public edm::EDAnalyzer{

public:

/// Constructor
EBHltTask(const edm::ParameterSet& ps);

/// Destructor
virtual ~EBHltTask();

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

EcalSubdetector subDet( const EBDetId& id ) { return( id.subdet() ); }

EcalSubdetector subDet( const EcalElectronicsId& id ) { return( id.subdet() ); }

int iSM( const EBDetId& id );

int iSM( const EcalElectronicsId& id );

private:

int ievt_;

DQMStore* dqmStore_;

std::string prefixME_;
std::string folderName_;

bool enableCleanup_;

bool mergeRuns_;

edm::EDGetTokenT<EBDetIdCollection> EBDetIdCollection0_;
edm::EDGetTokenT<EBDetIdCollection> EBDetIdCollection1_;
edm::EDGetTokenT<EBDetIdCollection> EBDetIdCollection2_;
edm::EDGetTokenT<EBDetIdCollection> EBDetIdCollection3_;
edm::EDGetTokenT<EBDetIdCollection> EBDetIdCollection4_;
edm::EDGetTokenT<EcalElectronicsIdCollection> EcalElectronicsIdCollection1_;
edm::EDGetTokenT<EcalElectronicsIdCollection> EcalElectronicsIdCollection2_;
edm::EDGetTokenT<EcalElectronicsIdCollection> EcalElectronicsIdCollection3_;
edm::EDGetTokenT<EcalElectronicsIdCollection> EcalElectronicsIdCollection4_;
edm::EDGetTokenT<EcalElectronicsIdCollection> EcalElectronicsIdCollection5_;
edm::EDGetTokenT<EcalElectronicsIdCollection> EcalElectronicsIdCollection6_;
edm::EDGetTokenT<FEDRawDataCollection> FEDRawDataCollection_;

MonitorElement* meEBFedsOccupancy_;
MonitorElement* meEBFedsSizeErrors_;
MonitorElement* meEBFedsIntegrityErrors_;

bool init_;
bool initGeometry_;

const EcalElectronicsMapping* map;

};

#endif
