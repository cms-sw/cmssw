#ifndef EBHltTask_H
#define EBHltTask_H

/*
 * \file EBHltTask.h
 *
 * $Date: 2012/04/27 13:46:00 $
 * $Revision: 1.10 $
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

edm::InputTag EBDetIdCollection0_;
edm::InputTag EBDetIdCollection1_;
edm::InputTag EBDetIdCollection2_;
edm::InputTag EBDetIdCollection3_;
edm::InputTag EBDetIdCollection4_;
edm::InputTag EcalElectronicsIdCollection1_;
edm::InputTag EcalElectronicsIdCollection2_;
edm::InputTag EcalElectronicsIdCollection3_;
edm::InputTag EcalElectronicsIdCollection4_;
edm::InputTag EcalElectronicsIdCollection5_;
edm::InputTag EcalElectronicsIdCollection6_;
edm::InputTag FEDRawDataCollection_;

MonitorElement* meEBFedsOccupancy_;
MonitorElement* meEBFedsSizeErrors_;
MonitorElement* meEBFedsIntegrityErrors_;

bool init_;
bool initGeometry_;

const EcalElectronicsMapping* map;

};

#endif
