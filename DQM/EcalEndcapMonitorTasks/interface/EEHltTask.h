#ifndef EEHltTask_H
#define EEHltTask_H

/*
 * \file EEHltTask.h
 *
 * $Date: 2010/03/26 11:24:50 $
 * $Revision: 1.6 $
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

edm::InputTag EEDetIdCollection0_;
edm::InputTag EEDetIdCollection1_;
edm::InputTag EEDetIdCollection2_;
edm::InputTag EEDetIdCollection3_;
edm::InputTag EEDetIdCollection4_;
edm::InputTag EcalElectronicsIdCollection1_;
edm::InputTag EcalElectronicsIdCollection2_;
edm::InputTag EcalElectronicsIdCollection3_;
edm::InputTag EcalElectronicsIdCollection4_;
edm::InputTag EcalElectronicsIdCollection5_;
edm::InputTag EcalElectronicsIdCollection6_;
edm::InputTag FEDRawDataCollection_;

MonitorElement* meEEFedsOccupancy_;
MonitorElement* meEEFedsSizeErrors_;
MonitorElement* meEEFedsIntegrityErrors_;

bool init_;
bool initGeometry_;

const EcalElectronicsMapping* map;

};

#endif
