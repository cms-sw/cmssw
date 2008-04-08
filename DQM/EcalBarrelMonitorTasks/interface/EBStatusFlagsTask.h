#ifndef EBStatusFlagsTask_H
#define EBStatusFlagsTask_H

/*
 * \file EBStatusFlagsTask.h
 *
 * $Date: 2008/04/08 15:06:23 $
 * $Revision: 1.4 $
 * \author G. Della Ricca
 *
*/

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

class MonitorElement;
class DQMStore;

class EBStatusFlagsTask: public edm::EDAnalyzer{

public:

/// Constructor
EBStatusFlagsTask(const edm::ParameterSet& ps);

/// Destructor
virtual ~EBStatusFlagsTask();

protected:

/// Analyze
void analyze(const edm::Event& e, const edm::EventSetup& c);

/// BeginJob
void beginJob(const edm::EventSetup& c);

/// EndJob
void endJob(void);

/// Setup
void setup(void);

/// Cleanup
void cleanup(void);

private:

int ievt_;

DQMStore* dqmStore_;

std::string prefixME_;

bool enableCleanup_;

edm::InputTag EcalRawDataCollection_;

MonitorElement* meEvtType_[36];

MonitorElement* meFEchErrors_[36][2];

bool init_;

};

#endif
