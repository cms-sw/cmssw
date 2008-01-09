#ifndef EBStatusFlagsTask_H
#define EBStatusFlagsTask_H

/*
 * \file EBStatusFlagsTask.h
 *
 * $Date: 2007/11/13 13:20:50 $
 * $Revision: 1.25 $
 * \author G. Della Ricca
 *
*/

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DQMServices/Core/interface/MonitorElement.h"

class MonitorElement;
class DaqMonitorBEInterface;

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

DaqMonitorBEInterface* dbe_;

bool enableCleanup_;

edm::InputTag EcalRawDataCollection_;

MonitorElement* meEvtType_[36];

MonitorElement* meFEchErrors_[36][2];

bool init_;

};

#endif
