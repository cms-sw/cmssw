#ifndef EBClusterTask_H
#define EBClusterTask_H

/*
 * \file EBClusterTask.h
 *
 * $Date: 2007/02/01 15:43:56 $
 * $Revision: 1.4 $
 * \author G. Della Ricca
 *
*/


#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DQMServices/Core/interface/MonitorElement.h"

class EBClusterTask: public edm::EDAnalyzer{

public:

/// Constructor
EBClusterTask(const edm::ParameterSet& ps);

/// Destructor
virtual ~EBClusterTask();

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

MonitorElement* meBEne_;
MonitorElement* meBNum_;
MonitorElement* meBCry_;

MonitorElement* meBEneMap_;
MonitorElement* meBNumMap_;

MonitorElement* meSEne_;
MonitorElement* meSNum_;
MonitorElement* meSSiz_;

MonitorElement* meSEneMap_;
MonitorElement* meSNumMap_;

bool init_;

};

#endif
