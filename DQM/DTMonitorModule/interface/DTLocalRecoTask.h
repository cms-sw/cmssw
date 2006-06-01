#ifndef DTLocalRecoTask_H
#define DTLocalRecoTask_H

/*
 * \file DTLocalRecoTask.h
 *
 * $Date: 2006/02/15 08:24:55 $
 * $Revision: 1.1 $
 * \author M. Zanetti & G. Cerminara - INFN Padova & Torino
 *
*/

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include <FWCore/Framework/interface/EDAnalyzer.h>




// #include <fstream>
// #include <vector>


class DaqMonitorBEInterface;
class DTSegmentAnalysis;

class DTLocalRecoTask: public edm::EDAnalyzer{

friend class DTMonitorModule;

public:

/// Constructor
DTLocalRecoTask(const edm::ParameterSet& pset);

/// Destructor
virtual ~DTLocalRecoTask();

protected:

/// Analyze
void analyze(const edm::Event& event, const edm::EventSetup& setup);

// BeginJob
void beginJob(const edm::EventSetup& setup);

// EndJob
void endJob();

private:
  
  // Switch for verbosity
  bool debug;

  // The BE interface
  DaqMonitorBEInterface* dbe;

  // Classes doing the analysis
  DTSegmentAnalysis *theSegmentAnalysis;

  // My monitor elements
  
  //  ofstream logFile;
  
};

#endif
