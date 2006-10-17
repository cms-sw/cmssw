#ifndef DTLocalRecoTask_H
#define DTLocalRecoTask_H

/*
 * \file DTLocalRecoTask.h
 *
 * $Date: 2006/06/28 09:21:25 $
 * $Revision: 1.5 $
 * \author M. Zanetti & G. Cerminara - INFN Padova & Torino
 *
*/

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include <FWCore/Framework/interface/EDAnalyzer.h>


#include <string>

// #include <fstream>
// #include <vector>


class DaqMonitorBEInterface;
class DTSegmentAnalysis;
class DTResolutionAnalysis;


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
  std::string theRootFileName;
  bool writeHisto;

  // The BE interface
  DaqMonitorBEInterface* dbe;

  // Classes doing the analysis
  DTSegmentAnalysis *theSegmentAnalysis;
  DTResolutionAnalysis *theResolutionAnalysis;
  
  bool doSegmentAnalysis;
  bool doResolutionAnalysis;

};

#endif
