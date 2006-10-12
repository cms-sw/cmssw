#ifndef DTSegmentAnalysis_H
#define DTSegmentAnalysis_H

/** \class DTTriggerCheck
 *
 *  \author S.Bolognesi - INFN Torino
 */

#include <FWCore/Framework/interface/MakerMacros.h>
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include <FWCore/Framework/interface/EDAnalyzer.h>
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DQMServices/Core/interface/DaqMonitorBEInterface.h"
#include "DQMServices/Daemon/interface/MonitorDaemon.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include <string>
#include <map>
#include <vector>
//#include <pair>

class DaqMonitorBEInterface;
class MonitorElement;

class DTTriggerCheck: public edm::EDAnalyzer{

friend class DTMonitorModule;
public:
  /// Constructor
  DTTriggerCheck(const edm::ParameterSet& pset);

  /// Destructor
  virtual ~DTTriggerCheck();

/// Analyze
void analyze(const edm::Event& event, const edm::EventSetup& setup);

// BeginJob
void beginJob(const edm::EventSetup& setup);

// EndJob
void endJob();

protected:

private:
  DaqMonitorBEInterface* theDbe;

  bool debug;

  edm::ParameterSet parameters;

  MonitorElement* histo;

};
#endif

