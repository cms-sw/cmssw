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
#include "FWCore/Utilities/interface/InputTag.h"

#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include <string>
#include <map>
#include <vector>
//#include <pair>

class DQMStore;
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
void beginJob();

// EndJob
void endJob();

protected:

private:
  DQMStore* theDbe;

  bool debug;

  MonitorElement* histo;

  bool isLocalRun;
  edm::InputTag ltcDigiCollectionTag;
};
#endif
