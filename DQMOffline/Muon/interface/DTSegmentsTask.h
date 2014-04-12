#ifndef DTSegmentsTask_H
#define DTSegmentsTask_H

/** \class DTSegmentsTask
 *  DQM Analysis of 4D DT segments
 *
 *  \author G. Mila - INFN Torino
 */

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include <FWCore/Framework/interface/EDAnalyzer.h>

//RecHit
#include "DataFormats/DTRecHit/interface/DTRecSegment4DCollection.h"
#include "CondFormats/DataRecord/interface/DTStatusFlagRcd.h"
#include "CondFormats/DTObjects/interface/DTStatusFlag.h"

#include <string>
#include <vector>

class DQMStore;
class MonitorElement;

class DTSegmentsTask: public edm::EDAnalyzer{
public:
  /// Constructor
  DTSegmentsTask(const edm::ParameterSet& pset);

  /// Destructor
  virtual ~DTSegmentsTask();

  /// book the histos
  void beginJob(void);
  void beginRun(const edm::Run&, const edm::EventSetup&);

  /// Endjob
  void endJob();
  void endRun(const edm::Run&, const edm::EventSetup&);

    // Operations
  void analyze(const edm::Event& event, const edm::EventSetup& setup);

protected:

private:

  // The BE interface
  DQMStore* theDbe;

  // Switch for verbosity
  bool debug;
  edm::ParameterSet parameters;
  
  // the histos
  std::vector<MonitorElement*> phiHistos;
  std::vector<MonitorElement*> thetaHistos;
  
  // Label of 4D segments in the event
  edm::EDGetTokenT<DTRecSegment4DCollection> theRecHits4DLabel_;
};
#endif

