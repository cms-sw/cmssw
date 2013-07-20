#ifndef DTSegmentsTask_H
#define DTSegmentsTask_H

/** \class DTSegmentsTask
 *  DQM Analysis of 4D DT segments
 *
 *  $Date: 2009/12/22 17:44:52 $
 *  $Revision: 1.2 $
 *  \author G. Mila - INFN Torino
 */

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include <FWCore/Framework/interface/EDAnalyzer.h>

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

  /// Endjob
  void endJob();

  // Operations
  void analyze(const edm::Event& event, const edm::EventSetup& setup);

protected:

private:

  // The BE interface
  DQMStore* theDbe;

  // Switch for verbosity
  bool debug;

  // Lable of 4D segments in the event
  std::string theRecHits4DLabel;

  edm::ParameterSet parameters;
  
  // the histos
  std::vector<MonitorElement*> phiHistos;
  std::vector<MonitorElement*> thetaHistos;

};
#endif

