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

#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "DQMServices/Core/interface/DQMEDAnalyzer.h"

#include <string>
#include <vector>

class DTSegmentsTask: public DQMEDAnalyzer{

public:
  /// Constructor
  DTSegmentsTask(const edm::ParameterSet& pset);

  /// Destructor
  virtual ~DTSegmentsTask();

  /// book the histos
  void analyze(const edm::Event&, const edm::EventSetup&);
  void bookHistograms(DQMStore::IBooker &, edm::Run const &, edm::EventSetup const &) override;

protected:

private:


  // Switch for verbosity
  bool debug;
  bool checkNoisyChannels;
  edm::ParameterSet parameters;
  
  // the histos
  std::vector<MonitorElement*> phiHistos;
  std::vector<MonitorElement*> thetaHistos;
  
  // Label of 4D segments in the event
  edm::EDGetTokenT<DTRecSegment4DCollection> theRecHits4DLabel_;
};
#endif

