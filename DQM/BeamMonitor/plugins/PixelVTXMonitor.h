/*
 * \class PixelVTXMonitor
 *
 * DQM FED Client
 *
 * $Date: 2011/07/18 15:52:06 $
 * $Revision: 1.0 $
 * \author  S. Dutta
 *
*/

#ifndef PIXELVTXMONITOR_H
#define PIXELVTXMONITORH

#include <string>
#include <vector>
#include <map>

#include <FWCore/Framework/interface/EDAnalyzer.h>
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "HLTrigger/HLTcore/interface/HLTConfigProvider.h"

//
// class declaration
//

class PixelVTXMonitor : public edm::EDAnalyzer {
public:
  PixelVTXMonitor( const edm::ParameterSet& );
  ~PixelVTXMonitor();

protected:

  void beginJob();
  void beginRun(edm::Run const& iRun, edm::EventSetup const& iSetup);
  void analyze(edm::Event const& iEvent, edm::EventSetup const& iSetup);
  void endRun(edm::Run const& iRun,  edm::EventSetup const& iSetup);
  void endJob();

private:

  void bookHistograms();

  edm::ParameterSet parameters_;

  std::string moduleName_;
  std::string folderName_;
  edm::InputTag pixelVertexInputTag_;
  edm::InputTag hltInputTag_;
  float minVtxDoF_;

  DQMStore * dbe_;
  HLTConfigProvider hltConfig_;

  std::map<std::string, MonitorElement*>  vtxHistoMap_;
};

#endif
