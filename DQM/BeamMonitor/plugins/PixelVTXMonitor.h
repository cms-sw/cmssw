/*
 * \class PixelVTXMonitor
 *
 * DQM FED Client
 *
 * \author  S. Dutta
 *
*/

#ifndef PIXELVTXMONITOR_H
#define PIXELVTXMONITOR_H

#include <string>
#include <vector>
#include <map>

#include <FWCore/Framework/interface/EDAnalyzer.h>
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "HLTrigger/HLTcore/interface/HLTConfigProvider.h"
#include "DataFormats/SiPixelCluster/interface/SiPixelCluster.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/Common/interface/TriggerResults.h"

//
// class declaration
//

class PixelVTXMonitor : public edm::EDAnalyzer {
public:
  PixelVTXMonitor(const edm::ParameterSet&);
  ~PixelVTXMonitor() override;

protected:
  void beginJob() override;
  void beginRun(edm::Run const& iRun, edm::EventSetup const& iSetup) override;
  void analyze(edm::Event const& iEvent, edm::EventSetup const& iSetup) override;
  void endRun(edm::Run const& iRun, edm::EventSetup const& iSetup) override;
  void endJob() override;

private:
  void bookHistograms();

  edm::ParameterSet parameters_;

  std::string moduleName_;
  std::string folderName_;
  edm::EDGetTokenT<SiPixelClusterCollectionNew> pixelClusterInputTagToken_;
  edm::EDGetTokenT<reco::VertexCollection> pixelVertexInputTagToken_;
  edm::EDGetTokenT<edm::TriggerResults> hltInputTagToken_;
  edm::InputTag pixelClusterInputTag_;
  edm::InputTag pixelVertexInputTag_;
  edm::InputTag hltInputTag_;
  float minVtxDoF_;

  DQMStore* dbe_;
  HLTConfigProvider hltConfig_;

  struct PixelMEs {
    MonitorElement* clusME;
    MonitorElement* vtxME;
  };

  std::map<std::string, PixelMEs> histoMap_;
};

#endif  // PIXELVTXMONITOR_H

// Local Variables:
// show-trailing-whitespace: t
// truncate-lines: t
// End:
