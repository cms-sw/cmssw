/*
 * \file PixelVTXMonitor.cc
 * \author S. Dutta
 * Last Update:
 *
 * Description: Pixel Vertex Monitoring for different HLT paths
 *
*/

// system includes
#include <string>
#include <vector>
#include <map>

// user includes
#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/TriggerResults.h"
#include "DataFormats/SiPixelCluster/interface/SiPixelCluster.h"
#include "DataFormats/SiPixelCluster/interface/SiPixelCluster.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "HLTrigger/HLTcore/interface/HLTConfigProvider.h"

// ROOT includes
#include "TPRegexp.h"

//
// class declaration
//

class PixelVTXMonitor : public DQMEDAnalyzer {
public:
  typedef dqm::legacy::MonitorElement MonitorElement;
  typedef dqm::legacy::DQMStore DQMStore;
  PixelVTXMonitor(const edm::ParameterSet&);
  ~PixelVTXMonitor() override = default;

protected:
  void bookHistograms(DQMStore::IBooker& iBooker, const edm::Run& iRun, const edm::EventSetup& iSetup) override;

private:
  void dqmBeginRun(const edm::Run& iRun, const edm::EventSetup& iSetup) override;
  void analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) override;

  edm::ParameterSet parameters_;

  const std::string moduleName_;
  const std::string folderName_;
  const edm::InputTag pixelClusterInputTag_;
  const edm::InputTag pixelVertexInputTag_;
  const edm::InputTag hltInputTag_;
  const edm::EDGetTokenT<SiPixelClusterCollectionNew> pixelClusterInputTagToken_;
  const edm::EDGetTokenT<reco::VertexCollection> pixelVertexInputTagToken_;
  const edm::EDGetTokenT<edm::TriggerResults> hltInputTagToken_;
  const float minVtxDoF_;

  HLTConfigProvider hltConfig_;

  struct PixelMEs {
    MonitorElement* clusME;
    MonitorElement* vtxME;
  };

  std::map<std::string, PixelMEs> histoMap_;
};

// -----------------------------
//  constructors and destructor
// -----------------------------

PixelVTXMonitor::PixelVTXMonitor(const edm::ParameterSet& ps)
    : parameters_(ps),
      moduleName_(parameters_.getParameter<std::string>("ModuleName")),
      folderName_(parameters_.getParameter<std::string>("FolderName")),
      pixelClusterInputTag_(parameters_.getParameter<edm::InputTag>("PixelClusterInputTag")),
      pixelVertexInputTag_(parameters_.getParameter<edm::InputTag>("PixelVertexInputTag")),
      hltInputTag_(parameters_.getParameter<edm::InputTag>("HLTInputTag")),
      pixelClusterInputTagToken_(consumes<SiPixelClusterCollectionNew>(pixelClusterInputTag_)),
      pixelVertexInputTagToken_(consumes<reco::VertexCollection>(pixelVertexInputTag_)),
      hltInputTagToken_(consumes<edm::TriggerResults>(hltInputTag_)),
      minVtxDoF_(parameters_.getParameter<double>("MinVtxDoF")) {}

void PixelVTXMonitor::bookHistograms(DQMStore::IBooker& iBooker, const edm::Run&, const edm::EventSetup&) {
  std::vector<std::string> hltPathsOfInterest =
      parameters_.getParameter<std::vector<std::string> >("HLTPathsOfInterest");
  if (hltPathsOfInterest.empty())
    return;

  const std::vector<std::string>& pathList = hltConfig_.triggerNames();
  std::vector<std::string> selectedPaths;
  for (const auto& it : pathList) {
    int nmatch = 0;
    for (const auto& kt : hltPathsOfInterest) {
      nmatch += TPRegexp(kt).Match(it);
    }
    if (!nmatch)
      continue;
    else
      selectedPaths.push_back(it);
  }

  edm::ParameterSet ClusHistoPar = parameters_.getParameter<edm::ParameterSet>("TH1ClusPar");
  edm::ParameterSet VtxHistoPar = parameters_.getParameter<edm::ParameterSet>("TH1VtxPar");

  std::string currentFolder = moduleName_ + "/" + folderName_;
  iBooker.setCurrentFolder(currentFolder);

  PixelMEs local_MEs;
  for (const auto& tag : selectedPaths) {
    std::map<std::string, PixelMEs>::iterator iPos = histoMap_.find(tag);
    if (iPos == histoMap_.end()) {
      std::string hname, htitle;

      hname = "nPxlClus_";
      hname += tag;
      htitle = "# of Pixel Clusters (";
      htitle += tag + ")";
      local_MEs.clusME = iBooker.book1D(hname,
                                        htitle,
                                        ClusHistoPar.getParameter<int32_t>("Xbins"),
                                        ClusHistoPar.getParameter<double>("Xmin"),
                                        ClusHistoPar.getParameter<double>("Xmax"));

      hname = "nPxlVtx_";
      hname += tag;
      htitle = "# of Pixel Vertices (";
      htitle += tag + ")";
      local_MEs.vtxME = iBooker.book1D(hname,
                                       htitle,
                                       VtxHistoPar.getParameter<int32_t>("Xbins"),
                                       VtxHistoPar.getParameter<double>("Xmin"),
                                       VtxHistoPar.getParameter<double>("Xmax"));

      histoMap_.insert(std::make_pair(tag, local_MEs));
    }
  }
}

void PixelVTXMonitor::dqmBeginRun(edm::Run const& iRun, edm::EventSetup const& iSetup) {
  bool changed = true;
  if (hltConfig_.init(iRun, iSetup, hltInputTag_.process(), changed)) {
    // if init returns TRUE, initialisation has succeeded!
    edm::LogInfo("PixelVTXMonitor") << "HLT config with process name " << hltInputTag_.process()
                                    << " successfully extracted";
  } else {
    // if init returns FALSE, initialisation has NOT succeeded, which indicates a problem
    // with the file and/or code and needs to be investigated!
    edm::LogError("PixelVTXMonotor") << "Error! HLT config extraction with process name " << hltInputTag_.process()
                                     << " failed";
    // In this case, all access methods will return empty values!
  }
}
void PixelVTXMonitor::analyze(edm::Event const& iEvent, edm::EventSetup const& iSetup) {
  if (histoMap_.empty())
    return;

  //Access Pixel Clusters
  edm::Handle<SiPixelClusterCollectionNew> siPixelClusters = iEvent.getHandle(pixelClusterInputTagToken_);

  if (!siPixelClusters.isValid()) {
    edm::LogError("PixelVTXMonotor") << "Could not find Cluster Collection " << pixelClusterInputTag_;
    return;
  }
  unsigned nClusters = siPixelClusters->size();

  //Access Pixel Verteces
  edm::Handle<reco::VertexCollection> pixelVertices = iEvent.getHandle(pixelVertexInputTagToken_);
  if (!pixelVertices.isValid()) {
    edm::LogError("PixelVTXMonotor") << "Could not find Vertex Collection " << pixelVertexInputTag_;
    return;
  }

  int nVtx = 0;
  for (const auto& ivtx : *pixelVertices) {
    if (minVtxDoF_ == -1)
      nVtx++;
    else {
      if ((ivtx.isValid() == true) && (ivtx.isFake() == false) && (ivtx.ndof() >= minVtxDoF_) &&
          (ivtx.tracksSize() != 0))
        nVtx++;
    }
  }
  // Access Trigger Results
  edm::Handle<edm::TriggerResults> triggerResults = iEvent.getHandle(hltInputTagToken_);
  if (!triggerResults.isValid())
    return;

  for (const auto& it : histoMap_) {
    std::string path = it.first;
    MonitorElement* me_clus = it.second.clusME;
    MonitorElement* me_vtx = it.second.vtxME;
    unsigned int index = hltConfig_.triggerIndex(path);
    if (index < triggerResults->size() && triggerResults->accept(index)) {
      if (me_vtx)
        me_vtx->Fill(nVtx);
      if (me_clus)
        me_clus->Fill(nClusters);
    }
  }
}

// Define this as a plug-in
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(PixelVTXMonitor);
