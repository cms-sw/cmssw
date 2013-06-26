/*
 * \file PixelVTXMonitor.cc
 * \author S. Dutta
 * Last Update:
 * $Date: 2011/08/30 06:21:16 $
 * $Revision: 1.6 $
 * $Author: dutta $
 *
 * Description: Pixel Vertex Monitoring for different HLT paths
 *
*/
#include "DQM/BeamMonitor/plugins/PixelVTXMonitor.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/SiPixelCluster/interface/SiPixelCluster.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/Common/interface/TriggerResults.h"
#include "TPRegexp.h"


// -----------------------------
//  constructors and destructor
// -----------------------------

PixelVTXMonitor::PixelVTXMonitor( const edm::ParameterSet& ps ) : parameters_(ps) {


  moduleName_          = parameters_.getParameter<std::string>("ModuleName");
  folderName_          = parameters_.getParameter<std::string>("FolderName");
  pixelClusterInputTag_= parameters_.getParameter<edm::InputTag>("PixelClusterInputTag");
  pixelVertexInputTag_ = parameters_.getParameter<edm::InputTag>("PixelVertexInputTag");
  hltInputTag_         = parameters_.getParameter<edm::InputTag>("HLTInputTag");
  minVtxDoF_           = parameters_.getParameter<double>("MinVtxDoF");  
}

PixelVTXMonitor::~PixelVTXMonitor() {

}

void PixelVTXMonitor::bookHistograms() {
  std::vector<std::string> hltPathsOfInterest = parameters_.getParameter<std::vector<std::string> > ("HLTPathsOfInterest");
  if (hltPathsOfInterest.size()  == 0) return;

  const std::vector<std::string>& pathList = hltConfig_.triggerNames();
  std::vector<std::string> selectedPaths;
  for (std::vector<std::string>::const_iterator it = pathList.begin();
       it != pathList.end(); ++it) {
    int nmatch = 0;
    for (std::vector<std::string>::const_iterator kt = hltPathsOfInterest.begin();
	 kt != hltPathsOfInterest.end(); ++kt) {
      nmatch += TPRegexp(*kt).Match(*it);
    }
    if (!nmatch) continue;
    else selectedPaths.push_back(*it);     
  }
    
  edm::ParameterSet ClusHistoPar =  parameters_.getParameter<edm::ParameterSet>("TH1ClusPar");
  edm::ParameterSet VtxHistoPar  =  parameters_.getParameter<edm::ParameterSet>("TH1VtxPar");


  std::string currentFolder = moduleName_ + "/" + folderName_ ;
  dbe_->setCurrentFolder(currentFolder.c_str());

  PixelMEs local_MEs;
  for (std::vector<std::string> ::iterator it = selectedPaths.begin();
       it != selectedPaths.end(); it++) {
    std::string tag = (*it) ;
    std::map<std::string, PixelMEs>::iterator iPos = histoMap_.find(tag); 
    if (iPos == histoMap_.end()) {
      
      std::string hname, htitle;

      hname  = "nPxlClus_";
      hname += tag;
      htitle= "# of Pixel Clusters (";
      htitle += tag +")";
      local_MEs.clusME= dbe_->book1D(hname, htitle, 
        ClusHistoPar.getParameter<int32_t>("Xbins"),
        ClusHistoPar.getParameter<double>("Xmin"),
        ClusHistoPar.getParameter<double>("Xmax"));

      hname = "nPxlVtx_";
      hname += tag;
      htitle= "# of Pixel Vertices (";
      htitle += tag +")";
      local_MEs.vtxME= dbe_->book1D(hname, htitle,
         VtxHistoPar.getParameter<int32_t>("Xbins"),
         VtxHistoPar.getParameter<double>("Xmin"),
         VtxHistoPar.getParameter<double>("Xmax"));

      histoMap_.insert(std::make_pair(tag, local_MEs)); 
    } 
  }
}

void PixelVTXMonitor::beginJob() {
  dbe_ = edm::Service<DQMStore>().operator->();
 
}

void PixelVTXMonitor::beginRun(edm::Run const& iRun, edm::EventSetup const& iSetup) {
  bool changed = true;
  if (hltConfig_.init(iRun, iSetup, hltInputTag_.process(), changed)) {
    // if init returns TRUE, initialisation has succeeded!
    edm::LogInfo("PixelVTXMonitor") << "HLT config with process name " 
				     << hltInputTag_.process() << " successfully extracted";
  }  else {
    // if init returns FALSE, initialisation has NOT succeeded, which indicates a problem
    // with the file and/or code and needs to be investigated!
    edm::LogError("PixelVTXMonotor") << "Error! HLT config extraction with process name " 
                                  <<hltInputTag_.process() << " failed";
    // In this case, all access methods will return empty values!
  }
  bookHistograms();

}
void PixelVTXMonitor::analyze(edm::Event const& iEvent, edm::EventSetup const& iSetup)  {
  if (!histoMap_.size()) return;

  //Access Pixel Clusters
  edm::Handle< SiPixelClusterCollectionNew > siPixelClusters;
  iEvent.getByLabel(pixelClusterInputTag_, siPixelClusters);
  
  if(!siPixelClusters.isValid()) {
    edm::LogError("PixelVTXMonotor") << "Could not find Cluster Collection " << pixelClusterInputTag_;
    return;
  }
  unsigned nClusters = siPixelClusters->size();
  

  //Access Pixel Verteces
  edm::Handle<reco::VertexCollection> pixelVertices;
  iEvent.getByLabel(pixelVertexInputTag_,pixelVertices);
  if (!pixelVertices.isValid()) {
    edm::LogError("PixelVTXMonotor") << "Could not find Vertex Collection " << pixelVertexInputTag_;
    return;
  }

  int nVtx = 0;
  for (reco::VertexCollection::const_iterator ivtx = pixelVertices->begin(); 
       ivtx != pixelVertices->end(); ++ivtx) {
    if (minVtxDoF_ == -1) nVtx++;
    else {
      if ((ivtx->isValid() == true) &&
	  (ivtx->isFake() == false) &&
	  (ivtx->ndof() >= minVtxDoF_) &&
	  (ivtx->tracksSize() != 0)) nVtx++;
    }
  }
  // Access Trigger Results
  edm::Handle<edm::TriggerResults> triggerResults;
  iEvent.getByLabel(hltInputTag_, triggerResults);
  if (!triggerResults.isValid()) return;

  for (std::map<std::string,PixelMEs>::iterator it = histoMap_.begin();
       it != histoMap_.end(); ++it) {
    std::string path = it->first; 
    MonitorElement* me_clus  = it->second.clusME;
    MonitorElement* me_vtx  = it->second.vtxME;
    unsigned int index = hltConfig_.triggerIndex(path);
    if ( index < triggerResults->size() && triggerResults->accept(index)) {
      if (me_vtx) me_vtx->Fill(nVtx);
      if (me_clus) me_clus->Fill(nClusters);
    } 
  } 
}


void PixelVTXMonitor::endRun(edm::Run const& iRun, edm::EventSetup const& iSetup) {

}


void PixelVTXMonitor::endJob() {

}
// Define this as a plug-in
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(PixelVTXMonitor);
