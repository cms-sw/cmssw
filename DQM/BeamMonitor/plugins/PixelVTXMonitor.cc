/*
 * \file PixelVTXMonitor.cc
 * \author S. Dutta
 * Last Update:
 * $Date: 2011/08/10 05:14:06 $
 * $Revision: 1.1 $
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
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/Common/interface/TriggerResults.h"


// -----------------------------
//  constructors and destructor
// -----------------------------

PixelVTXMonitor::PixelVTXMonitor( const edm::ParameterSet& ps ) : parameters_(ps) {


  moduleName_          = parameters_.getParameter<std::string>("ModuleName");
  folderName_          = parameters_.getParameter<std::string>("FolderName");
  pixelVertexInputTag_ = parameters_.getParameter<edm::InputTag>("PixelVertexInputTag");
  hltInputTag_         = parameters_.getParameter<edm::InputTag>("HLTInputTag");
  minVtxDoF_           = parameters_.getParameter<double>("MinVtxDoF");  
}

PixelVTXMonitor::~PixelVTXMonitor() {

}

void PixelVTXMonitor::bookHistograms() {
  std::vector<std::string> hltPathsOfInterest = parameters_.getParameter<std::vector<std::string> > ("HLTPathsOfInterest");
  for (std::vector<std::string> ::iterator it = hltPathsOfInterest.begin();
       it != hltPathsOfInterest.end(); it++) {
    std::string tag = (*it) ;
    std::string hname = "nPxlVtx_";
    hname += tag;

    std::string htitle= "# of Pixel Vertices (";
    htitle += tag +")";
    MonitorElement* me = dbe_->book1D(hname, htitle, 101, -0.5, 100.5);
    vtxHistoMap_.insert(std::make_pair(tag, me));  
  }
}

void PixelVTXMonitor::beginJob() {
  dbe_ = edm::Service<DQMStore>().operator->();
  std::string currentFolder = moduleName_ + "/" + folderName_ ;
  dbe_->setCurrentFolder(currentFolder.c_str());

  bookHistograms();
  
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

}
void PixelVTXMonitor::analyze(edm::Event const& iEvent, edm::EventSetup const& iSetup)  {
  if (!vtxHistoMap_.size()) return;

  //Access Pixel Verteces
  edm::Handle<reco::VertexCollection> pixelVertices;
  iEvent.getByLabel(pixelVertexInputTag_,pixelVertices);
  if (!pixelVertices.isValid()) return;
  std::cout << " for runs: " << iEvent.id().run() << std::endl;
  int nVtx = 0;
  for (reco::VertexCollection::const_iterator ivtx = pixelVertices->begin(); 
	 ivtx != pixelVertices->end(); ++ivtx) {
    if ((ivtx->isValid() == true) &&
        (ivtx->isFake() == false) &&
	(ivtx->ndof() >= minVtxDoF_) &&
	(ivtx->tracksSize() != 0)) nVtx++;
  }

  // Access Trigger Results
  edm::Handle<edm::TriggerResults> triggerResults;
  iEvent.getByLabel(hltInputTag_, triggerResults);
  if (!triggerResults.isValid()) return;

  for (std::map<std::string,MonitorElement*>::iterator it = vtxHistoMap_.begin();
       it != vtxHistoMap_.end(); ++it) {
    std::string path = it->first; 
    MonitorElement* me = it->second;
    unsigned int index = hltConfig_.triggerIndex(path);
    if (index < triggerResults->size() && triggerResults->accept(index)) {
      me->Fill(nVtx);
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
