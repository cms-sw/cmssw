// -*- C++ -*-
//
// Package:     HigPhotonJetHLTOfflineSource
// Class:       HigPhotonJetHLTOfflineSource
// 

//
// Author: Xin Shi <Xin.Shi@cern.ch> 
// Created: 2014.07.22 
//

// system include files
#include <memory>
#include <iostream>

// user include files

#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "DQMServices/Core/interface/DQMEDAnalyzer.h"

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "HLTrigger/HLTcore/interface/HLTConfigProvider.h"

#include "DataFormats/JetReco/interface/CaloJetCollection.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/EgammaCandidates/interface/Photon.h"
#include "DataFormats/EgammaCandidates/interface/PhotonFwd.h"

#include "DataFormats/METReco/interface/PFMETCollection.h"
#include "DataFormats/METReco/interface/PFMET.h"

#include "TPRegexp.h"
#include "TH1F.h" 

//  Define the interface

class HigPhotonJetHLTOfflineSource : public DQMEDAnalyzer {

public:

  explicit HigPhotonJetHLTOfflineSource(const edm::ParameterSet&);

private:

  // Analyzer Methods
  virtual void beginJob();
  virtual void dqmBeginRun(const edm::Run &, const edm::EventSetup &) override;
  virtual void bookHistograms(DQMStore::IBooker &,
			      edm::Run const &, edm::EventSetup const &) override;  
  virtual void analyze(const edm::Event &, const edm::EventSetup &) override;
  virtual void endRun(const edm::Run &, const edm::EventSetup &) override;
  virtual void endJob();

  // Extra Methods
  std::vector<std::string> moduleLabels(std::string);

  // Input from Configuration File
  edm::ParameterSet pset_;
  std::string hltProcessName_;
  std::vector<std::string> hltPathsToCheck_;
  std::string dirname_; 
  bool verbose_; 

  // Member Variables
  HLTConfigProvider hltConfig_;
  // DQMStore * dbe_;

  // CaloJet 
  edm::EDGetTokenT<reco::CaloJetCollection> caloJetsToken_;
  // Vertex 
  edm::EDGetTokenT<reco::VertexCollection> pvToken_;
  // Photon
  edm::EDGetTokenT<reco::PhotonCollection> photonsToken_;
  // PFMET
  edm::EDGetTokenT<reco::PFMETCollection> pfMetToken_;
  
  MonitorElement*  ncalojets_;
  MonitorElement*  nvertices_;
  MonitorElement*  nphotons_;
  MonitorElement*  photonpt_;
  MonitorElement*  pfmet_;
  
};



// Class Methods 

HigPhotonJetHLTOfflineSource::HigPhotonJetHLTOfflineSource(const edm::ParameterSet& pset) :
  pset_(pset)
{
  hltProcessName_ = pset.getParameter<std::string>("hltProcessName"); 
  hltPathsToCheck_ = pset.getParameter<std::vector<std::string>>("hltPathsToCheck"); 
  verbose_ = pset.getUntrackedParameter<bool>("verbose", false);
  dirname_ = pset.getUntrackedParameter<std::string>("dirname", std::string("HLT/xshi/"));
  caloJetsToken_ = consumes<reco::CaloJetCollection> (pset.getParameter<edm::InputTag>("caloJetsToken"));
  pvToken_ = consumes<reco::VertexCollection> (pset.getParameter<edm::InputTag>("pvToken"));
  photonsToken_ = consumes<reco::PhotonCollection> (pset.getParameter<edm::InputTag>("photonsToken"));
  pfMetToken_ = consumes<reco::PFMETCollection> (pset.getParameter<edm::InputTag>("pfMetToken"));


}

std::vector<std::string> 
HigPhotonJetHLTOfflineSource::moduleLabels(std::string path) 
{

  std::vector<std::string> modules = hltConfig_.moduleLabels(path);
  std::vector<std::string>::iterator iter = modules.begin();

  while (iter != modules.end())
    if (iter->find("Filtered") == std::string::npos) 
      iter = modules.erase(iter);
    else
      ++iter;

  return modules;
}


void 
HigPhotonJetHLTOfflineSource::dqmBeginRun(const edm::Run & iRun, 
				    const edm::EventSetup & iSetup) 
{

  // Initialize hltConfig
  bool changedConfig;
  if (!hltConfig_.init(iRun, iSetup, hltProcessName_, changedConfig)) {
    edm::LogError("HLTPhotonJetVal") << "Initialization of HLTConfigProvider failed!!"; 
    return;
  }
  
  // Get the set of trigger paths we want to make plots for
  std::set<std::string> hltPaths;
  for (size_t i = 0; i < hltPathsToCheck_.size(); i++) {
    TPRegexp pattern(hltPathsToCheck_[i]);
    for (size_t j = 0; j < hltConfig_.triggerNames().size(); j++)
      if (TString(hltConfig_.triggerNames()[j]).Contains(pattern))
        hltPaths.insert(hltConfig_.triggerNames()[j]);
  }
  
  // Initialize the plotters
  std::set<std::string>::iterator iPath;
  for (iPath = hltPaths.begin(); iPath != hltPaths.end(); iPath++) {
    std::string path = * iPath;
    std::vector<std::string> labels = moduleLabels(path);
    if (labels.size() > 0) {
      // plotterContainer_.addPlotter(pset_, path, moduleLabels(path));
    }
  }
}


void 
HigPhotonJetHLTOfflineSource::bookHistograms(DQMStore::IBooker & iBooker, 
				       edm::Run const & iRun,
				       edm::EventSetup const & iSetup)
{
  // plotterContainer_.beginRun(iBooker, iRun, iSetup);
  
  // TH1F *h = new TH1F(name.c_str(), title.c_str(), nBins, min, max);
  //Assuming you have a map of map<std::string, MonitorElements*> called “elements”
  // elements[“ptMuon”]->Fill(muon->pt());
 
  // TH1F *h = new TH1F("ncalojets", "Num of Calo Jets", 100, 0., 100.);
  // ncalojets_ = iBooker.book1D("ncalojets", h);
  // delete h;

  iBooker.setCurrentFolder(dirname_);
  
  // ncalojets_ = iBooker.book1D("ncalojets", "Num of Calo Jets", 100, 0., 100.);

  // std::string path = "Photon135_PFMET40"; 
  
  // iBooker.setCurrentFolder(dirname_ + "/" + path);

  ncalojets_ = iBooker.book1D("ncalojets", "Number of Calo Jets", 100, 0., 100.);
  
  nvertices_ = iBooker.book1D("nvertices", "Number of vertices", 100, 0, 100); 

  nphotons_ = iBooker.book1D("nphotons", "Number of photons", 100, 0, 100); 
  photonpt_ = iBooker.book1D("photonpt", "Photons pT", 100, 0, 100); 
  pfmet_ = iBooker.book1D("pfmet", "PF MET", 100, 0, 100); 

}


void
HigPhotonJetHLTOfflineSource::analyze(const edm::Event& iEvent, 
				const edm::EventSetup& iSetup)
{

  // plotterContainer_.analyze(iEvent, iSetup);
  // std::map<std::string, MonitorElements*> elements;
  
  // Get CaloJet
  edm::Handle<reco::CaloJetCollection> calojets;
  iEvent.getByToken(caloJetsToken_, calojets);
  if(!calojets.isValid()) return;
  // reco::CaloJetCollection calojet = *calojetColl_; 
  // edm::LogInfo(">>> xshi >>> Number of CaloJets : ") << calojet.size();   
  if (verbose_)
    std::cout << "xshi:: N calojets : " << calojets->size() << std::endl;

  ncalojets_->Fill(calojets->size()); 

  edm::Handle<reco::VertexCollection> vertices;
  iEvent.getByToken(pvToken_, vertices);
  if(!vertices.isValid()) return;  
  if (verbose_)
    std::cout << "xshi:: N vertices : " << vertices->size() << std::endl;

  nvertices_->Fill(vertices->size());

  // photons pT
  edm::Handle<reco::PhotonCollection> photons;
  iEvent.getByToken(photonsToken_, photons);
  if(!photons.isValid()) return;  
  if (verbose_)
    std::cout << "xshi:: N photons : " << photons->size() << std::endl;
  nphotons_->Fill(photons->size());
  for(reco::PhotonCollection::const_iterator phoIter=photons->begin();
      phoIter!=photons->end();++phoIter){
    photonpt_->Fill(phoIter->pt()); 
  }

  // PF MET
  edm::Handle<reco::PFMETCollection> pfmets;
  iEvent.getByToken(pfMetToken_, pfmets);
  if (!pfmets.isValid()) return;
  if (verbose_)
    std::cout << "xshi:: N pfmets: " << pfmets->size() << std::endl;

  // const PFMETCollection *pfmetcol = pfmetColl_.product();
  const reco::PFMET pfmet = pfmets->front();

  if (verbose_)
    std::cout << "xshi:: PFMET: " << pfmet.et() << std::endl;

  pfmet_->Fill(pfmet.et()); 
}


void 
HigPhotonJetHLTOfflineSource::beginJob()
{
  // dbe_ = edm::Service<DQMStore>().operator->();
  // if ( ! dbe_ ) {
  //   edm::LogError("HigPhotonJetHLTOfflineSource") <<
  //     "Unabel to get DQMStore service";
  // } else {
  //   dbe_->setCurrentFolder(dirname_);
  // }
  
}



void 
HigPhotonJetHLTOfflineSource::endRun(const edm::Run & iRun, 
			       const edm::EventSetup& iSetup)
{

  //   plotterContainer_.endRun(iRun, iSetup);

}



void 
HigPhotonJetHLTOfflineSource::endJob()
{
 
}



//define this as a plug-in
DEFINE_FWK_MODULE(HigPhotonJetHLTOfflineSource);
