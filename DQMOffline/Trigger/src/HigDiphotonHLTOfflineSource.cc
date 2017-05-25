// -*- C++ -*-
//
// Package:     HigPhotonJetHLTOfflineSource
// Class:       HigPhotonJetHLTOfflineSource
// 

//
// Author: Xin Shi <Xin.Shi@cern.ch> 
// Created: 2014.07.22 
//

//copied from HigPhotonJetHLTOfllineSource (by mplaner for diphoton hgg paths)

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
#include "FWCore/Common/interface/TriggerNames.h"

#include "HLTrigger/HLTcore/interface/HLTConfigProvider.h"

#include "DataFormats/Common/interface/TriggerResults.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/EgammaCandidates/interface/Photon.h"
#include "DataFormats/EgammaCandidates/interface/PhotonFwd.h"
#include "DataFormats/Math/interface/deltaPhi.h"

#include <TLorentzVector.h>
#include <TH2F.h>

//  Define the interface
class HigDiphotonHLTOfflineSource : public DQMEDAnalyzer {

public:

  explicit HigDiphotonHLTOfflineSource(const edm::ParameterSet&);

private:

  // Analyzer Methods
  virtual void dqmBeginRun(const edm::Run &,
			   const edm::EventSetup &) override;
  virtual void bookHistograms(DQMStore::IBooker &,
			      edm::Run const &,
			      edm::EventSetup const &) override;  
  virtual void analyze(const edm::Event &,
		       const edm::EventSetup &) override;
  virtual void endRun(const edm::Run &,
		      const edm::EventSetup &) override;
  bool isMonitoredTriggerAccepted(const edm::TriggerNames,
				  const edm::Handle<edm::TriggerResults>); 

  // Input from Configuration File
  edm::ParameterSet pset_;
  std::string hltProcessName_;
  std::vector<std::string> hltPathsToCheck_;
  std::string dirname_; 
  bool verbose_;
  bool triggerAccept_; 
    
  edm::EDGetTokenT <edm::TriggerResults> triggerResultsToken_;
  edm::EDGetTokenT<reco::VertexCollection> pvToken_;
  edm::EDGetTokenT<reco::PhotonCollection> photonsToken_;

  double photonMinPt_;  

  // Member Variables

  MonitorElement*  nvertices_reco_;
  MonitorElement*  nvertices_;
  MonitorElement*  nphotons_reco_;
  MonitorElement*  nphotons_;
  MonitorElement*  photonpt_reco_;
  MonitorElement*  photonpt_;
  MonitorElement*  photonrapidity_reco_;
  MonitorElement*  photonrapidity_;
  MonitorElement*  triggers_reco_;
  MonitorElement*  triggers_;
  MonitorElement*  trigvsnvtx_reco_;
  MonitorElement*  trigvsnvtx_;

  //  MonitorElement*  diphotonmass_;  //new!! FIXME
  //MonitorElement*  diphotonmass_reco_;  //new!! FIXME
  
  double evtsrun_; 
  
};


// Class Methods 

HigDiphotonHLTOfflineSource::HigDiphotonHLTOfflineSource(const edm::ParameterSet& pset) :
  pset_(pset)
{
  hltProcessName_ = pset.getParameter<std::string>("hltProcessName"); 
  hltPathsToCheck_ = pset.getParameter<std::vector<std::string>>("hltPathsToCheck"); 
  verbose_ = pset.getUntrackedParameter<bool>("verbose", false);
  triggerAccept_ = pset.getUntrackedParameter<bool>("triggerAccept", true);
  triggerResultsToken_ = consumes <edm::TriggerResults> (pset.getParameter<edm::InputTag>("triggerResultsToken"));
  dirname_ = pset.getUntrackedParameter<std::string>("dirname", std::string("HLT/Higgs/Diphoton/"));
  pvToken_ = consumes<reco::VertexCollection> (pset.getParameter<edm::InputTag>("pvToken"));
  photonsToken_ = consumes<reco::PhotonCollection> (pset.getParameter<edm::InputTag>("photonsToken"));
  photonMinPt_ = pset.getUntrackedParameter<double>("photonMinPt", 0.0);
}

void 
HigDiphotonHLTOfflineSource::dqmBeginRun(const edm::Run & iRun, 
					  const edm::EventSetup & iSetup) 
{ // Initialize hltConfig
  HLTConfigProvider hltConfig;
  bool changedConfig;
  if (!hltConfig.init(iRun, iSetup, hltProcessName_, changedConfig)) {
    edm::LogError("HLTDiphotonVal") << "Initialization of HLTConfigProvider failed!!"; 
    return;
  }
 
  evtsrun_ = 0; 
}


void 
HigDiphotonHLTOfflineSource::bookHistograms(DQMStore::IBooker & iBooker, 
					     edm::Run const & iRun,
					     edm::EventSetup const & iSetup)
{
  iBooker.setCurrentFolder(dirname_);
  nvertices_reco_ = iBooker.book1D("nvertices_reco", "Reco: Number of vertices", 100, 0, 100); 
  nvertices_ = iBooker.book1D("nvertices", "Number of vertices", 100, 0, 100); 
  nphotons_reco_ = iBooker.book1D("nphotons_reco", "Reco: Number of photons", 100, 0, 10); 
  nphotons_ = iBooker.book1D("nphotons", "Number of photons", 100, 0, 10); 
  photonpt_reco_ = iBooker.book1D("photonpt_reco", "Reco: Photons pT", 100, 0, 500); 
  //diphotonmass_reco_ = iBooker.book1D("photonmass_reco", "Reco: Photons mass", 50, 0, 200); //diphoton_mass_
  //  diphotonmass_ = iBooker.book1D("photonmass", "Photons mass", 50, 0, 200); //diphoton_mass_
  photonpt_ = iBooker.book1D("photonpt", "Photons pT", 100, 0, 500); 
  photonrapidity_reco_ = iBooker.book1D("photonrapidity_reco", "Reco: Photons rapidity;y_{#gamma}", 100, -2.5, 2.5); 
  photonrapidity_ = iBooker.book1D("photonrapidity", "Photons rapidity;y_{#gamma}", 100, -2.5, 2.5); 
  triggers_reco_ = iBooker.book1D("triggers_reco", "Reco: Triggers", hltPathsToCheck_.size(), 0, hltPathsToCheck_.size());
  triggers_ = iBooker.book1D("triggers", "Triggers", hltPathsToCheck_.size(), 0, hltPathsToCheck_.size());
  trigvsnvtx_reco_ = iBooker.book2D("trigvsnvtx_reco", "Reco: Trigger vs. # vertices;N_{vertices};Trigger", 100, 0, 100, hltPathsToCheck_.size(), 0, hltPathsToCheck_.size()); 
  trigvsnvtx_ = iBooker.book2D("trigvsnvtx", "Trigger vs. # vertices;N_{vertices};Trigger", 100, 0, 100, hltPathsToCheck_.size(), 0, hltPathsToCheck_.size()); 
}


void
HigDiphotonHLTOfflineSource::analyze(const edm::Event& iEvent, 
				      const edm::EventSetup& iSetup)
{
  // Count total number of events in one run
  evtsrun_++; 

  edm::Handle<edm::TriggerResults> triggerResults;
  iEvent.getByToken(triggerResultsToken_, triggerResults);
  if(!triggerResults.isValid()) {
      edm::LogError("HigDiphotonHLT")<<"Missing triggerResults collection" << std::endl;
      return;
  }

  // Check whether contains monitored trigger and accepted
  const edm::TriggerNames triggerNames = iEvent.triggerNames(*triggerResults); 
  bool triggered = isMonitoredTriggerAccepted(triggerNames, triggerResults); 

  // if (!triggered) return; 

  // Test scale 
  // if (evtsrun_ > 10) return;

  // N Vertices 
  edm::Handle<reco::VertexCollection> vertices;
  iEvent.getByToken(pvToken_, vertices);
  if(!vertices.isValid()) return;  
  if (verbose_)
    std::cout << "xshi:: N vertices : " << vertices->size() << std::endl;
  
  // Set trigger name labels
  for (size_t i = 0; i < hltPathsToCheck_.size(); i++) {
    triggers_->setBinLabel(i+1, hltPathsToCheck_[i]); 
  }

  // Fill trigger info
  for (unsigned int itrig = 0; itrig < triggerResults->size(); itrig++){
    std::string triggername = triggerNames.triggerName(itrig);
    for (size_t i = 0; i < hltPathsToCheck_.size(); i++) {
      if ( triggername.find(hltPathsToCheck_[i]) != std::string::npos) {
  	 triggers_reco_->Fill(i);
	 trigvsnvtx_reco_->Fill(vertices->size(), i); 
  	 if (triggered) triggers_->Fill(i);
  	 if (triggered) trigvsnvtx_->Fill(vertices->size(), i);
      }
    }
  }

  nvertices_reco_->Fill(vertices->size());
  if (triggered) nvertices_->Fill(vertices->size());

  // Photons
  edm::Handle<reco::PhotonCollection> photons;
  iEvent.getByToken(photonsToken_, photons);
  if(!photons.isValid()) return;
  int nphotons = 0; 
  for(reco::PhotonCollection::const_iterator phoIter=photons->begin();
      phoIter!=photons->end();++phoIter){
    if (phoIter->pt() < photonMinPt_ )  continue;
    nphotons++;
    photonpt_reco_->Fill(phoIter->pt());
    photonrapidity_reco_->Fill(phoIter->rapidity()); 
    if (triggered) photonpt_->Fill(phoIter->pt()); 
    if (triggered) photonrapidity_->Fill(phoIter->rapidity()); 
  }
  nphotons_reco_->Fill(nphotons);
  if (triggered)  nphotons_->Fill(nphotons);
}


void 
HigDiphotonHLTOfflineSource::endRun(const edm::Run & iRun, 
				     const edm::EventSetup& iSetup)
{
  // Normalize to the total number of events in the run
  TH2F* h = trigvsnvtx_->getTH2F();
  double integral = h->Integral();
  double norm = (integral > 0.) ? evtsrun_*hltPathsToCheck_.size()/integral : 1.;
  h->Scale(norm);
  if (verbose_) {
    std::cout << "xshi:: endRun total number of events: " << evtsrun_
  	      << ", integral = " << h->Integral()
  	      << ", norm = " << norm << std::endl;
  }
}

bool
HigDiphotonHLTOfflineSource::isMonitoredTriggerAccepted(const edm::TriggerNames triggerNames,
							 const edm::Handle<edm::TriggerResults> triggerResults )
{
  for (unsigned int itrig = 0; itrig < triggerResults->size(); itrig++){
    // Only consider the triggered case.
    if ( triggerAccept_ && ( (*triggerResults)[itrig].accept() != 1) ) continue; 
    std::string triggername = triggerNames.triggerName(itrig);
    for (size_t i = 0; i < hltPathsToCheck_.size(); i++) {
      if ( triggername.find(hltPathsToCheck_[i]) != std::string::npos) {
  	return true;
      }
    }
  }

  return false; 
}

//define this as a plug-in
DEFINE_FWK_MODULE(HigDiphotonHLTOfflineSource);
