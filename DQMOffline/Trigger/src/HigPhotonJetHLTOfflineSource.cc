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
#include "FWCore/Common/interface/TriggerNames.h"

#include "HLTrigger/HLTcore/interface/HLTConfigProvider.h"

#include "DataFormats/Common/interface/TriggerResults.h"
#include "DataFormats/JetReco/interface/CaloJetCollection.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/EgammaCandidates/interface/Photon.h"
#include "DataFormats/EgammaCandidates/interface/PhotonFwd.h"
#include "DataFormats/METReco/interface/PFMETCollection.h"
#include "DataFormats/METReco/interface/PFMET.h"
#include "DataFormats/JetReco/interface/PFJetCollection.h"
#include "DataFormats/JetReco/interface/PFJet.h"
#include "DataFormats/Math/interface/deltaPhi.h"

#include <TLorentzVector.h>
#include <TH2F.h>

//  Define the interface
class HigPhotonJetHLTOfflineSource : public DQMEDAnalyzer {

public:

  explicit HigPhotonJetHLTOfflineSource(const edm::ParameterSet&);

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
  edm::EDGetTokenT<reco::PFMETCollection> pfMetToken_;
  edm::EDGetTokenT<reco::PFJetCollection> pfJetsToken_;

  double pfjetMinPt_;  
  double photonMinPt_;  

  // Member Variables

  MonitorElement*  nvertices_gen_;
  MonitorElement*  nvertices_;
  MonitorElement*  nphotons_gen_;
  MonitorElement*  nphotons_;
  MonitorElement*  photonpt_gen_;
  MonitorElement*  photonpt_;
  MonitorElement*  photonrapidity_gen_;
  MonitorElement*  photonrapidity_;
  MonitorElement*  pfmet_gen_;
  MonitorElement*  pfmet_;
  MonitorElement*  pfmetphi_gen_;
  MonitorElement*  pfmetphi_;
  MonitorElement*  npfjets_gen_;
  MonitorElement*  npfjets_;
  MonitorElement*  delphiphomet_gen_;
  MonitorElement*  delphiphomet_;
  MonitorElement*  delphijetmet_gen_;
  MonitorElement*  delphijetmet_;
  MonitorElement*  invmassjj_gen_;
  MonitorElement*  invmassjj_;
  MonitorElement*  deletajj_gen_;
  MonitorElement*  deletajj_;
  MonitorElement*  triggers_gen_;
  MonitorElement*  triggers_;
  MonitorElement*  trigvsnvtx_gen_;
  MonitorElement*  trigvsnvtx_;
  
  double evtsrun_; 
  
};


// Class Methods 

HigPhotonJetHLTOfflineSource::HigPhotonJetHLTOfflineSource(const edm::ParameterSet& pset) :
  pset_(pset)
{
  hltProcessName_ = pset.getParameter<std::string>("hltProcessName"); 
  hltPathsToCheck_ = pset.getParameter<std::vector<std::string>>("hltPathsToCheck"); 
  verbose_ = pset.getUntrackedParameter<bool>("verbose", false);
  triggerAccept_ = pset.getUntrackedParameter<bool>("triggerAccept", true);
  triggerResultsToken_ = consumes <edm::TriggerResults> (pset.getParameter<edm::InputTag>("triggerResultsToken"));
  dirname_ = pset.getUntrackedParameter<std::string>("dirname", std::string("HLT/Higgs/PhotonJet/"));
  pvToken_ = consumes<reco::VertexCollection> (pset.getParameter<edm::InputTag>("pvToken"));
  photonsToken_ = consumes<reco::PhotonCollection> (pset.getParameter<edm::InputTag>("photonsToken"));
  pfMetToken_ = consumes<reco::PFMETCollection> (pset.getParameter<edm::InputTag>("pfMetToken"));
  pfJetsToken_ = consumes<reco::PFJetCollection> (pset.getParameter<edm::InputTag>("pfJetsToken"));
  pfjetMinPt_ = pset.getUntrackedParameter<double>("pfjetMinPt", 0.0);
  photonMinPt_ = pset.getUntrackedParameter<double>("photonMinPt", 0.0);
}

void 
HigPhotonJetHLTOfflineSource::dqmBeginRun(const edm::Run & iRun, 
					  const edm::EventSetup & iSetup) 
{ // Initialize hltConfig
  HLTConfigProvider hltConfig;
  bool changedConfig;
  if (!hltConfig.init(iRun, iSetup, hltProcessName_, changedConfig)) {
    edm::LogError("HLTPhotonJetVal") << "Initialization of HLTConfigProvider failed!!"; 
    return;
  }
 
  evtsrun_ = 0; 
}


void 
HigPhotonJetHLTOfflineSource::bookHistograms(DQMStore::IBooker & iBooker, 
					     edm::Run const & iRun,
					     edm::EventSetup const & iSetup)
{
  iBooker.setCurrentFolder(dirname_);
  nvertices_gen_ = iBooker.book1D("nvertices_gen", "Gen: Number of vertices", 100, 0, 100); 
  nvertices_ = iBooker.book1D("nvertices", "Number of vertices", 100, 0, 100); 
  nphotons_gen_ = iBooker.book1D("nphotons_gen", "Gen: Number of photons", 100, 0, 10); 
  nphotons_ = iBooker.book1D("nphotons", "Number of photons", 100, 0, 10); 
  photonpt_gen_ = iBooker.book1D("photonpt_gen", "Gen: Photons pT", 100, 0, 500); 
  photonpt_ = iBooker.book1D("photonpt", "Photons pT", 100, 0, 500); 
  photonrapidity_gen_ = iBooker.book1D("photonrapidity_gen", "Gen: Photons rapidity;y_{#gamma}", 100, -2.5, 2.5); 
  photonrapidity_ = iBooker.book1D("photonrapidity", "Photons rapidity;y_{#gamma}", 100, -2.5, 2.5); 
  pfmet_gen_ = iBooker.book1D("pfmet_gen", "Gen: PF MET", 100, 0, 250); 
  pfmet_ = iBooker.book1D("pfmet", "PF MET", 100, 0, 250); 
  pfmetphi_gen_ = iBooker.book1D("pfmetphi_gen", "Gen: PF MET phi;#phi_{PFMET}", 100, -4, 4); 
  pfmetphi_ = iBooker.book1D("pfmetphi", "PF MET phi;#phi_{PFMET}", 100, -4, 4); 
  delphiphomet_gen_ = iBooker.book1D("delphiphomet_gen", "Gen: #Delta#phi(photon, MET);#Delta#phi(#gamma,MET)", 100, 0, 4); 
  delphiphomet_ = iBooker.book1D("delphiphomet", "#Delta#phi(photon, MET);#Delta#phi(#gamma,MET)", 100, 0, 4); 
  npfjets_gen_ = iBooker.book1D("npfjets_gen", "Gen: Number of PF Jets", 100, 0, 20); 
  npfjets_ = iBooker.book1D("npfjets", "Number of PF Jets", 100, 0, 20); 
  delphijetmet_gen_ = iBooker.book1D("delphijetmet_gen", "Gen: #Delta#phi(PFJet, MET);#Delta#phi(Jet,MET)", 100, 0, 4); 
  delphijetmet_ = iBooker.book1D("delphijetmet", "#Delta#phi(PFJet, MET);#Delta#phi(Jet,MET)", 100, 0, 4); 
  invmassjj_gen_ = iBooker.book1D("invmassjj_gen", "Gen: Inv mass two leading jets;M_{jj}[GeV]", 100, 0, 2000); 
  invmassjj_ = iBooker.book1D("invmassjj", "Inv mass two leading jets;M_{jj}[GeV]", 100, 0, 2000); 
  deletajj_gen_ = iBooker.book1D("deletajj_gen", "Gen: #Delta#eta(jj);|#Delta#eta_{jj}|", 100, 0, 6); 
  deletajj_ = iBooker.book1D("deletajj", "#Delta#eta(jj);|#Delta#eta_{jj}|", 100, 0, 6); 
  triggers_gen_ = iBooker.book1D("triggers_gen", "Gen: Triggers", hltPathsToCheck_.size(), 0, hltPathsToCheck_.size());
  triggers_ = iBooker.book1D("triggers", "Triggers", hltPathsToCheck_.size(), 0, hltPathsToCheck_.size());
  trigvsnvtx_gen_ = iBooker.book2D("trigvsnvtx_gen", "Gen: Trigger vs. # vertices;N_{vertices};Trigger", 100, 0, 100, hltPathsToCheck_.size(), 0, hltPathsToCheck_.size()); 
  trigvsnvtx_ = iBooker.book2D("trigvsnvtx", "Trigger vs. # vertices;N_{vertices};Trigger", 100, 0, 100, hltPathsToCheck_.size(), 0, hltPathsToCheck_.size()); 
}


void
HigPhotonJetHLTOfflineSource::analyze(const edm::Event& iEvent, 
				      const edm::EventSetup& iSetup)
{
  // Count total number of events in one run
  evtsrun_++; 

  edm::Handle<edm::TriggerResults> triggerResults;
  iEvent.getByToken(triggerResultsToken_, triggerResults);
  if(!triggerResults.isValid()) {
      edm::LogError("HigPhotonJetHLT")<<"Missing triggerResults collection" << std::endl;
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
  	 triggers_gen_->Fill(i);
	 trigvsnvtx_gen_->Fill(vertices->size(), i); 
  	 if (triggered) triggers_->Fill(i);
  	 if (triggered) trigvsnvtx_->Fill(vertices->size(), i);
      }
    }
  }

  nvertices_gen_->Fill(vertices->size());
  if (triggered) nvertices_->Fill(vertices->size());

  // PF MET
  edm::Handle<reco::PFMETCollection> pfmets;
  iEvent.getByToken(pfMetToken_, pfmets);
  if (!pfmets.isValid()) return;
  const reco::PFMET pfmet = pfmets->front();
  pfmet_gen_->Fill(pfmet.et()); 
  if (triggered) pfmet_->Fill(pfmet.et()); 
  if (verbose_)
    std::cout << "xshi:: number of pfmets: " << pfmets->size() << std::endl;

  pfmetphi_gen_->Fill(pfmet.phi()); 
  if (triggered) pfmetphi_->Fill(pfmet.phi()); 
  
  // Photons
  edm::Handle<reco::PhotonCollection> photons;
  iEvent.getByToken(photonsToken_, photons);
  if(!photons.isValid()) return;
  int nphotons = 0; 
  for(reco::PhotonCollection::const_iterator phoIter=photons->begin();
      phoIter!=photons->end();++phoIter){
    if (phoIter->pt() < photonMinPt_ )  continue;
    nphotons++;
    photonpt_gen_->Fill(phoIter->pt());
    photonrapidity_gen_->Fill(phoIter->rapidity()); 
    if (triggered) photonpt_->Fill(phoIter->pt()); 
    if (triggered) photonrapidity_->Fill(phoIter->rapidity()); 
    double tmp_delphiphomet = fabs(deltaPhi(phoIter->phi(), pfmet.phi())); 
    delphiphomet_gen_->Fill(tmp_delphiphomet); 
    if (triggered) delphiphomet_->Fill(tmp_delphiphomet); 
  }
  nphotons_gen_->Fill(nphotons);
  if (triggered)  nphotons_->Fill(nphotons);
  
  // PF Jet
  edm::Handle<reco::PFJetCollection> pfjets;
  iEvent.getByToken(pfJetsToken_, pfjets);
  if(!pfjets.isValid()) return;
  if (verbose_)
    std::cout << "xshi:: N pfjets : " << pfjets->size() << std::endl;

  double min_delphijetmet = 6.0;
  TLorentzVector p4jet1, p4jet2, p4jj;
  // Two leading jets eta
  double etajet1(0), etajet2(0);
  int njet = 0;  
  for(reco::PFJetCollection::const_iterator jetIter=pfjets->begin();
      jetIter!=pfjets->end();++jetIter){
    if (jetIter->pt() < pfjetMinPt_ ) continue; 
    njet++;

    double tmp_delphijetmet = fabs(deltaPhi(jetIter->phi(), pfmet.phi())); 
    if (tmp_delphijetmet < min_delphijetmet)
      min_delphijetmet = tmp_delphijetmet;

    if (njet == 1) {
      p4jet1.SetXYZM(jetIter->px(), jetIter->py(), jetIter->pz(), jetIter->mass()); 
      etajet1 = jetIter->eta(); 
    }
    if (njet == 2){
      p4jet2.SetXYZM(jetIter->px(), jetIter->py(), jetIter->pz(), jetIter->mass()); 
      etajet2 = jetIter->eta(); 
    }
  }
  npfjets_gen_->Fill(njet);   
  if (triggered) npfjets_->Fill(njet);   
  
  delphijetmet_gen_->Fill(min_delphijetmet); 
  if (triggered) delphijetmet_->Fill(min_delphijetmet); 
  p4jj = p4jet1 + p4jet2; 
  double deletajj = etajet1 - etajet2 ; 
  if (verbose_) 
    std::cout << "xshi:: invmass jj " << p4jj.M() << std::endl;
  
  invmassjj_gen_->Fill(p4jj.M());
  deletajj_gen_->Fill(deletajj); 
  if (triggered) invmassjj_->Fill(p4jj.M());
  if (triggered) deletajj_->Fill(deletajj); 
}


void 
HigPhotonJetHLTOfflineSource::endRun(const edm::Run & iRun, 
				     const edm::EventSetup& iSetup)
{
  // Normalize to the total number of events in the run
  TH2F* h = trigvsnvtx_->getTH2F();
  double norm = evtsrun_*hltPathsToCheck_.size()/h->Integral(); 
  h->Scale(norm);
  if (verbose_) {
    std::cout << "xshi:: endRun total number of events: " << evtsrun_
  	      << ", integral = " << h->Integral()
  	      << ", norm = " << norm << std::endl;
  }
}

bool
HigPhotonJetHLTOfflineSource::isMonitoredTriggerAccepted(const edm::TriggerNames triggerNames,
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
DEFINE_FWK_MODULE(HigPhotonJetHLTOfflineSource);
