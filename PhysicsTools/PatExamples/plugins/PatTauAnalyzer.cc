#include "PhysicsTools/PatExamples/plugins/PatTauAnalyzer.h"

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "PhysicsTools/UtilAlgos/interface/TFileService.h"
#include "DataFormats/PatCandidates/interface/Tau.h"

#include <TMath.h>

const reco::GenParticle* getGenTau(const pat::Tau& patTau)
{
  std::vector<reco::GenParticleRef> associatedGenParticles = patTau.genParticleRefs();
  for ( std::vector<reco::GenParticleRef>::const_iterator it = associatedGenParticles.begin(); 
	it != associatedGenParticles.end(); ++it ) {
    if ( it->isAvailable() ) {
      const reco::GenParticleRef& genParticle = (*it);
      if ( genParticle->pdgId() == -15 || genParticle->pdgId() == +15 ) return genParticle.get();
    }
  }

  return 0;
}

PatTauAnalyzer::PatTauAnalyzer(const edm::ParameterSet& cfg)
{
  //std::cout << "<PatTauAnalyzer::PatTauAnalyzer>:" << std::endl;

//--- read name of pat::Tau collection
  src_ = cfg.getParameter<edm::InputTag>("src");
  //std::cout << " src = " << src_ << std::endl;

//--- fill histograms for all tau-jet candidates or for "real" taus only ?
  requireGenTauMatch_ = cfg.getParameter<bool>("requireGenTauMatch");
  //std::cout << " requireGenTauMatch = " << requireGenTauMatch_ << std::endl;

//--- read names of tau id. discriminators
  discrByLeadTrack_ = cfg.getParameter<std::string>("discrByLeadTrack");
  //std::cout << " discrByLeadTrack = " << discrByLeadTrack_ << std::endl;

  discrByIso_ = cfg.getParameter<std::string>("discrByIso");
  //std::cout << " discrByIso = " << discrByIso_ << std::endl;

  discrByTaNC_ = cfg.getParameter<std::string>("discrByTaNC");
  //std::cout << " discrByTaNC = " << discrByTaNC_ << std::endl;
}

PatTauAnalyzer::~PatTauAnalyzer()
{
  //std::cout << "<PatTauAnalyzer::~PatTauAnalyzer>:" << std::endl;

//--- clean-up memory;
//    delete all histograms
/*

  deletion of histograms taken care of by TFileService;
  do not delete them here (if the histograms are deleted here,
  they will not appear in the ROOT file written by TFileService)

  delete hGenTauEnergy_;
  delete hGenTauPt_;
  delete hGenTauEta_;
  delete hGenTauPhi_;
  delete hTauJetEnergy_;
  delete hTauJetPt_;
  delete hTauJetEta_;
  delete hTauJetPhi_;
  delete hNumTauJets_;
  delete hTauLeadTrackPt_;
  delete hTauNumSigConeTracks_;
  delete hTauNumIsoConeTracks_;
  delete hTauDiscrByIso_;
  delete hTauDiscrByTaNC_;
  delete hTauDiscrAgainstElectrons_;
  delete hTauDiscrAgainstMuons_;
  delete hTauJetEnergyIsoPassed_;
  delete hTauJetPtIsoPassed_;
  delete hTauJetEtaIsoPassed_;
  delete hTauJetPhiIsoPassed_;
  delete hTauJetEnergyTaNCpassed_;
  delete hTauJetPtTaNCpassed_;
  delete hTauJetEtaTaNCpassed_;
  delete hTauJetPhiTaNCpassed_;
 */
}

void PatTauAnalyzer::beginJob(const edm::EventSetup& es)
{
//--- retrieve handle to auxiliary service
//    used for storing histograms into ROOT file
  edm::Service<TFileService> fs;

//--- book generator level histograms
  hGenTauEnergy_ = fs->make<TH1F>("GenTauEnergy", "GenTauEnergy", 30, 0., 150.);
  hGenTauPt_ = fs->make<TH1F>("GenTauPt", "GenTauPt", 30, 0., 150.);
  hGenTauEta_ = fs->make<TH1F>("GenTauEta", "GenTauEta", 24, -3., +3.);
  hGenTauPhi_ = fs->make<TH1F>("GenTauPhi", "GenTauPhi", 18, -TMath::Pi(), +TMath::Pi());
  
//--- book reconstruction level histograms
//    for tau-jet Energy, Pt, Eta, Phi
  hTauJetEnergy_ = fs->make<TH1F>("TauJetEnergy", "TauJetEnergy", 30, 0., 150.);
  hTauJetPt_ = fs->make<TH1F>("TauJetPt", "TauJetPt", 30, 0., 150.);
  hTauJetEta_ = fs->make<TH1F>("TauJetEta", "TauJetEta", 24, -3., +3.);
  hTauJetPhi_ = fs->make<TH1F>("TauJetPhi", "TauJetPhi", 18, -TMath::Pi(), +TMath::Pi());

//... for number of tau-jet candidates
  hNumTauJets_ = fs->make<TH1F>("NumTauJets", "NumTauJets", 10, -0.5, 9.5);

//... for Pt of highest Pt track within signal cone tau-jet...
  hTauLeadTrackPt_ = fs->make<TH1F>("TauLeadTrackPt", "TauLeadTrackPt", 40, 0., 100.);
  
//... for total number of tracks within signal/isolation cones
  hTauNumSigConeTracks_ = fs->make<TH1F>("TauNumSigConeTracks", "TauNumSigConeTracks", 10, -0.5,  9.5);
  hTauNumIsoConeTracks_ = fs->make<TH1F>("TauNumIsoConeTracks", "TauNumIsoConeTracks", 20, -0.5, 19.5);

//... for values of tau id. discriminators based on track isolation cut/
//    neural network-based tau id.
  hTauDiscrByIso_ = fs->make<TH1F>("TauDiscrByIso", "TauDiscrByIso", 103, -0.015, 1.015);
  hTauDiscrByTaNC_ = fs->make<TH1F>("TauDiscrByTaNC", "TauDiscrByTaNC", 103, -0.015, 1.015);
  
//... for values of tau id. discriminators against (unidentified) electrons and muons
  hTauDiscrAgainstElectrons_ = fs->make<TH1F>("TauDiscrAgainstElectrons", "TauDiscrAgainstElectrons", 103, -0.015, 1.015);
  hTauDiscrAgainstMuons_ = fs->make<TH1F>("TauDiscrAgainstMuons", "TauDiscrAgainstMuons", 103, -0.015, 1.015);

//... for Energy, Pt, Eta, Phi of tau-jets passing the discriminatorByIsolation selection
  hTauJetEnergyIsoPassed_ = fs->make<TH1F>("TauJetEnergyIsoPassed", "TauJetEnergyIsoPassed", 30, 0., 150.);
  hTauJetPtIsoPassed_ = fs->make<TH1F>("TauJetPtIsoPassed", "TauJetPtIsoPassed", 30, 0., 150.);
  hTauJetEtaIsoPassed_ = fs->make<TH1F>("TauJetEtaIsoPassed", "TauJetEtaIsoPassed", 24, -3., +3.);
  hTauJetPhiIsoPassed_ = fs->make<TH1F>("TauJetPhiIsoPassed", "TauJetPhiIsoPassed", 18, -TMath::Pi(), +TMath::Pi());

//... for Energy, Pt, Eta, Phi of tau-jets passing the discriminatorByTaNC ("Tau Neural Classifier") selection
  hTauJetEnergyTaNCpassed_ = fs->make<TH1F>("TauJetEnergyTaNCpassed", "TauJetEnergyTaNCpassed", 30, 0., 150.);
  hTauJetPtTaNCpassed_ = fs->make<TH1F>("TauJetPtTaNCpassed", "TauJetPtTaNCpassed", 30, 0., 150.);
  hTauJetEtaTaNCpassed_ = fs->make<TH1F>("TauJetEtaTaNCpassed", "TauJetEtaTaNCpassed", 24, -3., +3.);
  hTauJetPhiTaNCpassed_ = fs->make<TH1F>("TauJetPhiTaNCpassed", "TauJetPhiTaNCpassed", 18, -TMath::Pi(), +TMath::Pi());
}

void PatTauAnalyzer::analyze(const edm::Event& evt, const edm::EventSetup& es)
{  
  //std::cout << "<PatTauAnalyzer::analyze>:" << std::endl; 

  edm::Handle<pat::TauCollection> patTaus;
  evt.getByLabel(src_, patTaus);

  hNumTauJets_->Fill(patTaus->size());

  for ( pat::TauCollection::const_iterator patTau = patTaus->begin(); 
	patTau != patTaus->end(); ++patTau ) {

//--- skip fake taus in case configuration parameters set to do so...
    const reco::GenParticle* genTau = getGenTau(*patTau);
    if ( requireGenTauMatch_ && !genTau ) continue;

//--- fill generator level histograms    
    if ( genTau ) {
      hGenTauEnergy_->Fill(genTau->energy());
      hGenTauPt_->Fill(genTau->pt());
      hGenTauEta_->Fill(genTau->eta());
      hGenTauPhi_->Fill(genTau->phi());
    }

//--- fill reconstruction level histograms
//    for Pt of highest Pt track within signal cone tau-jet...
    hTauJetEnergy_->Fill(patTau->energy());
    hTauJetPt_->Fill(patTau->pt());
    hTauJetEta_->Fill(patTau->eta());
    hTauJetPhi_->Fill(patTau->phi());

    if ( patTau->leadTrack().isAvailable() ) hTauLeadTrackPt_->Fill(patTau->leadTrack()->pt());
  
//... for total number of tracks within signal/isolation cones
    hTauNumSigConeTracks_->Fill(patTau->signalTracks().size());
    hTauNumIsoConeTracks_->Fill(patTau->isolationTracks().size());

//... for values of tau id. discriminators based on track isolation cut/
//    neural network-based tau id.
//    (combine with requirement of at least one "leading" track of Pt > 5. GeV 
//     within the signal cone of the tau-jet)
    float discrByIso = ( patTau->tauID(discrByLeadTrack_.data()) > 0.5 ) ? patTau->tauID(discrByIso_.data()) : 0.;
    hTauDiscrByIso_->Fill(discrByIso);
    float discrByTaNC = ( patTau->tauID(discrByLeadTrack_.data()) > 0.5 ) ? patTau->tauID(discrByTaNC_.data()) : 0.;
    hTauDiscrByTaNC_->Fill(discrByTaNC);

//... for values of tau id. discriminators against (unidentified) electrons and muons
    hTauDiscrAgainstElectrons_->Fill(patTau->tauID("againstElectron"));
    hTauDiscrAgainstMuons_->Fill(patTau->tauID("againstMuon"));

//... for Energy, Pt, Eta, Phi of tau-jets passing the discriminatorByIsolation selection
    if ( discrByIso > 0.5 ) {
      hTauJetEnergyIsoPassed_->Fill(patTau->energy());
      hTauJetPtIsoPassed_->Fill(patTau->pt());
      hTauJetEtaIsoPassed_->Fill(patTau->eta());
      hTauJetPhiIsoPassed_->Fill(patTau->phi());
    }

//... for Energy, Pt, Eta, Phi of tau-jets passing the discriminatorByTaNC ("Tau Neural Classifier") selection
    if ( discrByTaNC > 0.5 ) {
      hTauJetEnergyTaNCpassed_->Fill(patTau->energy());
      hTauJetPtTaNCpassed_->Fill(patTau->pt());
      hTauJetEtaTaNCpassed_->Fill(patTau->eta());
      hTauJetPhiTaNCpassed_->Fill(patTau->phi());
    }
  }
}

void PatTauAnalyzer::endJob()
{
//--- nothing to be done yet...
}

#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(PatTauAnalyzer);

