// -*- C++ -*-
//
// Package:    L1Trigger/L1TNtuples
// Class:      L1JetRecoTreeProducer
//
/**\class L1JetRecoTreeProducer L1JetRecoTreeProducer.cc L1Trigger/L1TNtuples/src/L1JetRecoTreeProducer.cc

 Description: Produces tree containing reco quantities


*/

// system include files
#include <memory>

// framework
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/ESHandle.h"

// cond formats
#include "JetMETCorrections/JetCorrector/interface/JetCorrector.h"
#include "CondFormats/EcalObjects/interface/EcalChannelStatus.h"
#include "CondFormats/DataRecord/interface/EcalChannelStatusRcd.h"

// data formats
#include "DataFormats/JetReco/interface/CaloJetCollection.h"
#include "DataFormats/JetReco/interface/PFJetCollection.h"
#include "DataFormats/JetReco/interface/JetID.h"
#include "DataFormats/METReco/interface/PFMETCollection.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "DataFormats/METReco/interface/PFMET.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/METReco/interface/CaloMETCollection.h"
#include "DataFormats/METReco/interface/CaloMET.h"
#include "DataFormats/PatCandidates/interface/Jet.h"

// ROOT output stuff
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "TH1.h"
#include "TTree.h"
#include "TF1.h"
#include <TVector2.h>

//local  data formats
#include "L1Trigger/L1TNtuples/interface/L1AnalysisRecoJetDataFormat.h"
#include "L1Trigger/L1TNtuples/interface/L1AnalysisRecoMetDataFormat.h"

//
// class declaration
//

class L1JetRecoTreeProducer : public edm::one::EDAnalyzer<edm::one::SharedResources> {
public:
  explicit L1JetRecoTreeProducer(const edm::ParameterSet&);
  ~L1JetRecoTreeProducer() override;

private:
  void beginJob(void) override;
  void analyze(const edm::Event&, const edm::EventSetup&) override;
  void endJob() override;

  void doPFJets(edm::Handle<reco::PFJetCollection> pfJets);
  void doPFJetCorr(edm::Handle<reco::PFJetCollection> pfJets, edm::Handle<reco::JetCorrector> pfJetCorr);
  void doPUPPIJets(edm::Handle<reco::PFJetCollection> puppiJets);
  void doCorrPUPPIJets(edm::Handle<std::vector<pat::Jet> > corrPuppiJets);
  void doCaloJets(edm::Handle<reco::CaloJetCollection> caloJets);
  void doCaloJetCorr(edm::Handle<reco::CaloJetCollection> caloJets, edm::Handle<reco::JetCorrector> caloJetCorr);
  void doCaloMet(edm::Handle<reco::CaloMETCollection> caloMet);
  void doCaloMetBE(edm::Handle<reco::CaloMETCollection> caloMetBE);

  void doPFMet(edm::Handle<reco::PFMETCollection> pfMet);
  void doPFMetNoMu(edm::Handle<reco::PFMETCollection> pfMet, edm::Handle<reco::MuonCollection>);
  void doPUPPIMetNoMu(edm::Handle<reco::PFMETCollection> puppiMet, edm::Handle<reco::MuonCollection>);

  void doZPt(edm::Handle<reco::MuonCollection> muons, edm::Handle<std::vector<pat::Jet> > corrPuppiJets);

  bool pfJetID(const reco::PFJet& jet);
  bool puppiJetID(const pat::Jet& jet);
  bool caloJetID(const reco::CaloJet& jet);

public:
  L1Analysis::L1AnalysisRecoJetDataFormat* jet_data;
  L1Analysis::L1AnalysisRecoMetDataFormat* met_data;

private:
  // tree
  TTree* tree_;

  // EDM input tags
  edm::EDGetTokenT<reco::PFJetCollection> pfJetToken_;
  edm::EDGetTokenT<reco::PFJetCollection> puppiJetToken_;
  edm::EDGetTokenT<std::vector<pat::Jet> > corrPuppiJetToken_;
  edm::EDGetTokenT<reco::CaloJetCollection> caloJetToken_;
  edm::EDGetTokenT<edm::ValueMap<reco::JetID> > caloJetIDToken_;
  edm::EDGetTokenT<reco::JetCorrector> pfJECToken_;
  edm::EDGetTokenT<reco::JetCorrector> caloJECToken_;

  edm::EDGetTokenT<reco::PFMETCollection> pfMetToken_;
  edm::EDGetTokenT<reco::PFMETCollection> puppiMetToken_;
  edm::EDGetTokenT<reco::CaloMETCollection> caloMetToken_;
  edm::EDGetTokenT<reco::CaloMETCollection> caloMetBEToken_;

  edm::EDGetTokenT<reco::MuonCollection> muonToken_;

  // debug stuff
  bool pfJetsMissing_;
  bool puppiJetsMissing_;
  bool corrPuppiJetsMissing_;
  double jetptThreshold_;
  double jetetaMax_;
  unsigned int maxCl_;
  unsigned int maxJet_;
  unsigned int maxVtx_;
  unsigned int maxTrk_;

  bool pfJetCorrMissing_;
  bool caloJetCorrMissing_;
  bool caloJetsMissing_;
  bool caloJetIDMissing_;
  bool pfMetMissing_;
  bool puppiMetMissing_;
  bool caloMetMissing_;
  bool caloMetBEMissing_;
  bool muonsMissing_;
};

L1JetRecoTreeProducer::L1JetRecoTreeProducer(const edm::ParameterSet& iConfig)
    : pfJetsMissing_(false),
      puppiJetsMissing_(false),
      corrPuppiJetsMissing_(false),
      pfJetCorrMissing_(false),
      caloJetCorrMissing_(false),
      caloJetsMissing_(false),
      caloJetIDMissing_(false),
      pfMetMissing_(false),
      puppiMetMissing_(false),
      caloMetMissing_(false),
      caloMetBEMissing_(false),
      muonsMissing_(false) {
  caloJetToken_ =
      consumes<reco::CaloJetCollection>(iConfig.getUntrackedParameter("caloJetToken", edm::InputTag("ak4CaloJets")));
  pfJetToken_ =
      consumes<reco::PFJetCollection>(iConfig.getUntrackedParameter("pfJetToken", edm::InputTag("ak4PFJetsCHS")));
  puppiJetToken_ = consumes<reco::PFJetCollection>(iConfig.getParameter<edm::InputTag>("puppiJetToken"));
  corrPuppiJetToken_ = consumes<std::vector<pat::Jet> >(
      iConfig.getUntrackedParameter("corrPuppiJetToken", edm::InputTag("patJetsCorrectedPuppiJets")));
  caloJetIDToken_ =
      consumes<edm::ValueMap<reco::JetID> >(iConfig.getUntrackedParameter("caloJetIDToken", edm::InputTag("ak4JetID")));
  pfJECToken_ = consumes<reco::JetCorrector>(
      iConfig.getUntrackedParameter<edm::InputTag>("pfJECToken", edm::InputTag("ak4PFCHSL1FastL2L3ResidualCorrector")));
  caloJECToken_ = consumes<reco::JetCorrector>(iConfig.getUntrackedParameter<edm::InputTag>(
      "caloJECToken", edm::InputTag("ak4CaloL1FastL2L3ResidualCorrector")));

  pfMetToken_ = consumes<reco::PFMETCollection>(iConfig.getUntrackedParameter("pfMetToken", edm::InputTag("pfMetT1")));
  puppiMetToken_ =
      consumes<reco::PFMETCollection>(iConfig.getUntrackedParameter("puppiMetToken", edm::InputTag("pfMetPuppi")));
  caloMetToken_ =
      consumes<reco::CaloMETCollection>(iConfig.getUntrackedParameter("caloMetToken", edm::InputTag("caloMet")));
  caloMetBEToken_ =
      consumes<reco::CaloMETCollection>(iConfig.getUntrackedParameter("caloMetBEToken", edm::InputTag("caloMetBE")));

  muonToken_ = consumes<reco::MuonCollection>(iConfig.getUntrackedParameter("muonToken", edm::InputTag("muons")));

  usesResource(TFileService::kSharedResource);

  jetptThreshold_ = iConfig.getParameter<double>("jetptThreshold");
  jetetaMax_ = iConfig.getParameter<double>("jetetaMax");
  maxJet_ = iConfig.getParameter<unsigned int>("maxJet");

  jet_data = new L1Analysis::L1AnalysisRecoJetDataFormat();
  met_data = new L1Analysis::L1AnalysisRecoMetDataFormat();

  // set up output
  edm::Service<TFileService> fs_;
  tree_ = fs_->make<TTree>("JetRecoTree", "JetRecoTree");
  tree_->Branch("Jet", "L1Analysis::L1AnalysisRecoJetDataFormat", &jet_data, 32000, 3);
  tree_->Branch("Sums", "L1Analysis::L1AnalysisRecoMetDataFormat", &met_data, 32000, 3);
}

L1JetRecoTreeProducer::~L1JetRecoTreeProducer() {
  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)
}

//
// member functions
//

// ------------ method called to for each event  ------------
void L1JetRecoTreeProducer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  jet_data->Reset();
  met_data->Reset();

  // get jets
  edm::Handle<reco::PFJetCollection> pfJets;
  iEvent.getByToken(pfJetToken_, pfJets);

  // get puppi jets
  edm::Handle<reco::PFJetCollection> puppiJets;
  iEvent.getByToken(puppiJetToken_, puppiJets);

  // get corrected puppi jets
  edm::Handle<std::vector<pat::Jet> > corrPuppiJets;
  iEvent.getByToken(corrPuppiJetToken_, corrPuppiJets);

  // get calo jets
  edm::Handle<reco::CaloJetCollection> caloJets;
  iEvent.getByToken(caloJetToken_, caloJets);

  //get sums
  edm::Handle<reco::PFMETCollection> pfMet;
  iEvent.getByToken(pfMetToken_, pfMet);

  //get sums
  edm::Handle<reco::PFMETCollection> puppiMet;
  iEvent.getByToken(puppiMetToken_, puppiMet);

  // get jet ID
  edm::Handle<edm::ValueMap<reco::JetID> > jetsID;
  iEvent.getByToken(caloJetIDToken_, jetsID);

  edm::Handle<reco::JetCorrector> pfJetCorr;
  iEvent.getByToken(pfJECToken_, pfJetCorr);

  edm::Handle<reco::JetCorrector> caloJetCorr;
  iEvent.getByToken(caloJECToken_, caloJetCorr);

  edm::Handle<reco::CaloMETCollection> caloMet;
  iEvent.getByToken(caloMetToken_, caloMet);

  edm::Handle<reco::CaloMETCollection> caloMetBE;
  iEvent.getByToken(caloMetBEToken_, caloMetBE);

  // get muons
  edm::Handle<reco::MuonCollection> muons;
  iEvent.getByToken(muonToken_, muons);

  if (pfJets.isValid()) {
    jet_data->nJets = 0;

    doPFJets(pfJets);

  } else {
    if (!pfJetsMissing_) {
      edm::LogWarning("MissingProduct") << "PFJets not found.  Branch will not be filled" << std::endl;
    }
    pfJetsMissing_ = true;
  }

  if (pfJetCorr.isValid()) {
    doPFJetCorr(pfJets, pfJetCorr);

  } else {
    if (!pfJetCorrMissing_) {
      edm::LogWarning("MissingProduct") << "PF Jet Corrector not found.  Branch will not be filled" << std::endl;
    }
    pfJetCorrMissing_ = true;
  }

  if (puppiJets.isValid()) {
    jet_data->puppi_nUncorrJets = 0;

    doPUPPIJets(puppiJets);

  } else {
    if (!puppiJetsMissing_) {
      edm::LogWarning("MissingProduct") << "PUPPIJets not found.  Branch will not be filled" << std::endl;
    }
    puppiJetsMissing_ = true;
  }

  if (corrPuppiJets.isValid()) {
    jet_data->puppi_nJets = 0;

    doCorrPUPPIJets(corrPuppiJets);

  } else {
    if (!corrPuppiJetsMissing_) {
      edm::LogWarning("MissingProduct") << "Corrected PUPPIJets not found.  Branch will not be filled" << std::endl;
    }
    corrPuppiJetsMissing_ = true;
  }

  if (caloJets.isValid()) {
    jet_data->nCaloJets = 0;

    doCaloJets(caloJets);

  } else {
    if (!caloJetsMissing_) {
      edm::LogWarning("MissingProduct") << "Calo Jets not found.  Branch will not be filled" << std::endl;
    }
    caloJetsMissing_ = true;
  }

  if (caloJetCorr.isValid()) {
    doCaloJetCorr(caloJets, caloJetCorr);

  } else {
    if (!caloJetCorrMissing_) {
      edm::LogWarning("MissingProduct") << "Calo Jet Corrector not found.  Branch will not be filled" << std::endl;
    }
    caloJetCorrMissing_ = true;
  }

  if (!jetsID.isValid()) {
    if (!caloJetIDMissing_) {
      edm::LogWarning("MissingProduct") << "Calo Jet ID not found.  Branch will not be filled" << std::endl;
    }
    caloJetIDMissing_ = true;
  }

  if (pfMet.isValid()) {
    doPFMet(pfMet);

    if (muons.isValid()) {
      doPFMetNoMu(pfMet, muons);

    } else {
      if (!muonsMissing_) {
        edm::LogWarning("MissingProduct") << "Muons not found.  PFMetNoMu branch will not be filled" << std::endl;
      }
      muonsMissing_ = true;
    }
  } else {
    if (!pfMetMissing_) {
      edm::LogWarning("MissingProduct") << "PFMet not found.  Branch will not be filled" << std::endl;
    }
    pfMetMissing_ = true;
  }

  if (puppiMet.isValid()) {
    if (muons.isValid()) {
      doPUPPIMetNoMu(puppiMet, muons);

    } else {
      if (!muonsMissing_) {
        edm::LogWarning("MissingProduct") << "Muons not found.  PUPPIMetNoMu branch will not be filled" << std::endl;
      }
      muonsMissing_ = true;
    }
  } else {
    if (!puppiMetMissing_) {
      edm::LogWarning("MissingProduct") << "PUPPIMet not found.  Branch will not be filled" << std::endl;
    }
    puppiMetMissing_ = true;
  }

  if (caloMet.isValid()) {
    doCaloMet(caloMet);

  } else {
    if (!caloMetMissing_) {
      edm::LogWarning("MissingProduct") << "CaloMet not found. Branch will not be filled" << std::endl;
    }
    caloMetMissing_ = true;
  }

  if (caloMetBE.isValid()) {
    doCaloMetBE(caloMetBE);

  } else {
    if (!caloMetBEMissing_) {
      edm::LogWarning("MissingProduct") << "CaloMetBE not found. Branch will not be filled" << std::endl;
    }
    caloMetBEMissing_ = true;
  }

  if (muons.isValid()) {
    if (puppiJets.isValid()) {
      doZPt(muons, corrPuppiJets);

    } else {
      if (!puppiJetsMissing_) {
        edm::LogWarning("MissingProduct") << "PUPPIJets not found.  Branch will not be filled" << std::endl;
      }
      puppiJetsMissing_ = true;
    }
  } else {
    if (!muonsMissing_) {
      edm::LogWarning("MissingProduct") << "Muons not found.  ZPt branch will not be filled" << std::endl;
    }
    muonsMissing_ = true;
  }

  tree_->Fill();
}

void L1JetRecoTreeProducer::doCaloJets(edm::Handle<reco::CaloJetCollection> caloJets) {
  for (auto it = caloJets->begin(); it != caloJets->end() && jet_data->nCaloJets < maxJet_; ++it) {
    if (!caloJetIDMissing_)
      if (!caloJetID(*it))
        continue;

    jet_data->caloEt.push_back(it->et());
    jet_data->caloEta.push_back(it->eta());
    jet_data->caloPhi.push_back(it->phi());
    jet_data->caloE.push_back(it->energy());

    jet_data->eEMF.push_back(it->emEnergyFraction());
    jet_data->eEmEB.push_back(it->emEnergyInEB());
    jet_data->eEmEE.push_back(it->emEnergyInEE());
    jet_data->eEmHF.push_back(it->emEnergyInHF());
    jet_data->eHadHB.push_back(it->hadEnergyInHB());
    jet_data->eHadHE.push_back(it->hadEnergyInHE());
    jet_data->eHadHO.push_back(it->hadEnergyInHO());
    jet_data->eHadHF.push_back(it->hadEnergyInHF());
    jet_data->eMaxEcalTow.push_back(it->maxEInEmTowers());
    jet_data->eMaxHcalTow.push_back(it->maxEInHadTowers());
    jet_data->towerArea.push_back(it->towersArea());
    jet_data->n60.push_back(it->n60());

    jet_data->nCaloJets++;
  }
}

void L1JetRecoTreeProducer::doPFJets(edm::Handle<reco::PFJetCollection> pfJets) {
  for (auto it = pfJets->begin(); it != pfJets->end() && jet_data->nJets < maxJet_; ++it) {
    if (!pfJetID(*it))
      continue;

    jet_data->et.push_back(it->et());
    jet_data->eta.push_back(it->eta());
    jet_data->phi.push_back(it->phi());
    jet_data->e.push_back(it->energy());

    jet_data->chef.push_back(it->chargedHadronEnergyFraction());
    jet_data->nhef.push_back(it->neutralHadronEnergyFraction());
    jet_data->pef.push_back(it->photonEnergyFraction());
    jet_data->eef.push_back(it->electronEnergyFraction());
    jet_data->mef.push_back(it->muonEnergyFraction());
    jet_data->hfhef.push_back(it->HFHadronEnergyFraction());
    jet_data->hfemef.push_back(it->HFEMEnergyFraction());
    jet_data->chMult.push_back(it->chargedHadronMultiplicity());
    jet_data->nhMult.push_back(it->neutralHadronMultiplicity());
    jet_data->phMult.push_back(it->photonMultiplicity());
    jet_data->elMult.push_back(it->electronMultiplicity());
    jet_data->muMult.push_back(it->muonMultiplicity());
    jet_data->hfhMult.push_back(it->HFHadronMultiplicity());
    jet_data->hfemMult.push_back(it->HFEMMultiplicity());

    jet_data->cemef.push_back(it->chargedEmEnergyFraction());
    jet_data->cmef.push_back(it->chargedMuEnergyFraction());
    jet_data->nemef.push_back(it->neutralEmEnergyFraction());
    jet_data->cMult.push_back(it->chargedMultiplicity());
    jet_data->nMult.push_back(it->neutralMultiplicity());

    jet_data->nJets++;
  }
}

void L1JetRecoTreeProducer::doPFJetCorr(edm::Handle<reco::PFJetCollection> pfJets,
                                        edm::Handle<reco::JetCorrector> pfJetCorr) {
  float corrFactor = 1.;
  unsigned int nJets = 0;

  float mHx = 0;
  float mHy = 0;

  met_data->Ht = 0;
  met_data->mHt = -999.;
  met_data->mHtPhi = -999.;

  for (auto it = pfJets->begin(); it != pfJets->end() && nJets < maxJet_; ++it) {
    if (!pfJetID(*it))
      continue;

    corrFactor = pfJetCorr.product()->correction(*it);

    jet_data->etCorr.push_back(it->et() * corrFactor);
    jet_data->corrFactor.push_back(corrFactor);

    nJets++;

    if (it->pt() * corrFactor > jetptThreshold_ && std::abs(it->eta()) < jetetaMax_) {
      mHx += -1. * it->px() * corrFactor;
      mHy += -1. * it->py() * corrFactor;
      met_data->Ht += it->pt() * corrFactor;
    }
  }

  TVector2 tv2 = TVector2(mHx, mHy);
  met_data->mHt = tv2.Mod();
  met_data->mHtPhi = tv2.Phi();
}

void L1JetRecoTreeProducer::doPUPPIJets(edm::Handle<reco::PFJetCollection> puppiJets) {
  for (auto it = puppiJets->begin(); it != puppiJets->end() && jet_data->puppi_nUncorrJets < maxJet_; ++it) {
    if (!puppiJetID(*it))
      continue;
    jet_data->puppi_et.push_back(it->et());
    jet_data->puppi_nUncorrJets++;
  }
}

void L1JetRecoTreeProducer::doCorrPUPPIJets(edm::Handle<std::vector<pat::Jet> > corrPuppiJets) {
  float mHx = 0;
  float mHy = 0;

  met_data->puppi_Ht = 0;
  met_data->puppi_mHt = -999.;
  met_data->puppi_mHtPhi = -999.;

  for (auto it = corrPuppiJets->begin(); it != corrPuppiJets->end() && jet_data->puppi_nJets < maxJet_; ++it) {
    if (!puppiJetID(*it))
      continue;

    jet_data->puppi_etCorr.push_back(it->et());
    jet_data->puppi_eta.push_back(it->eta());
    jet_data->puppi_phi.push_back(it->phi());
    jet_data->puppi_e.push_back(it->energy());

    jet_data->puppi_chef.push_back(it->chargedHadronEnergyFraction());
    jet_data->puppi_nhef.push_back(it->neutralHadronEnergyFraction());
    jet_data->puppi_pef.push_back(it->photonEnergyFraction());
    jet_data->puppi_eef.push_back(it->electronEnergyFraction());
    jet_data->puppi_mef.push_back(it->muonEnergyFraction());
    jet_data->puppi_hfhef.push_back(it->HFHadronEnergyFraction());
    jet_data->puppi_hfemef.push_back(it->HFEMEnergyFraction());
    jet_data->puppi_chMult.push_back(it->chargedHadronMultiplicity());
    jet_data->puppi_nhMult.push_back(it->neutralHadronMultiplicity());
    jet_data->puppi_phMult.push_back(it->photonMultiplicity());
    jet_data->puppi_elMult.push_back(it->electronMultiplicity());
    jet_data->puppi_muMult.push_back(it->muonMultiplicity());
    jet_data->puppi_hfhMult.push_back(it->HFHadronMultiplicity());
    jet_data->puppi_hfemMult.push_back(it->HFEMMultiplicity());

    jet_data->puppi_cemef.push_back(it->chargedEmEnergyFraction());
    jet_data->puppi_cmef.push_back(it->chargedMuEnergyFraction());
    jet_data->puppi_nemef.push_back(it->neutralEmEnergyFraction());
    jet_data->puppi_cMult.push_back(it->chargedMultiplicity());
    jet_data->puppi_nMult.push_back(it->neutralMultiplicity());

    jet_data->puppi_nJets++;

    if (it->pt() > jetptThreshold_ && std::abs(it->eta()) < jetetaMax_) {
      mHx += -1. * it->px();
      mHy += -1. * it->py();
      met_data->puppi_Ht += it->pt();
    }
  }

  TVector2 tv2 = TVector2(mHx, mHy);
  met_data->puppi_mHt = tv2.Mod();
  met_data->puppi_mHtPhi = tv2.Phi();
}

void L1JetRecoTreeProducer::doCaloJetCorr(edm::Handle<reco::CaloJetCollection> caloJets,
                                          edm::Handle<reco::JetCorrector> caloJetCorr) {
  float caloCorrFactor = 1.;
  unsigned int nCaloJets = 0;

  met_data->caloHt = 0;

  for (auto it = caloJets->begin(); it != caloJets->end() && nCaloJets < maxJet_; ++it) {
    if (!caloJetIDMissing_)
      if (!caloJetID(*it))
        continue;

    caloCorrFactor = caloJetCorr.product()->correction(*it);

    jet_data->caloEtCorr.push_back(it->et() * caloCorrFactor);
    jet_data->caloCorrFactor.push_back(caloCorrFactor);

    nCaloJets++;

    if (it->pt() * caloCorrFactor > jetptThreshold_ && std::abs(it->eta()) < jetetaMax_) {
      met_data->caloHt += it->pt() * caloCorrFactor;
    }
  }
}

void L1JetRecoTreeProducer::doPFMet(edm::Handle<reco::PFMETCollection> pfMet) {
  const reco::PFMETCollection* metCol = pfMet.product();
  const reco::PFMET theMet = metCol->front();

  met_data->met = theMet.et();
  met_data->metPhi = theMet.phi();
  met_data->sumEt = theMet.sumEt();
  met_data->metPx = theMet.px();
  met_data->metPy = theMet.py();
}

void L1JetRecoTreeProducer::doPFMetNoMu(edm::Handle<reco::PFMETCollection> pfMet,
                                        edm::Handle<reco::MuonCollection> muons) {
  const reco::PFMETCollection* metCol = pfMet.product();
  const reco::PFMET theMet = metCol->front();
  reco::PFMET thePFMetNoMu = metCol->front();

  double pfMetNoMuPx = theMet.px();
  double pfMetNoMuPy = theMet.py();

  double muPx(0.), muPy(0.);

  for (auto it = muons->begin(); it != muons->end(); ++it) {
    if (it->isPFMuon()) {
      muPx += it->px();
      muPy += it->py();
    }
  }

  pfMetNoMuPx += muPx;
  pfMetNoMuPy += muPy;

  math::XYZTLorentzVector pfMetNoMuP4(pfMetNoMuPx, pfMetNoMuPy, 0, hypot(pfMetNoMuPx, pfMetNoMuPy));

  thePFMetNoMu.setP4(pfMetNoMuP4);

  met_data->pfMetNoMu = thePFMetNoMu.et();
  met_data->pfMetNoMuPhi = thePFMetNoMu.phi();
  met_data->pfMetNoMuPx = thePFMetNoMu.px();
  met_data->pfMetNoMuPy = thePFMetNoMu.py();
}

void L1JetRecoTreeProducer::doPUPPIMetNoMu(edm::Handle<reco::PFMETCollection> puppiMet,
                                           edm::Handle<reco::MuonCollection> muons) {
  const reco::PFMETCollection* metCol = puppiMet.product();
  const reco::PFMET theMet = metCol->front();
  reco::PFMET thePUPPIMetNoMu = metCol->front();

  double puppiMetNoMuPx = theMet.px();
  double puppiMetNoMuPy = theMet.py();

  double muPx(0.), muPy(0.);

  for (auto it = muons->begin(); it != muons->end(); ++it) {
    if (it->isPFMuon()) {
      muPx += it->px();
      muPy += it->py();
    }
  }

  puppiMetNoMuPx += muPx;
  puppiMetNoMuPy += muPy;

  math::XYZTLorentzVector puppiMetNoMuP4(puppiMetNoMuPx, puppiMetNoMuPy, 0, hypot(puppiMetNoMuPx, puppiMetNoMuPy));

  thePUPPIMetNoMu.setP4(puppiMetNoMuP4);

  met_data->puppi_metNoMu = thePUPPIMetNoMu.et();
  met_data->puppi_metNoMuPhi = thePUPPIMetNoMu.phi();
  met_data->puppi_metNoMuPx = thePUPPIMetNoMu.px();
  met_data->puppi_metNoMuPy = thePUPPIMetNoMu.py();
}

void L1JetRecoTreeProducer::doCaloMet(edm::Handle<reco::CaloMETCollection> caloMet) {
  const reco::CaloMETCollection* metCol = caloMet.product();
  const reco::CaloMET theMet = metCol->front();

  met_data->caloMet = theMet.et();
  met_data->caloMetPhi = theMet.phi();
  met_data->caloSumEt = theMet.sumEt();
}

void L1JetRecoTreeProducer::doCaloMetBE(edm::Handle<reco::CaloMETCollection> caloMetBE) {
  const reco::CaloMETCollection* metCol = caloMetBE.product();
  const reco::CaloMET theMet = metCol->front();

  met_data->caloMetBE = theMet.et();
  met_data->caloMetPhiBE = theMet.phi();
  met_data->caloSumEtBE = theMet.sumEt();
}

void L1JetRecoTreeProducer::doZPt(edm::Handle<reco::MuonCollection> muons,
                                  edm::Handle<std::vector<pat::Jet> > corrPuppiJets) {
  bool passPuppiJetPtCut = false;

  for (auto it = corrPuppiJets->begin(); it != corrPuppiJets->end(); ++it) {
    if (!puppiJetID(*it))
      continue;
    if (it->muonEnergyFraction() > 0.5 || it->chargedEmEnergyFraction() > 0.5)
      continue;
    if (it->pt() > 30)
      passPuppiJetPtCut = true;
  }

  if (!passPuppiJetPtCut) {
    met_data->zPt = -999;
    return;
  }

  reco::Muon muon1;
  reco::Muon muon2;

  float zMass = 91.2;
  float diMuMass = 0;
  float closestDiff = 999.;
  bool found2PFMuons = false;

  for (auto it1 = muons->begin(); it1 != muons->end(); ++it1) {
    if (!it1->isPFMuon())
      continue;
    for (auto it2 = std::next(it1); it2 != muons->end(); ++it2) {
      if (!it2->isPFMuon())
        continue;
      if (it1->charge() != (-1 * it2->charge()))
        continue;

      found2PFMuons = true;
      diMuMass = (it1->p4() + it2->p4()).M();
      float diff = abs(diMuMass - zMass);
      if (diff < closestDiff) {
        closestDiff = diff;
        muon1 = *it1;
        muon2 = *it2;
      }
    }
  }

  diMuMass = (muon1.p4() + muon2.p4()).M();
  if (abs(diMuMass - zMass) > 30 || !found2PFMuons) {
    met_data->zPt = -999;
    return;
  }

  float zPt = (muon1.p4() + muon2.p4()).pt();
  met_data->zPt = zPt;
}

bool L1JetRecoTreeProducer::pfJetID(const reco::PFJet& jet) {
  bool tmp = true;
  if (std::abs(jet.eta()) <= 2.6) {
    tmp &= jet.neutralHadronEnergyFraction() < 0.9;
    tmp &= jet.neutralEmEnergyFraction() < 0.9;
    tmp &= (jet.chargedMultiplicity() + jet.neutralMultiplicity()) > 1;
    tmp &= jet.muonEnergyFraction() < 0.8;
    tmp &= jet.chargedHadronEnergyFraction() > 0.01;
    tmp &= jet.chargedMultiplicity() > 0;
    tmp &= jet.chargedEmEnergyFraction() < 0.8;
  }
  if (std::abs(jet.eta()) > 2.6 && std::abs(jet.eta()) <= 2.7) {
    tmp &= jet.neutralHadronEnergyFraction() < 0.9;
    tmp &= jet.neutralEmEnergyFraction() < 0.99;
    tmp &= jet.muonEnergyFraction() < 0.8;
    tmp &= jet.chargedMultiplicity() > 0;
    tmp &= jet.chargedEmEnergyFraction() < 0.8;
  }
  if (std::abs(jet.eta()) > 2.7 && std::abs(jet.eta()) < 3.0) {
    tmp &= jet.neutralEmEnergyFraction() < 0.99;
    tmp &= jet.neutralMultiplicity() > 1;
  }
  if (std::abs(jet.eta()) > 3.0) {
    tmp &= jet.neutralHadronEnergyFraction() > 0.2;
    tmp &= jet.neutralEmEnergyFraction() < 0.9;
    tmp &= jet.neutralMultiplicity() > 10;
  }

  // our custom selection
  //tmp &= jet.muonMultiplicity() == 0;
  //tmp &= jet.electronMultiplicity() == 0;

  return tmp;
}

//https://twiki.cern.ch/twiki/bin/view/CMS/JetID13p6TeV
bool L1JetRecoTreeProducer::puppiJetID(const pat::Jet& jet) {
  bool tmp = true;
  if (std::abs(jet.eta()) <= 2.6) {
    tmp &= jet.neutralHadronEnergyFraction() < 0.9;
    tmp &= jet.neutralEmEnergyFraction() < 0.9;
    tmp &= (jet.chargedMultiplicity() + jet.neutralMultiplicity()) > 1;
    tmp &= jet.muonEnergyFraction() < 0.8;
    tmp &= jet.chargedHadronEnergyFraction() > 0.01;
    tmp &= jet.chargedMultiplicity() > 0;
    tmp &= jet.chargedEmEnergyFraction() < 0.8;
  }
  if (std::abs(jet.eta()) > 2.6 && std::abs(jet.eta()) <= 2.7) {
    tmp &= jet.neutralHadronEnergyFraction() < 0.9;
    tmp &= jet.neutralEmEnergyFraction() < 0.99;
    tmp &= jet.muonEnergyFraction() < 0.8;
    tmp &= jet.chargedEmEnergyFraction() < 0.8;
  }
  if (std::abs(jet.eta()) > 2.7 && std::abs(jet.eta()) <= 3.0) {
    tmp &= jet.neutralHadronEnergyFraction() < 0.9999;
  }
  if (std::abs(jet.eta()) > 3.0) {
    tmp &= jet.neutralEmEnergyFraction() < 0.9;
    tmp &= jet.neutralMultiplicity() > 2;
  }

  return tmp;
}

bool L1JetRecoTreeProducer::caloJetID(const reco::CaloJet& jet) {
  bool tmp = true;

  return tmp;
}

// ------------ method called once each job just before starting event loop  ------------
void L1JetRecoTreeProducer::beginJob(void) {}

// ------------ method called once each job just after ending the event loop  ------------
void L1JetRecoTreeProducer::endJob() {}

//define this as a plug-in
DEFINE_FWK_MODULE(L1JetRecoTreeProducer);
