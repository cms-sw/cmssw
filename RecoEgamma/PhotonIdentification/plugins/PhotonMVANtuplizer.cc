// -*- C++ -*-
//
// Package:    RecoEgamma/PhotonIdentification
// Class:      PhotonMVANtuplizer
//
/**\class PhotonMVANtuplizer PhotonMVANtuplizer.cc RecoEgamma/PhotonIdentification/plugins/PhotonMVANtuplizer.cc

 Description: Ntuplizer to use for testing photon MVA IDs.

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  Jonas REMBSER
//         Created:  Thu, 22 Mar 2018 14:54:24 GMT
//
//

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "DataFormats/EgammaCandidates/interface/Photon.h"
#include "DataFormats/PatCandidates/interface/Photon.h"
#include "RecoEgamma/EgammaTools/interface/MVAVariableHelper.h"
#include "RecoEgamma/EgammaTools/interface/MVAVariableManager.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "SimDataFormats/PileupSummaryInfo/interface/PileupSummaryInfo.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "RecoEcal/EgammaCoreTools/interface/EcalClusterLazyTools.h"

#include <TTree.h>
#include <TFile.h>
#include <Math/VectorUtil.h>

//
// class declaration
//

class PhotonMVANtuplizer : public edm::one::EDAnalyzer<edm::one::SharedResources> {
public:
  explicit PhotonMVANtuplizer(const edm::ParameterSet&);

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void analyze(const edm::Event&, const edm::EventSetup&) override;

  // ----------member data ---------------------------

  // other
  TTree* tree_;

  // global variables
  int nEvent_, nRun_, nLumi_;
  int genNpu_;
  int vtxN_;

  // photon variables
  double pT_, eta_;
  std::vector<float> energyMatrix_;

  // photon genMatch variable
  int matchedToGenPh_;
  int matchedGenIdx_;

  // ID decisions objects
  const std::vector<std::string> phoMapTags_;
  std::vector<edm::EDGetTokenT<edm::ValueMap<bool>>> phoMapTokens_;
  const std::vector<std::string> phoMapBranchNames_;
  const size_t nPhoMaps_;

  // MVA values and categories (optional)
  const std::vector<std::string> valMapTags_;
  std::vector<edm::EDGetTokenT<edm::ValueMap<float>>> valMapTokens_;
  const std::vector<std::string> valMapBranchNames_;
  const size_t nValMaps_;

  const std::vector<std::string> mvaCatTags_;
  std::vector<edm::EDGetTokenT<edm::ValueMap<int>>> mvaCatTokens_;
  const std::vector<std::string> mvaCatBranchNames_;
  const size_t nCats_;

  // config
  const bool isMC_;
  const double ptThreshold_;
  const double deltaR_;

  // Tokens
  const edm::EDGetTokenT<edm::View<reco::Photon>> src_;
  const edm::EDGetTokenT<std::vector<reco::Vertex>> vertices_;
  const edm::EDGetTokenT<std::vector<PileupSummaryInfo>> pileup_;
  const edm::EDGetTokenT<edm::View<reco::GenParticle>> genParticles_;
  const edm::EDGetTokenT<EcalRecHitCollection> ebRecHits_;
  const edm::EDGetTokenT<EcalRecHitCollection> eeRecHits_;

  const EcalClusterLazyTools::ESGetTokens ecalClusterToolsESGetTokens_;

  // to hold ID decisions and categories
  std::vector<int> mvaPasses_;
  std::vector<float> mvaValues_;
  std::vector<int> mvaCats_;

  // To get the auxiliary MVA variables
  const MVAVariableHelper variableHelper_;

  // To manage the variables which are parsed from the text file
  MVAVariableManager<reco::Photon> mvaVarMngr_;

  const int nVars_;
  std::vector<float> vars_;

  const bool doEnergyMatrix_;
  const int energyMatrixSize_;
};

enum PhotonMatchType {
  FAKE_PHOTON,
  TRUE_PROMPT_PHOTON,
  TRUE_NON_PROMPT_PHOTON,
};

namespace {

  int matchToTruth(const reco::Photon& ph, const edm::View<reco::GenParticle>& genParticles, double deltaR) {
    // Find the closest status 1 gen photon to the reco photon
    double dR = 999;
    reco::GenParticle const* closestPhoton = &genParticles[0];
    for (auto& particle : genParticles) {
      // Drop everything that is not photon or not status 1
      if (abs(particle.pdgId()) != 22 || particle.status() != 1)
        continue;

      double dRtmp = ROOT::Math::VectorUtil::DeltaR(ph.p4(), particle.p4());
      if (dRtmp < dR) {
        dR = dRtmp;
        closestPhoton = &particle;
      }
    }
    // See if the closest photon (if it exists) is close enough.
    // If not, no match found.
    if (dR < deltaR) {
      if (closestPhoton->isPromptFinalState())
        return TRUE_PROMPT_PHOTON;
      else
        return TRUE_NON_PROMPT_PHOTON;
    }
    return FAKE_PHOTON;
  }

};  // namespace

// constructor
PhotonMVANtuplizer::PhotonMVANtuplizer(const edm::ParameterSet& iConfig)
    : phoMapTags_(iConfig.getParameter<std::vector<std::string>>("phoMVAs")),
      phoMapBranchNames_(iConfig.getParameter<std::vector<std::string>>("phoMVALabels")),
      nPhoMaps_(phoMapBranchNames_.size()),
      valMapTags_(iConfig.getParameter<std::vector<std::string>>("phoMVAValMaps")),
      valMapBranchNames_(iConfig.getParameter<std::vector<std::string>>("phoMVAValMapLabels")),
      nValMaps_(valMapBranchNames_.size()),
      mvaCatTags_(iConfig.getParameter<std::vector<std::string>>("phoMVACats")),
      mvaCatBranchNames_(iConfig.getParameter<std::vector<std::string>>("phoMVACatLabels")),
      nCats_(mvaCatBranchNames_.size()),
      isMC_(iConfig.getParameter<bool>("isMC")),
      ptThreshold_(iConfig.getParameter<double>("ptThreshold")),
      deltaR_(iConfig.getParameter<double>("deltaR")),
      src_(consumes<edm::View<reco::Photon>>(iConfig.getParameter<edm::InputTag>("src"))),
      vertices_(consumes<std::vector<reco::Vertex>>(iConfig.getParameter<edm::InputTag>("vertices"))),
      pileup_(consumes<std::vector<PileupSummaryInfo>>(iConfig.getParameter<edm::InputTag>("pileup"))),
      genParticles_(consumes<edm::View<reco::GenParticle>>(iConfig.getParameter<edm::InputTag>("genParticles"))),
      ebRecHits_(consumes<EcalRecHitCollection>(iConfig.getParameter<edm::InputTag>("ebReducedRecHitCollection"))),
      eeRecHits_(consumes<EcalRecHitCollection>(iConfig.getParameter<edm::InputTag>("eeReducedRecHitCollection"))),
      ecalClusterToolsESGetTokens_{consumesCollector()},
      mvaPasses_(nPhoMaps_),
      mvaValues_(nValMaps_),
      mvaCats_(nCats_),
      variableHelper_(consumesCollector()),
      mvaVarMngr_(iConfig.getParameter<std::string>("variableDefinition"), MVAVariableHelper::indexMap()),
      nVars_(mvaVarMngr_.getNVars()),
      vars_(nVars_),
      doEnergyMatrix_(iConfig.getParameter<bool>("doEnergyMatrix")),
      energyMatrixSize_(iConfig.getParameter<int>("energyMatrixSize")) {
  // phoMaps
  for (auto const& tag : phoMapTags_) {
    phoMapTokens_.push_back(consumes<edm::ValueMap<bool>>(edm::InputTag(tag)));
  }
  // valMaps
  for (auto const& tag : valMapTags_) {
    valMapTokens_.push_back(consumes<edm::ValueMap<float>>(edm::InputTag(tag)));
  }
  // categories
  for (auto const& tag : mvaCatTags_) {
    mvaCatTokens_.push_back(consumes<edm::ValueMap<int>>(edm::InputTag(tag)));
  }

  // Book tree
  usesResource(TFileService::kSharedResource);
  edm::Service<TFileService> fs;
  tree_ = fs->make<TTree>("tree", "tree");

  tree_->Branch("nEvent", &nEvent_);
  tree_->Branch("nRun", &nRun_);
  tree_->Branch("nLumi", &nLumi_);
  if (isMC_) {
    tree_->Branch("genNpu", &genNpu_);
    tree_->Branch("matchedToGenPh", &matchedToGenPh_);
  }
  tree_->Branch("vtxN", &vtxN_);
  tree_->Branch("pT", &pT_);
  tree_->Branch("eta", &eta_);

  if (doEnergyMatrix_)
    tree_->Branch("energyMatrix", &energyMatrix_);

  for (int i = 0; i < nVars_; ++i) {
    tree_->Branch(mvaVarMngr_.getName(i).c_str(), &vars_[i]);
  }

  // IDs
  for (size_t k = 0; k < nValMaps_; ++k) {
    tree_->Branch(valMapBranchNames_[k].c_str(), &mvaValues_[k]);
  }

  for (size_t k = 0; k < nPhoMaps_; ++k) {
    tree_->Branch(phoMapBranchNames_[k].c_str(), &mvaPasses_[k]);
  }

  for (size_t k = 0; k < nCats_; ++k) {
    tree_->Branch(mvaCatBranchNames_[k].c_str(), &mvaCats_[k]);
  }
}

// ------------ method called for each event  ------------
void PhotonMVANtuplizer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  // Fill global event info
  nEvent_ = iEvent.id().event();
  nRun_ = iEvent.id().run();
  nLumi_ = iEvent.luminosityBlock();

  // Get Handles
  auto src = iEvent.getHandle(src_);
  auto vertices = iEvent.getHandle(vertices_);
  auto pileup = iEvent.getHandle(pileup_);
  auto genParticles = iEvent.getHandle(genParticles_);

  vtxN_ = vertices->size();

  // initialize cluster tools
  std::unique_ptr<noZS::EcalClusterLazyTools> lazyTools;
  if (doEnergyMatrix_) {
    // Configure Lazy Tools, which will compute 5x5 quantities
    lazyTools = std::make_unique<noZS::EcalClusterLazyTools>(
        iEvent, ecalClusterToolsESGetTokens_.get(iSetup), ebRecHits_, eeRecHits_);
  }

  // Fill with true number of pileup
  if (isMC_) {
    for (const auto& pu : *pileup) {
      int bx = pu.getBunchCrossing();
      if (bx == 0) {
        genNpu_ = pu.getPU_NumInteractions();
        break;
      }
    }
  }

  // Get MVA decisions
  edm::Handle<edm::ValueMap<bool>> decisions[nPhoMaps_];
  for (size_t k = 0; k < nPhoMaps_; ++k) {
    iEvent.getByToken(phoMapTokens_[k], decisions[k]);
  }

  // Get MVA values
  edm::Handle<edm::ValueMap<float>> values[nValMaps_];
  for (size_t k = 0; k < nValMaps_; ++k) {
    iEvent.getByToken(valMapTokens_[k], values[k]);
  }

  // Get MVA categories
  edm::Handle<edm::ValueMap<int>> mvaCats[nCats_];
  for (size_t k = 0; k < nCats_; ++k) {
    iEvent.getByToken(mvaCatTokens_[k], mvaCats[k]);
  }

  std::vector<float> extraVariables = variableHelper_.getAuxVariables(iEvent);

  for (auto const& pho : src->ptrs()) {
    if (pho->pt() < ptThreshold_)
      continue;

    pT_ = pho->pt();
    eta_ = pho->eta();

    // Fill the energy matrix around the seed
    if (doEnergyMatrix_) {
      const auto& seed = *(pho->superCluster()->seed());
      energyMatrix_ = lazyTools->energyMatrix(seed, energyMatrixSize_);
    }

    // variables from the text file
    for (int iVar = 0; iVar < nVars_; ++iVar) {
      vars_[iVar] = mvaVarMngr_.getValue(iVar, *pho, extraVariables);
    }

    if (isMC_)
      matchedToGenPh_ = matchToTruth(*pho, *genParticles, deltaR_);

    //
    // Look up and save the ID decisions
    //
    for (size_t k = 0; k < nPhoMaps_; ++k)
      mvaPasses_[k] = static_cast<int>((*decisions[k])[pho]);
    for (size_t k = 0; k < nValMaps_; ++k)
      mvaValues_[k] = (*values[k])[pho];
    for (size_t k = 0; k < nCats_; ++k)
      mvaCats_[k] = (*mvaCats[k])[pho];

    tree_->Fill();
  }
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void PhotonMVANtuplizer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("src", edm::InputTag("slimmedPhotons"));
  desc.add<edm::InputTag>("vertices", edm::InputTag("offlineSlimmedPrimaryVertices"));
  desc.add<edm::InputTag>("pileup", edm::InputTag("slimmedAddPileupInfo"));
  desc.add<edm::InputTag>("genParticles", edm::InputTag("prunedGenParticles"));
  desc.add<edm::InputTag>("ebReducedRecHitCollection", edm::InputTag("reducedEgamma", "reducedEBRecHits"));
  desc.add<edm::InputTag>("eeReducedRecHitCollection", edm::InputTag("reducedEgamma", "reducedEERecHits"));
  desc.add<std::vector<std::string>>("phoMVAs", {});
  desc.add<std::vector<std::string>>("phoMVALabels", {});
  desc.add<std::vector<std::string>>("phoMVAValMaps", {});
  desc.add<std::vector<std::string>>("phoMVAValMapLabels", {});
  desc.add<std::vector<std::string>>("phoMVACats", {});
  desc.add<std::vector<std::string>>("phoMVACatLabels", {});
  desc.add<bool>("doEnergyMatrix", false);
  desc.add<int>("energyMatrixSize", 2)->setComment("extension of crystals in each direction away from the seed");
  desc.add<bool>("isMC", true);
  desc.add<double>("ptThreshold", 15.0);
  desc.add<double>("deltaR", 0.1);
  desc.add<std::string>("variableDefinition");
  descriptions.addDefault(desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(PhotonMVANtuplizer);
