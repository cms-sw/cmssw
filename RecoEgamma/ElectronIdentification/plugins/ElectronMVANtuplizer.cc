// -*- C++ -*-
//
// Package:    RecoEgamma/ElectronIdentification
// Class:      ElectronMVANtuplizer
//
/**\class ElectronMVANtuplizer ElectronMVANtuplizer.cc RecoEgamma/ElectronIdentification/plugins/ElectronMVANtuplizer.cc

 Description: Ntuplizer for training and testing electron MVA IDs.

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  Jonas REMBSER
//         Created:  Thu, 22 Mar 2018 14:54:24 GMT
//
//

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/PatCandidates/interface/Electron.h"
#include "RecoEgamma/EgammaTools/interface/MVAVariableManager.h"
#include "RecoEgamma/EgammaTools/interface/MVAVariableHelper.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "RecoEcal/EgammaCoreTools/interface/EcalClusterLazyTools.h"
#include "SimDataFormats/PileupSummaryInfo/interface/PileupSummaryInfo.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"

#include <TTree.h>
#include <TFile.h>
#include <Math/VectorUtil.h>

//
// class declaration
//

class ElectronMVANtuplizer : public edm::one::EDAnalyzer<edm::one::SharedResources> {
public:
  explicit ElectronMVANtuplizer(const edm::ParameterSet&);

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void analyze(const edm::Event&, const edm::EventSetup&) override;

  // method called once each job just before starting event loop
  void beginJob() override{};
  // method called once each job just after ending the event loop
  void endJob() override{};

  int matchToTruth(reco::GsfElectron const& electron, edm::View<reco::GenParticle> const& genParticles) const;

  // ----------member data ---------------------------

  //global variables
  int nEvent_;
  int nRun_;
  int nLumi_;
  int genNpu_;
  int vtxN_;

  // electron variables
  float eleQ_;
  int ele3Q_;
  int matchedToGenEle_;

  std::vector<float> energyMatrix_;

  // gap variables
  bool eleIsEB_;
  bool eleIsEE_;
  bool eleIsEBEtaGap_;
  bool eleIsEBPhiGap_;
  bool eleIsEBEEGap_;
  bool eleIsEEDeeGap_;
  bool eleIsEERingGap_;

  // config
  const bool isMC_;
  const double deltaR_;
  const double ptThreshold_;

  // ID decisions objects
  const std::vector<std::string> eleMapTags_;
  std::vector<edm::EDGetTokenT<edm::ValueMap<bool>>> eleMapTokens_;
  const std::vector<std::string> eleMapBranchNames_;
  const size_t nEleMaps_;

  // MVA values and categories (optional)
  const std::vector<std::string> valMapTags_;
  std::vector<edm::EDGetTokenT<edm::ValueMap<float>>> valMapTokens_;
  const std::vector<std::string> valMapBranchNames_;
  const size_t nValMaps_;

  const std::vector<std::string> mvaCatTags_;
  std::vector<edm::EDGetTokenT<edm::ValueMap<int>>> mvaCatTokens_;
  const std::vector<std::string> mvaCatBranchNames_;
  const size_t nCats_;

  // Tokens
  const edm::EDGetTokenT<edm::View<reco::GsfElectron>> src_;
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

  // other
  TTree* tree_;

  MVAVariableManager<reco::GsfElectron> mvaVarMngr_;
  const int nVars_;
  std::vector<float> vars_;

  const bool doEnergyMatrix_;
  const int energyMatrixSize_;
};

//
// constants, enums and typedefs
//

enum ElectronMatchType {
  UNMATCHED,
  TRUE_PROMPT_ELECTRON,
  TRUE_ELECTRON_FROM_TAU,
  TRUE_NON_PROMPT_ELECTRON,
};  // The last does not include tau parents

//
// constructors and destructor
//
ElectronMVANtuplizer::ElectronMVANtuplizer(const edm::ParameterSet& iConfig)
    : isMC_(iConfig.getParameter<bool>("isMC")),
      deltaR_(iConfig.getParameter<double>("deltaR")),
      ptThreshold_(iConfig.getParameter<double>("ptThreshold")),
      eleMapTags_(iConfig.getParameter<std::vector<std::string>>("eleMVAs")),
      eleMapBranchNames_(iConfig.getParameter<std::vector<std::string>>("eleMVALabels")),
      nEleMaps_(eleMapBranchNames_.size()),
      valMapTags_(iConfig.getParameter<std::vector<std::string>>("eleMVAValMaps")),
      valMapBranchNames_(iConfig.getParameter<std::vector<std::string>>("eleMVAValMapLabels")),
      nValMaps_(valMapBranchNames_.size()),
      mvaCatTags_(iConfig.getParameter<std::vector<std::string>>("eleMVACats")),
      mvaCatBranchNames_(iConfig.getParameter<std::vector<std::string>>("eleMVACatLabels")),
      nCats_(mvaCatBranchNames_.size()),
      src_(consumes<edm::View<reco::GsfElectron>>(iConfig.getParameter<edm::InputTag>("src"))),
      vertices_(consumes<std::vector<reco::Vertex>>(iConfig.getParameter<edm::InputTag>("vertices"))),
      pileup_(consumes<std::vector<PileupSummaryInfo>>(iConfig.getParameter<edm::InputTag>("pileup"))),
      genParticles_(consumes<edm::View<reco::GenParticle>>(iConfig.getParameter<edm::InputTag>("genParticles"))),
      ebRecHits_(consumes<EcalRecHitCollection>(iConfig.getParameter<edm::InputTag>("ebReducedRecHitCollection"))),
      eeRecHits_(consumes<EcalRecHitCollection>(iConfig.getParameter<edm::InputTag>("eeReducedRecHitCollection"))),
      ecalClusterToolsESGetTokens_{consumesCollector()},
      mvaPasses_(nEleMaps_),
      mvaValues_(nValMaps_),
      mvaCats_(nCats_),
      variableHelper_(consumesCollector()),
      mvaVarMngr_(iConfig.getParameter<std::string>("variableDefinition"), MVAVariableHelper::indexMap()),
      nVars_(mvaVarMngr_.getNVars()),
      vars_(nVars_),
      doEnergyMatrix_(iConfig.getParameter<bool>("doEnergyMatrix")),
      energyMatrixSize_(iConfig.getParameter<int>("energyMatrixSize")) {
  // eleMaps
  for (auto const& tag : eleMapTags_) {
    eleMapTokens_.push_back(consumes<edm::ValueMap<bool>>(edm::InputTag(tag)));
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
  if (isMC_)
    tree_->Branch("genNpu", &genNpu_);
  tree_->Branch("vtxN", &vtxN_);

  tree_->Branch("ele_q", &eleQ_);
  tree_->Branch("ele_3q", &ele3Q_);

  if (doEnergyMatrix_)
    tree_->Branch("energyMatrix", &energyMatrix_);

  if (isMC_)
    tree_->Branch("matchedToGenEle", &matchedToGenEle_);

  for (int i = 0; i < nVars_; ++i)
    tree_->Branch(mvaVarMngr_.getName(i).c_str(), &vars_[i]);

  tree_->Branch("ele_isEB", &eleIsEB_);
  tree_->Branch("ele_isEE", &eleIsEE_);
  tree_->Branch("ele_isEBEtaGap", &eleIsEBEtaGap_);
  tree_->Branch("ele_isEBPhiGap", &eleIsEBPhiGap_);
  tree_->Branch("ele_isEBEEGap", &eleIsEBEEGap_);
  tree_->Branch("ele_isEEDeeGap", &eleIsEEDeeGap_);
  tree_->Branch("ele_isEERingGap", &eleIsEERingGap_);

  // IDs
  for (size_t k = 0; k < nValMaps_; ++k) {
    tree_->Branch(valMapBranchNames_[k].c_str(), &mvaValues_[k]);
  }

  for (size_t k = 0; k < nEleMaps_; ++k) {
    tree_->Branch(eleMapBranchNames_[k].c_str(), &mvaPasses_[k]);
  }

  for (size_t k = 0; k < nCats_; ++k) {
    tree_->Branch(mvaCatBranchNames_[k].c_str(), &mvaCats_[k]);
  }
}

// ------------ method called for each event  ------------
void ElectronMVANtuplizer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  // Fill global event info
  nEvent_ = iEvent.id().event();
  nRun_ = iEvent.id().run();
  nLumi_ = iEvent.luminosityBlock();

  // Get Handles
  auto src = iEvent.getHandle(src_);
  auto vertices = iEvent.getHandle(vertices_);

  // initialize cluster tools
  std::unique_ptr<noZS::EcalClusterLazyTools> lazyTools;
  if (doEnergyMatrix_) {
    // Configure Lazy Tools, which will compute 5x5 quantities
    lazyTools = std::make_unique<noZS::EcalClusterLazyTools>(
        iEvent, ecalClusterToolsESGetTokens_.get(iSetup), ebRecHits_, eeRecHits_);
  }

  // Get MC only Handles, which are allowed to be non-valid
  auto genParticles = iEvent.getHandle(genParticles_);
  auto pileup = iEvent.getHandle(pileup_);

  vtxN_ = vertices->size();

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
  edm::Handle<edm::ValueMap<bool>> decisions[nEleMaps_];
  for (size_t k = 0; k < nEleMaps_; ++k) {
    iEvent.getByToken(eleMapTokens_[k], decisions[k]);
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

  for (auto const& ele : src->ptrs()) {
    if (ele->pt() < ptThreshold_)
      continue;

    // Fill the energy matrix around the seed
    if (doEnergyMatrix_) {
      const auto& seed = *(ele->superCluster()->seed());
      energyMatrix_ = lazyTools->energyMatrix(seed, energyMatrixSize_);
    }

    // Fill various tree variable
    eleQ_ = ele->charge();
    ele3Q_ = ele->chargeInfo().isGsfCtfScPixConsistent;

    for (int iVar = 0; iVar < nVars_; ++iVar) {
      vars_[iVar] = mvaVarMngr_.getValue(iVar, *ele, extraVariables);
    }

    if (isMC_) {
      matchedToGenEle_ = matchToTruth(*ele, *genParticles);
    }

    // gap variables
    eleIsEB_ = ele->isEB();
    eleIsEE_ = ele->isEE();
    eleIsEBEEGap_ = ele->isEBEEGap();
    eleIsEBEtaGap_ = ele->isEBEtaGap();
    eleIsEBPhiGap_ = ele->isEBPhiGap();
    eleIsEEDeeGap_ = ele->isEEDeeGap();
    eleIsEERingGap_ = ele->isEERingGap();

    //
    // Look up and save the ID decisions
    //
    for (size_t k = 0; k < nEleMaps_; ++k)
      mvaPasses_[k] = static_cast<int>((*decisions[k])[ele]);
    for (size_t k = 0; k < nValMaps_; ++k)
      mvaValues_[k] = (*values[k])[ele];
    for (size_t k = 0; k < nCats_; ++k)
      mvaCats_[k] = (*mvaCats[k])[ele];

    tree_->Fill();
  }
}

int ElectronMVANtuplizer::matchToTruth(reco::GsfElectron const& electron,
                                       edm::View<reco::GenParticle> const& genParticles) const {
  //
  // Explicit loop and geometric matching method (advised by Josh Bendavid)
  //

  // Find the closest status 1 gen electron to the reco electron
  double dR = 999;
  reco::GenParticle const* closestElectron = nullptr;
  for (auto const& particle : genParticles) {
    // Drop everything that is not electron or not status 1
    if (std::abs(particle.pdgId()) != 11 || particle.status() != 1)
      continue;
    //
    double dRtmp = ROOT::Math::VectorUtil::DeltaR(electron.p4(), particle.p4());
    if (dRtmp < dR) {
      dR = dRtmp;
      closestElectron = &particle;
    }
  }
  // See if the closest electron is close enough. If not, no match found.
  if (closestElectron == nullptr || dR >= deltaR_)
    return UNMATCHED;

  if (closestElectron->fromHardProcessFinalState())
    return TRUE_PROMPT_ELECTRON;

  if (closestElectron->isDirectHardProcessTauDecayProductFinalState())
    return TRUE_ELECTRON_FROM_TAU;

  // What remains is true non-prompt electrons
  return TRUE_NON_PROMPT_ELECTRON;
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void ElectronMVANtuplizer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("src", edm::InputTag("slimmedElectrons"));
  desc.add<edm::InputTag>("vertices", edm::InputTag("offlineSlimmedPrimaryVertices"));
  desc.add<edm::InputTag>("pileup", edm::InputTag("slimmedAddPileupInfo"));
  desc.add<edm::InputTag>("genParticles", edm::InputTag("prunedGenParticles"));
  desc.add<edm::InputTag>("ebReducedRecHitCollection", edm::InputTag("reducedEgamma", "reducedEBRecHits"));
  desc.add<edm::InputTag>("eeReducedRecHitCollection", edm::InputTag("reducedEgamma", "reducedEERecHits"));
  desc.add<std::string>("variableDefinition");
  desc.add<bool>("doEnergyMatrix", false);
  desc.add<int>("energyMatrixSize", 2)->setComment("extension of crystals in each direction away from the seed");
  desc.add<bool>("isMC", true);
  desc.add<double>("deltaR", 0.1);
  desc.add<double>("ptThreshold", 5.0);
  desc.add<std::vector<std::string>>("eleMVAs", {});
  desc.add<std::vector<std::string>>("eleMVALabels", {});
  desc.add<std::vector<std::string>>("eleMVAValMaps", {});
  desc.add<std::vector<std::string>>("eleMVAValMapLabels", {});
  desc.add<std::vector<std::string>>("eleMVACats", {});
  desc.add<std::vector<std::string>>("eleMVACatLabels", {});
  descriptions.addDefault(desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(ElectronMVANtuplizer);
