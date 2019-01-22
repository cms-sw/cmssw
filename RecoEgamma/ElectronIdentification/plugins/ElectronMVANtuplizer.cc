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
#include "RecoEgamma/EgammaTools/interface/MultiToken.h"


#include "SimDataFormats/PileupSummaryInfo/interface/PileupSummaryInfo.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"

#include <TTree.h>
#include <TFile.h>
#include <Math/VectorUtil.h>

//
// class declaration
//

// If the analyzer does not use TFileService, please remove
// the template argument to the base class so the class inherits
// from  edm::one::EDAnalyzer<>
// This will improve performance in multithreaded jobs.
//

class ElectronMVANtuplizer : public edm::one::EDAnalyzer<edm::one::SharedResources>  {
   public:
      explicit ElectronMVANtuplizer(const edm::ParameterSet&);
      ~ElectronMVANtuplizer() override;

      static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);


   private:
      void analyze(const edm::Event&, const edm::EventSetup&) override;

      // method called once each job just before starting event loop
      void beginJob() override {};
      // method called once each job just after ending the event loop
      void endJob() override {};

      template<class T, class V>
      int matchToTruth(const T &el, const V &genParticles, int &genIdx);

      // ----------member data ---------------------------

      //global variables
      int nEvent_, nRun_, nLumi_;
      int genNpu_;
      int vtxN_;

      // electron variables
      float eleQ_;
      int ele3Q_;
      int matchedToGenEle_;
      int matchedGenIdx_;


      // gap variables
      bool eleIsEB_;
      bool eleIsEE_;
      bool eleIsEBEtaGap_;
      bool eleIsEBPhiGap_;
      bool eleIsEBEEGap_;
      bool eleIsEEDeeGap_;
      bool eleIsEERingGap_;

      int eleIndex_;

      // config
      const bool isMC_;
      const double deltaR_;
      const double ptThreshold_;

      // ID decisions objects
      const std::vector< std::string > eleMapTags_;
      std::vector< edm::EDGetTokenT< edm::ValueMap<bool> > > eleMapTokens_;
      const std::vector< std::string > eleMapBranchNames_;
      const size_t nEleMaps_;

      // MVA values and categories (optional)
      const std::vector< std::string > valMapTags_;
      std::vector< edm::EDGetTokenT<edm::ValueMap<float> > > valMapTokens_;
      const std::vector< std::string > valMapBranchNames_;
      const size_t nValMaps_;

      const std::vector< std::string > mvaCatTags_;
      std::vector< edm::EDGetTokenT<edm::ValueMap<int> > > mvaCatTokens_;
      const std::vector< std::string > mvaCatBranchNames_;
      const size_t nCats_;

      // Tokens for AOD and MiniAOD case
      const MultiTokenT<edm::View<reco::GsfElectron>>   src_;
      const MultiTokenT<std::vector<reco::Vertex>>      vertices_;
      const MultiTokenT<std::vector<PileupSummaryInfo>> pileup_;
      const MultiTokenT<edm::View<reco::GenParticle>>   genParticles_;

      // to hold ID decisions and categories
      std::vector<int> mvaPasses_;
      std::vector<float> mvaValues_;
      std::vector<int> mvaCats_;

      // To get the auxiliary MVA variables
      const MVAVariableHelper<reco::GsfElectron> variableHelper_;

      // other
      TTree* tree_;

      MVAVariableManager<reco::GsfElectron> mvaVarMngr_;
      const int nVars_;
      std::vector<float> vars_;

};

//
// constants, enums and typedefs
//

enum ElectronMatchType {
                        UNMATCHED,
                        TRUE_PROMPT_ELECTRON,
                        TRUE_ELECTRON_FROM_TAU,
                        TRUE_NON_PROMPT_ELECTRON,
                       }; // The last does not include tau parents

//
// static data member definitions
//

//
// constructors and destructor
//
ElectronMVANtuplizer::ElectronMVANtuplizer(const edm::ParameterSet& iConfig)
  : isMC_                  (iConfig.getParameter<bool>("isMC"))
  , deltaR_                (iConfig.getParameter<double>("deltaR"))
  , ptThreshold_           (iConfig.getParameter<double>("ptThreshold"))
  , eleMapTags_            (iConfig.getUntrackedParameter<std::vector<std::string>>("eleMVAs"))
  , eleMapBranchNames_     (iConfig.getUntrackedParameter<std::vector<std::string>>("eleMVALabels"))
  , nEleMaps_              (eleMapBranchNames_.size())
  , valMapTags_            (iConfig.getUntrackedParameter<std::vector<std::string>>("eleMVAValMaps"))
  , valMapBranchNames_     (iConfig.getUntrackedParameter<std::vector<std::string>>("eleMVAValMapLabels"))
  , nValMaps_              (valMapBranchNames_.size())
  , mvaCatTags_            (iConfig.getUntrackedParameter<std::vector<std::string>>("eleMVACats"))
  , mvaCatBranchNames_     (iConfig.getUntrackedParameter<std::vector<std::string>>("eleMVACatLabels"))
  , nCats_                 (mvaCatBranchNames_.size())
  , src_                   (consumesCollector(), iConfig, "src"         , "srcMiniAOD")
  , vertices_        (src_, consumesCollector(), iConfig, "vertices"    , "verticesMiniAOD")
  , pileup_          (src_, consumesCollector(), iConfig, "pileup"      , "pileupMiniAOD")
  , genParticles_    (src_, consumesCollector(), iConfig, "genParticles", "genParticlesMiniAOD")
  , mvaPasses_             (nEleMaps_)
  , mvaValues_             (nValMaps_)
  , mvaCats_               (nCats_)
  , variableHelper_        (consumesCollector())
  , mvaVarMngr_            (iConfig.getParameter<std::string>("variableDefinition"))
  , nVars_                 (mvaVarMngr_.getNVars())
  , vars_                  (nVars_)
{
    // eleMaps
    for (auto const& tag : eleMapTags_) {
        eleMapTokens_.push_back(consumes<edm::ValueMap<bool> >(edm::InputTag(tag)));
    }
    // valMaps
    for (auto const& tag : valMapTags_) {
        valMapTokens_.push_back(consumes<edm::ValueMap<float> >(edm::InputTag(tag)));
    }
    // categories
    for (auto const& tag : mvaCatTags_) {
        mvaCatTokens_.push_back(consumes<edm::ValueMap<int> >(edm::InputTag(tag)));
    }

   // Book tree
   usesResource(TFileService::kSharedResource);
   edm::Service<TFileService> fs ;
   tree_  = fs->make<TTree>("tree","tree");

   tree_->Branch("nEvent",  &nEvent_);
   tree_->Branch("nRun",    &nRun_);
   tree_->Branch("nLumi",   &nLumi_);
   if (isMC_) tree_->Branch("genNpu", &genNpu_);
   tree_->Branch("vtxN",   &vtxN_);

   tree_->Branch("ele_q",&eleQ_);
   tree_->Branch("ele_3q",&ele3Q_);

   if (isMC_) {
       tree_->Branch("matchedToGenEle",   &matchedToGenEle_);
   }

   for (int i = 0; i < nVars_; ++i) {
       tree_->Branch(mvaVarMngr_.getName(i).c_str(), &vars_[i]);
   }

   tree_->Branch("ele_isEB",&eleIsEB_);
   tree_->Branch("ele_isEE",&eleIsEE_);
   tree_->Branch("ele_isEBEtaGap",&eleIsEBEtaGap_);
   tree_->Branch("ele_isEBPhiGap",&eleIsEBPhiGap_);
   tree_->Branch("ele_isEBEEGap", &eleIsEBEEGap_);
   tree_->Branch("ele_isEEDeeGap",&eleIsEEDeeGap_);
   tree_->Branch("ele_isEERingGap",&eleIsEERingGap_);

   tree_->Branch("ele_index",&eleIndex_);

   // IDs
   for (size_t k = 0; k < nValMaps_; ++k) {
       tree_->Branch(valMapBranchNames_[k].c_str() ,  &mvaValues_[k]);
   }

   for (size_t k = 0; k < nEleMaps_; ++k) {
       tree_->Branch(eleMapBranchNames_[k].c_str() ,  &mvaPasses_[k]);
   }

   for (size_t k = 0; k < nCats_; ++k) {
       tree_->Branch(mvaCatBranchNames_[k].c_str() ,  &mvaCats_[k]);
   }
}


ElectronMVANtuplizer::~ElectronMVANtuplizer()
{

   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called for each event  ------------
void
ElectronMVANtuplizer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
    // Fill global event info
    nEvent_ = iEvent.id().event();
    nRun_   = iEvent.id().run();
    nLumi_  = iEvent.luminosityBlock();

    // Get Handles
    auto src          = src_.getValidHandle(iEvent);
    auto vertices     = vertices_.getValidHandle(iEvent);

    // Get MC only Handles, which are allowed to be non-valid
    auto genParticles = genParticles_.getHandle(iEvent);
    auto pileup       = pileup_.getHandle(iEvent);

    vtxN_ = vertices->size();

    // Fill with true number of pileup
    if(isMC_) {
       for(const auto& pu : *pileup)
       {
           int bx = pu.getBunchCrossing();
           if(bx == 0)
           {
               genNpu_ = pu.getPU_NumInteractions();
               break;
           }
       }
    }

    // Get MVA decisions
    edm::Handle<edm::ValueMap<bool> > decisions[nEleMaps_];
    for (size_t k = 0; k < nEleMaps_; ++k) {
        iEvent.getByToken(eleMapTokens_[k],decisions[k]);
    }

    // Get MVA values
    edm::Handle<edm::ValueMap<float> > values[nValMaps_];
    for (size_t k = 0; k < nValMaps_; ++k) {
        iEvent.getByToken(valMapTokens_[k],values[k]);
    }

    // Get MVA categories
    edm::Handle<edm::ValueMap<int> > mvaCats[nCats_];
    for (size_t k = 0; k < nCats_; ++k) {
        iEvent.getByToken(mvaCatTokens_[k],mvaCats[k]);
    }

    eleIndex_ = -1;
    for(size_t iEle = 0; iEle < src->size(); ++iEle) {

        ++eleIndex_;

        const auto ele =  src->ptrAt(iEle);

        eleQ_ = ele->charge();
        ele3Q_ = ele->chargeInfo().isGsfCtfScPixConsistent;

        if (ele->pt() < ptThreshold_) {
            continue;
        }

        for (int iVar = 0; iVar < nVars_; ++iVar) {
            std::vector<float> extraVariables = variableHelper_.getAuxVariables(ele, iEvent);
            vars_[iVar] = mvaVarMngr_.getValue(iVar, *ele, extraVariables);
        }

        if (isMC_) {
            matchedToGenEle_ = matchToTruth( ele, genParticles, matchedGenIdx_);
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
        for (size_t k = 0; k < nEleMaps_; ++k) {
          mvaPasses_[k] = static_cast<int>((*decisions[k])[ele]);
        }

        for (size_t k = 0; k < nValMaps_; ++k) {
          mvaValues_[k] = (*values[k])[ele];
        }

        for (size_t k = 0; k < nCats_; ++k) {
          mvaCats_[k] = (*mvaCats[k])[ele];
        }


        tree_->Fill();
    }

}

template<class T, class V>
int ElectronMVANtuplizer::matchToTruth(const T &el, const V &genParticles, int &genIdx){

  genIdx = -1;

  //
  // Explicit loop and geometric matching method (advised by Josh Bendavid)
  //

  // Find the closest status 1 gen electron to the reco electron
  double dR = 999;
  for(size_t i=0; i<genParticles->size();i++){
    const auto particle = genParticles->ptrAt(i);
    // Drop everything that is not electron or not status 1
    if( abs(particle->pdgId()) != 11 || particle->status() != 1 )
      continue;
    //
    double dRtmp = ROOT::Math::VectorUtil::DeltaR( el->p4(), particle->p4() );
    if( dRtmp < dR ){
      dR = dRtmp;
      genIdx = i;
    }
  }
  // See if the closest electron is close enough. If not, no match found.
  if( genIdx == -1 || dR >= deltaR_ ) {
    return UNMATCHED;
  }

  const auto closestElectron = genParticles->ptrAt(genIdx);

  if( closestElectron->fromHardProcessFinalState() )
    return TRUE_PROMPT_ELECTRON;

  if( closestElectron->isDirectHardProcessTauDecayProductFinalState() )
    return TRUE_ELECTRON_FROM_TAU;

  // What remains is true non-prompt electrons
  return TRUE_NON_PROMPT_ELECTRON;
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void
ElectronMVANtuplizer::fillDescriptions(edm::ConfigurationDescriptions& descriptions)
{
    edm::ParameterSetDescription desc;
    desc.add<edm::InputTag>("src",                 edm::InputTag("gedGsfElectrons"));
    desc.add<edm::InputTag>("vertices",            edm::InputTag("offlinePrimaryVertices"));
    desc.add<edm::InputTag>("pileup",              edm::InputTag("addPileupInfo"));
    desc.add<edm::InputTag>("genParticles",        edm::InputTag("genParticles"));
    desc.add<edm::InputTag>("srcMiniAOD",          edm::InputTag("slimmedElectrons"));
    desc.add<edm::InputTag>("verticesMiniAOD",     edm::InputTag("offlineSlimmedPrimaryVertices"));
    desc.add<edm::InputTag>("pileupMiniAOD",       edm::InputTag("slimmedAddPileupInfo"));
    desc.add<edm::InputTag>("genParticlesMiniAOD", edm::InputTag("prunedGenParticles"));
    desc.add<std::string>("variableDefinition");
    desc.add<bool>("isMC", true);
    desc.add<double>("deltaR", 0.1);
    desc.add<double>("ptThreshold", 5.0);
    desc.addUntracked<std::vector<std::string>>("eleMVAs", {});
    desc.addUntracked<std::vector<std::string>>("eleMVALabels", {});
    desc.addUntracked<std::vector<std::string>>("eleMVAValMaps", {});
    desc.addUntracked<std::vector<std::string>>("eleMVAValMapLabels", {});
    desc.addUntracked<std::vector<std::string>>("eleMVACats", {});
    desc.addUntracked<std::vector<std::string>>("eleMVACatLabels", {});
    descriptions.addDefault(desc);

}

//define this as a plug-in
DEFINE_FWK_MODULE(ElectronMVANtuplizer);
