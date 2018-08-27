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


// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"

#include "DataFormats/EgammaCandidates/interface/Photon.h"

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

class PhotonMVANtuplizer : public edm::one::EDAnalyzer<edm::one::SharedResources>  {

   public:
      explicit PhotonMVANtuplizer(const edm::ParameterSet&);
      ~PhotonMVANtuplizer() override;

      static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);


   private:
      void analyze(const edm::Event&, const edm::EventSetup&) override;

      // method called once each job just before starting event loop
      void beginJob() override {};
      // method called once each job just after ending the event loop
      void endJob() override {};

      // ----------member data ---------------------------

      // other
      TTree* tree_;

      std::vector<float> vars_;
      int nVars_;

      //global variables
      int nEvent_, nRun_, nLumi_;
      int genNpu_;
      int vtxN_;

      // to hold ID decisions and categories
      std::vector<int> mvaPasses_;
      std::vector<float> mvaValues_;
      std::vector<int> mvaCats_;

      // ID decisions objects
      const std::vector< std::string > phoMapTags_;
      std::vector< edm::EDGetTokenT< edm::ValueMap<bool> > > phoMapTokens_;
      const std::vector< std::string > phoMapBranchNames_;
      const size_t nPhoMaps_;

      // MVA values and categories (optional)
      const std::vector< std::string > valMapTags_;
      std::vector< edm::EDGetTokenT<edm::ValueMap<float> > > valMapTokens_;
      const std::vector< std::string > valMapBranchNames_;
      const size_t nValMaps_;

      const std::vector< std::string > mvaCatTags_;
      std::vector< edm::EDGetTokenT<edm::ValueMap<int> > > mvaCatTokens_;
      const std::vector< std::string > mvaCatBranchNames_;
      const size_t nCats_;

      // config
      const bool isMC_;
      const double ptThreshold_;

      // for AOD or MiniAOD case
      MultiTokenT<edm::View<reco::Photon>>        src_;
      MultiTokenT<std::vector<reco::Vertex>>      vertices_;
      MultiTokenT<std::vector<PileupSummaryInfo>> pileup_;

};

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
PhotonMVANtuplizer::PhotonMVANtuplizer(const edm::ParameterSet& iConfig)
 : phoMapTags_            (iConfig.getUntrackedParameter<std::vector<std::string>>("phoMVAs"))
 , phoMapBranchNames_     (iConfig.getUntrackedParameter<std::vector<std::string>>("phoMVALabels"))
 , nPhoMaps_              (phoMapBranchNames_.size())
 , valMapTags_            (iConfig.getUntrackedParameter<std::vector<std::string>>("phoMVAValMaps"))
 , valMapBranchNames_     (iConfig.getUntrackedParameter<std::vector<std::string>>("phoMVAValMapLabels"))
 , nValMaps_              (valMapBranchNames_.size())
 , mvaCatTags_            (iConfig.getUntrackedParameter<std::vector<std::string>>("phoMVACats"))
 , mvaCatBranchNames_     (iConfig.getUntrackedParameter<std::vector<std::string>>("phoMVACatLabels"))
 , nCats_                 (mvaCatBranchNames_.size())
 , isMC_                  (iConfig.getParameter<bool>("isMC"))
 , ptThreshold_           (iConfig.getParameter<double>("ptThreshold"))
 , src_                   (consumesCollector(), iConfig, "src"     , "srcMiniAOD")
 , vertices_        (src_, consumesCollector(), iConfig, "vertices", "verticesMiniAOD")
 , pileup_          (src_, consumesCollector(), iConfig, "pileup"  , "pileupMiniAOD")
{
    // phoMaps
    for (size_t k = 0; k < nPhoMaps_; ++k) {

        phoMapTokens_.push_back(consumes<edm::ValueMap<bool> >(edm::InputTag(phoMapTags_[k])));

        // Initialize vectors for holding ID decisions
        mvaPasses_.push_back(0);
    }

    // valMaps
    for (size_t k = 0; k < nValMaps_; ++k) {
        valMapTokens_.push_back(consumes<edm::ValueMap<float> >(edm::InputTag(valMapTags_[k])));

        // Initialize vectors for holding MVA values
        mvaValues_.push_back(0.0);
    }

    // categories
    for (size_t k = 0; k < nCats_; ++k) {
        mvaCatTokens_.push_back(consumes<edm::ValueMap<int> >(edm::InputTag(mvaCatTags_[k])));

        // Initialize vectors for holding MVA values
        mvaCats_.push_back(0);
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

   // Has to be in two different loops
   for (int i = 0; i < nVars_; ++i) {
       vars_.push_back(0.0);
   }

   // IDs
   for (size_t k = 0; k < nValMaps_; ++k) {
       tree_->Branch(valMapBranchNames_[k].c_str() ,  &mvaValues_[k]);
   }

   for (size_t k = 0; k < nPhoMaps_; ++k) {
       tree_->Branch(phoMapBranchNames_[k].c_str() ,  &mvaPasses_[k]);
   }

   for (size_t k = 0; k < nCats_; ++k) {
       tree_->Branch(mvaCatBranchNames_[k].c_str() ,  &mvaCats_[k]);
   }
}


PhotonMVANtuplizer::~PhotonMVANtuplizer()
{

   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called for each event  ------------
void
PhotonMVANtuplizer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
    // Fill global event info
    nEvent_ = iEvent.id().event();
    nRun_   = iEvent.id().run();
    nLumi_  = iEvent.luminosityBlock();

    // Get Handles
    auto src      = src_.getValidHandle(iEvent);
    auto vertices = vertices_.getValidHandle(iEvent);
    auto pileup   = pileup_.getValidHandle(iEvent);

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
    edm::Handle<edm::ValueMap<bool> > decisions[nPhoMaps_];
    for (size_t k = 0; k < nPhoMaps_; ++k) {
        iEvent.getByToken(phoMapTokens_[k],decisions[k]);
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

    int nPho = src->size();

    for(int iPho = 0; iPho < nPho; ++iPho) {

        const auto pho =  src->ptrAt(iPho);

        if (pho->pt() < ptThreshold_) {
            continue;
        }

        //
        // Look up and save the ID decisions
        //
        for (size_t k = 0; k < nPhoMaps_; ++k) {
          mvaPasses_[k] = (int)(*decisions[k])[pho];
        }

        for (size_t k = 0; k < nValMaps_; ++k) {
          mvaValues_[k] = (*values[k])[pho];
        }

        for (size_t k = 0; k < nCats_; ++k) {
          mvaCats_[k] = (*mvaCats[k])[pho];
        }


        tree_->Fill();
    }

}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void
PhotonMVANtuplizer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {

    edm::ParameterSetDescription desc;
    desc.add<edm::InputTag>("src");
    desc.add<edm::InputTag>("vertices");
    desc.add<edm::InputTag>("pileup");
    desc.add<edm::InputTag>("srcMiniAOD");
    desc.add<edm::InputTag>("verticesMiniAOD");
    desc.add<edm::InputTag>("pileupMiniAOD");
    desc.addUntracked<std::vector<std::string>>("phoMVAs");
    desc.addUntracked<std::vector<std::string>>("phoMVALabels");
    desc.addUntracked<std::vector<std::string>>("phoMVAValMaps");
    desc.addUntracked<std::vector<std::string>>("phoMVAValMapLabels");
    desc.addUntracked<std::vector<std::string>>("phoMVACats");
    desc.addUntracked<std::vector<std::string>>("phoMVACatLabels");
    desc.add<bool>("isMC");
    desc.add<double>("ptThreshold", 5.0);
    descriptions.addDefault(desc);

}

//define this as a plug-in
DEFINE_FWK_MODULE(PhotonMVANtuplizer);
