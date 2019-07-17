// -*- C++ -*-
//
// Package:    Giovanni/NTuplizer
// Class:      NTuplizer
// 
/**\class NTuplizer NTuplizer.cc Giovanni/NTuplizer/plugins/NTuplizer.cc

 Description: [one line class summary]

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  Giovanni Petrucciani
//         Created:  Thu, 01 Sep 2016 11:30:38 GMT
//
//

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/View.h"

#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "DataFormats/JetReco/interface/GenJet.h"
#include "DataFormats/JetReco/interface/PFJet.h"
#include "DataFormats/METReco/interface/CaloMET.h"
#include "DataFormats/METReco/interface/PFMET.h"
#include "DataFormats/METReco/interface/GenMET.h"

#include "DataFormats/Math/interface/deltaR.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "CommonTools/Utils/interface/StringCutObjectSelector.h"
#include "CommonTools/Utils/interface/StringObjectFunction.h"
#include "L1Trigger/Phase2L1ParticleFlow/interface/SimpleCalibrations.h"

#include <cstdint>
#include <TTree.h>
#include <TRandom3.h>

class L1TPFJetMetTreeProducer : public edm::one::EDAnalyzer<edm::one::SharedResources>  {
   public:
      explicit L1TPFJetMetTreeProducer(const edm::ParameterSet&);
      ~L1TPFJetMetTreeProducer();

   private:
      virtual void analyze(const edm::Event&, const edm::EventSetup&) override;

      struct JetInput {
         std::string name;
         edm::EDGetTokenT<edm::View<reco::LeafCandidate>> token;
         l1tpf::SimpleCorrEm jec;
         JetInput() {}
         JetInput(const std::string &n, const edm::EDGetTokenT<edm::View<reco::LeafCandidate>> &t, const l1tpf::SimpleCorrEm & c) : name(n), token(t), jec(c) {}
      };
      std::vector<JetInput> jets_;
      std::vector<std::pair<std::string,StringCutObjectSelector<reco::LeafCandidate>>> jetSels_;

      struct MetInput {
         std::string name;
         edm::EDGetTokenT<edm::View<reco::LeafCandidate>> token;
         MetInput() {}
         MetInput(const std::string &n, const edm::EDGetTokenT<edm::View<reco::LeafCandidate>> &t) : name(n), token(t) {}
      };
      std::vector<MetInput> mets_;

      struct JetCollectionBranch {
        std::string name;
	std::vector<float> ak4jet_pt_corr;
	std::vector<float> ak4jet_pt_raw;
	std::vector<float> ak4jet_eta;
	std::vector<float> ak4jet_phi;
	std::vector<float> ak4jet_mass_raw;
	std::vector<float> ak4jet_mass_corr;
	std::vector<float> ak4jet_e_raw;
	std::vector<float> ak4jet_e_corr;
	int ak4jet_N;
        JetCollectionBranch(const std::string &n) : name(n) {}
        void makeBranches(TTree *tree) {
            tree->Branch((name+"_N").c_str()        ,   &ak4jet_N        );
	    tree->Branch((name+"_pt_raw").c_str()   ,   &ak4jet_pt_raw   );
	    tree->Branch((name+"_pt_corr").c_str()  ,   &ak4jet_pt_corr  );
	    tree->Branch((name+"_eta").c_str()      ,   &ak4jet_eta      );
	    tree->Branch((name+"_phu").c_str()      ,   &ak4jet_phi      );
	    tree->Branch((name+"_mass_raw").c_str() ,   &ak4jet_mass_raw );
	    tree->Branch((name+"_mass_corr").c_str(),   &ak4jet_mass_corr);
	    tree->Branch((name+"_e_raw").c_str()    ,   &ak4jet_pt_raw   );
	    tree->Branch((name+"_e_corr").c_str()   ,   &ak4jet_pt_corr  );
        }
        void fill(const edm::View<reco::LeafCandidate> & jets, const  l1tpf::SimpleCorrEm & corr) {  
	    ak4jet_pt_corr.clear();
	    ak4jet_pt_raw.clear();
	    ak4jet_eta.clear();
	    ak4jet_phi.clear();
	    ak4jet_mass_raw.clear();
	    ak4jet_mass_corr.clear();
	    ak4jet_e_raw.clear();
	    ak4jet_e_corr.clear();
	    ak4jet_N = 0;
            reco::Particle::LorentzVector sum_raw, sum_corr;
            for (reco::LeafCandidate jet : jets) {
	        ak4jet_N++;
	        ak4jet_pt_raw.push_back(jet.pt());
		ak4jet_e_raw.push_back(jet.energy());
		ak4jet_mass_raw.push_back(jet.mass());
		ak4jet_eta.push_back(jet.eta());
		ak4jet_phi.push_back(jet.phi());
		float correction = corr(jet.pt(), std::abs(jet.eta()));
                jet.setP4(reco::Particle::PolarLorentzVector(correction, jet.eta(), jet.phi(), jet.mass()));
		correction = correction/jet.pt();
		ak4jet_pt_corr.push_back(jet.pt());
		ak4jet_mass_corr.push_back(correction*jet.mass());
		ak4jet_e_corr.push_back(correction*jet.energy());
            }
        }
      };
      std::vector<JetCollectionBranch> jetCollbranches_;
      
      struct JetBranch {
        std::string name;
        int   count_raw, count_corr;
        float ht_raw, ht_corr;
        float mht_raw, mht_corr;
        JetBranch(const std::string &n) : name(n) {}
        void makeBranches(TTree *tree) {
            tree->Branch((name+"_ht_raw").c_str()   ,   &ht_raw          ,   (name+"_ht_raw/F").c_str());
            tree->Branch((name+"_mht_raw").c_str()  ,   &mht_raw         ,   (name+"_mht_raw/F").c_str());
            tree->Branch((name+"_num_raw").c_str()  ,   &count_raw       ,   (name+"_num_raw/I").c_str());
            tree->Branch((name+"_ht_corr").c_str()  ,   &ht_corr         ,   (name+"_ht_corr/F").c_str());
            tree->Branch((name+"_mht_corr").c_str() ,   &mht_corr        ,   (name+"_mht_corr/F").c_str());
            tree->Branch((name+"_num_corr").c_str() ,   &count_corr      ,   (name+"_num_corr/I").c_str());
        }
        void fill(const edm::View<reco::LeafCandidate> & jets, const  l1tpf::SimpleCorrEm & corr, const StringCutObjectSelector<reco::LeafCandidate> & sel) {  
            count_raw = 0; ht_raw = 0;
            count_corr = 0; ht_corr = 0;
            reco::Particle::LorentzVector sum_raw, sum_corr;
            for (reco::LeafCandidate jet : jets) {
                if (sel(jet)) { count_raw++; ht_raw += std::min(jet.pt(),500.); sum_raw += jet.p4(); }
                jet.setP4(reco::Particle::PolarLorentzVector(corr(jet.pt(), std::abs(jet.eta())), jet.eta(), jet.phi(), jet.mass()));
                if (sel(jet)) { count_corr++; ht_corr += std::min(jet.pt(),500.); sum_corr += jet.p4(); }
            }
            mht_raw = sum_raw.Pt();
            mht_corr = sum_corr.Pt();
        }
      };
      std::vector<JetBranch> jetbranches_;

      struct MetBranch {
       std::string name;
       float met, met_phi;
       MetBranch(const std::string &n) : name(n) {}
       void makeBranches(TTree *tree) {
         tree->Branch((name).c_str(),   &met,   (name+"/F").c_str());
         tree->Branch((name+"_phi").c_str(), &met_phi, (name+"_phi/F").c_str());
       }
       void fill(const edm::View<reco::LeafCandidate> & recoMets) {  
         met = 0; met_phi = 0;
         for (reco::LeafCandidate recoMet : recoMets) { met = recoMet.et(); met_phi = recoMet.phi(); }
       }
      };
      std::vector<MetBranch> metbranches_;

      struct SpecialBranch {
        std::string name;
        edm::EDGetTokenT<edm::View<reco::Candidate>> token;
        StringCutObjectSelector<reco::Candidate> sel;
        StringObjectFunction<reco::Candidate> func;
        float val;
        SpecialBranch() : sel(""), func("1") {}
        SpecialBranch(const std::string &n, const edm::EDGetTokenT<edm::View<reco::Candidate>> &t, const std::string & cut, const std::string & expr) : name(n), token(t), sel(cut,true), func(expr,true) {}
        void makeBranches(TTree *tree) {
            tree->Branch((name).c_str(), &val, (name+"/F").c_str());
        }
        void fill(const edm::View<reco::Candidate> & view) {
            val = 0;
            for (const reco::Candidate & c : view) {
                if (sel(c)) { val += func(c); }
            } 
        }
      };
      std::vector<SpecialBranch> specials_;
       
      TTree *tree_;
      uint32_t run_, lumi_; uint64_t event_;      
};

L1TPFJetMetTreeProducer::L1TPFJetMetTreeProducer(const edm::ParameterSet& iConfig) 
{
    usesResource("TFileService");
    edm::Service<TFileService> fs;
    tree_ = fs->make<TTree>("tree","tree");
    tree_->Branch("run",  &run_,  "run/i");
    tree_->Branch("lumi", &lumi_, "lumi/i");
    tree_->Branch("event", &event_, "event/l");

    // jet branches
    edm::ParameterSet jets = iConfig.getParameter<edm::ParameterSet>("jets");
    edm::ParameterSet jecs = iConfig.getParameter<edm::ParameterSet>("jecs");
    for (const std::string & name : jets.getParameterNamesForType<edm::InputTag>()) {
        jets_.emplace_back(name, consumes<edm::View<reco::LeafCandidate>>(jets.getParameter<edm::InputTag>(name)), l1tpf::SimpleCorrEm(jecs, name));
    }
    edm::ParameterSet sels = iConfig.getParameter<edm::ParameterSet>("sels");
    for (const std::string & name : sels.getParameterNamesForType<std::string>()) {
        jetSels_.emplace_back(name, StringCutObjectSelector<reco::LeafCandidate>(sels.getParameter<std::string>(name)));
    }
    jetbranches_.reserve(jets_.size() * jetSels_.size());
    for (const auto & j : jets_) {
        for (const auto & s : jetSels_) {
            jetbranches_.emplace_back(j.name + s.first);
        }
    }
    for (auto & b : jetbranches_) b.makeBranches(tree_);

    jetCollbranches_.reserve(jets_.size());
    for (const auto & j : jets_) jetCollbranches_.emplace_back(j.name);
    for (auto & b : jetCollbranches_) b.makeBranches(tree_);
    
    //met branches
    edm::ParameterSet mets = iConfig.getParameter<edm::ParameterSet>("mets");
    for (const std::string & name : mets.getParameterNamesForType<edm::InputTag>()) {
        mets_.emplace_back(name, consumes<edm::View<reco::LeafCandidate> >(mets.getParameter<edm::InputTag>(name)));
    }
    metbranches_.reserve(mets_.size());
    for (const auto & m : mets_) metbranches_.emplace_back(m.name);
    for (auto & b : metbranches_) b.makeBranches(tree_);

    edm::ParameterSet specials = iConfig.getParameter<edm::ParameterSet>("specials");
    for (const std::string & name : specials.getParameterNamesForType<edm::ParameterSet>()) {
        edm::ParameterSet conf = specials.getParameter<edm::ParameterSet>(name);
        specials_.emplace_back(name, 
                               consumes<edm::View<reco::Candidate>>(conf.getParameter<edm::InputTag>("src")),
                               conf.getParameter<std::string>("cut"),
                               conf.getParameter<std::string>("expr"));
    }
    for (auto & b : specials_) b.makeBranches(tree_);
}

L1TPFJetMetTreeProducer::~L1TPFJetMetTreeProducer() { }

// ------------ method called for each event  ------------
void
L1TPFJetMetTreeProducer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
    run_  = iEvent.id().run();
    lumi_ = iEvent.id().luminosityBlock();
    event_ = iEvent.id().event();

    int ibranch = 0;
    for (const auto & j : jets_) {
        edm::Handle<edm::View<reco::LeafCandidate>> jets;
        iEvent.getByToken(j.token, jets);
        jetCollbranches_[ibranch++].fill(*jets, j.jec);
    }
        
    ibranch = 0;
    for (const auto & j : jets_) {
        edm::Handle<edm::View<reco::LeafCandidate>> jets;
        iEvent.getByToken(j.token, jets);
        for (const auto & s : jetSels_) {
            jetbranches_[ibranch++].fill(*jets, j.jec, s.second);
        } 
    }

    ibranch = 0;
    for (const auto & m : mets_) {
        edm::Handle<edm::View<reco::LeafCandidate>> mets;
        iEvent.getByToken(m.token, mets);
        metbranches_[ibranch++].fill(*mets);
    }


    
    for (SpecialBranch & s : specials_) {
        edm::Handle<edm::View<reco::Candidate>> specials;
        iEvent.getByToken(s.token, specials);
        s.fill(*specials);
    }
    tree_->Fill();
}

//define this as a plug-in
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(L1TPFJetMetTreeProducer);
