
// system include files
#include <memory>

// stl
#include <algorithm>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

// ana
#include "HeavyIonsAnalysis/JetAnalysis/interface/HiPFCandAnalyzer.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "RecoJets/JetAlgorithms/interface/JetAlgoHelper.h"
#include "RecoHI/HiJetAlgos/interface/UEParameters.h"
#include "FWCore/Common/interface/TriggerNames.h"
#include "DataFormats/Common/interface/TriggerResults.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/HepMCCandidate/interface/GenParticleFwd.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "DataFormats/Math/interface/LorentzVector.h"
#include "DataFormats/Math/interface/deltaR.h"
#include "DataFormats/Math/interface/deltaPhi.h"
#include "TMath.h"
#include "TStopwatch.h"

#include "DataFormats/HeavyIonEvent/interface/Centrality.h"
#include "SimDataFormats/HiGenData/interface/GenHIEvent.h"


using namespace std;
using namespace edm;
using namespace reco;

//
// constructors and destructor
//
HiPFCandAnalyzer::HiPFCandAnalyzer(const edm::ParameterSet& iConfig)
{
  // Event source
  // Event Info
  pfCandidateLabel_ = iConfig.getParameter<edm::InputTag>("pfCandidateLabel");
  pfCandidatePF_ = consumes<reco::PFCandidateCollection> (pfCandidateLabel_);
  pfCandidateView_ = consumes<reco::CandidateView> (pfCandidateLabel_);
  pfPtMin_ = iConfig.getParameter<double>("pfPtMin");
  genPtMin_ = iConfig.getParameter<double>("genPtMin");
  jetPtMin_ = iConfig.getParameter<double>("jetPtMin");

  etaBins_ = iConfig.getParameter<int>("etaBins");
  fourierOrder_ = iConfig.getParameter<int>("fourierOrder");

  doVS_ = iConfig.getUntrackedParameter<bool>("doVS",false);
  if(doVS_){
    edm::InputTag vsTag = iConfig.getParameter<edm::InputTag>("bkg");
    srcVorFloat_ = consumes<std::vector<float> >(vsTag);
    srcVorMap_ = consumes<reco::VoronoiMap>(vsTag);
  }


  // debug
  verbosity_ = iConfig.getUntrackedParameter<int>("verbosity", false);

  doJets_ = iConfig.getUntrackedParameter<bool>("doJets",false);
  if(doJets_){
    jetLabel_ = consumes<pat::JetCollection>(iConfig.getParameter<edm::InputTag>("jetLabel"));
  }
  doUEraw_ = iConfig.getUntrackedParameter<bool>("doUEraw",false);

  doMC_ = iConfig.getUntrackedParameter<bool>("doMC",false);

  if(doMC_){
    genLabel_ = consumes<reco::GenParticleCollection>(iConfig.getParameter<edm::InputTag>("genLabel"));
  }
  skipCharged_ = iConfig.getUntrackedParameter<bool>("skipCharged",false);




}


HiPFCandAnalyzer::~HiPFCandAnalyzer()
{

  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called to for each event  ------------
void
HiPFCandAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  pfEvt_.Clear();

  // Fill PF info

  edm::Handle<reco::PFCandidateCollection> pfCandidates;
  iEvent.getByToken(pfCandidatePF_,pfCandidates);
  iEvent.getByToken(pfCandidateView_,candidates_);
  const reco::PFCandidateCollection *pfCandidateColl = pfCandidates.product();
  if (doVS_) {
   iEvent.getByToken(srcVorMap_,backgrounds_);
   iEvent.getByToken(srcVorFloat_,vn_);
   UEParameters vnUE(vn_.product(),fourierOrder_,etaBins_);
   const std::vector<float>& vue = vnUE.get_raw();


   for(int ieta = 0; ieta < etaBins_; ++ieta){
     pfEvt_.sumpt[ieta] = vnUE.get_sum_pt(ieta);
     for(int ifour = 0; ifour < fourierOrder_; ++ifour){
       pfEvt_.vn[ifour * etaBins_ + ieta] = vnUE.get_vn(ifour,ieta);
       pfEvt_.psin[ifour * etaBins_ + ieta] = vnUE.get_psin(ifour,ieta);
     }
   }

   for(int iue = 0; iue < etaBins_*fourierOrder_*2*3; ++iue){
     pfEvt_.ueraw[iue] = vue[iue];
   }
  }

  for(unsigned icand=0;icand<pfCandidateColl->size(); icand++) {
    const reco::PFCandidate pfCandidate = pfCandidateColl->at(icand);
    reco::CandidateViewRef ref(candidates_,icand);

    double vsPtInitial=-999, vsPt=-999, vsArea = -999;

    if (doVS_) {
      const reco::VoronoiBackground& voronoi = (*backgrounds_)[ref];
      vsPt = voronoi.pt();
      vsPtInitial = voronoi.pt_subtracted();
      vsArea = voronoi.area();
    }

    double pt =  pfCandidate.pt();
    double energy = pfCandidate.energy();
    if(pt<=pfPtMin_) continue;

    int id = pfCandidate.particleId();
    if(skipCharged_ && (abs(id) == 1 || abs(id) == 3)) continue;

    pfEvt_.pfId_.push_back( id );
    pfEvt_.pfPt_.push_back( rndSF(pt,4) );
    pfEvt_.pfEnergy_.push_back( rndSF(energy,4) );
    pfEvt_.pfVsPt_.push_back( rndSF(vsPt,4) );
    pfEvt_.pfVsPtInitial_.push_back( rndSF(vsPtInitial,4) );
    pfEvt_.pfArea_.push_back( rndSF(vsArea,4) );
    pfEvt_.pfEta_.push_back( rndDP(pfCandidate.eta(),3) );
    pfEvt_.pfPhi_.push_back( rndDP(pfCandidate.phi(),3) );
    pfEvt_.nPFpart_++;

  }

  // Fill GEN info
  if(doMC_){
    edm::Handle<reco::GenParticleCollection> genParticles;
    iEvent.getByToken(genLabel_,genParticles);
    const reco::GenParticleCollection* genColl= &(*genParticles);

    for(unsigned igen=0;igen<genColl->size(); igen++) {

      const reco::GenParticle gen = genColl->at(igen);
      double eta = gen.eta();
      double pt = gen.pt();

      if(gen.status()==1 && fabs(eta)<3.0 && pt> genPtMin_){
	pfEvt_.genPDGId_.push_back( gen.pdgId() );
	pfEvt_.genPt_.push_back( rndSF(pt,4) );
	pfEvt_.genEta_.push_back( rndDP(eta,3) );
	pfEvt_.genPhi_.push_back( rndDP(gen.phi(),3) );
	pfEvt_.nGENpart_++;
      }
    }
  }

  // Fill Jet info
  if(doJets_){
    edm::Handle<pat::JetCollection> jets;
    iEvent.getByToken(jetLabel_,jets);
    const pat::JetCollection *jetColl = &(*jets);


    for(unsigned ijet=0;ijet<jetColl->size(); ijet++) {
      const pat::Jet jet = jetColl->at(ijet);

      double pt =  jet.pt();
      double energy =  jet.energy();
      if(pt>jetPtMin_){
	pfEvt_.jetPt_.push_back( pt );
	pfEvt_.jetEnergy_.push_back( energy );
	pfEvt_.jetEta_.push_back( jet.eta() );
	pfEvt_.jetPhi_.push_back( jet.phi() );
	pfEvt_.njets_++;
      }
    }
  }

  // All done
  pfTree_->Fill();
}

void HiPFCandAnalyzer::beginJob()
{

  // -- trees --
  pfTree_ = fs->make<TTree>("pfTree","dijet tree");
  pfEvt_.SetTree(pfTree_);
  pfEvt_.doMC = doMC_;
  pfEvt_.doJets = doJets_;

  pfEvt_.SetBranches(etaBins_, fourierOrder_, doUEraw_);
}

// ------------ method called once each job just after ending the event loop  ------------
void
HiPFCandAnalyzer::endJob() {
}

// constructors
TreePFCandEventData::TreePFCandEventData(){
}


// set branches
void TreePFCandEventData::SetBranches(int etaBins, int fourierOrder, bool doUEraw)
{
  // --event level--

  // -- particle info --
  tree_->Branch("nPFpart",&(this->nPFpart_),"nPFpart/I");
  tree_->Branch("pfId",&(this->pfId_));
  tree_->Branch("pfPt",&(this->pfPt_));
  tree_->Branch("pfEnergy",&(this->pfEnergy_));
  tree_->Branch("pfVsPtInitial",&(this->pfVsPtInitial_));

  tree_->Branch("pfEta",&(this->pfEta_));
  tree_->Branch("pfPhi",&(this->pfPhi_));

  // -- jet info --
  if(doJets){
    tree_->Branch("njets",&(this->njets_),"njets/I");
    tree_->Branch("jetPt",&(this->jetPt_));
    tree_->Branch("jetEta",&(this->jetEta_));
    tree_->Branch("jetPhi",&(this->jetPhi_));
  }

  tree_->Branch("vn",this->vn,Form("vn[%d][%d]/F",fourierOrder,etaBins));
  tree_->Branch("psin",this->psin,Form("vpsi[%d][%d]/F",fourierOrder,etaBins));
  tree_->Branch("sumpt",this->sumpt,Form("sumpt[%d]/F",etaBins));
  if(doUEraw){
    tree_->Branch("ueraw",this->ueraw,Form("ueraw[%d]/F",(fourierOrder*etaBins*2*3)));
  }

  // -- gen info --
  if(doMC){
    tree_->Branch("nGENpart",&(this->nGENpart_),"nGENpart/I");
    tree_->Branch("genPDGId",&(this->genPDGId_));
    tree_->Branch("genPt",&(this->genPt_));
    tree_->Branch("genEta",&(this->genEta_));
    tree_->Branch("genPhi",&(this->genPhi_));
  }

}
void TreePFCandEventData::Clear()
{
  // event

  nPFpart_      = 0;
  njets_        = 0;
  nGENpart_     = 0;

  pfId_.clear();
  genPDGId_.clear();
  pfEnergy_.clear();
  jetEnergy_.clear();
  pfPt_.clear();
  genPt_.clear();
  jetPt_.clear();
  pfEta_.clear();
  genEta_.clear();
  jetEta_.clear();
  pfPhi_.clear();
  genPhi_.clear();
  jetPhi_.clear();
  pfVsPt_.clear();
  pfVsPtInitial_.clear();
  pfVsPtEqualized_.clear();
  pfArea_.clear();
}

DEFINE_FWK_MODULE(HiPFCandAnalyzer);
