#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
//#include "DataFormats/PatCandidates/interface/CompositeCandidate.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "DataFormats/PatCandidates/interface/Tau.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"

#include "TH1D.h"
#include "TH2D.h"
#include "TGraphAsymmErrors.h"

class rerunMVAIsolationOnMiniAOD_Phase2 : public edm::EDAnalyzer {
public:
  explicit rerunMVAIsolationOnMiniAOD_Phase2(const edm::ParameterSet&);
private:
   virtual void analyze(const edm::Event&, const edm::EventSetup&) override;
   virtual void endJob() override;

   edm::EDGetTokenT<pat::TauCollection> tauToken_;
 
   TH1D *h_raw_bkg, *h_raw_sig;
 
   TH1D *h_pt_denom_bkg, *h_pt_denom_sig;
   TH1D *h_pt_num_bkg[7], *h_pt_num_sig[7];
   TGraphAsymmErrors *eff[7], *fak[7];
   
   TH1D *h_inc_denom_bkg, *h_inc_num_bkg;
   TH1D *h_inc_denom_sig, *h_inc_num_sig;
   TGraphAsymmErrors *inceff, *incfak;

   TH2D *h2_denom_bkg, *h2_denom_sig;
   TH2D *h2_num_bkg[7], *h2_num_sig[7];
   TH2D *eff2[7], *fak2[7];

   bool haveGenJets;
   edm::EDGetTokenT<std::vector<reco::GenJet>> genJetToken_;   
   //bool haveGenVisTaus;
   //edm::EDGetTokenT<pat::CompositeCandidateCollection> genVisTauToken_;
   bool haveGenParticles;
   edm::EDGetTokenT<std::vector<reco::GenParticle>> genParticleToken_;
};

rerunMVAIsolationOnMiniAOD_Phase2::rerunMVAIsolationOnMiniAOD_Phase2(const edm::ParameterSet& iConfig)
{
   tauToken_ = consumes<pat::TauCollection>(edm::InputTag("newTauIDsEmbedded"));
 
   haveGenJets = false;
   if (iConfig.existsAs<edm::InputTag>("genJetCollection")) {  
      haveGenJets = true;
      genJetToken_ = consumes<std::vector<reco::GenJet>>(iConfig.getParameter<edm::InputTag>("genJetCollection"));
   }
   //haveGenVisTaus = false;
   //if (iConfig.existsAs<edm::InputTag>("genVisTauCollection")) {
   //   haveGenVisTaus = true;
   //   genVisTauToken_ = consumes<pat::CompositeCandidateCollection>(iConfig.getParameter<edm::InputTag>("genVisTauCollection"));
   //}
   haveGenParticles = false;
   if (iConfig.existsAs<edm::InputTag>("genParticleCollection")) {
      haveGenParticles = true;
      genParticleToken_ = consumes<std::vector<reco::GenParticle>>(iConfig.getParameter<edm::InputTag>("genParticleCollection"));
   }

   edm::Service<TFileService> fileService;
   h_raw_sig = fileService->make<TH1D>("h_raw_sig", ";BDT output;taus / 0.02", 110, -1.1, 1.1);
   h_raw_bkg = fileService->make<TH1D>("h_raw_bkg", ";BDT output;taus / 0.02", 110, -1.1, 1.1);
   
   h_inc_num_sig = fileService->make<TH1D>("h_inc_num_sig", ";working point;taus / bin", 7, 0.5, 7.5);
   h_inc_denom_sig = fileService->make<TH1D>("h_inc_denom_sig", ";working point;taus / bin", 7, 0.5, 7.5);
   h_inc_num_bkg = fileService->make<TH1D>("h_inc_num_bkg", ";working point;taus / bin", 7, 0.5, 7.5);
   h_inc_denom_bkg = fileService->make<TH1D>("h_inc_denom_bkg", ";working point;taus / bin", 7, 0.5, 7.5);
   inceff = fileService->make<TGraphAsymmErrors>(7);
   inceff->SetName("inceff");
   incfak = fileService->make<TGraphAsymmErrors>(7);
   incfak->SetName("incfak");

   const int n = 10;
   const double x[n+1] = {20., 25.4196, 32.3079, 41.0627, 52.19, 66.3325, 84.3074, 107.153, 136.19, 173.095, 220.};
   h_pt_denom_sig = fileService->make<TH1D>("h_pt_denom_sig", ";p_{T} [GeV];taus / bin", n, x);
   h_pt_denom_bkg = fileService->make<TH1D>("h_pt_denom_bkg", ";p_{T} [GeV];taus / bin", n, x);
   for (int i = 0; i < 7; ++i) {
      const TString tag = TString::Itoa(i, 10);
      h_pt_num_sig[i] = fileService->make<TH1D>("h_pt_num_sig_"+tag, ";p_{T} [GeV];taus / bin", n, x);
      h_pt_num_bkg[i] = fileService->make<TH1D>("h_pt_num_bkg_"+tag, ";p_{T} [GeV];taus / bin", n, x);
      eff[i] = fileService->make<TGraphAsymmErrors>(10);
      eff[i]->SetName("eff_"+TString::Itoa(i, 10));
      fak[i] = fileService->make<TGraphAsymmErrors>(10);
      fak[i]->SetName("fak_"+TString::Itoa(i, 10));
   }
   const double xeta[4] = {0., 1.5, 2.3, 3.};
   const double xpt[3] = {20., 50., 220.};
   h2_denom_sig = fileService->make<TH2D>("h2_denom_sig", ";|#eta|;p_{T} [GeV]", 3, xeta, 2, xpt);
   h2_denom_bkg = fileService->make<TH2D>("h2_denom_bkg", ";|#eta|;p_{T} [GeV]", 3, xeta, 2, xpt);
   for (int i = 0; i < 7; ++i) {
      const TString tag = TString::Itoa(i, 10);
      h2_num_sig[i] = fileService->make<TH2D>("h2_num_sig_"+tag, ";|#eta|;p_{T} [GeV]", 3, xeta, 2, xpt);
      h2_num_sig[i]->Sumw2();
      h2_num_bkg[i] = fileService->make<TH2D>("h2_num_bkg_"+tag, ";|#eta|;p_{T} [GeV]", 3, xeta, 2, xpt);
      h2_num_bkg[i]->Sumw2();
      eff2[i] = fileService->make<TH2D>("eff2_"+tag, ";|#eta|;p_{T} [GeV]", 3, xeta, 2, xpt);
      eff2[i]->Sumw2();
      fak2[i] = fileService->make<TH2D>("fak2_"+tag, ";|#eta|;p_{T} [GeV]", 3, xeta, 2, xpt);  
      fak2[i]->Sumw2();
   }
}

void rerunMVAIsolationOnMiniAOD_Phase2::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   edm::Handle<pat::TauCollection> taus;
   iEvent.getByToken(tauToken_, taus);

   edm::Handle<std::vector<reco::GenJet>> genJets;
   if (haveGenJets) iEvent.getByToken(genJetToken_, genJets);

   //edm::Handle<pat::CompositeCandidateCollection> genVisTaus;
   //if (haveGenVisTaus) iEvent.getByToken(genVisTauToken_, genVisTaus);

   edm::Handle<std::vector<reco::GenParticle>> genParticles;
   if (haveGenParticles) iEvent.getByToken(genParticleToken_, genParticles);
 
   for (auto i = taus->begin(); i != taus->end(); ++i) {
      
      const double pt = i->pt();
      const double eta = std::abs(i->eta());
      if (pt<20. || pt>=220. || eta>=3.) continue;
      if (!i->tauID("decayModeFindingNewDMs")) continue;

      bool isPU = false;
      if (haveGenJets) {
         double mindr_jet = 9.;
         for (auto j = genJets->begin(); j != genJets->end(); ++j) {
            if (reco::deltaR(*i, *j)<mindr_jet) {
               mindr_jet = reco::deltaR(*i, *j);
            }
         }
         if (mindr_jet>=0.4) isPU = true;
      }
      if (isPU) continue;

      //bool isSig = false;
      //if (haveGenVisTaus) {
      //   double mindr = 99.;
      //   for (auto j = genVisTaus->begin(); j != genVisTaus->end(); ++j) {
      //      if (std::abs(j->pdgId())==15 && reco::deltaR(*i, *j)<mindr) {
      //         mindr = reco::deltaR(*i, *j);
      //      }
      //   }
      //   if (mindr<0.4) isSig = true;
      //}

      bool isSig = false;
      if (haveGenParticles) {
         double mindr_e = 99.;
         double mindr_m = 99.;
         double mindr_t = 99.;
         for (auto j = genParticles->begin(); j != genParticles->end(); ++j) {
            const double dr = reco::deltaR(*i, *j);
            const int pdgid = std::abs(j->pdgId());
            if (pdgid==11 && j->status()==1 && dr<mindr_e) mindr_e = dr;
            if (pdgid==13 && j->status()==1 && dr<mindr_m) mindr_m = dr;
            if (pdgid==15 && j->isLastCopy() && dr<mindr_t) mindr_t = dr;
         }
         if (mindr_e>=0.4 && mindr_m>=0.4 && mindr_t<0.5) isSig = true;
      }

      const double byIsolationMVAPhase2raw = i->tauID("byIsolationMVAPhase2raw");
      double wp[7];
      wp[0] = i->tauID("byVVLooseIsolationMVAPhase2New");
      wp[1] = i->tauID("byVLooseIsolationMVAPhase2New");
      wp[2] = i->tauID("byLooseIsolationMVAPhase2New");
      wp[3] = i->tauID("byMediumIsolationMVAPhase2New");
      wp[4] = i->tauID("byTightIsolationMVAPhase2New");
      wp[5] = i->tauID("byVTightIsolationMVAPhase2New");
      wp[6] = i->tauID("byVVTightIsolationMVAPhase2New");
      
      if (isSig) {
         h_raw_sig->Fill(byIsolationMVAPhase2raw);
         h_pt_denom_sig->Fill(pt);
         h2_denom_sig->Fill(eta, pt);
         for (int j = 0; j < 7; ++j) {
            h_inc_denom_sig->Fill(j+1);
            if (wp[j]) {
               h_inc_num_sig->Fill(j+1);
               h_pt_num_sig[j]->Fill(pt);
               h2_num_sig[j]->Fill(eta, pt);
            }
         }
      } else {
         h_raw_bkg->Fill(byIsolationMVAPhase2raw);
         h_pt_denom_bkg->Fill(pt);
         h2_denom_bkg->Fill(eta, pt);
         for (int j = 0; j < 7; ++j) {
            h_inc_denom_bkg->Fill(j+1);
            if (wp[j]) {
               h_inc_num_bkg->Fill(j+1);
               h_pt_num_bkg[j]->Fill(pt);
               h2_num_bkg[j]->Fill(eta, pt);
            }
         }
      }
   }

}

void rerunMVAIsolationOnMiniAOD_Phase2::endJob()
{
   inceff->Divide(h_inc_num_sig, h_inc_denom_sig, "e0");
   inceff->SetTitle(";working point;tagging efficiency");
   incfak->Divide(h_inc_num_bkg, h_inc_denom_bkg, "e0");
   incfak->SetTitle(";working point;tagging efficiency");
   for (int i = 0; i < 7; ++i) {
      eff[i]->Divide(h_pt_num_sig[i], h_pt_denom_sig, "e0");
      eff[i]->SetTitle(";p_{T} [GeV];tagging efficiency");
      fak[i]->Divide(h_pt_num_bkg[i], h_pt_denom_bkg, "e0");
      fak[i]->SetTitle(";p_{T} [GeV];tagging efficiency");
      eff2[i]->Divide(h2_num_sig[i], h2_denom_sig, 1., 1., "B");
      eff2[i]->SetTitle("tagging efficiency;|eta|;p_{T} [GeV]");
      fak2[i]->Divide(h2_num_bkg[i], h2_denom_bkg, 1., 1., "B");
      fak2[i]->SetTitle("tagging efficiency;|eta|;p_{T} [GeV]");
   }
}

//define this as a plug-in
DEFINE_FWK_MODULE(rerunMVAIsolationOnMiniAOD_Phase2);

