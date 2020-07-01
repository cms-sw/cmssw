#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

class TH1D;

class ExhumeAnalyzer : public edm::EDAnalyzer {
public:
  explicit ExhumeAnalyzer(const edm::ParameterSet&);
  ~ExhumeAnalyzer() override;

  void analyze(const edm::Event& event, const edm::EventSetup& eventSetup) override;

  //virtual void beginJob(const edm::EventSetup& eventSetup);
  void beginJob() override;
  void endJob() override;

private:
  edm::InputTag genParticlesTag_;

  double Ebeam_;

  // Histograms
  TH1D* hist_eta_;
  TH1D* hist_phi1_;
  TH1D* hist_t1_;
  TH1D* hist_xigen1_;
  TH1D* hist_phi2_;
  TH1D* hist_t2_;
  TH1D* hist_xigen2_;
  TH1D* hist_sHat_;
  TH1D* hist_MX_;
};

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "DataFormats/Common/interface/Handle.h"

#include "DataFormats/Common/interface/View.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"

#include "TH1D.h"

ExhumeAnalyzer::ExhumeAnalyzer(const edm::ParameterSet& pset)
    : genParticlesTag_(pset.getParameter<edm::InputTag>("GenParticleTag")),
      Ebeam_(pset.getParameter<double>("EBeam")) {}

void ExhumeAnalyzer::beginJob() {
  edm::Service<TFileService> fs;

  hist_eta_ = fs->make<TH1D>("hist_eta", "#eta system", 100, -4.5, 4.5);
  hist_phi1_ = fs->make<TH1D>("hist_phi1", "#phi proton 1", 100, -1.1 * M_PI, 1.1 * M_PI);
  hist_t1_ = fs->make<TH1D>("hist_t1", "t proton 1", 100, -1.4, 0);
  hist_xigen1_ = fs->make<TH1D>("hist_xigen1", "#xi proton 1", 100, 0., 0.1);
  hist_phi2_ = fs->make<TH1D>("hist_phi2", "#phi proton 2", 100, -1.1 * M_PI, 1.1 * M_PI);
  hist_t2_ = fs->make<TH1D>("hist_t2", "t proton 1", 100, -1.4, 0);
  hist_xigen2_ = fs->make<TH1D>("hist_xigen2", "#xi proton 2", 100, 0., 0.1);
  hist_sHat_ = fs->make<TH1D>("hist_sHat", "Central inv. mass", 100, 80., 150.);
  hist_MX_ = fs->make<TH1D>("hist_MX", "Missing mass", 100, 80., 150.);
}

ExhumeAnalyzer::~ExhumeAnalyzer() {}

void ExhumeAnalyzer::analyze(const edm::Event& event, const edm::EventSetup& setup) {
  // Generator Information
  edm::Handle<reco::GenParticleCollection> genParticles;
  event.getByLabel(genParticlesTag_, genParticles);

  // Look for protons
  double pz1max = 0.;
  double pz2min = 0.;
  reco::GenParticleCollection::const_iterator proton1 = genParticles->end();
  reco::GenParticleCollection::const_iterator proton2 = genParticles->end();
  for (reco::GenParticleCollection::const_iterator genpart = genParticles->begin(); genpart != genParticles->end();
       ++genpart) {
    if (genpart->status() != 1)
      continue;

    double pz = genpart->pz();
    if ((genpart->pdgId() == 2212) && (pz > 0.75 * Ebeam_)) {
      if (pz > pz1max) {
        proton1 = genpart;
        pz1max = pz;
      }
    } else if ((genpart->pdgId() == 2212) && (pz < -0.75 * Ebeam_)) {
      if (pz < pz2min) {
        proton2 = genpart;
        pz2min = pz;
      }
    }
  }

  if ((proton1 != genParticles->end()) && (proton2 != genParticles->end())) {
    std::cout << "Proton 1: " << proton1->pt() << "  " << proton1->eta() << "  " << proton1->phi() << std::endl;
    std::cout << "Proton 2: " << proton2->pt() << "  " << proton2->eta() << "  " << proton2->phi() << std::endl;

    math::XYZTLorentzVector proton1in(0., 0., Ebeam_, Ebeam_);
    math::XYZTLorentzVector proton1diff(proton1->px(), proton1->py(), proton1->pz(), proton1->energy());
    double t1 = (proton1diff - proton1in).M2();
    double xigen1 = 1 - proton1diff.pz() / Ebeam_;
    math::XYZTLorentzVector proton2in(0., 0., -Ebeam_, Ebeam_);
    math::XYZTLorentzVector proton2diff(proton2->px(), proton2->py(), proton2->pz(), proton2->energy());
    double t2 = (proton2diff - proton2in).M2();
    double xigen2 = 1 + proton2diff.pz() / Ebeam_;

    double eta = 0.5 * log(xigen1 / xigen2);
    double pt1 = sqrt(-t1);
    double pt2 = sqrt(-t2);
    double phi1 = proton1diff.phi();
    double phi2 = proton2diff.phi();
    double s = 4 * Ebeam_ * Ebeam_;
    double sHat = t1 + t2 - 2 * pt1 * pt2 * cos(phi1 - phi2) + s * xigen1 * xigen2;
    double MX = sqrt(s * xigen1 * xigen2);

    //Fill histograms
    hist_eta_->Fill(eta);
    hist_phi1_->Fill(phi1);
    hist_t1_->Fill(t1);
    hist_xigen1_->Fill(xigen1);
    hist_phi2_->Fill(phi2);
    hist_t2_->Fill(t2);
    hist_xigen2_->Fill(xigen2);
    hist_sHat_->Fill(sqrt(sHat));
    hist_MX_->Fill(MX);
  }
}

void ExhumeAnalyzer::endJob() {}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(ExhumeAnalyzer);
