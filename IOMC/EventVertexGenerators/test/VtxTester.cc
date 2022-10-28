#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "DataFormats/Common/interface/ValidHandle.h"

#include "FWCore/Framework/interface/Event.h"
#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"

#include "TFile.h"
#include "TH1.h"
#include "TH2.h"

#include "CLHEP/Units/PhysicalConstants.h"
#include "CLHEP/Units/SystemOfUnits.h"

#include "FWCore/Framework/interface/MakerMacros.h"

class VtxTester : public edm::one::EDAnalyzer<> {
public:
  //
  explicit VtxTester(const edm::ParameterSet&);
  virtual ~VtxTester() {}

  void beginJob() override {}
  void analyze(const edm::Event&, const edm::EventSetup&) override;
  void endJob() override {}

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  edm::EDGetTokenT<edm::HepMCProduct> srcHepMCToken_;

  TH1D* fVtxHistTime_;
  TH1D* fVtxHistz_;
  TH2D* fVtxHistzTime_;
  TH1D* fVtxHistx_;
  TH1D* fVtxHisty_;
  TH2D* fVtxHistxy_;
  TH1D* fPhiHist_;
  TH1D* fEtaHist_;
};

VtxTester::VtxTester(const edm::ParameterSet& cfg)
    : srcHepMCToken_(consumes<edm::HepMCProduct>(cfg.getParameter<edm::InputTag>("src"))) {
  edm::Service<TFileService> fs;

  fVtxHistTime_ = fs->make<TH1D>("VtxHistTime", "#vtx, t [ns]", 60, -0.9, 0.9);
  fVtxHistz_ = fs->make<TH1D>("VtxHistz", "#vtx, z [mm]", 400, -200., 200.);
  fVtxHistzTime_ = fs->make<TH2D>("VtxHistzTime", "#vtx time [ns] vs z [mm]", 400, -200., 200., 60, -0.9, 0.9);
  fVtxHistx_ = fs->make<TH1D>("VtxHistx", "#vtx, x [mm]", 200, -1., 1.);
  fVtxHisty_ = fs->make<TH1D>("VtxHisty", "#vtx, y [mm]", 200, -1., 1.);
  fVtxHistxy_ = fs->make<TH2D>("VtxHistxy", "#vtx y vs x [mm]", 200, -1., 1., 200, -1., 1.);

  fPhiHist_ = fs->make<TH1D>("PhiHist", "#vtx phi", 70, -3.5, 3.5);
  fEtaHist_ = fs->make<TH1D>("EtaHist", "#vtx eta", 120, -6., 6.);
}

void VtxTester::analyze(const edm::Event& e, const edm::EventSetup&) {
  auto theGenEvent = makeValid(e.getHandle(srcHepMCToken_));

  const HepMC::GenEvent* Evt = theGenEvent->GetEvent();

  // take only 1st vertex for now - it's been tested only of PGuns...

  HepMC::GenEvent::vertex_const_iterator Vtx = Evt->vertices_begin();

  fVtxHistTime_->Fill((*Vtx)->position().t() * CLHEP::mm / CLHEP::c_light);
  fVtxHistz_->Fill((*Vtx)->position().z() * CLHEP::mm);
  fVtxHistzTime_->Fill((*Vtx)->position().z() * CLHEP::mm, (*Vtx)->position().t() * CLHEP::mm / CLHEP::c_light);
  fVtxHistx_->Fill((*Vtx)->position().x() * CLHEP::mm);
  fVtxHisty_->Fill((*Vtx)->position().y() * CLHEP::mm);
  fVtxHistxy_->Fill((*Vtx)->position().x() * CLHEP::mm, (*Vtx)->position().y() * CLHEP::mm);

  for (HepMC::GenEvent::particle_const_iterator Part = Evt->particles_begin(); Part != Evt->particles_end(); Part++) {
    HepMC::FourVector Mom = (*Part)->momentum();
    double Phi = Mom.phi();
    double Eta = -log(tan(Mom.theta() / 2.));

    fPhiHist_->Fill(Phi);
    fEtaHist_->Fill(Eta);
  }

  return;
}

void VtxTester::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("src", edm::InputTag("generatorSmeared"))
      ->setComment("Input generated HepMC event after vtx smearing");
  descriptions.add("vtxTester", desc);
}

DEFINE_FWK_MODULE(VtxTester);
