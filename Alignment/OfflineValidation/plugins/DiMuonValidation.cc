// system includes
#include <memory>
#include <cmath>
#include <fmt/printf.h>

// user include files
#include "Alignment/OfflineValidation/interface/DiLeptonVertexHelpers.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "DataFormats/Candidate/interface/Particle.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "Geometry/CommonDetUnit/interface/GeomDet.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"

// ROOT includes
#include "TH1D.h"
#include "TH2D.h"
#include "TH3D.h"

namespace DiMuonValid {
  using LV = reco::Particle::LorentzVector;
}
//
// class declaration
//
class DiMuonValidation : public edm::one::EDAnalyzer<edm::one::SharedResources> {
public:
  explicit DiMuonValidation(const edm::ParameterSet& pset)
      : eBeam_(pset.getParameter<double>("eBeam")),
        compressionSettings_(pset.getUntrackedParameter<int>("compressionSettings", -1)),
        TkTag_(pset.getParameter<std::string>("TkTag")),
        pair_mass_min_(pset.getParameter<double>("Pair_mass_min")),
        pair_mass_max_(pset.getParameter<double>("Pair_mass_max")),
        pair_mass_nbins_(pset.getParameter<int>("Pair_mass_nbins")),
        pair_etaminpos_(pset.getParameter<double>("Pair_etaminpos")),
        pair_etamaxpos_(pset.getParameter<double>("Pair_etamaxpos")),
        pair_etaminneg_(pset.getParameter<double>("Pair_etaminneg")),
        pair_etamaxneg_(pset.getParameter<double>("Pair_etamaxneg")),
        variable_CosThetaCS_xmin_(pset.getParameter<double>("Variable_CosThetaCS_xmin")),
        variable_CosThetaCS_xmax_(pset.getParameter<double>("Variable_CosThetaCS_xmax")),
        variable_CosThetaCS_nbins_(pset.getParameter<int>("Variable_CosThetaCS_nbins")),
        variable_DeltaEta_xmin_(pset.getParameter<double>("Variable_DeltaEta_xmin")),
        variable_DeltaEta_xmax_(pset.getParameter<double>("Variable_DeltaEta_xmax")),
        variable_DeltaEta_nbins_(pset.getParameter<int>("Variable_DeltaEta_nbins")),
        variable_EtaMinus_xmin_(pset.getParameter<double>("Variable_EtaMinus_xmin")),
        variable_EtaMinus_xmax_(pset.getParameter<double>("Variable_EtaMinus_xmax")),
        variable_EtaMinus_nbins_(pset.getParameter<int>("Variable_EtaMinus_nbins")),
        variable_EtaPlus_xmin_(pset.getParameter<double>("Variable_EtaPlus_xmin")),
        variable_EtaPlus_xmax_(pset.getParameter<double>("Variable_EtaPlus_xmax")),
        variable_EtaPlus_nbins_(pset.getParameter<int>("Variable_EtaPlus_nbins")),
        variable_PhiCS_xmin_(pset.getParameter<double>("Variable_PhiCS_xmin")),
        variable_PhiCS_xmax_(pset.getParameter<double>("Variable_PhiCS_xmax")),
        variable_PhiCS_nbins_(pset.getParameter<int>("Variable_PhiCS_nbins")),
        variable_PhiMinus_xmin_(pset.getParameter<double>("Variable_PhiMinus_xmin")),
        variable_PhiMinus_xmax_(pset.getParameter<double>("Variable_PhiMinus_xmax")),
        variable_PhiMinus_nbins_(pset.getParameter<int>("Variable_PhiMinus_nbins")),
        variable_PhiPlus_xmin_(pset.getParameter<double>("Variable_PhiPlus_xmin")),
        variable_PhiPlus_xmax_(pset.getParameter<double>("Variable_PhiPlus_xmax")),
        variable_PhiPlus_nbins_(pset.getParameter<int>("Variable_PhiPlus_nbins")),
        variable_PairPt_xmin_(pset.getParameter<double>("Variable_PairPt_xmin")),
        variable_PairPt_xmax_(pset.getParameter<double>("Variable_PairPt_xmax")),
        variable_PairPt_nbins_(pset.getParameter<int>("Variable_PairPt_nbins")) {
    usesResource(TFileService::kSharedResource);
    theTrackCollectionToken_ = consumes<reco::TrackCollection>(TkTag_);

    variables_min_[Variable::CosThetaCS] = variable_CosThetaCS_xmin_;
    variables_min_[Variable::DeltaEta] = variable_DeltaEta_xmin_;
    variables_min_[Variable::EtaMinus] = variable_EtaMinus_xmin_;
    variables_min_[Variable::EtaPlus] = variable_EtaPlus_xmin_;
    variables_min_[Variable::PhiCS] = variable_PhiCS_xmin_;
    variables_min_[Variable::PhiMinus] = variable_PhiMinus_xmin_;
    variables_min_[Variable::PhiPlus] = variable_PhiPlus_xmin_;
    variables_min_[Variable::Pt] = variable_PairPt_xmin_;

    variables_max_[Variable::CosThetaCS] = variable_CosThetaCS_xmax_;
    variables_max_[Variable::DeltaEta] = variable_DeltaEta_xmax_;
    variables_max_[Variable::EtaMinus] = variable_EtaMinus_xmax_;
    variables_max_[Variable::EtaPlus] = variable_EtaPlus_xmax_;
    variables_max_[Variable::PhiCS] = variable_PhiCS_xmax_;
    variables_max_[Variable::PhiMinus] = variable_PhiMinus_xmax_;
    variables_max_[Variable::PhiPlus] = variable_PhiPlus_xmax_;
    variables_max_[Variable::Pt] = variable_PairPt_xmax_;

    variables_bins_number_[Variable::CosThetaCS] = variable_CosThetaCS_nbins_;
    variables_bins_number_[Variable::DeltaEta] = variable_DeltaEta_nbins_;
    variables_bins_number_[Variable::EtaMinus] = variable_EtaMinus_nbins_;
    variables_bins_number_[Variable::EtaPlus] = variable_EtaPlus_nbins_;
    variables_bins_number_[Variable::PhiCS] = variable_PhiCS_nbins_;
    variables_bins_number_[Variable::PhiMinus] = variable_PhiMinus_nbins_;
    variables_bins_number_[Variable::PhiPlus] = variable_PhiPlus_nbins_;
    variables_bins_number_[Variable::Pt] = variable_PairPt_nbins_;
  }

  ~DiMuonValidation() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  // enumeration to contain the variables to plot
  enum Variable {
    CosThetaCS = 0,
    DeltaEta = 1,
    EtaMinus = 2,
    EtaPlus = 3,
    PhiCS = 4,
    PhiMinus = 5,
    PhiPlus = 6,
    Pt = 7,
    VarNumber = 8
  };

  static constexpr double mu_mass2_ = 0.105658 * 0.105658;  //The invariant mass of muon 105.658MeV

private:
  void beginJob() override;
  void analyze(const edm::Event&, const edm::EventSetup&) override;

  // ----------member data ---------------------------
  float eBeam_;
  int compressionSettings_;
  std::string TkTag_;

  double pair_mass_min_;
  double pair_mass_max_;
  int pair_mass_nbins_;
  double pair_etaminpos_;
  double pair_etamaxpos_;
  double pair_etaminneg_;
  double pair_etamaxneg_;

  double variable_CosThetaCS_xmin_;
  double variable_CosThetaCS_xmax_;
  int variable_CosThetaCS_nbins_;

  double variable_DeltaEta_xmin_;
  double variable_DeltaEta_xmax_;
  int variable_DeltaEta_nbins_;

  double variable_EtaMinus_xmin_;
  double variable_EtaMinus_xmax_;
  int variable_EtaMinus_nbins_;

  double variable_EtaPlus_xmin_;
  double variable_EtaPlus_xmax_;
  int variable_EtaPlus_nbins_;

  double variable_PhiCS_xmin_;
  double variable_PhiCS_xmax_;
  int variable_PhiCS_nbins_;

  double variable_PhiMinus_xmin_;
  double variable_PhiMinus_xmax_;
  int variable_PhiMinus_nbins_;

  double variable_PhiPlus_xmin_;
  double variable_PhiPlus_xmax_;
  int variable_PhiPlus_nbins_;

  double variable_PairPt_xmin_;
  double variable_PairPt_xmax_;
  int variable_PairPt_nbins_;

  edm::EDGetTokenT<reco::TrackCollection> theTrackCollectionToken_;

  TH1F* th1f_mass;
  TH2D* th2d_mass_variables_[Variable::VarNumber];  // actual histograms
  TH3D* th3d_mass_vs_eta_phi_plus_;                 // 3D histogram for scatter plot vs eta / phi (mu+)
  TH3D* th3d_mass_vs_eta_phi_minus_;                // 3D histogram for scatter plot vs eta / phi (mu-)

  std::string variables_name_[Variable::VarNumber] = {
      "CosThetaCS", "DeltaEta", "EtaMinus", "EtaPlus", "PhiCS", "PhiMinus", "PhiPlus", "Pt"};

  std::string variables_title_[Variable::VarNumber] = {"Cos#theta_{CS}",
                                                       "#Delta#eta(#mu^{-},#mu^{+})",
                                                       "#mu^{-} #eta",
                                                       "#mu^{+} #eta",
                                                       "#phi_{CS} [rad]",
                                                       "#mu^{-} #phi [rad]",
                                                       "#mu^{+} #phi [rad]",
                                                       "p_{T} [GeV]"};

  int variables_bins_number_[Variable::VarNumber];  // = {20, 20, 12, 12, 20, 16, 16, 100};
  double variables_min_[Variable::VarNumber];       // = {-1, -4.8, -2.4, -2.4, -M_PI / 2, -M_PI, -M_PI, 0};
  double variables_max_[Variable::VarNumber];       // = {+1, +4.8, +2.4, +2.4, +M_PI / 2, +M_PI, +M_PI, 100};

  DiLeptonHelp::PlotsVsDiLeptonRegion InvMassInEtaBins = DiLeptonHelp::PlotsVsDiLeptonRegion(1.5);
  DiLeptonHelp::PlotsVsDiLeptonRegion InvMassVsPhiPlusInEtaBins = DiLeptonHelp::PlotsVsDiLeptonRegion(1.5);
  DiLeptonHelp::PlotsVsDiLeptonRegion InvMassVsPhiMinusInEtaBins = DiLeptonHelp::PlotsVsDiLeptonRegion(1.5);
  DiLeptonHelp::PlotsVsDiLeptonRegion InvMassVsCosThetaCSInEtaBins = DiLeptonHelp::PlotsVsDiLeptonRegion(1.5);
};

//
// member functions
//

// ------------ method called for each event  ------------
void DiMuonValidation::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  using namespace edm;

  const reco::TrackCollection& tC = iEvent.get(theTrackCollectionToken_);

  DiMuonValid::LV LV_mother(0., 0., 0., 0.);
  for (reco::TrackCollection::const_iterator track1 = tC.begin(); track1 != tC.end(); track1++) {
    DiMuonValid::LV LV_track1(track1->px(),
                              track1->py(),
                              track1->pz(),
                              sqrt((track1->p() * track1->p()) + mu_mass2_));  //old 106

    for (reco::TrackCollection::const_iterator track2 = track1 + 1; track2 != tC.end(); track2++) {
      if (track1->charge() == track2->charge()) {
        continue;
      }  // only reconstruct opposite charge pair

      DiMuonValid::LV LV_track2(
          track2->px(), track2->py(), track2->pz(), sqrt((track2->p() * track2->p()) + mu_mass2_));

      LV_mother = LV_track1 + LV_track2;
      double mother_mass = LV_mother.M();
      th1f_mass->Fill(mother_mass);
      double mother_pt = LV_mother.Pt();

      int charge1 = track1->charge();
      double etaMu1 = track1->eta();
      double phiMu1 = track1->phi();
      double ptMu1 = track1->pt();

      int charge2 = track2->charge();
      double etaMu2 = track2->eta();
      double phiMu2 = track2->phi();
      double ptMu2 = track2->pt();

      if (charge1 < 0) {  // use Mu+ for charge1, Mu- for charge2
        std::swap(charge1, charge2);
        std::swap(etaMu1, etaMu2);
        std::swap(phiMu1, phiMu2);
        std::swap(ptMu1, ptMu2);
      }

      const auto& tplus = track1->charge() > 0 ? track1 : track2;
      const auto& tminus = track1->charge() < 0 ? track1 : track2;
      TLorentzVector p4_tplus(tplus->px(), tplus->py(), tplus->pz(), sqrt((tplus->p() * tplus->p()) + mu_mass2_));
      TLorentzVector p4_tminus(tminus->px(), tminus->py(), tminus->pz(), sqrt((tminus->p() * tminus->p()) + mu_mass2_));
      std::pair<TLorentzVector, TLorentzVector> tktk_p4 = std::make_pair(p4_tplus, p4_tminus);
      InvMassInEtaBins.fillTH1Plots(mother_mass, tktk_p4);
      InvMassVsPhiPlusInEtaBins.fillTH2Plots(mother_mass, phiMu1, tktk_p4);
      InvMassVsPhiMinusInEtaBins.fillTH2Plots(mother_mass, phiMu2, tktk_p4);

      //eta cut
      if (etaMu1 < pair_etaminpos_ || etaMu1 > pair_etamaxpos_ || etaMu2 < pair_etaminneg_ ||
          etaMu2 > pair_etamaxneg_) {
        continue;
      }

      double delta_eta = etaMu1 - etaMu2;

      double muplus = 1.0 / sqrt(2.0) * (LV_track1.E() + LV_track1.Z());
      double muminus = 1.0 / sqrt(2.0) * (LV_track1.E() - LV_track1.Z());
      double mubarplus = 1.0 / sqrt(2.0) * (LV_track2.E() + LV_track2.Z());
      double mubarminus = 1.0 / sqrt(2.0) * (LV_track2.E() - LV_track2.Z());
      //double costheta = 2.0 / Q.mag() / sqrt(pow(Q.mag(), 2) + pow(Q.Pt(), 2)) * (muplus * mubarminus - muminus * mubarplus);
      double costhetaCS = 2.0 / LV_mother.mag() / sqrt(pow(LV_mother.mag(), 2) + pow(LV_mother.Pt(), 2)) *
                          (muplus * mubarminus - muminus * mubarplus);

      InvMassVsCosThetaCSInEtaBins.fillTH2Plots(mother_mass, costhetaCS, tktk_p4);

      DiMuonValid::LV Pbeam(0., 0., eBeam_, eBeam_);
      auto R = Pbeam.Vect().Cross(LV_mother.Vect());
      auto Runit = R.Unit();
      auto Qt = LV_mother.Vect();
      Qt.SetZ(0);
      auto Qtunit = Qt.Unit();
      DiMuonValid::LV D(LV_track1 - LV_track2);
      auto Dt = D.Vect();
      Dt.SetZ(0);
      double tanphi =
          sqrt(pow(LV_mother.mag(), 2) + pow(LV_mother.Pt(), 2)) / LV_mother.mag() * Dt.Dot(Runit) / Dt.Dot(Qtunit);
      double phiCS = atan(tanphi);

      if (mother_mass > pair_mass_min_ && mother_mass < pair_mass_max_) {
        th2d_mass_variables_[Variable::CosThetaCS]->Fill(mother_mass, costhetaCS, 1);
        th2d_mass_variables_[Variable::DeltaEta]->Fill(mother_mass, delta_eta, 1);
        th2d_mass_variables_[Variable::EtaMinus]->Fill(mother_mass, etaMu2, 1);
        th2d_mass_variables_[Variable::EtaPlus]->Fill(mother_mass, etaMu1, 1);
        th2d_mass_variables_[Variable::PhiCS]->Fill(mother_mass, phiCS, 1);
        th2d_mass_variables_[Variable::PhiMinus]->Fill(mother_mass, phiMu2, 1);
        th2d_mass_variables_[Variable::PhiPlus]->Fill(mother_mass, phiMu1, 1);
        th2d_mass_variables_[Variable::Pt]->Fill(mother_mass, mother_pt, 1);

        th3d_mass_vs_eta_phi_plus_->Fill(mother_mass, etaMu1, phiMu1);
        th3d_mass_vs_eta_phi_minus_->Fill(mother_mass, etaMu2, phiMu2);
      }
    }
  }
}

// ------------ method called once each job just before starting event loop  ------------
void DiMuonValidation::beginJob() {
  edm::Service<TFileService> fs;
  if (compressionSettings_ > 0) {
    fs->file().SetCompressionSettings(compressionSettings_);
  }

  th1f_mass = fs->make<TH1F>("hMass", "mass;m_{#mu#mu} [GeV];events", 200, 0., 200.);

  for (int i = 0; i < Variable::VarNumber; i++) {
    std::string th2d_name = fmt::sprintf("th2d_mass_%s", variables_name_[i].c_str());
    th2d_mass_variables_[i] =
        fs->make<TH2D>(th2d_name.c_str(),
                       fmt::format("{};M_{{#mu^{{-}}#mu^{{+}}}} [GeV];{}", th2d_name, variables_title_[i]).c_str(),
                       pair_mass_nbins_,
                       pair_mass_min_,
                       pair_mass_max_,
                       variables_bins_number_[i],
                       variables_min_[i],
                       variables_max_[i]);
  }

  // 3D histogram for eta/phi map (mu+)
  th3d_mass_vs_eta_phi_plus_ =
      fs->make<TH3D>("th3d_mass_vs_eta_phi_plus",
                     "th3d_mass_vs_eta_phi_plus;M_{#mu^{-}#mu^{+}} [GeV];#mu^{+} #eta;#mu^{+} #phi [rad]",
                     pair_mass_nbins_,
                     pair_mass_min_,
                     pair_mass_max_,
                     variables_bins_number_[Variable::EtaPlus],
                     variables_min_[Variable::EtaPlus],
                     variables_max_[Variable::EtaPlus],
                     variables_bins_number_[Variable::PhiPlus],
                     variables_min_[Variable::PhiPlus],
                     variables_max_[Variable::PhiPlus]);

  // 3D histogram for eta/phi map (mu+)
  th3d_mass_vs_eta_phi_minus_ =
      fs->make<TH3D>("th3d_mass_vs_eta_phi_minus",
                     "th3d_mass_vs_eta_phi_minus;M_{#mu^{-}#mu^{+}} [GeV];#mu^{-} #eta;#mu^{-} #phi [rad]",
                     pair_mass_nbins_,
                     pair_mass_min_,
                     pair_mass_max_,
                     variables_bins_number_[Variable::EtaMinus],
                     variables_min_[Variable::EtaMinus],
                     variables_max_[Variable::EtaMinus],
                     variables_bins_number_[Variable::PhiMinus],
                     variables_min_[Variable::PhiMinus],
                     variables_max_[Variable::PhiMinus]);

  // Z-> mm mass in eta bins
  TFileDirectory dirResMassEta = fs->mkdir("TkTkMassInEtaBins");
  InvMassInEtaBins.bookSet(dirResMassEta, th1f_mass);

  // Z->mm mass vs cos phi Collins-Soper in eta bins
  TFileDirectory dirResMassVsCosThetaCSInEta = fs->mkdir("TkTkMassVsCosThetaCSInEtaBins");
  InvMassVsCosThetaCSInEtaBins.bookSet(dirResMassVsCosThetaCSInEta, th2d_mass_variables_[Variable::CosThetaCS]);

  // Z->mm mass vs phi minus in eta bins
  TFileDirectory dirResMassVsPhiMinusInEta = fs->mkdir("TkTkMassVsPhiMinusInEtaBins");
  InvMassVsPhiMinusInEtaBins.bookSet(dirResMassVsPhiMinusInEta, th2d_mass_variables_[Variable::PhiMinus]);

  // Z->mm mass vs phi plus in eta bins
  TFileDirectory dirResMassVsPhiPlusInEta = fs->mkdir("TkTkMassVsPhiPlusInEtaBins");
  InvMassVsPhiPlusInEtaBins.bookSet(dirResMassVsPhiPlusInEta, th2d_mass_variables_[Variable::PhiPlus]);
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void DiMuonValidation::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.setComment("Validates alignment payloads by evaluating bias in Z->mm mass distributions");
  desc.addUntracked<int>("compressionSettings", -1);

  desc.add<double>("eBeam", 3500.)->setComment("beam energy in GeV");
  desc.add<std::string>("TkTag", "ALCARECOTkAlZMuMu");

  desc.add<double>("Pair_mass_min", 60.);
  desc.add<double>("Pair_mass_max", 120.);
  desc.add<int>("Pair_mass_nbins", 120);
  desc.add<double>("Pair_etaminpos", -2.4);
  desc.add<double>("Pair_etamaxpos", 2.4);
  desc.add<double>("Pair_etaminneg", -2.4);
  desc.add<double>("Pair_etamaxneg", 2.4);

  desc.add<double>("Variable_CosThetaCS_xmin", -1.);
  desc.add<double>("Variable_CosThetaCS_xmax", 1.);
  desc.add<int>("Variable_CosThetaCS_nbins", 20);

  desc.add<double>("Variable_DeltaEta_xmin", -4.8);
  desc.add<double>("Variable_DeltaEta_xmax", 4.8);
  desc.add<int>("Variable_DeltaEta_nbins", 20);

  desc.add<double>("Variable_EtaMinus_xmin", -2.4);
  desc.add<double>("Variable_EtaMinus_xmax", 2.4);
  desc.add<int>("Variable_EtaMinus_nbins", 12);

  desc.add<double>("Variable_EtaPlus_xmin", -2.4);
  desc.add<double>("Variable_EtaPlus_xmax", 2.4);
  desc.add<int>("Variable_EtaPlus_nbins", 12);

  desc.add<double>("Variable_PhiCS_xmin", -M_PI / 2);
  desc.add<double>("Variable_PhiCS_xmax", M_PI / 2);
  desc.add<int>("Variable_PhiCS_nbins", 20);

  desc.add<double>("Variable_PhiMinus_xmin", -M_PI);
  desc.add<double>("Variable_PhiMinus_xmax", M_PI);
  desc.add<int>("Variable_PhiMinus_nbins", 16);

  desc.add<double>("Variable_PhiPlus_xmin", -M_PI);
  desc.add<double>("Variable_PhiPlus_xmax", M_PI);
  desc.add<int>("Variable_PhiPlus_nbins", 16);

  desc.add<double>("Variable_PairPt_xmin", 0.);
  desc.add<double>("Variable_PairPt_xmax", 100.);
  desc.add<int>("Variable_PairPt_nbins", 100);

  descriptions.addWithDefaultLabel(desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(DiMuonValidation);
-- dummy change --
