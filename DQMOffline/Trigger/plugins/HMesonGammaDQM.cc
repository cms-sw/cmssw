#include "DQMOffline/Trigger/plugins/HMesonGammaDQM.h"
#include "DataFormats/Math/interface/deltaR.h"

HMesonGammaDQM::HMesonGammaDQM() = default;

HMesonGammaDQM::~HMesonGammaDQM() = default;

void HMesonGammaDQM::initialise(const edm::ParameterSet& iConfig) {
  gammapt_variable_binning_ =
      iConfig.getParameter<edm::ParameterSet>("histoPSet").getParameter<std::vector<double> >("gammaptBinning");
  mesonpt_variable_binning_ =
      iConfig.getParameter<edm::ParameterSet>("histoPSet").getParameter<std::vector<double> >("mesonptBinning");
  eta_binning_ =
      getHistoPSet(iConfig.getParameter<edm::ParameterSet>("histoPSet").getParameter<edm::ParameterSet>("hmgetaPSet"));
  ls_binning_ =
      getHistoPSet(iConfig.getParameter<edm::ParameterSet>("histoPSet").getParameter<edm::ParameterSet>("hmglsPSet"));
}

void HMesonGammaDQM::bookHistograms(DQMStore::IBooker& ibooker) {
  std::string histname, histtitle;

  histname = "gammapt";
  histtitle = "Gamma pT";
  bookME(ibooker, gammaptME_, histname, histtitle, gammapt_variable_binning_);
  setMETitle(gammaptME_, "Gamma p_{T} [GeV]", "events / [GeV]");

  histname = "mesonpt";
  histtitle = "Meson pT";
  bookME(ibooker, mesonptME_, histname, histtitle, mesonpt_variable_binning_);
  setMETitle(mesonptME_, "Meson p_{T} [GeV]", "events / [GeV]");

  histname = "gammaeta";
  histtitle = "Gamma eta";
  bookME(ibooker, gammaetaME_, histname, histtitle, eta_binning_.nbins, eta_binning_.xmin, eta_binning_.xmax);
  setMETitle(gammaetaME_, "Gamma #eta", "events");

  histname = "mesoneta";
  histtitle = "Meson eta";
  bookME(ibooker, mesonetaME_, histname, histtitle, eta_binning_.nbins, eta_binning_.xmin, eta_binning_.xmax);
  setMETitle(mesonetaME_, "Meson #eta", "events");

  histname = "gammaetaVsLS";
  histtitle = "Gamma eta vs LS";
  bookME(ibooker,
         gammaetaVsLS_,
         histname,
         histtitle,
         ls_binning_.nbins,
         ls_binning_.xmin,
         ls_binning_.xmax,
         eta_binning_.xmin,
         eta_binning_.xmax);
  setMETitle(gammaetaVsLS_, "LS", "Gamma #eta");
}

void HMesonGammaDQM::fillHistograms(const reco::PhotonCollection& photons,
                                    const std::vector<TLorentzVector>& mesons,
                                    const int ls,
                                    const bool passCond) {
  // filling histograms (denominator)
  if (!photons.empty()) {
    double eta1 = photons[0].eta();
    gammaptME_.denominator->Fill(photons[0].pt());
    gammaetaME_.denominator->Fill(eta1);
    gammaetaVsLS_.denominator->Fill(ls, eta1);
  }
  if (!mesons.empty()) {
    double eta2 = mesons[0].Eta();
    mesonptME_.denominator->Fill(mesons[0].Pt());
    mesonetaME_.denominator->Fill(eta2);
  }

  // applying selection for numerator
  if (passCond) {
    if (!photons.empty()) {
      double eta1 = photons[0].eta();
      gammaptME_.numerator->Fill(photons[0].pt());
      gammaetaME_.numerator->Fill(eta1);
      gammaetaVsLS_.numerator->Fill(ls, eta1);
    }
    if (!mesons.empty()) {
      double eta2 = mesons[0].Eta();
      mesonptME_.numerator->Fill(mesons[0].Pt());
      mesonetaME_.numerator->Fill(eta2);
    }
  }
}

void HMesonGammaDQM::fillHmgDescription(edm::ParameterSetDescription& histoPSet) {
  edm::ParameterSetDescription hmgetaPSet;
  fillHistoPSetDescription(hmgetaPSet);
  histoPSet.add<edm::ParameterSetDescription>("hmgetaPSet", hmgetaPSet);

  std::vector<double> pt1bins = {0.,   20.,  40.,  60.,  80.,  90.,  100., 110., 120., 130., 140., 150.,  160.,
                                 180., 210., 240., 270., 300., 330., 360., 400., 450., 500., 750., 1000., 1500.};
  histoPSet.add<std::vector<double> >("gammaptBinning", pt1bins);

  std::vector<double> pt2bins = {0.,   20.,  40.,  45.,  50.,  55.,  60.,  65.,  70.,  80.,  90.,  100.,
                                 110., 120., 150., 180., 210., 240., 270., 300., 350., 400., 1000.};
  histoPSet.add<std::vector<double> >("mesonptBinning", pt2bins);

  edm::ParameterSetDescription lsPSet;
  fillHistoLSPSetDescription(lsPSet);
  histoPSet.add<edm::ParameterSetDescription>("hmglsPSet", lsPSet);
}
