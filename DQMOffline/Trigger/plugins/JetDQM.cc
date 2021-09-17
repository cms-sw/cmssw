#include "DQMOffline/Trigger/plugins/JetDQM.h"
//#include "DataFormats/Math/interface/deltaR.h"

JetDQM::JetDQM() = default;

JetDQM::~JetDQM() = default;

void JetDQM::initialise(const edm::ParameterSet& iConfig) {
  jetpt_variable_binning_ =
      iConfig.getParameter<edm::ParameterSet>("histoPSet").getParameter<std::vector<double> >("jetptBinning");
  jet1pt_variable_binning_ =
      iConfig.getParameter<edm::ParameterSet>("histoPSet").getParameter<std::vector<double> >("jet1ptBinning");
  jet2pt_variable_binning_ =
      iConfig.getParameter<edm::ParameterSet>("histoPSet").getParameter<std::vector<double> >("jet2ptBinning");
  mjj_variable_binning_ =
      iConfig.getParameter<edm::ParameterSet>("histoPSet").getParameter<std::vector<double> >("mjjBinning");
  jeteta_binning_ =
      getHistoPSet(iConfig.getParameter<edm::ParameterSet>("histoPSet").getParameter<edm::ParameterSet>("jetetaPSet"));
  detajj_binning_ =
      getHistoPSet(iConfig.getParameter<edm::ParameterSet>("histoPSet").getParameter<edm::ParameterSet>("detajjPSet"));
  dphijj_binning_ =
      getHistoPSet(iConfig.getParameter<edm::ParameterSet>("histoPSet").getParameter<edm::ParameterSet>("dphijjPSet"));
  mindphijmet_binning_ = getHistoPSet(
      iConfig.getParameter<edm::ParameterSet>("histoPSet").getParameter<edm::ParameterSet>("mindphijmetPSet"));
  ls_binning_ =
      getHistoPSet(iConfig.getParameter<edm::ParameterSet>("histoPSet").getParameter<edm::ParameterSet>("jetlsPSet"));
}

void JetDQM::bookHistograms(DQMStore::IBooker& ibooker) {
  std::string histname, histtitle;

  histname = "jet1pt";
  histtitle = "PFJet1 pT";
  bookME(ibooker, jet1ptME_, histname, histtitle, jet1pt_variable_binning_);
  setMETitle(jet1ptME_, "PFJet1 p_{T} [GeV]", "events / [GeV]");

  histname = "jet2pt";
  histtitle = "PFJet2 pT";
  bookME(ibooker, jet2ptME_, histname, histtitle, jet2pt_variable_binning_);
  setMETitle(jet2ptME_, "PFJet2 p_{T} [GeV]", "events / [GeV]");

  histname = "jet1eta";
  histtitle = "PFJet1 eta";
  bookME(ibooker, jet1etaME_, histname, histtitle, jeteta_binning_.nbins, jeteta_binning_.xmin, jeteta_binning_.xmax);
  setMETitle(jet1etaME_, "PFJet1 #eta", "events / [rad]");

  histname = "jet2eta";
  histtitle = "PFJet2 eta";
  bookME(ibooker, jet2etaME_, histname, histtitle, jeteta_binning_.nbins, jeteta_binning_.xmin, jeteta_binning_.xmax);
  setMETitle(jet2etaME_, "PFJet2 #eta", "events / [rad]");

  histname = "cjetpt";
  histtitle = "central PFJet pT";
  bookME(ibooker, cjetptME_, histname, histtitle, jetpt_variable_binning_);
  setMETitle(cjetptME_, "central PFJet p_{T} [GeV]", "events / [GeV]");

  histname = "fjetpt";
  histtitle = "forward PFJet pT";
  bookME(ibooker, fjetptME_, histname, histtitle, jetpt_variable_binning_);
  setMETitle(fjetptME_, "forward PFJet p_{T} [GeV]", "events / [GeV]");

  histname = "cjeteta";
  histtitle = "central PFJet eta";
  bookME(ibooker, cjetetaME_, histname, histtitle, jeteta_binning_.nbins, jeteta_binning_.xmin, jeteta_binning_.xmax);
  setMETitle(cjetetaME_, "central PFJet #eta", "events / [rad]");

  histname = "fjeteta";
  histtitle = "forward PFJet eta";
  bookME(ibooker, fjetetaME_, histname, histtitle, jeteta_binning_.nbins, jeteta_binning_.xmin, jeteta_binning_.xmax);
  setMETitle(fjetetaME_, "forward PFJet #eta", "events / [rad]");

  histname = "mjj";
  histtitle = "PFDiJet M";
  bookME(ibooker, mjjME_, histname, histtitle, mjj_variable_binning_);
  setMETitle(mjjME_, "PFDiJet M [GeV]", "events / [GeV]");

  histname = "detajj";
  histtitle = "PFDiJet DeltaEta";
  bookME(ibooker, detajjME_, histname, histtitle, detajj_binning_.nbins, detajj_binning_.xmin, detajj_binning_.xmax);
  setMETitle(detajjME_, "PFDiJet #Delta#eta", "events / [rad]");

  histname = "dphijj";
  histtitle = "PFDiJet DeltaPhi";
  bookME(ibooker, dphijjME_, histname, histtitle, dphijj_binning_.nbins, dphijj_binning_.xmin, dphijj_binning_.xmax);
  setMETitle(dphijjME_, "PFDiJet #Delta#phi", "events / [rad]");

  histname = "mindphijmet";
  histtitle = "minDeltaPhi(PFJets,MET)";
  bookME(ibooker,
         mindphijmetME_,
         histname,
         histtitle,
         mindphijmet_binning_.nbins,
         mindphijmet_binning_.xmin,
         mindphijmet_binning_.xmax);
  setMETitle(mindphijmetME_, "min#Delta#phi(jets,MET)", "events / [rad]");

  histname = "jet1etaVsLS";
  histtitle = "PFJet1 eta vs LS";
  bookME(ibooker,
         jet1etaVsLS_,
         histname,
         histtitle,
         ls_binning_.nbins,
         ls_binning_.xmin,
         ls_binning_.xmax,
         jeteta_binning_.xmin,
         jeteta_binning_.xmax);
  setMETitle(jet1etaVsLS_, "LS", "PF Jet1 #eta");

  histname = "mjjVsLS";
  histtitle = "PFDiJet M vs LS";
  bookME(ibooker, mjjVsLS_, histname, histtitle, ls_binning_.nbins, ls_binning_.xmin, ls_binning_.xmax, 0, 14000);
  setMETitle(mjjVsLS_, "LS", "PFDiJet M [GeV]");

  histname = "mindphijmetVsLS";
  histtitle = "minDeltaPhi(PFJets,MET) vs LS";
  bookME(ibooker,
         mindphijmetVsLS_,
         histname,
         histtitle,
         ls_binning_.nbins,
         ls_binning_.xmin,
         ls_binning_.xmax,
         mindphijmet_binning_.xmin,
         mindphijmet_binning_.xmax);
  setMETitle(mindphijmetVsLS_, "LS", "min#Delta#phi(jets,MET)");
}

void JetDQM::fillHistograms(const std::vector<reco::PFJet>& jets,
                            const reco::PFMET& pfmet,
                            const int ls,
                            const bool passCond) {
  // filling histograms (denominator)
  if (!jets.empty()) {
    double eta1 = jets[0].eta();
    jet1ptME_.denominator->Fill(jets[0].pt());
    jet1etaME_.denominator->Fill(eta1);
    jet1etaVsLS_.denominator->Fill(ls, eta1);
    if (jets.size() > 1) {
      double eta2 = jets[1].eta();
      jet2ptME_.denominator->Fill(jets[1].pt());
      jet2etaME_.denominator->Fill(eta2);
      if (fabs(eta1) < fabs(eta2)) {
        cjetetaME_.denominator->Fill(eta1);
        fjetetaME_.denominator->Fill(eta2);
        cjetptME_.denominator->Fill(jets[0].pt());
        fjetptME_.denominator->Fill(jets[1].pt());
      } else {
        cjetetaME_.denominator->Fill(eta2);
        fjetetaME_.denominator->Fill(eta1);
        cjetptME_.denominator->Fill(jets[1].pt());
        fjetptME_.denominator->Fill(jets[0].pt());
      }
      double mass = (jets[0].p4() + jets[1].p4()).mass();
      mjjME_.denominator->Fill(mass);
      mjjVsLS_.denominator->Fill(ls, mass);
      detajjME_.denominator->Fill(fabs(eta1 - eta2));
      dphijjME_.denominator->Fill(fabs(reco::deltaPhi(jets[0].phi(), jets[1].phi())));

      double mindphi = fabs(reco::deltaPhi(jets[0].phi(), pfmet.phi()));
      for (unsigned ij(0); ij < jets.size(); ++ij) {
        if (ij > 4)
          break;
        double dphi = fabs(reco::deltaPhi(jets[ij].phi(), pfmet.phi()));
        if (dphi < mindphi)
          mindphi = dphi;
      }

      mindphijmetME_.denominator->Fill(mindphi);
      mindphijmetVsLS_.denominator->Fill(ls, mindphi);
    }
  }  //at least 1 jet

  // applying selection for numerator
  if (passCond) {
    // filling histograms (num_genTriggerEventFlag_)
    if (!jets.empty()) {
      double eta1 = jets[0].eta();
      jet1ptME_.numerator->Fill(jets[0].pt());
      jet1etaME_.numerator->Fill(eta1);
      jet1etaVsLS_.numerator->Fill(ls, eta1);
      if (jets.size() > 1) {
        double eta2 = jets[1].eta();
        jet2ptME_.numerator->Fill(jets[1].pt());
        jet2etaME_.numerator->Fill(eta2);
        if (fabs(eta1) < fabs(eta2)) {
          cjetetaME_.numerator->Fill(eta1);
          fjetetaME_.numerator->Fill(eta2);
          cjetptME_.numerator->Fill(jets[0].pt());
          fjetptME_.numerator->Fill(jets[1].pt());
        } else {
          cjetetaME_.numerator->Fill(eta2);
          fjetetaME_.numerator->Fill(eta1);
          cjetptME_.numerator->Fill(jets[1].pt());
          fjetptME_.numerator->Fill(jets[0].pt());
        }
        double mass = (jets[0].p4() + jets[1].p4()).mass();
        mjjME_.numerator->Fill(mass);
        mjjVsLS_.numerator->Fill(ls, mass);
        detajjME_.numerator->Fill(fabs(eta1 - eta2));
        dphijjME_.numerator->Fill(fabs(reco::deltaPhi(jets[0].phi(), jets[1].phi())));

        double mindphi = fabs(reco::deltaPhi(jets[0].phi(), pfmet.phi()));
        for (unsigned ij(0); ij < jets.size(); ++ij) {
          if (ij > 4)
            break;
          double dphi = fabs(reco::deltaPhi(jets[ij].phi(), pfmet.phi()));
          if (dphi < mindphi)
            mindphi = dphi;
        }

        mindphijmetME_.numerator->Fill(mindphi);
        mindphijmetVsLS_.numerator->Fill(ls, mindphi);
      }
    }  //at least 1 jet
  }
}

void JetDQM::fillJetDescription(edm::ParameterSetDescription& histoPSet) {
  edm::ParameterSetDescription jetetaPSet;
  fillHistoPSetDescription(jetetaPSet);
  histoPSet.add<edm::ParameterSetDescription>("jetetaPSet", jetetaPSet);

  edm::ParameterSetDescription detajjPSet;
  fillHistoPSetDescription(detajjPSet);
  histoPSet.add<edm::ParameterSetDescription>("detajjPSet", detajjPSet);

  edm::ParameterSetDescription dphijjPSet;
  fillHistoPSetDescription(dphijjPSet);
  histoPSet.add<edm::ParameterSetDescription>("dphijjPSet", dphijjPSet);

  edm::ParameterSetDescription mindphijmetPSet;
  fillHistoPSetDescription(mindphijmetPSet);
  histoPSet.add<edm::ParameterSetDescription>("mindphijmetPSet", mindphijmetPSet);

  std::vector<double> bins = {0.,   20.,  40.,  60.,  80.,  100., 120., 140., 160., 180.,  200., 220.,
                              240., 260., 280., 300., 350., 400., 450., 500,  750,  1000., 1500.};
  histoPSet.add<std::vector<double> >("jetptBinning", bins);

  std::vector<double> pt1bins = {0.,   20.,  40.,  60.,  80.,  90.,  100., 110., 120., 130., 140., 150.,  160.,
                                 180., 210., 240., 270., 300., 330., 360., 400., 450., 500., 750., 1000., 1500.};
  histoPSet.add<std::vector<double> >("jet1ptBinning", pt1bins);

  std::vector<double> pt2bins = {0.,   20.,  40.,  45.,  50.,  55.,  60.,  65.,  70.,  80.,  90.,  100.,
                                 110., 120., 150., 180., 210., 240., 270., 300., 350., 400., 1000.};
  histoPSet.add<std::vector<double> >("jet2ptBinning", pt2bins);

  std::vector<double> mjjbins = {0.,  200, 400, 600,  620,  640,  660,  680,  700,  720,  740,  760,  780, 800,
                                 850, 900, 950, 1000, 1200, 1400, 1600, 1800, 2000, 2500, 3000, 4000, 6000};
  histoPSet.add<std::vector<double> >("mjjBinning", mjjbins);

  edm::ParameterSetDescription lsPSet;
  fillHistoLSPSetDescription(lsPSet);
  histoPSet.add<edm::ParameterSetDescription>("jetlsPSet", lsPSet);
}
