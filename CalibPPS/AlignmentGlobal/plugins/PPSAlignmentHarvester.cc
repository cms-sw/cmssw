/****************************************************************************
 *
 *  CalibPPS/AlignmentGlobal/plugins/PPSAlignmentHarvester.cc
 *
 *  Description : PPS Alignment DQM harvester
 *
 *  Authors:
 *  - Jan Kašpar
 *  - Mateusz Kocot
 *
 ****************************************************************************/

#include "DQMServices/Core/interface/DQMEDHarvester.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "CondFormats/PPSObjects/interface/CTPPSRPAlignmentCorrectionData.h"
#include "CondFormats/PPSObjects/interface/CTPPSRPAlignmentCorrectionsData.h"

#include "CondFormats/PPSObjects/interface/PPSAlignmentConfig.h"
#include "CondFormats/DataRecord/interface/PPSAlignmentConfigRcd.h"

#include <map>
#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <cmath>

#include "TH1D.h"
#include "TH2D.h"
#include "TGraph.h"
#include "TGraphErrors.h"
#include "TF1.h"
#include "TProfile.h"
#include "TFile.h"
#include "TKey.h"
#include "TSystemFile.h"
#include "TSpline.h"
#include "TCanvas.h"

//----------------------------------------------------------------------------------------------------

class PPSAlignmentHarvester : public DQMEDHarvester {
public:
  PPSAlignmentHarvester(const edm::ParameterSet &iConfig);

private:
  void dqmEndJob(DQMStore::IBooker &iBooker, DQMStore::IGetter &iGetter) override;
  void dqmEndRun(DQMStore::IBooker &iBooker,
                 DQMStore::IGetter &iGetter,
                 edm::Run const &iRun,
                 edm::EventSetup const &iSetup) override;

  // ------------ x alignment ------------
  static int fitProfile(TProfile *p,
                        double x_mean,
                        double x_rms,
                        unsigned int fitProfileMinBinEntries,
                        unsigned int fitProfileMinNReasonable,
                        double &sl,
                        double &sl_unc);
  TGraphErrors *buildGraphFromVector(const std::vector<PointErrors> &pv);
  TGraphErrors *buildGraphFromMonitorElements(DQMStore::IGetter &iGetter,
                                              const RPConfig &rpd,
                                              const std::vector<MonitorElement *> &mes,
                                              unsigned int fitProfileMinBinEntries,
                                              unsigned int fitProfileMinNReasonable);
  void doMatch(DQMStore::IBooker &iBooker,
               const PPSAlignmentConfig &cfg,
               const RPConfig &rpd,
               TGraphErrors *g_ref,
               TGraphErrors *g_test,
               const SelectionRange &range_ref,
               double sh_min,
               double sh_max,
               double &sh_best,
               double &sh_best_unc);

  void xAlignment(DQMStore::IBooker &iBooker,
                  DQMStore::IGetter &iGetter,
                  const PPSAlignmentConfig &cfg,
                  const PPSAlignmentConfig &cfg_ref,
                  int seqPos);

  std::map<unsigned int, double> sh_x_map;

  // ------------ x alignment relative ------------
  void xAlignmentRelative(DQMStore::IBooker &iBooker,
                          DQMStore::IGetter &iGetter,
                          const PPSAlignmentConfig &cfg,
                          int seqPos);

  // ------------ y alignment ------------
  static double findMax(TF1 *ff_fit);
  TGraphErrors *buildModeGraph(DQMStore::IBooker &iBooker,
                               MonitorElement *h2_y_vs_x,
                               const PPSAlignmentConfig &cfg,
                               const RPConfig &rpd);

  void yAlignment(DQMStore::IBooker &iBooker, DQMStore::IGetter &iGetter, const PPSAlignmentConfig &cfg, int seqPos);

  // ------------ other member data and methods ------------
  static TH1D *getTH1DFromTGraphErrors(TGraphErrors *graph,
                                       std::string title = "",
                                       std::string labels = "",
                                       int n = -1,
                                       double binWidth = -1.,
                                       double min = -1.);

  edm::ESGetToken<PPSAlignmentConfig, PPSAlignmentConfigRcd> esTokenTest_;
  edm::ESGetToken<PPSAlignmentConfig, PPSAlignmentConfigRcd> esTokenReference_;

  const std::string folder_;
  const bool debug_;
  TFile *debugFile_;
  std::ofstream resultsFile_;
};

// -------------------------------- x alignment methods --------------------------------

// Fits a linear function to a TProfile (similar method in PPSAlignmentConfigESSource).
int PPSAlignmentHarvester::fitProfile(TProfile *p,
                                      double x_mean,
                                      double x_rms,
                                      unsigned int fitProfileMinBinEntries,
                                      unsigned int fitProfileMinNReasonable,
                                      double &sl,
                                      double &sl_unc) {
  unsigned int n_reasonable = 0;
  for (int bi = 1; bi <= p->GetNbinsX(); bi++) {
    if (p->GetBinEntries(bi) < fitProfileMinBinEntries) {
      p->SetBinContent(bi, 0.);
      p->SetBinError(bi, 0.);
    } else {
      n_reasonable++;
    }
  }

  if (n_reasonable < fitProfileMinNReasonable)
    return 1;

  double xMin = x_mean - x_rms, xMax = x_mean + x_rms;

  TF1 *ff_pol1 = new TF1("ff_pol1", "[0] + [1]*x");

  ff_pol1->SetParameter(0., 0.);
  p->Fit(ff_pol1, "Q", "", xMin, xMax);

  sl = ff_pol1->GetParameter(1);
  sl_unc = ff_pol1->GetParError(1);

  return 0;
}

// Builds graph from a vector of points (with errors).
TGraphErrors *PPSAlignmentHarvester::buildGraphFromVector(const std::vector<PointErrors> &pv) {
  TGraphErrors *g = new TGraphErrors();

  for (unsigned int i = 0; i < pv.size(); i++) {
    const auto &p = pv[i];
    g->SetPoint(i, p.x_, p.y_);
    g->SetPointError(i, p.ex_, p.ey_);
  }
  g->Sort();

  return g;
}

// Builds a TGraphErrors from slice plots represented as MonitorElements.
TGraphErrors *PPSAlignmentHarvester::buildGraphFromMonitorElements(DQMStore::IGetter &iGetter,
                                                                   const RPConfig &rpd,
                                                                   const std::vector<MonitorElement *> &mes,
                                                                   unsigned int fitProfileMinBinEntries,
                                                                   unsigned int fitProfileMinNReasonable) {
  TGraphErrors *g = new TGraphErrors();

  for (auto *me : mes) {
    if (me->getName() == "h_y")  // find "h_y"
    {
      // retrieve parent directory
      std::string parentPath = me->getPathname();
      size_t parentPos = parentPath.substr(0, parentPath.size() - 1).find_last_of('/') + 1;
      std::string parentName = parentPath.substr(parentPos);
      size_t d = parentName.find('-');
      const double xMin = std::stod(parentName.substr(0, d));
      const double xMax = std::stod(parentName.substr(d + 1));

      TH1D *h_y = me->getTH1D();

      // collect "p_y_diffFN_vs_y" corresponding to found "h_y"
      auto *p_y_diffFN_vs_y_monitor = iGetter.get(parentPath + "p_y_diffFN_vs_y");
      if (p_y_diffFN_vs_y_monitor == nullptr) {
        edm::LogWarning("PPS") << "[x_alignment] could not find p_y_diffFN_vs_y in: " << parentPath;
        continue;
      }
      TProfile *p_y_diffFN_vs_y = p_y_diffFN_vs_y_monitor->getTProfile();

      double y_cen = h_y->GetMean();
      double y_width = h_y->GetRMS();

      y_cen += rpd.y_cen_add_;
      y_width *= rpd.y_width_mult_;

      double sl = 0., sl_unc = 0.;
      int fr =
          fitProfile(p_y_diffFN_vs_y, y_cen, y_width, fitProfileMinBinEntries, fitProfileMinNReasonable, sl, sl_unc);
      if (fr != 0)
        continue;

      if (debug_)
        p_y_diffFN_vs_y->Write(parentName.c_str());

      int idx = g->GetN();
      g->SetPoint(idx, (xMax + xMin) / 2., sl);
      g->SetPointError(idx, (xMax - xMin) / 2., sl_unc);
    }
  }
  g->Sort();

  return g;
}

// Matches reference data with test data.
void PPSAlignmentHarvester::doMatch(DQMStore::IBooker &iBooker,
                                    const PPSAlignmentConfig &cfg,
                                    const RPConfig &rpd,
                                    TGraphErrors *g_ref,
                                    TGraphErrors *g_test,
                                    const SelectionRange &range_ref,
                                    double sh_min,
                                    double sh_max,
                                    double &sh_best,
                                    double &sh_best_unc) {
  const auto range_test = cfg.alignment_x_meth_o_ranges().at(rpd.id_);

  // print config
  edm::LogInfo("PPS") << std::fixed << std::setprecision(3) << "[x_alignment] "
                      << "ref: x_min = " << range_ref.x_min_ << ", x_max = " << range_ref.x_max_ << "\n"
                      << "test: x_min = " << range_test.x_min_ << ", x_max = " << range_test.x_max_;

  // make spline from g_ref
  TSpline3 *s_ref = new TSpline3("s_ref", g_ref->GetX(), g_ref->GetY(), g_ref->GetN());

  // book match-quality graphs
  TGraph *g_n_points = new TGraph();
  g_n_points->SetName("g_n_points");
  g_n_points->SetTitle(";sh;N");
  TGraph *g_chi_sq = new TGraph();
  g_chi_sq->SetName("g_chi_sq");
  g_chi_sq->SetTitle(";sh;S2");
  TGraph *g_chi_sq_norm = new TGraph();
  g_chi_sq_norm->SetName("g_chi_sq_norm");
  g_chi_sq_norm->SetTitle(";sh;S2 / N");

  // optimalisation variables
  double S2_norm_best = 1E100;

  for (double sh = sh_min; sh <= sh_max; sh += cfg.x_ali_sh_step()) {
    // calculate chi^2
    int n_points = 0;
    double S2 = 0.;

    for (int i = 0; i < g_test->GetN(); ++i) {
      const double x_test = g_test->GetX()[i];
      const double y_test = g_test->GetY()[i];
      const double y_test_unc = g_test->GetErrorY(i);

      const double x_ref = x_test + sh;

      if (x_ref < range_ref.x_min_ || x_ref > range_ref.x_max_ || x_test < range_test.x_min_ ||
          x_test > range_test.x_max_)
        continue;

      const double y_ref = s_ref->Eval(x_ref);

      int js = -1, jg = -1;
      double xs = -1E100, xg = +1E100;
      for (int j = 0; j < g_ref->GetN(); ++j) {
        const double x = g_ref->GetX()[j];
        if (x < x_ref && x > xs) {
          xs = x;
          js = j;
        }
        if (x > x_ref && x < xg) {
          xg = x;
          jg = j;
        }
      }
      if (jg == -1)
        jg = js;

      const double y_ref_unc = (g_ref->GetErrorY(js) + g_ref->GetErrorY(jg)) / 2.;

      n_points++;
      const double S2_inc = pow(y_test - y_ref, 2.) / (y_ref_unc * y_ref_unc + y_test_unc * y_test_unc);
      S2 += S2_inc;
    }

    // update best result
    double S2_norm = S2 / n_points;

    if (S2_norm < S2_norm_best) {
      S2_norm_best = S2_norm;
      sh_best = sh;
    }

    // fill in graphs
    int idx = g_n_points->GetN();
    g_n_points->SetPoint(idx, sh, n_points);
    g_chi_sq->SetPoint(idx, sh, S2);
    g_chi_sq_norm->SetPoint(idx, sh, S2_norm);
  }

  TF1 *ff_pol2 = new TF1("ff_pol2", "[0] + [1]*x + [2]*x*x");

  // determine uncertainty
  double fit_range = cfg.methOUncFitRange();
  g_chi_sq->Fit(ff_pol2, "Q", "", sh_best - fit_range, sh_best + fit_range);
  sh_best_unc = 1. / sqrt(ff_pol2->GetParameter(2));

  // print results
  edm::LogInfo("PPS") << std::fixed << std::setprecision(3) << "[x_alignment] "
                      << "sh_best = (" << sh_best << " +- " << sh_best_unc << ") mm";

  TGraphErrors *g_test_shifted = new TGraphErrors(*g_test);
  for (int i = 0; i < g_test_shifted->GetN(); ++i) {
    g_test_shifted->GetX()[i] += sh_best;
  }

  iBooker.book1DD(
      "h_test_shifted",
      getTH1DFromTGraphErrors(
          g_test_shifted, "test_shifted", ";x (mm);S", rpd.x_slice_n_, rpd.x_slice_w_, rpd.x_slice_min_ + sh_best));

  if (debug_) {
    // save graphs
    g_n_points->Write();
    g_chi_sq->Write();
    g_chi_sq_norm->Write();
    g_test_shifted->SetTitle(";x (mm);S");
    g_test_shifted->Write("g_test_shifted");

    // save results
    TGraph *g_results = new TGraph();
    g_results->SetName("g_results");
    g_results->SetPoint(0, sh_best, sh_best_unc);
    g_results->SetPoint(1, range_ref.x_min_, range_ref.x_max_);
    g_results->SetPoint(2, range_test.x_min_, range_test.x_max_);
    g_results->Write();

    // save debug canvas
    TCanvas *c_cmp = new TCanvas("c_cmp");
    g_ref->SetLineColor(1);
    g_ref->SetName("g_ref");
    g_ref->Draw("apl");

    g_test->SetLineColor(4);
    g_test->SetName("g_test");
    g_test->Draw("pl");

    g_test_shifted->SetLineColor(2);
    g_test_shifted->SetName("g_test_shifted");

    g_test_shifted->Draw("pl");
    c_cmp->Write();

    delete c_cmp;
  }

  // clean up
  delete s_ref;
}

// method o
void PPSAlignmentHarvester::xAlignment(DQMStore::IBooker &iBooker,
                                       DQMStore::IGetter &iGetter,
                                       const PPSAlignmentConfig &cfg,
                                       const PPSAlignmentConfig &cfg_ref,
                                       int seqPos) {
  TDirectory *xAliDir = nullptr;
  if (debug_)
    xAliDir = debugFile_->mkdir((std::to_string(seqPos + 1) + ": x alignment").c_str());

  // prepare results
  CTPPSRPAlignmentCorrectionsData results;

  for (const auto &sdp : {std::make_pair(cfg.sectorConfig45(), cfg_ref.sectorConfig45()),
                          std::make_pair(cfg.sectorConfig56(), cfg_ref.sectorConfig56())}) {
    const auto &sd = sdp.first;
    for (const auto &rpdp : {std::make_pair(sd.rp_F_, sdp.second.rp_F_), std::make_pair(sd.rp_N_, sdp.second.rp_N_)}) {
      const auto &rpd = rpdp.first;

      auto mes_test = iGetter.getAllContents(folder_ + "/worker/" + sd.name_ + "/near_far/x slices, " + rpd.position_);
      if (mes_test.empty()) {
        edm::LogWarning("PPS") << "[x_alignment] " << rpd.name_ << ": could not load mes_test";
        continue;
      }

      TDirectory *rpDir = nullptr;
      if (debug_)
        rpDir = xAliDir->mkdir(rpd.name_.c_str());

      auto vec_ref = cfg_ref.matchingReferencePoints().at(rpd.id_);
      if (vec_ref.empty()) {
        edm::LogInfo("PPS") << "[x_alignment] " << rpd.name_ << ": reference points vector is empty";
        continue;
      }

      TGraphErrors *g_ref = buildGraphFromVector(vec_ref);

      if (debug_)
        gDirectory = rpDir->mkdir("fits_test");
      TGraphErrors *g_test = buildGraphFromMonitorElements(
          iGetter, rpd, mes_test, cfg.fitProfileMinBinEntries(), cfg.fitProfileMinNReasonable());

      // require minimal number of points
      if (g_ref->GetN() < (int)cfg.methOGraphMinN() || g_test->GetN() < (int)cfg.methOGraphMinN()) {
        edm::LogWarning("PPS") << "[x_alignment] " << rpd.name_ << ": insufficient data, skipping (g_ref "
                               << g_ref->GetN() << "/" << cfg.methOGraphMinN() << ", g_test " << g_test->GetN() << "/"
                               << cfg.methOGraphMinN() << ")";
        continue;
      }

      iBooker.setCurrentFolder(folder_ + "/harvester/x alignment/" + rpd.name_);
      iBooker.book1DD(
          "h_ref",
          getTH1DFromTGraphErrors(
              g_ref, "ref", ";x (mm);S", rpdp.second.x_slice_n_, rpdp.second.x_slice_w_, rpdp.second.x_slice_min_));
      iBooker.book1DD(
          "h_test",
          getTH1DFromTGraphErrors(g_test, "test", ";x (mm);S", rpd.x_slice_n_, rpd.x_slice_w_, rpd.x_slice_min_));

      if (debug_) {
        gDirectory = rpDir;
        g_ref->SetTitle(";x (mm);S");
        g_ref->Write("g_ref");
        g_test->SetTitle(";x (mm);S");
        g_test->Write("g_test");
      }

      const auto &shiftRange = cfg_ref.matchingShiftRanges().at(rpd.id_);
      double sh = 0., sh_unc = 0.;
      doMatch(iBooker,
              cfg,
              rpd,
              g_ref,
              g_test,
              cfg_ref.alignment_x_meth_o_ranges().at(rpd.id_),
              shiftRange.x_min_,
              shiftRange.x_max_,
              sh,
              sh_unc);

      CTPPSRPAlignmentCorrectionData rpResult(sh, sh_unc, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.);
      results.setRPCorrection(rpd.id_, rpResult);
      edm::LogInfo("PPS") << std::fixed << std::setprecision(3) << "[x_alignment] "
                          << "Setting sh_x of " << rpd.name_ << " to " << sh;
      sh_x_map[rpd.id_] = sh;
    }
  }

  edm::LogInfo("PPS") << seqPos + 1 << ": x_alignment:\n" << results;

  if (resultsFile_.is_open())
    resultsFile_ << seqPos + 1 << ": x_alignment:\n" << results << "\n\n";
}

// -------------------------------- x alignment relative methods --------------------------------

void PPSAlignmentHarvester::xAlignmentRelative(DQMStore::IBooker &iBooker,
                                               DQMStore::IGetter &iGetter,
                                               const PPSAlignmentConfig &cfg,
                                               int seqPos) {
  TDirectory *xAliRelDir = nullptr;
  if (debug_)
    xAliRelDir = debugFile_->mkdir((std::to_string(seqPos + 1) + ": x_alignment_relative").c_str());

  // prepare results
  CTPPSRPAlignmentCorrectionsData results;
  CTPPSRPAlignmentCorrectionsData results_sl_fix;

  TF1 *ff = new TF1("ff", "[0] + [1]*(x - [2])");
  TF1 *ff_sl_fix = new TF1("ff_sl_fix", "[0] + [1]*(x - [2])");

  // processing
  for (const auto &sd : {cfg.sectorConfig45(), cfg.sectorConfig56()}) {
    TDirectory *sectorDir = nullptr;
    if (debug_) {
      sectorDir = xAliRelDir->mkdir(sd.name_.c_str());
      gDirectory = sectorDir;
    }

    auto *p_x_diffFN_vs_x_N_monitor = iGetter.get(folder_ + "/worker/" + sd.name_ + "/near_far/p_x_diffFN_vs_x_N");
    if (p_x_diffFN_vs_x_N_monitor == nullptr) {
      edm::LogWarning("PPS") << "[x_alignment_relative] " << sd.name_ << ": cannot load data, skipping";
      continue;
    }
    TProfile *p_x_diffFN_vs_x_N = p_x_diffFN_vs_x_N_monitor->getTProfile();

    if (p_x_diffFN_vs_x_N->GetEntries() < cfg.nearFarMinEntries()) {
      edm::LogWarning("PPS") << "[x_alignment_relative] " << sd.name_ << ": insufficient data, skipping (near_far "
                             << p_x_diffFN_vs_x_N->GetEntries() << "/" << cfg.nearFarMinEntries() << ")";
      continue;
    }

    const double xMin = cfg.alignment_x_relative_ranges().at(sd.rp_N_.id_).x_min_;
    const double xMax = cfg.alignment_x_relative_ranges().at(sd.rp_N_.id_).x_max_;

    const double sh_x_N = sh_x_map[sd.rp_N_.id_];
    double slope = sd.slope_;

    ff->SetParameters(0., slope, 0.);
    ff->FixParameter(2, -sh_x_N);
    ff->SetLineColor(2);
    p_x_diffFN_vs_x_N->Fit(ff, "Q", "", xMin, xMax);

    const double a = ff->GetParameter(1), a_unc = ff->GetParError(1);
    const double b = ff->GetParameter(0), b_unc = ff->GetParError(0);

    edm::LogInfo("PPS") << "[x_alignment_relative] " << sd.name_ << ":\n"
                        << std::fixed << std::setprecision(3) << "    x_min = " << xMin << ", x_max = " << xMax << "\n"
                        << "    sh_x_N = " << sh_x_N << ", slope (fix) = " << slope << ", slope (fitted) = " << a;

    CTPPSRPAlignmentCorrectionData rpResult_N(+b / 2., b_unc / 2., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.);
    results.setRPCorrection(sd.rp_N_.id_, rpResult_N);
    CTPPSRPAlignmentCorrectionData rpResult_F(-b / 2., b_unc / 2., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.);
    results.setRPCorrection(sd.rp_F_.id_, rpResult_F);

    ff_sl_fix->SetParameters(0., slope, 0.);
    ff_sl_fix->FixParameter(1, slope);
    ff_sl_fix->FixParameter(2, -sh_x_N);
    ff_sl_fix->SetLineColor(4);
    p_x_diffFN_vs_x_N->Fit(ff_sl_fix, "Q+", "", xMin, xMax);

    const double b_fs = ff_sl_fix->GetParameter(0), b_fs_unc = ff_sl_fix->GetParError(0);

    CTPPSRPAlignmentCorrectionData rpResult_sl_fix_N(+b_fs / 2., b_fs_unc / 2., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.);
    results_sl_fix.setRPCorrection(sd.rp_N_.id_, rpResult_sl_fix_N);
    CTPPSRPAlignmentCorrectionData rpResult_sl_fix_F(-b_fs / 2., b_fs_unc / 2., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.);
    results_sl_fix.setRPCorrection(sd.rp_F_.id_, rpResult_sl_fix_F);

    edm::LogInfo("PPS") << "[x_alignment_relative] " << std::fixed << std::setprecision(3)
                        << "ff: " << ff->GetParameter(0) << " + " << ff->GetParameter(1) << " * (x - "
                        << ff->GetParameter(2) << "), ff_sl_fix: " << ff_sl_fix->GetParameter(0) << " + "
                        << ff_sl_fix->GetParameter(1) << " * (x - " << ff_sl_fix->GetParameter(2) << ")";

    if (debug_) {
      p_x_diffFN_vs_x_N->Write("p_x_diffFN_vs_x_N");

      TGraph *g_results = new TGraph();
      g_results->SetPoint(0, sh_x_N, 0.);
      g_results->SetPoint(1, a, a_unc);
      g_results->SetPoint(2, b, b_unc);
      g_results->SetPoint(3, b_fs, b_fs_unc);
      g_results->Write("g_results");
    }
  }

  // write results
  edm::LogInfo("PPS") << seqPos + 1 << ": x_alignment_relative:\n"
                      << results << seqPos + 1 << ": x_alignment_relative_sl_fix:\n"
                      << results_sl_fix;

  if (resultsFile_.is_open()) {
    resultsFile_ << seqPos + 1 << ": x_alignment_relative:\n" << results << "\n";
    resultsFile_ << seqPos + 1 << ": x_alignment_relative_sl_fix:\n" << results_sl_fix << "\n\n";
  }
}

// -------------------------------- y alignment methods --------------------------------

double PPSAlignmentHarvester::findMax(TF1 *ff_fit) {
  const double mu = ff_fit->GetParameter(1);
  const double si = ff_fit->GetParameter(2);

  // unreasonable fit?
  if (si > 25. || std::fabs(mu) > 100.)
    return 1E100;

  double xMax = 1E100;
  double yMax = -1E100;
  for (double x = mu - si; x <= mu + si; x += 0.001) {
    double y = ff_fit->Eval(x);
    if (y > yMax) {
      xMax = x;
      yMax = y;
    }
  }

  return xMax;
}

TGraphErrors *PPSAlignmentHarvester::buildModeGraph(DQMStore::IBooker &iBooker,
                                                    MonitorElement *h2_y_vs_x,
                                                    const PPSAlignmentConfig &cfg,
                                                    const RPConfig &rpd) {
  TDirectory *d_top = nullptr;
  if (debug_)
    d_top = gDirectory;

  TF1 *ff_fit = new TF1("ff_fit", "[0] * exp(-(x-[1])*(x-[1])/2./[2]/[2]) + [3] + [4]*x");

  TGraphErrors *g_y_mode_vs_x = new TGraphErrors();

  int h_n = h2_y_vs_x->getNbinsX();
  double diff = h2_y_vs_x->getTH2D()->GetXaxis()->GetBinWidth(1) / 2.;
  auto h_mode = iBooker.book1DD("mode",
                                ";x (mm); mode of y (mm)",
                                h_n,
                                h2_y_vs_x->getTH2D()->GetXaxis()->GetBinCenter(1) - diff,
                                h2_y_vs_x->getTH2D()->GetXaxis()->GetBinCenter(h_n) + diff);

  for (int bix = 1; bix <= h_n; bix++) {
    const double x = h2_y_vs_x->getTH2D()->GetXaxis()->GetBinCenter(bix);
    const double x_unc = h2_y_vs_x->getTH2D()->GetXaxis()->GetBinWidth(bix) / 2.;

    char buf[100];
    sprintf(buf, "h_y_x=%.3f", x);
    TH1D *h_y = h2_y_vs_x->getTH2D()->ProjectionY(buf, bix, bix);

    if (h_y->GetEntries() < cfg.multSelProjYMinEntries())
      continue;

    if (debug_) {
      sprintf(buf, "x=%.3f", x);
      gDirectory = d_top->mkdir(buf);
    }

    double conMax = -1.;
    double conMax_x = 0.;
    for (int biy = 1; biy < h_y->GetNbinsX(); biy++) {
      if (h_y->GetBinContent(biy) > conMax) {
        conMax = h_y->GetBinContent(biy);
        conMax_x = h_y->GetBinCenter(biy);
      }
    }

    ff_fit->SetParameters(conMax, conMax_x, h_y->GetRMS() * 0.75, 0., 0.);
    ff_fit->FixParameter(4, 0.);

    double xMin = rpd.x_min_fit_mode_, xMax = rpd.x_max_fit_mode_;
    h_y->Fit(ff_fit, "Q", "", xMin, xMax);

    ff_fit->ReleaseParameter(4);
    double w = std::min(4., 2. * ff_fit->GetParameter(2));
    xMin = ff_fit->GetParameter(1) - w;
    xMax = std::min(rpd.y_max_fit_mode_, ff_fit->GetParameter(1) + w);

    h_y->Fit(ff_fit, "Q", "", xMin, xMax);

    if (debug_)
      h_y->Write("h_y");

    double y_mode = findMax(ff_fit);
    const double y_mode_fit_unc = ff_fit->GetParameter(2) / 10;
    const double y_mode_sys_unc = cfg.y_mode_sys_unc();
    double y_mode_unc = std::sqrt(y_mode_fit_unc * y_mode_fit_unc + y_mode_sys_unc * y_mode_sys_unc);

    const double chiSqThreshold = cfg.chiSqThreshold();

    const bool valid =
        !(std::fabs(y_mode_unc) > cfg.y_mode_unc_max_valid() || std::fabs(y_mode) > cfg.y_mode_max_valid() ||
          ff_fit->GetChisquare() / ff_fit->GetNDF() > chiSqThreshold);

    if (debug_) {
      TGraph *g_data = new TGraph();
      g_data->SetPoint(0, y_mode, y_mode_unc);
      g_data->SetPoint(1, ff_fit->GetChisquare(), ff_fit->GetNDF());
      g_data->SetPoint(2, valid, 0.);
      g_data->Write("g_data");
    }

    if (!valid)
      continue;

    int idx = g_y_mode_vs_x->GetN();
    g_y_mode_vs_x->SetPoint(idx, x, y_mode);
    g_y_mode_vs_x->SetPointError(idx, x_unc, y_mode_unc);

    h_mode->Fill(x, y_mode);
    h_mode->setBinError(bix, y_mode_unc);
  }

  return g_y_mode_vs_x;
}

void PPSAlignmentHarvester::yAlignment(DQMStore::IBooker &iBooker,
                                       DQMStore::IGetter &iGetter,
                                       const PPSAlignmentConfig &cfg,
                                       int seqPos) {
  TDirectory *yAliDir = nullptr;
  if (debug_)
    yAliDir = debugFile_->mkdir((std::to_string(seqPos + 1) + ": y_alignment").c_str());

  // prepare results
  CTPPSRPAlignmentCorrectionsData results;
  CTPPSRPAlignmentCorrectionsData results_sl_fix;

  TF1 *ff = new TF1("ff", "[0] + [1]*(x - [2])");
  TF1 *ff_sl_fix = new TF1("ff_sl_fix", "[0] + [1]*(x - [2])");

  // processing
  for (const auto &sd : {cfg.sectorConfig45(), cfg.sectorConfig56()}) {
    for (const auto &rpd : {sd.rp_F_, sd.rp_N_}) {
      TDirectory *rpDir = nullptr;
      if (debug_) {
        rpDir = yAliDir->mkdir(rpd.name_.c_str());
        gDirectory = rpDir->mkdir("x");
      }

      auto *h2_y_vs_x =
          iGetter.get(folder_ + "/worker/" + sd.name_ + "/multiplicity selection/" + rpd.name_ + "/h2_y_vs_x");

      if (h2_y_vs_x == nullptr) {
        edm::LogWarning("PPS") << "[y_alignment] " << rpd.name_ << ": cannot load data, skipping";
        continue;
      }

      iBooker.setCurrentFolder(folder_ + "/harvester/y alignment/" + rpd.name_);
      auto *g_y_cen_vs_x = buildModeGraph(iBooker, h2_y_vs_x, cfg, rpd);

      if ((unsigned int)g_y_cen_vs_x->GetN() < cfg.modeGraphMinN()) {
        edm::LogWarning("PPS") << "[y_alignment] " << rpd.name_ << ": insufficient data, skipping (mode graph "
                               << g_y_cen_vs_x->GetN() << "/" << cfg.modeGraphMinN() << ")";
        continue;
      }

      const double xMin = cfg.alignment_y_ranges().at(rpd.id_).x_min_;
      const double xMax = cfg.alignment_y_ranges().at(rpd.id_).x_max_;

      const double sh_x = sh_x_map[rpd.id_];
      double slope = rpd.slope_;

      ff->SetParameters(0., 0., 0.);
      ff->FixParameter(2, -sh_x);
      ff->SetLineColor(2);
      g_y_cen_vs_x->Fit(ff, "Q", "", xMin, xMax);

      const double a = ff->GetParameter(1), a_unc = ff->GetParError(1);
      const double b = ff->GetParameter(0), b_unc = ff->GetParError(0);

      edm::LogInfo("PPS") << "[y_alignment] " << rpd.name_ << ":\n"
                          << std::fixed << std::setprecision(3) << "    x_min = " << xMin << ", x_max = " << xMax
                          << "\n"
                          << "    sh_x = " << sh_x << ", slope (fix) = " << slope << ", slope (fitted) = " << a;

      CTPPSRPAlignmentCorrectionData rpResult(0., 0., b, b_unc, 0., 0., 0., 0., 0., 0., 0., 0.);
      results.setRPCorrection(rpd.id_, rpResult);

      ff_sl_fix->SetParameters(0., 0., 0.);
      ff_sl_fix->FixParameter(1, slope);
      ff_sl_fix->FixParameter(2, -sh_x);
      ff_sl_fix->SetLineColor(4);
      g_y_cen_vs_x->Fit(ff_sl_fix, "Q+", "", xMin, xMax);

      const double b_fs = ff_sl_fix->GetParameter(0), b_fs_unc = ff_sl_fix->GetParError(0);

      CTPPSRPAlignmentCorrectionData rpResult_sl_fix(0., 0., b_fs, b_fs_unc, 0., 0., 0., 0., 0., 0., 0., 0.);
      results_sl_fix.setRPCorrection(rpd.id_, rpResult_sl_fix);

      edm::LogInfo("PPS") << "[y_alignment] " << std::fixed << std::setprecision(3) << "ff: " << ff->GetParameter(0)
                          << " + " << ff->GetParameter(1) << " * (x - " << ff->GetParameter(2)
                          << "), ff_sl_fix: " << ff_sl_fix->GetParameter(0) << " + " << ff_sl_fix->GetParameter(1)
                          << " * (x - " << ff_sl_fix->GetParameter(2) << ")";

      if (debug_) {
        gDirectory = rpDir;

        g_y_cen_vs_x->SetTitle(";x (mm); mode of y (mm)");
        g_y_cen_vs_x->Write("g_y_cen_vs_x");

        TGraph *g_results = new TGraph();
        g_results->SetPoint(0, sh_x, 0.);
        g_results->SetPoint(1, a, a_unc);
        g_results->SetPoint(2, b, b_unc);
        g_results->SetPoint(3, b_fs, b_fs_unc);
        g_results->Write("g_results");
      }
    }
  }

  // write results
  edm::LogInfo("PPS") << seqPos + 1 << ": y_alignment:\n"
                      << results << seqPos + 1 << ": y_alignment_sl_fix:\n"
                      << results_sl_fix;

  if (resultsFile_.is_open()) {
    resultsFile_ << seqPos + 1 << ": y_alignment:\n" << results << "\n";
    resultsFile_ << seqPos + 1 << ": y_alignment_sl_fix:\n" << results_sl_fix << "\n\n";
  }
}

// -------------------------------- PPSAlignmentHarvester methods --------------------------------

// Points in TGraph should be sorted (TGraph::Sort())
// if n, binWidth, or min is set to -1, method will find it on its own
TH1D *PPSAlignmentHarvester::getTH1DFromTGraphErrors(
    TGraphErrors *graph, std::string title, std::string labels, int n, double binWidth, double min) {
  TH1D *hist;
  if (n == 0) {
    hist = new TH1D(title.c_str(), labels.c_str(), 0, -10., 10.);
  } else if (n == 1) {
    hist = new TH1D(title.c_str(), labels.c_str(), 1, graph->GetPointX(0) - 5., graph->GetPointX(0) + 5.);
  } else {
    n = n == -1 ? graph->GetN() : n;
    binWidth = binWidth == -1 ? graph->GetPointX(1) - graph->GetPointX(0) : binWidth;
    double diff = binWidth / 2.;
    min = min == -1 ? graph->GetPointX(0) - diff : min;
    double max = min + n * binWidth;
    hist = new TH1D(title.c_str(), labels.c_str(), n, min, max);
  }

  for (int i = 0; i < graph->GetN(); i++) {
    double x, y;
    graph->GetPoint(i, x, y);
    hist->Fill(x, y);
    hist->SetBinError(hist->GetXaxis()->FindBin(x), graph->GetErrorY(i));
  }
  return hist;
}

PPSAlignmentHarvester::PPSAlignmentHarvester(const edm::ParameterSet &iConfig)
    : esTokenTest_(
          esConsumes<PPSAlignmentConfig, PPSAlignmentConfigRcd, edm::Transition::EndRun>(edm::ESInputTag("", ""))),
      esTokenReference_(esConsumes<PPSAlignmentConfig, PPSAlignmentConfigRcd, edm::Transition::EndRun>(
          edm::ESInputTag("", "reference"))),
      folder_(iConfig.getParameter<std::string>("folder")),
      debug_(iConfig.getParameter<bool>("debug")) {}

void PPSAlignmentHarvester::dqmEndJob(DQMStore::IBooker &iBooker, DQMStore::IGetter &iGetter) {}

void PPSAlignmentHarvester::dqmEndRun(DQMStore::IBooker &iBooker,
                                      DQMStore::IGetter &iGetter,
                                      edm::Run const &iRun,
                                      edm::EventSetup const &iSetup) {
  const auto &cfg = iSetup.getData(esTokenTest_);

  const auto &cfg_ref = iSetup.getData(esTokenReference_);

  if (debug_)
    debugFile_ = new TFile("debug_harvester.root", "recreate");

  if (!cfg.resultsDir().empty())
    resultsFile_.open(cfg.resultsDir(), std::ios::out | std::ios::trunc);

  // setting default sh_x values from config
  for (const auto &sd : {cfg.sectorConfig45(), cfg.sectorConfig56()}) {
    for (const auto &rpd : {sd.rp_N_, sd.rp_F_}) {
      edm::LogInfo("PPS") << "[harvester] " << std::fixed << std::setprecision(3) << "Setting sh_x of " << rpd.name_
                          << " to " << rpd.sh_x_;
      sh_x_map[rpd.id_] = rpd.sh_x_;
    }
  }

  for (unsigned int i = 0; i < cfg.sequence().size(); i++) {
    if (cfg.sequence()[i] == "x_alignment")
      xAlignment(iBooker, iGetter, cfg, cfg_ref, i);
    else if (cfg.sequence()[i] == "x_alignment_relative")
      xAlignmentRelative(iBooker, iGetter, cfg, i);
    else if (cfg.sequence()[i] == "y_alignment")
      yAlignment(iBooker, iGetter, cfg, i);
    else
      edm::LogError("PPS") << "[harvester] " << cfg.sequence()[i] << " is a wrong method name.";
  }

  if (debug_)
    delete debugFile_;

  if (resultsFile_.is_open())
    resultsFile_.close();
}

DEFINE_FWK_MODULE(PPSAlignmentHarvester);