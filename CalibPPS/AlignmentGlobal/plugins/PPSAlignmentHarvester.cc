/****************************************************************************
* Authors: 
*  Jan Kašpar (jan.kaspar@gmail.com) 
*  Mateusz Kocot (mateuszkocot99@gmail.com)
****************************************************************************/

#include "DQMServices/Core/interface/DQMEDHarvester.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"

#include "DataFormats/CTPPSDetId/interface/CTPPSDetId.h"

#include "CondFormats/PPSObjects/interface/CTPPSRPAlignmentCorrectionData.h"
#include "CondFormats/PPSObjects/interface/CTPPSRPAlignmentCorrectionsData.h"
#include "CondFormats/DataRecord/interface/CTPPSRPAlignmentCorrectionsDataRcd.h"

#include "CondFormats/PPSObjects/interface/PPSAlignmentConfiguration.h"
#include "CondFormats/DataRecord/interface/PPSAlignmentConfigurationRcd.h"

#include "CalibPPS/AlignmentGlobal/interface/utils.h"

#include <memory>
#include <map>
#include <vector>
#include <string>
#include <fstream>
#include <iomanip>
#include <cmath>
#include <utility>
#include <algorithm>

#include "TH1D.h"
#include "TH2D.h"
#include "TGraph.h"
#include "TGraphErrors.h"
#include "TF1.h"
#include "TProfile.h"
#include "TFile.h"
#include "TKey.h"
#include "TSpline.h"
#include "TCanvas.h"

//----------------------------------------------------------------------------------------------------

class PPSAlignmentHarvester : public DQMEDHarvester {
public:
  PPSAlignmentHarvester(const edm::ParameterSet& iConfig);
  ~PPSAlignmentHarvester() override;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void dqmEndJob(DQMStore::IBooker& iBooker, DQMStore::IGetter& iGetter) override;
  void dqmEndRun(DQMStore::IBooker& iBooker,
                 DQMStore::IGetter& iGetter,
                 edm::Run const& iRun,
                 edm::EventSetup const& iSetup) override;

  // ------------ x alignment ------------
  std::unique_ptr<TGraphErrors> buildGraphFromVector(const std::vector<PPSAlignmentConfiguration::PointErrors>& pv);
  std::unique_ptr<TGraphErrors> buildGraphFromMonitorElements(DQMStore::IGetter& iGetter,
                                                              const PPSAlignmentConfiguration::RPConfig& rpc,
                                                              const std::vector<MonitorElement*>& mes,
                                                              const unsigned int fitProfileMinBinEntries,
                                                              const unsigned int fitProfileMinNReasonable);
  void doMatch(DQMStore::IBooker& iBooker,
               const PPSAlignmentConfiguration& cfg,
               const PPSAlignmentConfiguration::RPConfig& rpc,
               TGraphErrors* g_ref,
               TGraphErrors* g_test,
               const PPSAlignmentConfiguration::SelectionRange& range_ref,
               const double sh_min,
               const double sh_max,
               double& sh_best,
               double& sh_best_unc);

  void xAlignment(DQMStore::IBooker& iBooker,
                  DQMStore::IGetter& iGetter,
                  const PPSAlignmentConfiguration& cfg,
                  const PPSAlignmentConfiguration& cfg_ref);

  std::map<unsigned int, double> sh_x_map_;

  // ------------ x alignment relative ------------
  void xAlignmentRelative(DQMStore::IBooker& iBooker, DQMStore::IGetter& iGetter, const PPSAlignmentConfiguration& cfg);

  // ------------ y alignment ------------
  static double findMax(const TF1* ff_fit);
  TH1D* buildModeGraph(DQMStore::IBooker& iBooker,
                       const MonitorElement* h2_y_vs_x,
                       const PPSAlignmentConfiguration& cfg,
                       const PPSAlignmentConfiguration::RPConfig& rpc);

  void yAlignment(DQMStore::IBooker& iBooker, DQMStore::IGetter& iGetter, const PPSAlignmentConfiguration& cfg);

  // ------------ other member data and methods ------------
  static void writeCutPlot(
      TH2D* h, const double a, const double c, const double si, const double n_si, const std::string& label);
  static std::unique_ptr<TH1D> getTH1DFromTGraphErrors(TGraphErrors* graph,
                                                       const std::string& title = "",
                                                       const std::string& labels = "",
                                                       int n = -1,
                                                       double binWidth = -1.,
                                                       double min = -1.);

  CTPPSRPAlignmentCorrectionsData getLongIdResults(CTPPSRPAlignmentCorrectionsData finalResults);

  edm::ESGetToken<PPSAlignmentConfiguration, PPSAlignmentConfigurationRcd> esTokenTest_;
  edm::ESGetToken<PPSAlignmentConfiguration, PPSAlignmentConfigurationRcd> esTokenReference_;

  // variables from parameters
  const std::string dqmDir_;
  const std::vector<std::string> sequence_;
  const bool overwriteShX_;
  const bool writeSQLiteResults_;
  const bool xAliRelFinalSlopeFixed_;
  const bool yAliFinalSlopeFixed_;
  const std::pair<double, double> xCorrRange_;
  const std::pair<double, double> yCorrRange_;
  const unsigned int detectorId_;
  const unsigned int subdetectorId_;
  const bool debug_;

  // other class variables
  std::unique_ptr<TFile> debugFile_;
  std::ofstream textResultsFile_;
  int seqPos = 1;  // position in sequence_

  CTPPSRPAlignmentCorrectionsData xAliResults_;

  CTPPSRPAlignmentCorrectionsData xAliRelResults_;
  CTPPSRPAlignmentCorrectionsData xAliRelResultsSlopeFixed_;

  CTPPSRPAlignmentCorrectionsData yAliResults_;
  CTPPSRPAlignmentCorrectionsData yAliResultsSlopeFixed_;
};

// -------------------------------- DQMEDHarvester methods --------------------------------

PPSAlignmentHarvester::PPSAlignmentHarvester(const edm::ParameterSet& iConfig)
    : esTokenTest_(esConsumes<PPSAlignmentConfiguration, PPSAlignmentConfigurationRcd, edm::Transition::EndRun>(
          edm::ESInputTag("", ""))),
      esTokenReference_(esConsumes<PPSAlignmentConfiguration, PPSAlignmentConfigurationRcd, edm::Transition::EndRun>(
          edm::ESInputTag("", "reference"))),
      dqmDir_(iConfig.getParameter<std::string>("dqm_dir")),
      sequence_(iConfig.getParameter<std::vector<std::string>>("sequence")),
      overwriteShX_(iConfig.getParameter<bool>("overwrite_sh_x")),
      writeSQLiteResults_(iConfig.getParameter<bool>("write_sqlite_results")),
      xAliRelFinalSlopeFixed_(iConfig.getParameter<bool>("x_ali_rel_final_slope_fixed")),
      yAliFinalSlopeFixed_(iConfig.getParameter<bool>("y_ali_final_slope_fixed")),
      xCorrRange_(std::make_pair(iConfig.getParameter<double>("x_corr_min") / 1000.,
                                 iConfig.getParameter<double>("x_corr_max") / 1000.)),  // um -> mm
      yCorrRange_(std::make_pair(iConfig.getParameter<double>("y_corr_min") / 1000.,
                                 iConfig.getParameter<double>("y_corr_max") / 1000.)),  // um -> mm
      detectorId_(iConfig.getParameter<unsigned int>("detector_id")),
      subdetectorId_(iConfig.getParameter<unsigned int>("subdetector_id")),
      debug_(iConfig.getParameter<bool>("debug")) {
  auto textResultsPath = iConfig.getParameter<std::string>("text_results_path");
  if (!textResultsPath.empty()) {
    textResultsFile_.open(textResultsPath, std::ios::out | std::ios::trunc);
  }
  if (debug_) {
    debugFile_ = std::make_unique<TFile>("debug_harvester.root", "recreate");
  }

  edm::LogInfo("PPSAlignmentHarvester").log([&](auto& li) {
    li << "parameters:\n";
    li << "* dqm_dir: " << dqmDir_ << "\n";
    li << "* sequence:\n";
    for (unsigned int i = 0; i < sequence_.size(); i++) {
      li << "    " << i + 1 << ": " << sequence_[i] << "\n";
    }
    li << "* overwrite_sh_x: " << std::boolalpha << overwriteShX_ << "\n";
    li << "* text_results_path: " << textResultsPath << "\n";
    li << "* write_sqlite_results: " << std::boolalpha << writeSQLiteResults_ << "\n";
    li << "* x_ali_rel_final_slope_fixed: " << std::boolalpha << xAliRelFinalSlopeFixed_ << "\n";
    li << "* y_ali_final_slope_fixed: " << std::boolalpha << yAliFinalSlopeFixed_ << "\n";
    // print in um
    li << "* x_corr_min: " << std::fixed << xCorrRange_.first * 1000. << ", x_corr_max: " << xCorrRange_.second * 1000.
       << "\n";
    // print in um
    li << "* y_corr_min: " << std::fixed << yCorrRange_.first * 1000. << ", y_corr_max: " << yCorrRange_.second * 1000.
       << "\n";
    li << "* detector_id: " << detectorId_ << "\n";
    li << "* subdetector_id: " << subdetectorId_ << "\n";
    li << "* debug: " << std::boolalpha << debug_;
  });
}

PPSAlignmentHarvester::~PPSAlignmentHarvester() {
  if (textResultsFile_.is_open()) {
    textResultsFile_.close();
  }
}

void PPSAlignmentHarvester::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;

  desc.add<std::string>("dqm_dir", "AlCaReco/PPSAlignment");
  desc.add<std::vector<std::string>>("sequence", {"x_alignment", "x_alignment_relative", "y_alignment"});
  desc.add<bool>("overwrite_sh_x", true);
  desc.add<std::string>("text_results_path", "./alignment_results.txt");
  desc.add<bool>("write_sqlite_results", false);
  desc.add<bool>("x_ali_rel_final_slope_fixed", true);
  desc.add<bool>("y_ali_final_slope_fixed", true);
  desc.add<double>("x_corr_min", -1'000'000.);
  desc.add<double>("x_corr_max", 1'000'000.);
  desc.add<double>("y_corr_min", -1'000'000.);
  desc.add<double>("y_corr_max", 1'000'000.);
  desc.add<unsigned int>("detector_id", 7);
  desc.add<unsigned int>("subdetector_id", 4);
  desc.add<bool>("debug", false);

  descriptions.addWithDefaultLabel(desc);
}

void PPSAlignmentHarvester::dqmEndJob(DQMStore::IBooker& iBooker, DQMStore::IGetter& iGetter) {}

void PPSAlignmentHarvester::dqmEndRun(DQMStore::IBooker& iBooker,
                                      DQMStore::IGetter& iGetter,
                                      edm::Run const& iRun,
                                      edm::EventSetup const& iSetup) {
  const auto& cfg = iSetup.getData(esTokenTest_);

  const auto& cfg_ref = iSetup.getData(esTokenReference_);

  // setting default sh_x values from config
  for (const auto& sc : {cfg.sectorConfig45(), cfg.sectorConfig56()}) {
    for (const auto& rpc : {sc.rp_N_, sc.rp_F_}) {
      sh_x_map_[rpc.id_] = rpc.sh_x_;
    }
  }
  edm::LogInfo("PPSAlignmentHarvester").log([&](auto& li) {
    li << "Setting sh_x from config of:\n";
    for (const auto& sc : {cfg.sectorConfig45(), cfg.sectorConfig56()}) {
      for (const auto& rpc : {sc.rp_N_, sc.rp_F_}) {
        li << "    " << rpc.name_ << " to " << std::fixed << std::setprecision(3) << rpc.sh_x_;
        if (rpc.name_ != "R_2_F")
          li << "\n";
      }
    }
  });

  bool doXAli = false, doXAliRel = false, doYAli = false;
  for (const std::string& aliMethod : sequence_) {
    if (aliMethod == "x_alignment") {
      xAlignment(iBooker, iGetter, cfg, cfg_ref);
      doXAli = true;
    } else if (aliMethod == "x_alignment_relative") {
      xAlignmentRelative(iBooker, iGetter, cfg);
      doXAliRel = true;
    } else if (aliMethod == "y_alignment") {
      yAlignment(iBooker, iGetter, cfg);
      doYAli = true;
    } else
      edm::LogError("PPSAlignmentHarvester") << aliMethod << " is a wrong method name.";
    seqPos++;
  }

  // merge results from all the specified methods
  CTPPSRPAlignmentCorrectionsData finalResults;
  if (doXAli) {  // x alignment
    finalResults.addCorrections(xAliResults_);
    if (doXAliRel) {  // merge with x alignment relative
      for (const auto& sc : {cfg.sectorConfig45(), cfg.sectorConfig56()}) {
        // extract shifts
        double d_x_N = xAliResults_.getRPCorrection(sc.rp_N_.id_).getShX();
        double d_x_F = xAliResults_.getRPCorrection(sc.rp_F_.id_).getShX();

        double d_x_rel_N, d_x_rel_F;
        if (xAliRelFinalSlopeFixed_) {
          d_x_rel_N = xAliRelResultsSlopeFixed_.getRPCorrection(sc.rp_N_.id_).getShX();
          d_x_rel_F = xAliRelResultsSlopeFixed_.getRPCorrection(sc.rp_F_.id_).getShX();
        } else {
          d_x_rel_N = xAliRelResults_.getRPCorrection(sc.rp_N_.id_).getShX();
          d_x_rel_F = xAliRelResults_.getRPCorrection(sc.rp_F_.id_).getShX();
        }

        // merge the results
        double b = d_x_rel_N - d_x_rel_F;
        double xCorrRel = b + d_x_F - d_x_N;

        CTPPSRPAlignmentCorrectionData corrRelN(xCorrRel / 2., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.);
        finalResults.addRPCorrection(sc.rp_N_.id_, corrRelN);
        CTPPSRPAlignmentCorrectionData corrRelF(-xCorrRel / 2., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.);
        finalResults.addRPCorrection(sc.rp_F_.id_, corrRelF);
      }
    }
  }
  if (doYAli) {  // y alignment
    if (yAliFinalSlopeFixed_) {
      finalResults.addCorrections(yAliResultsSlopeFixed_);
    } else {
      finalResults.addCorrections(yAliResults_);
    }
  }

  // check if the results are within the reasonability ranges xCorrRange and yCorrRange
  for (const auto& sc : {cfg.sectorConfig45(), cfg.sectorConfig56()}) {
    for (const auto& rpc : {sc.rp_F_, sc.rp_N_}) {
      auto& rpResults = finalResults.getRPCorrection(rpc.id_);

      if (!(xCorrRange_.first <= rpResults.getShX() && rpResults.getShX() <= xCorrRange_.second)) {
        edm::LogWarning("PPSAlignmentHarvester")
            << "The horizontal shift of " << rpc.name_ << " (" << std::fixed << std::setw(9) << std::setprecision(1)
            << rpResults.getShX() * 1000. << " um) outside of the reasonability range. Setting it to 0.";
        rpResults.setShX(0.);
        rpResults.setShXUnc(0.);
      }

      if (!(yCorrRange_.first <= rpResults.getShY() && rpResults.getShY() <= yCorrRange_.second)) {
        edm::LogWarning("PPSAlignmentHarvester")
            << "The vertical shift of " << rpc.name_ << " (" << std::fixed << std::setw(9) << std::setprecision(1)
            << rpResults.getShY() * 1000. << " um) outside of the reasonability range. Setting it to 0.";
        rpResults.setShY(0.);
        rpResults.setShYUnc(0.);
      }
    }
  }

  // print the text results
  edm::LogInfo("PPSAlignmentHarvester") << "final merged results:\n" << finalResults;

  if (textResultsFile_.is_open()) {
    textResultsFile_ << "final merged results:\n" << finalResults;
  }

  // if requested, store the results in a DB object
  if (writeSQLiteResults_) {
    CTPPSRPAlignmentCorrectionsData longIdFinalResults = getLongIdResults(finalResults);
    edm::LogInfo("PPSAlignmentHarvester") << "trying to store final merged results with long ids:\n"
                                          << longIdFinalResults;

    edm::Service<cond::service::PoolDBOutputService> poolDbService;
    if (poolDbService.isAvailable()) {
      poolDbService->writeOneIOV(
          longIdFinalResults, poolDbService->currentTime(), "CTPPSRPAlignmentCorrectionsDataRcd");
    } else {
      edm::LogWarning("PPSAlignmentHarvester")
          << "Could not store the results in a DB object. PoolDBService not available.";
    }
  }

  // if debug_, save nice-looking cut plots with the worker data in the debug ROOT file
  if (debug_) {
    TDirectory* cutsDir = debugFile_->mkdir("cuts");
    for (const auto& sc : {cfg.sectorConfig45(), cfg.sectorConfig56()}) {
      TDirectory* sectorDir = cutsDir->mkdir(sc.name_.c_str());

      gDirectory = sectorDir->mkdir("cut_h");
      auto* h2_cut_h_bef_monitor = iGetter.get(dqmDir_ + "/worker/" + sc.name_ + "/cuts/cut_h/h2_cut_h_bef");
      auto* h2_cut_h_aft_monitor = iGetter.get(dqmDir_ + "/worker/" + sc.name_ + "/cuts/cut_h/h2_cut_h_aft");
      writeCutPlot(
          h2_cut_h_bef_monitor->getTH2D(), sc.cut_h_a_, sc.cut_h_c_, cfg.n_si(), sc.cut_h_si_, "canvas_before");
      writeCutPlot(h2_cut_h_aft_monitor->getTH2D(), sc.cut_h_a_, sc.cut_h_c_, cfg.n_si(), sc.cut_h_si_, "canvas_after");

      gDirectory = sectorDir->mkdir("cut_v");
      auto* h2_cut_v_bef_monitor = iGetter.get(dqmDir_ + "/worker/" + sc.name_ + "/cuts/cut_v/h2_cut_v_bef");
      auto* h2_cut_v_aft_monitor = iGetter.get(dqmDir_ + "/worker/" + sc.name_ + "/cuts/cut_v/h2_cut_v_aft");
      writeCutPlot(
          h2_cut_v_bef_monitor->getTH2D(), sc.cut_v_a_, sc.cut_v_c_, cfg.n_si(), sc.cut_v_si_, "canvas_before");
      writeCutPlot(h2_cut_v_aft_monitor->getTH2D(), sc.cut_v_a_, sc.cut_v_c_, cfg.n_si(), sc.cut_v_si_, "canvas_after");
    }
  }
}

// -------------------------------- x alignment methods --------------------------------

// Builds graph from a vector of points (with errors).
std::unique_ptr<TGraphErrors> PPSAlignmentHarvester::buildGraphFromVector(
    const std::vector<PPSAlignmentConfiguration::PointErrors>& pv) {
  auto g = std::make_unique<TGraphErrors>();

  for (unsigned int i = 0; i < pv.size(); i++) {
    const auto& p = pv[i];
    g->SetPoint(i, p.x_, p.y_);
    g->SetPointError(i, p.ex_, p.ey_);
  }
  g->Sort();

  return g;
}

// Builds a TGraphErrors from slice plots represented as MonitorElements.
std::unique_ptr<TGraphErrors> PPSAlignmentHarvester::buildGraphFromMonitorElements(
    DQMStore::IGetter& iGetter,
    const PPSAlignmentConfiguration::RPConfig& rpc,
    const std::vector<MonitorElement*>& mes,
    const unsigned int fitProfileMinBinEntries,
    const unsigned int fitProfileMinNReasonable) {
  auto g = std::make_unique<TGraphErrors>();

  for (auto* me : mes) {
    if (me->getName() == "h_y")  // find "h_y"
    {
      // retrieve parent directory
      std::string parentPath = me->getPathname();
      size_t parentPos = parentPath.substr(0, parentPath.size() - 1).find_last_of('/') + 1;
      std::string parentName = parentPath.substr(parentPos);
      std::replace(parentName.begin(), parentName.end(), '_', '.');  // replace _ with .
      size_t d = parentName.find('-');
      const double x_min = std::stod(parentName.substr(0, d));
      const double x_max = std::stod(parentName.substr(d + 1));

      TH1D* h_y = me->getTH1D();

      // collect "p_y_diffFN_vs_y" corresponding to found "h_y"
      auto* p_y_diffFN_vs_y_monitor = iGetter.get(parentPath + "p_y_diffFN_vs_y");
      if (p_y_diffFN_vs_y_monitor == nullptr) {
        edm::LogWarning("PPSAlignmentHarvester") << "[x_alignment] could not find p_y_diffFN_vs_y in: " << parentPath;
        continue;
      }
      TProfile* p_y_diffFN_vs_y = p_y_diffFN_vs_y_monitor->getTProfile();

      double y_cen = h_y->GetMean() + rpc.y_cen_add_;
      double y_width = h_y->GetRMS() * rpc.y_width_mult_;

      double sl, sl_unc;
      int fr = alig_utils::fitProfile(
          p_y_diffFN_vs_y, y_cen, y_width, fitProfileMinBinEntries, fitProfileMinNReasonable, sl, sl_unc);
      if (fr != 0)
        continue;

      if (debug_)
        p_y_diffFN_vs_y->Write(parentName.c_str());

      int idx = g->GetN();
      g->SetPoint(idx, (x_max + x_min) / 2., sl);
      g->SetPointError(idx, (x_max - x_min) / 2., sl_unc);
    }
  }
  g->Sort();

  return g;
}

// Matches reference data with test data.
void PPSAlignmentHarvester::doMatch(DQMStore::IBooker& iBooker,
                                    const PPSAlignmentConfiguration& cfg,
                                    const PPSAlignmentConfiguration::RPConfig& rpc,
                                    TGraphErrors* g_ref,
                                    TGraphErrors* g_test,
                                    const PPSAlignmentConfiguration::SelectionRange& range_ref,
                                    const double sh_min,
                                    const double sh_max,
                                    double& sh_best,
                                    double& sh_best_unc) {
  const auto range_test = cfg.alignment_x_meth_o_ranges().at(rpc.id_);

  // print config
  edm::LogInfo("PPSAlignmentHarvester") << std::fixed << std::setprecision(3) << "[x_alignment] "
                                        << "ref: x_min = " << range_ref.x_min_ << ", x_max = " << range_ref.x_max_
                                        << "\n"
                                        << "test: x_min = " << range_test.x_min_ << ", x_max = " << range_test.x_max_;

  // make spline from g_ref
  auto s_ref = std::make_unique<TSpline3>("s_ref", g_ref->GetX(), g_ref->GetY(), g_ref->GetN());

  // book match-quality graphs
  auto g_n_points = std::make_unique<TGraph>();
  g_n_points->SetName("g_n_points");
  g_n_points->SetTitle(";sh;N");
  auto g_chi_sq = std::make_unique<TGraph>();
  g_chi_sq->SetName("g_chi_sq");
  g_chi_sq->SetTitle(";sh;S2");
  auto g_chi_sq_norm = std::make_unique<TGraph>();
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

  auto ff_pol2 = std::make_unique<TF1>("ff_pol2", "[0] + [1]*x + [2]*x*x");

  // determine uncertainty
  double fit_range = cfg.methOUncFitRange();
  g_chi_sq->Fit(ff_pol2.get(), "Q", "", sh_best - fit_range, sh_best + fit_range);
  sh_best_unc = 1. / sqrt(ff_pol2->GetParameter(2));

  // print results
  edm::LogInfo("PPSAlignmentHarvester") << std::fixed << std::setprecision(3) << "[x_alignment] "
                                        << "sh_best = (" << sh_best << " +- " << sh_best_unc << ") mm";

  auto g_test_shifted = std::make_unique<TGraphErrors>(*g_test);
  for (int i = 0; i < g_test_shifted->GetN(); ++i) {
    g_test_shifted->GetX()[i] += sh_best;
  }

  std::unique_ptr<TH1D> histPtr = getTH1DFromTGraphErrors(
      g_test_shifted.get(), "test_shifted", ";x (mm);S", rpc.x_slice_n_, rpc.x_slice_w_, rpc.x_slice_min_ + sh_best);
  iBooker.book1DD("h_test_shifted", histPtr.get());

  if (debug_) {
    // save graphs
    g_n_points->Write();
    g_chi_sq->Write();
    g_chi_sq_norm->Write();
    g_test_shifted->SetTitle(";x (mm);S");
    g_test_shifted->Write("g_test_shifted");

    // save results
    auto g_results = std::make_unique<TGraph>();
    g_results->SetName("g_results");
    g_results->SetPoint(0, sh_best, sh_best_unc);
    g_results->SetPoint(1, range_ref.x_min_, range_ref.x_max_);
    g_results->SetPoint(2, range_test.x_min_, range_test.x_max_);
    g_results->Write();

    // save debug canvas
    auto c_cmp = std::make_unique<TCanvas>("c_cmp");
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
  }
}

// method o
void PPSAlignmentHarvester::xAlignment(DQMStore::IBooker& iBooker,
                                       DQMStore::IGetter& iGetter,
                                       const PPSAlignmentConfiguration& cfg,
                                       const PPSAlignmentConfiguration& cfg_ref) {
  TDirectory* xAliDir = nullptr;
  if (debug_)
    xAliDir = debugFile_->mkdir((std::to_string(seqPos) + ": x alignment").c_str());

  for (const auto& [sc, sc_ref] : {std::make_pair(cfg.sectorConfig45(), cfg_ref.sectorConfig45()),
                                   std::make_pair(cfg.sectorConfig56(), cfg_ref.sectorConfig56())}) {
    for (const auto& [rpc, rpc_ref] :
         {std::make_pair(sc.rp_F_, sc_ref.rp_F_), std::make_pair(sc.rp_N_, sc_ref.rp_N_)}) {
      auto mes_test = iGetter.getAllContents(dqmDir_ + "/worker/" + sc.name_ + "/near_far/x slices " + rpc.position_);
      if (mes_test.empty()) {
        edm::LogWarning("PPSAlignmentHarvester") << "[x_alignment] " << rpc.name_ << ": could not load mes_test";
        continue;
      }

      TDirectory* rpDir = nullptr;
      if (debug_)
        rpDir = xAliDir->mkdir(rpc.name_.c_str());

      auto vec_ref = cfg_ref.matchingReferencePoints().at(rpc.id_);
      if (vec_ref.empty()) {
        edm::LogInfo("PPSAlignmentHarvester") << "[x_alignment] " << rpc.name_ << ": reference points vector is empty";
        continue;
      }

      std::unique_ptr<TGraphErrors> g_ref = buildGraphFromVector(vec_ref);

      if (debug_)
        gDirectory = rpDir->mkdir("fits_test");
      std::unique_ptr<TGraphErrors> g_test = buildGraphFromMonitorElements(
          iGetter, rpc, mes_test, cfg.fitProfileMinBinEntries(), cfg.fitProfileMinNReasonable());

      // require minimal number of points
      if (g_ref->GetN() < (int)cfg.methOGraphMinN() || g_test->GetN() < (int)cfg.methOGraphMinN()) {
        edm::LogWarning("PPSAlignmentHarvester")
            << "[x_alignment] " << rpc.name_ << ": insufficient data, skipping (g_ref " << g_ref->GetN() << "/"
            << cfg.methOGraphMinN() << ", g_test " << g_test->GetN() << "/" << cfg.methOGraphMinN() << ")";
        continue;
      }

      iBooker.setCurrentFolder(dqmDir_ + "/harvester/x alignment/" + rpc.name_);

      std::unique_ptr<TH1D> histPtr = getTH1DFromTGraphErrors(
          g_ref.get(), "ref", ";x (mm);S", rpc_ref.x_slice_n_, rpc_ref.x_slice_w_, rpc_ref.x_slice_min_);
      iBooker.book1DD("h_ref", histPtr.get());

      histPtr =
          getTH1DFromTGraphErrors(g_test.get(), "test", ";x (mm);S", rpc.x_slice_n_, rpc.x_slice_w_, rpc.x_slice_min_);
      iBooker.book1DD("h_test", histPtr.get());

      if (debug_) {
        gDirectory = rpDir;
        g_ref->SetTitle(";x (mm);S");
        g_ref->Write("g_ref");
        g_test->SetTitle(";x (mm);S");
        g_test->Write("g_test");
      }

      const auto& shiftRange = cfg.matchingShiftRanges().at(rpc.id_);
      double sh = 0., sh_unc = 0.;

      // matching
      doMatch(iBooker,
              cfg,
              rpc,
              g_ref.get(),
              g_test.get(),
              cfg_ref.alignment_x_meth_o_ranges().at(rpc.id_),
              shiftRange.x_min_,
              shiftRange.x_max_,
              sh,
              sh_unc);

      // save the results
      CTPPSRPAlignmentCorrectionData rpResult(sh, sh_unc, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.);
      xAliResults_.setRPCorrection(rpc.id_, rpResult);
      edm::LogInfo("PPSAlignmentHarvester") << std::fixed << std::setprecision(3) << "[x_alignment] "
                                            << "Setting sh_x of " << rpc.name_ << " to " << sh;

      // update the shift
      if (overwriteShX_) {
        sh_x_map_[rpc.id_] = sh;
      }
    }
  }

  edm::LogInfo("PPSAlignmentHarvester") << seqPos << ": x_alignment:\n" << xAliResults_;

  if (textResultsFile_.is_open())
    textResultsFile_ << seqPos << ": x_alignment:\n" << xAliResults_ << "\n\n";
}

// -------------------------------- x alignment relative methods --------------------------------

void PPSAlignmentHarvester::xAlignmentRelative(DQMStore::IBooker& iBooker,
                                               DQMStore::IGetter& iGetter,
                                               const PPSAlignmentConfiguration& cfg) {
  TDirectory* xAliRelDir = nullptr;
  if (debug_)
    xAliRelDir = debugFile_->mkdir((std::to_string(seqPos) + ": x_alignment_relative").c_str());

  auto ff = std::make_unique<TF1>("ff", "[0] + [1]*(x - [2])");
  auto ff_sl_fix = std::make_unique<TF1>("ff_sl_fix", "[0] + [1]*(x - [2])");

  // processing
  for (const auto& sc : {cfg.sectorConfig45(), cfg.sectorConfig56()}) {
    TDirectory* sectorDir = nullptr;
    if (debug_) {
      sectorDir = xAliRelDir->mkdir(sc.name_.c_str());
      gDirectory = sectorDir;
    }

    auto* p_x_diffFN_vs_x_N_monitor = iGetter.get(dqmDir_ + "/worker/" + sc.name_ + "/near_far/p_x_diffFN_vs_x_N");
    if (p_x_diffFN_vs_x_N_monitor == nullptr) {
      edm::LogWarning("PPSAlignmentHarvester")
          << "[x_alignment_relative] " << sc.name_ << ": cannot load data, skipping";
      continue;
    }
    TProfile* p_x_diffFN_vs_x_N = p_x_diffFN_vs_x_N_monitor->getTProfile();

    if (p_x_diffFN_vs_x_N->GetEntries() < cfg.nearFarMinEntries()) {
      edm::LogWarning("PPSAlignmentHarvester")
          << "[x_alignment_relative] " << sc.name_ << ": insufficient data, skipping (near_far "
          << p_x_diffFN_vs_x_N->GetEntries() << "/" << cfg.nearFarMinEntries() << ")";
      continue;
    }

    const double x_min = cfg.alignment_x_relative_ranges().at(sc.rp_N_.id_).x_min_;
    const double x_max = cfg.alignment_x_relative_ranges().at(sc.rp_N_.id_).x_max_;

    const double sh_x_N = sh_x_map_[sc.rp_N_.id_];
    double slope = sc.slope_;

    // calculate the results without slope fixed
    ff->SetParameters(0., slope, 0.);
    ff->FixParameter(2, -sh_x_N);
    ff->SetLineColor(2);
    p_x_diffFN_vs_x_N->Fit(ff.get(), "Q", "", x_min, x_max);

    const double a = ff->GetParameter(1), a_unc = ff->GetParError(1);
    const double b = ff->GetParameter(0), b_unc = ff->GetParError(0);

    edm::LogInfo("PPSAlignmentHarvester")
        << "[x_alignment_relative] " << sc.name_ << ":\n"
        << std::fixed << std::setprecision(3) << "    x_min = " << x_min << ", x_max = " << x_max << "\n"
        << "    sh_x_N = " << sh_x_N << ", slope (fix) = " << slope << ", slope (fitted) = " << a;

    CTPPSRPAlignmentCorrectionData rpResult_N(+b / 2., b_unc / 2., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.);
    xAliRelResults_.setRPCorrection(sc.rp_N_.id_, rpResult_N);
    CTPPSRPAlignmentCorrectionData rpResult_F(-b / 2., b_unc / 2., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.);
    xAliRelResults_.setRPCorrection(sc.rp_F_.id_, rpResult_F);

    // calculate the results with slope fixed
    ff_sl_fix->SetParameters(0., slope, 0.);
    ff_sl_fix->FixParameter(1, slope);
    ff_sl_fix->FixParameter(2, -sh_x_N);
    ff_sl_fix->SetLineColor(4);
    p_x_diffFN_vs_x_N->Fit(ff_sl_fix.get(), "Q+", "", x_min, x_max);

    const double b_fs = ff_sl_fix->GetParameter(0), b_fs_unc = ff_sl_fix->GetParError(0);

    CTPPSRPAlignmentCorrectionData rpResult_sl_fix_N(+b_fs / 2., b_fs_unc / 2., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.);
    xAliRelResultsSlopeFixed_.setRPCorrection(sc.rp_N_.id_, rpResult_sl_fix_N);
    CTPPSRPAlignmentCorrectionData rpResult_sl_fix_F(-b_fs / 2., b_fs_unc / 2., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.);
    xAliRelResultsSlopeFixed_.setRPCorrection(sc.rp_F_.id_, rpResult_sl_fix_F);

    edm::LogInfo("PPSAlignmentHarvester")
        << "[x_alignment_relative] " << std::fixed << std::setprecision(3) << "ff: " << ff->GetParameter(0) << " + "
        << ff->GetParameter(1) << " * (x - " << ff->GetParameter(2) << "), ff_sl_fix: " << ff_sl_fix->GetParameter(0)
        << " + " << ff_sl_fix->GetParameter(1) << " * (x - " << ff_sl_fix->GetParameter(2) << ")";

    // rebook the diffFN plot in the harvester
    iBooker.setCurrentFolder(dqmDir_ + "/harvester/x_alignment_relative/" + sc.name_);
    iBooker.bookProfile("p_x_diffFN_vs_x_N", p_x_diffFN_vs_x_N);

    if (debug_) {
      p_x_diffFN_vs_x_N->Write("p_x_diffFN_vs_x_N");

      auto g_results = std::make_unique<TGraph>();
      g_results->SetPoint(0, sh_x_N, 0.);
      g_results->SetPoint(1, a, a_unc);
      g_results->SetPoint(2, b, b_unc);
      g_results->SetPoint(3, b_fs, b_fs_unc);
      g_results->Write("g_results");
    }
  }

  // write results
  edm::LogInfo("PPSAlignmentHarvester") << seqPos << ": x_alignment_relative:\n"
                                        << xAliRelResults_ << seqPos + 1 << ": x_alignment_relative_sl_fix:\n"
                                        << xAliRelResultsSlopeFixed_;

  if (textResultsFile_.is_open()) {
    textResultsFile_ << seqPos << ": x_alignment_relative:\n" << xAliRelResults_ << "\n";
    textResultsFile_ << seqPos << ": x_alignment_relative_sl_fix:\n" << xAliRelResultsSlopeFixed_ << "\n\n";
  }
}

// -------------------------------- y alignment methods --------------------------------

double PPSAlignmentHarvester::findMax(const TF1* ff_fit) {
  const double mu = ff_fit->GetParameter(1);
  const double si = ff_fit->GetParameter(2);

  // unreasonable fit?
  if (si > 25. || std::fabs(mu) > 100.)
    return 1E100;

  double x_max = 1E100;
  double y_max = -1E100;
  for (double x = mu - si; x <= mu + si; x += 0.001) {
    double y = ff_fit->Eval(x);
    if (y > y_max) {
      x_max = x;
      y_max = y;
    }
  }

  return x_max;
}

TH1D* PPSAlignmentHarvester::buildModeGraph(DQMStore::IBooker& iBooker,
                                            const MonitorElement* h2_y_vs_x,
                                            const PPSAlignmentConfiguration& cfg,
                                            const PPSAlignmentConfiguration::RPConfig& rpc) {
  TDirectory* d_top = nullptr;
  if (debug_)
    d_top = gDirectory;

  auto ff_fit = std::make_unique<TF1>("ff_fit", "[0] * exp(-(x-[1])*(x-[1])/2./[2]/[2]) + [3] + [4]*x");

  int h_n = h2_y_vs_x->getNbinsX();
  double diff = h2_y_vs_x->getTH2D()->GetXaxis()->GetBinWidth(1) / 2.;
  auto h_mode = iBooker.book1DD("mode",
                                ";x (mm); mode of y (mm)",
                                h_n,
                                h2_y_vs_x->getTH2D()->GetXaxis()->GetBinCenter(1) - diff,
                                h2_y_vs_x->getTH2D()->GetXaxis()->GetBinCenter(h_n) + diff);

  // find mode for each bin
  for (int bix = 1; bix <= h_n; bix++) {
    const double x = h2_y_vs_x->getTH2D()->GetXaxis()->GetBinCenter(bix);

    char buf[100];
    sprintf(buf, "h_y_x=%.3f", x);
    TH1D* h_y = h2_y_vs_x->getTH2D()->ProjectionY(buf, bix, bix);

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

    double x_min = rpc.x_min_fit_mode_, x_max = rpc.x_max_fit_mode_;
    h_y->Fit(ff_fit.get(), "Q", "", x_min, x_max);

    ff_fit->ReleaseParameter(4);
    double w = std::min(4., 2. * ff_fit->GetParameter(2));
    x_min = ff_fit->GetParameter(1) - w;
    x_max = std::min(rpc.y_max_fit_mode_, ff_fit->GetParameter(1) + w);

    h_y->Fit(ff_fit.get(), "Q", "", x_min, x_max);

    if (debug_)
      h_y->Write("h_y");

    double y_mode = findMax(ff_fit.get());
    const double y_mode_fit_unc = ff_fit->GetParameter(2) / 10;
    const double y_mode_sys_unc = cfg.y_mode_sys_unc();
    double y_mode_unc = std::sqrt(y_mode_fit_unc * y_mode_fit_unc + y_mode_sys_unc * y_mode_sys_unc);

    const double chiSqThreshold = cfg.chiSqThreshold();

    const bool valid =
        !(std::fabs(y_mode_unc) > cfg.y_mode_unc_max_valid() || std::fabs(y_mode) > cfg.y_mode_max_valid() ||
          ff_fit->GetChisquare() / ff_fit->GetNDF() > chiSqThreshold);

    if (debug_) {
      auto g_data = std::make_unique<TGraph>();
      g_data->SetPoint(0, y_mode, y_mode_unc);
      g_data->SetPoint(1, ff_fit->GetChisquare(), ff_fit->GetNDF());
      g_data->SetPoint(2, valid, 0.);
      g_data->Write("g_data");
    }

    if (!valid)
      continue;

    h_mode->Fill(x, y_mode);
    h_mode->setBinError(bix, y_mode_unc);
  }

  return h_mode->getTH1D();
}

void PPSAlignmentHarvester::yAlignment(DQMStore::IBooker& iBooker,
                                       DQMStore::IGetter& iGetter,
                                       const PPSAlignmentConfiguration& cfg) {
  TDirectory* yAliDir = nullptr;
  if (debug_)
    yAliDir = debugFile_->mkdir((std::to_string(seqPos) + ": y_alignment").c_str());

  auto ff = std::make_unique<TF1>("ff", "[0] + [1]*(x - [2])");
  auto ff_sl_fix = std::make_unique<TF1>("ff_sl_fix", "[0] + [1]*(x - [2])");

  // processing
  for (const auto& sc : {cfg.sectorConfig45(), cfg.sectorConfig56()}) {
    for (const auto& rpc : {sc.rp_F_, sc.rp_N_}) {
      TDirectory* rpDir = nullptr;
      if (debug_) {
        rpDir = yAliDir->mkdir(rpc.name_.c_str());
        gDirectory = rpDir->mkdir("x");
      }

      auto* h2_y_vs_x =
          iGetter.get(dqmDir_ + "/worker/" + sc.name_ + "/multiplicity selection/" + rpc.name_ + "/h2_y_vs_x");

      if (h2_y_vs_x == nullptr) {
        edm::LogWarning("PPSAlignmentHarvester") << "[y_alignment] " << rpc.name_ << ": cannot load data, skipping";
        continue;
      }

      iBooker.setCurrentFolder(dqmDir_ + "/harvester/y alignment/" + rpc.name_);
      auto* h_y_cen_vs_x = buildModeGraph(iBooker, h2_y_vs_x, cfg, rpc);

      if ((unsigned int)h_y_cen_vs_x->GetEntries() < cfg.modeGraphMinN()) {
        edm::LogWarning("PPSAlignmentHarvester")
            << "[y_alignment] " << rpc.name_ << ": insufficient data, skipping (mode graph "
            << h_y_cen_vs_x->GetEntries() << "/" << cfg.modeGraphMinN() << ")";
        continue;
      }

      const double x_min = cfg.alignment_y_ranges().at(rpc.id_).x_min_;
      const double x_max = cfg.alignment_y_ranges().at(rpc.id_).x_max_;

      const double sh_x = sh_x_map_[rpc.id_];
      double slope = rpc.slope_;

      // calculate the results without slope fixed
      ff->SetParameters(0., 0., 0.);
      ff->FixParameter(2, -sh_x);
      ff->SetLineColor(2);
      h_y_cen_vs_x->Fit(ff.get(), "Q", "", x_min, x_max);

      const double a = ff->GetParameter(1), a_unc = ff->GetParError(1);
      const double b = ff->GetParameter(0), b_unc = ff->GetParError(0);

      edm::LogInfo("PPSAlignmentHarvester")
          << "[y_alignment] " << rpc.name_ << ":\n"
          << std::fixed << std::setprecision(3) << "    x_min = " << x_min << ", x_max = " << x_max << "\n"
          << "    sh_x = " << sh_x << ", slope (fix) = " << slope << ", slope (fitted) = " << a;

      CTPPSRPAlignmentCorrectionData rpResult(0., 0., b, b_unc, 0., 0., 0., 0., 0., 0., 0., 0.);
      yAliResults_.setRPCorrection(rpc.id_, rpResult);

      // calculate the results with slope fixed
      ff_sl_fix->SetParameters(0., 0., 0.);
      ff_sl_fix->FixParameter(1, slope);
      ff_sl_fix->FixParameter(2, -sh_x);
      ff_sl_fix->SetLineColor(4);
      h_y_cen_vs_x->Fit(ff_sl_fix.get(), "Q+", "", x_min, x_max);

      const double b_fs = ff_sl_fix->GetParameter(0), b_fs_unc = ff_sl_fix->GetParError(0);

      CTPPSRPAlignmentCorrectionData rpResult_sl_fix(0., 0., b_fs, b_fs_unc, 0., 0., 0., 0., 0., 0., 0., 0.);
      yAliResultsSlopeFixed_.setRPCorrection(rpc.id_, rpResult_sl_fix);

      edm::LogInfo("PPSAlignmentHarvester")
          << "[y_alignment] " << std::fixed << std::setprecision(3) << "ff: " << ff->GetParameter(0) << " + "
          << ff->GetParameter(1) << " * (x - " << ff->GetParameter(2) << "), ff_sl_fix: " << ff_sl_fix->GetParameter(0)
          << " + " << ff_sl_fix->GetParameter(1) << " * (x - " << ff_sl_fix->GetParameter(2) << ")";

      if (debug_) {
        gDirectory = rpDir;

        h_y_cen_vs_x->SetTitle(";x (mm); mode of y (mm)");
        h_y_cen_vs_x->Write("h_y_cen_vs_x");

        auto g_results = std::make_unique<TGraph>();
        g_results->SetPoint(0, sh_x, 0.);
        g_results->SetPoint(1, a, a_unc);
        g_results->SetPoint(2, b, b_unc);
        g_results->SetPoint(3, b_fs, b_fs_unc);
        g_results->Write("g_results");
      }
    }
  }

  // write results
  edm::LogInfo("PPSAlignmentHarvester") << seqPos << ": y_alignment:\n"
                                        << yAliResults_ << seqPos << ": y_alignment_sl_fix:\n"
                                        << yAliResultsSlopeFixed_;

  if (textResultsFile_.is_open()) {
    textResultsFile_ << seqPos << ": y_alignment:\n" << yAliResults_ << "\n";
    textResultsFile_ << seqPos << ": y_alignment_sl_fix:\n" << yAliResultsSlopeFixed_ << "\n\n";
  }
}

// -------------------------------- other methods --------------------------------

// Creates a plot showing a cut applied by the worker. Used only for debug purposes.
void PPSAlignmentHarvester::writeCutPlot(
    TH2D* h, const double a, const double c, const double n_si, const double si, const std::string& label) {
  auto canvas = std::make_unique<TCanvas>();
  canvas->SetName(label.c_str());
  canvas->SetLogz(1);

  h->Draw("colz");

  double x_min = -30.;
  double x_max = 30.;

  auto g_up = std::make_unique<TGraph>();
  g_up->SetName("g_up");
  g_up->SetPoint(0, x_min, -a * x_min - c + n_si * si);
  g_up->SetPoint(1, x_max, -a * x_max - c + n_si * si);
  g_up->SetLineColor(1);
  g_up->Draw("l");

  auto g_down = std::make_unique<TGraph>();
  g_down->SetName("g_down");
  g_down->SetPoint(0, x_min, -a * x_min - c - n_si * si);
  g_down->SetPoint(1, x_max, -a * x_max - c - n_si * si);
  g_down->SetLineColor(1);
  g_down->Draw("l");

  canvas->Write();
}

// Points in TGraph should be sorted (TGraph::Sort())
// if n, binWidth, or min is set to -1, method will find it on its own
std::unique_ptr<TH1D> PPSAlignmentHarvester::getTH1DFromTGraphErrors(
    TGraphErrors* graph, const std::string& title, const std::string& labels, int n, double binWidth, double min) {
  std::unique_ptr<TH1D> hist;
  if (n == 0) {
    hist = std::make_unique<TH1D>(title.c_str(), labels.c_str(), 0, -10., 10.);
  } else if (n == 1) {
    hist = std::make_unique<TH1D>(title.c_str(), labels.c_str(), 1, graph->GetPointX(0) - 5., graph->GetPointX(0) + 5.);
  } else {
    n = n == -1 ? graph->GetN() : n;
    binWidth = binWidth == -1 ? graph->GetPointX(1) - graph->GetPointX(0) : binWidth;
    double diff = binWidth / 2.;
    min = min == -1 ? graph->GetPointX(0) - diff : min;
    double max = min + n * binWidth;
    hist = std::make_unique<TH1D>(title.c_str(), labels.c_str(), n, min, max);
  }

  for (int i = 0; i < graph->GetN(); i++) {
    double x, y;
    graph->GetPoint(i, x, y);
    hist->Fill(x, y);
    hist->SetBinError(hist->GetXaxis()->FindBin(x), graph->GetErrorY(i));
  }
  return hist;
}

// Get Long 32-bit detector ID from short 3-digit ID
CTPPSRPAlignmentCorrectionsData PPSAlignmentHarvester::getLongIdResults(CTPPSRPAlignmentCorrectionsData shortIdResults) {
  CTPPSRPAlignmentCorrectionsData longIdResults;
  for (const auto& [shortId, correction] : shortIdResults.getRPMap()) {
    unsigned int arm = shortId / 100;
    unsigned int station = (shortId / 10) % 10;
    unsigned int rp = shortId % 10;

    uint32_t longDetId = detectorId_ << 28 | subdetectorId_ << 25 | arm << 24 | station << 22 | rp << 19;

    longIdResults.addRPCorrection(longDetId, correction);
  }

  return longIdResults;
}

DEFINE_FWK_MODULE(PPSAlignmentHarvester);
