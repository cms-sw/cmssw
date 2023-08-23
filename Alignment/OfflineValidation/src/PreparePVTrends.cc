#include "Alignment/OfflineValidation/interface/PreparePVTrends.h"

namespace ph = std::placeholders;  // for _1, _2, _3...
namespace pt = boost::property_tree;

PreparePVTrends::PreparePVTrends(const char *outputFileName, int nWorkers, pt::ptree &json)
    : outputFileName_(outputFileName), nWorkers_(nWorkers) {
  setDirsAndLabels(json);
}

void PreparePVTrends::setDirsAndLabels(pt::ptree &json) {
  DirList.clear();
  LabelList.clear();
  for (const auto &childTree : json) {
    DirList.push_back(childTree.first.c_str());
    LabelList.push_back(childTree.second.get<std::string>("title"));
  }
}

void PreparePVTrends::multiRunPVValidation(bool useRMS, TString lumiInputFile, bool doUnitTest) {
  TStopwatch timer;
  timer.Start();

  gROOT->ProcessLine("gErrorIgnoreLevel = kError;");

  ROOT::EnableThreadSafety();
  TH1::AddDirectory(kFALSE);

  std::ofstream outfile("log.txt");

  const Int_t nDirs_ = DirList.size();
  TString LegLabels[10];
  const char *dirs[10];

  std::vector<int> intersection;
  std::vector<double> runs;
  std::vector<double> x_ticks;
  std::vector<double> ex_ticks = {0.};

  for (Int_t j = 0; j < nDirs_; j++) {
    // Retrieve labels
    LegLabels[j] = LabelList[j];
    dirs[j] = DirList[j].data();

    std::vector<int> currentList = list_files(dirs[j]);
    std::vector<int> tempSwap;

    std::sort(currentList.begin(), currentList.end());

    if (j == 0) {
      intersection = currentList;
    }

    std::sort(intersection.begin(), intersection.end());

    std::set_intersection(
        currentList.begin(), currentList.end(), intersection.begin(), intersection.end(), std::back_inserter(tempSwap));

    intersection.clear();
    intersection = tempSwap;
    tempSwap.clear();
  }

  std::ifstream lumifile(lumiInputFile.Data());

  std::string line;
  while (std::getline(lumifile, line)) {
    std::istringstream iss(line);
    std::string a, b;
    if (!(iss >> a >> b)) {
      break;
    }  // error
    int run = std::stoi(a);

    // check if the run is in the list
    if (std::find(intersection.begin(), intersection.end(), run) == intersection.end()) {
      logWarning << " Run: " << run << " is not found in the intersection" << std::endl;
    }
  }

  // book the vectors of values
  alignmentTrend dxyPhiMeans_;
  alignmentTrend dxyPhiChi2_;
  alignmentTrend dxyPhiHiErr_;
  alignmentTrend dxyPhiLoErr_;
  alignmentTrend dxyPhiKS_;
  alignmentTrend dxyPhiHi_;
  alignmentTrend dxyPhiLo_;

  alignmentTrend dxyEtaMeans_;
  alignmentTrend dxyEtaChi2_;
  alignmentTrend dxyEtaHiErr_;
  alignmentTrend dxyEtaLoErr_;
  alignmentTrend dxyEtaKS_;
  alignmentTrend dxyEtaHi_;
  alignmentTrend dxyEtaLo_;

  alignmentTrend dzPhiMeans_;
  alignmentTrend dzPhiChi2_;
  alignmentTrend dzPhiHiErr_;
  alignmentTrend dzPhiLoErr_;
  alignmentTrend dzPhiKS_;
  alignmentTrend dzPhiHi_;
  alignmentTrend dzPhiLo_;

  alignmentTrend dzEtaMeans_;
  alignmentTrend dzEtaChi2_;
  alignmentTrend dzEtaHiErr_;
  alignmentTrend dzEtaLoErr_;
  alignmentTrend dzEtaKS_;
  alignmentTrend dzEtaHi_;
  alignmentTrend dzEtaLo_;

  // unrolled histos

  std::map<TString, std::vector<unrolledHisto> > dxyVect;
  std::map<TString, std::vector<unrolledHisto> > dzVect;

  logInfo << " pre do-stuff: " << runs.size() << std::endl;

  //we should use std::bind to create a functor and then pass it to the procPool
  auto f_processData =
      std::bind(processData, ph::_1, intersection, nDirs_, dirs, LegLabels, useRMS, nWorkers_, doUnitTest);

  //f_processData(0);
  //logInfo<<" post do-stuff: " <<  runs.size() << std::endl;

  TProcPool procPool(std::min(nWorkers_, intersection.size()));
  std::vector<size_t> range(std::min(nWorkers_, intersection.size()));
  std::iota(range.begin(), range.end(), 0);
  //procPool.Map([&f_processData](size_t a) { f_processData(a); },{1,2,3});
  auto extracts = procPool.Map(f_processData, range);

  // sort the extracts according to the global index
  std::sort(extracts.begin(), extracts.end(), [](const outPVtrends &a, const outPVtrends &b) -> bool {
    return a.m_index < b.m_index;
  });

  // re-assemble everything together
  for (auto extractedTrend : extracts) {
    runs.insert(std::end(runs), std::begin(extractedTrend.m_runs), std::end(extractedTrend.m_runs));

    for (const auto &label : LegLabels) {
      //******************************//
      dxyPhiMeans_[label].insert(std::end(dxyPhiMeans_[label]),
                                 std::begin(extractedTrend.m_dxyPhiMeans[label]),
                                 std::end(extractedTrend.m_dxyPhiMeans[label]));
      dxyPhiChi2_[label].insert(std::end(dxyPhiChi2_[label]),
                                std::begin(extractedTrend.m_dxyPhiChi2[label]),
                                std::end(extractedTrend.m_dxyPhiChi2[label]));
      dxyPhiKS_[label].insert(std::end(dxyPhiKS_[label]),
                              std::begin(extractedTrend.m_dxyPhiKS[label]),
                              std::end(extractedTrend.m_dxyPhiKS[label]));

      dxyPhiHi_[label].insert(std::end(dxyPhiHi_[label]),
                              std::begin(extractedTrend.m_dxyPhiHi[label]),
                              std::end(extractedTrend.m_dxyPhiHi[label]));
      dxyPhiLo_[label].insert(std::end(dxyPhiLo_[label]),
                              std::begin(extractedTrend.m_dxyPhiLo[label]),
                              std::end(extractedTrend.m_dxyPhiLo[label]));

      //******************************//
      dzPhiMeans_[label].insert(std::end(dzPhiMeans_[label]),
                                std::begin(extractedTrend.m_dzPhiMeans[label]),
                                std::end(extractedTrend.m_dzPhiMeans[label]));
      dzPhiChi2_[label].insert(std::end(dzPhiChi2_[label]),
                               std::begin(extractedTrend.m_dzPhiChi2[label]),
                               std::end(extractedTrend.m_dzPhiChi2[label]));
      dzPhiKS_[label].insert(std::end(dzPhiKS_[label]),
                             std::begin(extractedTrend.m_dzPhiKS[label]),
                             std::end(extractedTrend.m_dzPhiKS[label]));

      dzPhiHi_[label].insert(std::end(dzPhiHi_[label]),
                             std::begin(extractedTrend.m_dzPhiHi[label]),
                             std::end(extractedTrend.m_dzPhiHi[label]));
      dzPhiLo_[label].insert(std::end(dzPhiLo_[label]),
                             std::begin(extractedTrend.m_dzPhiLo[label]),
                             std::end(extractedTrend.m_dzPhiLo[label]));

      //******************************//
      dxyEtaMeans_[label].insert(std::end(dxyEtaMeans_[label]),
                                 std::begin(extractedTrend.m_dxyEtaMeans[label]),
                                 std::end(extractedTrend.m_dxyEtaMeans[label]));
      dxyEtaChi2_[label].insert(std::end(dxyEtaChi2_[label]),
                                std::begin(extractedTrend.m_dxyEtaChi2[label]),
                                std::end(extractedTrend.m_dxyEtaChi2[label]));
      dxyEtaKS_[label].insert(std::end(dxyEtaKS_[label]),
                              std::begin(extractedTrend.m_dxyEtaKS[label]),
                              std::end(extractedTrend.m_dxyEtaKS[label]));

      dxyEtaHi_[label].insert(std::end(dxyEtaHi_[label]),
                              std::begin(extractedTrend.m_dxyEtaHi[label]),
                              std::end(extractedTrend.m_dxyEtaHi[label]));
      dxyEtaLo_[label].insert(std::end(dxyEtaLo_[label]),
                              std::begin(extractedTrend.m_dxyEtaLo[label]),
                              std::end(extractedTrend.m_dxyEtaLo[label]));

      //******************************//
      dzEtaMeans_[label].insert(std::end(dzEtaMeans_[label]),
                                std::begin(extractedTrend.m_dzEtaMeans[label]),
                                std::end(extractedTrend.m_dzEtaMeans[label]));
      dzEtaChi2_[label].insert(std::end(dzEtaChi2_[label]),
                               std::begin(extractedTrend.m_dzEtaChi2[label]),
                               std::end(extractedTrend.m_dzEtaChi2[label]));
      dzEtaKS_[label].insert(std::end(dzEtaKS_[label]),
                             std::begin(extractedTrend.m_dzEtaKS[label]),
                             std::end(extractedTrend.m_dzEtaKS[label]));

      dzEtaHi_[label].insert(std::end(dzEtaHi_[label]),
                             std::begin(extractedTrend.m_dzEtaHi[label]),
                             std::end(extractedTrend.m_dzEtaHi[label]));
      dzEtaLo_[label].insert(std::end(dzEtaLo_[label]),
                             std::begin(extractedTrend.m_dzEtaLo[label]),
                             std::end(extractedTrend.m_dzEtaLo[label]));

      //******************************//
      dxyVect[label].insert(std::end(dxyVect[label]),
                            std::begin(extractedTrend.m_dxyVect[label]),
                            std::end(extractedTrend.m_dxyVect[label]));
      dzVect[label].insert(std::end(dzVect[label]),
                           std::begin(extractedTrend.m_dzVect[label]),
                           std::end(extractedTrend.m_dzVect[label]));
    }
  }
  // extra vectors for low and high boundaries

  for (const auto &label : LegLabels) {
    for (unsigned int it = 0; it < dxyPhiMeans_[label].size(); it++) {
      dxyPhiHiErr_[label].push_back(std::abs(dxyPhiHi_[label][it] - dxyPhiMeans_[label][it]));
      dxyPhiLoErr_[label].push_back(std::abs(dxyPhiLo_[label][it] - dxyPhiMeans_[label][it]));
      dxyEtaHiErr_[label].push_back(std::abs(dxyEtaHi_[label][it] - dxyEtaMeans_[label][it]));
      dxyEtaLoErr_[label].push_back(std::abs(dxyEtaLo_[label][it] - dxyEtaMeans_[label][it]));

      if (VERBOSE) {
        logInfo << "label: " << label << " means:" << dxyEtaMeans_[label][it] << " low: " << dxyEtaLo_[label][it]
                << " loErr: " << dxyEtaLoErr_[label][it] << std::endl;
      }

      dzPhiHiErr_[label].push_back(std::abs(dzPhiHi_[label][it] - dzPhiMeans_[label][it]));
      dzPhiLoErr_[label].push_back(std::abs(dzPhiLo_[label][it] - dzPhiMeans_[label][it]));
      dzEtaHiErr_[label].push_back(std::abs(dzEtaHi_[label][it] - dzEtaMeans_[label][it]));
      dzEtaLoErr_[label].push_back(std::abs(dzEtaLo_[label][it] - dzEtaMeans_[label][it]));
    }
  }
  // bias on the mean

  TGraph *g_dxy_phi_vs_run[nDirs_];
  TGraphAsymmErrors *gerr_dxy_phi_vs_run[nDirs_];
  TGraph *g_chi2_dxy_phi_vs_run[nDirs_];
  TGraph *g_KS_dxy_phi_vs_run[nDirs_];
  //TGraph *gprime_dxy_phi_vs_run[nDirs_];
  TGraph *g_dxy_phi_hi_vs_run[nDirs_];
  TGraph *g_dxy_phi_lo_vs_run[nDirs_];

  TGraph *g_dxy_eta_vs_run[nDirs_];
  TGraphAsymmErrors *gerr_dxy_eta_vs_run[nDirs_];
  TGraph *g_chi2_dxy_eta_vs_run[nDirs_];
  TGraph *g_KS_dxy_eta_vs_run[nDirs_];
  //TGraph *gprime_dxy_eta_vs_run[nDirs_];
  TGraph *g_dxy_eta_hi_vs_run[nDirs_];
  TGraph *g_dxy_eta_lo_vs_run[nDirs_];

  TGraph *g_dz_phi_vs_run[nDirs_];
  TGraphAsymmErrors *gerr_dz_phi_vs_run[nDirs_];
  TGraph *g_chi2_dz_phi_vs_run[nDirs_];
  TGraph *g_KS_dz_phi_vs_run[nDirs_];
  //TGraph *gprime_dz_phi_vs_run[nDirs_];
  TGraph *g_dz_phi_hi_vs_run[nDirs_];
  TGraph *g_dz_phi_lo_vs_run[nDirs_];

  TGraph *g_dz_eta_vs_run[nDirs_];
  TGraphAsymmErrors *gerr_dz_eta_vs_run[nDirs_];
  TGraph *g_chi2_dz_eta_vs_run[nDirs_];
  TGraph *g_KS_dz_eta_vs_run[nDirs_];
  //TGraph *gprime_dz_eta_vs_run[nDirs_];
  TGraph *g_dz_eta_hi_vs_run[nDirs_];
  TGraph *g_dz_eta_lo_vs_run[nDirs_];

  // resolutions

  TH1F *h_RMS_dxy_phi_vs_run[nDirs_];
  TH1F *h_RMS_dxy_eta_vs_run[nDirs_];
  TH1F *h_RMS_dz_phi_vs_run[nDirs_];
  TH1F *h_RMS_dz_eta_vs_run[nDirs_];

  // scatters of integrated bias

  TH2F *h2_scatter_dxy_vs_run[nDirs_];
  TH2F *h2_scatter_dz_vs_run[nDirs_];

  // decide the type
  TString theType = "run number";
  TString theTypeLabel = "run number";
  x_ticks = runs;

  pv::bundle theBundle = pv::bundle(nDirs_, theType, theTypeLabel, useRMS);
  theBundle.printAll();

  TFile *fout = TFile::Open(outputFileName_, "RECREATE");

  for (Int_t j = 0; j < nDirs_; j++) {
    // check on the sanity
    logInfo << "x_ticks.size()= " << x_ticks.size() << " dxyPhiMeans_[LegLabels[" << j
            << "]].size()=" << dxyPhiMeans_[LegLabels[j]].size() << std::endl;

    // otherwise something very bad has happened
    assert(x_ticks.size() == dxyPhiMeans_[LegLabels[j]].size());

    // *************************************
    // dxy vs phi
    // *************************************

    auto dxyPhiInputs =
        pv::wrappedTrends(dxyPhiMeans_, dxyPhiLo_, dxyPhiHi_, dxyPhiLoErr_, dxyPhiHiErr_, dxyPhiChi2_, dxyPhiKS_);

    outputGraphs(dxyPhiInputs,
                 x_ticks,
                 ex_ticks,
                 g_dxy_phi_vs_run[j],
                 g_chi2_dxy_phi_vs_run[j],
                 g_KS_dxy_phi_vs_run[j],
                 g_dxy_phi_lo_vs_run[j],
                 g_dxy_phi_hi_vs_run[j],
                 gerr_dxy_phi_vs_run[j],
                 h_RMS_dxy_phi_vs_run,
                 theBundle,
                 pv::dxyphi,
                 j,
                 LegLabels[j]);

    // *************************************
    // dxy vs eta
    // *************************************

    auto dxyEtaInputs =
        pv::wrappedTrends(dxyEtaMeans_, dxyEtaLo_, dxyEtaHi_, dxyEtaLoErr_, dxyEtaHiErr_, dxyEtaChi2_, dxyEtaKS_);

    outputGraphs(dxyEtaInputs,
                 x_ticks,
                 ex_ticks,
                 g_dxy_eta_vs_run[j],
                 g_chi2_dxy_eta_vs_run[j],
                 g_KS_dxy_eta_vs_run[j],
                 g_dxy_eta_lo_vs_run[j],
                 g_dxy_eta_hi_vs_run[j],
                 gerr_dxy_eta_vs_run[j],
                 h_RMS_dxy_eta_vs_run,
                 theBundle,
                 pv::dxyeta,
                 j,
                 LegLabels[j]);

    // *************************************
    // dz vs phi
    // *************************************

    auto dzPhiInputs =
        pv::wrappedTrends(dzPhiMeans_, dzPhiLo_, dzPhiHi_, dzPhiLoErr_, dzPhiHiErr_, dzPhiChi2_, dzPhiKS_);

    outputGraphs(dzPhiInputs,
                 x_ticks,
                 ex_ticks,
                 g_dz_phi_vs_run[j],
                 g_chi2_dz_phi_vs_run[j],
                 g_KS_dz_phi_vs_run[j],
                 g_dz_phi_lo_vs_run[j],
                 g_dz_phi_hi_vs_run[j],
                 gerr_dz_phi_vs_run[j],
                 h_RMS_dz_phi_vs_run,
                 theBundle,
                 pv::dzphi,
                 j,
                 LegLabels[j]);

    // *************************************
    // dz vs eta
    // *************************************

    auto dzEtaInputs =
        pv::wrappedTrends(dzEtaMeans_, dzEtaLo_, dzEtaHi_, dzEtaLoErr_, dzEtaHiErr_, dzEtaChi2_, dzEtaKS_);

    outputGraphs(dzEtaInputs,
                 x_ticks,
                 ex_ticks,
                 g_dz_eta_vs_run[j],
                 g_chi2_dz_eta_vs_run[j],
                 g_KS_dz_eta_vs_run[j],
                 g_dz_eta_lo_vs_run[j],
                 g_dz_eta_hi_vs_run[j],
                 gerr_dz_eta_vs_run[j],
                 h_RMS_dz_eta_vs_run,
                 theBundle,
                 pv::dzeta,
                 j,
                 LegLabels[j]);

    // *************************************
    // Integrated bias dxy scatter plots
    // *************************************

    h2_scatter_dxy_vs_run[j] =
        new TH2F(Form("h2_scatter_dxy_%s", LegLabels[j].Data()),
                 Form("scatter of d_{xy} vs %s;%s;d_{xy} [cm]", theType.Data(), theTypeLabel.Data()),
                 x_ticks.size() - 1,
                 &(x_ticks[0]),
                 dxyVect[LegLabels[j]][0].get_n_bins(),
                 dxyVect[LegLabels[j]][0].get_y_min(),
                 dxyVect[LegLabels[j]][0].get_y_max());
    h2_scatter_dxy_vs_run[j]->SetStats(kFALSE);
    h2_scatter_dxy_vs_run[j]->SetTitle(LegLabels[j]);

    for (unsigned int runindex = 0; runindex < x_ticks.size(); runindex++) {
      for (unsigned int binindex = 0; binindex < dxyVect[LegLabels[j]][runindex].get_n_bins(); binindex++) {
        h2_scatter_dxy_vs_run[j]->SetBinContent(runindex + 1,
                                                binindex + 1,
                                                dxyVect[LegLabels[j]][runindex].get_bin_contents().at(binindex) /
                                                    dxyVect[LegLabels[j]][runindex].get_integral());
      }
    }

    // *************************************
    // Integrated bias dz scatter plots
    // *************************************

    h2_scatter_dz_vs_run[j] =
        new TH2F(Form("h2_scatter_dz_%s", LegLabels[j].Data()),
                 Form("scatter of d_{z} vs %s;%s;d_{z} [cm]", theType.Data(), theTypeLabel.Data()),
                 x_ticks.size() - 1,
                 &(x_ticks[0]),
                 dzVect[LegLabels[j]][0].get_n_bins(),
                 dzVect[LegLabels[j]][0].get_y_min(),
                 dzVect[LegLabels[j]][0].get_y_max());
    h2_scatter_dz_vs_run[j]->SetStats(kFALSE);
    h2_scatter_dz_vs_run[j]->SetTitle(LegLabels[j]);

    for (unsigned int runindex = 0; runindex < x_ticks.size(); runindex++) {
      for (unsigned int binindex = 0; binindex < dzVect[LegLabels[j]][runindex].get_n_bins(); binindex++) {
        h2_scatter_dz_vs_run[j]->SetBinContent(runindex + 1,
                                               binindex + 1,
                                               dzVect[LegLabels[j]][runindex].get_bin_contents().at(binindex) /
                                                   dzVect[LegLabels[j]][runindex].get_integral());
      }
    }

    TString modified_label = (LegLabels[j].ReplaceAll(" ", "_"));
    g_dxy_phi_vs_run[j]->Write("mean_" + modified_label + "_dxy_phi_vs_run");
    g_chi2_dxy_phi_vs_run[j]->Write("chi2_" + modified_label + "_dxy_phi_vs_run");
    g_KS_dxy_phi_vs_run[j]->Write("KS_" + modified_label + "_dxy_phi_vs_run");
    g_dxy_phi_hi_vs_run[j]->Write("hi_" + modified_label + "_dxy_phi_hi_vs_run");
    g_dxy_phi_lo_vs_run[j]->Write("lo_" + modified_label + "_dxy_phi_lo_vs_run");

    g_dxy_eta_vs_run[j]->Write("mean_" + modified_label + "_dxy_eta_vs_run");
    g_chi2_dxy_eta_vs_run[j]->Write("chi2_" + modified_label + "_dxy_eta_vs_run");
    g_KS_dxy_eta_vs_run[j]->Write("KS_" + modified_label + "_dxy_eta_vs_run");
    g_dxy_eta_hi_vs_run[j]->Write("hi_" + modified_label + "_dxy_eta_hi_vs_run");
    g_dxy_eta_lo_vs_run[j]->Write("lo_" + modified_label + "_dxy_eta_lo_vs_run");

    g_dz_phi_vs_run[j]->Write("mean_" + modified_label + "_dz_phi_vs_run");
    g_chi2_dz_phi_vs_run[j]->Write("chi2_" + modified_label + "_dz_phi_vs_run");
    g_KS_dz_phi_vs_run[j]->Write("KS_" + modified_label + "_dz_phi_vs_run");
    g_dz_phi_hi_vs_run[j]->Write("hi_" + modified_label + "_dz_phi_hi_vs_run");
    g_dz_phi_lo_vs_run[j]->Write("lo_" + modified_label + "_dz_phi_lo_vs_run");

    g_dz_eta_vs_run[j]->Write("mean_" + modified_label + "_dz_eta_vs_run");
    g_chi2_dz_eta_vs_run[j]->Write("chi2_" + modified_label + "_dz_eta_vs_run");
    g_KS_dz_eta_vs_run[j]->Write("KS_" + modified_label + "_dz_eta_vs_run");
    g_dz_eta_hi_vs_run[j]->Write("hi_" + modified_label + "_dz_eta_hi_vs_run");
    g_dz_eta_lo_vs_run[j]->Write("lo_" + modified_label + "_dz_eta_lo_vs_run");

    h_RMS_dxy_phi_vs_run[j]->Write("RMS_" + modified_label + "_dxy_phi_vs_run");
    h_RMS_dxy_eta_vs_run[j]->Write("RMS_" + modified_label + "_dxy_eta_vs_run");
    h_RMS_dz_phi_vs_run[j]->Write("RMS_" + modified_label + "_dz_phi_vs_run");
    h_RMS_dz_eta_vs_run[j]->Write("RMS_" + modified_label + "_dz_eta_vs_run");

    // scatter

    h2_scatter_dxy_vs_run[j]->Write("Scatter_" + modified_label + "_dxy_vs_run");
    h2_scatter_dz_vs_run[j]->Write("Scatter_" + modified_label + "_dz_vs_run");
  }
  // do all the deletes

  for (int iDir = 0; iDir < nDirs_; iDir++) {
    delete g_dxy_phi_vs_run[iDir];
    delete g_chi2_dxy_phi_vs_run[iDir];
    delete g_KS_dxy_phi_vs_run[iDir];
    delete g_dxy_phi_hi_vs_run[iDir];
    delete g_dxy_phi_lo_vs_run[iDir];

    delete g_dxy_eta_vs_run[iDir];
    delete g_chi2_dxy_eta_vs_run[iDir];
    delete g_KS_dxy_eta_vs_run[iDir];
    delete g_dxy_eta_hi_vs_run[iDir];
    delete g_dxy_eta_lo_vs_run[iDir];

    delete g_dz_phi_vs_run[iDir];
    delete g_chi2_dz_phi_vs_run[iDir];
    delete g_KS_dz_phi_vs_run[iDir];
    delete g_dz_phi_hi_vs_run[iDir];
    delete g_dz_phi_lo_vs_run[iDir];

    delete g_dz_eta_vs_run[iDir];
    delete g_chi2_dz_eta_vs_run[iDir];
    delete g_KS_dz_eta_vs_run[iDir];
    delete g_dz_eta_hi_vs_run[iDir];
    delete g_dz_eta_lo_vs_run[iDir];

    delete h_RMS_dxy_phi_vs_run[iDir];
    delete h_RMS_dxy_eta_vs_run[iDir];
    delete h_RMS_dz_phi_vs_run[iDir];
    delete h_RMS_dz_eta_vs_run[iDir];

    delete h2_scatter_dxy_vs_run[iDir];
    delete h2_scatter_dz_vs_run[iDir];
  }

  fout->Close();

  timer.Stop();
  timer.Print();
}

/*! \fn outputGraphs
 *  \brief function to build the output graphs
 */

/*--------------------------------------------------------------------*/
void PreparePVTrends::outputGraphs(const pv::wrappedTrends &allInputs,
                                   const std::vector<double> &ticks,
                                   const std::vector<double> &ex_ticks,
                                   TGraph *&g_mean,
                                   TGraph *&g_chi2,
                                   TGraph *&g_KS,
                                   TGraph *&g_low,
                                   TGraph *&g_high,
                                   TGraphAsymmErrors *&g_asym,
                                   TH1F *h_RMS[],
                                   const pv::bundle &mybundle,
                                   const pv::view &theView,
                                   const int index,
                                   const TString &label)
/*--------------------------------------------------------------------*/
{
  g_mean = new TGraph(ticks.size(), &(ticks[0]), &((allInputs.getMean()[label])[0]));
  g_chi2 = new TGraph(ticks.size(), &(ticks[0]), &((allInputs.getChi2()[label])[0]));
  g_KS = new TGraph(ticks.size(), &(ticks[0]), &((allInputs.getKS()[label])[0]));
  g_high = new TGraph(ticks.size(), &(ticks[0]), &((allInputs.getHigh()[label])[0]));
  g_low = new TGraph(ticks.size(), &(ticks[0]), &((allInputs.getLow()[label])[0]));

  g_asym = new TGraphAsymmErrors(ticks.size(),
                                 &(ticks[0]),
                                 &((allInputs.getMean()[label])[0]),
                                 &(ex_ticks[0]),
                                 &(ex_ticks[0]),
                                 &((allInputs.getLowErr()[label])[0]),
                                 &((allInputs.getHighErr()[label])[0]));

  g_mean->SetTitle(label);
  g_asym->SetTitle(label);

  // scatter or RMS TH1
  h_RMS[index] = new TH1F(Form("h_RMS_dz_eta_%s", label.Data()), label, ticks.size() - 1, &(ticks[0]));
  h_RMS[index]->SetStats(kFALSE);

  for (size_t bincounter = 1; bincounter < ticks.size(); bincounter++) {
    h_RMS[index]->SetBinContent(
        bincounter, std::abs(allInputs.getHigh()[label][bincounter - 1] - allInputs.getLow()[label][bincounter - 1]));
    h_RMS[index]->SetBinError(bincounter, 0.01);
  }
}

/*! \fn list_files
 *  \brief utility function to list of filles in a directory
 */

/*--------------------------------------------------------------------*/
std::vector<int> PreparePVTrends::list_files(const char *dirname, const char *ext)
/*--------------------------------------------------------------------*/
{
  std::vector<int> theRunNumbers;

  TSystemDirectory dir(dirname, dirname);
  TList *files = dir.GetListOfFiles();
  if (files) {
    TSystemFile *file;
    TString fname;
    TIter next(files);
    while ((file = (TSystemFile *)next())) {
      fname = file->GetName();
      if (!file->IsDirectory() && fname.EndsWith(ext) && fname.BeginsWith("PVValidation")) {
        //logInfo << fname.Data() << std::endl;
        TObjArray *bits = fname.Tokenize("_");
        TString theRun = bits->At(2)->GetName();
        //logInfo << theRun << std::endl;
        TString formatRun = (theRun.ReplaceAll(".root", "")).ReplaceAll("_", "");
        //logInfo << dirname << " "<< formatRun.Atoi() << std::endl;
        theRunNumbers.push_back(formatRun.Atoi());
      }
    }
  }
  return theRunNumbers;
}

/*! \fn DrawConstant
 *  \brief utility function to draw a constant histogram with erros !=0
 */

/*--------------------------------------------------------------------*/
TH1F *PreparePVTrends::drawConstantWithErr(TH1F *hist, Int_t iter, Double_t theConst)
/*--------------------------------------------------------------------*/
{
  Int_t nbins = hist->GetNbinsX();
  Double_t lowedge = hist->GetBinLowEdge(1);
  Double_t highedge = hist->GetBinLowEdge(nbins + 1);

  TH1F *hzero = new TH1F(Form("hconst_%s_%i", hist->GetName(), iter),
                         Form("hconst_%s_%i", hist->GetName(), iter),
                         nbins,
                         lowedge,
                         highedge);
  for (Int_t i = 0; i <= hzero->GetNbinsX(); i++) {
    hzero->SetBinContent(i, theConst);
    hzero->SetBinError(i, hist->GetBinError(i));
  }
  hzero->SetLineWidth(2);
  hzero->SetLineStyle(9);
  hzero->SetLineColor(kMagenta);

  return hzero;
}

/*! \fn getUnrolledHisto
 *  \brief utility function to tranform a TH1 into a vector of floats
 */

/*--------------------------------------------------------------------*/
unrolledHisto PreparePVTrends::getUnrolledHisto(TH1F *hist)
/*--------------------------------------------------------------------*/
{
  /*
    Double_t y_min = hist->GetBinLowEdge(1);
    Double_t y_max = hist->GetBinLowEdge(hist->GetNbinsX()+1);
  */

  Double_t y_min = -0.1;
  Double_t y_max = 0.1;

  std::vector<Double_t> contents;
  for (int j = 0; j < hist->GetNbinsX(); j++) {
    if (std::abs(hist->GetXaxis()->GetBinCenter(j)) <= 0.1)
      contents.push_back(hist->GetBinContent(j + 1));
  }

  auto ret = unrolledHisto(y_min, y_max, contents.size(), contents);
  return ret;
}

/*! \fn getBiases
 *  \brief utility function to extract characterization of the PV bias plot
 */

/*--------------------------------------------------------------------*/
pv::biases PreparePVTrends::getBiases(TH1F *hist)
/*--------------------------------------------------------------------*/
{
  int nbins = hist->GetNbinsX();
  // if there are no bins in the histogram then return default constructed object
  // shouldn't really ever happen
  if (nbins <= 0) {
    logError << "No bins in the input histogram";
    return pv::biases();
  }

  //extract median from histogram
  double *y = new double[nbins];
  double *err = new double[nbins];

  // remember for weight means <x> = sum_i (x_i* w_i) / sum_i w_i ; where w_i = 1/sigma^2_i

  for (int j = 0; j < nbins; j++) {
    y[j] = hist->GetBinContent(j + 1);
    if (hist->GetBinError(j + 1) != 0.) {
      err[j] = 1. / (hist->GetBinError(j + 1) * hist->GetBinError(j + 1));
    } else {
      err[j] = 0.;
    }
  }

  Double_t w_mean = TMath::Mean(nbins, y, err);
  Double_t w_rms = TMath::RMS(nbins, y, err);

  Double_t mean = TMath::Mean(nbins, y);
  Double_t rms = TMath::RMS(nbins, y);

  Double_t max = hist->GetMaximum();
  Double_t min = hist->GetMinimum();

  // in case one would like to use a pol0 fit
  hist->Fit("pol0", "Q0+");
  TF1 *f = (TF1 *)hist->FindObject("pol0");
  //f->SetLineColor(hist->GetLineColor());
  //f->SetLineStyle(hist->GetLineStyle());
  Double_t chi2 = f->GetChisquare();
  Int_t ndf = f->GetNDF();

  TH1F *theZero = drawConstantWithErr(hist, 1, 1.);
  TH1F *displaced = (TH1F *)hist->Clone("displaced");
  displaced->Add(theZero);
  Double_t ksScore = std::max(-20., TMath::Log10(displaced->KolmogorovTest(theZero)));

  /*
    std::pair<std::pair<Double_t,Double_t>, Double_t> result;
    std::pair<Double_t,Double_t> resultBounds;
    resultBounds = useRMS_ ? std::make_pair(mean-rms,mean+rms) :  std::make_pair(min,max)  ;
    result = make_pair(resultBounds,mean);
  */

  pv::biases result(mean, rms, w_mean, w_rms, min, max, chi2, ndf, ksScore);

  delete theZero;
  delete displaced;
  delete[] y;
  delete[] err;
  return result;
}

/*! \fn processData
 *  \brief function where the magic happens, take the raw inputs and creates the output Trends
 */

/*--------------------------------------------------------------------*/
outPVtrends PreparePVTrends::processData(size_t iter,
                                         std::vector<int> intersection,
                                         const Int_t nDirs_,
                                         const char *dirs[10],
                                         TString LegLabels[10],
                                         bool useRMS,
                                         const size_t nWorkers,
                                         bool doUnitTest)
/*--------------------------------------------------------------------*/
{
  outPVtrends ret;

  unsigned int effSize = std::min(nWorkers, intersection.size());

  unsigned int pitch = std::floor(intersection.size() / effSize);
  unsigned int first = iter * pitch;
  unsigned int last = (iter == (effSize - 1)) ? intersection.size() : ((iter + 1) * pitch);

  logInfo << "iter:" << iter << "| pitch: " << pitch << " [" << first << "-" << last << ")" << std::endl;

  ret.m_index = iter;

  for (unsigned int n = first; n < last; n++) {
    //in case of debug, use only 50
    //for(unsigned int n=0; n<50;n++){

    //if(intersection.at(n)!=283946)
    //  continue;

    if (VERBOSE) {
      logInfo << "iter: " << iter << " " << n << " " << intersection.at(n) << std::endl;
    }

    TFile *fins[nDirs_];

    TH1F *dxyPhiMeanTrend[nDirs_];
    TH1F *dxyPhiWidthTrend[nDirs_];
    TH1F *dzPhiMeanTrend[nDirs_];
    TH1F *dzPhiWidthTrend[nDirs_];

    //TH1F *dxyLadderMeanTrend[nDirs_];
    //TH1F *dxyLadderWidthTrend[nDirs_];
    //TH1F *dzLadderWidthTrend[nDirs_];
    //TH1F *dzLadderMeanTrend[nDirs_];

    //TH1F *dxyModZMeanTrend[nDirs_];
    //TH1F *dxyModZWidthTrend[nDirs_];
    //TH1F *dzModZMeanTrend[nDirs_];
    //TH1F *dzModZWidthTrend[nDirs_];

    TH1F *dxyEtaMeanTrend[nDirs_];
    TH1F *dxyEtaWidthTrend[nDirs_];
    TH1F *dzEtaMeanTrend[nDirs_];
    TH1F *dzEtaWidthTrend[nDirs_];

    TH1F *dxyNormPhiWidthTrend[nDirs_];
    TH1F *dxyNormEtaWidthTrend[nDirs_];
    TH1F *dzNormPhiWidthTrend[nDirs_];
    TH1F *dzNormEtaWidthTrend[nDirs_];

    TH1F *dxyNormPtWidthTrend[nDirs_];
    TH1F *dzNormPtWidthTrend[nDirs_];
    TH1F *dxyPtWidthTrend[nDirs_];
    TH1F *dzPtWidthTrend[nDirs_];

    TH1F *dxyIntegralTrend[nDirs_];
    TH1F *dzIntegralTrend[nDirs_];

    bool areAllFilesOK = true;
    Int_t lastOpen = 0;

    // loop over the objects
    for (Int_t j = 0; j < nDirs_; j++) {
      //fins[j] = TFile::Open(Form("%s/PVValidation_%s_%i.root",dirs[j],dirs[j],intersection[n]));
      size_t position = std::string(dirs[j]).find('/');
      std::string stem = std::string(dirs[j]).substr(position + 1);  // get from position to the end

      fins[j] = new TFile(Form("%s/PVValidation_%s_%i.root", dirs[j], stem.c_str(), intersection[n]));
      if (fins[j]->IsZombie()) {
        logError << Form("%s/PVValidation_%s_%i.root", dirs[j], stem.c_str(), intersection[n])
                 << " is a Zombie! cannot combine" << std::endl;
        areAllFilesOK = false;
        lastOpen = j;
        break;
      }

      if (VERBOSE) {
        logInfo << Form("%s/PVValidation_%s_%i.root", dirs[j], stem.c_str(), intersection[n])
                << " has size: " << fins[j]->GetSize() << " b ";
      }

      // sanity check
      TH1F *h_tracks = (TH1F *)fins[j]->Get("PVValidation/EventFeatures/h_nTracks");
      Double_t numEvents = h_tracks->GetEntries();

      if (!doUnitTest) {
        if (numEvents < 2500) {
          logWarning << "excluding run " << intersection[n] << " because it has less than 2.5k events" << std::endl;
          areAllFilesOK = false;
          lastOpen = j;
          break;
        }
      } else {
        if (numEvents == 0) {
          logWarning << "excluding run " << intersection[n] << " because it has 0 events" << std::endl;
          areAllFilesOK = false;
          lastOpen = j;
          break;
        }
      }

      dxyPhiMeanTrend[j] = (TH1F *)fins[j]->Get("PVValidation/MeanTrends/means_dxy_phi");
      dxyPhiWidthTrend[j] = (TH1F *)fins[j]->Get("PVValidation/WidthTrends/widths_dxy_phi");
      dzPhiMeanTrend[j] = (TH1F *)fins[j]->Get("PVValidation/MeanTrends/means_dz_phi");
      dzPhiWidthTrend[j] = (TH1F *)fins[j]->Get("PVValidation/WidthTrends/widths_dz_phi");

      //dxyLadderMeanTrend[j] = (TH1F *)fins[j]->Get("PVValidation/MeanTrends/means_dxy_ladder");
      //dxyLadderWidthTrend[j] = (TH1F *)fins[j]->Get("PVValidation/WidthTrends/widths_dxy_ladder");
      //dzLadderMeanTrend[j] = (TH1F *)fins[j]->Get("PVValidation/MeanTrends/means_dz_ladder");
      //dzLadderWidthTrend[j] = (TH1F *)fins[j]->Get("PVValidation/WidthTrends/widths_dz_ladder");

      dxyEtaMeanTrend[j] = (TH1F *)fins[j]->Get("PVValidation/MeanTrends/means_dxy_eta");
      dxyEtaWidthTrend[j] = (TH1F *)fins[j]->Get("PVValidation/WidthTrends/widths_dxy_eta");
      dzEtaMeanTrend[j] = (TH1F *)fins[j]->Get("PVValidation/MeanTrends/means_dz_eta");
      dzEtaWidthTrend[j] = (TH1F *)fins[j]->Get("PVValidation/WidthTrends/widths_dz_eta");

      //dxyModZMeanTrend[j] = (TH1F *)fins[j]->Get("PVValidation/MeanTrends/means_dxy_modZ");
      //dxyModZWidthTrend[j] = (TH1F *)fins[j]->Get("PVValidation/WidthTrends/widths_dxy_modZ");
      //dzModZMeanTrend[j] = (TH1F *)fins[j]->Get("PVValidation/MeanTrends/means_dz_modZ");
      //dzModZWidthTrend[j] = (TH1F *)fins[j]->Get("PVValidation/WidthTrends/widths_dz_modZ");

      dxyNormPhiWidthTrend[j] = (TH1F *)fins[j]->Get("PVValidation/WidthTrends/norm_widths_dxy_phi");
      dxyNormEtaWidthTrend[j] = (TH1F *)fins[j]->Get("PVValidation/WidthTrends/norm_widths_dxy_eta");
      dzNormPhiWidthTrend[j] = (TH1F *)fins[j]->Get("PVValidation/WidthTrends/norm_widths_dz_phi");
      dzNormEtaWidthTrend[j] = (TH1F *)fins[j]->Get("PVValidation/WidthTrends/norm_widths_dz_eta");

      dxyNormPtWidthTrend[j] = (TH1F *)fins[j]->Get("PVValidation/WidthTrends/norm_widths_dxy_pTCentral");
      dzNormPtWidthTrend[j] = (TH1F *)fins[j]->Get("PVValidation/WidthTrends/norm_widths_dz_pTCentral");
      dxyPtWidthTrend[j] = (TH1F *)fins[j]->Get("PVValidation/WidthTrends/widths_dxy_pTCentral");
      dzPtWidthTrend[j] = (TH1F *)fins[j]->Get("PVValidation/WidthTrends/widths_dz_pTCentral");

      dxyIntegralTrend[j] = (TH1F *)fins[j]->Get("PVValidation/ProbeTrackFeatures/h_probedxyRefitV");
      dzIntegralTrend[j] = (TH1F *)fins[j]->Get("PVValidation/ProbeTrackFeatures/h_probedzRefitV");

      // fill the vectors of biases

      auto dxyPhiBiases = getBiases(dxyPhiMeanTrend[j]);

      //logInfo<<"\n" <<j<<" "<< LegLabels[j] << " dxy(phi) mean: "<< dxyPhiBiases.getWeightedMean()
      //       <<" dxy(phi) max: "<< dxyPhiBiases.getMax()
      //       <<" dxy(phi) min: "<< dxyPhiBiases.getMin()
      //       << std::endl;

      ret.m_dxyPhiMeans[LegLabels[j]].push_back(dxyPhiBiases.getWeightedMean());
      ret.m_dxyPhiChi2[LegLabels[j]].push_back(TMath::Log10(dxyPhiBiases.getNormChi2()));
      ret.m_dxyPhiKS[LegLabels[j]].push_back(dxyPhiBiases.getKSScore());

      //logInfo<<"\n" <<j<<" "<< LegLabels[j] << " dxy(phi) ks score: "<< dxyPhiBiases.getKSScore() << std::endl;

      useRMS
          ? ret.m_dxyPhiLo[LegLabels[j]].push_back(dxyPhiBiases.getWeightedMean() - 2 * dxyPhiBiases.getWeightedRMS())
          : ret.m_dxyPhiLo[LegLabels[j]].push_back(dxyPhiBiases.getMin());
      useRMS
          ? ret.m_dxyPhiHi[LegLabels[j]].push_back(dxyPhiBiases.getWeightedMean() + 2 * dxyPhiBiases.getWeightedRMS())
          : ret.m_dxyPhiHi[LegLabels[j]].push_back(dxyPhiBiases.getMax());

      auto dxyEtaBiases = getBiases(dxyEtaMeanTrend[j]);
      ret.m_dxyEtaMeans[LegLabels[j]].push_back(dxyEtaBiases.getWeightedMean());
      ret.m_dxyEtaChi2[LegLabels[j]].push_back(TMath::Log10(dxyEtaBiases.getNormChi2()));
      ret.m_dxyEtaKS[LegLabels[j]].push_back(dxyEtaBiases.getKSScore());
      useRMS
          ? ret.m_dxyEtaLo[LegLabels[j]].push_back(dxyEtaBiases.getWeightedMean() - 2 * dxyEtaBiases.getWeightedRMS())
          : ret.m_dxyEtaLo[LegLabels[j]].push_back(dxyEtaBiases.getMin());
      useRMS
          ? ret.m_dxyEtaHi[LegLabels[j]].push_back(dxyEtaBiases.getWeightedMean() + 2 * dxyEtaBiases.getWeightedRMS())
          : ret.m_dxyEtaHi[LegLabels[j]].push_back(dxyEtaBiases.getMax());

      auto dzPhiBiases = getBiases(dzPhiMeanTrend[j]);
      ret.m_dzPhiMeans[LegLabels[j]].push_back(dzPhiBiases.getWeightedMean());
      ret.m_dzPhiChi2[LegLabels[j]].push_back(TMath::Log10(dzPhiBiases.getNormChi2()));
      ret.m_dzPhiKS[LegLabels[j]].push_back(dzPhiBiases.getKSScore());
      useRMS ? ret.m_dzPhiLo[LegLabels[j]].push_back(dzPhiBiases.getWeightedMean() - 2 * dzPhiBiases.getWeightedRMS())
             : ret.m_dzPhiLo[LegLabels[j]].push_back(dzPhiBiases.getMin());
      useRMS ? ret.m_dzPhiHi[LegLabels[j]].push_back(dzPhiBiases.getWeightedMean() + 2 * dzPhiBiases.getWeightedRMS())
             : ret.m_dzPhiHi[LegLabels[j]].push_back(dzPhiBiases.getMax());

      auto dzEtaBiases = getBiases(dzEtaMeanTrend[j]);
      ret.m_dzEtaMeans[LegLabels[j]].push_back(dzEtaBiases.getWeightedMean());
      ret.m_dzEtaChi2[LegLabels[j]].push_back(TMath::Log10(dzEtaBiases.getNormChi2()));
      ret.m_dzEtaKS[LegLabels[j]].push_back(dzEtaBiases.getKSScore());
      useRMS ? ret.m_dzEtaLo[LegLabels[j]].push_back(dzEtaBiases.getWeightedMean() - 2 * dzEtaBiases.getWeightedRMS())
             : ret.m_dzEtaLo[LegLabels[j]].push_back(dzEtaBiases.getMin());
      useRMS ? ret.m_dzEtaHi[LegLabels[j]].push_back(dzEtaBiases.getWeightedMean() + 2 * dzEtaBiases.getWeightedRMS())
             : ret.m_dzEtaHi[LegLabels[j]].push_back(dzEtaBiases.getMax());

      // unrolled histograms
      ret.m_dxyVect[LegLabels[j]].push_back(getUnrolledHisto(dxyIntegralTrend[j]));
      ret.m_dzVect[LegLabels[j]].push_back(getUnrolledHisto(dzIntegralTrend[j]));
    }

    if (!areAllFilesOK) {
      // do all the necessary deletions
      logWarning << "====> not all files are OK" << std::endl;

      for (int i = 0; i < lastOpen; i++) {
        fins[i]->Close();
      }
      continue;
    } else {
      ret.m_runs.push_back(intersection.at(n));
    }

    if (VERBOSE) {
      logInfo << "I am still here - runs.size(): " << ret.m_runs.size() << std::endl;
    }

    // do all the necessary deletions

    for (int i = 0; i < nDirs_; i++) {
      delete dxyPhiMeanTrend[i];
      delete dzPhiMeanTrend[i];
      delete dxyEtaMeanTrend[i];
      delete dzEtaMeanTrend[i];

      delete dxyPhiWidthTrend[i];
      delete dzPhiWidthTrend[i];
      delete dxyEtaWidthTrend[i];
      delete dzEtaWidthTrend[i];

      delete dxyNormPhiWidthTrend[i];
      delete dxyNormEtaWidthTrend[i];
      delete dzNormPhiWidthTrend[i];
      delete dzNormEtaWidthTrend[i];

      delete dxyNormPtWidthTrend[i];
      delete dzNormPtWidthTrend[i];
      delete dxyPtWidthTrend[i];
      delete dzPtWidthTrend[i];

      fins[i]->Close();
    }

    if (VERBOSE) {
      logInfo << std::endl;
    }
  }

  return ret;
}
