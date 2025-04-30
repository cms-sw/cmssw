#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>

#include "toolbox.h"
#include "Options.h"

#include "TAxis.h"
#include "TBranch.h"
#include "TCanvas.h"
#include "TChain.h"
#include "TCut.h"
#include "TF1.h"
#include "TFile.h"
#include "TFrame.h"
#include "TH1.h"
#include "TH1D.h"
#include "TH1F.h"
#include "TH2.h"
#include "TH2F.h"
#include "THStack.h"
#include "TLatex.h"
#include "TLegend.h"
#include "TMath.h"
#include "TPad.h"
#include "TPaveText.h"
#include "TRandom.h"
#include "TString.h"
#include "TStyle.h"
#include "TSystem.h"
#include "TTree.h"
//#include "RooGlobalFunc.h"

#include <boost/filesystem.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>
#include <boost/range/adaptor/indexed.hpp>

#include "Alignment/OfflineValidation/interface/FitWithRooFit.h"
#include "Alignment/OfflineValidation/macros/CMS_lumi.h"

using namespace RooFit;
using namespace std;
using namespace AllInOneConfig;
namespace pt = boost::property_tree;
/*--------------------------------------------------------------------*/
void makeNicePlotStyle(RooPlot* plot) {
  plot->GetXaxis()->CenterTitle(true);
  plot->GetYaxis()->CenterTitle(true);
  plot->GetXaxis()->SetTitleFont(42);
  plot->GetYaxis()->SetTitleFont(42);
  plot->GetXaxis()->SetTitleSize(0.05);
  plot->GetYaxis()->SetTitleSize(0.05);
  plot->GetXaxis()->SetTitleOffset(0.9);
  plot->GetYaxis()->SetTitleOffset(1.3);
  plot->GetXaxis()->SetLabelFont(42);
  plot->GetYaxis()->SetLabelFont(42);
  plot->GetYaxis()->SetLabelSize(.05);
  plot->GetXaxis()->SetLabelSize(.05);
}
/*--------------------------------------------------------------------*/

RooRealVar MuMu_mass("MuMu_mass", "MuMu_mass", 70, 110);
static TString GT = "";
TLatex* tlxg = new TLatex();
class FitOut {
public:
  double mean;
  double mean_err;
  double sigma;
  double sigma_err;
  double chi2;
  FitOut(double a, double b, double c, double d) : mean(a), mean_err(b), sigma(c), sigma_err(d) {}
};

FitOut ZMassBinFit_OldTool(TH1D* th1d_input, TString s_name = "zmumu_fitting", TString output_path = "./") {
  // silence messages
  RooMsgService::instance().setGlobalKillBelow(RooFit::FATAL);

  double xMean = 91.1876;
  double xMin = th1d_input->GetXaxis()->GetXmin();
  double xMax = th1d_input->GetXaxis()->GetXmax();

  double sigma(2.);
  double sigmaMin(0.1);
  double sigmaMax(10.);

  double sigma2(0.1);
  double sigma2Min(0.);
  double sigma2Max(10.);

  std::unique_ptr<FitWithRooFit> fitter = std::make_unique<FitWithRooFit>();

  bool useChi2(false);

  fitter->useChi2_ = useChi2;
  fitter->initMean(xMean, xMin, xMax);
  fitter->initSigma(sigma, sigmaMin, sigmaMax);
  fitter->initSigma2(sigma2, sigma2Min, sigma2Max);
  fitter->initAlpha(1.5, 0.05, 10.);
  fitter->initN(1, 0.01, 100.);
  fitter->initFGCB(0.4, 0., 1.);
  fitter->initGamma(2.4952, 0., 10.);
  fitter->gamma()->setConstant(kTRUE);
  fitter->initMean2(0., -20., 20.);
  fitter->mean2()->setConstant(kTRUE);
  fitter->initSigma(1.2, 0., 5.);
  fitter->initAlpha(1.5, 0.05, 10.);
  fitter->initN(1, 0.01, 100.);
  fitter->initExpCoeffA0(-1., -10., 10.);
  fitter->initExpCoeffA1(0., -10., 10.);
  fitter->initExpCoeffA2(0., -2., 2.);
  fitter->initFsig(0.9, 0., 1.);
  fitter->initA0(0., -10., 10.);
  fitter->initA1(0., -10., 10.);
  fitter->initA2(0., -10., 10.);
  fitter->initA3(0., -10., 10.);
  fitter->initA4(0., -10., 10.);
  fitter->initA5(0., -10., 10.);
  fitter->initA6(0., -10., 10.);

  TCanvas* c1 = new TCanvas();
  c1->Clear();
  c1->SetLeftMargin(0.15);
  c1->SetRightMargin(0.10);

  fitter->fit(th1d_input, "breitWignerTimesCB", "exponential", xMin, xMax, false);

  c1->Print(Form("%s/fitResultPlot/%s_oldtool.pdf", output_path.Data(), s_name.Data()));
  c1->Print(Form("%s/fitResultPlot/%s_oldtool.root", output_path.Data(), s_name.Data()));

  FitOut fitRes(
      fitter->mean()->getVal(), fitter->mean()->getError(), fitter->sigma()->getVal(), fitter->sigma()->getError());

  return fitRes;
}
void Draw_th1d(TH1D* th1d_input, TString variable_name, TString output_path) {
  TCanvas* c = new TCanvas();
  c->cd();
  gStyle->SetOptStat(0);
  th1d_input->SetMarkerStyle(kFullCircle);
  th1d_input->SetMarkerColor(kRed);
  th1d_input->SetLineColor(kRed);
  th1d_input->SetMaximum(91.4);
  th1d_input->SetMinimum(90.85);
  th1d_input->GetXaxis()->SetTitle(variable_name.Data());
  th1d_input->GetXaxis()->SetTitleOffset(1.2);
  th1d_input->GetYaxis()->SetTitle("Mass mean (GeV)");
  th1d_input->Draw();
  tlxg->DrawLatexNDC(0.2, 0.8, Form("%s", GT.Data()));
  c->Print(Form("%s/fitResultPlot/mass_VS_%s.pdf", output_path.Data(), variable_name.Data()));
}

const static int variables_number = 8;
const TString tstring_variables_name[variables_number] = {
    "CosThetaCS", "DeltaEta", "EtaMinus", "EtaPlus", "PhiCS", "PhiMinus", "PhiPlus", "Pt"};
const TString tstring_variables_name_label[variables_number] = {"cos #theta_{CS}",
                                                                "#Delta #eta",
                                                                "#eta_{#mu^{-}}",
                                                                "#eta_{#mu^{+}}",
                                                                "#phi_{CS}",
                                                                "#phi_{#mu^{-}}",
                                                                "#phi_{#mu^{+}}",
                                                                "p_{T}"};

void Fitting_GetMassmeanVSvariables(TString inputfile_name, TString output_path) {
  TH2D* th2d_mass_variables[variables_number];
  TFile* inputfile = TFile::Open(inputfile_name.Data());
  TDirectoryFile* tdirectory = (TDirectoryFile*)inputfile->Get("DiMuonMassValidation");
  for (int i = 0; i < variables_number; i++) {
    TString th2d_name = Form("th2d_mass_%s", tstring_variables_name[i].Data());
    th2d_mass_variables[i] = (TH2D*)tdirectory->Get(th2d_name);
  }

  gSystem->Exec(Form("mkdir -p %s", output_path.Data()));
  gSystem->Exec(Form("mkdir -p %s/fitResultPlot", output_path.Data()));
  TFile* outputfile = TFile::Open(Form("%s/fitting_output.root", output_path.Data()), "RECREATE");
  TH1D* th1d_variables_meanmass[variables_number];
  TH1D* th1d_variables_entries[variables_number];
  const int variables_rebin[variables_number] = {1, 1, 1, 1, 1, 1, 1, 1};
  const double xaxis_range[variables_number][2] = {
      {-1, 1}, {-4.8, 4.8}, {-2.4, 2.4}, {-2.4, 2.4}, {-1, 1}, {-M_PI, M_PI}, {-M_PI, M_PI}, {0, 100}};
  for (int i = 0; i < variables_number; i++) {
    TString th1d_name = Form("th1d_meanmass_%s", tstring_variables_name[i].Data());

    th2d_mass_variables[i]->RebinY(variables_rebin[i]);
    th1d_variables_meanmass[i] = th2d_mass_variables[i]->ProjectionY(th1d_name, 1, 1, "d");
    for (int j = 0; j < th1d_variables_meanmass[i]->GetNbinsX(); j++) {
      if (i == 7 and j > 25) {
        continue;
      }
      std::cout << __PRETTY_FUNCTION__
                << " th1d_variables_meanmass[i]->GetNbinsX()=" << th1d_variables_meanmass[i]->GetNbinsX() << endl;
      std::cout << __PRETTY_FUNCTION__ << " th2d_mass_variables[i]->GetNbinsY()=" << th2d_mass_variables[i]->GetNbinsY()
                << endl;
      th1d_variables_meanmass[i]->SetBinContent(j, 0);
      th1d_variables_meanmass[i]->SetBinError(j, 0);

      TString th1d_mass_temp_name = Form("th1d_mass_%s_%d", tstring_variables_name[i].Data(), j);
      TH1D* th1d_i = th2d_mass_variables[i]->ProjectionX(th1d_mass_temp_name, j, j, "d");
      th1d_i->Write(th1d_mass_temp_name);
      TString s_cut = Form("nocut");
      TString s_name = Form("%s_%d", tstring_variables_name[i].Data(), j);

      FitOut fitR = ZMassBinFit_OldTool(th1d_i, s_name, output_path);

      th1d_variables_meanmass[i]->SetBinContent(j, fitR.mean);
      th1d_variables_meanmass[i]->SetBinError(j, fitR.mean_err);
    }

    th1d_variables_meanmass[i]->GetXaxis()->SetRangeUser(xaxis_range[i][0], xaxis_range[i][1]);
    Draw_th1d(th1d_variables_meanmass[i], tstring_variables_name[i], output_path);
    outputfile->cd();
    th1d_variables_meanmass[i]->Write(th1d_name);

    TString th1d_name_entries = Form("th1d_entries_%s", tstring_variables_name[i].Data());
    th1d_variables_entries[i] = th2d_mass_variables[i]->ProjectionY(th1d_name_entries, 0, -1, "d");
    th1d_variables_entries[i]->GetXaxis()->SetTitle(tstring_variables_name[i].Data());
    th1d_variables_entries[i]->GetYaxis()->SetTitle("Entry");
    outputfile->cd();
    th1d_variables_entries[i]->Write(th1d_name_entries);
  }

  if (outputfile->IsOpen()) {
    // Get the path (current working directory) in which the file is going to be written
    const char* path = outputfile->GetPath();

    if (path) {
      std::cout << "File is going to be written in the directory: " << path << " for input file: " << inputfile_name
                << std::endl;
    } else {
      std::cerr << "Error: Unable to determine the path." << std::endl;
    }
    outputfile->Close();
    delete outputfile;
  }
}

const static int max_file_number = 10;
void Draw_TH1D_forMultiRootFiles(const vector<TString>& file_names,
                                 const vector<TString>& label_names,
                                 const vector<int>& colors,
                                 const vector<int>& styles,
                                 const TString& Rlabel,
                                 const TString& th1d_name,
                                 const TString& xlabel,
                                 const TString& ylabel,
                                 const TString& output_name) {
  if (file_names.empty() || label_names.empty()) {
    std::cout << "Provided an empty list of file and label names" << std::endl;
    return;
  }

  // do not allow the list of files and labels names to differ
  assert(file_names.size() == label_names.size());

  TH1D* th1d_input[max_file_number];
  TFile* file_input[max_file_number];
  for (auto const& filename : file_names | boost::adaptors::indexed(0)) {
    file_input[filename.index()] = TFile::Open(filename.value());
    th1d_input[filename.index()] = (TH1D*)file_input[filename.index()]->Get(th1d_name);
    th1d_input[filename.index()]->SetTitle("");
  }

  int W = 800;
  int H = 800;
  // references for T, B, L, R
  float T = 0.08 * H;
  float B = 0.12 * H;
  float L = 0.12 * W;
  float R = 0.04 * W;

  // Form the canvas name by appending th1d_name
  TString canvasName;
  canvasName.Form("canv_%s", th1d_name.Data());

  // Create a new canvas with the formed name
  TCanvas* canv = new TCanvas(canvasName, canvasName, W, H);
  canv->SetFillColor(0);
  canv->SetBorderMode(0);
  canv->SetFrameFillStyle(0);
  canv->SetFrameBorderMode(0);
  canv->SetLeftMargin(L / W + 0.05);
  canv->SetRightMargin(R / W);
  canv->SetTopMargin(T / H);
  canv->SetBottomMargin(B / H);
  canv->SetTickx(0);
  canv->SetTicky(0);
  canv->SetGrid();
  canv->cd();

  gStyle->SetOptStat(0);

  TLegend* lg = new TLegend(0.3, 0.7, 0.7, 0.9);
  lg->SetFillStyle(0);
  lg->SetLineColor(0);
  lg->SetEntrySeparation(0.05);

  double ymin = 0.;
  double ymax = 0.;

  for (auto const& labelname : label_names | boost::adaptors::indexed(0)) {
    double temp_ymin = th1d_input[labelname.index()]->GetMinimum();
    double temp_ymax = th1d_input[labelname.index()]->GetMaximum();
    if (labelname.index() == 0) {
      ymin = temp_ymin;
      ymax = temp_ymax;
    }
    if (temp_ymin <= ymin) {
      ymin = temp_ymin;
    }
    if (temp_ymax >= ymax) {
      ymax = temp_ymax;
    }
  }

  for (auto const& labelname : label_names | boost::adaptors::indexed(0)) {
    th1d_input[labelname.index()]->SetMarkerColor(colors[labelname.index()]);
    th1d_input[labelname.index()]->SetLineColor(colors[labelname.index()]);
    th1d_input[labelname.index()]->SetMarkerStyle(styles[labelname.index()]);
    th1d_input[labelname.index()]->GetXaxis()->SetTitle(xlabel);
    th1d_input[labelname.index()]->GetYaxis()->SetTitle(ylabel);
    th1d_input[labelname.index()]->GetYaxis()->SetTitleOffset(2.0);
    lg->AddEntry(th1d_input[labelname.index()], labelname.value());

    TString label_meanmass_plot = "Mean M_{#mu#mu} (GeV)";
    if (ylabel.EqualTo(label_meanmass_plot)) {
      double ycenter = (ymax + ymin) / 2.0;
      double yrange = (ymax - ymin) * 2;
      double Ymin = ycenter - yrange;
      double Ymax = ycenter + yrange * 1.10;
      th1d_input[labelname.index()]->SetAxisRange(Ymin, Ymax, "Y");
      th1d_input[labelname.index()]->Draw("PEX0same");
    } else {
      double Ymin = ymin - ymin * 0.07;
      double Ymax = ymax + ymax * 0.35;
      th1d_input[labelname.index()]->SetAxisRange(Ymin, Ymax, "Y");
      th1d_input[labelname.index()]->Draw("HIST E1 same");
    }
  }

  CMS_lumi(canv, 0, 3, Rlabel);

  lg->Draw("same");

  canv->Update();
  canv->RedrawAxis();
  canv->GetFrame()->Draw();
  canv->SaveAs(output_name);

  if (output_name.Contains(".pdf")) {
    TString output_name_png(output_name);  // output_name is const, copy to modify
    output_name_png.Replace(output_name_png.Index(".pdf"), 4, ".png");
    canv->SaveAs(output_name_png);
  }
}

int Zmumumerge(int argc, char* argv[]) {
  vector<TString> vec_single_file_path;
  vector<TString> vec_single_file_name;
  vector<TString> vec_global_tag;
  vector<TString> vec_title;
  vector<int> vec_color;
  vector<int> vec_style;
  vector<TString> vec_right_title;

  Options options;
  options.helper(argc, argv);
  options.parser(argc, argv);
  pt::ptree main_tree;
  pt::read_json(options.config, main_tree);
  pt::ptree alignments = main_tree.get_child("alignments");
  pt::ptree validation = main_tree.get_child("validation");

  //Load defined order
  std::vector<std::pair<std::string, pt::ptree>> alignmentsOrdered;
  for (const auto& childTree : alignments) {
    alignmentsOrdered.push_back(childTree);
  }
  std::sort(alignmentsOrdered.begin(),
            alignmentsOrdered.end(),
            [](const std::pair<std::string, pt::ptree>& left, const std::pair<std::string, pt::ptree>& right) {
              return left.second.get<int>("index") < right.second.get<int>("index");
            });

  for (const auto& childTree : alignmentsOrdered) {
    // do not consider the nodes with a "file" to merge
    if (childTree.second.find("file") == childTree.second.not_found()) {
      std::cerr << "Ignoring alignment: " << childTree.second.get<std::string>("title") << ".\nNo file to merged found!"
                << std::endl;
      continue;
    } else {
      std::cout << "Storing alignment: " << childTree.second.get<std::string>("title") << std::endl;
    }
    vec_single_file_path.push_back(childTree.second.get<std::string>("file"));
    vec_single_file_name.push_back(childTree.second.get<std::string>("file") + "/Zmumu.root");
    vec_color.push_back(childTree.second.get<int>("color"));
    vec_style.push_back(childTree.second.get<int>("style"));
    if (childTree.second.find("customrighttitle") == childTree.second.not_found()) {
      vec_right_title.push_back("");
    } else {
      vec_right_title.push_back(childTree.second.get<std::string>("customrighttitle"));
    }
    vec_global_tag.push_back(childTree.second.get<std::string>("globaltag"));
    vec_title.push_back(childTree.second.get<std::string>("title"));

    //Fitting_GetMassmeanVSvariables(childTree.second.get<std::string>("file") + "/Zmumu.root", childTree.second.get<std::string>("file"));
  }

  TString merge_output = main_tree.get<std::string>("output");
  //=============================================
  vector<TString> vec_single_fittingoutput;
  vec_single_fittingoutput.clear();
  for (unsigned i = 0; i < vec_single_file_path.size(); i++) {
    Fitting_GetMassmeanVSvariables(vec_single_file_name[i], vec_single_file_path[i]);
    vec_single_fittingoutput.push_back(vec_single_file_path[i] + "/fitting_output.root");
  }

  int files_number = vec_single_file_path.size();
  cout << "files_number=" << files_number << endl;
  for (int idx_variable = 0; idx_variable < variables_number; idx_variable++) {
    TString th1d_name = Form("th1d_meanmass_%s", tstring_variables_name[idx_variable].Data());

    Draw_TH1D_forMultiRootFiles(
        vec_single_fittingoutput,
        vec_title,
        vec_color,
        vec_style,
        vec_right_title[0],
        th1d_name,
        tstring_variables_name_label[idx_variable].Data(),
        "Mean M_{#mu#mu} (GeV)",
        merge_output + Form("/meanmass_%s_GTs.pdf", tstring_variables_name[idx_variable].Data()));

    TString th1d_name_entries = Form("th1d_entries_%s", tstring_variables_name[idx_variable].Data());

    Draw_TH1D_forMultiRootFiles(
        vec_single_fittingoutput,
        vec_title,
        vec_color,
        vec_style,
        vec_right_title[0],
        th1d_name_entries,
        tstring_variables_name_label[idx_variable].Data(),
        "Events",
        merge_output + Form("/entries_%s_GTs.pdf", tstring_variables_name[idx_variable].Data()));
  }

  //=============================================
  return EXIT_SUCCESS;
}
#ifndef DOXYGEN_SHOULD_SKIP_THIS
int main(int argc, char* argv[]) { return Zmumumerge(argc, argv); }
#endif
-- dummy change --
