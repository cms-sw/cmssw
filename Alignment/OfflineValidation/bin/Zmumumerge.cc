#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>

#include "toolbox.h"
#include "FitWithRooFit.cc"
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
  double xMin(75), xMax(105), xMean(91);
  double sigma = 2;
  double sigmaMin = 0.1;
  double sigmaMax = 10;

  FitWithRooFit* fitter = new FitWithRooFit();
  double sigma2(0.1), sigma2Min(0.), sigma2Max(10.), useChi2(false);
  fitter->useChi2_ = useChi2;
  fitter->initMean(xMean, xMin, xMax);
  fitter->initSigma(sigma, sigmaMin, sigmaMax);
  fitter->initSigma2(sigma2, sigma2Min, sigma2Max);
  fitter->initAlpha(1.5, 0.05, 10.);
  fitter->initN(1, 0.01, 100.);
  fitter->initFGCB(0.4, 0., 1.);

  fitter->initMean(91.1876, xMin, xMax);
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
      fitter->mean()->getValV(), fitter->mean()->getError(), fitter->sigma()->getValV(), fitter->sigma()->getError());
  return fitRes;
}
void Draw_th1d(TH1D* th1d_input, TString variable_name) {
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
  c->Print(Form("%s/fitResultPlot/mass_VS_%s.pdf", GT.Data(), variable_name.Data()));
}

const static int variables_number = 8;
const TString tstring_variables_name[variables_number] = {
    "CosThetaCS", "DeltaEta", "EtaMinus", "EtaPlus", "PhiCS", "PhiMinus", "PhiPlus", "Pt"};
void Fitting_GetMassmeanVSvariables(TString inputfile_name, TString output_path) {
  TH2D* th2d_mass_variables[variables_number];
  TFile* inputfile = TFile::Open(inputfile_name.Data());
  TDirectoryFile* tdirectory = (TDirectoryFile*)inputfile->Get("myanalysis");
  for (int i = 0; i < variables_number; i++) {
    TString th2d_name = Form("th2d_mass_%s", tstring_variables_name[i].Data());
    th2d_mass_variables[i] = (TH2D*)tdirectory->Get(th2d_name);
  }

  gSystem->Exec(Form("mkdir -p %s", output_path.Data()));
  gSystem->Exec(Form("mkdir -p %s/fitResultPlot", output_path.Data()));
  TFile* outpufile = TFile::Open(Form("%s/fitting_output.root", output_path.Data()), "recreate");
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
      cout << "th1d_variables_meanmass[i]->GetNbinsX()=" << th1d_variables_meanmass[i]->GetNbinsX() << endl;
      cout << "th2d_mass_variables[i]->GetNbinsY()=" << th2d_mass_variables[i]->GetNbinsY() << endl;
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
    Draw_th1d(th1d_variables_meanmass[i], tstring_variables_name[i]);
    outpufile->cd();
    th1d_variables_meanmass[i]->Write(th1d_name);

    TString th1d_name_entries = Form("th1d_entries_%s", tstring_variables_name[i].Data());
    th1d_variables_entries[i] = th2d_mass_variables[i]->ProjectionY(th1d_name_entries, 0, -1, "d");
    th1d_variables_entries[i]->GetXaxis()->SetTitle(tstring_variables_name[i].Data());
    th1d_variables_entries[i]->GetYaxis()->SetTitle("Entry");
    outpufile->cd();
    th1d_variables_entries[i]->Write(th1d_name_entries);
  }

  outpufile->Write();
  outpufile->Close();
  delete outpufile;
}

const static int max_file_number = 10;
void Draw_TH1D_forMultiRootFiles(const vector<TString>& file_names,
                                 const vector<TString>& label_names,
                                 const vector<int>& colors,
                                 const vector<int>& styles,
                                 const TString& th1d_name,
                                 const TString& output_name) {
  if (file_names.empty() || label_names.empty()) {
    cout << "Provided an empty list of file and label names" << std::endl;
    return;
  }

  // do not allow the list of files and labels names to differ
  assert(file_names.size() == label_names.size());

  TH1D* th1d_input[max_file_number];
  TFile* file_input[max_file_number];
  for (auto const& filename : file_names | boost::adaptors::indexed(0)) {
    file_input[filename.index()] = TFile::Open(filename.value());
    th1d_input[filename.index()] = (TH1D*)file_input[filename.index()]->Get(th1d_name);
  }

  TCanvas* c = new TCanvas();
  TLegend* lg = new TLegend(0.2, 0.7, 0.5, 0.95);
  c->cd();
  gStyle->SetOptStat(0);
  th1d_input[0]->SetTitle("");

  for (auto const& labelname : label_names | boost::adaptors::indexed(0)) {
    th1d_input[labelname.index()]->SetMarkerColor(colors[labelname.index()]);
    th1d_input[labelname.index()]->SetLineColor(colors[labelname.index()]);
    th1d_input[labelname.index()]->SetMarkerStyle(styles[labelname.index()]);
    th1d_input[labelname.index()]->Draw("same");
    lg->AddEntry(th1d_input[labelname.index()], labelname.value());
  }
  lg->Draw("same");
  c->SaveAs(output_name);
}

int Zmumumerge(int argc, char* argv[]) {
  vector<TString> vec_single_file_path;
  vector<TString> vec_single_file_name;
  vector<TString> vec_global_tag;
  vector<int> vec_color;
  vector<int> vec_style;

  Options options;
  options.helper(argc, argv);
  options.parser(argc, argv);
  pt::ptree main_tree;
  pt::read_json(options.config, main_tree);
  pt::ptree alignments = main_tree.get_child("alignments");
  pt::ptree validation = main_tree.get_child("validation");
  for (const auto& childTree : alignments) {
    vec_single_file_path.push_back(childTree.second.get<std::string>("file"));
    vec_single_file_name.push_back(childTree.second.get<std::string>("file") + "/Zmumu.root");
    vec_color.push_back(childTree.second.get<int>("color"));
    vec_style.push_back(childTree.second.get<int>("style"));
    vec_global_tag.push_back(childTree.second.get<std::string>("globaltag"));

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
        vec_global_tag,
        vec_color,
        vec_style,
        th1d_name,
        merge_output + Form("/meanmass_%s_GTs.pdf", tstring_variables_name[idx_variable].Data()));
    TString th1d_name_entries = Form("th1d_entries_%s", tstring_variables_name[idx_variable].Data());
    Draw_TH1D_forMultiRootFiles(
        vec_single_fittingoutput,
        vec_global_tag,
        vec_color,
        vec_style,
        th1d_name_entries,
        merge_output + Form("/entries_%s_GTs.pdf", tstring_variables_name[idx_variable].Data()));
  }
  //=============================================
  return EXIT_SUCCESS;
}
#ifndef DOXYGEN_SHOULD_SKIP_THIS
int main(int argc, char* argv[]) { return Zmumumerge(argc, argv); }
#endif
