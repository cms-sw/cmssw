// STL headers
#include <iostream>
#include <iomanip>
#include <sstream>

// ROOT headers
#include <TStyle.h>
#include <TCanvas.h>
#include <TTree.h>
#include <TString.h>
#include <TAxis.h>
#include <TProfile.h>
#include <TF1.h>
#include <TH1.h>
#include <TH2F.h>
#include <TGraphErrors.h>
#include <TROOT.h>
#include <TDirectory.h>
#include <TFile.h>
#include <TDirectoryFile.h>
#include <TLegend.h>
#include <TChain.h>
#include <TMath.h>
#include <TLatex.h>
#include <TVirtualFitter.h>
#include <TLorentzVector.h>
#include <TMatrixD.h>
#include <vector>

// CMSSW headers
#include "Alignment/OfflineValidation/interface/EopElecVariables.h"

// New enumeration for kind of plots
enum ModeType { TRACK_ETA, TRACK_PHI };

// New structure for storing information relative to one file
struct HistoType {
  TString label;
  TFile* file;
  TTree* tree;

  std::vector<Double_t> etaRange;
  std::vector<Double_t> zRange;
  std::vector<Double_t> eRange;

  // Basic histograms
  // dimension = numberOfBins x numberOfSteps
  UInt_t** selectedTracks;
  TH1F*** xAxisBin;
  TH1F*** negative;
  TH1F*** positive;
  TH2F*** Enegative;
  TH2F*** Epositive;

  // Intermediate histograms
  TH1F** fit;
  TH1F** combinedxAxisBin;

  // Final histograms
  TGraphErrors* overallGraph;
  TH1F* overallhisto;

  // Final fit
  TF1* f2;
  TString misalignmentfit;

  HistoType() {
    label = "";
    file = 0;
    tree = 0;
    overallGraph = 0;
    overallhisto = 0;
    f2 = 0;
    misalignmentfit = "";
  }

  ~HistoType() {}
};

// Prototype of functions
void initializeHistograms(ModeType mode, HistoType& histos);
Bool_t checkArguments(TString variable,  //input
                      TString path,
                      TString alignmentWithLabel,
                      TString outputType,
                      Double_t radius,
                      Bool_t verbose,
                      Double_t givenMin,
                      Double_t givenMax,
                      ModeType& mode,  // ouput
                      std::vector<TFile*>& files,
                      std::vector<TString>& labels);
void configureROOTstyle(Bool_t verbose);
Bool_t initializeTree(std::vector<TFile*>& files, std::vector<TTree*>& trees, EopElecVariables* track);
void readTree(ModeType mode, TString variable, HistoType& histo, EopElecVariables* track, Double_t radius);
void fillIntermediateHisto(
    ModeType mode, Bool_t verbose, HistoType& histo, TCanvas* ccontrol, TString outputType, Double_t radius);
void fillFinalHisto(ModeType mode, Bool_t verbose, HistoType& histo);
void layoutFinalPlot(ModeType mode,
                     std::vector<HistoType>& histos,
                     TString outputType,
                     TString variable,
                     TCanvas* c,
                     Double_t givenMin,
                     Double_t givenMax);

std::vector<Double_t> extractgausparams(TH1F* histo, TString option1, TString option2);

// -----------------------------------------------------------------------------
//
//    Main function : momentumElectronBiasValidation
//
// -----------------------------------------------------------------------------
void momentumElectronBiasValidation(TString variable,
                                    TString path,
                                    TString alignmentWithLabel,
                                    TString outputType,
                                    Double_t radius = 1.,
                                    Bool_t verbose = false,
                                    Double_t givenMin = 0.,
                                    Double_t givenMax = 0.) {
  // Displaying first message
  std::cout << "!!! Welcome to momentumElectronBiasValidation !!!" << std::endl;
  std::cout << std::endl;
  time_t start = time(0);

  // Checking and decoding the arguments
  std::cout << "Checking arguments ..." << std::endl;
  ModeType mode;                // variable to study
  std::vector<TFile*> files;    // list of input files
  std::vector<TString> labels;  // list of input labels
  if (!checkArguments(
          variable, path, alignmentWithLabel, outputType, radius, verbose, givenMin, givenMax, mode, files, labels))
    return;
  else {
    std::cout << "-> Number of files: " << files.size() << std::endl;
  }

  // Initializing the TTree
  std::cout << "Initializing the TTree ..." << std::endl;
  std::vector<TTree*> trees(files.size(), 0);
  EopElecVariables* track = new EopElecVariables();
  if (!initializeTree(files, trees, track)) {
    delete track;
    return;
  }

  // Configuring ROOT style
  std::cout << "Configuring the ROOT style ..." << std::endl;
  configureROOTstyle(verbose);

  TCanvas* c = new TCanvas("cElectron", "Canvas", 0, 0, 1150, 800);
  TCanvas* ccontrol = new TCanvas("ccontrolElectron", "controlCanvas", 0, 0, 600, 300);

  // Creating histos
  std::vector<HistoType> histos(files.size());

  // Loop over files

  // To save the table cut
  TFile* histo1 = nullptr;
  std::string fileName;

  for (unsigned int ifile = 0; ifile < histos.size(); ifile++) {
    HistoType& histo = histos[ifile];

    // setting labels
    histo.label = labels[ifile];
    histo.file = files[ifile];
    histo.tree = trees[ifile];

    fileName = histo.file->GetName();
    fileName = fileName.substr(path.Sizeof() - 1);
    fileName = "Plots" + fileName;

    std::cout << "NAME :" << fileName << std::endl;
    histo1 = new TFile(fileName.c_str(), "RECREATE");

    // Getting and saving cut flow values from tree producer step
    TH1D* Nevents;
    TH1D* NeventsTriggered;
    TH1D* Nevents2elec;
    TH1D* Ntracks;
    TH1D* NtracksFiltered;
    TH1D* NtracksFirstPtcut;
    TH1D* NtracksOneSC;

    Nevents = dynamic_cast<TH1D*>(histo.file->Get("energyOverMomentumTree/nEvents"));
    NeventsTriggered = dynamic_cast<TH1D*>(histo.file->Get("energyOverMomentumTree/nEventsTriggered"));
    Nevents2elec = dynamic_cast<TH1D*>(histo.file->Get("energyOverMomentumTree/nEvents2Elec"));
    Ntracks = dynamic_cast<TH1D*>(histo.file->Get("energyOverMomentumTree/nTracks"));
    NtracksFiltered = dynamic_cast<TH1D*>(histo.file->Get("energyOverMomentumTree/nTracksFiltered"));
    NtracksFirstPtcut = dynamic_cast<TH1D*>(histo.file->Get("energyOverMomentumTree/cut_Ptmin"));
    NtracksOneSC = dynamic_cast<TH1D*>(histo.file->Get("energyOverMomentumTree/cut_OneSCmatch"));

    Nevents->Write();
    NeventsTriggered->Write();
    Nevents2elec->Write();
    Ntracks->Write();
    NtracksFiltered->Write();
    NtracksFirstPtcut->Write();
    NtracksOneSC->Write();

    // Initializing the range in Energy
    histo.eRange.clear();
    histo.eRange.push_back(30);
    histo.eRange.push_back(40);
    histo.eRange.push_back(60);
    //histo.eRange.push_back(100);

    // Dump at screen range Energy
    if (ifile == 0) {
      std::cout << "Range used for energy : ";
      for (unsigned int i = 0; i < (histo.eRange.size() - 1); i++)
        std::cout << "[" << histo.eRange[i] << "]-[" << histo.eRange[i + 1] << "] ";
      std::cout << std::endl;
    }

    // Initializing the range in Phi
    if (mode == TRACK_ETA) {
      Double_t begin = -0.9;  //-1.40;//-1.65;
      Double_t end = 0.9;     //1.40;//1.65;
      //      UInt_t   nsteps  = 18;
      UInt_t nsteps = 11;  //8;
      Double_t binsize = (end - begin) / static_cast<Float_t>(nsteps);

      histo.etaRange.resize(nsteps + 1, 0.);
      histo.zRange.resize(nsteps + 1, 0.);
      for (UInt_t i = 0; i < histo.etaRange.size(); i++) {
        histo.etaRange[i] = begin + i * binsize;
        histo.zRange[i] = radius * 100. / TMath::Tan(2 * TMath::ATan(TMath::Exp(-(begin + i * binsize))));
      }
    }

    // Initializing the range in Eta
    else if (mode == TRACK_PHI) {
      Double_t begin = -TMath::Pi();
      Double_t end = +TMath::Pi();
      UInt_t nsteps = 8;  //
      Double_t binsize = (end - begin) / static_cast<Float_t>(nsteps);

      histo.etaRange.resize(nsteps + 1, 0.);
      for (UInt_t i = 0; i < histo.etaRange.size(); i++) {
        histo.etaRange[i] = begin + i * binsize;
      }
    }

    // Dump at screen the range in Eta (or Phi)
    if (ifile == 0) {
      std::cout << "Range used for ";
      if (mode == TRACK_ETA)
        std::cout << "eta";
      else
        std::cout << "phi";
      std::cout << " : " << std::endl;
      ;
      for (UInt_t i = 0; i < (histo.etaRange.size() - 1); i++)
        std::cout << "[" << histo.etaRange[i] << "]-[" << histo.etaRange[i + 1] << "] ";
      std::cout << std::endl;
    }

    // Initializing histos
    std::cout << "Initialzing histograms ..." << std::endl;
    initializeHistograms(mode, histo);

    // Filling with events
    std::cout << "Reading the TFile ..." << std::endl;
    readTree(mode, variable, histo, track, radius);

    //Filling histograms
    std::cout << "Filling histograms ..." << std::endl;
    fillIntermediateHisto(mode, verbose, histo, ccontrol, outputType, radius);
    fillFinalHisto(mode, verbose, histo);
  }

  // Achieving the final plot
  std::cout << "Final plot ..." << std::endl;
  layoutFinalPlot(mode, histos, outputType, variable, c, givenMin, givenMax);

  // Displaying final message
  time_t end = time(0);
  std::cout << "Done in " << static_cast<int>(difftime(end, start)) / 60 << " min and "
            << static_cast<int>(difftime(end, start)) % 60 << " sec." << std::endl;

  delete track;
  delete histo1;
}

// -----------------------------------------------------------------------------
//
//    Auxiliary function : fillIntermediateHisto
//
// -----------------------------------------------------------------------------
void fillIntermediateHisto(
    ModeType mode, Bool_t verbose, HistoType& histo, TCanvas* ccontrol, TString outputType, Double_t radius) {
  // Loop over eta or phi range
  for (UInt_t j = 0; j < (histo.etaRange.size() - 1); j++) {
    // Display eta (or phi) range
    if (verbose) {
      std::cout << histo.etaRange[j] << " < ";
      if (mode == TRACK_ETA)
        std::cout << "eta";
      else
        std::cout << "phi";
      std::cout << " < " << histo.etaRange[j + 1] << std::endl;
    }

    // Loop over energy range
    for (UInt_t i = 0; i < (histo.eRange.size() - 1); i++) {
      // Verbose mode : saving histo positive and negative
      TString controlName;
      if (verbose) {
        // filename for control plots
        controlName = "controlEOP/control_eop_";
        controlName += histo.label;
        controlName += "_bin";
        controlName += j;
        controlName += "_energy";
        controlName += i;
        controlName += ".";
        controlName += outputType;
        ccontrol->cd();

        Double_t posMax = histo.positive[i][j]->GetMaximum();
        Double_t negMax = histo.negative[i][j]->GetMaximum();
        if (posMax > negMax)
          histo.negative[i][j]->SetMaximum(1.1 * posMax);

        histo.negative[i][j]->DrawClone();
        histo.positive[i][j]->DrawClone("same");
      }

      // Fitting by a gaussian and extracting fit parameters
      std::vector<Double_t> curvNeg = extractgausparams(histo.negative[i][j], "", "");
      std::vector<Double_t> curvPos = extractgausparams(histo.positive[i][j], "", "same");

      // Verbose mode : saving gaussian fit plots
      if (verbose) {
        ccontrol->Print(controlName);
      }

      // Initial misalignment value
      Double_t misalignment = 0.;
      Double_t misaliUncert = 1000.;

      // Compute misalignment only if there are selected tracks
      if (histo.selectedTracks[i][j] != 0) {
        // Setting gaussian range
        histo.Enegative[i][j]->GetYaxis()->SetRangeUser(curvNeg[2], curvNeg[3]);
        histo.Epositive[i][j]->GetYaxis()->SetRangeUser(curvPos[2], curvPos[3]);

        // Getting mean values from histogram
        Double_t meanEnergyNeg = histo.Enegative[i][j]->GetMean();
        Double_t meanEnergyPos = histo.Epositive[i][j]->GetMean();
        Double_t meanEnergy = (meanEnergyNeg + meanEnergyPos) /
                              2.;  // use mean of positive and negative tracks to reduce energy dependence

        // Verbose mode : displaying difference between positive and negative means
        if (verbose) {
          std::cout << "difference in energy between positive and negative tracks: " << meanEnergyNeg - meanEnergyPos
                    << std::endl;
        }

        // Compute misalignment
        if (mode == TRACK_ETA) {
          misalignment = 1000000. * 0.5 *
                         (-TMath::ASin((0.57 * radius / meanEnergy) * curvNeg[0]) +
                          TMath::ASin((0.57 * radius / meanEnergy) * curvPos[0]));
          misaliUncert =
              1000000. * 0.5 *
              (TMath::Sqrt((0.57 * 0.57 * radius * radius * curvNeg[1] * curvNeg[1]) /
                               (meanEnergy * meanEnergy - 0.57 * 0.57 * radius * radius * curvNeg[0] * curvNeg[0]) +
                           (0.57 * 0.57 * radius * radius * curvPos[1] * curvPos[1]) /
                               (meanEnergy * meanEnergy - 0.57 * 0.57 * radius * radius * curvPos[0] * curvPos[0])));
        } else if (mode == TRACK_PHI) {
          misalignment = 1000. * (curvPos[0] - curvNeg[0]) / (curvPos[0] + curvNeg[0]);
          misaliUncert = 1000. * 2 / ((curvPos[0] + curvNeg[0]) * (curvPos[0] + curvNeg[0])) *
                         TMath::Sqrt((curvPos[0] * curvPos[0] * curvPos[1] * curvPos[1]) +
                                     (curvNeg[0] * curvNeg[0] * curvNeg[1] * curvNeg[1]));
        }
      }

      // Verbose mode : displaying computed misalignment
      if (verbose)
        std::cout << "misalignment: " << misalignment << "+-" << misaliUncert << std::endl << std::endl;

      // Fill intermediate histogram : histo.fit
      histo.fit[j]->SetBinContent(i + 1, misalignment);
      histo.fit[j]->SetBinError(i + 1, misaliUncert);

      // Fill intermediate histogram : histo.combinedxAxisBin
      Double_t xBinCentre = histo.xAxisBin[i][j]->GetMean();
      Double_t xBinCenUnc = histo.xAxisBin[i][j]->GetMeanError();
      histo.combinedxAxisBin[j]->SetBinContent(i + 1, xBinCentre);
      histo.combinedxAxisBin[j]->SetBinError(i + 1, xBinCenUnc);
    }
  }
}

// -----------------------------------------------------------------------------
//
//    Auxiliary function : fillFinalHisto
//
// -----------------------------------------------------------------------------
void fillFinalHisto(ModeType mode, Bool_t verbose, HistoType& histo) {
  TString fitOption;
  if (verbose)
    fitOption = "";
  else
    fitOption = "Q";
  for (UInt_t i = 0; i < (histo.etaRange.size() - 1); i++) {
    Double_t overallmisalignment;
    Double_t overallmisaliUncert;
    Double_t overallxBin;
    Double_t overallxBinUncert;

    // calculate mean of different energy bins
    TF1* fit = new TF1("fit", "pol0", 0, histo.eRange.size() - 1);
    TF1* fit2 = new TF1("fit2", "pol0", 0, histo.eRange.size() - 1);

    if (histo.fit[i]->GetEntries() < 10) {
      return;
    }

    histo.fit[i]->Fit("fit", fitOption + "0");
    histo.combinedxAxisBin[i]->Fit("fit2", "Q0");
    overallmisalignment = fit->GetParameter(0);
    overallmisaliUncert = fit->GetParError(0);
    overallxBin = fit2->GetParameter(0);
    overallxBinUncert = fit2->GetParError(0);
    fit->Delete();
    fit2->Delete();

    // Fill final histograms
    histo.overallhisto->SetBinContent(i + 1, overallmisalignment);
    histo.overallhisto->SetBinError(i + 1, overallmisaliUncert);
    histo.overallGraph->SetPoint(i, overallxBin, overallmisalignment);
    histo.overallGraph->SetPointError(i, overallxBinUncert, overallmisaliUncert);
  }

  // Fit to final histogram
  TString func = "func";
  func += histo.label;
  if (mode == TRACK_ETA)
    histo.f2 = new TF1(func, "[0]+[1]*x/100.", -500, 500);  //Divide by 100. cm->m
  if (mode == TRACK_PHI)
    histo.f2 = new TF1(func, "[0]+[1]*TMath::Cos(x+[2])", -500, 500);

  // Fitting final histogram
  histo.overallGraph->Fit(func, fitOption + "mR0+");

  // Verbose mode : displaying covariance from fit
  if (verbose) {
    std::cout << "Covariance Matrix:" << std::endl;
    TVirtualFitter* fitter = TVirtualFitter::GetFitter();
    TMatrixD matrix(2, 2, fitter->GetCovarianceMatrix());
    Double_t oneOne = fitter->GetCovarianceMatrixElement(0, 0);
    Double_t oneTwo = fitter->GetCovarianceMatrixElement(0, 1);
    Double_t twoOne = fitter->GetCovarianceMatrixElement(1, 0);
    Double_t twoTwo = fitter->GetCovarianceMatrixElement(1, 1);

    std::cout << "( " << oneOne << ", " << twoOne << ")" << std::endl;
    std::cout << "( " << oneTwo << ", " << twoTwo << ")" << std::endl;
  }

  // Displaying at screen  fit parameters
  if (mode == TRACK_ETA) {
    std::cout << "const: " << histo.f2->GetParameter(0) << "+-" << histo.f2->GetParError(0)
              << ", slope: " << histo.f2->GetParameter(1) << "+-" << histo.f2->GetParError(1) << std::endl;
  } else if (mode == TRACK_PHI) {
    std::cout << "const: " << histo.f2->GetParameter(0) << "+-" << histo.f2->GetParError(0)
              << ", amplitude: " << histo.f2->GetParameter(1) << "+-" << histo.f2->GetParError(1)
              << ", shift: " << histo.f2->GetParameter(2) << "+-" << histo.f2->GetParError(2) << std::endl;
  }
  std::cout << "fit probability: " << histo.f2->GetProb() << std::endl;
  std::cout << "fit chi2       : " << histo.f2->GetChisquare() << std::endl;
  std::cout << "fit chi2/Ndof  : " << histo.f2->GetChisquare() / static_cast<Double_t>(histo.f2->GetNDF()) << std::endl;

  // Adding the fit function for the legend
  Char_t misalignmentfitchar[20];
  sprintf(misalignmentfitchar, "%1.f", histo.f2->GetParameter(0));
  histo.misalignmentfit += "(";
  histo.misalignmentfit += misalignmentfitchar;
  histo.misalignmentfit += "#pm";
  sprintf(misalignmentfitchar, "%1.f", histo.f2->GetParError(0));
  histo.misalignmentfit += misalignmentfitchar;
  if (mode == TRACK_ETA) {
    histo.misalignmentfit += ")#murad #upoint r[m] + (";
    sprintf(misalignmentfitchar, "%1.f", histo.f2->GetParameter(1));
    histo.misalignmentfit += misalignmentfitchar;
    histo.misalignmentfit += "#pm";
    sprintf(misalignmentfitchar, "%1.f", histo.f2->GetParError(1));
    histo.misalignmentfit += misalignmentfitchar;
    histo.misalignmentfit += ")#murad #upoint z[m]";
  } else if (mode == TRACK_PHI) {
    histo.misalignmentfit += ") + (";
    sprintf(misalignmentfitchar, "%1.f", histo.f2->GetParameter(1));
    histo.misalignmentfit += misalignmentfitchar;
    histo.misalignmentfit += "#pm";
    sprintf(misalignmentfitchar, "%1.f", histo.f2->GetParError(1));
    histo.misalignmentfit += misalignmentfitchar;
    histo.misalignmentfit += ") #upoint cos(#phi";
    if (histo.f2->GetParameter(2) > 0.)
      histo.misalignmentfit += "+";
    sprintf(misalignmentfitchar, "%1.1f", histo.f2->GetParameter(2));
    histo.misalignmentfit += misalignmentfitchar;
    histo.misalignmentfit += "#pm";
    sprintf(misalignmentfitchar, "%1.1f", histo.f2->GetParError(2));
    histo.misalignmentfit += misalignmentfitchar;
    histo.misalignmentfit += ")";
  }
}

// -----------------------------------------------------------------------------
//
//    Auxiliary function : LayoutFinalPlot
//
// -----------------------------------------------------------------------------
void layoutFinalPlot(ModeType mode,
                     std::vector<HistoType>& histos,
                     TString outputType,
                     TString variable,
                     TCanvas* c,
                     Double_t givenMin,
                     Double_t givenMax) {
  // Create a legend
  TLegend* leg = new TLegend(0.13, 0.78, 0.98, 0.98);
  leg->SetFillColor(10);
  leg->SetTextFont(42);
  leg->SetTextSize(0.038);
  if (variable == "phi1")
    leg->SetHeader("low #eta tracks (#eta < -0.9)");
  if (variable == "phi2")
    leg->SetHeader("central tracks (|#eta| < 0.9)");
  if (variable == "phi3")
    leg->SetHeader("high #eta tracks (#eta > 0.9)");

  // Setting display options to final histos
  for (UInt_t i = 0; i < histos.size(); i++) {
    // Setting axis title to final histos
    if (mode == TRACK_ETA) {
      histos[i].overallhisto->GetXaxis()->SetTitle("z [cm]");
      histos[i].overallhisto->GetYaxis()->SetTitle("misalignment #Delta#phi[#murad]");
    }
    if (mode == TRACK_PHI) {
      histos[i].overallhisto->GetXaxis()->SetTitle("#phi");
      histos[i].overallhisto->GetYaxis()->SetTitle("misalignment #Delta#phi[a.u.]");
    }

    // Fit function
    if (histos[i].f2) {
      histos[i].f2->SetLineColor(i + 1);
      histos[i].f2->SetLineStyle(i + 1);
      histos[i].f2->SetLineWidth(2);
    }

    // Other settings
    histos[i].overallhisto->GetYaxis()->SetTitleOffset(1.05);
    histos[i].overallhisto->GetYaxis()->SetTitleSize(0.065);
    histos[i].overallhisto->GetYaxis()->SetLabelSize(0.065);
    histos[i].overallhisto->GetXaxis()->SetTitleOffset(0.8);
    histos[i].overallhisto->GetXaxis()->SetTitleSize(0.065);
    histos[i].overallhisto->GetXaxis()->SetLabelSize(0.065);
    histos[i].overallhisto->SetLineWidth(2);
    histos[i].overallhisto->SetLineColor(i + 1);
    histos[i].overallhisto->SetMarkerColor(i + 1);
    histos[i].overallGraph->SetLineWidth(2);
    histos[i].overallGraph->SetLineColor(i + 1);
    histos[i].overallGraph->SetMarkerColor(i + 1);
    histos[i].overallGraph->SetMarkerStyle(i + 20);
    histos[i].overallGraph->SetMarkerSize(2);
  }

  // set pad margins
  c->cd();
  gPad->SetTopMargin(0.02);
  gPad->SetBottomMargin(0.12);
  gPad->SetLeftMargin(0.13);
  gPad->SetRightMargin(0.02);

  // Determining common reasonable y-axis range
  Double_t overallmax = 0.;
  Double_t overallmin = 0.;
  for (UInt_t i = 0; i < histos.size(); i++) {
    // Getting maximum from overallmax
    overallmax = TMath::Max(overallmax,
                            histos[i].overallhisto->GetMaximum() +
                                histos[i].overallhisto->GetBinError(histos[i].overallhisto->GetMaximumBin()) +
                                0.55 * (histos[i].overallhisto->GetMaximum() +
                                        histos[i].overallhisto->GetBinError(histos[i].overallhisto->GetMaximumBin()) -
                                        histos[i].overallhisto->GetMinimum() +
                                        histos[i].overallhisto->GetBinError(histos[i].overallhisto->GetMinimumBin())));

    // Getting minimum from overallmin
    overallmin = TMath::Min(overallmin,
                            histos[i].overallhisto->GetMinimum() -
                                fabs(histos[i].overallhisto->GetBinError(histos[i].overallhisto->GetMinimumBin())) -
                                0.1 * (histos[i].overallhisto->GetMaximum() +
                                       histos[i].overallhisto->GetBinError(histos[i].overallhisto->GetMaximumBin()) -
                                       histos[i].overallhisto->GetMinimum() +
                                       histos[i].overallhisto->GetBinError(histos[i].overallhisto->GetMinimumBin())));
  }

  // Applying common y-axis range to each final histo
  for (UInt_t i = 0; i < histos.size(); i++) {
    histos[i].overallhisto->SetMaximum(overallmax);
    histos[i].overallhisto->SetMinimum(overallmin);
    if (givenMax != 0)
      histos[i].overallhisto->SetMaximum(givenMax);
    if (givenMin != 0)
      histos[i].overallhisto->SetMinimum(givenMin);
    histos[i].overallhisto->DrawClone("axis");

    // set histogram errors to a small value as only errors of the graph should be shown
    for (Int_t j = 0; j < histos[i].overallhisto->GetNbinsX(); j++)
      histos[i].overallhisto->SetBinError(j + 1, 0.00001);

    // draw final histogram
    histos[i].overallhisto->DrawClone("pe1 same");
    if (histos[i].f2) {
      histos[i].f2->DrawClone("same");
    }
    histos[i].overallGraph->DrawClone("|| same");
    histos[i].overallGraph->DrawClone("pz same");
    histos[i].overallGraph->SetLineStyle(i + 1);
    leg->AddEntry(histos[i].overallGraph, histos[i].label + " (" + histos[i].misalignmentfit + ")", "Lp");
  }

  leg->Draw();

  // Saving plots
  if (variable == "eta")
    c->Print("twist_validation." + outputType);
  else if (variable == "phi")
    c->Print("sagitta_validation_all." + outputType);
  else if (variable == "phi1")
    c->Print("sagitta_validation_lowEta." + outputType);
  else if (variable == "phi2")
    c->Print("sagitta_validation_centralEta." + outputType);
  else if (variable == "phi3")
    c->Print("sagitta_validation_highEta." + outputType);

  delete leg;
}

// -----------------------------------------------------------------------------
//
//    Auxiliary function : initialize Histograms
//
// -----------------------------------------------------------------------------
void initializeHistograms(ModeType mode, HistoType& histo) {
  // -----------------------------------------
  //      Initializing Basic Histograms
  // -----------------------------------------

  // Allocated memory
  histo.selectedTracks = new UInt_t*[histo.eRange.size() - 1];
  histo.xAxisBin = new TH1F**[histo.eRange.size() - 1];
  histo.negative = new TH1F**[histo.eRange.size() - 1];
  histo.positive = new TH1F**[histo.eRange.size() - 1];
  histo.Enegative = new TH2F**[histo.eRange.size() - 1];
  histo.Epositive = new TH2F**[histo.eRange.size() - 1];
  for (unsigned int i = 0; i < (histo.eRange.size() - 1); i++) {
    histo.selectedTracks[i] = new UInt_t[histo.etaRange.size() - 1];
    histo.xAxisBin[i] = new TH1F*[histo.etaRange.size() - 1];
    histo.negative[i] = new TH1F*[histo.etaRange.size() - 1];
    histo.positive[i] = new TH1F*[histo.etaRange.size() - 1];
    histo.Enegative[i] = new TH2F*[histo.etaRange.size() - 1];
    histo.Epositive[i] = new TH2F*[histo.etaRange.size() - 1];
  }

  // Loop over energy range
  unsigned int index = 0;
  for (unsigned int i = 0; i < (histo.eRange.size() - 1); i++) {
    // Labels for histo title
    Char_t tmpstep[10];
    sprintf(tmpstep, "%1.1fGeV", histo.eRange[i]);
    TString lowEnergyBorder = tmpstep;
    sprintf(tmpstep, "%1.1fGeV", histo.eRange[i + 1]);
    TString highEnergyBorder = tmpstep;
    TString eTitle = lowEnergyBorder + " #leq ";
    if (mode == TRACK_ETA)
      eTitle += "E";
    else
      eTitle += "E_{T}";
    eTitle += " < " + highEnergyBorder;

    for (unsigned int j = 0; j < (histo.etaRange.size() - 1); j++) {
      // Labels for histo name
      Char_t tmpstep[10];
      sprintf(tmpstep, "%1.1fGeV", histo.eRange[i]);
      TString lowEnergyBorder = tmpstep;
      sprintf(tmpstep, "%1.1fGeV", histo.eRange[i + 1]);
      TString highEnergyBorder = tmpstep;
      TString eTitle = lowEnergyBorder + " #leq ";
      if (mode == TRACK_ETA)
        eTitle += "E";
      else
        eTitle += "E_{T}";
      eTitle += " < " + highEnergyBorder;

      index++;
      Char_t histochar[20];
      sprintf(histochar, "%i", index);
      TString histotrg = histochar;
      histotrg += " (" + histo.label + ")";

      // Creating histograms
      histo.negative[i][j] = new TH1F("negative" + histotrg, "negative (" + eTitle + ")", 50, 0, 2);
      histo.positive[i][j] = new TH1F("positive" + histotrg, "positive (" + eTitle + ")", 50, 0, 2);
      histo.Enegative[i][j] = new TH2F("Enegative" + histotrg, "Enegative (" + eTitle + ")", 5000, 0, 500, 50, 0, 2);
      histo.Epositive[i][j] = new TH2F("Epositive" + histotrg, "Epositive (" + eTitle + ")", 5000, 0, 500, 50, 0, 2);
      histo.xAxisBin[i][j] = new TH1F("xAxisBin" + histotrg, "xAxisBin (" + eTitle + ")", 1000, -500, 500);

      // Sum of squares of weight for each histogram
      histo.Enegative[i][j]->Sumw2();
      histo.Epositive[i][j]->Sumw2();
      histo.xAxisBin[i][j]->Sumw2();
      histo.negative[i][j]->Sumw2();
      histo.positive[i][j]->Sumw2();

      //SetDirectory(0)
      histo.Enegative[i][j]->SetDirectory(0);
      histo.Epositive[i][j]->SetDirectory(0);
      histo.xAxisBin[i][j]->SetDirectory(0);
      histo.negative[i][j]->SetDirectory(0);
      histo.positive[i][j]->SetDirectory(0);

      // Set color
      histo.negative[i][j]->SetLineColor(kGreen);
      histo.positive[i][j]->SetLineColor(kRed);
    }
  }

  // -----------------------------------------
  //   Initializing Intermediate Histograms
  // -----------------------------------------
  histo.fit = new TH1F*[histo.etaRange.size() - 1];
  histo.combinedxAxisBin = new TH1F*[histo.etaRange.size() - 1];

  for (UInt_t j = 0; j < (histo.etaRange.size() - 1); j++) {
    Char_t histochar[20];
    sprintf(histochar, "%i", j + 1);
    TString histotrg = histochar;
    histotrg += "(" + histo.label + ")";
    histo.fit[j] = new TH1F("fithisto" + histotrg,       //name
                            "fithisto" + histotrg,       //title
                            (histo.eRange.size() - 1),   //nbins
                            0.,                          //min
                            (histo.eRange.size() - 1));  //max
    histo.fit[j]->SetDirectory(0);
    histo.combinedxAxisBin[j] = new TH1F("combinedxAxisBin" + histotrg,  //name
                                         "combinedxAxisBin" + histotrg,  //title
                                         (histo.eRange.size() - 1),      //nbins
                                         0.,                             //min
                                         (histo.eRange.size() - 1));     //max
    histo.combinedxAxisBin[j]->SetDirectory(0);
  }

  // -----------------------------------------
  //       Initializing Final Histograms
  // -----------------------------------------
  if (mode == TRACK_ETA) {
    histo.overallhisto = new TH1F(
        "overallhisto (" + histo.label + ")", "overallhisto", (histo.zRange.size() - 1), &histo.zRange.front());
  } else {
    histo.overallhisto = new TH1F("overallhisto (" + histo.label + ")",
                                  "overallhisto",
                                  (histo.etaRange.size() - 1),
                                  histo.etaRange[0],
                                  histo.etaRange[histo.etaRange.size() - 1]);
  }
  histo.overallhisto->SetDirectory(0);

  histo.overallGraph = new TGraphErrors(histo.overallhisto);
}

// -----------------------------------------------------------------------------
//
//    Auxiliary function : readTree
//
// -----------------------------------------------------------------------------
void readTree(ModeType mode, TString variable, HistoType& histo, EopElecVariables* track, Double_t radius) {
  // Displaying sample name
  Long64_t nevent = (Long64_t)histo.tree->GetEntries();
  std::cout << "Reading sample labeled by '" << histo.label << "' containing " << nevent << " events ..." << std::endl;

  // Reseting counters
  TH1D* NtracksMacro = new TH1D("NtracksMacro", "NtracksMacro", 1, 0, 1);
  TH1D* usedtracks = new TH1D("usedtracks", "usedtracks", 1, 0, 1);
  TH1D* cut_Mz = new TH1D("cut_Mz", "cut_Mz", 1, 0, 1);
  TH1D* cut_eta = new TH1D("cut_eta", "cut_eta", 1, 0, 1);
  TH1D* cut_ChargedIso = new TH1D("cut_ChargedIso", "cut_ChargedIso", 1, 0, 1);
  TH1D* cut_Ecal = new TH1D("cut_Ecal", "cut_Ecal", 1, 0, 1);
  TH1D* cut_HoverE = new TH1D("cut_HoverE", "cut_HoverE", 1, 0, 1);
  TH1D* cut_NeutralIso = new TH1D("cut_NeutralIso", "cut_NeutralIso", 1, 0, 1);
  TH1D* cut_nHits = new TH1D("cut_nHits", "cut_nHits", 1, 0, 1);
  TH1D* cut_nLostHits = new TH1D("cut_nLostHits", "cut_nLostHits", 1, 0, 1);
  TH1D* cut_outerRadius = new TH1D("cut_outerRadius", "cut_outerRadius", 1, 0, 1);
  TH1D* cut_normalizedChi2 = new TH1D("cut_normalizedChi2", "cut_normalizedChi2", 1, 0, 1);
  TH1D* cut_fbrem = new TH1D("cut_fbrem", "cut_fbrem", 1, 0, 1);
  TH1D* cut_nCluster = new TH1D("cut_nCluster", "cut_nCluster", 1, 0, 1);
  TH1D* cut_EcalRange = new TH1D("cut_EcalRange", "cut_EcalRange", 1, 0, 1);
  TH1D* cut_trigger = new TH1D("cut_trigger", "cut_trigger", 1, 0, 1);

  TH1D* P = new TH1D("P", "P", 180, 0, 180);
  TH1D* eta = new TH1D("eta", "eta", 100, -5., 5.);
  TH1D* Pt = new TH1D("Pt", "Pt", 1000, 0, 1000);
  TH1D* Ecal = new TH1D("Ecal", "Ecal", 100, 0, 180);
  TH1D* HcalEnergyIn01 = new TH1D("HcalEnergyIn01", "HcalEnergyIn01", 100, 0, 40);
  TH1D* HcalEnergyIn02 = new TH1D("HcalEnergyIn02", "HcalEnergyIn02", 100, 0, 40);
  TH1D* HcalEnergyIn03 = new TH1D("HcalEnergyIn03", "HcalEnergyIn03", 100, 0, 40);
  TH1D* HcalEnergyIn04 = new TH1D("HcalEnergyIn04", "HcalEnergyIn04", 100, 0, 40);
  TH1D* HcalEnergyIn05 = new TH1D("HcalEnergyIn05", "HcalEnergyIn05", 100, 0, 40);
  TH1D* SumPt = new TH1D("SumPt", "SumPt", 100, 0, 2);
  TH1D* HoverE = new TH1D("HoverE", "HoverE", 50, 0, 0.6);
  TH1D* distTo1stSC = new TH1D("distTo1stSC", "distTo1stSC", 50, 0, 0.1);
  TH1D* distTo2ndSC = new TH1D("distTo2ndSC", "distTo2ndSC", 50, 0, 1.5);
  TH1D* fbrem = new TH1D("fbrem", "fbrem", 50, -0.2, 1.);
  TH1D* nBasicClus = new TH1D("nBasicClus", "nBasicClus", 12, 0, 12);
  TH1D* ScPhiWidth = new TH1D("ScPhiWidth", "ScPhiWidth", 50, 0.005, 0.1);

  TH2D* PhiWidthVSnBC = new TH2D("PhiWidthVSnBC", "PhiWidthVSnBC", 12, 0, 12, 50, 0.005, 0.1);
  TH2D* PhiWidthVSfbrem = new TH2D("PhiWidthVSfbrem", "PhiWidthVSfbrem", 100, -0.2, 1., 50, 0.005, 0.1);
  TH2D* nBasicClusVSfbrem = new TH2D("nBasicClusVSfbrem", "nBasicClusVSfbrem", 100, -0.2, 1., 12, 0, 12);
  TH2D* EopVSfbrem = new TH2D("EopVSfbrem", "EopVSfbrem", 100, -0.2, 1., 100, 0, 3);

  TH1D* EopNegFwd = new TH1D("EopNegFwd", "EopNegFwd", 180, 0, 3);
  TH1D* EopNegBwd = new TH1D("EopNegBwd", "EopNegBwd", 180, 0, 3);

  double Mass;
  Bool_t MzTag = false;
  Bool_t BARREL = true;

  // Loop over tracks
  for (Long64_t ientry = 0; ientry < nevent; ientry++) {
    // Load the event information
    histo.tree->GetEntry(ientry);

    //Control plots
    if (track->charge < 0) {
      if (track->eta > 0.5)
        EopNegFwd->Fill(track->SC_energy / track->p);
      if (track->eta < 0.5)
        EopNegBwd->Fill(track->SC_energy / track->p);
    }

    PhiWidthVSfbrem->Fill(track->fbrem, track->SC_phiWidth);

    // ---- Track Selection ----
    NtracksMacro->Fill(0.5);

    //Mz calculation
    MzTag = false;
    Mass = 0.;
    TLorentzVector a(0., 0., 0., 0.);
    a.SetXYZM(track->px, track->py, track->pz, 0.511);
    TLorentzVector b(0., 0., 0., 0.);
    b.SetXYZM(track->px_rejected_track, track->py_rejected_track, track->pz_rejected_track, 0.511);

    Mass = pow((a.E() + b.E()), 2) - pow((a.Px() + b.Px()), 2) - pow((a.Py() + b.Py()), 2) - pow((a.Pz() + b.Pz()), 2);

    Mass = sqrt(Mass);

    if (Mass < 110. && Mass > 70.)
      MzTag = true;

    // Z mass window
    if (!MzTag)
      continue;
    cut_Mz->Fill(0.5);

    // -----------        ENDCAP SELECTION    ----------------

    if (!BARREL) {
      eta->Fill(track->eta);
      if (!track->SC_isEndcap)
        continue;
      cut_eta->Fill(0.5);

      // Isolation against charged particles
      SumPt->Fill(track->SumPtIn05 / track->pt);
      if (!track->NoTrackIn0015 || (track->SumPtIn05 / track->pt) > 0.05)
        continue;
      cut_ChargedIso->Fill(0.5);

      // Ecal energy deposit
      Ecal->Fill(track->SC_energy);
      if (track->SC_energy < 30.)
        continue;
      cut_Ecal->Fill(0.5);

      // Hcal over Ecal energy deposit
      HoverE->Fill(track->HcalEnergyIn03 / track->SC_energy);
      if (track->HcalEnergyIn03 / track->SC_energy > 0.06)
        continue;  // before: > 0.08
      cut_HoverE->Fill(0.5);

      // Track-SuperCluster matching radius
      distTo1stSC->Fill(track->dRto1stSC);
      if (track->dRto1stSC > 0.05)
        continue;
      distTo2ndSC->Fill(track->dRto2ndSC);
      if (track->dRto2ndSC < 0.25)
        continue;  //min=0.09(1st SC)
      cut_NeutralIso->Fill(0.5);

      // a high number of valid hits
      if (track->nHits < 13)
        continue;
      cut_nHits->Fill(0.5);

      // no lost hits
      if (track->nLostHits != 0)
        continue;
      cut_nLostHits->Fill(0.5);

      // outerRadius
      //if (track->outerRadius<=99.) continue;
      cut_outerRadius->Fill(0.5);

      // good chi2 value
      if (track->normalizedChi2 >= 5.)
        continue;
      cut_normalizedChi2->Fill(0.5);

      // less than 10% bremmstrahlung radiated energy
      fbrem->Fill(track->fbrem);
      if (track->fbrem > 0.10 || track->fbrem < -0.10)
        continue;
      cut_fbrem->Fill(0.5);

      // only tracks with associated SuperCluster composed of ONE BasicCluster
      nBasicClus->Fill(track->SC_nBasicClus);
      if (track->SC_nBasicClus != 1)
        continue;
      cut_nCluster->Fill(0.5);
    }

    // -----------        BARREL SELECTION    ----------------

    if (BARREL) {
      eta->Fill(track->eta);
      if (!track->SC_isBarrel)
        continue;
      cut_eta->Fill(0.5);

      // Isolation against charged particles
      SumPt->Fill(track->SumPtIn05 / track->pt);
      if (!track->NoTrackIn0015 || (track->SumPtIn05 / track->pt) > 0.05)
        continue;
      cut_ChargedIso->Fill(0.5);

      // Ecal energy deposit
      Ecal->Fill(track->SC_energy);
      if (track->SC_energy < 25.)
        continue;
      cut_Ecal->Fill(0.5);

      // Hcal over Ecal energy deposit
      HoverE->Fill(track->HcalEnergyIn03 / track->SC_energy);
      if (track->HcalEnergyIn03 / track->SC_energy > 0.06)
        continue;  // before: > 0.08
      cut_HoverE->Fill(0.5);

      // Track-SuperCluster matching radius
      distTo1stSC->Fill(track->dRto1stSC);
      if (track->dRto1stSC > 0.04)
        continue;
      distTo2ndSC->Fill(track->dRto2ndSC);
      if (track->dRto2ndSC < 0.35)
        continue;  //min=0.09(1st SC)
      cut_NeutralIso->Fill(0.5);

      // a high number of valid hits
      if (track->nHits < 13)
        continue;
      cut_nHits->Fill(0.5);

      // no lost hits
      if (track->nLostHits != 0)
        continue;
      cut_nLostHits->Fill(0.5);

      // outerRadius
      if (track->outerRadius <= 99.)
        continue;
      cut_outerRadius->Fill(0.5);

      // good chi2 value
      if (track->normalizedChi2 >= 5.)
        continue;
      cut_normalizedChi2->Fill(0.5);

      // less than 10% bremmstrahlung radiated energy
      fbrem->Fill(track->fbrem);
      if (track->fbrem > 0.1 || track->fbrem < -0.1)
        continue;
      cut_fbrem->Fill(0.5);

      // only tracks with associated SuperCluster composed of ONE BasicCluster
      nBasicClus->Fill(track->SC_nBasicClus);
      if (track->SC_nBasicClus != 1)
        continue;
      cut_nCluster->Fill(0.5);
    }

    PhiWidthVSnBC->Fill(track->SC_nBasicClus, track->SC_phiWidth);
    nBasicClusVSfbrem->Fill(track->fbrem, track->SC_nBasicClus);
    EopVSfbrem->Fill(track->fbrem, track->SC_energy / track->p);

    // only track in studied ECAL range
    if (track->SC_energy <= histo.eRange[0] || track->SC_energy >= histo.eRange[histo.eRange.size() - 1])
      continue;
    cut_EcalRange->Fill(0.5);

    // Loop over eta or phi bins
    for (UInt_t j = 0; j < (histo.etaRange.size() - 1); j++) {
      // Loop over energy steps
      for (UInt_t i = 0; i < (histo.eRange.size() - 1); i++) {
        Double_t lowerE = histo.eRange[i];
        Double_t upperE = histo.eRange[i + 1];
        Double_t lowerEta = histo.etaRange[j];
        Double_t upperEta = histo.etaRange[j + 1];

        // Select only tracks in E range
        if (!(track->SC_energy > lowerE && track->SC_energy < upperE))
          continue;

        // Select only tracks in eta range
        if (mode == TRACK_ETA) {
          if (!(track->eta >= lowerEta && track->eta < upperEta))
            continue;

          usedtracks->Fill(0.5);
          histo.selectedTracks[i][j]++;
          histo.xAxisBin[i][j]->Fill(radius * 100. / TMath::Tan(2 * TMath::ATan(TMath::Exp(-(track->eta)))));
        }

        // Select only tracks in phi range
        else if (mode == TRACK_PHI) {
          if (!((variable == "phi1" && track->eta < -0.9) || (variable == "phi2" && TMath::Abs(track->eta) < 0.9) ||
                (variable == "phi3" && track->eta > 0.9) || variable == "phi"))
            continue;

          if (!(track->phi >= lowerEta && track->phi < upperEta))
            continue;

          usedtracks->Fill(0.5);
          histo.selectedTracks[i][j]++;
          histo.xAxisBin[i][j]->Fill(track->phi);
        }

        // e over p
        if (track->charge < 0) {
          histo.negative[i][j]->Fill((track->SC_energy) / track->p);
          histo.Enegative[i][j]->Fill(track->SC_energy * TMath::Sin(track->theta), (track->SC_energy) / track->p);
        } else {
          histo.positive[i][j]->Fill((track->SC_energy) / track->p);
          histo.Epositive[i][j]->Fill(track->SC_energy * TMath::Sin(track->theta), (track->SC_energy) / track->p);
        }
      }
    }
  }

  //############ Saving cut table and control histogramms #############

  NtracksMacro->Write();
  cut_Ecal->Write();
  cut_ChargedIso->Write();
  cut_NeutralIso->Write();
  cut_HoverE->Write();
  cut_nHits->Write();
  cut_nLostHits->Write();
  cut_outerRadius->Write();
  cut_normalizedChi2->Write();
  cut_fbrem->Write();
  cut_nCluster->Write();
  cut_Mz->Write();
  cut_EcalRange->Write();
  cut_eta->Write();
  cut_trigger->Write();
  usedtracks->Write();

  Pt->Write();
  P->Write();
  eta->Write();
  Ecal->Write();
  HcalEnergyIn01->Write();
  HcalEnergyIn02->Write();
  HcalEnergyIn03->Write();
  HcalEnergyIn04->Write();
  HcalEnergyIn05->Write();
  SumPt->Write();
  HoverE->Write();
  distTo1stSC->Write();
  distTo2ndSC->Write();
  fbrem->Write();
  nBasicClus->Write();
  ScPhiWidth->Write();
  PhiWidthVSnBC->Write();
  PhiWidthVSfbrem->Write();
  nBasicClusVSfbrem->Write();
  EopVSfbrem->Write();

  EopNegFwd->Write();
  EopNegBwd->Write();

  //############ CUTS EFFICIENCY ##############

  double NTracksMacro = NtracksMacro->GetBinContent(1);
  double usedTracks = usedtracks->GetBinContent(1);
  double Cut_Ecal = cut_Ecal->GetBinContent(1);
  double Cut_ChargedIso = cut_ChargedIso->GetBinContent(1);
  double Cut_NeutralIso = cut_NeutralIso->GetBinContent(1);
  double Cut_HoverE = cut_HoverE->GetBinContent(1);
  double Cut_nHits = cut_nHits->GetBinContent(1);
  double Cut_nLostHits = cut_nLostHits->GetBinContent(1);
  double Cut_outerRadius = cut_outerRadius->GetBinContent(1);
  double Cut_normalizedChi2 = cut_normalizedChi2->GetBinContent(1);
  double Cut_fbrem = cut_fbrem->GetBinContent(1);
  double Cut_nCluster = cut_nCluster->GetBinContent(1);
  double Cut_Mz = cut_Mz->GetBinContent(1);
  double Cut_EcalRange = cut_EcalRange->GetBinContent(1);
  double Cut_eta = cut_eta->GetBinContent(1);

  std::cout.setf(std::ios::fixed);
  UInt_t precision = std::cout.precision();
  std::cout.precision(2);
  std::cout << "##################   CUTS EFFICIENCY  ########################" << std::endl;
  std::cout << std::endl;
  std::cout << "   Cut          | Number of tracks | Subsisting percentage % " << std::endl;
  std::cout << "--------------------------------------------------------------" << std::endl;
  std::cout << " Trigger selection  | ";
  std::cout.width(16);
  std::cout << std::right << NTracksMacro << " | ";
  std::cout.width(16);
  std::cout << NTracksMacro / static_cast<Double_t>(NTracksMacro) * 100. << std::endl;
  std::cout << "--------------------------------------------------------------" << std::endl;
  std::cout << " Mz                 | ";
  std::cout.width(16);
  std::cout << std::right << Cut_Mz << " | ";
  std::cout.width(16);
  std::cout << Cut_Mz / static_cast<Double_t>(NTracksMacro) * 100. << std::endl;
  std::cout << "--------------------------------------------------------------" << std::endl;
  std::cout << " Eta range          | ";
  std::cout.width(16);
  std::cout << std::right << Cut_eta << " | ";
  std::cout.width(16);
  std::cout << Cut_eta / static_cast<Double_t>(NTracksMacro) * 100. << std::endl;
  std::cout << "--------------------------------------------------------------" << std::endl;
  std::cout << " Iso charged        | ";
  std::cout.width(16);
  std::cout << std::right << Cut_ChargedIso << " | ";
  std::cout.width(16);
  std::cout << Cut_ChargedIso / static_cast<Double_t>(NTracksMacro) * 100. << std::endl;
  std::cout << "--------------------------------------------------------------" << std::endl;
  std::cout << " Ecal               | ";
  std::cout.width(16);
  std::cout << std::right << Cut_Ecal << " | ";
  std::cout.width(16);
  std::cout << Cut_Ecal / static_cast<Double_t>(NTracksMacro) * 100. << std::endl;
  std::cout << "--------------------------------------------------------------" << std::endl;
  std::cout << " H over E           | ";
  std::cout.width(16);
  std::cout << std::right << Cut_HoverE << " | ";
  std::cout.width(16);
  std::cout << Cut_HoverE / static_cast<Double_t>(NTracksMacro) * 100. << std::endl;
  std::cout << "--------------------------------------------------------------" << std::endl;
  std::cout << " Iso Neutral        | ";
  std::cout.width(16);
  std::cout << std::right << Cut_NeutralIso << " | ";
  std::cout.width(16);
  std::cout << Cut_NeutralIso / static_cast<Double_t>(NTracksMacro) * 100. << std::endl;
  std::cout << "--------------------------------------------------------------" << std::endl;
  std::cout << " nHits              | ";
  std::cout.width(16);
  std::cout << std::right << Cut_nHits << " | ";
  std::cout.width(16);
  std::cout << Cut_nHits / static_cast<Double_t>(NTracksMacro) * 100. << std::endl;
  std::cout << "--------------------------------------------------------------" << std::endl;
  std::cout << " nLostHits          | ";
  std::cout.width(16);
  std::cout << std::right << Cut_nLostHits << " | ";
  std::cout.width(16);
  std::cout << Cut_nLostHits / static_cast<Double_t>(NTracksMacro) * 100. << std::endl;
  std::cout << "--------------------------------------------------------------" << std::endl;
  std::cout << " outerRadius        | ";
  std::cout.width(16);
  std::cout << std::right << Cut_outerRadius << " | ";
  std::cout.width(16);
  std::cout << Cut_outerRadius / static_cast<Double_t>(NTracksMacro) * 100. << std::endl;
  std::cout << "--------------------------------------------------------------" << std::endl;
  std::cout << " normalizedChi2     | ";
  std::cout.width(16);
  std::cout << std::right << Cut_normalizedChi2 << " | ";
  std::cout.width(16);
  std::cout << Cut_normalizedChi2 / static_cast<Double_t>(NTracksMacro) * 100. << std::endl;
  std::cout << "--------------------------------------------------------------" << std::endl;
  std::cout << " fbrem              | ";
  std::cout.width(16);
  std::cout << std::right << Cut_fbrem << " | ";
  std::cout.width(16);
  std::cout << Cut_fbrem / static_cast<Double_t>(NTracksMacro) * 100. << std::endl;
  std::cout << "--------------------------------------------------------------" << std::endl;
  std::cout << " nCluster           | ";
  std::cout.width(16);
  std::cout << std::right << Cut_nCluster << " | ";
  std::cout.width(16);
  std::cout << Cut_nCluster / static_cast<Double_t>(NTracksMacro) * 100. << std::endl;
  std::cout << "--------------------------------------------------------------" << std::endl;
  std::cout << " EcalEnergy Range   | ";
  std::cout.width(16);
  std::cout << std::right << Cut_EcalRange << " | ";
  std::cout.width(16);
  std::cout << Cut_EcalRange / static_cast<Double_t>(NTracksMacro) * 100. << std::endl;
  std::cout << "--------------------------------------------------------------" << std::endl;
  std::cout << " Used Tracks        | ";
  std::cout.width(16);
  std::cout << std::right << usedTracks << " | ";
  std::cout.width(16);
  std::cout << usedTracks / static_cast<Double_t>(NTracksMacro) * 100. << std::endl;
  std::cout << std::endl;
  std::cout << "##############################################################" << std::endl;
  std::cout.unsetf(std::ios::fixed);
  std::cout.precision(precision);
}

// -----------------------------------------------------------------------------
//
//    Auxiliary function : extractgausparams
//
// -----------------------------------------------------------------------------
std::vector<Double_t> extractgausparams(TH1F* histo, TString option1, TString option2) {
  if (histo->GetEntries() < 10) {
    return {0., 0., 0., 0.};
  }

  // Fitting the histogram with a gaussian function
  TF1* f1 = new TF1("f1", "gaus");
  f1->SetRange(0., 2.);
  histo->Fit("f1", "QR0L");

  // Getting parameters estimated by the fit
  double mean = f1->GetParameter(1);
  double deviation = f1->GetParameter(2);

  // Iinitializing internal parameters of the iteration algorithms
  double lowLim = 0;
  double upLim = 0;
  double degrade = 0.05;
  unsigned int iteration = 0;

  // Iteration procedure
  // Iteration is stopped when :
  //      more than 10 iterations
  //   or fit probability > 0.001
  while (iteration == 0 || (f1->GetProb() < 0.001 && iteration < 10)) {
    // Computing new bounds for the fit range (2.0 sigma)
    lowLim = mean - (2.0 * deviation * (1 - degrade * iteration));
    upLim = mean + (2.0 * deviation * (1 - degrade * iteration));
    f1->SetRange(lowLim, upLim);

    // Fitting the histo with new bounds
    histo->Fit("f1", "QRL0");

    // If the fit succeeds -> extract the new estimated mean value
    double newmean = 0;
    if (f1->GetParameter(1) < 2 && f1->GetParameter(1) > 0)
      newmean = f1->GetParameter(1);
    // Else -> keep the previous mean value
    else
      newmean = mean;

    // Computing new bounds for the fit with the new mean value (1.5 sigma)
    lowLim = newmean - (1.5 * deviation);  //*(1-degrade*iteration));
    upLim = newmean + (1.5 * deviation);   //*(1-degrade*iteration));
    f1->SetRange(lowLim, upLim);
    f1->SetLineWidth(1);

    // Fitting the histo with new bounds
    histo->Fit("f1", "QRL+i" + option1, option2);

    // if the fit succeeds -> extract the new estimated mean value
    if (f1->GetParameter(1) < 2 && f1->GetParameter(1) > 0)
      mean = f1->GetParameter(1);

    // Computing new deviation
    deviation = f1->GetParameter(2);

    // Next iteration
    iteration++;
  }

  // return the mean value + its error
  std::vector<Double_t> params;
  params.push_back(f1->GetParameter(1));
  params.push_back(f1->GetParError(1));
  params.push_back(lowLim);
  params.push_back(upLim);
  delete f1;
  return params;
}

// -----------------------------------------------------------------------------
//
//    Auxiliary function : CheckArguments
//
// -----------------------------------------------------------------------------
Bool_t checkArguments(TString variable,  //input
                      TString path,
                      TString alignmentWithLabel,
                      TString outputType,
                      Double_t radius,
                      Bool_t verbose,
                      Double_t givenMin,
                      Double_t givenMax,
                      ModeType& mode,  // ouput
                      std::vector<TFile*>& files,
                      std::vector<TString>& labels) {
  // Determining mode
  if (variable == "eta")
    mode = TRACK_ETA;
  else if (variable == "phi" || variable == "phi1" || variable == "phi2" || variable == "phi3")
    mode = TRACK_PHI;
  else {
    std::cout << "variable can be eta or phi" << std::endl;
    std::cout << "phi1, phi2 and phi3 are for phi in different eta bins" << std::endl;
    return false;
  }

  // Checking outputType
  TString allowed[] = {"eps", "pdf", "svg", "gif", "xpm", "png", "jpg", "tiff", "C", "cxx", "root", "xml"};
  bool find = false;
  for (unsigned int i = 0; i < 12; i++) {
    if (outputType == allowed[i]) {
      find = true;
      break;
    }
  }
  if (!find) {
    std::cout << "ERROR: output type called '" + outputType + "' is unknown !" << std::endl
              << "The available output types are : " << std::endl;
    for (unsigned int i = 0; i < 12; i++) {
      std::cout << " " << allowed[i];
    }
    std::cout << std::endl;
    return false;
  }

  // Checking radius
  if (radius <= 0) {
    std::cout << "ERROR: The radius cannot be null or negative" << std::endl;
    return false;
  }

  // Checking bounds
  if (givenMin > givenMax) {
    std::cout << "ERROR: Min value is greater than Max value" << std::endl;
    return false;
  }

  // Reading the list of files
  TObjArray* fileLabelPairs = alignmentWithLabel.Tokenize("\\");
  std::cout << "Reading input ROOT files ..." << std::endl;
  for (Int_t i = 0; i < fileLabelPairs->GetEntries(); i++) {
    TObjArray* singleFileLabelPair = TString(fileLabelPairs->At(i)->GetName()).Tokenize("=");

    if (singleFileLabelPair->GetEntries() == 2) {
      TFile* theFile = new TFile(path + (TString(singleFileLabelPair->At(0)->GetName())));
      if (!theFile->IsOpen()) {
        std::cout << "ERROR : the file '" << theFile->GetName() << "' is not found" << std::endl;
        return false;
      }
      files.push_back(theFile);
      labels.push_back(singleFileLabelPair->At(1)->GetName());
    } else {
      std::cout << "ERROR : Please give file name and legend entry in the following form:\n"
                << " filename1=legendentry1\\filename2=legendentry2\\..." << std::endl;
      return false;
    }
  }

  // Check there is at least of file to analyze
  if (files.size() == 0) {
    std::cout << "-> No file to analyze" << std::endl;
    return false;
  }

  // Check labels are unique
  for (unsigned int i = 0; i < labels.size(); i++)
    for (unsigned int j = i + 1; j < labels.size(); j++) {
      if (labels[i] == labels[j]) {
        std::cout << "the label '" << labels[i] << "' is twice" << std::endl;
        return false;
      }
    }

  return true;
}

// -----------------------------------------------------------------------------
//
//    Main (only for stand alone compiled program)
//
// -----------------------------------------------------------------------------
void configureROOTstyle(Bool_t verbose) {
  // Configuring style
  gROOT->cd();
  gROOT->SetStyle("Plain");
  if (verbose) {
    std::cout << "\n all formerly produced control plots in ./controlEOP/ deleted.\n" << std::endl;
    gROOT->ProcessLine(".!if [ -d controlEOP ]; then rm controlEOP/control_eop*; else mkdir controlEOP; fi");
  }
  gStyle->SetPadBorderMode(0);
  gStyle->SetOptStat(0);
  gStyle->SetTitleFont(42, "XYZ");
  gStyle->SetLabelFont(42, "XYZ");
  gStyle->SetEndErrorSize(5);
  gStyle->SetLineStyleString(2, "80 20");
  gStyle->SetLineStyleString(3, "40 18");
  gStyle->SetLineStyleString(4, "20 16");
}

// -----------------------------------------------------------------------------
//
//    Initialize Tree
//
// -----------------------------------------------------------------------------
Bool_t initializeTree(std::vector<TFile*>& files, std::vector<TTree*>& trees, EopElecVariables* track) {
  // resize the tree size
  trees.resize(files.size());

  // Loop over the different ROOT files
  for (unsigned int i = 0; i < files.size(); i++) {
    // Displaying the file name
    std::cout << "Opening the file : " << files[i]->GetName() << std::endl;

    // Extracting the TTree from the ROOT file
    TTree* theTree = dynamic_cast<TTree*>(files[i]->Get("energyOverMomentumTree/EopTree"));

    // Checking if the TTree is found
    if (theTree == 0) {
      std::cout << "ERROR : the tree 'EopTree' is not found ! This file is skipped !" << std::endl;
      return false;
    }

    // Storing the TTree in the container
    trees[i] = theTree;

    // Connecting the branches of the TTree to the event object
    trees[i]->SetMakeClass(1);
    trees[i]->SetBranchAddress("EopElecVariables", &track);
    trees[i]->SetBranchAddress("charge", &track->charge);
    trees[i]->SetBranchAddress("nHits", &track->nHits);
    trees[i]->SetBranchAddress("nLostHits", &track->nLostHits);
    trees[i]->SetBranchAddress("innerOk", &track->innerOk);
    trees[i]->SetBranchAddress("outerRadius", &track->outerRadius);
    trees[i]->SetBranchAddress("chi2", &track->chi2);
    trees[i]->SetBranchAddress("normalizedChi2", &track->normalizedChi2);
    trees[i]->SetBranchAddress("px_rejected_track", &track->px_rejected_track);
    trees[i]->SetBranchAddress("py_rejected_track", &track->py_rejected_track);
    trees[i]->SetBranchAddress("pz_rejected_track", &track->pz_rejected_track);
    trees[i]->SetBranchAddress("px", &track->px);
    trees[i]->SetBranchAddress("py", &track->py);
    trees[i]->SetBranchAddress("pz", &track->pz);
    trees[i]->SetBranchAddress("p", &track->p);
    trees[i]->SetBranchAddress("pIn", &track->pIn);
    trees[i]->SetBranchAddress("etaIn", &track->etaIn);
    trees[i]->SetBranchAddress("phiIn", &track->phiIn);
    trees[i]->SetBranchAddress("pOut", &track->pOut);
    trees[i]->SetBranchAddress("etaOut", &track->etaOut);
    trees[i]->SetBranchAddress("phiOut", &track->phiOut);
    trees[i]->SetBranchAddress("pt", &track->pt);
    trees[i]->SetBranchAddress("ptError", &track->ptError);
    trees[i]->SetBranchAddress("theta", &track->theta);
    trees[i]->SetBranchAddress("eta", &track->eta);
    trees[i]->SetBranchAddress("phi", &track->phi);
    trees[i]->SetBranchAddress("fbrem", &track->fbrem);
    trees[i]->SetBranchAddress("MaxPtIn01", &track->MaxPtIn01);
    trees[i]->SetBranchAddress("SumPtIn01", &track->SumPtIn01);
    trees[i]->SetBranchAddress("NoTrackIn0015", &track->NoTrackIn0015);
    trees[i]->SetBranchAddress("MaxPtIn02", &track->MaxPtIn02);
    trees[i]->SetBranchAddress("SumPtIn02", &track->SumPtIn02);
    trees[i]->SetBranchAddress("NoTrackIn0020", &track->NoTrackIn0020);
    trees[i]->SetBranchAddress("MaxPtIn03", &track->MaxPtIn03);
    trees[i]->SetBranchAddress("SumPtIn03", &track->SumPtIn03);
    trees[i]->SetBranchAddress("NoTrackIn0025", &track->NoTrackIn0025);
    trees[i]->SetBranchAddress("MaxPtIn04", &track->MaxPtIn04);
    trees[i]->SetBranchAddress("SumPtIn04", &track->SumPtIn04);
    trees[i]->SetBranchAddress("NoTrackIn0030", &track->NoTrackIn0030);
    trees[i]->SetBranchAddress("MaxPtIn05", &track->MaxPtIn05);
    trees[i]->SetBranchAddress("SumPtIn05", &track->SumPtIn05);
    trees[i]->SetBranchAddress("NoTrackIn0035", &track->NoTrackIn0035);
    trees[i]->SetBranchAddress("NoTrackIn0040", &track->NoTrackIn0040);
    trees[i]->SetBranchAddress("SC_algoID", &track->SC_algoID);
    trees[i]->SetBranchAddress("SC_energy", &track->SC_energy);
    trees[i]->SetBranchAddress("SC_nBasicClus", &track->SC_nBasicClus);
    trees[i]->SetBranchAddress("SC_etaWidth", &track->SC_etaWidth);
    trees[i]->SetBranchAddress("SC_phiWidth", &track->SC_phiWidth);
    trees[i]->SetBranchAddress("SC_eta", &track->SC_eta);
    trees[i]->SetBranchAddress("SC_phi", &track->SC_phi);
    trees[i]->SetBranchAddress("SC_isBarrel", &track->SC_isBarrel);
    trees[i]->SetBranchAddress("SC_isEndcap", &track->SC_isEndcap);
    trees[i]->SetBranchAddress("dRto1stSC", &track->dRto1stSC);
    trees[i]->SetBranchAddress("dRto2ndSC", &track->dRto2ndSC);
    trees[i]->SetBranchAddress("HcalEnergyIn01", &track->HcalEnergyIn01);
    trees[i]->SetBranchAddress("HcalEnergyIn02", &track->HcalEnergyIn02);
    trees[i]->SetBranchAddress("HcalEnergyIn03", &track->HcalEnergyIn03);
    trees[i]->SetBranchAddress("HcalEnergyIn04", &track->HcalEnergyIn04);
    trees[i]->SetBranchAddress("HcalEnergyIn05", &track->HcalEnergyIn05);
    trees[i]->SetBranchAddress("isEcalDriven", &track->isEcalDriven);
    trees[i]->SetBranchAddress("isTrackerDriven", &track->isTrackerDriven);
    trees[i]->SetBranchAddress("RunNumber", &track->RunNumber);
    trees[i]->SetBranchAddress("EvtNumber", &track->EvtNumber);
  }
  return true;
}
