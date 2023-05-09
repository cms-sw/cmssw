#include <TStyle.h>
#include <TCanvas.h>
#include <TTree.h>

#include <iostream>
#include "TString.h"
#include "TAxis.h"
#include "TProfile.h"
#include "TF1.h"
#include "TH1.h"
#include "TH2F.h"
#include "TGraphErrors.h"
#include "TROOT.h"
#include "TDirectory.h"
#include "TFile.h"
#include "TDirectoryFile.h"
#include "TLegend.h"
#include "TPaveLabel.h"
#include "TChain.h"
#include "TMath.h"
#include "TLatex.h"
#include "TVirtualFitter.h"
#include "TMatrixD.h"
#include "../interface/EopVariables.h"

using namespace std;

namespace eop {

  vector<Double_t> extractgausparams(TH1F *histo, TString option1 = "0", TString option2 = "");
  vector<Double_t> extractgausparams(TH1F *histo, TString option1, TString option2) {
    if (histo->GetEntries() < 10) {
      return {0., 0., 0., 0.};
    }

    TF1 *f1 = new TF1("f1", "gaus");
    f1->SetRange(0., 2.);
    histo->Fit("f1", "QR0L");

    double mean = f1->GetParameter(1);
    double deviation = f1->GetParameter(2);

    double lowLim;
    double upLim;
    double newmean;

    double degrade = 0.05;
    unsigned int iteration = 0;

    while (iteration == 0 || (f1->GetProb() < 0.001 && iteration < 10)) {
      lowLim = mean - (2.0 * deviation * (1 - degrade * iteration));
      upLim = mean + (2.0 * deviation * (1 - degrade * iteration));

      f1->SetRange(lowLim, upLim);
      histo->Fit("f1", "QRL0");

      newmean = mean;
      if (f1->GetParameter(1) < 2 && f1->GetParameter(1) > 0)
        newmean = f1->GetParameter(1);
      lowLim = newmean - (1.5 * deviation);  //*(1-degrade*iteration));
      upLim = newmean + (1.5 * deviation);   //*(1-degrade*iteration));
      f1->SetRange(lowLim, upLim);
      f1->SetLineWidth(1);
      histo->Fit("f1", "QRL+i" + option1, option2);
      if (f1->GetParameter(1) < 2 && f1->GetParameter(1) > 0)
        mean = f1->GetParameter(1);
      deviation = f1->GetParameter(2);
      iteration++;
    }
    vector<Double_t> params;
    params.push_back(f1->GetParameter(1));
    params.push_back(f1->GetParError(1));
    params.push_back(lowLim);
    params.push_back(upLim);
    return params;
  }

  void momentumBiasValidation(
      TString variable = "eta",
      TString path = "/scratch/hh/current/cms/user/henderle/",
      TString alignmentWithLabel =
          "EOP_TBDkbMinBias_2011.root=Summer 2011\\EOP_TBD_2011.root=no bows\\EOP_TBDfrom0T_2011.root=from 0T, no "
          "bows\\EOP_plain_2011.root=no mass constraint",
      bool verbose = false,
      Double_t givenMin = 0.,
      Double_t givenMax = 0.) {
    time_t start = time(0);

    // give radius at which the misalignment is calculated (arbitrary)
    Double_t radius = 1.;

    // configure style
    gROOT->cd();
    gROOT->SetStyle("Plain");
    if (verbose) {
      std::cout << "\n all formerly produced control plots in ./controlEOP/ deleted.\n" << std::endl;
      gROOT->ProcessLine(".!if [ -d controlEOP ]; then rm controlEOP/control_eop*.eps; else mkdir controlEOP; fi");
    }
    gStyle->SetPadBorderMode(0);
    gStyle->SetOptStat(0);
    gStyle->SetTitleFont(42, "XYZ");
    gStyle->SetLabelFont(42, "XYZ");
    gStyle->SetEndErrorSize(5);
    gStyle->SetLineStyleString(2, "80 20");
    gStyle->SetLineStyleString(3, "40 18");
    gStyle->SetLineStyleString(4, "20 16");

    // create canvas
    TCanvas *c = new TCanvas("c", "Canvas", 0, 0, 1150, 880);
    TCanvas *ccontrol = new TCanvas("ccontrol", "controlCanvas", 0, 0, 600, 300);

    // create angular binning
    TString binVar;
    if (variable == "eta")
      binVar = "track_eta";
    else if (variable == "phi" || variable == "phi1" || variable == "phi2" || variable == "phi3")
      binVar = "track_phi";
    else {
      std::cout << "variable can be eta or phi" << std::endl;
      std::cout << "phi1, phi2 and phi3 are for phi in different eta bins" << std::endl;
    }

    Int_t numberOfBins = 0;
    Double_t startbin = 0.;
    Double_t lastbin = 0.;
    if (binVar == "track_eta") {
      // tracks beyond abs(eta)=1.65 do not have radii > 99cm (cut value).
      startbin = -1.65;
      lastbin = 1.65;
      numberOfBins = 18;  // ~ outermost TEC
      //startbin = -.917; lastbin = .917; numberOfBins = 10;// ~ outermost TOB
      //startbin = -1.28; lastbin = 1.28; numberOfBins = 14;// ~ innermost TOB
    }
    if (binVar == "track_phi") {
      startbin = -TMath::Pi();
      lastbin = TMath::Pi();
      numberOfBins = 18;
    }
    Double_t binsize = (lastbin - startbin) / numberOfBins;
    vector<Double_t> binningstrvec;
    vector<Double_t> zBinning_;
    for (Int_t i = 0; i <= numberOfBins; i++) {
      binningstrvec.push_back(startbin + i * binsize);
      zBinning_.push_back(radius * 100. / TMath::Tan(2 * TMath::ATan(TMath::Exp(-(startbin + i * binsize)))));
    }

    // create energy binning
    Int_t startStep = 0;
    const Int_t NumberOFSteps = 2;
    Double_t steparray[NumberOFSteps + 1] = {50, 58, 80};  //data 2 steps
    //const Int_t NumberOFSteps = 6;Double_t steparray[NumberOFSteps+1] = {50,55,60,65,70,75,80};//mc 6 steps

    vector<Double_t> steps;
    vector<TString> stepstrg;
    Char_t tmpstep[10];
    for (Int_t ii = 0; ii <= NumberOFSteps; ii++) {
      steps.push_back(steparray[ii]);
      sprintf(tmpstep, "%1.1fGeV", steparray[ii]);
      stepstrg.push_back(tmpstep);
    }

    // read files
    vector<TFile *> file;
    vector<TString> label;

    TObjArray *fileLabelPairs = alignmentWithLabel.Tokenize("\\");
    for (Int_t i = 0; i < fileLabelPairs->GetEntries(); ++i) {
      TObjArray *singleFileLabelPair = TString(fileLabelPairs->At(i)->GetName()).Tokenize("=");

      if (singleFileLabelPair->GetEntries() == 2) {
        file.push_back(new TFile(path + (TString(singleFileLabelPair->At(0)->GetName()))));
        label.push_back(singleFileLabelPair->At(1)->GetName());
      } else {
        std::cout << "Please give file name and legend entry in the following form:\n"
                  << " filename1=legendentry1\\filename2=legendentry2\\..." << std::endl;
      }
    }
    cout << "number of files: " << file.size() << endl;

    // create trees
    vector<TTree *> tree;
    EopVariables *track = new EopVariables();

    for (unsigned int i = 0; i < file.size(); i++) {
      tree.push_back((TTree *)file[i]->Get("energyOverMomentumTree/EopTree"));
      tree[i]->SetMakeClass(1);
      tree[i]->SetBranchAddress("EopVariables", &track);
      tree[i]->SetBranchAddress("track_outerRadius", &(track->track_outerRadius));
      tree[i]->SetBranchAddress("track_chi2", &track->track_chi2);
      tree[i]->SetBranchAddress("track_normalizedChi2", &track->track_normalizedChi2);
      tree[i]->SetBranchAddress("track_p", &track->track_p);
      tree[i]->SetBranchAddress("track_pt", &track->track_pt);
      tree[i]->SetBranchAddress("track_ptError", &track->track_ptError);
      tree[i]->SetBranchAddress("track_theta", &track->track_theta);
      tree[i]->SetBranchAddress("track_eta", &track->track_eta);
      tree[i]->SetBranchAddress("track_phi", &track->track_phi);
      tree[i]->SetBranchAddress("track_emc1", &track->track_emc1);
      tree[i]->SetBranchAddress("track_emc3", &track->track_emc3);
      tree[i]->SetBranchAddress("track_emc5", &track->track_emc5);
      tree[i]->SetBranchAddress("track_hac1", &track->track_hac1);
      tree[i]->SetBranchAddress("track_hac3", &track->track_hac3);
      tree[i]->SetBranchAddress("track_hac5", &track->track_hac5);
      tree[i]->SetBranchAddress("track_maxPNearby", &track->track_maxPNearby);
      tree[i]->SetBranchAddress("track_EnergyIn", &track->track_EnergyIn);
      tree[i]->SetBranchAddress("track_EnergyOut", &track->track_EnergyOut);
      tree[i]->SetBranchAddress("distofmax", &track->distofmax);
      tree[i]->SetBranchAddress("track_charge", &track->track_charge);
      tree[i]->SetBranchAddress("track_nHits", &track->track_nHits);
      tree[i]->SetBranchAddress("track_nLostHits", &track->track_nLostHits);
      tree[i]->SetBranchAddress("track_innerOk", &track->track_innerOk);
    }

    // create histogram vectors

    // basic histograms
    vector<TH1F *> histonegative;
    vector<TH1F *> histopositive;
    vector<TH2F *> Energynegative;
    vector<TH2F *> Energypositive;
    vector<TH1F *> xAxisBin;
    vector<Int_t> selectedTracks;
    // intermediate histograms
    vector<TH1F *> fithisto;
    vector<TH1F *> combinedxAxisBin;
    // final histograms
    vector<TGraphErrors *> overallGraph;
    vector<TH1F *> overallhisto;

    // book histograms
    // basic histograms
    for (UInt_t i = 0; i < NumberOFSteps * numberOfBins * file.size(); i++) {
      Char_t histochar[20];
      sprintf(histochar, "%i", i + 1);
      TString histostrg = histochar;
      TString lowEnergyBorder = stepstrg[i % NumberOFSteps];
      TString highEnergyBorder = stepstrg[i % NumberOFSteps + 1];
      if (binVar == "track_eta")
        histonegative.push_back(
            new TH1F("histonegative" + histostrg, lowEnergyBorder + " #leq E < " + highEnergyBorder, 50, 0, 2));
      else
        histonegative.push_back(
            new TH1F("histonegative" + histostrg, lowEnergyBorder + " #leq E_{T} < " + highEnergyBorder, 50, 0, 2));
      histopositive.push_back(new TH1F("histopositive" + histostrg, "histopositive" + histostrg, 50, 0, 2));
      Energynegative.push_back(
          new TH2F("Energynegative" + histostrg, "Energynegative" + histostrg, 5000, 0, 500, 50, 0, 2));
      Energypositive.push_back(
          new TH2F("Energypositive" + histostrg, "Energypositive" + histostrg, 5000, 0, 500, 50, 0, 2));
      xAxisBin.push_back(new TH1F("xAxisBin" + histostrg, "xAxisBin" + histostrg, 1000, -500, 500));
      selectedTracks.push_back(0);
      Energynegative[i]->Sumw2();
      Energypositive[i]->Sumw2();
      xAxisBin[i]->Sumw2();
      histonegative[i]->Sumw2();
      histopositive[i]->Sumw2();
      histonegative[i]->SetLineColor(kGreen);
      histopositive[i]->SetLineColor(kRed);
    }
    // intermediate histograms
    for (UInt_t i = 0; i < numberOfBins * file.size(); i++) {
      Char_t histochar[20];
      sprintf(histochar, "%i", i + 1);
      TString histostrg = histochar;
      fithisto.push_back(new TH1F("fithisto" + histostrg, "fithisto" + histostrg, NumberOFSteps, 0, NumberOFSteps));
      combinedxAxisBin.push_back(
          new TH1F("combinedxAxisBin" + histostrg, "combinedxAxisBin" + histostrg, NumberOFSteps, 0, NumberOFSteps));
    }
    // final histograms
    for (UInt_t i = 0; i < file.size(); i++) {
      TString cnt;
      cnt += i;
      if (binVar == "track_eta") {
        overallhisto.push_back(new TH1F("overallhisto" + cnt, "overallhisto" + cnt, numberOfBins, &zBinning_.front()));
        overallGraph.push_back(new TGraphErrors(overallhisto.back()));
      } else {
        overallhisto.push_back(new TH1F(
            "overallhisto" + cnt, "overallhisto" + cnt, numberOfBins, startbin, startbin + numberOfBins * binsize));
        overallGraph.push_back(new TGraphErrors(overallhisto.back()));
      }
    }

    // fill basic histograms
    Long64_t nevent;
    Int_t usedtracks;
    for (UInt_t isample = 0; isample < file.size(); isample++) {
      usedtracks = 0;
      nevent = (Long64_t)tree[isample]->GetEntries();
      cout << nevent << " tracks in sample " << isample + 1 << endl;
      for (Long64_t ientry = 0; ientry < nevent; ++ientry) {
        tree[isample]->GetEntry(ientry);
        for (Int_t i = 0; i < numberOfBins; i++) {
          for (Int_t iStep = startStep; iStep < NumberOFSteps; iStep++) {
            Double_t lowerEcut = steps[iStep];
            Double_t upperEcut = steps[iStep + 1];
            Double_t lowcut = binningstrvec[i];
            Double_t highcut = binningstrvec[i + 1];
            // track selection
            if (track->track_nHits >= 13 && track->track_nLostHits == 0 && track->track_outerRadius > 99. &&
                track->track_normalizedChi2 < 5. && track->track_EnergyIn < 1. && track->track_EnergyOut < 8. &&
                track->track_hac3 > lowerEcut && track->track_hac3 < upperEcut) {
              if (binVar == "track_eta") {
                if (track->track_eta >= lowcut && track->track_eta < highcut) {
                  usedtracks++;
                  selectedTracks[i * NumberOFSteps + iStep + isample * (NumberOFSteps * numberOfBins)]++;
                  xAxisBin[i * NumberOFSteps + iStep + isample * (NumberOFSteps * numberOfBins)]->Fill(
                      radius * 100. / TMath::Tan(2 * TMath::ATan(TMath::Exp(-(track->track_eta)))));
                  if (track->track_charge < 0) {
                    histonegative[i * NumberOFSteps + iStep + isample * (NumberOFSteps * numberOfBins)]->Fill(
                        (track->track_hac3) / track->track_p);
                    Energynegative[i * NumberOFSteps + iStep + isample * (NumberOFSteps * numberOfBins)]->Fill(
                        track->track_hac3 * TMath::Sin(track->track_theta), (track->track_hac3) / track->track_p);
                  }
                  if (track->track_charge > 0) {
                    histopositive[i * NumberOFSteps + iStep + isample * (NumberOFSteps * numberOfBins)]->Fill(
                        (track->track_hac3) / track->track_p);
                    Energypositive[i * NumberOFSteps + iStep + isample * (NumberOFSteps * numberOfBins)]->Fill(
                        track->track_hac3 * TMath::Sin(track->track_theta), (track->track_hac3) / track->track_p);
                  }
                }
              } else if (binVar == "track_phi") {
                if ((variable == "phi1" && track->track_eta < -0.9) ||
                    (variable == "phi2" && TMath::Abs(track->track_eta) < 0.9) ||
                    (variable == "phi3" && track->track_eta > 0.9) || variable == "phi")
                  if (track->track_phi >= lowcut && track->track_phi < highcut) {
                    usedtracks++;
                    selectedTracks[i * NumberOFSteps + iStep + isample * (NumberOFSteps * numberOfBins)]++;
                    xAxisBin[i * NumberOFSteps + iStep + isample * (NumberOFSteps * numberOfBins)]->Fill(
                        track->track_phi);
                    if (track->track_charge < 0) {
                      histonegative[i * NumberOFSteps + iStep + isample * (NumberOFSteps * numberOfBins)]->Fill(
                          (track->track_hac3) / track->track_p);
                      Energynegative[i * NumberOFSteps + iStep + isample * (NumberOFSteps * numberOfBins)]->Fill(
                          track->track_hac3 * TMath::Sin(track->track_theta), (track->track_hac3) / track->track_p);
                    }
                    if (track->track_charge > 0) {
                      histopositive[i * NumberOFSteps + iStep + isample * (NumberOFSteps * numberOfBins)]->Fill(
                          (track->track_hac3) / track->track_p);
                      Energypositive[i * NumberOFSteps + iStep + isample * (NumberOFSteps * numberOfBins)]->Fill(
                          track->track_hac3 * TMath::Sin(track->track_theta), (track->track_hac3) / track->track_p);
                    }
                  }
              }
            }
          }
        }
      }
      cout << "number of used tracks in this sample: " << usedtracks << endl;
    }
    // calculate misalignment per bin
    Double_t misalignment;
    Double_t misaliUncert;
    vector<TString> misalignmentfit;
    vector<TF1 *> f2;

    for (UInt_t isample = 0; isample < file.size(); isample++) {
      for (Int_t i = 0; i < numberOfBins; i++) {
        if (verbose)
          cout << binningstrvec[i] << " < " + binVar + " < " << binningstrvec[i + 1] << endl;
        for (Int_t iStep = startStep; iStep < NumberOFSteps; iStep++) {
          vector<Double_t> curvNeg;
          vector<Double_t> curvPos;
          if (verbose) {
            TString controlName = "controlEOP/control_eop_sample";
            controlName += isample;
            controlName += "_bin";
            controlName += i;
            controlName += "_energy";
            controlName += iStep;
            controlName += ".eps";
            ccontrol->cd();
            Double_t posMax =
                histopositive[i * NumberOFSteps + iStep + isample * (NumberOFSteps * numberOfBins)]->GetMaximum();
            Double_t negMax =
                histonegative[i * NumberOFSteps + iStep + isample * (NumberOFSteps * numberOfBins)]->GetMaximum();
            if (posMax > negMax)
              histonegative[i * NumberOFSteps + iStep + isample * (NumberOFSteps * numberOfBins)]->SetMaximum(1.1 *
                                                                                                              posMax);
            histonegative[i * NumberOFSteps + iStep + isample * (NumberOFSteps * numberOfBins)]->DrawClone();
            histopositive[i * NumberOFSteps + iStep + isample * (NumberOFSteps * numberOfBins)]->DrawClone("same");
            curvNeg = eop::extractgausparams(
                histonegative[i * NumberOFSteps + iStep + isample * (NumberOFSteps * numberOfBins)], "", "");
            curvPos = eop::extractgausparams(
                histopositive[i * NumberOFSteps + iStep + isample * (NumberOFSteps * numberOfBins)], "", "same");
            ccontrol->Print(controlName);
          } else {
            curvNeg = eop::extractgausparams(
                histonegative[i * NumberOFSteps + iStep + isample * (NumberOFSteps * numberOfBins)]);
            curvPos = eop::extractgausparams(
                histopositive[i * NumberOFSteps + iStep + isample * (NumberOFSteps * numberOfBins)]);
          }

          misalignment = 0.;
          misaliUncert = 1000.;

          if (selectedTracks[i * NumberOFSteps + iStep + isample * (NumberOFSteps * numberOfBins)] != 0) {
            Energynegative[i * NumberOFSteps + iStep + isample * (NumberOFSteps * numberOfBins)]
                ->GetYaxis()
                ->SetRangeUser(curvNeg[2], curvNeg[3]);
            Energypositive[i * NumberOFSteps + iStep + isample * (NumberOFSteps * numberOfBins)]
                ->GetYaxis()
                ->SetRangeUser(curvPos[2], curvPos[3]);

            Double_t meanEnergyNeg =
                Energynegative[i * NumberOFSteps + iStep + isample * (NumberOFSteps * numberOfBins)]->GetMean();
            Double_t meanEnergyPos =
                Energypositive[i * NumberOFSteps + iStep + isample * (NumberOFSteps * numberOfBins)]->GetMean();
            Double_t meanEnergy = (meanEnergyNeg + meanEnergyPos) /
                                  2.;  // use mean of positive and negative tracks to reduce energy dependence
            if (verbose)
              std::cout << "difference in energy between positive and negative tracks: "
                        << meanEnergyNeg - meanEnergyPos << std::endl;

            if (binVar == "track_eta") {
              misalignment = 1000000. * 0.5 *
                             (-TMath::ASin((0.57 * radius / meanEnergy) * curvNeg[0]) +
                              TMath::ASin((0.57 * radius / meanEnergy) * curvPos[0]));
              misaliUncert =
                  1000000. * 0.5 *
                  (TMath::Sqrt(((0.57 * 0.57 * radius * radius * curvNeg[1] * curvNeg[1]) /
                                (meanEnergy * meanEnergy - 0.57 * 0.57 * radius * radius * curvNeg[0] * curvNeg[0])) +
                               ((0.57 * 0.57 * radius * radius * curvPos[1] * curvPos[1]) /
                                (meanEnergy * meanEnergy - 0.57 * 0.57 * radius * radius * curvPos[0] * curvPos[0]))));
            }
            if (binVar == "track_phi") {
              misalignment = 1000. * (curvPos[0] - curvNeg[0]) / (curvPos[0] + curvNeg[0]);
              misaliUncert = 1000. * 2 / ((curvPos[0] + curvNeg[0]) * (curvPos[0] + curvNeg[0])) *
                             TMath::Sqrt((curvPos[0] * curvPos[0] * curvPos[1] * curvPos[1]) +
                                         (curvNeg[0] * curvNeg[0] * curvNeg[1] * curvNeg[1]));
            }
          }
          if (verbose)
            cout << "misalignment: " << misalignment << "+-" << misaliUncert << endl << endl;
          // fill intermediate histograms
          fithisto[i + isample * numberOfBins]->SetBinContent(iStep + 1, misalignment);
          fithisto[i + isample * numberOfBins]->SetBinError(iStep + 1, misaliUncert);
          Double_t xBinCentre =
              xAxisBin[i * NumberOFSteps + iStep + isample * (NumberOFSteps * numberOfBins)]->GetMean();
          Double_t xBinCenUnc =
              xAxisBin[i * NumberOFSteps + iStep + isample * (NumberOFSteps * numberOfBins)]->GetMeanError();
          combinedxAxisBin[i + isample * numberOfBins]->SetBinContent(iStep + 1, xBinCentre);
          combinedxAxisBin[i + isample * numberOfBins]->SetBinError(iStep + 1, xBinCenUnc);
        }

        Double_t overallmisalignment;
        Double_t overallmisaliUncert;
        Double_t overallxBin{0.f};
        Double_t overallxBinUncert{0.f};
        // calculate mean of different energy bins
        if (NumberOFSteps > 1 && (fithisto[i + isample * numberOfBins]->Integral() > 0)) {
          TF1 *fit = new TF1("fit", "pol0", startStep, NumberOFSteps);
          TF1 *fit2 = new TF1("fit2", "pol0", startStep, NumberOFSteps);
          if (verbose)
            fithisto[i + isample * numberOfBins]->Fit("fit", "0");
          else
            fithisto[i + isample * numberOfBins]->Fit("fit", "Q0");
          combinedxAxisBin[i + isample * numberOfBins]->Fit("fit2", "Q0");
          overallmisalignment = fit->GetParameter(0);
          overallmisaliUncert = fit->GetParError(0);
          overallxBin = fit2->GetParameter(0);
          overallxBinUncert = fit2->GetParError(0);
          fit->Delete();
        } else {
          overallmisalignment = misalignment;
          overallmisaliUncert = misaliUncert;
        }
        // fill final histograms
        overallhisto[isample]->SetBinContent(i + 1, overallmisalignment);
        overallhisto[isample]->SetBinError(i + 1, overallmisaliUncert);
        overallGraph[isample]->SetPoint(i, overallxBin, overallmisalignment);
        overallGraph[isample]->SetPointError(i, overallxBinUncert, overallmisaliUncert);
      }
    }

    // create legend
    TLegend *leg = new TLegend(0.13, 0.74, 0.98, 0.94);
    leg->SetFillColor(10);
    leg->SetTextFont(42);
    leg->SetTextSize(0.038);
    if (variable == "phi1")
      leg->SetHeader("low #eta tracks (#eta < -0.9)");
    if (variable == "phi2")
      leg->SetHeader("central tracks (|#eta| < 0.9)");
    if (variable == "phi3")
      leg->SetHeader("high #eta tracks (#eta > 0.9)");

    for (UInt_t isample = 0; isample < file.size(); isample++) {
      // configure final histogram
      if (binVar == "track_eta") {
        overallhisto[isample]->GetXaxis()->SetTitle("z [cm]");
        overallhisto[isample]->GetYaxis()->SetTitle("#Delta#phi [#murad]");
      }
      if (binVar == "track_phi") {
        overallhisto[isample]->GetXaxis()->SetTitle("#phi");
        overallhisto[isample]->GetYaxis()->SetTitle("#Delta#phi [a.u.]");
      }
      overallhisto[isample]->GetYaxis()->SetTitleOffset(1.05);
      overallhisto[isample]->GetYaxis()->SetTitleSize(0.065);
      overallhisto[isample]->GetYaxis()->SetLabelSize(0.065);
      overallhisto[isample]->GetXaxis()->SetTitleOffset(0.8);
      overallhisto[isample]->GetXaxis()->SetTitleSize(0.065);
      overallhisto[isample]->GetXaxis()->SetLabelSize(0.065);
      overallhisto[isample]->SetLineWidth(2);
      overallhisto[isample]->SetLineColor(isample + 1);
      overallhisto[isample]->SetMarkerColor(isample + 1);
      overallGraph[isample]->SetLineWidth(2);
      overallGraph[isample]->SetLineColor(isample + 1);
      overallGraph[isample]->SetMarkerColor(isample + 1);
      if (isample == 2) {
        overallhisto[isample]->SetLineColor(kGreen + 3);
        overallhisto[isample]->SetMarkerColor(kGreen + 3);
        overallGraph[isample]->SetLineColor(kGreen + 3);
        overallGraph[isample]->SetMarkerColor(kGreen + 3);
      }
      overallGraph[isample]->SetMarkerStyle(isample + 20);
      overallGraph[isample]->SetMarkerSize(2);

      // fit to final histogram
      Char_t funchar[10];
      sprintf(funchar, "func%i", isample + 1);
      TString func = funchar;
      if (binVar == "track_eta")
        f2.push_back(new TF1(func, "[0]+[1]*x/100.", -500, 500));  //Divide by 100. cm->m
      if (binVar == "track_phi")
        f2.push_back(new TF1(func, "[0]+[1]*TMath::Cos(x+[2])", -500, 500));

      f2[isample]->SetLineColor(isample + 1);
      if (isample == 2)
        f2[isample]->SetLineColor(kGreen + 3);
      f2[isample]->SetLineStyle(isample + 1);
      f2[isample]->SetLineWidth(2);

      if (verbose) {
        if (overallGraph[isample]->Integral() != 0.f) {
          overallGraph[isample]->Fit(func, "mR0+");

          cout << "Covariance Matrix:" << endl;
          TVirtualFitter *fitter = TVirtualFitter::GetFitter();
          TMatrixD matrix(2, 2, fitter->GetCovarianceMatrix());
          Double_t oneOne = fitter->GetCovarianceMatrixElement(0, 0);
          Double_t oneTwo = fitter->GetCovarianceMatrixElement(0, 1);
          Double_t twoOne = fitter->GetCovarianceMatrixElement(1, 0);
          Double_t twoTwo = fitter->GetCovarianceMatrixElement(1, 1);

          cout << "( " << oneOne << ", " << twoOne << ")" << endl;
          cout << "( " << oneTwo << ", " << twoTwo << ")" << endl;
        }
      } else {
        if (overallGraph[isample]->Integral() != 0) {
          overallGraph[isample]->Fit(func, "QmR0+");
        }
      }

      // print fit parameter
      if (binVar == "track_eta")
        cout << "const: " << f2[isample]->GetParameter(0) << "+-" << f2[isample]->GetParError(0)
             << ", slope: " << f2[isample]->GetParameter(1) << "+-" << f2[isample]->GetParError(1) << endl;
      if (binVar == "track_phi")
        cout << "const: " << f2[isample]->GetParameter(0) << "+-" << f2[isample]->GetParError(0)
             << ", amplitude: " << f2[isample]->GetParameter(1) << "+-" << f2[isample]->GetParError(1)
             << ", shift: " << f2[isample]->GetParameter(2) << "+-" << f2[isample]->GetParError(2) << endl;
      cout << "fit probability: " << f2[isample]->GetProb() << endl;
      if (verbose)
        cout << "chi^2/Ndof: " << f2[isample]->GetChisquare() / f2[isample]->GetNDF() << endl;
      // write fit function to legend
      Char_t misalignmentfitchar[20];
      sprintf(misalignmentfitchar, "%1.f", f2[isample]->GetParameter(0));
      misalignmentfit.push_back("(");
      misalignmentfit[isample] += misalignmentfitchar;
      misalignmentfit[isample] += "#pm";
      sprintf(misalignmentfitchar, "%1.f", f2[isample]->GetParError(0));
      misalignmentfit[isample] += misalignmentfitchar;
      if (variable == "eta") {
        misalignmentfit[isample] += ")#murad #upoint r[m] + (";
        sprintf(misalignmentfitchar, "%1.f", f2[isample]->GetParameter(1));
        misalignmentfit[isample] += misalignmentfitchar;
        misalignmentfit[isample] += "#pm";
        sprintf(misalignmentfitchar, "%1.f", f2[isample]->GetParError(1));
        misalignmentfit[isample] += misalignmentfitchar;
        misalignmentfit[isample] += ")#murad #upoint z[m]";
      } else if (variable.Contains("phi")) {
        misalignmentfit[isample] += ") + (";
        sprintf(misalignmentfitchar, "%1.f", f2[isample]->GetParameter(1));
        misalignmentfit[isample] += misalignmentfitchar;
        misalignmentfit[isample] += "#pm";
        sprintf(misalignmentfitchar, "%1.f", f2[isample]->GetParError(1));
        misalignmentfit[isample] += misalignmentfitchar;
        misalignmentfit[isample] += ") #upoint cos(#phi";
        if (f2[isample]->GetParameter(2) > 0.)
          misalignmentfit[isample] += "+";
        sprintf(misalignmentfitchar, "%1.1f", f2[isample]->GetParameter(2));
        misalignmentfit[isample] += misalignmentfitchar;
        misalignmentfit[isample] += "#pm";
        sprintf(misalignmentfitchar, "%1.1f", f2[isample]->GetParError(2));
        misalignmentfit[isample] += misalignmentfitchar;
        misalignmentfit[isample] += ")";
      }

      // set pad margins
      c->cd();
      gPad->SetTopMargin(0.06);
      gPad->SetBottomMargin(0.12);
      gPad->SetLeftMargin(0.13);
      gPad->SetRightMargin(0.02);

      // determine resonable y-axis range
      Double_t overallmax = 0.;
      Double_t overallmin = 0.;
      if (isample == 0) {
        for (UInt_t i = 0; i < file.size(); i++) {
          overallmax = max(
              overallhisto[i]->GetMaximum() + overallhisto[i]->GetBinError(overallhisto[i]->GetMaximumBin()) +
                  0.55 *
                      (overallhisto[i]->GetMaximum() + overallhisto[i]->GetBinError(overallhisto[i]->GetMaximumBin()) -
                       overallhisto[i]->GetMinimum() + overallhisto[i]->GetBinError(overallhisto[i]->GetMinimumBin())),
              overallmax);
          overallmin = min(
              overallhisto[i]->GetMinimum() - fabs(overallhisto[i]->GetBinError(overallhisto[i]->GetMinimumBin())) -
                  0.1 *
                      (overallhisto[i]->GetMaximum() + overallhisto[i]->GetBinError(overallhisto[i]->GetMaximumBin()) -
                       overallhisto[i]->GetMinimum() + overallhisto[i]->GetBinError(overallhisto[i]->GetMinimumBin())),
              overallmin);
        }
        overallhisto[isample]->SetMaximum(overallmax);
        overallhisto[isample]->SetMinimum(overallmin);
        if (givenMax)
          overallhisto[isample]->SetMaximum(givenMax);
        if (givenMin)
          overallhisto[isample]->SetMinimum(givenMin);
        overallhisto[isample]->DrawClone("axis");
      }
      // set histogram errors to a small value as only errors of the graph should be shown
      for (int i = 0; i < overallhisto[isample]->GetNbinsX(); i++) {
        overallhisto[isample]->SetBinError(i + 1, 0.00001);
      }
      // draw final histogram
      overallhisto[isample]->DrawClone("pe1 same");
      f2[isample]->DrawClone("same");
      overallGraph[isample]->DrawClone("|| same");
      overallGraph[isample]->DrawClone("pz same");
      overallGraph[isample]->SetLineStyle(isample + 1);
      leg->AddEntry(overallGraph[isample], label[isample] + " (" + misalignmentfit[isample] + ")", "Lp");
    }
    leg->Draw();

    TPaveLabel *CMSlabel = new TPaveLabel(0.13, 0.94, 0.5, 1., "CMS preliminary 2012", "br NDC");
    CMSlabel->SetFillStyle(0);
    CMSlabel->SetBorderSize(0);
    CMSlabel->SetTextSize(0.95);
    CMSlabel->SetTextAlign(12);
    CMSlabel->SetTextFont(42);
    CMSlabel->Draw("same");

    // print plots
    if (variable == "eta") {
      c->Print("twist_validation.eps");
      if (verbose)
        c->Print("twist_validation.root");
    } else if (variable == "phi") {
      c->Print("sagitta_validation_all.eps");
      if (verbose)
        c->Print("sagitta_validation_all.root");
    } else if (variable == "phi1") {
      c->Print("sagitta_validation_lowEta.eps");
      if (verbose)
        c->Print("sagitta_validation_lowEta.root");
    } else if (variable == "phi2") {
      c->Print("sagitta_validation_centralEta.eps");
      if (verbose)
        c->Print("sagitta_validation_centralEta.root");
    } else if (variable == "phi3") {
      c->Print("sagitta_validation_highEta.eps");
      if (verbose)
        c->Print("sagitta_validation_highEta.root");
    }

    time_t end = time(0);
    cout << "Done in " << int(difftime(end, start)) / 60 << " min and " << int(difftime(end, start)) % 60 << " sec."
         << endl;
  }

}  // namespace eop
