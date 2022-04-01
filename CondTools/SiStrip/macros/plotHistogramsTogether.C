/*************************************************
  Automatically plots histograms from two files
  onto the same plot and saves them.
  It looksfor ALL histograms in the first file
  and plots the corresponding histogram in the 2nd
  file onto the sample plot.
  
  Can be run from a bash prompt as well:
    root -b -l -q "plotHistogramsTogether.C(\"fileA.root\",\"fileB.root\")"
    root -b -l -q "plotHistogramsTogether.C(\"fileA.root\",\"fileB.root\",\"Signal\",\"Background\",10,2,1)"

  Michael B. Anderson
  Sept 5, 2008
*************************************************/

#include <string.h>
#include "TFile.h"
#include "TH1.h"
#include "TKey.h"
#include "Riostream.h"

// Accesable everywhere
TObject *obj;
TFile *sourceFile1, *sourceFile2;
TString label1, label2;
TString outputFolder, outputFilename;
TCanvas *canvasDefault;
Float_t scale1, scale2;
bool showStatsBoxes;

// *******************************************
// Variables
TString imageType = "png";
int outputWidth = 480;
int outputHeight = 360;
bool yAxisLogScale = false;
// End of Variables
// *******************************************

void recurseOverKeys(TDirectory *target1);
void plot2Histograms(TH1 *htemp1, TH1 *htemp2, TString filename);

void plotHistogramsTogether(TString fileName1,
                            TString fileName2,
                            TString fileLabel1 = "",
                            TString fileLabel2 = "",
                            Float_t fileScale1 = 1.0,
                            Float_t fileScale2 = 1.0,
                            bool showStats = false) {
  // If file labels were not given as argument,
  // use the filename as a label
  if (fileLabel1 == "") {
    fileLabel1 = fileName1;
    fileLabel2 = fileName2;
    fileLabel1.ReplaceAll(".root", "");
    fileLabel1.ReplaceAll(".root", "");
  }
  label1 = fileLabel1;
  label2 = fileLabel2;

  // Set the scale of the histograms.
  // If they are < 0.0, they will be area normalized
  scale1 = fileScale1;
  scale2 = fileScale2;
  showStatsBoxes = showStats;

  sourceFile1 = TFile::Open(fileName1);
  sourceFile2 = TFile::Open(fileName2);

  outputFolder = "HistogramsTogether/";  // Blank to use current directory,
                                         // or, for a specific dir type
                                         // something like "images/"

  gSystem->MakeDirectory(outputFolder);

  canvasDefault = new TCanvas("canvasDefault", "testCanvas", outputWidth, outputHeight);

  // This function will plot all histograms from
  // file1 against matching histogram from file2
  recurseOverKeys(sourceFile1);

  sourceFile1->Close();
  sourceFile2->Close();

  TString currentDir = gSystem->pwd();
  cout << "Done. See images in:" << endl << currentDir << "/" << outputFolder << endl;
}

void recurseOverKeys(TDirectory *target1) {
  // Figure out where we are
  TString path((char *)strstr(target1->GetPath(), ":"));
  path.Remove(0, 2);

  sourceFile1->cd(path);

  TDirectory *current_sourcedir = gDirectory;

  TKey *key;
  TIter nextkey(current_sourcedir->GetListOfKeys());

  while (key = (TKey *)nextkey()) {
    obj = key->ReadObj();

    // Check if this is a 1D histogram or a directory
    if (obj->IsA()->InheritsFrom("TH1")) {
      // **************************
      // Plot & Save this Histogram
      TH1 *htemp1, *htemp2;

      htemp1 = (TH1 *)obj;
      TString histName = htemp1->GetName();

      if (path != "") {
        sourceFile2->GetObject(path + "/" + histName, htemp2);
      } else {
        sourceFile2->GetObject(histName, htemp2);
      }

      outputFilename = histName;
      plot2Histograms(htemp1, htemp2, outputFolder + path + "/" + outputFilename + "." + imageType);

    } else if (obj->IsA()->InheritsFrom("TDirectory")) {
      // it's a subdirectory

      cout << "Found subdirectory " << obj->GetName() << endl;
      gSystem->MakeDirectory(outputFolder + path + "/" + obj->GetName());

      // obj is now the starting point of another round of merging
      // obj still knows its depth within the target file via
      // GetPath(), so we can still figure out where we are in the recursion
      recurseOverKeys((TDirectory *)obj);

    }  // end of IF a TDriectory
  }
}

void plot2Histograms(TH1 *htemp1, TH1 *htemp2, TString filename) {
  //TString title = htemp1->GetName();
  TString title = htemp1->GetTitle();

  // Make sure histograms exist
  if (!htemp2) {
    cout << "Histogram missing from 2nd file: " << htemp1->GetName() << endl;
    return;
  }

  // Scale by given factor.
  // If given factor is negative, area normalize
  if (scale1 > 0.0) {
    htemp1->Scale(scale1);
  } else {
    Double_t integral = htemp1->Integral();
    if (integral > 0.0)
      htemp1->Scale(1 / integral);
  }
  if (scale2 > 0.0) {
    htemp2->Scale(scale2);
  } else {
    Double_t integral = htemp2->Integral();
    if (integral > 0.0)
      htemp2->Scale(1 / integral);
  }

  // Set the histogram colors & lines
  htemp1->SetLineColor(kRed);
  htemp2->SetLineColor(kBlue);
  htemp1->SetLineWidth(1);
  htemp2->SetLineWidth(2);

  // Turn off stats
  if (!showStatsBoxes) {
    gStyle->SetOptStat(0);
  }

  // Create TStack but we will draw without stacking
  THStack *tempStack = new THStack();
  tempStack->Add(htemp1, "sames");
  tempStack->Add(htemp2, "sames");

  // Draw the histogram and titles
  tempStack->Draw("hist nostack");
  tempStack->SetTitle(title);
  tempStack->GetXaxis()->SetTitle(htemp1->GetXaxis()->GetTitle());

  // Draw the legend
  TLegend *infoBox = new TLegend(0.75, 0.83, 0.99, 0.99, "");
  infoBox->AddEntry(htemp1, label1, "L");
  infoBox->AddEntry(htemp2, label2, "L");
  infoBox->SetShadowColor(0);  // 0 = transparent
  infoBox->SetFillColor(kWhite);
  infoBox->Draw();

  // Place the stats boxes to be non-overlapping
  if (showStatsBoxes) {
    canvasDefault->SetRightMargin(0.2);
    canvasDefault->Update();
    TPaveStats *st1 = (TPaveStats *)htemp1->GetListOfFunctions()->FindObject("stats");
    TPaveStats *st2 = (TPaveStats *)htemp2->GetListOfFunctions()->FindObject("stats");
    st1->SetX1NDC(.79);
    st1->SetX2NDC(.99);
    st1->SetY1NDC(.6);
    st1->SetY2NDC(.8);
    st2->SetX1NDC(.79);
    st2->SetX2NDC(.99);
    st2->SetY1NDC(.38);
    st2->SetY2NDC(.58);
    canvasDefault->Modified();
  }

  // Set log y axis
  if (yAxisLogScale)
    canvasDefault->SetLogy(1);
  // Save the canvas
  canvasDefault->SaveAs(filename);
}
