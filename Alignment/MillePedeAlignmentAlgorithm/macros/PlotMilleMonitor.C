// Original Author: Gero Flucke
// last change    : $Date: 2009/01/20 20:21:38 $
// by             : $Author: flucke $

#include "PlotMilleMonitor.h"
#include <TH1.h>
// #include <TH2.h>
// #include <TProfile.h>
#include <TF1.h>
#include <TObjArray.h>

#include <TError.h>
#include <TFile.h>
#include <TROOT.h>
//#include <TCanvas.h>

//#include <iostream>

#include "GFUtils/GFHistManager.h"

////////////////////////////////////////////////////////////////////////////////////////////////////
PlotMilleMonitor::PlotMilleMonitor(const char *fileLegendList)
  : fHistManager(new GFHistManager) //, fFile(TFile::Open(fileName))
{
  //  if (!fFile) ::Error("PlotMilleMonitor", "Could not open file '%s'", fileName);

  if (!this->OpenFilesLegends(fileLegendList)) {
    ::Error("PlotMilleMonitor", "Problem opening files from '%s'", fileLegendList);
  }
  fHistManager->SameWithStats(true);
  fHistManager->SetLegendX1Y1X2Y2(0.14, 0.7, 0.45, 0.9);

}

////////////////////////////////////////////////////////////////////////////////////////////////////
PlotMilleMonitor::~PlotMilleMonitor()
{
  delete fHistManager;

  for (unsigned int i = 0; i < fFileLegends.size(); ++i) delete fFileLegends[i].first;
  //  delete fFile;
}

// ////////////////////////////////////////////////////////////////////////////////////////////////////
// void PlotMilleMonitor::DrawAll(Option_t *opt)
// {
  
//   const TString o(opt);
//   bool wasBatch = fHistManager->SetBatch();
//   fHistManager->Clear();
  
// //   if (o.Contains("d", TString::kIgnoreCase)) this->DrawParamDiff(true);
//   if (o.Contains("r", TString::kIgnoreCase)) this->DrawParamResult(true);
//   if (o.Contains("o", TString::kIgnoreCase)) this->DrawOrigParam(true);
//   if (o.Contains("g", TString::kIgnoreCase)) this->DrawGlobCorr(true);
//   if (o.Contains("p", TString::kIgnoreCase)) this->DrawPull("add");
//   if (o.Contains("m", TString::kIgnoreCase)) this->DrawMisVsLocation(true);
//   if (o.Contains("e", TString::kIgnoreCase)) this->DrawErrorVsHit(true);
//   if (o.Contains("h", TString::kIgnoreCase)) this->DrawHitMaps(true);
  
//   fHistManager->SetBatch(wasBatch);
//   if (!wasBatch) fHistManager->Draw();
//}

////////////////////////////////////////////////////////////////////////////////////////////////////
void PlotMilleMonitor::DrawAllByHit(const char *xy, Option_t * option)
{
  bool wasBatch = fHistManager->SetBatch();
  
  TString opt(option);
  this->DrawResidualsByHit("resid", xy, opt);
  opt += " add";
  this->DrawResidualsByHit("reduResid", xy, opt);
  this->DrawResidualsByHit("sigma", xy, opt);
  this->DrawResidualsByHit("angle", xy, opt);

  fHistManager->SetBatch(wasBatch);
  if (!wasBatch) fHistManager->Draw();
}

// ////////////////////////////////////////////////////////////////////////////////////////////////////
// void PlotMilleMonitor::DrawResidualsByHit(const char *histName, const char *xy, Option_t * option)
// {
//   const TString opt(option);
//   const Int_t layer = this->PrepareAdd(opt.Contains("add", TString::kIgnoreCase));
//   const bool addSumary = opt.Contains("sum", TString::kIgnoreCase);
//   const bool onlySummary = opt.Contains("sumonly", TString::kIgnoreCase);

//   const unsigned int nHitsMax = 29;
//   TH1 *hMean = 0, *hRms = 0;
//   if (addSumary) {
//     hMean = new TH1F(this->Unique(Form("%sMean", histName)),
//                      Form("mean of %s;N(hit)", histName), nHitsMax, 0, nHitsMax);
//     hRms  = new TH1F(this->Unique(Form("%sRms", histName)),
//                      Form("RMS of %s;N(hit)", histName, histName), nHitsMax, 0, nHitsMax);
//   }
//   for (unsigned int i = 0; i < nHitsMax; ++i) {
//     TString completeName(Form("residuals/%s/%s_%u", xy, histName, i));
//     TH1 *h = 0;
//     if (fFile) fFile->GetObject(completeName, h);
//     if (!h) {
//       ::Warning("PlotMilleMonitor::DrawResidualsByHit", "No hist '%s'!", completeName.Data());
//     } else {
//       if (addSumary) {
//         if (i == 0) {
//           hMean->SetYTitle(Form("<%s>", h->GetXaxis()->GetTitle()));
//           hRms->SetYTitle(Form("RMS(%s)", h->GetXaxis()->GetTitle()));
//         }
//         hMean->SetBinContent(i, h->GetMean());
//         hMean->SetBinError(i, h->GetMeanError());
//         hRms->SetBinContent(i, h->GetRMS());
//         hRms->SetBinError(i, h->GetRMSError());
//       }
//       if (onlySummary) {
//         delete h;
//       } else {
//         fHistManager->AddHist(h, layer);
//       }
//     }
//   }
//   fHistManager->AddHist(hMean, layer + (1 * !onlySummary));
//   fHistManager->AddHist(hRms,  layer + (1 * !onlySummary));
    
//   fHistManager->Draw();
//}

////////////////////////////////////////////////////////////////////////////////////////////////////
void PlotMilleMonitor::DrawResidualsByHit(const char *histName, const char *xy, Option_t *option)
{
  Int_t layer = this->PrepareAdd(TString(option).Contains("add", TString::kIgnoreCase));

  for (unsigned int iFile = 0; iFile < fFileLegends.size(); ++iFile) {
    this->AddResidualsByHit(histName, fFileLegends[iFile], layer, xy, option);
  }

  fHistManager->Draw();
}


////////////////////////////////////////////////////////////////////////////////////////////////////
Int_t PlotMilleMonitor::AddResidualsByHit(const char *histName,
                                          std::pair<TFile*,TString> &fileLeg, Int_t layer,
                                          const char *xy, Option_t *option)
{
  // options: - 'sum' to draw mean and RMS vs hit
  //          - 'sumonly' to skip single hit hists  
  // returns how many layers have been added

  const TString opt(option);
  const bool addSumary = opt.Contains("sum", TString::kIgnoreCase);
  const bool onlySummary = opt.Contains("sumonly", TString::kIgnoreCase);
  const bool norm = opt.Contains("norm", TString::kIgnoreCase);
  const bool gaus = opt.Contains("gaus", TString::kIgnoreCase);

  const unsigned int nHitsMax = 30;
  TH1 *hMean = 0, *hRms = 0; //, *hMeanGaus = 0, *hSigmaGaus = 0;
  if (addSumary) {
    hMean = new TH1F(this->Unique(Form("%sMean", histName)),
                     Form("mean of %s;N(hit)", histName), nHitsMax, 0, nHitsMax);
    hRms  = new TH1F(this->Unique(Form("%sRms", histName)),
                     Form("%s of %s;N(hit)", (gaus ? "#sigma" : "RMS"), histName),
		     nHitsMax, 0, nHitsMax);
  }
  for (unsigned int i = 0; i < nHitsMax; ++i) {
    const TString completeName(Form("residuals/%s/%s_%u", xy, histName, i));
    TH1 *h = 0;
    fileLeg.first->GetObject(completeName, h);
    if (!h) {
      ::Warning("PlotMilleMonitor::DrawResidualsByHit", "No hist '%s'!", completeName.Data());
    } else {
      if (gaus) {
	h->Fit("gaus", "Q0L"); // "0": do not directly draw, "Q": quiet, "L" likelihood for bin=0 treatment
	h->GetFunction("gaus")->ResetBit(TF1::kNotDraw);
      }
      if (addSumary) {
        if (i == 0) {
          hMean->SetYTitle(Form("<%s>", h->GetXaxis()->GetTitle()));
          hRms->SetYTitle(Form("%s(%s)", (gaus ? "#sigma" : "RMS"), h->GetXaxis()->GetTitle()));
        }
	if (gaus) {
	  TF1 *func = static_cast<TF1*>(h->GetFunction("gaus"));
	  hMean->SetBinContent(i+1, func->GetParameter(1));
	  hMean->SetBinError  (i+1, func->GetParError(1));
	  hRms-> SetBinContent(i+1, func->GetParameter(2));
	  hRms-> SetBinError  (i+1, func->GetParError(2));
	} else {
	  hMean->SetBinContent(i+1, h->GetMean());
	  hMean->SetBinError  (i+1, h->GetMeanError());
	  hRms-> SetBinContent(i+1, h->GetRMS());
	  hRms-> SetBinError  (i+1, h->GetRMSError());
	}
      }
      if (onlySummary) {
        delete h;
      } else {
        if (norm && h->GetEntries()) h->Scale(1./h->GetEntries());
        fHistManager->AddHistSame(h, layer, i, fileLeg.second);
      }
    }
  }
  if (hMean) fHistManager->AddHistSame(hMean, layer + (1 * !onlySummary), 0, fileLeg.second);
  if (hRms)  fHistManager->AddHistSame(hRms,  layer + (1 * !onlySummary), 1, fileLeg.second);

  return ((addSumary && onlySummary) || (!addSumary) ? 1 : 2);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
Int_t PlotMilleMonitor::PrepareAdd(bool addPlots)
{
  if (addPlots) {
    return fHistManager->GetNumLayers();
  } else {
    fHistManager->Clear();
    return 0;
  }
}

// ////////////////////////////////////////////////////////////////////////////////////////////////////
TString PlotMilleMonitor::Unique(const char *name) const
{
  if (!gROOT->FindObject(name)) return name;
  
  UInt_t i = 1;
  while (gROOT->FindObject(Form("%s_%u", name, i))) ++i;
  
  return Form("%s_%u", name, i);
}


//________________________________________________________
bool PlotMilleMonitor::OpenFilesLegends(const char *fileLegendList)
{
  bool allOk = true;
  
  TObjArray *fileLegPairs = TString(fileLegendList).Tokenize(",");
  for (Int_t iF = 0; iF < fileLegPairs->GetEntriesFast(); ++iF) {
    TObjArray *aFileLegPair = TString(fileLegPairs->At(iF)->GetName()).Tokenize("=");

    const char *legend = "";
    if (aFileLegPair->GetEntriesFast() >= 2) {
      if (aFileLegPair->GetEntriesFast() > 2) {
	::Error("TifResidualOverlay::OpenFilesLegends",
                "File-legend pair %s: %d (>2) '=' separated parts!",
		fileLegPairs->At(iF)->GetName(), aFileLegPair->GetEntriesFast());
      }
      legend = aFileLegPair->At(1)->GetName();
    } else if (aFileLegPair->GetEntriesFast() < 1) {
      continue; // empty: should report error?
    } // else if (aFileLegPair->GetEntriesFast() == 1): That's OK, use empty legend.

    TFile *file = TFile::Open(aFileLegPair->At(0)->GetName());
    if (!file) {
      allOk = false;
    } else {
      // std::pair<TFile*, const TString> fileLeg(file, legend);
      fFileLegends.push_back(std::make_pair(file, legend));
    }
    delete aFileLegPair;
  }
  delete fileLegPairs;

  return allOk;
}
