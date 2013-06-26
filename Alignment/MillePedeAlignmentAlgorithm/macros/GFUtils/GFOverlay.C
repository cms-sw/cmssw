//   Author:      Gero Flucke
//   Date:        October 2007
//   last update: $Date: 2013/04/12 07:06:16 $  
//   by:          $Author: flucke $

#include "GFOverlay.h"
#include <TError.h>
#include <TFile.h>
#include <TList.h>
#include <TObjArray.h>
#include <TString.h>
#include <TObjString.h>
#include <TKey.h>
#include <TH1.h>
#include "GFUtils/GFHistManager.h"

#include <iostream>
// for strlen:
#include <string.h>

//________________________________________________________
GFOverlay::GFOverlay(const char *fileLegendList, Option_t *option) :
  fHistMan(new GFHistManager), fLayer(0),
  fFiles(), fLegends(),
  fNormalise(TString(option).Contains("norm", TString::kIgnoreCase)),
  fSummaries(TString(option).Contains("sumperdir", TString::kIgnoreCase)),
  fDirNames(), fSkipDirNames(), fHistNames(), fSkipHistNames()
{
  if (fNormalise) ::Info("GFOverlay", "Normalising hists.");
  if (fSummaries) ::Info("GFOverlay", "Add summary hists for each directory.");

  fFiles.SetOwner();
  fLegends.SetOwner();

  const TString opt(option);

  // directory parts to require: either generally selecetd or only for dirs
  fDirNames = this->FindAllBetween(opt, "name(", ")");
  TObjArray tmp(this->FindAllBetween(opt, "nameDir(", ")"));
  fDirNames.AddAll(&tmp);
  fDirNames.SetOwner();
  for (Int_t iN = 0; iN < fDirNames.GetEntriesFast(); ++iN) {
    ::Info("GFOverlay", "Use only directories containing '%s'.", fDirNames[iN]->GetName());
  }
  // hist name parts to require: either generally selecetd or only for hists
  fHistNames = this->FindAllBetween(opt, "name(", ")");
  tmp = this->FindAllBetween(opt, "nameHist(", ")");
  fHistNames.AddAll(&tmp);
  fHistNames.SetOwner();
  for (Int_t iN = 0; iN < fHistNames.GetEntriesFast(); ++iN) {
    ::Info("GFOverlay", "Use only hists containing '%s'.", fHistNames[iN]->GetName());
  }

  // directory name parts to skip: either generally selecetd or only for dirs
  fSkipDirNames = this->FindAllBetween(opt, "skip(", ")");
  tmp = this->FindAllBetween(opt, "skipDir(", ")");
  fSkipDirNames.AddAll(&tmp);
  fSkipDirNames.SetOwner();
  for (Int_t iS = 0; iS < fSkipDirNames.GetEntriesFast(); ++iS) {
    ::Info("GFOverlay", "Skip directories containing '%s'.", fSkipDirNames[iS]->GetName());
  }
  // hist name parts to skip: either generally selecetd or only for hists
  fSkipHistNames = this->FindAllBetween(opt, "skip(", ")");
  tmp = this->FindAllBetween(opt, "skipHist(", ")");
  fSkipHistNames.AddAll(&tmp);
  fSkipHistNames.SetOwner();
  for (Int_t iS = 0; iS < fSkipHistNames.GetEntriesFast(); ++iS) {
    ::Info("GFOverlay", "Skip hists containing '%s'.", fSkipHistNames[iS]->GetName());
  }

  fHistMan->SameWithStats(true); // to draw both statistic boxes
  fHistMan->SetLegendX1Y1X2Y2(.14, .72, .42, .9); // defines (absolute) coordinates of legends

  this->OpenFilesLegends(fileLegendList);
  this->Overlay(fFiles, fLegends);

  fHistMan->SetNumHistsXY(3,2); // how many hists in x/y per canvas
  fHistMan->Draw();
}

//________________________________________________________
GFOverlay::~GFOverlay()
{
  delete fHistMan;
}

//________________________________________________________
TObjArray GFOverlay::FindAllBetween(const TString &text,
				    const char *startStr, const char *endStr) const
{
  TObjArray result; // TObjStrings...

  if (text.Contains(startStr, TString::kIgnoreCase)) {
    Ssiz_t start = text.Index(startStr);
    while (start != kNPOS && start < text.Length()) {
      TString name = this->FindNextBetween(text, start, startStr, endStr);
      if (!name.IsNull()) {
	result.Add(new TObjString(name));
	start = text.Index(startStr, start + name.Length() + TString(endStr).Length());
      } else {
	break;
      }
    }
  }

  return result;
}

//________________________________________________________
TString GFOverlay::FindNextBetween(const TString &input, Ssiz_t startInd,
				   const char *startStr, const char *endStr) const
{
  // search for startStr in input, starting at index startInd
  // if found, return what comes after that and before endStr, otherwise return empty string
  TString string(input);
  const Ssiz_t start = string.Index(startStr, startInd, TString::kIgnoreCase)
    + (startStr ? strlen(startStr) : 0);
  if (start != kNPOS) {
    const Ssiz_t end = string.Index(endStr, start);
    if (end == kNPOS) {
      ::Error("GFOverlay::FindNextBetween", "Miss closing '%s' after '%s'", endStr, startStr);
      string = "";
    } else {
      string = string(start, end - start);
    }
  } else {
    string = "";
  }

  return string;
}

//________________________________________________________
bool GFOverlay::OpenFilesLegends(const char *fileLegendList)
{
  bool allOk = true;
  
  TObjArray *fileLegPairs = TString(fileLegendList).Tokenize(",");
  for (Int_t iF = 0; iF < fileLegPairs->GetEntriesFast(); ++iF) {
    TObjArray *aFileLegPair = TString(fileLegPairs->At(iF)->GetName()).Tokenize("=");

    const char *legend = "";
    if (aFileLegPair->GetEntriesFast() >= 2) {
      if (aFileLegPair->GetEntriesFast() > 2) {
	::Error("GFOverlay::OpenFilesLegends", "File-legend pair %s: %d (>2) '=' separated parts!",
		fileLegPairs->At(iF)->GetName(), aFileLegPair->GetEntriesFast());
      }
      legend = aFileLegPair->At(1)->GetName();
    } else if (aFileLegPair->GetEntriesFast() < 1) {
      continue; // empty: should report error?
    } // else if (aFileLegPair->GetEntriesFast() == 1): That's OK, use empty legend.

    TFile *file = TFile::Open(aFileLegPair->At(0)->GetName());
    if (!file) {
      ::Error("GFOverlay::OpenFilesLegends", "Skip file-legend pair %s due to opening problems!",
	      fileLegPairs->At(iF)->GetName());
      allOk = false;
    } else {
      fFiles.Add(file);
      fLegends.Add(new TObjString(legend));
    }
    delete aFileLegPair;
  }

  delete fileLegPairs;
  return allOk;
}

//________________________________________________________
void GFOverlay::Overlay(const TObjArray &dirs, const TObjArray &legends)
{
  // 'directories' must contain TDirectory and inheriting, being parallel with legends,
  // method is called recursively
  TDirectory *dir1 = 0;
  for (Int_t iDir = 0; !dir1 && iDir < dirs.GetEntriesFast(); ++iDir) {
    dir1 = static_cast<TDirectory*>(dirs[iDir]);
  }
  if (!dir1) return;

  const Int_t currentLayer = fLayer;
  fLayer += (fSummaries ? 2 : 1);
  std::vector<TH1*> meanHists, rmsHists;

  // Now loop on keys of first directory (i.e. first file). If not deselected, foresee hists
  // for plotting and directories for recursion. Other objects are ignored. 
  UInt_t counter = 0;
  TIter nextKey(dir1->GetListOfKeys());
//   //  while(TKey* key = static_cast <TKey*> (nextKey())) {
//   // OK, make CINT happy, i.e. make .L GFOverlay.C work without extending '+':
  TKey* key = NULL; 
  while ((key = static_cast <TKey*> (nextKey()))) {
    // If key's name is deselected for both cases (hist and dir), avoid reading object:
    if (!fHistNames.IsEmpty()   && !this->KeyContainsListMember(key->GetName(), fHistNames)
	&& !fDirNames.IsEmpty() && !this->KeyContainsListMember(key->GetName(), fDirNames )) {
      continue; // type (dir/hist) independent skip: not actively selected!
    }
    if (this->KeyContainsListMember(key->GetName(), fSkipHistNames)
	&& this->KeyContainsListMember(key->GetName(), fSkipDirNames)) {
      continue; // type (dir/hist) independent skip: actively deselected!
    }
    // Now read object (only of first directory/file!) to be able to check type specific
    // skipping (hist or dir):
    TObject *obj = key->ReadObj();
    if (!obj) continue; // What ? How that?
    if (obj->InheritsFrom(TH1::Class())) {
      if (!fHistNames.IsEmpty() && !this->KeyContainsListMember(key->GetName(), fHistNames)) {
	continue;
      }
      if (this->KeyContainsListMember(key->GetName(), fSkipHistNames)) continue;
    } else if (obj->InheritsFrom(TDirectory::Class())) {
      if (!fDirNames.IsEmpty() && !this->KeyContainsListMember(key->GetName(), fDirNames)) {
	continue;
      }
      if (this->KeyContainsListMember(key->GetName(), fSkipDirNames)) continue;
    }
    // Now key survived! First treat for hist case:
    TObjArray hists(this->GetTypeWithNameFromDirs(TH1::Class(), key->GetName(), dirs));
    if (this->AddHistsAt(hists, legends, currentLayer, counter) > 0) {
      if (fSummaries) {
	this->CreateFillMeanRms(hists, currentLayer, dir1->GetName(), meanHists, rmsHists);
      }
      ++counter;
    }
    // Now treat directory case:
    TObjArray subDirs(this->GetTypeWithNameFromDirs(TDirectory::Class(), key->GetName(), dirs));
    if (subDirs.GetEntries()) { // NOT GetEntriesFast()!
      ::Info("GFOverlay::Overlay", "Key '%s' is directory to do recursion.", key->GetName());
      this->Overlay(subDirs, legends);
    }
  } // end of loop on keys
   
  // If mean/rms hists created, add them to manager:
  for (unsigned int iMean = 0; iMean < meanHists.size(); ++iMean) {
    fHistMan->AddHistSame(meanHists[iMean], currentLayer + 1, 0, legends[iMean]->GetName());
    fHistMan->AddHistSame(rmsHists[iMean], currentLayer + 1, 1, legends[iMean]->GetName());
  }
}

//________________________________________________________
bool GFOverlay::KeyContainsListMember(const TString &key, const TObjArray &list) const
{
  for (Int_t i = 0; i < list.GetEntriesFast(); ++i) {
    if (key.Contains(list[i]->GetName())) return true; 
  }

  return false;
}

//________________________________________________________
TObjArray GFOverlay::GetTypeWithNameFromDirs(const TClass *aType, const char *name,
					     const TObjArray &dirs) const
{
  // dirs must contain TDirectory only!
  // do Get(name) for all dirs and adds result to return value if type fits, otherwise Add(NULL)
  // result has length of dirs

  TObjArray result(dirs.GetEntriesFast()); // default length
  for (Int_t iDir = 0; iDir < dirs.GetEntriesFast(); ++iDir) {
    TDirectory *aDir = static_cast<TDirectory*>(dirs[iDir]);
    TObject *obj = (aDir ? aDir->Get(name) : 0);
    if (obj && !obj->InheritsFrom(aType)) {
      // delete obj; NO! deletes things found in previous calls...
      obj = 0;
    }
    result.Add(obj); // might be NULL
  }

  return result;
}

//________________________________________________________
Int_t GFOverlay::AddHistsAt(const TObjArray &hists, const TObjArray &legends, Int_t layer,Int_t pos)
{
  // hists and legends must have same length, but might have gaps...
  // return number of hists found and added

  Int_t nHists = 0;
  for (Int_t iHist = 0; iHist < hists.GetEntriesFast(); ++iHist) {
    TH1 *hist = static_cast<TH1*>(hists[iHist]);
    if (!hist) continue;

    if (fNormalise && hist->GetEntries()) {
      hist->Scale(1./hist->GetEntries());
    }

    fHistMan->AddHistSame(hist, layer, pos, (legends[iHist] ? legends[iHist]->GetName() : 0));
    ++nHists;
  }

  return nHists;
}

//________________________________________________________
void GFOverlay::CreateFillMeanRms(const TObjArray &hists, Int_t layer, const char *dirName,
				  std::vector<TH1*> &meanHists, std::vector<TH1*> &rmsHists) const
{
  // fill mean/rms from hists into the corresponding meanHists/rmsHists
  // if these are empty, create one hist for each slot of hists (even for empty ones!)
  if (hists.IsEmpty()) return;
  TH1 *h1 = 0;
  for (Int_t iH = 0; !h1 && iH < hists.GetEntriesFast(); ++iH) {
    h1 = static_cast<TH1*>(hists[iH]);
  }
  if (!h1 || h1->GetDimension() > 1) return; // only for 1D hists
  
  if (meanHists.empty()) { // create mean/RMS hists if not yet done
    const Float_t min = h1->GetXaxis()->GetXmin()/3.;
    const Float_t max = h1->GetXaxis()->GetXmax()/3.;
    const Int_t nBins = h1->GetNbinsX()/2;
    for (Int_t iHist = 0; iHist < hists.GetEntriesFast(); ++iHist) {
      TH1 *hMean = new TH1F(Form("mean%d_%d", layer, iHist), Form("%s: mean", dirName),
			    nBins, min, max);
      meanHists.push_back(hMean);
      TH1 *hRms = new TH1F(Form("rms%d_%d", layer, iHist), Form("%s: RMS", dirName),
			   nBins, 0., max);
      rmsHists.push_back(hRms);
    }
  }

  // now fill mean and rms hists
  for (Int_t iHist = 0; iHist < hists.GetEntriesFast(); ++iHist) {
    TH1 *h = static_cast<TH1*>(hists[iHist]);
    if (!h) continue;
    meanHists[iHist]->Fill(h->GetMean());
    rmsHists[iHist]->Fill(h->GetRMS());
  }
}
