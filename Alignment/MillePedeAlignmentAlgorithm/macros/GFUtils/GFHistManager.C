// Author: Gero Flucke <mailto:flucke@mail.desy.de>
//____________________________________
// GFHistManager
//   Author:      Gero Flucke
//   Date:        Feb. 10th, 2002
//   last update: $Date: 2012/12/07 10:07:24 $  
//   by:          $Author: flucke $
//

#include <string.h>

// #include <iostream>
// #include <vector>
// RooT header:
#include <TROOT.h>
#include <TError.h>
#include <TH1.h>
#include <TH2.h>
#include <TF1.h>
#include <TCanvas.h>
#include <TFile.h>
#include <TString.h>
#include <TObjArray.h>
#include <TList.h>
#include <TLegend.h>
#include <TLegendEntry.h>
#include <TPaveStats.h>
#include <TStyle.h>

// my header:
#include "GFHistManager.h"
#include "GFUtils/GFHistArray.h"

ClassImp(GFHistManager)

const Int_t GFHistManager::kDefaultPadsPerCanX = 4;
const Int_t GFHistManager::kDefaultPadsPerCanY = 3;
const Int_t GFHistManager::kDefaultDepth = 0;
TString GFHistManager::fgLegendEntryOption = "l";


GFHistManager::GFHistManager()
{
  this->Initialise();
  fBatch = kFALSE;
  fDrawDiffStyle = kTRUE;
  fSameWithStats = kFALSE;
  fLegendY1 = 0.75;
  fLegendX1 = 0.7;
  fLegendY2 = 0.99;
  fLegendX2 = 0.99;
  fStatsX1 = 0.72, fStatsX2 = 0.995, fStatsY1 = .8, fStatsY2 = .995;
  fCanvasWidth = 600;
  fCanvasHeight = 600;
}

GFHistManager::GFHistManager(TH1* hist)
{
  // constructing with hist as first histogram 
  this->Initialise();
  
  this->AddHist(hist);
  fBatch = kFALSE;
  fDrawDiffStyle = kTRUE;
  fSameWithStats = kFALSE;
  fLegendY1 = 0.75;
  fLegendX1 = 0.7;
  fLegendY2 = 0.99;
  fLegendX2 = 0.99;
  fStatsX1 = 0.72, fStatsX2 = 0.995, fStatsY1 = .8, fStatsY2 = .995;
  fCanvasWidth = 600;
  fCanvasHeight = 600;
}

GFHistManager::GFHistManager(TCollection* hists)
{
  // constructing with histos in 'hists' as first histogram 
  this->Initialise();

  this->AddHists(hists);
  fBatch = kFALSE;
  fDrawDiffStyle = kTRUE;
  fSameWithStats = kFALSE;
  fLegendY1 = 0.75;
  fLegendX1 = 0.7;
  fLegendY2 = 0.99;
  fLegendX2 = 0.99;
  fStatsX1 = 0.72, fStatsX2 = 0.995, fStatsY1 = .8, fStatsY2 = .995;
  fCanvasWidth = 600;
  fCanvasHeight = 600;
}

//________________________________________________________
void GFHistManager::Initialise()
{
  fDepth = kDefaultDepth;
  fNoX.Set(fDepth);
  fNoY.Set(fDepth);
#if ROOT_VERSION_CODE >= ROOT_VERSION(3,10,2) && ROOT_VERSION_CODE <= ROOT_VERSION(4,0,3)
  // TArrayI::Reset(Int_t) buggy in 3.10_02:
  for(Int_t i = 0; i < fNoX.GetSize(); ++i) fNoX[i] = kDefaultPadsPerCanX;
  for(Int_t i = 0; i < fNoY.GetSize(); ++i) fNoY[i] = kDefaultPadsPerCanY;
#else
  fNoX.Reset(kDefaultPadsPerCanX);
  fNoY.Reset(kDefaultPadsPerCanY);
#endif  
  fLogY.Set(fDepth);
  fLogY.Reset(); // set to 0

//   fCanvasWidth = 700;
//   fCanvasHeight = 500;
//   fCanvasWidth = 600;
//   fCanvasHeight = 600;

  fHistArrays = new TObjArray;
  for(Int_t i = 0; i < fDepth; ++i){
    fHistArrays->Add(new TObjArray);
  }
  fCanArrays = new TObjArray;
  fLegendArrays = NULL;
  fObjLists = NULL;
}

//________________________________________________________
GFHistManager::~GFHistManager()
{
  // destructor: delete stored arrays, legends and canvases, but not the hists and objects!
  if(fHistArrays) {
    TIter iter(fHistArrays);
    while(TObjArray* array = static_cast<TObjArray*>(iter.Next())){
      array->Delete();
    }
    fHistArrays->Delete();
    delete fHistArrays;
  }
  if(fLegendArrays) {
    TIter legIter(fLegendArrays);
    while(TObjArray* array = static_cast<TObjArray*>(legIter.Next())){
      array->Delete();
    }
    fLegendArrays->Delete();
    delete fLegendArrays;
  }

  if(fObjLists) { 
    TIter listArrayIter(fObjLists);
    while(TObjArray* listArray = static_cast<TObjArray*>(listArrayIter.Next())){
      listArray->Delete();; // delete lists, but not its objects
    }
    fObjLists->Delete(); // delete arrays of lists
    delete fObjLists;// delete array of arrays of lists
  }

  if(fCanArrays) {
    TIter canIter(fCanArrays);
    while(TObjArray* array = static_cast<TObjArray*>(canIter.Next())){
      array->Delete();
    }
    fCanArrays->Delete();
    delete fCanArrays;
  }
}

//________________________________________________________
void GFHistManager::Clear(Bool_t deleteHists)
{
  // delete all canvases and clear the lists of hists stored in the manager 
  // (hists and objects are deleted if deleteHists is true [by default it is false])

  TIter iterCanArrays(fCanArrays);
  while(TObjArray* arr = static_cast<TObjArray*>(iterCanArrays.Next())){
    // A simple 'arr->Delete();' causes a crash if the user has closed a canvas
    // via the GUI - so delete only those that are known to gROOT:
    TIter cans(arr);
    while (TObject *c = cans.Next()) delete gROOT->GetListOfCanvases()->FindObject(c);
  }
  fCanArrays->Delete(); // delete arrays of canvases
  delete fCanArrays;

  if(fLegendArrays){ // by default there are no legends...
    TIter iterLegArrays(fLegendArrays);
    while(TObjArray* arr = static_cast<TObjArray*>(iterLegArrays.Next())){
      arr->Delete(); // delete legends
    }
    fLegendArrays->Delete(); // delete arrays of legends
    delete fLegendArrays;
  }

  if(fObjLists) { 
    TIter listArrayIter(fObjLists);
    while(TObjArray* listArray = static_cast<TObjArray*>(listArrayIter.Next())){
      if(deleteHists) {
	TIter listIter(listArray);
	while(TList* list = static_cast<TList*>(listIter.Next())){
	  list->Delete(); // delete objects if requested
	}
      }
      listArray->Delete(); // delete lists
    }
    fObjLists->Delete(); // delete arrays of lists
    delete fObjLists; // delete array of arrays of lists
  }

  TIter iterHistArrays(fHistArrays);
  while(TObjArray* arr = static_cast<TObjArray*>(iterHistArrays.Next())){
    TIter iterHistArrays2(arr);
    while(TObjArray* arr2 = static_cast<TObjArray*>(iterHistArrays2.Next())){
      if(deleteHists) arr2->Delete(); // delete histograms
      else            arr2->Clear();
    }
    arr->Delete();
  }
  fHistArrays->Delete(); // delete arrays of arrays of histograms
  delete fHistArrays;

  this->Initialise(); // here the arrays are rebuild and fDepth etc. adjusted
}

//________________________________________________________
void GFHistManager::Draw(Option_t *)
{
  // draw all layers of histograms (ignore if in batch mode)
  if(fBatch) return; // for speed up...
  for(Int_t i = 0; i < fDepth; ++i){
    this->Draw(i);
  }
}

//________________________________________________________
void GFHistManager::Draw(Int_t layer)
{
  if(fBatch) return;
  this->DrawReally(layer);
}


//________________________________________________________
void GFHistManager::DrawReally(Int_t layer)
{
  if(layer < 0 || layer > fDepth-1) {
    this->Warning("DrawReally","Layer %d does not exist, possible are 0 to %d.", 
		  layer, fDepth-1);
    return;
  }

  this->MakeCanvases(layer);

  TIter canIter(static_cast<TObjArray*>(fCanArrays->At(layer)));
  TIter histIter(static_cast<TObjArray*>(fHistArrays->At(layer)));

  Int_t histNo = 0; // becomes number of histograms in layer
  while(TCanvas* can = static_cast<TCanvas*>(canIter.Next())){
    Int_t nPads = this->NumberOfSubPadsOf(can);
    if (fNoX[layer] * fNoY[layer] != nPads && 
	!(nPads == 0 && fNoX[layer] * fNoY[layer] == 1)) {
      this->Warning("DrawReally", "inconsistent number of pads %d, expect %d*%d",
		    nPads, fNoX[layer], fNoY[layer]);
    }
    for(Int_t i = 0; i <= nPads; ++i){
      if (i == 0 && nPads != 0) i = 1;
      can->cd(i);
      if(GFHistArray* histsOfPad = static_cast<GFHistArray*>(histIter.Next())){
	TIter hists(histsOfPad);
	TH1* firstHist = static_cast<TH1*>(hists.Next());
	firstHist->Draw();
	this->DrawFuncs(firstHist);
	while(TH1* h = static_cast<TH1*>(hists.Next())){
	  h->Draw(Form("SAME%s%s", (fSameWithStats ? "S" : ""), h->GetOption()));
	  this->DrawFuncs(h);
	}
	if(histsOfPad->GetEntriesFast() > 1){
	  const Double_t max = this->MaxOfHists(histsOfPad);
	  if(//firstHist->GetMaximumStored() != -1111. &&  ????
	     //max > firstHist->GetMaximumStored()){
	     max > firstHist->GetMaximum()){
	    firstHist->SetMaximum((fLogY[layer] ? 1.1 : 1.05) * max);
	  }
	  const Double_t min = this->MinOfHists(histsOfPad);
	  if (min < 0.) {
	    firstHist->SetMinimum(min * 1.05);
	  } else if (gStyle->GetHistMinimumZero()) {
	    // nothing to do
	  } else if (min != 0. || !fLogY[layer]) {
	    // Do not set to zero: log scale issue!
	    firstHist->SetMinimum(min * 0.95);
	  }
	}
	if(fLogY[layer] 
	   && (firstHist->GetMinimum() > 0. 
	      || (firstHist->GetMinimum() == 0. 
		  && firstHist->GetMinimumStored() == -1111.)))gPad->SetLogy();
	// draw other objects:
	this->DrawObjects(layer, histNo);
	// make hist style differ
	if(fDrawDiffStyle) GFHistManager::MakeDifferentStyle(histsOfPad);
	// draw legends on top of all
	if(fLegendArrays && layer <= fLegendArrays->GetLast() && fLegendArrays->At(layer)){
	  this->DrawLegend(layer, histNo);
	}
	gPad->Modified();
	this->ColourFuncs(histsOfPad);
	if (fSameWithStats) {
	  gPad->Update(); // not nice to need this here, makes use over network impossible...
	  this->ColourStatsBoxes(histsOfPad);
	}
	histNo++;
      }
    } // loop over pads
  } // loop over canvases
}

//________________________________________________________
void GFHistManager::DrawLegend(Int_t layer, Int_t histNo)
{
  // histNo starting at '0'
  // We must already be in the correct pad, layer and histNo must exist

  if(fLegendArrays && layer <= fLegendArrays->GetLast() && fLegendArrays->At(layer)){
    TObjArray* legends = static_cast<TObjArray*>(fLegendArrays->At(layer));
    TObject* legend = (histNo <= legends->GetLast() ? legends->At(histNo) : NULL);
    if(legend) legend->Draw();
  }
}

//________________________________________________________
void GFHistManager::DrawObjects(Int_t layer, Int_t histNo)
{
  // histNo starting at '0'
  // We must already be in the correct pad, layer and histNo must exist
  if(fObjLists && layer <= fObjLists->GetLast() && fObjLists->At(layer)){
    TObjArray* layerLists = static_cast<TObjArray*>(fObjLists->At(layer));
    if(histNo <= layerLists->GetLast() && layerLists->At(histNo)){
      TObjLink *lnk = static_cast<TList*>(layerLists->At(histNo))->FirstLink();
      while (lnk) {
	lnk->GetObject()->Draw(lnk->GetOption());
	lnk = lnk->Next();
      }
    }
  }
}

//________________________________________________________
void GFHistManager::Print(const char* filename, Bool_t add)
{
  // print all layers of histograms to ps-file 'filename'
  // if 'add == true' puts '(' or ')' only if 'filename' ends with it, 
  // e.g. if i is loop variable 
  //   GFHistManager *man = ...;
  //   TString name("XXX.ps");
  //   if(i == 0) man->Print(name + '(');
  //   else if(i == last) man->Print(name + ')');
  //   else man->Print(name, kTRUE);


  const Bool_t rootIsBatch = gROOT->IsBatch();
  if(fBatch){
    gROOT->SetBatch();
    for(Int_t i = 0; i < fDepth; ++i){
      this->DrawReally(i);
    }
  }
  gROOT->SetBatch(rootIsBatch);

  TObjArray cans;
  TIter canArrayIter(fCanArrays);
  while(TObjArray* canArray = static_cast<TObjArray*>(canArrayIter.Next())){
    cans.AddAll(canArray);
  }

  const Int_t nCans = cans.GetEntriesFast();
  if(nCans == 1) {
    cans.At(0)->Print(filename);
    return;
  }

  TString plainName(filename);
  const Bool_t starting = plainName.EndsWith("(");
  if(starting) {
    const Ssiz_t ind = plainName.Last('(');
    plainName.Remove(ind);
    //    plainName.ReplaceAll("(", "");
  }
  const Bool_t ending = plainName.EndsWith(")");
  if(ending) {
    const Ssiz_t ind = plainName.Last(')');
    plainName.Remove(ind);
//     plainName.ReplaceAll(")", "");
  }

  for(Int_t i = 0; i < nCans; ++i){
    if(i == 0 && !ending && (!add || starting)) {
      cans.At(i)->Print(plainName + "(");
    } else if(i == nCans - 1 && !starting && (!add || ending)) {
      cans.At(i)->Print(plainName + ")");
    } else {
      cans.At(i)->Print(plainName);
    }
  }
}

// //________________________________________________________
// void GFHistManager::Print(const char* filename, Int_t layer)
// {
//   // print all canvases of layer into .ps-files with name of their first histogram
//   Bool_t rootIsBatch = gROOT->IsBatch();
//   gROOT->SetBatch();
//   this->DrawReally(layer);

//   TIter canIter(static_cast<TObjArray*>(fCanArrays->At(layer)));
//   Int_t nCan = 0;
//   while(TCanvas* can = static_cast<TCanvas*>(canIter.Next())){
//     TObjArray* histLayer = static_cast<TObjArray*>(fHistArrays->At(layer));
//     TObjArray* firstHistInCan =
//       static_cast<TObjArray*>(histLayer->At(nCan*fNoX[layer]*fNoY[layer]));
//     ++nCan;
//     if(firstHistInCan){
//       TString psFile("/tmp/");
//       psFile+=firstHistInCan->First()->GetName();
//       can->Print(psFile+=".ps");
//     }
//   } // loop over canvases
//   gROOT->SetBatch(rootIsBatch);
// }


//________________________________________________________
void GFHistManager::Update()
{
  // call Update() to all canvases
  for(Int_t i = 0; i < fDepth; ++i){
    this->Update(i);
  }
}

//________________________________________________________
void GFHistManager::Update(Int_t layer)
{
  if(!this->CheckDepth("Update", layer, kFALSE)) {
    return;
  }

  // First loop on canvases:
  // If meanwhile the setting of fNoX/fNoY has changed, we are better with
  // drawing from scratch:
  Bool_t drawFromScratch = kFALSE;
  TIter canIter(static_cast<TObjArray*>(fCanArrays->At(layer)));
  while(TCanvas* can = static_cast<TCanvas*>(canIter.Next())){
    const Int_t nPads = this->NumberOfSubPadsOf(can);
    if (fNoX[layer] * fNoY[layer] != nPads && // does not fit...
	!(nPads == 0 && fNoX[layer] * fNoY[layer] == 1)) {// ...nor single hist canvas
      drawFromScratch = kTRUE;
      break;
    }
  }
  if (drawFromScratch) {
    this->Draw(layer);
    return; // nothing else to be done...
  }

  // Now second loop doing the real Update work:
  canIter = static_cast<TObjArray*>(fCanArrays->At(layer));
  Int_t numPreviousCansHists = 0;
  const Int_t numHistsLayer = this->GetNumHistsOf(layer);
  while(TCanvas* can = static_cast<TCanvas*>(canIter.Next())){
    const Int_t nPads = this->NumberOfSubPadsOf(can); // get numbers of first loop?
    for(Int_t i = 0; i <= nPads; ++i){
      if (i == 0 && nPads != 0) i = 1;// i==0: single hist canvas, else step into pad
      can->cd(i);

      const Int_t histNo = TMath::Max(0, numPreviousCansHists + i - 1);// for nPad == 0
      if (histNo >= numHistsLayer) continue;
      // draw other objects
      this->DrawObjects(layer, histNo);
      // draw legends on top of all
      if(fLegendArrays && fLegendArrays->GetSize() > layer && fLegendArrays->At(layer)){
	this->DrawLegend(layer, histNo);
      }

      if(fLogY[layer]) {
	GFHistArray *histsOfPad = this->GetHistsOf(layer, histNo);
	TH1 *h1 = histsOfPad->First();
	if (h1->GetMinimumStored() == 0. && histsOfPad->GetEntriesFast() > 1
	    && this->MinOfHists(histsOfPad) == 0.) {
	  // trouble with log scale, but assume that 0. set in DrawReally(..)!
	  h1->SetMinimum(-1111.);
	}
	if ((h1->GetMinimum() > 0. 
	     || (h1->GetMinimum() == 0. && h1->GetMinimumStored() == -1111.))) {
	  gPad->SetLogy();
	} else {
	  gPad->SetLogy(kFALSE);
	}
      } else {
	gPad->SetLogy(kFALSE);
      }
      gPad->Modified();
//       gPad->Update();
//       gPad->Modified();
//       gPad->Update();
    }
//     can->Update();
    can->Modified();
    can->Update();
//     can->Modified();
    numPreviousCansHists += nPads;
  } // loop over canvases
}


//_____________________________________________________
TLegendEntry* GFHistManager::AddHist(TH1* hist, Int_t layer, const char* legendTitle,
				     const char* legOpt)
{
  // add hist to 'layer'th list  of histos (expands, if layer does not already exist!)
  if(!hist){
    this->Warning("AddHist", "adding NULL pointer will be ignored!");
    return NULL;
  }

  if(!this->CheckDepth("AddHist", layer)) return NULL;
  GFHistArray* newHist = new GFHistArray;
  newHist->Add(hist);
  TObjArray* layerHistArrays = static_cast<TObjArray*>(fHistArrays->At(layer));
  layerHistArrays->Add(newHist);
  if(legendTitle){
    TObjArray* legends = this->MakeLegends(layer);
    TLegend* legend = new TLegend(fLegendX1, fLegendY1, fLegendX2, fLegendY2);
#if ROOT_VERSION_CODE < ROOT_VERSION(5,6,0)
    if (TString(gStyle->GetName()) == "Plain") legend->SetBorderSize(1);
#endif
    legends->AddAtAndExpand(legend, layerHistArrays->IndexOf(newHist));
    return legend->AddEntry(hist, legendTitle, legOpt ? legOpt : fgLegendEntryOption.Data());
  }
  return NULL;
}

//_____________________________________________________
void GFHistManager::AddHists(TCollection* hists, Int_t layer, const char* legendTitle,
			     const char* legOpt)
{
  // add contents of 'hists' to 'layer'th list of histos (should be histograms!!!)
  TIter iter(hists);
  while(TObject* hist = iter.Next()){
    if(!hist->InheritsFrom(TH1::Class())){
      this->Warning("AddHists", "Trying to add a non-histogram object, ignore!");
    } else this->AddHist(static_cast<TH1*>(hist), layer, legendTitle, legOpt);
  }
}

//_____________________________________________________
TLegendEntry* GFHistManager::AddHistSame(TH1* hist, Int_t layer, Int_t histNum,
					 const char* legendTitle, const char* legOpt)
{
  // adds hist to layer to draw it in the same pad as histNum's histo of that layer 
  if(!hist){
    this->Warning("AddHistSame", "adding NULL pointer will be ignored!");
    return NULL;
  }
  if (histNum > 0 && this->CheckDepth("AddHistSame", layer, kTRUE) //maybe added layer?
      && !this->GetHistsOf(layer, histNum-1)) {
    this->Error("AddHistSame", "usage as AddHist only for next free histNum, not %d", histNum);
    return NULL;
  }
  GFHistArray *histsArray = this->GetHistsOf(layer, histNum, kTRUE);// expand!
  TLegendEntry* result = NULL;
  if(histsArray) {
    histsArray->Add(hist); 
    if(legendTitle && strlen(legendTitle)){
      TObjArray* legends = this->MakeLegends(layer);
      TLegend* legend = NULL;
      if(legends->GetLast() >= histNum
	 && legends->At(histNum)){
	legend = static_cast<TLegend*>(legends->At(histNum));
      } else {
	legend = new TLegend(fLegendX1, fLegendY1, fLegendX2, fLegendY2);
#if ROOT_VERSION_CODE < ROOT_VERSION(5,6,0)
	if (TString(gStyle->GetName()) == "Plain") legend->SetBorderSize(1);
#endif
	legends->AddAtAndExpand(legend, histNum);
      }
      result = legend->AddEntry(hist,legendTitle, legOpt ? legOpt : fgLegendEntryOption.Data());
    }
  }
  return result;
}

//_____________________________________________________
void GFHistManager::AddHistsSame(GFHistArray* hists, Int_t layer,
				 const char* legendTitle, const char* legOpt)
{
  // adds hists to layer
  // each hist is AddHistSame(hist, layer, pad = 0...n)
  if(!hists){
    this->Warning("AddHistsSame", "adding NULL pointer will be ignored!");
    return;
  }
  for(Int_t i = 0; i < hists->GetEntriesFast(); ++i){
    this->AddHistSame(hists->At(i), layer, i, legendTitle, legOpt);
  }
}

//_____________________________________________________
void GFHistManager::AddHistsSame(GFHistArray* hists, Int_t layer, Int_t histNum)
{
  // adds hists to layer to draw it in the same pad as histNum's histo of that layer
  if(!hists){
    this->Warning("AddHistsSame", "adding NULL pointer will be ignored!");
    return;
  }
  GFHistArray* histsArray = this->GetHistsOf(layer, histNum, kTRUE);
  if(histsArray) histsArray->AddAll(hists);
}

//_____________________________________________________
void GFHistManager::AddLayers(GFHistManager* other)
{
  // append the layers from other to this, hists are not cloned, but legends?
  if(!other) return;
  const Int_t oldDepth = fDepth;
  for(Int_t iLayer = 0; iLayer < other->GetNumLayers(); ++iLayer){
    for(Int_t iPad = 0; iPad < other->GetNumHistsOf(iLayer); ++iPad){
      GFHistArray* hists = other->GetHistsOf(iLayer, iPad);
      this->AddHistsSame(hists, oldDepth + iLayer, iPad);
      TLegend* leg = other->GetLegendOf(iLayer, iPad);
      if(leg) this->AddLegend(static_cast<TLegend*>(leg->Clone()), iLayer, iPad);
    }
  }
}

//_____________________________________________________
void GFHistManager::AddLayer(GFHistManager* other, Int_t layer)
{
  // append the layer 'layer' from other to this, hists are not cloned, but legends?
  if(!other || layer >= other->GetNumLayers()) return;

  const Int_t newLayer = fDepth;
  for(Int_t iPad = 0; iPad < other->GetNumHistsOf(layer); ++iPad){
    GFHistArray* hists = other->GetHistsOf(layer, iPad);
    this->AddHist(hists->At(0), newLayer);
    for(Int_t iHist = 1; iHist < hists->GetEntriesFast(); ++iHist){
      this->AddHistSame(hists->At(iHist), newLayer, iPad);
    }
    TLegend* leg = other->GetLegendOf(layer, iPad);
    if(leg) this->AddLegend(static_cast<TLegend*>(leg->Clone()), newLayer, iPad);
  }
}

//_____________________________________________________
void GFHistManager::Overlay(GFHistManager* other, Int_t otherLayer, Int_t myLayer,
			    const char* legendTitle)
{
  if (!other || otherLayer >= other->GetNumLayers()
      || myLayer >= other->GetNumLayers()
      || other->GetNumHistsOf(otherLayer) != this->GetNumHistsOf(myLayer)) return;

  const Int_t histNo = 0;
  for (Int_t iPad = 0; iPad < other->GetNumHistsOf(otherLayer); ++iPad) {
    GFHistArray* hists = other->GetHistsOf(otherLayer, iPad);
    this->AddHistSame(hists->At(histNo), myLayer, iPad, legendTitle);//, legOpt)
  }
}

//_____________________________________________________
TLegend* GFHistManager::AddLegend(Int_t layer, Int_t histoNum, 
				  const char* header, Bool_t referAll)
{
  // adds a legend referencing all hists in same pad 'histoNum' of layer
  // 

  // FIXME: use help of other AddLegend method?
  if(!this->CheckHistNum("AddLegend", layer, histoNum)) return NULL;

  TObjArray* legendsOfLayer = this->MakeLegends(layer);
  TLegend* legend = (legendsOfLayer->GetSize() <= histoNum ? 
		     NULL : static_cast<TLegend*>(legendsOfLayer->At(histoNum)));
  if(!legend) {
    legend = new TLegend(fLegendX1, fLegendY1, fLegendX2, fLegendY2);
#if ROOT_VERSION_CODE < ROOT_VERSION(5,6,0)
    if (TString(gStyle->GetName()) == "Plain") legend->SetBorderSize(1);
#endif
    legendsOfLayer->AddAtAndExpand(legend, histoNum);
  }

  if(header) legend->SetHeader(header);
  GFHistArray* hists = this->GetHistsOf(layer, histoNum);
  TList* legendEntries = legend->GetListOfPrimitives();

  if(referAll){
    TIter histsIter(hists);
    while(TObject* hist = histsIter.Next()){
      Bool_t addNew = kTRUE;
      TIter legEntrIter(legendEntries);
      while(TLegendEntry* entry = static_cast<TLegendEntry*>(legEntrIter())){
	if(hist == entry->GetObject()) {addNew = kFALSE; break;}
      }  
      if(addNew) legend->AddEntry(hist, hist->GetName(), fgLegendEntryOption);
    }
  }

  if(layer < fCanArrays->GetEntriesFast()) {
    this->Update(layer); // if canvas already drawn
  }
  return legend;
}

//_____________________________________________________
Bool_t GFHistManager::RemoveLegend(Int_t layer, Int_t nPad)
{
  // true if there was a legend
  if(!this->CheckHistNum("RemoveLegend", layer, nPad)) return kFALSE;

  TLegend* leg = this->GetLegendOf(layer, nPad);
  if(!leg) return kFALSE;

  TObjArray* legendsOfLayer = this->MakeLegends(layer);
  if(!legendsOfLayer->Remove(leg)) {
    this->Error("RemoveLegend", "inconsistent state for layer %d, nPad %d", layer, nPad);
    return kFALSE;
  }
  delete leg;

  if(layer < fCanArrays->GetEntriesFast()) {
    this->Update(layer); // if canvas already drawn
  }

  return kTRUE;
}

//_____________________________________________________
void GFHistManager::AddLegend(TLegend* leg, Int_t layer, Int_t histoNum)
{
  // hist and layer must already exist
  if(!this->CheckHistNum("AddLegend", layer, histoNum)) return;

  TObjArray* legendsOfLayer = this->MakeLegends(layer);
  TLegend* legend = (legendsOfLayer->GetSize() < histoNum ? 
		     NULL : static_cast<TLegend*>(legendsOfLayer->At(histoNum)));
  if(legend) {
    this->Error("AddLegend", "legend exists, replacing it");
    delete legend;
  }
  legend = leg;
  legendsOfLayer->AddAtAndExpand(legend, histoNum);

  if(layer < fCanArrays->GetEntriesFast()) {
    this->Update(layer); // if canvas already drawn
  }
}


//_____________________________________________________
void GFHistManager::AddObject(TObject* obj, Int_t layer, Int_t histoNum, Option_t* opt)
{
  // Hist and layer must already exist.
  // If the given pad is already drawn, it will get updated to display the object.
  // If you add many objects, this can become pretty slow, so it is recommended
  // to first work in batch mode (SetBatch()), add all hists and objects and then
  // go back and draw: SetBatch(false); Draw();// or Draw(layer)
  if(!this->CheckHistNum("AddObject", layer, histoNum)) return;

  TList* objList = this->MakeObjList(layer, histoNum);
  objList->Add(obj, opt);

  if(layer < fCanArrays->GetEntriesFast()) {
    // Would be nice to update only for histoNum to speed up...
    this->Update(layer); // if canvas already drawn
  }
}


//_____________________________________________________
void GFHistManager::WriteCanvases(TFile* file)
{
  // write canvases with their content to file  (overwrite)
  if(!file) {
    this->Warning("WriteCanvases", "no file given, ignore!");
    return;
  }

  TDirectory* saveDir = gDirectory;
  file->cd();
  for(Int_t i = 0; i < fDepth; ++i){
    TIter canvases(static_cast<TObjArray*>(fCanArrays->At(i)));
    while(TCanvas* can = static_cast<TCanvas*>(canvases.Next())){
      can->Write(0, kOverwrite);
    }
  }
  saveDir->cd();
}

//_____________________________________________________
void GFHistManager::WriteHistos(TFile* file)
{
  // write histos to file (overwrite)

  if(!file) {
    this->Warning("WriteHistos", "no file given, ignore!");
    return;
  }

  TDirectory* saveDir = gDirectory;
  file->cd();
  for(Int_t i = 0; i < fDepth; ++i){
    TIter iterHistsArr(static_cast<TObjArray*>(fHistArrays->At(i)));
    while(TObjArray* arr2 = static_cast<TObjArray*>(iterHistsArr.Next())){
      TIter iterHistArr2(arr2);
      while(TObject* hist = iterHistArr2.Next()){
	hist->Write(0, kOverwrite);
      }
    }
  }
  saveDir->cd();
}

//_____________________________________________________
void GFHistManager::MakeCanvases(Int_t layer)
{
  // no check done whether layer is consistent with depth...

  Int_t nHists = static_cast<TObjArray*>(fHistArrays->At(layer))->GetEntriesFast();
  Int_t nCanvases = nHists / (fNoX[layer] * fNoY[layer]);
  if(nHists > nCanvases * fNoX[layer] * fNoY[layer]){
    ++nCanvases;
  }

  Bool_t oneCanvas = kFALSE;
  while(nHists < fNoX[layer] * fNoY[layer]){
    oneCanvas = kTRUE;
//     fNoX[layer] > 1 ? --(fNoX[layer]) : --(fNoY[layer]);
    (fNoX[layer] > 1 && fNoX[layer] >= fNoY[layer]) ? --(fNoX[layer]) : --(fNoY[layer]);
//  if(nHists < fNoX[layer] * fNoY[layer]) fNoY[layer] > 1 ? --(fNoY[layer]) : --(fNoX[layer]);
    if(nHists < fNoX[layer] * fNoY[layer])
      (fNoY[layer] > 1 && fNoY[layer] >= fNoX[layer]) ? --(fNoY[layer]) : --(fNoX[layer]);
  }
//   if(oneCanvas && nHists > fNoX[layer] * fNoY[layer]) ++(fNoX[layer]);
//   if(oneCanvas && nHists > fNoX[layer] * fNoY[layer]) 
//     (fNoX[layer] > fNoY[layer]) ? ++(fNoY[layer]) : ++(fNoX[layer]);
  while(oneCanvas && nHists > fNoX[layer] * fNoY[layer]){
    (fNoX[layer] > fNoY[layer]) ? ++(fNoY[layer]) : ++(fNoX[layer]);
  }

  if(fCanArrays->GetSize() > layer && fCanArrays->At(layer)){
    static_cast<TObjArray*>(fCanArrays->At(layer))->Delete();
  } else {
    fCanArrays->AddAtAndExpand(new TObjArray, layer);
  }

  TString canName("canvas");
  (canName += layer) += "_";

  for(Long_t i = 0; i < nCanvases; i++){
    Int_t width = fCanvasWidth, height = fCanvasHeight;
//  on screen this is nice, but Print for different canvas sizes in one .ps fails...
//     if(fNoX[layer] < fNoY[layer]){
//       width = (width * 11) / 14;
//       height *= 73; height /= 50;
//     } else if(fNoX[layer] > fNoY[layer]){ // new!
//       width = (width * 73) / 50;
//       //      height *= 11; height /= 14;
//     }

    while(gROOT->FindObject(canName+i)){
      canName += 'n';
    }

    TCanvas* can = new TCanvas(canName+i, canName+i, 10, 10, width, height);
    if (fNoX[layer] != 1 || fNoY[layer] != 1) can->Divide(fNoX[layer], fNoY[layer]);
    static_cast<TObjArray*>(fCanArrays->At(layer))->Add(can);
  }
}


//________________________________________________________
Int_t GFHistManager::NumberOfSubPadsOf(TCanvas* can)
{
  Int_t n = 0;

  TIter next(can ? can->GetListOfPrimitives() : NULL);
  while (TObject* obj = next()) {
    if (obj->InheritsFrom(TVirtualPad::Class())){
      ++n;
    }
  }

  return n;
}

//________________________________________________________
void GFHistManager::SetLegendX1Y1X2Y2(Double_t x1, Double_t y1, Double_t x2,Double_t y2)
{
  fLegendX1 = x1;
  fLegendY1 = y1;
  fLegendX2 = x2;
  fLegendY2 = y2;
}

//________________________________________________________
void GFHistManager::SetLegendX1(Double_t x1) {fLegendX1 = x1;}
//________________________________________________________
void GFHistManager::SetLegendY1(Double_t y1) {fLegendY1 = y1;}
//________________________________________________________
void GFHistManager::SetLegendX2(Double_t x2) {fLegendX2 = x2;}
//________________________________________________________
void GFHistManager::SetLegendY2(Double_t y2) {fLegendY2 = y2;}

//________________________________________________________
void GFHistManager::SetStatsX1Y1X2Y2(Double_t x1, Double_t y1, Double_t x2, Double_t y2)
{
  fStatsX1 = x1;
  fStatsX2 = x2;
  fStatsY1 = y1;
  fStatsY2 = y2;
}

//________________________________________________________
void GFHistManager::SetNumHistsX(UInt_t numX)
{
  for(Int_t i = 0; i < fDepth; ++i){
    fNoX[i] = numX;
  }
}

//________________________________________________________
void GFHistManager::SetNumHistsX(UInt_t numX, Int_t layer)
{
  if(this->CheckDepth("SetNumHistsX", layer, kFALSE)) {
    fNoX[layer] = numX;
  }
}

//________________________________________________________
void GFHistManager::SetNumHistsY(UInt_t numY)
{
  for(Int_t i = 0; i < fDepth; ++i){
    fNoY[i] = numY;
  }
}

//________________________________________________________
void GFHistManager::SetNumHistsY(UInt_t numY, Int_t layer)
{
  if(this->CheckDepth("SetNumHistsY", layer, kFALSE)) {
    fNoY[layer] = numY;
  }
}

//________________________________________________________
void GFHistManager::SetNumHistsXY(UInt_t numX, UInt_t numY)
{
  this->SetNumHistsX(numX);
  this->SetNumHistsY(numY);
}

//________________________________________________________
void GFHistManager::SetNumHistsXY(UInt_t numX, UInt_t numY, Int_t layer)
{
  this->SetNumHistsX(numX, layer);
  this->SetNumHistsY(numY, layer);
}

//________________________________________________________
void GFHistManager::SetLogY(Bool_t yesNo)
{
  for(Int_t i = 0; i < fDepth; ++i){
    this->SetLogY(i, yesNo);
  }
}

//________________________________________________________
void GFHistManager::SetLogY(Int_t layer, Bool_t yesNo)
{
  if(this->CheckDepth("SetLogY", layer, kFALSE)) {
    fLogY[layer] = yesNo ? 1 : 0;
    if(layer < fCanArrays->GetEntriesFast()) {
      this->Update(layer); // if canvas already drawn
    }
  }
}

//________________________________________________________
void GFHistManager::SetHistsOption(Option_t* option)
{
  for(Int_t i = 0; i < fDepth; ++i){
    this->SetHistsOption(option, i);
  }
}

//________________________________________________________
void GFHistManager::SetHistsOption(Option_t* option, Int_t layer)
{
  if(!this->CheckDepth("SetHistsOption", layer, kFALSE)) return;

  TIter iter2(static_cast<TObjArray*>(fHistArrays->At(layer)));
  while(TObjArray* arr = static_cast<TObjArray*>(iter2.Next())){
    TIter iter(arr); // arr is GFHistArray* !
    while(TH1* hist = static_cast<TH1*>(iter.Next())){
      TString opt(option); opt.ToLower();
      if(!hist->InheritsFrom(TH2::Class()) && opt.Contains("box")){
	opt.ReplaceAll("box",0);
      }
      hist->SetOption(opt);
    }
  }
}

//________________________________________________________
void GFHistManager::SetHistsMinMax(Double_t minMax, Bool_t min)
{
  for(Int_t i = 0; i < fDepth; ++i){
    this->SetHistsMinMax(minMax, min, i);
  }
}

//________________________________________________________
void GFHistManager::SetHistsMinMax(Double_t minMax, Bool_t min, Int_t layer)
{
  if(!this->CheckDepth("SetHistsMinMax", layer, kFALSE)) return;

  TIter iter2(static_cast<TObjArray*>(fHistArrays->At(layer)));
  while(TObjArray* arr = static_cast<TObjArray*>(iter2.Next())){
    TIter iter(arr); // arr is GFHistArray* !
    while(TH1* hist = static_cast<TH1*>(iter.Next())){
      if(min) hist->SetMinimum(minMax);
      else hist->SetMaximum(minMax);
    }
  }
  if(layer < fCanArrays->GetEntriesFast()) {
    this->Update(layer); // if canvas already drawn
  }
}


//________________________________________________________
void GFHistManager::AddHistsOption(Option_t* option)
{
  for(Int_t i = 0; i < fDepth; ++i){
    this->AddHistsOption(option, i);
  }
}

//________________________________________________________
void GFHistManager::AddHistsOption(Option_t* option, Int_t layer)
{
  // if 'layer' exists, add 'option' to all hists so far existing in this layer
  // (ignore option 'box' for 1-D histograms)
  if(!this->CheckDepth("AddHistsOption", layer, kFALSE)) return;

  TIter iter2(static_cast<TObjArray*>(fHistArrays->At(layer)));
  while(TObjArray* arr = static_cast<TObjArray*>(iter2.Next())){
    TIter iter(arr); // arr is GFHistArray* !
    while(TH1* hist = static_cast<TH1*>(iter.Next())){
      TString opt(option); opt.ToLower();
      if(!hist->InheritsFrom(TH2::Class()) && opt.Contains("box",TString::kIgnoreCase)){
	opt.ReplaceAll("box",0);
      }
      hist->SetOption(opt += hist->GetOption());
    }
  }
}


//________________________________________________________
void GFHistManager::SetHistsXTitle(const char* title)
{
  for(Int_t i = 0; i < fDepth; ++i){
    this->SetHistsXTitle(title, i);
  }
}

//________________________________________________________
void GFHistManager::SetHistsXTitle(const char* title, Int_t layer)
{
  if(!this->CheckDepth("SetHistsXTitle", layer, kFALSE)) return;

  TIter iter2(static_cast<TObjArray*>(fHistArrays->At(layer)));
  while(TObjArray* arr = static_cast<TObjArray*>(iter2.Next())){
    TIter iter(arr); // arr is GFHistArray* !
    while(TH1* hist = static_cast<TH1*>(iter.Next())){
      hist->SetXTitle(title);
    }
  }
}

//________________________________________________________
void GFHistManager::SetHistsYTitle(const char* title)
{
  for(Int_t i = 0; i < fDepth; ++i){
    this->SetHistsYTitle(title, i);
  }
}

//________________________________________________________
void GFHistManager::SetHistsYTitle(const char* title, Int_t layer)
{
  if(!this->CheckDepth("SetHistsYTitle", layer, kFALSE)) return;

  TIter iter2(static_cast<TObjArray*>(fHistArrays->At(layer)));
  while(TObjArray* arr = static_cast<TObjArray*>(iter2.Next())){
    TIter iter(arr);// arr is GFHistArray* !
    while(TH1* hist = static_cast<TH1*>(iter.Next())){
      hist->SetYTitle(title);
    }
  }
}


//________________________________________________________
void GFHistManager::SetHistsFillColor(Color_t color)
{
  for(Int_t i = 0; i < fDepth; ++i){
    this->SetHistsFillColor(color, i);
  }
}

//________________________________________________________
void GFHistManager::SetHistsFillColor(Color_t color, Int_t layer)
{
  if(!this->CheckDepth("SetHistsFillColor", layer, kFALSE)) return;

  TIter iter2(static_cast<TObjArray*>(fHistArrays->At(layer)));
  while(TObjArray* arr = static_cast<TObjArray*>(iter2.Next())){
    TIter iter(arr); // arr is GFHistArray* !
    while(TH1* hist = static_cast<TH1*>(iter.Next())){
      hist->SetFillColor(color);
    }
  }
}

//________________________________________________________
void GFHistManager::SetHistsLineWidth(Width_t width)
{
  for(Int_t i = 0; i < fDepth; ++i){
    this->SetHistsLineWidth(width, i);
  }
}

//________________________________________________________
void GFHistManager::SetHistsLineWidth(Width_t width, Int_t layer)
{
  if(!this->CheckDepth("SetHistsLineWidth", layer, kFALSE)) return;

  TIter iter2(static_cast<TObjArray*>(fHistArrays->At(layer)));
  while(TObjArray* arr = static_cast<TObjArray*>(iter2.Next())){
    TIter iter(arr); // arr is GFHistArray* !
    while(TH1* hist = static_cast<TH1*>(iter.Next())){
      hist->SetLineWidth(width);
    }
  }
}

//________________________________________________________
void GFHistManager::SetHistsLineStyle(Int_t s)
{
  for(Int_t i = 0; i < fDepth; ++i){
    this->SetHistsLineStyle(s, i);
  }
}

//________________________________________________________
void GFHistManager::SetHistsLineStyle(Int_t s, Int_t layer, Int_t numHistInPad)
{
  // sets style 's' toall hists in 'layer'.
  // if numHistInPad >= 0: only the given histNum in each pad is changed in style
  // (default is numHistInPad = -1)
  if(!this->CheckDepth("SetHistsLineStyle", layer, kFALSE)) return;

  TIter iter2(static_cast<TObjArray*>(fHistArrays->At(layer)));
  while(const GFHistArray* arr = static_cast<GFHistArray*>(iter2.Next())){
    if(numHistInPad < 0 || numHistInPad >= arr->GetEntriesFast()){ // all hist in pad
      TIter iter(arr); // arr is GFHistArray* !
      while(TH1* hist = static_cast<TH1*>(iter.Next())){
	hist->SetLineStyle(s);
      }
    } else {
      (*arr)[numHistInPad]->SetLineStyle(s);
    }
  }
}


//________________________________________________________
void GFHistManager::SetHistsLineColor(Color_t color)
{
  for(Int_t i = 0; i < fDepth; ++i){
    this->SetHistsLineColor(color, i);
  }
}

//________________________________________________________
void GFHistManager::SetHistsLineColor(Color_t color, Int_t layer)
{
  if(!this->CheckDepth("SetHistsLineColor", layer, kFALSE)) return;

  TIter iter2(static_cast<TObjArray*>(fHistArrays->At(layer)));
  while(TObjArray* arr = static_cast<TObjArray*>(iter2.Next())){
    TIter iter(arr); // arr is GFHistArray* !
    while(TH1* hist = static_cast<TH1*>(iter.Next())){
      hist->SetLineColor(color);
    }
  }
}

//________________________________________________________
Bool_t GFHistManager::CheckDepth(const char* method, Int_t layer, 
				 Bool_t mayExpand)
{
  // true, if layer is possible, if (mayExpand) expands to layer >= 0
  if(layer < 0){
    this->Warning("CheckDepth", "Layer below 0 (%d) called in '%s'!",
		  layer, method);
    return kFALSE;
  }

  if(layer > fDepth-1){
    if(mayExpand) {
      this->ExpandTo(layer);
      return kTRUE;
    } else {
      this->Warning("CheckDepth", "Layer %d called in '%s', max. is %d",
		  layer, method, fDepth-1);
      return kFALSE;
    }
  }
  return kTRUE;
}

//________________________________________________________
void GFHistManager::ExpandTo(Int_t layer)
{
  if(layer+1 <= fDepth){
    this->Error("ExpandTo",
		"Shrinking forbidden, fDepth = %d, should expand to = %d",
		fDepth, layer+1);
    return;
  }

  fNoX.Set(layer+1);
  fNoY.Set(layer+1);
  fLogY.Set(layer+1);

  for(Int_t i = fDepth; i <= layer; ++i){
    fNoX[i] = kDefaultPadsPerCanX;
    fNoY[i] = kDefaultPadsPerCanY;
    fLogY[i]= 0;
    fHistArrays->AddAtAndExpand(new TObjArray, i);
    fCanArrays->AddAtAndExpand(new TObjArray, i);
  }

  fDepth = layer+1;
}

//________________________________________________________
Bool_t GFHistManager::CheckHistNum(const char* method, Int_t layer, 
				 Int_t histNum, Bool_t mayExpand)
{
  // true if hist 'histNum' exists in 'layer' 
  // if(mayExpand == kTRUE) expands to this size if necessary! (default: kFALSE)
  if(!this->CheckDepth(method, layer, mayExpand)) return kFALSE; 
  

  TObjArray * layerArr = static_cast<TObjArray*>(fHistArrays->At(layer));
  if(histNum < 0) {
    this->Warning("CheckHistNum", "histogram number %d requested!", histNum);
    return kFALSE;
  }
  while(histNum >= layerArr->GetEntriesFast()){
    if(mayExpand){
      layerArr->AddAtAndExpand(new GFHistArray, layerArr->GetEntriesFast());
    } else {
      this->Warning("CheckHistNum", "layer %d has only %d histograms, number %d requested!",
		    layer, layerArr->GetEntriesFast(), histNum);
      return kFALSE;
    }
  }
  return kTRUE;
}

//________________________________________________________
TObjArray* GFHistManager::MakeLegends(Int_t layer)
{
  //  returns array of legends of 'layer' (to be called if 'layer' really exist!)
  // creates if necessary
  if(!fLegendArrays) fLegendArrays = new TObjArray(fDepth);
  if(layer > fLegendArrays->GetLast() || !fLegendArrays->At(layer)) {
    fLegendArrays->AddAtAndExpand(new TObjArray, layer);
  }

  return static_cast<TObjArray*>(fLegendArrays->At(layer));
}

//________________________________________________________
TList* GFHistManager::MakeObjList(Int_t layer, Int_t histoNum)
{
  // return list of objects to be drawn upon hists in pad histoNum of 'layer' 
  // (to be called if 'layer' really exist!)
  if(!fObjLists) fObjLists = new TObjArray(fDepth);
  if(layer > fObjLists->GetLast() || !fObjLists->At(layer)){
    fObjLists->AddAtAndExpand(new TObjArray(this->GetNumHistsOf(layer)),layer);
  }
  TObjArray* layerLists = static_cast<TObjArray*>(fObjLists->At(layer));
  if(histoNum > layerLists->GetLast() || !layerLists->At(histoNum)){
    layerLists->AddAtAndExpand(new TList, histoNum); 
  }

  return static_cast<TList*>(layerLists->At(histoNum));
}

//________________________________________________________
GFHistArray* GFHistManager::GetHistsOf(Int_t layer, Int_t histNum, Bool_t mayExpand)
{
  //  returns array of histograms for pad 'histNum' of 'layer'
  // if(mayExpand) creates if necessary!
  if(!this->CheckHistNum("GetHistsOf", layer, histNum, mayExpand)) return NULL;
  TObjArray* layerHists = static_cast<TObjArray*>(fHistArrays->At(layer));
  return static_cast<GFHistArray*>(layerHists->At(histNum));
}

//________________________________________________________
TList* GFHistManager::GetObjectsOf(Int_t layer, Int_t histNo)
{
  if(!this->CheckHistNum("GetObjectsOf", layer, histNo, kFALSE)) return NULL;

  if(fObjLists && layer <= fObjLists->GetLast() && fObjLists->At(layer)){
    TObjArray* layerLists = static_cast<TObjArray*>(fObjLists->At(layer));
    if(histNo <= layerLists->GetLast() && layerLists->At(histNo)){
      return static_cast<TList*>(layerLists->At(histNo));
    }
  }

  return NULL;
}

//________________________________________________________
Int_t GFHistManager::GetNumHistsOf(Int_t layer)
{
  if(!this->CheckDepth("GetNumHistsOf", layer, kFALSE)) return 0;
  TObjArray* layerHists = static_cast<TObjArray*>(fHistArrays->At(layer));
  if(layerHists) return layerHists->GetEntriesFast();
  return 0;
}

//________________________________________________________
TLegend* GFHistManager::GetLegendOf(Int_t layer, Int_t histoNum)
{
  // if it already exists!
  if(!this->CheckHistNum("AddLegend", layer, histoNum)) return NULL;

  TObjArray* legendsOfLayer = this->MakeLegends(layer);
  TLegend* legend = (legendsOfLayer->GetSize() < histoNum ? 
		     NULL : static_cast<TLegend*>(legendsOfLayer->At(histoNum)));
  return legend;
}


//________________________________________________________
void GFHistManager::MakeDifferentStyle(GFHistArray* hists)
{
  // easy version: adjust the histogram lines to colors
  // kBlack, kRed, kGreen, kBlue, kYellow, kMagenta, kCyan (c.f. $ROOTSYS/include/Gtypes.h)
  // 1       2     3       4      5        6         7 (skip 3 and 5: kGreen hardly visible...)
  //                                                           (... kYellow dito)
  // also set line style, set marker color to line color

  if (!hists || hists->GetEntriesFast() < 2) return; // nothing to do

  Color_t color = 0;
  for(Int_t i = 0; i < hists->GetEntriesFast(); ++i){
//     Color_t color = i+1;
    ++color;
    Style_t style = i+1;
    while(style > 4) style -= 4;
    if(color == 3) ++color; //omit kGreen
    if(color == 5) ++color; // and kYellow 
    if(color > 7 ) {
      ::Error("GFHistManager::MakeDifferentStyle", "Out of colors");
      color = 0;//      break;
    }
    hists->At(i)->SetLineColor(color);
    hists->At(i)->SetMarkerColor(color);
    hists->At(i)->SetLineStyle(style);
  }
}

//________________________________________________________
Double_t GFHistManager::MaxOfHist(const TH1* h) const
{
  // Take bin with max content, adds error and returns (no *1.05 applied!),
  // but cares that - together with an error bar - another bin might be higher...
  // If the hists option contains nothing about errors: error is NOT added!

  TString option = h->GetOption();
  option.ToLower();
  option.ReplaceAll("same", 0);
  
  Int_t maxBin = h->GetMaximumBin();
  Double_t result = h->GetBinContent(maxBin);
  if(option.Contains('e') || (h->GetSumw2N() && !option.Contains("hist"))){
    for(Int_t bin = 1; bin <= h->GetNbinsX(); ++bin){ //FIXME: for 2/3D: loop over y/z!
      result = TMath::Max(result, (h->GetBinContent(bin) + h->GetBinError(bin))); 
    }
  }
  return result;
}

//________________________________________________________
Double_t GFHistManager::MaxOfHists(const TObjArray* hists) const
{
  Double_t result = 0.;
  TIter nextHist(hists);
  while(TObject* hist = nextHist()){
    if(hist->InheritsFrom(TH1::Class())){
      result = TMath::Max(result, this->MaxOfHist(static_cast<TH1*>(hist)));
    } else {
      this->Warning("MaxOfHists", "Entry in input array is not a histogram!");
    }
  }
  return result;
}

//________________________________________________________
Double_t GFHistManager::MinOfHist(const TH1* h) const
{
  // Take bin with min content, subtract error and returns (no *1.05 applied!),
  // but cares that - together with an error bar - another bin might be lower...
  // If the hists option contains nothing about errors: error is NOT subtracted!

  TString option = h->GetOption();
  option.ToLower();
  option.ReplaceAll("same", 0);
  
  const Int_t minBin = h->GetMinimumBin();
  Double_t result = h->GetBinContent(minBin);
  if(option.Contains('e') || (h->GetSumw2N() && !option.Contains("hist"))){
    for(Int_t bin = 1; bin <= h->GetNbinsX(); ++bin){ //FIXME: for 2/3D: loop over y/z!
      result = TMath::Min(result, (h->GetBinContent(bin) - h->GetBinError(bin))); 
    }
  }
  return result;
}

//________________________________________________________
Double_t GFHistManager::MinOfHists(const TObjArray* hists) const
{
  Double_t result = DBL_MAX;
  TIter nextHist(hists);
  while(TObject* hist = nextHist()){
    if(hist->InheritsFrom(TH1::Class())){
      result = TMath::Min(result, this->MinOfHist(static_cast<TH1*>(hist)));
    } else {
      this->Warning("MinOfHists", "Entry in input array is not a histogram!");
    }
  }

  return result;
}
//________________________________________________________
TCanvas*  GFHistManager::GetCanvas(Int_t layer, Int_t number)
{
  // after draw!!
  if(!fCanArrays || fCanArrays->GetEntriesFast() <= layer) return NULL;
  TObjArray* cans = static_cast<TObjArray*>(fCanArrays->At(layer));
  if(cans && cans->GetEntriesFast() > number){
    return static_cast<TCanvas*>(cans->At(number));
  }
  return NULL;
}

//________________________________________________________
TVirtualPad* GFHistManager::GetPad(Int_t layer, Int_t histNum)
{
  // pointer to pad where hists from layer/histNum are painted in
  // callable after draw!
    
  Int_t totHistsYet = 0;
  Int_t numHists = 0;
  TCanvas *can = NULL;

  for (Int_t numCan = 0; ; ++numCan) {
    can = this->GetCanvas(layer, numCan);
    if (!can) break;
    numHists = TMath::Max(1, this->NumberOfSubPadsOf(can));
    totHistsYet += numHists;
    if (totHistsYet > histNum) {
      totHistsYet -= numHists;
      break;
    }
  }

  TVirtualPad *result = NULL;
  if (can) {
    TVirtualPad *oldPad = gPad;
    if (numHists <= 1) can->cd(0); // one hist per canvas: no pads!
    else can->cd(histNum - totHistsYet + 1);
    result = gPad;
    oldPad->cd();
  }

  return result;
}

//________________________________________________________
Int_t GFHistManager::GetNumHistsX(Int_t layer) const
{
  if(layer >= 0 && layer < fDepth) return fNoX[layer];
  else return 0;
}

//________________________________________________________
Int_t GFHistManager::GetNumHistsY(Int_t layer) const
{
  if(layer >= 0 && layer < fDepth) return fNoY[layer];
  else return 0;
}

//________________________________________________________
void GFHistManager::GetLegendX1Y1X2Y2(Double_t& x1, Double_t& y1, 
				      Double_t& x2, Double_t& y2) const
{
   x1 = fLegendX1;
   y1 = fLegendY1;
   x2 = fLegendX2;
   y2 = fLegendY2;
}

//________________________________________________________
void GFHistManager::DrawFuncs(const TH1* hist) const
{
  // calls Draw("SAME") for all TF1 in hist's GetListOfFunctions if necessary
  if(!hist || !TString(hist->GetOption()).Contains("HIST", TString::kIgnoreCase)) return;
  TIter nextprim(hist->GetListOfFunctions());
  while(TObject* next = nextprim()){
    if(next->InheritsFrom(TF1::Class())){
      next->Draw("SAME");
    }
  }
}

//________________________________________________________
void GFHistManager::ColourStatsBoxes(GFHistArray *hists) const
{
  // colours stats boxes like hists' line colors and moves the next to each other
  if (!hists) return;
  Double_t x1 = fStatsX1, x2 = fStatsX2, y1 = fStatsY1, y2 = fStatsY2;
  for (Int_t iH = 0; iH < hists->GetEntriesFast(); ++iH) {
    TH1 *h = hists->At(iH);
    if (!h) continue;
    TObject *statObj = h->GetListOfFunctions()->FindObject("stats");
    if (statObj && statObj->InheritsFrom(TPaveStats::Class())) {
      TPaveStats *stats = static_cast<TPaveStats*>(statObj);
      stats->SetLineColor(hists->At(iH)->GetLineColor());
      stats->SetTextColor(hists->At(iH)->GetLineColor());
      stats->SetX1NDC(x1);
      stats->SetX2NDC(x2);
      stats->SetY1NDC(y1);
      stats->SetY2NDC(y2);
      y2 = y1 - 0.005; // shift down 2
      y1 = y2 - (fStatsY2 - fStatsY1); // shift down 1
      if (y1 < 0.) {
	y1 = fStatsY1; y2 = fStatsY2; // restart y-positions
	x2 = x1 - 0.005; // shift left 2
	x1 = x2 - (fStatsX2 - fStatsX1); // shift left 1
	if (x1 < 0.) { // give up, start again:
	  x1 = fStatsX1, x2 = fStatsX2, y1 = fStatsY1, y2 = fStatsY2;
	}
      }
    } else if (gStyle->GetOptStat() != 0) { // failure in case changed in list via TExec....
      this->Warning("ColourStatsBoxes", "No stats found for %s", hists->At(iH)->GetName());
    }
  }
}

//________________________________________________________
void GFHistManager::ColourFuncs(GFHistArray *hists) const
{
  // adjust colour of funcs to match hist, but only if exactly one function per hist
  if (!hists) return;
  for (Int_t iH = 0; iH < hists->GetEntriesFast(); ++iH) {
    TH1 *h = hists->At(iH);
    if (!h) continue;

    // look for _the_ TF1 (not > 1!)
    TF1 *func = NULL;
    TIter nextprim(h->GetListOfFunctions());
    while (TObject* next = nextprim()) {
      if (next->InheritsFrom(TF1::Class())) {
	if (func) { // there is already a TF1, so...
	  func = NULL; // remove it again...
	  break;       // ...and stop searching for  more!
	} else {
	  func = static_cast<TF1*>(next);
	}
      }
    }
// if exactly 1 found, adjust line style/colour
    if (func) {
      func->SetLineColor(h->GetLineColor());
      func->SetLineStyle(h->GetLineStyle());
    }
  } 
}
