// Original Author: Gero Flucke
// last change    : $Date: 2012/03/29 08:42:23 $
// by             : $Author: flucke $

#include <map>
#include <vector>
#include <iostream>

#include "TString.h"
#include "TTree.h"
#include "TGraph.h"
#include "TGraphErrors.h"
#include "TMultiGraph.h"
#include "TCanvas.h"
#include "TROOT.h"
#include "TFile.h"
#include "TList.h"
#include "TLegend.h"

#include "PlotMillePede.h"
#include "PlotMillePedeIOV.h"
#include "GFUtils/GFHistManager.h"

PlotMillePedeIOV::PlotMillePedeIOV(const char *fileName, Int_t maxIov,
				   Int_t hieraLevel)
  : fHistManager(new GFHistManager), fTitle("")
{
  if (maxIov <= 0) { // find maximum IOV in file if not specified
    maxIov = 0;
    TFile *file = TFile::Open(fileName);
    if (!file) return;

    // Loop as long as we find MillePedeUser_<n> trees:
    TTree *tree = 0;
    do {
      file->GetObject(Form("MillePedeUser_%d", ++maxIov), tree);
    } while (tree);
    --maxIov; // one step back since last try did not succeed

    delete file;
  }

  for (Int_t iov = 1; iov <= maxIov; ++iov) {
    fIovs.push_back(new PlotMillePede(fileName, iov, hieraLevel));
    fIovs.back()->GetHistManager()->SetBatch();
    TTree *tree = fIovs.back()->GetMainTree();
    tree->SetEstimate(tree->GetEntries()); // for safe use of tree->GetV1() etc.
  }

  fHistManager->SetLegendX1Y1X2Y2(0.14, 0.7, 0.45, 0.9);
}

PlotMillePedeIOV::~PlotMillePedeIOV()
{
  delete fHistManager;
  for (unsigned int i = 0; i < fIovs.size(); ++i) {
    delete fIovs[i];
  }
}

////////////////////////////////////////////////////////////////////////////////
void PlotMillePedeIOV::DrawPedeParam(Option_t *option, unsigned int nNonRigidParam)
{
  const unsigned int nPar = PlotMillePede::kNpar + nNonRigidParam;
  const Int_t layer = this->PrepareAdd(TString(option).Contains("add", TString::kIgnoreCase));
  const Int_t xInTitle = TString(option).Contains("x", TString::kIgnoreCase);
  const Int_t yInTitle = TString(option).Contains("y", TString::kIgnoreCase);
  const Int_t zInTitle = TString(option).Contains("z", TString::kIgnoreCase);
  // if no position is selected for legend, use DetId:
  const Int_t idInTitle = (!xInTitle && !yInTitle && !zInTitle)
    || TString(option).Contains("id", TString::kIgnoreCase);
  const Int_t doErr = TString(option).Contains("err", TString::kIgnoreCase);
  const Int_t noErr = TString(option).Contains("val", TString::kIgnoreCase);
  
  typedef std::map<ParId, TGraphErrors*> GraphMap;
  GraphMap graphs; // size will be nPar*numAlis (if same nPar for all alignables)

  unsigned int maxPar = 0;
  for (unsigned int iIov = 0; iIov < fIovs.size(); ++iIov) { // loop in IOVs
    PlotMillePede *iov = fIovs[iIov];
    TTree *tree = iov->GetMainTree();
    for (unsigned int iPar = 0; iPar < nPar; ++iPar) { // loop on parameters
      if (iPar > maxPar) maxPar = iPar;
      const TString pedePar(iov->MpT() += iov->Par(iPar) += iov->ToMumMuRadPede(iPar));
      const TString parSi(iov->ParSi(iPar) += iov->ToMumMuRadPede(iPar));
      const TString par(doErr ? parSi : pedePar); // decide to do param or error
      // The selection is tree-name (but not parameter...) dependent:
      TString selection;//(iov->Valid(iPar)); // FIXME??
      iov->AddBasicSelection(selection);
      // main command to get parameter & also (Det)Id and ObjId (e.g. TPBLayer)
      const Long64_t numAlis = tree->Draw(par + ":Id:ObjId:" + parSi, selection, "goff");
      // copy result of above Draw(..) 
      const std::vector<double> values(tree->GetV1(), tree->GetV1() + numAlis);
      const std::vector<double> ids   (tree->GetV2(), tree->GetV2() + numAlis);
      const std::vector<double> objIds(tree->GetV3(), tree->GetV3() + numAlis);
      const std::vector<double> sigmas(tree->GetV4(), tree->GetV4() + numAlis);

      // now loop on selected alignables and create/fill graphs
      for (Long64_t iAli = 0; iAli < numAlis; ++iAli) {
	// ParId is Id, ObjId and parameter number - used as key for the map
	const ParId id(ids[iAli], objIds[iAli], iPar);
	TGraphErrors *&gr = graphs[id]; // pointer by ref (might be created!)
	if (!gr) {
	  // this tree->Draw(..) is only needed if xyz position requested...
	  tree->Draw(iov->XPos() += ":" + iov->YPos() +=
		     ":" + iov->ZPos(), selection, "goff");
	  gr = new TGraphErrors; // Assigns value to map-internal pointer. (!)
	  // We define title for legend here:
	  TString title(iov->AlignableObjIdString(id.objId_));
	  if (idInTitle)title += Form(", %d", id.id_);
	  if (xInTitle) title += Form(", x=%.1f", tree->GetV1()[iAli]);
	  if (yInTitle) title += Form(", y=%.1f", tree->GetV2()[iAli]);
	  if (zInTitle) title += Form(", z=%.f", tree->GetV3()[iAli]);
	  // if (title.Last(',') != kNPOS) title.Remove(title.Last(','));
	  gr->SetTitle(title);
	}
	gr->SetPoint(gr->GetN(), iIov+1, values[iAli]); // add new point for IOV
	if (!doErr      // not if we plot error instead of value
	    && !noErr   // not if error bar is deselected
	    && sigmas[iAli] > 0.) { // determined by pede (inversion...)
	  gr->SetPointError(gr->GetN()-1, 0., sigmas[iAli]); // add error in y
        }
      } // end loop on alignables
    } // end loop on parameters 
  } // end loop on IOVs

  //  if (graphs.empty()) return; // 

  // Now we have all graphs filled - order them such that 
  // the same parameter is attached to the same multigraph:
  std::vector<TMultiGraph*> multis(maxPar+1); // one multigraph per parameter
  for (GraphMap::const_iterator iGr = graphs.begin(); iGr != graphs.end();
       ++iGr) { // loop on map of graphs
    const ParId &id = iGr->first;
    if (!multis[id.par_]) multis[id.par_] = new TMultiGraph;
    multis[id.par_]->Add(iGr->second, ""); // add an option?
  }

  // Need to draw the multigraph to get its histogram for the axes.
  // This histogram has to be given to the hist manager before the multigraphs.
  // Therefore set ROOT to batch and prepare a temporary TCanvas for drawing.
  const bool isBatch = gROOT->IsBatch(); gROOT->SetBatch();
  TCanvas c; // On stack: later goes out of scope...
  // Finally loop on multigraphs 
  for (unsigned int iMulti = 0; iMulti < multis.size(); ++iMulti) {
    if (multis[iMulti] == 0) continue;
    TIter iter(multis[iMulti]->GetListOfGraphs());
    multis[iMulti]->Draw("ALP"); //'Draw' graph into batch canvas to create hist
    TH1 *h = static_cast<TH1*>(multis[iMulti]->GetHistogram() //...but clone it:
			       ->Clone(this->Unique(Form("IOV%u", iMulti))));
    const PlotMillePede *i0 = fIovs[0]; // IOV does not matter here...
    if (doErr) { // title if we draw error instead of value
      const TString errPar(Form("#sigma(%s)", i0->NamePede(iMulti).Data()));
      h->SetTitle(errPar + " IOVs" += i0->TitleAdd() += ";IOV;"
		  + errPar + i0->UnitPede(iMulti));
    } else if (fTitle != "" ) {
      h->SetTitle(i0->NamePede(iMulti) + " IOVs" + i0->TitleAdd() + ", " + fTitle + ";IOV;"
		  + i0->NamePede(iMulti) += i0->UnitPede(iMulti));
    } else {     // 'usual' title for drawing parameter values
      h->SetTitle((i0->NamePede(iMulti) += " IOVs") += i0->TitleAdd() += ";IOV;"
		  + i0->NamePede(iMulti) += i0->UnitPede(iMulti));
    }
    fHistManager->AddHistSame(h, layer, iMulti); // cloned hist for axes
    fHistManager->AddObject(multis[iMulti], layer, iMulti, "LP");
    // Create legend refering to graphs and add to manager:  
    Double_t x1, x2, y1, y2; fHistManager->GetLegendX1Y1X2Y2(x1, y1, x2, y2);
    TLegend *legend = new TLegend(x1, y1, x2, y2);
    legend->SetFillColor(kWhite);
    legend->SetTextFont(42);
    legend->SetBorderSize(1);
    fHistManager->AddLegend(legend, layer, iMulti);
    Int_t nGr = 0;
    while (TGraph* graph = static_cast<TGraph*>(iter.Next())){
      legend->AddEntry(graph, graph->GetTitle(), "lp"); // title set above
      this->SetLineMarkerStyle(*graph, nGr++);
    }
  }
  gROOT->SetBatch(isBatch); // reset batch mode 

  fHistManager->Draw();
}

//////////////////////////////////////////////////////////////////////////
void PlotMillePedeIOV::SetSubDetId(Int_t subDet)
{
  for (unsigned int iIov = 0; iIov < fIovs.size(); ++iIov) {
    fIovs[iIov]->SetSubDetId(subDet);
  }
}

//////////////////////////////////////////////////////////////////////////
void PlotMillePedeIOV::SetSubDetIds(Int_t id1, Int_t id2, Int_t id3, Int_t id4, Int_t id5)
{
  for (unsigned int iIov = 0; iIov < fIovs.size(); ++iIov) {
    fIovs[iIov]->SetSubDetIds(id1, id2, id3, id4, id5);
  }
}

//////////////////////////////////////////////////////////////////////////
void PlotMillePedeIOV::SetAlignableTypeId(Int_t alignableTypeId)
{
  for (unsigned int iIov = 0; iIov < fIovs.size(); ++iIov) {
    fIovs[iIov]->SetAlignableTypeId(alignableTypeId);
  }
}

//////////////////////////////////////////////////////////////////////////
void PlotMillePedeIOV::SetHieraLevel(Int_t hieraLevel)
{
  for (unsigned int iIov = 0; iIov < fIovs.size(); ++iIov) {
    fIovs[iIov]->SetHieraLevel(hieraLevel);
  }
}

//////////////////////////////////////////////////////////////////////////
void PlotMillePedeIOV::SetBowsParameters(bool use)
{
  for (unsigned int iIov = 0; iIov < fIovs.size(); ++iIov) {
    fIovs[iIov]->SetBowsParameters(use);
  }
}

//////////////////////////////////////////////////////////////////////////
void PlotMillePedeIOV::AddAdditionalSel(const TString &xyzrPhiNhit, Float_t min, Float_t max)
{
  for (unsigned int iIov = 0; iIov < fIovs.size(); ++iIov) {
    fIovs[iIov]->AddAdditionalSel(xyzrPhiNhit, min, max);
  }
}

//////////////////////////////////////////////////////////////////////////
void PlotMillePedeIOV::ClearAdditionalSel()
{
  for (unsigned int iIov = 0; iIov < fIovs.size(); ++iIov) {
    fIovs[iIov]->ClearAdditionalSel();
  }
}

//////////////////////////////////////////////////////////////////////////
TString PlotMillePedeIOV::Unique(const char *name) const
{
  if (!gROOT->FindObject(name)) return name;

  UInt_t i = 1;
  while (gROOT->FindObject(Form("%s_%u", name, i))) ++i;

  return Form("%s_%u", name, i);
}

//////////////////////////////////////////////////////////////////////////
Int_t PlotMillePedeIOV::PrepareAdd(bool addPlots)
{
  if (addPlots) {
    return fHistManager->GetNumLayers();
  } else {
    fHistManager->Clear();
    return 0;
  }
}

//////////////////////////////////////////////////////////////////////////
template<class T> 
void PlotMillePedeIOV::SetLineMarkerStyle(T &object, Int_t num) const
{
  // styles start with num=0
  // color: use 1-4, 6-9, 41, 46, 49
  Int_t colour = num%11 + 1; // i.e. 1-11 
  if (colour > 4) ++colour; // skip 5=yellow
  if (colour == 10) colour = 41;
  else if (colour == 11) colour = 46;
  else if (colour == 12) colour = 49;
  object.SetLineColor(colour);
  object.SetMarkerColor(object.GetLineColor());

  // style
  Int_t marker = 0; // use 14 styles
  switch (num%14) {
  case 0: marker = kFullCircle;
    break;
  case 1: marker = kFullSquare;
    break;
  case 2: marker = kFullTriangleUp;
    break;
  case 3: marker = kFullTriangleDown;
    break;
  case 4: marker = kOpenCircle;
    break;
  case 5: marker = kOpenSquare;
    break;
  case 6: marker = kOpenTriangleUp;
    break;
  case 7: marker = kOpenTriangleDown; // 32
    break;
  case 8: marker = kOpenStar;
    break;
  case 9: marker = kFullStar;
    break;
  case 10: marker = kOpenCross;
    break;
  case 11: marker = kFullCross;
    break;
  case 12: marker = kOpenDiamond;
    break;
  case 13: marker = kFullDiamond;
    break;
  default: marker = kPlus; // should not be reached...
  }
  object.SetMarkerStyle(marker);
  
  //object.SetLineStyle(num%10 + 1); // use (default) line styles 1-10
  object.SetLineStyle(num%4 + 1); // use only line styles 1-4

}

bool PlotMillePedeIOV::ParId::operator< (const ParId& other) const
{
  // Sorting needed for use as key in std::map:
  // first sort by id_, second objId_, finally par_.
  if (id_ < other.id_) return true;
  else if (id_ > other.id_) return false;
  else { // id the same
    if (objId_ < other.objId_) return true;
    else if (objId_ > other.objId_) return false;
    else { // id and objId the same
      if (par_ < other.par_) return true;
      // redundant: else if (par_ > other.par_) return false;
      else return false; // all are the same!
    }
  }
}
