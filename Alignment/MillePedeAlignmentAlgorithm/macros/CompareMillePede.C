// Original Author: Gero Flucke
// last change    : $Date: 2013/03/07 12:31:06 $
// by             : $Author: flucke $

#include "CompareMillePede.h"
#include "PlotMillePede.h"
#include "GFUtils/GFHistManager.h"

#include <TTree.h>
#include <TString.h>
#include <TError.h>
#include <TH1.h>
#include <TH2.h>
#include <TLine.h>

#include <iostream>
const unsigned int CompareMillePede::kNpar = MillePedeTrees::kNpar; // number of parameters we have...

//_________________________________________________________________________________________________
CompareMillePede::CompareMillePede(const char *fileName1, const char *fileName2,
                                   Int_t iov1, Int_t iov2, Int_t hieraLevel) :
  fPlotMp1(new PlotMillePede(fileName1, iov1, hieraLevel, "first")),
  fPlotMp2(new PlotMillePede(fileName2, iov2, hieraLevel, "second")),
  fHistManager(new GFHistManager)
{
  fHistManager->SetLegendX1Y1X2Y2(0.14, 0.7, 0.45, 0.9);
  
  TTree *tree1 = fPlotMp1->GetMainTree();
  TTree *tree2 = fPlotMp2->GetMainTree();
  if (!tree1 || !tree2) {
    ::Error("CompareMillePede", "Stop here: Previous problems...");
  } else if (tree1->GetEntries() != tree2->GetEntries()) {
    // '%lld' for Long64_t
    ::Error("CompareMillePede", "Stop here: %lld alignables in tree1, %lld in tree2.",
	    tree1->GetEntries(), tree2->GetEntries());
  } else {
    tree1->AddFriend(tree2);
    if (!this->IsConsistent()) {
      ::Error("CompareMillePede", "Alignables in inconsistent order.");
    } 
  }
}

//_________________________________________________________________________________________________
CompareMillePede::~CompareMillePede()
{
  delete fHistManager;
  delete fPlotMp1;
  delete fPlotMp2;
}

//_________________________________________________________________________________________________
void CompareMillePede::DrawPedeParam(Option_t *option, unsigned int nNonRigidParam)
{

  const TString opt(option);

  const Int_t layer = this->PrepareAdd(opt.Contains("add", TString::kIgnoreCase));
  const bool lineVs = opt.Contains("line", TString::kIgnoreCase);

  const TString titleAdd = this->TitleAdd();

  const PlotMillePede *m = fPlotMp1;
  UInt_t nPlot = 0;
  for (UInt_t iPar = 0; iPar < kNpar + nNonRigidParam; ++iPar) { // 
    TString sel("");
    this->AddIsFreeSel(sel, opt, iPar);
    fPlotMp1->AddBasicSelection(sel);
    fPlotMp2->AddBasicSelection(sel);
    
    const TString pedePar1(fPlotMp1->MpT() += m->Par(iPar));
    const TString pedePar2(fPlotMp2->MpT() += m->Par(iPar));
    const TString deltaPedePar(m->Parenth(pedePar2 + m->Min() += pedePar1)
                               += m->ToMumMuRadPede(iPar));
    const TString deltaName(m->Unique(Form("deltaPedePar%d", iPar)));
    TH1 *h = fPlotMp1->CreateHist(deltaPedePar, sel, deltaName);
    if (0. == h->GetEntries()) {
      delete h;
      continue;
    }

    const TString name2D(m->Unique(Form("pedePar2D%d", iPar)));
    TH2 *hVs = fPlotMp1->CreateHist2D(pedePar1 + m->ToMumMuRadPede(iPar),
				      pedePar2 + m->ToMumMuRadPede(iPar), sel, name2D, "BOX");

    const TString diff(Form("%s_{2}-%s_{1}", m->NamePede(iPar).Data(), m->NamePede(iPar).Data()));
    h->SetTitle(diff + titleAdd + ";" + diff + m->UnitPede(iPar) +=";#parameters");

    hVs->SetTitle(m->NamePede(iPar).Data() + titleAdd
		  + Form(";%s_{1}", m->NamePede(iPar).Data()) + m->UnitPede(iPar)
		  += Form(";%s_{2}", m->NamePede(iPar).Data()) + m->UnitPede(iPar));
    fHistManager->AddHist(h, layer);
    fHistManager->AddHist(hVs, layer+1);

    if (lineVs) { // add a TLine
      const Double_t xMin = hVs->GetXaxis()->GetXmin();
      const Double_t xMax = hVs->GetXaxis()->GetXmax();
      TLine *line = new TLine(xMin, xMin, xMax, xMax);
      line->SetLineColor(kRed); line->SetLineWidth(2);
      fHistManager->AddObject(line, layer + 1, iPar);
    }

    ++nPlot;
  }

  fHistManager->Draw();
}

//_________________________________________________________________________________________________
void CompareMillePede::DrawPedeParamVsLocation(Option_t *option, unsigned int nNonRigidParam)
{

  ::Warning("CompareMillePede::DrawPedeParamVsLocation",
	    "Not yet fully adjusted to possible bows");
  const TString opt(option);
  const Int_t layer = this->PrepareAdd(opt.Contains("add", TString::kIgnoreCase));
  const TString titleAdd = this->TitleAdd();

  const PlotMillePede *m = fPlotMp1;
  UInt_t nPlot = 0;
  for (UInt_t iPar = 0; iPar < kNpar + nNonRigidParam; ++iPar) { // 
    TString sel("");
    this->AddIsFreeSel(sel, opt, iPar);
    fPlotMp1->AddBasicSelection(sel);
    fPlotMp2->AddBasicSelection(sel);

    const TString deltaPedePar(m->Parenth(fPlotMp2->MpT() += m->Par(iPar) += m->Min()
                                          += fPlotMp1->MpT() += m->Par(iPar))
                               += m->ToMumMuRadPede(iPar));

    const TString deltaNameR(fPlotMp1->Unique(Form("deltaPedePar%dR", iPar))); // 
    TH2 *hDr = fPlotMp1->CreateHist2D(fPlotMp1->RPos(fPlotMp1->OrgPosT()), deltaPedePar, sel,
                                      deltaNameR, "BOX");
    if (0. == hDr->GetEntries()) {
      delete hDr;
      continue;
    }

    const TString deltaNamePhi(fPlotMp1->Unique(Form("deltaPedePar%dPhi", iPar))); // 
    TH2 *hDphi = fPlotMp1->CreateHist2D(fPlotMp1->Phi(fPlotMp1->OrgPosT()), deltaPedePar, sel,
                                        deltaNamePhi, "BOX");
    const TString deltaNameZ(fPlotMp1->Unique(Form("deltaPedePar%dZ", iPar))); // 
    TH2 *hDz = fPlotMp1->CreateHist2D(fPlotMp1->OrgPosT() += fPlotMp1->ZPos(), deltaPedePar, sel,
                                      deltaNameZ, "BOX");

    const TString deltaNameH(fPlotMp1->Unique(Form("deltaPedePar%dH", iPar))); // 
    TH2 *hDhit = fPlotMp1->CreateHist2D(fPlotMp1->HitsX(), deltaPedePar, sel, deltaNameH, "BOX");

//     // now devided by sigma prediction from 1
//     const TString deltaBySi1(this->DeltaParBySigma(iPar, fPlotMp1));
//     TString sel1Ok(fPlotMp1->ParSiOk(iPar));
//     if (!sel.IsNull()) sel1Ok.Append(fPlotMp1->AndL() += sel);
//     const TString deltaBySiName1R(fPlotMp1->Unique(Form("deltaPar%dRRel1", iPar))); // 
//     TH2 *hDrS1 = fPlotMp1->CreateHist2D(fPlotMp1->RPos(fPlotMp1->OrgPosT()),
//                                         deltaBySi1, sel1Ok, deltaBySiName1R, "BOX");
//     const TString deltaBySiName1Phi(fPlotMp1->Unique(Form("deltaPar%dPhiRel1", iPar))); // 
//     TH2 *hDphiS1 = fPlotMp1->CreateHist2D(fPlotMp1->Phi(fPlotMp1->OrgPosT()),
//                                           deltaBySi1, sel1Ok, deltaBySiName1Phi, "BOX");
//     const TString deltaBySiName1Z(fPlotMp1->Unique(Form("deltaPar%dZRel1", iPar))); // 
//     TH2 *hDzS1 = fPlotMp1->CreateHist2D(fPlotMp1->OrgPosT() += fPlotMp1->ZPos(),
//                                         deltaBySi1, sel1Ok, deltaBySiName1Z, "BOX");

//     // same now devided by sigma prediction from 2 (but CreateHist from 1!)
//     const TString deltaBySi2(this->DeltaParBySigma(iPar, fPlotMp2));
//     TString sel2Ok(fPlotMp2->ParSiOk(iPar));
//     if (!sel.IsNull()) sel2Ok.Append(fPlotMp2->AndL() += sel);
//     const TString deltaBySiName2R(fPlotMp2->Unique(Form("deltaPar%dRRel2", iPar))); // 
//     TH2 *hDrS2 = fPlotMp1->CreateHist2D(fPlotMp2->RPos(fPlotMp2->OrgPosT()),
//                                         deltaBySi2, sel2Ok, deltaBySiName2R, "BOX");
//     const TString deltaBySiName2Phi(fPlotMp2->Unique(Form("deltaPar%dPhiRel2", iPar))); // 
//     TH2 *hDphiS2 = fPlotMp1->CreateHist2D(fPlotMp2->Phi(fPlotMp2->OrgPosT()),
//                                           deltaBySi2, sel2Ok, deltaBySiName2Phi, "BOX");
//     const TString deltaBySiName2Z(fPlotMp2->Unique(Form("deltaPar%dZRel2", iPar))); // 
//     TH2 *hDzS2 = fPlotMp1->CreateHist2D(fPlotMp2->OrgPosT() += fPlotMp2->ZPos(),
//                                         deltaBySi2, sel2Ok, deltaBySiName2Z, "BOX");

    const TString diff(Form("%s_{2}-%s_{1}", m->NamePede(iPar).Data(), m->NamePede(iPar).Data()));

    hDr->SetTitle(m->DelName(iPar) += " vs r" + titleAdd + ";r[cm];" + diff + ' ' + m->UnitPede(iPar));
    hDphi->SetTitle(m->DelName(iPar) += " vs #phi" + titleAdd + ";#phi;" + diff + ' '
		    + m->Unit(iPar));
    hDz->SetTitle(m->DelName(iPar) += " vs z" + titleAdd + ";z[cm];" + diff + ' ' + m->UnitPede(iPar));
    hDhit->SetTitle(m->DelName(iPar) += " vs #hits" + titleAdd + ";#hits_{x};" + diff + ' '
		    + m->UnitPede(iPar));

    fHistManager->AddHist(hDr, layer + nPlot);
    fHistManager->AddHist(hDphi, layer + nPlot);
    fHistManager->AddHist(hDz, layer + nPlot);
    fHistManager->AddHist(hDhit, layer + nPlot);

//     if (hDrS1->GetEntries()) {
//       hDrS1->SetTitle(m->DelName(iPar) += "/#sigma vs r;r[cm];(" + diff + ")/#sigma");
//       hDphiS1->SetTitle(m->DelName(iPar) += "/#sigma vs #phi;#phi;(" + diff + ")/#sigma"); 
//       hDzS1->SetTitle(m->DelName(iPar) += "/#sigma vs z;z[cm];(" + diff + ")/#sigma"); 
//       fHistManager->AddHist(hDrS1, layer + nPlot + 1, "by #sigma_{1}", "f");
//       fHistManager->AddHist(hDphiS1, layer + nPlot + 1, "by #sigma_{1}", "f");
//       fHistManager->AddHist(hDzS1, layer + nPlot + 1, "by #sigma_{1}", "f");
//     } else {
//       delete hDrS1;
//       delete hDphiS1;
//       delete hDzS1;
//     }
//     if (hDrS2->GetEntries()) {
//       hDrS2->SetLineColor(kRed);
//       hDphiS2->SetLineColor(kRed);
//       hDzS2->SetLineColor(kRed);
//       fHistManager->AddHistSame(hDrS2, layer + nPlot + 1, 0, "by #sigma_{2}", "f");
//       fHistManager->AddHistSame(hDphiS2, layer + nPlot + 1, 1, "by #sigma_{2}", "f");
//       fHistManager->AddHistSame(hDzS2, layer + nPlot + 1, 2, "by #sigma_{2}", "f");
//     } else {
//       delete hDrS2;
//       delete hDphiS2;
//       delete hDzS2;
//     }
//     nPlot += 2;
    ++nPlot;
  }

  const bool oldDiffStyle = fHistManager->DrawDiffStyle(false);//avoid automatic hist style changes
  fHistManager->Draw();
  fHistManager->DrawDiffStyle(oldDiffStyle);
}

//_________________________________________________________________________________________________
void CompareMillePede::DrawParam(Option_t *option)
{

  const TString opt(option);

  const Int_t layer = this->PrepareAdd(opt.Contains("add", TString::kIgnoreCase));
  const TString titleAdd = this->TitleAdd();

  const PlotMillePede *m = fPlotMp1;
  UInt_t nPlot = 0;
  for (UInt_t iPar = 0; iPar < kNpar; ++iPar) { // 
    TString sel("");
    this->AddIsFreeSel(sel, opt, iPar);
    fPlotMp1->AddBasicSelection(sel);
    fPlotMp2->AddBasicSelection(sel);
    
    const TString deltaPar(m->Parenth(this->DeltaPar(iPar)) += m->ToMumMuRad(iPar));
    const TString deltaName(m->Unique(Form("deltaPar%d", iPar))); // 
    TH1 *hD = fPlotMp1->CreateHist(deltaPar, sel, deltaName);
    if (0. == hD->GetEntries()) {
      delete hD;
      continue;
    }

    const TString deltaBySi1(this->DeltaParBySigma(iPar, fPlotMp1));
    const TString deltaBySiName1(m->Unique(Form("deltaPar%dRel1", iPar))); // 
    TString sel1Ok(fPlotMp1->ParSiOk(iPar));
    if (!sel.IsNull()) sel1Ok.Append(m->AndL() += sel);
    TH1 *hDs1 = fPlotMp1->CreateHist(deltaBySi1, sel1Ok, deltaBySiName1);

    const TString deltaBySi2(this->DeltaParBySigma(iPar, fPlotMp2));
    TString deltaBySiName2(m->Unique(Form("deltaPar%dRel2", iPar))); // 
    m->CopyAddBinning(deltaBySiName2, hDs1); 
    TString sel2Ok(fPlotMp2->ParSiOk(iPar));
    if (!sel.IsNull()) sel2Ok.Append(m->AndL() += sel);
    TH1 *hDs2 = fPlotMp1->CreateHist(deltaBySi2, sel2Ok, deltaBySiName2);

    const TString diff(Form("%s_{2}-%s_{1}", m->Name(iPar).Data(), m->Name(iPar).Data()));
    hD->SetTitle(m->DelName(iPar) += titleAdd + ";" + diff + ' ' + m->Unit(iPar) +=";#parameters");
    hDs1->SetTitle(m->DelName(iPar) += "/#sigma" + titleAdd + ";(" + diff +")/#sigma;#parameters");

    fHistManager->AddHist(hD, layer);
    if (hDs1->GetEntries()) {
      fHistManager->AddHist(hDs1, layer+1, "by #sigma_{1}");
    } else delete hDs1;
    if (hDs2->GetEntries()) { // does DrawSame work if hDs1 was empty?
      fHistManager->AddHistSame(hDs2, layer+1, nPlot, "by #sigma_{2}");
    } else delete hDs2;

    ++nPlot;
  }

  fHistManager->Draw();
}

//_________________________________________________________________________________________________
void CompareMillePede::DrawParamVsLocation(Option_t *option)
{

  const TString opt(option);
  const Int_t layer = this->PrepareAdd(opt.Contains("add", TString::kIgnoreCase));
  const TString titleAdd = this->TitleAdd();

  const PlotMillePede *m = fPlotMp1;
  UInt_t nPlot = 0;
  for (UInt_t iPar = 0; iPar < kNpar; ++iPar) { // 
    TString sel("");
    this->AddIsFreeSel(sel, opt, iPar);
    fPlotMp1->AddBasicSelection(sel);
    fPlotMp2->AddBasicSelection(sel);

    const TString deltaPar(fPlotMp1->Parenth(this->DeltaPar(iPar)) += fPlotMp1->ToMumMuRad(iPar));

    const TString deltaNameR(fPlotMp1->Unique(Form("deltaPar%dR", iPar))); // 
    TH2 *hDr = fPlotMp1->CreateHist2D(fPlotMp1->RPos(fPlotMp1->OrgPosT()), deltaPar, sel,
                                      deltaNameR, "BOX");
    if (0. == hDr->GetEntries()) {
      delete hDr;
      continue;
    }

    const TString deltaNamePhi(fPlotMp1->Unique(Form("deltaPar%dPhi", iPar))); // 
    TH2 *hDphi = fPlotMp1->CreateHist2D(fPlotMp1->Phi(fPlotMp1->OrgPosT()), deltaPar, sel,
                                      deltaNamePhi, "BOX");
    const TString deltaNameZ(fPlotMp1->Unique(Form("deltaPar%dZ", iPar))); // 
    TH2 *hDz = fPlotMp1->CreateHist2D(fPlotMp1->OrgPosT() += fPlotMp1->ZPos(), deltaPar, sel,
                                      deltaNameZ, "BOX");
    // now devided by sigma prediction from 1
    const TString deltaBySi1(this->DeltaParBySigma(iPar, fPlotMp1));
    TString sel1Ok(fPlotMp1->ParSiOk(iPar));
    if (!sel.IsNull()) sel1Ok.Append(fPlotMp1->AndL() += sel);
    const TString deltaBySiName1R(fPlotMp1->Unique(Form("deltaPar%dRRel1", iPar))); // 
    TH2 *hDrS1 = fPlotMp1->CreateHist2D(fPlotMp1->RPos(fPlotMp1->OrgPosT()),
                                        deltaBySi1, sel1Ok, deltaBySiName1R, "BOX");
    const TString deltaBySiName1Phi(fPlotMp1->Unique(Form("deltaPar%dPhiRel1", iPar))); // 
    TH2 *hDphiS1 = fPlotMp1->CreateHist2D(fPlotMp1->Phi(fPlotMp1->OrgPosT()),
                                          deltaBySi1, sel1Ok, deltaBySiName1Phi, "BOX");
    const TString deltaBySiName1Z(fPlotMp1->Unique(Form("deltaPar%dZRel1", iPar))); // 
    TH2 *hDzS1 = fPlotMp1->CreateHist2D(fPlotMp1->OrgPosT() += fPlotMp1->ZPos(),
                                        deltaBySi1, sel1Ok, deltaBySiName1Z, "BOX");

    // same now devided by sigma prediction from 2 (but CreateHist from 1!)
    const TString deltaBySi2(this->DeltaParBySigma(iPar, fPlotMp2));
    TString sel2Ok(fPlotMp2->ParSiOk(iPar));
    if (!sel.IsNull()) sel2Ok.Append(fPlotMp2->AndL() += sel);
    const TString deltaBySiName2R(fPlotMp2->Unique(Form("deltaPar%dRRel2", iPar))); // 
    TH2 *hDrS2 = fPlotMp1->CreateHist2D(fPlotMp2->RPos(fPlotMp2->OrgPosT()),
                                        deltaBySi2, sel2Ok, deltaBySiName2R, "BOX");
    const TString deltaBySiName2Phi(fPlotMp2->Unique(Form("deltaPar%dPhiRel2", iPar))); // 
    TH2 *hDphiS2 = fPlotMp1->CreateHist2D(fPlotMp2->Phi(fPlotMp2->OrgPosT()),
                                          deltaBySi2, sel2Ok, deltaBySiName2Phi, "BOX");
    const TString deltaBySiName2Z(fPlotMp2->Unique(Form("deltaPar%dZRel2", iPar))); // 
    TH2 *hDzS2 = fPlotMp1->CreateHist2D(fPlotMp2->OrgPosT() += fPlotMp2->ZPos(),
                                        deltaBySi2, sel2Ok, deltaBySiName2Z, "BOX");

    const TString diff(Form("%s_{2}-%s_{1}", m->Name(iPar).Data(), m->Name(iPar).Data()));

    hDr->SetTitle(m->DelName(iPar) += " vs r" + titleAdd + ";r[cm];" + diff +' '+ m->Unit(iPar));
    hDphi->SetTitle(m->DelName(iPar) += " vs #phi" + titleAdd + ";#phi;"+diff +' '+ m->Unit(iPar));
    hDz->SetTitle(m->DelName(iPar) += " vs z" + titleAdd + ";z[cm];" + diff +' '+ m->Unit(iPar));

    fHistManager->AddHist(hDr, layer + nPlot);
    fHistManager->AddHist(hDphi, layer + nPlot);
    fHistManager->AddHist(hDz, layer + nPlot);

    if (hDrS1->GetEntries()) {
      hDrS1->SetTitle(m->DelName(iPar) += "/#sigma vs r" +titleAdd+ ";r[cm];(" + diff + ")/#sigma");
      hDphiS1->SetTitle(m->DelName(iPar) += "/#sigma vs #phi" + titleAdd + ";#phi;("
			+ diff + ")/#sigma");
      hDzS1->SetTitle(m->DelName(iPar) += "/#sigma vs z" +titleAdd+ ";z[cm];(" + diff + ")/#sigma");
      fHistManager->AddHist(hDrS1, layer + nPlot + 1, "by #sigma_{1}", "f");
      fHistManager->AddHist(hDphiS1, layer + nPlot + 1, "by #sigma_{1}", "f");
      fHistManager->AddHist(hDzS1, layer + nPlot + 1, "by #sigma_{1}", "f");
    } else {
      delete hDrS1;
      delete hDphiS1;
      delete hDzS1;
    }
    if (hDrS2->GetEntries()) {
      hDrS2->SetLineColor(kRed);
      hDphiS2->SetLineColor(kRed);
      hDzS2->SetLineColor(kRed);
      fHistManager->AddHistSame(hDrS2, layer + nPlot + 1, 0, "by #sigma_{2}", "f");
      fHistManager->AddHistSame(hDphiS2, layer + nPlot + 1, 1, "by #sigma_{2}", "f");
      fHistManager->AddHistSame(hDzS2, layer + nPlot + 1, 2, "by #sigma_{2}", "f");
    } else {
      delete hDrS2;
      delete hDphiS2;
      delete hDzS2;
    }
    nPlot += 2;
  }

  const bool oldDiffStyle = fHistManager->DrawDiffStyle(false);//avoid automatic hist style changes
  fHistManager->Draw();
  fHistManager->DrawDiffStyle(oldDiffStyle);
}

//_________________________________________________________________________________________________
void CompareMillePede::DrawParamDeltaMis(Option_t *option)
{
  //"add": keep old canvas
  const TString opt(option);

  const Int_t layer = this->PrepareAdd(opt.Contains("add", TString::kIgnoreCase));
  const TString titleAdd = this->TitleAdd();

  const PlotMillePede *m = fPlotMp1;
  UInt_t nPlot = 0;
  for (UInt_t iPar = 0; iPar < kNpar; ++iPar) { // 
    TString sel("");
    this->AddIsFreeSel(sel, opt, iPar);
    fPlotMp1->AddBasicSelection(sel);
    fPlotMp2->AddBasicSelection(sel);
    
    const TString deltaMisPar(m->Parenth(this->DeltaMisPar(iPar)) += m->ToMumMuRad(iPar));
    const TString deltaName(m->Unique(Form("deltaMisPar%d", iPar))); // 
    TH1 *hD = fPlotMp1->CreateHist(deltaMisPar, sel, deltaName);
    if (0. == hD->GetEntries()) {
      delete hD;
      continue;
    }

    const TString deltaMisBySi1(this->DeltaMisParBySigma(iPar, fPlotMp1));
    const TString deltaBySiName1(m->Unique(Form("deltaMisPar%dRel1", iPar))); // 
    TString sel1Ok(fPlotMp1->ParSiOk(iPar));
    if (!sel.IsNull()) sel1Ok.Append(m->AndL() += sel);
    TH1 *hDs1 = fPlotMp1->CreateHist(deltaMisBySi1, sel1Ok, deltaBySiName1);

    const TString deltaMisBySi2(this->DeltaMisParBySigma(iPar, fPlotMp2));
    TString deltaBySiName2(m->Unique(Form("deltaMisPar%dRel2", iPar)));
    m->CopyAddBinning(deltaBySiName2, hDs1); 
    TString sel2Ok(fPlotMp2->ParSiOk(iPar));
    if (!sel.IsNull()) sel2Ok.Append(m->AndL() += sel);
    TH1 *hDs2 = fPlotMp1->CreateHist(deltaMisBySi2, sel2Ok, deltaBySiName2);
    
    const TString dlNm = m->DelName(iPar);
    hD->SetTitle(m->Name(iPar) += titleAdd
		 + Form(";|%s_{2}|-|%s_{1}| ",dlNm.Data(), dlNm.Data())
		 + m->Unit(iPar) += ";#parameters");

    hDs1->SetTitle(m->Name(iPar) += titleAdd
		   + Form(";(|%s_{2}|-|%s_{1}|)/#sigma%s", dlNm.Data(), dlNm.Data(),
			  ";#parameters"));

    fHistManager->AddHist(hD, layer);
    if (hDs1->GetEntries()) {
      fHistManager->AddHist(hDs1, layer+1, "by #sigma_{1}");
    } else delete hDs1;
    if (hDs2->GetEntries()) { // does DrawSame work if hDs1 was empty?
      fHistManager->AddHistSame(hDs2, layer+1, nPlot, "by #sigma_{2}");
    } else delete hDs2;

    ++nPlot;
  }

  fHistManager->Draw();

}

//_________________________________________________________________________________________________
void CompareMillePede::DrawParamDeltaMisVsLoc(Option_t *option)
{

  const TString opt(option);
  const Int_t layer = this->PrepareAdd(opt.Contains("add", TString::kIgnoreCase));
  const TString titleAdd = this->TitleAdd();

  const PlotMillePede *m = fPlotMp1;
  UInt_t nPlot = 0;
  for (UInt_t iPar = 0; iPar < kNpar; ++iPar) { // 
    TString sel("");
    this->AddIsFreeSel(sel, opt, iPar);
    fPlotMp1->AddBasicSelection(sel);
    fPlotMp2->AddBasicSelection(sel);

    const TString deltaMisPar(m->Parenth(this->DeltaMisPar(iPar)) += m->ToMumMuRad(iPar));

    const TString deltaNameR(m->Unique(Form("deltaMisPar%dR", iPar))); // 
    TH2 *hDr = fPlotMp1->CreateHist2D(fPlotMp1->RPos(fPlotMp1->OrgPosT()), deltaMisPar, sel,
                                      deltaNameR, "BOX");
    if (0. == hDr->GetEntries()) {
      delete hDr;
      continue;
    }
    const TString deltaNamePhi(m->Unique(Form("deltaMisPar%dPhi", iPar))); // 
    TH2 *hDphi = fPlotMp1->CreateHist2D(fPlotMp1->Phi(fPlotMp1->OrgPosT()), deltaMisPar, sel,
					deltaNamePhi, "BOX");
    const TString deltaNameZ(m->Unique(Form("deltaMisPar%dZ", iPar))); // 
    TH2 *hDz = fPlotMp1->CreateHist2D(fPlotMp1->OrgPosT() += fPlotMp1->ZPos(), deltaMisPar, sel,
                                      deltaNameZ, "BOX");
    const TString deltaNameNh1(m->Unique(Form("deltaMisPar%dNh1", iPar))); // 
    TH2 *hDnh1 = fPlotMp1->CreateHist2D(fPlotMp1->HitsX(), deltaMisPar, sel,
					deltaNameNh1, "BOX");
    const TString deltaNameNh2(m->Unique(Form("deltaMisPar%dNh2", iPar))); // 
    TH2 *hDnh2 = fPlotMp1->CreateHist2D(fPlotMp2->HitsX(), deltaMisPar, sel,
					deltaNameNh2, "BOX");

    // now devided by sigma prediction from 1
    const TString deltaMisBySi1(this->DeltaMisParBySigma(iPar, fPlotMp1));
    TString sel1Ok(fPlotMp1->ParSiOk(iPar));
    if (!sel.IsNull()) sel1Ok.Append(m->AndL() += sel);
    const TString deltaBySiName1R(m->Unique(Form("deltaMisPar%dRRel1", iPar))); // 
    TH2 *hDrS1 = fPlotMp1->CreateHist2D(fPlotMp1->RPos(fPlotMp1->OrgPosT()),
                                        deltaMisBySi1, sel1Ok, deltaBySiName1R, "BOX");
    const TString deltaBySiName1Phi(m->Unique(Form("deltaMisPar%dPhiRel1", iPar))); // 
    TH2 *hDphiS1 = fPlotMp1->CreateHist2D(fPlotMp1->Phi(fPlotMp1->OrgPosT()),
                                          deltaMisBySi1, sel1Ok, deltaBySiName1Phi, "BOX");
    const TString deltaBySiName1Z(m->Unique(Form("deltaMisPar%dZRel1", iPar))); // 
    TH2 *hDzS1 = fPlotMp1->CreateHist2D(fPlotMp1->OrgPosT() += fPlotMp1->ZPos(),
                                        deltaMisBySi1, sel1Ok, deltaBySiName1Z, "BOX");

    // same now devided by sigma prediction from 2 (but CreateHist from 1!)
    const TString deltaMisBySi2(this->DeltaMisParBySigma(iPar, fPlotMp2));
    TString sel2Ok(fPlotMp2->ParSiOk(iPar));
    if (!sel.IsNull()) sel2Ok.Append(m->AndL() += sel);
    const TString deltaBySiName2R(m->Unique(Form("deltaMisPar%dRRel2", iPar))); // 
    TH2 *hDrS2 = fPlotMp1->CreateHist2D(fPlotMp2->RPos(fPlotMp2->OrgPosT()),
                                        deltaMisBySi2, sel2Ok, deltaBySiName2R, "BOX");
    const TString deltaBySiName2Phi(m->Unique(Form("deltaMisPar%dPhiRel2", iPar))); // 
    TH2 *hDphiS2 = fPlotMp1->CreateHist2D(fPlotMp2->Phi(fPlotMp2->OrgPosT()),
                                          deltaMisBySi2, sel2Ok, deltaBySiName2Phi, "BOX");
    const TString deltaBySiName2Z(m->Unique(Form("deltaMisPar%dZRel2", iPar))); // 
    TH2 *hDzS2 = fPlotMp1->CreateHist2D(fPlotMp2->OrgPosT() += fPlotMp2->ZPos(),
                                        deltaMisBySi2, sel2Ok, deltaBySiName2Z, "BOX");


    const TString diff(Form("|%s_{2}|-|%s_{1}|",m->DelName(iPar).Data(),m->DelName(iPar).Data()));

    hDr->SetTitle(m->DelName(iPar) += " vs r" + titleAdd + ";r[cm];" + diff + ' ' + m->Unit(iPar));
    hDphi->SetTitle(m->DelName(iPar) += " vs #phi" + titleAdd + ";#phi;" + diff + ' '
		    + m->Unit(iPar));
    hDz->SetTitle(m->DelName(iPar) += " vs z" + titleAdd + ";z[cm];" + diff + ' ' + m->Unit(iPar));
    hDnh1->SetTitle(m->DelName(iPar) += " vs N_{1}(hit" + titleAdd + ";N_{1}(hit);"
		    + diff + ' ' + m->Unit(iPar));
    hDnh2->SetTitle(m->DelName(iPar) += " vs N_{2}(hit" + titleAdd + ";N_{2}(hit);"
		    + diff + ' ' + m->Unit(iPar));

    fHistManager->AddHist(hDr, layer + nPlot);
    fHistManager->AddHist(hDphi, layer + nPlot);
    fHistManager->AddHist(hDz, layer + nPlot);
    fHistManager->AddHist(hDnh1, layer + nPlot);
    fHistManager->AddHist(hDnh2, layer + nPlot);

    if (hDrS1->GetEntries()) {
      hDrS1->SetTitle(m->DelName(iPar) += "/#sigma vs r" + titleAdd + ";r[cm];(" + diff
		      + ")/#sigma");
      hDphiS1->SetTitle(m->DelName(iPar) += "/#sigma vs #phi" + titleAdd + ";#phi;(" + diff
			+ ")/#sigma"); 
      hDzS1->SetTitle(m->DelName(iPar) += "/#sigma vs z" + titleAdd + ";z[cm];(" + diff
		      + ")/#sigma");
      fHistManager->AddHist(hDrS1, layer + nPlot + 1, "by #sigma_{1}", "f");
      fHistManager->AddHist(hDphiS1, layer + nPlot + 1, "by #sigma_{1}", "f");
      fHistManager->AddHist(hDzS1, layer + nPlot + 1, "by #sigma_{1}", "f");
    } else {
      delete hDrS1;
      delete hDphiS1;
      delete hDzS1;
    }
    if (hDrS2->GetEntries()) {
      hDrS2->SetLineColor(kRed);
      hDphiS2->SetLineColor(kRed);
      hDzS2->SetLineColor(kRed);
      fHistManager->AddHistSame(hDrS2, layer + nPlot + 1, 0, "by #sigma_{2}", "f");
      fHistManager->AddHistSame(hDphiS2, layer + nPlot + 1, 1, "by #sigma_{2}", "f");
      fHistManager->AddHistSame(hDzS2, layer + nPlot + 1, 2, "by #sigma_{2}", "f");
    } else {
      delete hDrS2;
      delete hDphiS2;
      delete hDzS2;
    }
    nPlot += 2;
  }

  const bool oldDiffStyle = fHistManager->DrawDiffStyle(false);//avoid automatic hist style changes
  fHistManager->Draw();
  fHistManager->DrawDiffStyle(oldDiffStyle);
}

//_________________________________________________________________________________________________
void CompareMillePede::DrawNumHits(Option_t *option)
{
  const TString opt(option);
  const Int_t layer = this->PrepareAdd(opt.Contains("add", TString::kIgnoreCase));
  const TString titleAdd = this->TitleAdd();

  const PlotMillePede *m = fPlotMp1;

  TString sel;
  fPlotMp1->AddBasicSelection(sel);
  fPlotMp2->AddBasicSelection(sel);

  const TString nX1vs2(m->Unique("hitsX1vs2"));
  TH2 *hX1vs2 = fPlotMp1->CreateHist2D(fPlotMp1->HitsX(), fPlotMp2->HitsX(), sel, nX1vs2, "BOX");

  const TString nY1vs2(m->Unique("hitsY1vs2"));
  TH2 *hY1vs2 = fPlotMp1->CreateHist2D(fPlotMp1->HitsY(), fPlotMp2->HitsY(), sel, nY1vs2, "BOX");

  const TString nDiffX(m->Unique("deltaHitsX"));
  const TString diffX(m->Parenth(fPlotMp2->HitsX() += m->Min() += fPlotMp1->HitsX()));
  TH1 *hDiffX = fPlotMp1->CreateHist(diffX, sel, nDiffX);

  const TString nDiffY(m->Unique("deltaHitsY"));
  const TString diffY(m->Parenth(fPlotMp2->HitsY() += m->Min() += fPlotMp1->HitsY()));
  TH1 *hDiffY = fPlotMp1->CreateHist(diffY, sel, nDiffY);

  const TString ratioX(m->Parenth(fPlotMp2->HitsX() += m->Div() += fPlotMp1->HitsX()));
  const TString selRatioX(fPlotMp1->HitsX() += ">0" + m->AndL() += sel);
  const TString nRatioX(m->Unique("ratioHitsX"));
  TH1 *hRatioX = fPlotMp1->CreateHist(ratioX, selRatioX, nRatioX);

  const TString ratioY(m->Parenth(fPlotMp2->HitsY() += m->Div() += fPlotMp1->HitsY()));
  const TString selRatioY(fPlotMp1->HitsY() += ">0" + m->AndL() += sel);
  const TString nRatioY(m->Unique("ratioHitsY"));
  TH1 *hRatioY = fPlotMp1->CreateHist(ratioY, selRatioY, nRatioY);

//   const TString nDiffYvs1(m->Unique("deltaHitsYvs1"));
//   TH2 *hDiffYvs1 = fPlotMp1->CreateHist2D(diffY, fPlotMp1->HitsY(), sel, nDiffXvs1, "BOX");

  // diff vs location/hits
  const TString nDiffXvsR(m->Unique("deltaHitsXvsR"));
  TH2 *hDiffXvsR = fPlotMp1->CreateHist2D(diffX, fPlotMp1->RPos(fPlotMp1->OrgPosT()),
                                          sel, nDiffXvsR, "BOX");
  const TString nDiffXvsPhi(m->Unique("deltaHitsXvsPhi"));
  TH2 *hDiffXvsPhi = fPlotMp1->CreateHist2D(diffX, fPlotMp1->Phi(fPlotMp1->OrgPosT()),
                                            sel, nDiffXvsPhi, "BOX");
  const TString nDiffXvsZ(m->Unique("deltaHitsXvsZ"));
  TH2 *hDiffXvsZ = fPlotMp1->CreateHist2D(diffX, fPlotMp1->OrgPosT() += fPlotMp1->ZPos(),
                                          sel, nDiffXvsZ, "BOX");
  const TString nDiffXvsX(m->Unique("deltaHitsXvsX"));
  TH2 *hDiffXvsX = fPlotMp1->CreateHist2D(diffX, fPlotMp1->OrgPosT() += fPlotMp1->XPos(),
                                          sel, nDiffXvsX, "BOX");
  const TString nDiffXvsY(m->Unique("deltaHitsXvsY"));
  TH2 *hDiffXvsY = fPlotMp1->CreateHist2D(diffX, fPlotMp1->OrgPosT() += fPlotMp1->YPos(),
                                          sel, nDiffXvsY, "BOX");
  const TString nDiffXvs1(m->Unique("deltaHitsXvs1"));
  TH2 *hDiffXvs1 = fPlotMp1->CreateHist2D(diffX, fPlotMp1->HitsX(), sel, nDiffXvs1, "BOX");

  // ratio vs location/hits
  const TString nRatioXvsR(m->Unique("ratioHitsXvsR"));
  TH2 *hRatioXvsR = fPlotMp1->CreateHist2D(ratioX, fPlotMp1->RPos(fPlotMp1->OrgPosT()),
					   selRatioX, nRatioXvsR, "BOX");
  const TString nRatioXvsPhi(m->Unique("ratioHitsXvsPhi"));
  TH2 *hRatioXvsPhi = fPlotMp1->CreateHist2D(ratioX, fPlotMp1->Phi(fPlotMp1->OrgPosT()),
					     selRatioX, nRatioXvsPhi, "BOX");
  const TString nRatioXvsZ(m->Unique("ratioHitsXvsZ"));
  TH2 *hRatioXvsZ = fPlotMp1->CreateHist2D(ratioX, fPlotMp1->OrgPosT() += fPlotMp1->ZPos(),
					   selRatioX, nRatioXvsZ, "BOX");
  const TString nRatioXvsX(m->Unique("ratioHitsXvsX"));
  TH2 *hRatioXvsX = fPlotMp1->CreateHist2D(ratioX, fPlotMp1->OrgPosT() += fPlotMp1->XPos(),
					   selRatioX, nRatioXvsX, "BOX");
  const TString nRatioXvsY(m->Unique("ratioHitsXvsY"));
  TH2 *hRatioXvsY = fPlotMp1->CreateHist2D(ratioX, fPlotMp1->OrgPosT() += fPlotMp1->YPos(),
					   selRatioX, nRatioXvsY, "BOX");
  const TString nRatioXvs1(m->Unique("ratioHitsXvs1"));
  TH2 *hRatioXvs1 = fPlotMp1->CreateHist2D(ratioX, fPlotMp1->HitsX(), selRatioX, nRatioXvs1, "BOX");

  
  hX1vs2->SetTitle("#hits_{x}" + titleAdd + ";N_{hit,x}^{1};N_{hit,x}^{2}");
  hY1vs2->SetTitle("#hits_{y}" + titleAdd + ";N_{hit,y}^{1};N_{hit,y}^{2}");
  hDiffX->SetTitle("#Delta#hits_{x}" + titleAdd + ";N_{hit,x}^{2} - N_{hit,x}^{1};#alignables");
  hDiffY->SetTitle("#Delta#hits_{y}" + titleAdd + ";N_{hit,y}^{2} - N_{hit,y}^{1};#alignables");
//   hDiffYvs1->SetTitle("#Delta#hits_{y} vs #hits_{y}^{1}" + titleAdd
// 		      + ";N_{hit,y}^{2} - N_{hit,y}^{1};N_{hit,y}^{1}");
  hRatioX->SetTitle("ratio #hits_{x}" + titleAdd + ";N_{hit,x}^{2} / N_{hit,x}^{1};#alignables");
  hRatioY->SetTitle("ratio #hits_{y}" + titleAdd + ";N_{hit,y}^{2} / N_{hit,y}^{1};#alignables");

  hDiffXvsR->SetTitle("#Delta#hits_{x} vs r" + titleAdd + ";N_{hit,x}^{2} - N_{hit,x}^{1};r[cm]");
  hDiffXvsPhi->SetTitle("#Delta#hits_{x} vs #phi" +titleAdd+ ";N_{hit,x}^{2} - N_{hit,x}^{1};#phi");
  hDiffXvsZ->SetTitle("#Delta#hits_{x} vs z" + titleAdd + ";N_{hit,x}^{2} - N_{hit,x}^{1};z[cm]");
  hDiffXvsX->SetTitle("#Delta#hits_{x} vs x" + titleAdd + ";N_{hit,x}^{2} - N_{hit,x}^{1};x[cm]");
  hDiffXvsY->SetTitle("#Delta#hits_{x} vs y" + titleAdd + ";N_{hit,x}^{2} - N_{hit,x}^{1};y[cm]");
  hDiffXvs1->SetTitle("#Delta#hits_{x} vs #hits_{x}^{1}" + titleAdd
		      + ";N_{hit,x}^{2} - N_{hit,x}^{1};N_{hit,x}^{1}");

  hRatioXvsR->SetTitle("hit ratio vs r" + titleAdd + ";N_{hit,x}^{2} / N_{hit,x}^{1};r[cm]");
  hRatioXvsPhi->SetTitle("hit ratio vs #phi" +titleAdd+ ";N_{hit,x}^{2} / N_{hit,x}^{1};#phi");
  hRatioXvsZ->SetTitle("hit ratio vs z" + titleAdd + ";N_{hit,x}^{2} / N_{hit,x}^{1};z[cm]");
  hRatioXvsX->SetTitle("hit ratio vs x" + titleAdd + ";N_{hit,x}^{2} / N_{hit,x}^{1};x[cm]");
  hRatioXvsY->SetTitle("hit ratio vs y" + titleAdd + ";N_{hit,x}^{2} / N_{hit,x}^{1};y[cm]");
  hRatioXvs1->SetTitle("hit ratio vs #hits_{x}^{1}" + titleAdd
		       + ";N_{hit,x}^{2} / N_{hit,x}^{1};N_{hit,x}^{1}");

  fHistManager->AddHist(hX1vs2, layer);
  fHistManager->AddHist(hDiffX, layer);
  fHistManager->AddHist(hRatioX, layer);
  fHistManager->AddHist(hY1vs2, layer);
  fHistManager->AddHist(hDiffY, layer);
//   fHistManager->AddHist(hDiffYvs1, layer);
  fHistManager->AddHist(hRatioY, layer);

  fHistManager->AddHist(hDiffXvsR, layer+1);
  fHistManager->AddHist(hDiffXvsPhi, layer+1);
  fHistManager->AddHist(hDiffXvsZ, layer+1);
  fHistManager->AddHist(hDiffXvsX, layer+1);
  fHistManager->AddHist(hDiffXvsY, layer+1);
  fHistManager->AddHist(hDiffXvs1, layer+1);

  fHistManager->AddHist(hRatioXvsR, layer+2);
  fHistManager->AddHist(hRatioXvsPhi, layer+2);
  fHistManager->AddHist(hRatioXvsZ, layer+2);
  fHistManager->AddHist(hRatioXvsX, layer+2);
  fHistManager->AddHist(hRatioXvsY, layer+2);
  fHistManager->AddHist(hRatioXvs1, layer+2);

  fHistManager->Draw();
}

//_________________________________________________________________________________________________
void CompareMillePede::DrawAbsPos(Option_t *option)
{
//"start": at start (else end), "add": keep old canvas

  const TString opt(option);

  const Int_t layer = this->PrepareAdd(opt.Contains("add", TString::kIgnoreCase));
  const TString titleAdd = this->TitleAdd();

  const TString tree1(opt.Contains("start", TString::kIgnoreCase) ? fPlotMp1->OrgPosT() : fPlotMp1->PosT()); 
  const TString tree2(opt.Contains("start", TString::kIgnoreCase) ? fPlotMp2->OrgPosT() : fPlotMp2->PosT()); 

  const PlotMillePede *m = fPlotMp1;
  //UInt_t nPlot = 0;

  // const TString posNames[] = {"rphi", "r", "z", "phi", "x", "y"};

//   for (UInt_t iPos = 0; iPos < sizeof(posNames)/sizeof(posNames[0]); ++iPos) {
//     const TString &posName = posNames[iPos];
  for (UInt_t iPos = 0; iPos < 3; ++iPos) { // 
    TString sel("");
    //this->AddIsFreeSel(sel, opt, iPar);
    fPlotMp1->AddBasicSelection(sel);
    fPlotMp2->AddBasicSelection(sel);
    
//   // values: absolute positions and orientations; need a 'position tree' as argument or in front
//   TString Pos() const { return "Pos";}
//   TString Pos(UInt_t ui) const { return Pos() += Bracket(ui);}
//   TString XPos() const { return Pos(0);}
//   TString YPos() const { return Pos(1);}
//   TString ZPos() const { return Pos(2);}

//   TString DelPos(UInt_t ui, const TString &tree1, const TString &tree2) const;

//   // units and names for position strings (r, rphi, phi, x, y, z)
//   TString ToMumMuRad(const TString &pos) const;
//   TString Name(const TString &pos) const;

//   TString OrgPosT() const { return fOrgPos + Dot();} // nominal global positions
//   TString PosT() const { return fPos + Dot();} // aligned global positions


    const TString deltaName(m->Unique(Form("deltaPos%d", iPos))); // 
    //const TString deltaName(m->Unique("deltaPos" + posName)); // 
    TH1 *h = fPlotMp1->CreateHist(m->DelPos(iPos, tree2, tree1), sel, deltaName);
    if (0. == h->GetEntries()) {
      delete h;
      continue;
    }

    const TString diff(Form("%s_{2}-%s_{1}", m->NamePos(iPos).Data(), m->NamePos(iPos).Data()));
    h->SetTitle(diff + titleAdd + ";" + diff + " [cm];#alignables");

    fHistManager->AddHist(h, layer);
    //    ++nPlot;
  }

  fHistManager->Draw();
}

//_______________________________________________________________________________
void CompareMillePede::DrawSurfaceDeformations(Option_t *option, unsigned int firstPar,
					       unsigned int lastPar, const TString &whichOne)
{
   const TString opt(option);

   const Int_t layer = this->PrepareAdd(opt.Contains("add", TString::kIgnoreCase));
   const bool limits = opt.Contains("limit", TString::kIgnoreCase);
   const bool noVs = opt.Contains("novs", TString::kIgnoreCase);
   const bool noDiff = opt.Contains("noDiff", TString::kIgnoreCase);
   const TString titleAdd = this->TitleAdd();
   const PlotMillePede *m = fPlotMp1;

   UInt_t nPlot = 0;
   for (UInt_t iPar = firstPar; iPar <= lastPar; ++iPar) { // 
     TString sel("");
     fPlotMp1->AddBasicSelection(sel);
     fPlotMp2->AddBasicSelection(sel);
     sel += m->AndL() += m->Parenth(fPlotMp1->NumDeformValues(whichOne) += Form(">%u", iPar))
       += m->AndL() += m->Parenth(fPlotMp2->NumDeformValues(whichOne) += Form(">%u", iPar));

     const TString deform1(fPlotMp1->DeformValue(iPar, whichOne));
     const TString deform2(fPlotMp2->DeformValue(iPar, whichOne));
     const TString diff(m->Parenth(deform2 + m->Min() += deform1)
			+= m->ToMumMuRadSurfDef(iPar));
     TString hNameDiff(m->Unique(Form("hSurfDiff%s%u", whichOne.Data(), iPar)));
     if (limits) hNameDiff += Form("(101,%f,%f)", m->GetMaxDevDown(), m->GetMaxDevUp());

     // delta plot
     TH1 *hDiff = fPlotMp1->CreateHist(diff, sel, hNameDiff);
     if (0. == hDiff->GetEntries()) {
       delete hDiff;
       continue;
     }

     if (noDiff) {
       delete hDiff;
     } else {
       // titles
       const TString diffTit(m->NameSurfDef(iPar) += "_{2}-"
			     + m->NameSurfDef(iPar) += "_{1}");
       hDiff->SetTitle(diffTit + titleAdd + ";"
		       + diffTit + m->UnitSurfDef(iPar) +=";#parameters");
       fHistManager->AddHist(hDiff, layer);
     }

     if (!noVs) { // 2D plot
       TString hNameVs(m->Unique(Form("hSurfVs%s%u", whichOne.Data(), iPar)));
       if (limits) hNameVs += Form("(101,%f,%f, 101,%f,%f)", m->GetMaxDevDown(),
				   m->GetMaxDevUp(), m->GetMaxDevDown(), m->GetMaxDevUp());
       TH2 *hVs = fPlotMp1->CreateHist2D(deform1 + m->ToMumMuRadSurfDef(iPar),
					 deform2 + m->ToMumMuRadSurfDef(iPar), sel,
					 hNameVs, "BOX");
       hVs->SetTitle(m->NameSurfDef(iPar) += titleAdd + ";"
		   + m->NameSurfDef(iPar) += "_{1}" + m->UnitSurfDef(iPar) +=";"
		   + m->NameSurfDef(iPar) += "_{2}" + m->UnitSurfDef(iPar));

       fHistManager->AddHist(hVs, layer + !noDiff);
       // add a TLine
       const Double_t xMin = hVs->GetXaxis()->GetXmin();
       const Double_t xMax = hVs->GetXaxis()->GetXmax();
       TLine *line = new TLine(xMin, xMin, xMax, xMax);
       line->SetLineColor(kRed); line->SetLineWidth(2);
       fHistManager->AddObject(line, layer + !noDiff, nPlot);
     }

     ++nPlot;
   }
   
   fHistManager->Draw();
}
//_________________________________________________________________________________________________
//_________________________________________________________________________________________________
//_________________________________________________________________________________________________
bool CompareMillePede::IsConsistent()
{

  // Id and ObjId identify an alignable, so the trees are in same order if the difference
  // is always zero. We check this by checking mean and RMS of the difference.

  TH1 *hId = fPlotMp1->CreateHist(fPlotMp1->PosT() += "Id - " + fPlotMp2->PosT() += "Id", "");
  TH1 *hObjId = fPlotMp1->CreateHist(fPlotMp1->PosT()+="ObjId - " + fPlotMp2->PosT()+="ObjId", "");

  if (hId->GetMean() == 0. && hId->GetRMS() == 0. 
      && hObjId->GetMean() == 0. && hObjId->GetRMS() == 0.) {
    delete hId;
    delete hObjId;
    return true;
  } else {
    fHistManager->Clear();
    fHistManager->AddHist(hId);
    fHistManager->AddHist(hObjId);
    fHistManager->Draw();

    return false;
  }
}

//_________________________________________________________________________________________________
TString CompareMillePede::DeltaPar(UInt_t iPar) const
{
  // '2' - '1'
  return fPlotMp2->FinalMisAlignment(iPar) += fPlotMp1->Min() += fPlotMp1->FinalMisAlignment(iPar);
}

//_________________________________________________________________________________________________
TString CompareMillePede::DeltaParBySigma(UInt_t iPar, const PlotMillePede *sigmaSource) const
{
  
  return 
    sigmaSource->Parenth(this->DeltaPar(iPar)) += sigmaSource->Div() += sigmaSource->ParSi(iPar);
}

//_________________________________________________________________________________________________
TString CompareMillePede::DeltaMisPar(UInt_t iPar) const
{
  // '2' - '1'
  const PlotMillePede *m = fPlotMp1;
  return            m->Abs(fPlotMp2->FinalMisAlignment(iPar)) 
    += m->Min() += m->Abs(fPlotMp1->FinalMisAlignment(iPar));
}

//_________________________________________________________________________________________________
TString CompareMillePede::DeltaMisParBySigma(UInt_t iPar, const PlotMillePede *sigmaSource) const
{
  
  return sigmaSource->Parenth(this->DeltaMisPar(iPar)) 
    += sigmaSource->Div() += sigmaSource->ParSi(iPar);
}

//_________________________________________________________________________________________________
TString CompareMillePede::DeltaPos(UInt_t iPos) const
{
  TString pos1(fPlotMp1->Parenth(fPlotMp1->PosT() += fPlotMp1->Pos(iPos)));
  TString pos2(fPlotMp2->Parenth(fPlotMp2->PosT() += fPlotMp2->Pos(iPos)));

  return pos1 += fPlotMp1->Min() += pos2;
}

//_________________________________________________________________________________________________
void CompareMillePede::AddIsFreeSel(TString &sel, const TString &option, UInt_t iPar) const
{
  if (option.Contains("free1", TString::kIgnoreCase)) {
    if (sel.IsNull()) sel = fPlotMp1->Fixed(iPar, false);
    else        sel.Prepend(fPlotMp1->Fixed(iPar, false) += fPlotMp1->AndL());
  }
  if (option.Contains("free2", TString::kIgnoreCase)) {
    if (sel.IsNull()) sel = fPlotMp2->Fixed(iPar, false);
    else        sel.Prepend(fPlotMp2->Fixed(iPar, false) += fPlotMp2->AndL());
  }
}

//_________________________________________________________________________________________________
void CompareMillePede::SetSubDetId(Int_t subDetId)
{
  fPlotMp1->SetSubDetId(subDetId);
  fPlotMp2->SetSubDetId(subDetId);
}

//_________________________________________________________________________________________________
void CompareMillePede::AddSubDetId(Int_t subDetId)
{
  fPlotMp1->AddSubDetId(subDetId);
  fPlotMp2->AddSubDetId(subDetId);
}

//_________________________________________________________________________________________________
void CompareMillePede::SetAlignableTypeId(Int_t alignableTypeId)
{
  fPlotMp1->SetAlignableTypeId(alignableTypeId);
  fPlotMp2->SetAlignableTypeId(alignableTypeId);
}

//_________________________________________________________________________________________________
void CompareMillePede::SetHieraLevel(Int_t hieraLevel)
{
  fPlotMp1->SetHieraLevel(hieraLevel);
  fPlotMp2->SetHieraLevel(hieraLevel);
}

//_________________________________________________________________________________________________
void CompareMillePede::AddAdditionalSel(const char *selection)
{
  fPlotMp1->AddAdditionalSel(selection);
  fPlotMp2->AddAdditionalSel(selection);
}

//_________________________________________________________________________________________________
void CompareMillePede::AddAdditionalSel(const TString &xyzrPhiNhit, Float_t min, Float_t max)
{
  fPlotMp1->AddAdditionalSel(xyzrPhiNhit, min, max);
  fPlotMp2->AddAdditionalSel(xyzrPhiNhit, min, max);
}

//_________________________________________________________________________________________________
void CompareMillePede::ClearAdditionalSel ()
{
  fPlotMp1->ClearAdditionalSel();
  fPlotMp2->ClearAdditionalSel();
}

//_________________________________________________________________________________________________
void CompareMillePede::SetSurfDefDeltaBows(bool deltaBows)
{
  fPlotMp1->SetSurfDefDeltaBows(deltaBows);
  fPlotMp2->SetSurfDefDeltaBows(deltaBows);
}

//_________________________________________________________________________________________________
Int_t CompareMillePede::PrepareAdd(bool addPlots)
{
  if (addPlots) {
    return fHistManager->GetNumLayers();
  } else {
    fHistManager->Clear();
    return 0;
  }
}

//_________________________________________________________________________________________________
TString CompareMillePede::TitleAdd() const
{
  // try to avoid duplication of cuts in title

  // get title add from titleAdd for first
  TString titleAdd1 = fPlotMp1->TitleAdd();
  const TString title1(fPlotMp1->GetTitle());
  TString titleAdd1noTitle(titleAdd1);
  const Ssiz_t tit1pos = titleAdd1.First(title1);
  if (tit1pos != kNPOS) titleAdd1noTitle.Remove(tit1pos, title1.Length() + 2); //+2: ', '
  // now for second
  TString titleAdd2 = fPlotMp2->TitleAdd();
  const TString title2(fPlotMp2->GetTitle());
  TString titleAdd2noTitle(titleAdd2);
  const Ssiz_t tit2pos = titleAdd2.First(title2);
  if (tit2pos != kNPOS) titleAdd2noTitle.Remove(tit2pos, title2.Length() + 2); //+2: ', '

  if (titleAdd1noTitle != titleAdd2noTitle) {
    // shortened titleAdds different: keep full length, adding 1/2 info
    titleAdd1 +=  " (1), ";
    if (titleAdd2(0, 2) == ": ") titleAdd2.Remove(0, 2); // remove ": "
    titleAdd1 += titleAdd2 += " (2)";
    return titleAdd1;
  } else {
    // otherwise keep only one, add titles at the end
    return (titleAdd1noTitle += ", 1: " + title1 + ", 2: " + title2);
  }
}
