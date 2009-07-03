// Original Author: Gero Flucke
// last change    : $Date: 2009/06/26 13:39:29 $
// by             : $Author: flucke $

#include "PlotMillePede.h"
#include <TH1.h>
#include <TH2.h>
#include <TProfile.h>
#include <TArrow.h>
#include <TEllipse.h>
#include <TF1.h>
#include <TMath.h>
#include <TTree.h>

#include <TError.h>
#include <TROOT.h>
#include <TCanvas.h>

#include <iostream>

#include "GFUtils/GFHistManager.h"

////////////////////////////////////////////////////////////////////////////////////////////////////
PlotMillePede::PlotMillePede(const char *fileName, Int_t iter, Int_t hieraLevel, bool useDiff)
  : MillePedeTrees(fileName, iter), fHistManager(new GFHistManager), fHieraLevel(hieraLevel),
    fUseDiff(useDiff), fSubDetIds(), fAlignableTypeId(-1), fMaxDev(500.)
{
  fHistManager->SetLegendX1Y1X2Y2(0.14, 0.7, 0.45, 0.9);
}
////////////////////////////////////////////////////////////////////////////////////////////////////
PlotMillePede::PlotMillePede(const char *fileName, Int_t iter, Int_t hieraLevel, const char *treeNameAdd)
  : MillePedeTrees(fileName, iter, treeNameAdd),
    fHistManager(new GFHistManager), fHieraLevel(hieraLevel),
    fUseDiff(false), fSubDetIds(), fAlignableTypeId(-1), fMaxDev(500.)
{
  fHistManager->SetLegendX1Y1X2Y2(0.14, 0.7, 0.45, 0.9);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
PlotMillePede::~PlotMillePede()
{
  delete fHistManager;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
void PlotMillePede::DrawAll(Option_t *opt)
{
  
  const TString o(opt);
  bool wasBatch = fHistManager->SetBatch();
  fHistManager->Clear();
  
//   if (o.Contains("d", TString::kIgnoreCase)) this->DrawParamDiff(true);
  if (o.Contains("r", TString::kIgnoreCase)) this->DrawParamResult(true);
  if (o.Contains("o", TString::kIgnoreCase)) this->DrawOrigParam(true);
  if (o.Contains("g", TString::kIgnoreCase)) this->DrawGlobCorr(true);
  if (o.Contains("p", TString::kIgnoreCase)) this->DrawPull("add");
  if (o.Contains("m", TString::kIgnoreCase)) this->DrawMisVsLocation(true);
  if (o.Contains("e", TString::kIgnoreCase)) this->DrawErrorVsHit(true);
  if (o.Contains("h", TString::kIgnoreCase)) this->DrawHitMaps(true);
  
  fHistManager->SetBatch(wasBatch);
  if (!wasBatch) fHistManager->Draw();
}
////////////////////////////////////////////////////////////////////////////////////////////////////
// void PlotMillePede::DrawParamVsMisaligned()
// {

//   ::Info("PlotMillePede::DrawParamVsMisaligned", "does not work...");
//   TCanvas *can = new TCanvas;
//   can->Divide(2, 3);

//   for (UInt_t iPar = 0; iPar < kNpar; ++iPar) { // 
//     const TString dPar(Dot() += Par(iPar));
//     const TString orig();
//     const TString toMum(iPar < 3 ? "*10000" : "");
//     const TString hNameS(Form("start%d(100, -500, 500)", iPar));
//     const TString hNameA(Form("after%d(100, -500, 500)", iPar));
//     TH1 *hStart = this->CreateHist(Parenth(MisParT() += dPar) += toMum, "!" + Fixed(iPar), hNameS);
//     TH1 *hAfter = this->CreateHist(Parenth(MisParT() += dPar + Min() += ParT() += dPar) += toMum,
//                                    "!" + Fixed(iPar), hNameA);
//     can->cd(iPar+1);
//     hStart->Draw();
//     hAfter->Draw("SAME");
//   }
// }

////////////////////////////////////////////////////////////////////////////////////////////////////
void PlotMillePede::DrawParam(bool addPlots, const TString &sel)
{

  const Int_t layer = this->PrepareAdd(addPlots);

  const TString titleAdd = this->TitleAdd();
  UInt_t nPlot = 0;
  for (UInt_t iPar = 0; iPar < kNpar; ++iPar) { // 
    TString parSel(sel.Length() ? sel : Fixed(iPar, false));
    this->AddBasicSelection(parSel);
    const TString toMum(this->ToMumMuRad(iPar));
//     const TString hNameA(Form("after%d(100, -500, 500)", iPar)); // 
//     const TString hNameB(Form("before%d(100, -500, 500)", iPar)); // 
    const TString hNameA(this->Unique(Form("after%d", iPar))); // 
    const TString hNameB(this->Unique(Form("before%d", iPar))); // 
    TH1 *hAfter = this->CreateHist(FinalMisAlignment(iPar) += toMum, parSel, hNameA);
    TH1 *hBefore = this->CreateHist(Parenth(MisParT() += Par(iPar)) += toMum, parSel, hNameB);

    if (0. == hAfter->GetEntries()) continue;
    hBefore->SetTitle(DelName(iPar)+=titleAdd+";"+DelNameU(iPar)+=";#parameters");
    hAfter ->SetTitle(DelName(iPar)+=titleAdd+";"+DelNameU(iPar)+=";#parameters");
    fHistManager->AddHist(hBefore, layer, "before");
    fHistManager->AddHistSame(hAfter, layer, nPlot, "after");
    fHistManager->AddHist(static_cast<TH1*>(hAfter->Clone()), layer + 1, "after");
    fHistManager->AddHistSame(static_cast<TH1*>(hBefore->Clone()), layer + 1, nPlot, "before");
    ++nPlot;
  }

  fHistManager->Draw();
}

////////////////////////////////////////////////////////////////////////////////////////////////////
void PlotMillePede::DrawPedeParam(Option_t *option)
{

  const Int_t layer = this->PrepareAdd(TString(option).Contains("add", TString::kIgnoreCase));

  const TString titleAdd = this->TitleAdd();
  UInt_t nPlot = 0;
  for (UInt_t iPar = 0; iPar < kNpar; ++iPar) { // 
    TString parSel(Fixed(iPar, false) += AndL() += Valid(iPar));// += AndL() += HitsX() += "<500");
    this->AddBasicSelection(parSel);
    const TString toMum(this->ToMumMuRad(iPar));
    const TString hName(this->Unique(Form("pedePar%d", iPar))); // 
    TH1 *h = this->CreateHist(Parenth(MpT() += Par(iPar)) += toMum, parSel, hName);

    if (0. == h->GetEntries()) continue;

    // parameters vs each other
    TObjArray histsParVsPar;
    for (UInt_t iPar2 = iPar + 1; iPar2 < kNpar; ++iPar2) {
      const TString hNameVs(this->Unique(Form("pedePar%d_%d", iPar, iPar2))); // 
      const TString toMum2(this->ToMumMuRad(iPar2));
      TH1 *hVs = this->CreateHist2D(Parenth(MpT() += Par(iPar)) += toMum,
                                  Parenth(MpT() += Par(iPar2)) += toMum2, // Valid(iPar2??)
                                  Parenth(parSel) += AndL() += Fixed(iPar2, false), hNameVs, "BOX");
      if (0. == hVs->GetEntries()) continue;// delete hVs;
      else {
        hVs->SetTitle("pede: " + Name(iPar2) += " vs. " + Name(iPar) += titleAdd + ";"
                      + Name(iPar) += Unit(iPar) += ";" + Name(iPar2) += Unit(iPar2));
        histsParVsPar.Add(hVs);
      }
    }

    // parameters and errors
    const TString hNameBySi(this->Unique(Form("pedeParBySi%d", iPar))); // 
    TH1 *hBySi = this->CreateHist(Parenth(MpT() += Par(iPar)) += Div() += ParSi(iPar),
                                  parSel + AndL() += ParSiOk(iPar), hNameBySi);
    TH1 *hBySiInv = 0;
    if (hBySi->GetEntries() == 0.) {
      delete hBySi; hBySi = 0;
    } else {
      const TString hNameBySiInv(this->Unique(Form("pedeParBySiInv%d", iPar)) += "(100,-20,20)"); // 
      hBySiInv = this->CreateHist(ParSi(iPar) += Div() += Parenth(MpT() += Par(iPar)),
                                  parSel + AndL() += ParSiOk(iPar), hNameBySiInv);
    }

    // parameters vs hits
    const TString hNameH(this->Unique(Form("pedeParVsHits%d", iPar))); // 
    TH2 *hHits = this->CreateHist2D(HitsX(), Parenth(MpT()+=Par(iPar)) += toMum, parSel,
				    hNameH, "BOX");

    // parameters vs global correlation
    const TString hNameG(this->Unique(Form("pedeParVsGlob%d", iPar))); // 
    TH2 *hGlobCor = this->CreateHist2D(Cor(iPar), Parenth(MpT()+=Par(iPar)) += toMum, parSel,
				       hNameG, "BOX");

    h->SetTitle("determined pede " + Name(iPar) += titleAdd + ";"
		+ Name(iPar) += Unit(iPar) += ";#alignables");
    hHits->SetTitle("determined pede " + Name(iPar) += titleAdd + " vs #n(hit_{x});N_{hit,x};"
		+ Name(iPar) += Unit(iPar));
    hGlobCor->SetTitle("determined pede " + Name(iPar) += titleAdd + 
		       " vs glob. corr;Global Correlation;" + Name(iPar) += Unit(iPar));
    fHistManager->AddHist(h, layer);
    fHistManager->AddHist(hHits, layer+1);
    fHistManager->AddHist(hGlobCor, layer+2);
    fHistManager->AddHists(&histsParVsPar, layer+3);

    if (hBySi) {
      const TString namI(Name(iPar));
      hBySi->SetTitle("pede: " + namI + "/#sigma_{" + namI + "}" + titleAdd + ";"
                      + namI + Div() += Fun("#sigma", namI) += ";#alignables");
      fHistManager->AddHist(hBySi, layer+4);
      hBySiInv->SetTitle("pede: #sigma_{" + namI + "}/" + namI + titleAdd + ";"
                         + Fun("#sigma", namI) += Div() += namI + ";#alignables");
      fHistManager->AddHist(hBySiInv, layer+5);
    }
    ++nPlot;
  }

  fHistManager->Draw();
}

////////////////////////////////////////////////////////////////////////////////////////////////////
void PlotMillePede::DrawOrigParam(bool addPlots, const TString &sel)
{
  // all alignables, even fixed...
  const Int_t layer = this->PrepareAdd(addPlots);
  const TString titleAdd = this->TitleAdd();
  TString aSel(sel);
  this->AddBasicSelection(aSel);

  for (UInt_t iPar = 0; iPar < kNpar; ++iPar) { // 
    const TString toMum(this->ToMumMuRad(iPar));
//     const TString hNameB(Form("beforeO%d(100, -500, 500)", iPar)); // 
    const TString hNameB(this->Unique(Form("beforeO%d", iPar))); // 
    TH1 *hBefore = this->CreateHist(Parenth(MisParT() += Par(iPar)) += toMum,
                                    aSel, hNameB);
    if (0. == hBefore->GetEntries()) continue;
    hBefore->SetTitle(DelName(iPar)+=": misplacement" + titleAdd + ";" 
		      + DelNameU(iPar) += ";#parameters"); 
    fHistManager->AddHist(hBefore, layer);//, "before");
  }

  fHistManager->Draw();
}

////////////////////////////////////////////////////////////////////////////////////////////////////
void PlotMillePede::DrawOrigPos(bool addPlots, const TString &sel)
{

  const Int_t layer = this->PrepareAdd(addPlots);
  const TString titleAdd = this->TitleAdd();
  TString aSel(sel);
  this->AddBasicSelection(aSel);

  TH1 *hPhi = this->CreateHist(Phi(OrgPosT()), aSel, this->Unique("orgPhi"));
  TH1 *hR = this->CreateHist(RPos(OrgPosT()), aSel, this->Unique("orgR"));
  TH1 *hZ = this->CreateHist(OrgPosT() += ZPos(), aSel, this->Unique("orgZ"));

  hPhi->SetTitle("original position #phi" + titleAdd + ";#phi;#alignables"); 
  hR->SetTitle("original position r[cm]" + titleAdd + ";r;#alignables"); 
  hZ->SetTitle("original position z[cm]" + titleAdd + ";z;#alignables"); 

  fHistManager->AddHist(hPhi, layer);
  fHistManager->AddHist(hR, layer);
  fHistManager->AddHist(hZ, layer);

  fHistManager->Draw();
}

////////////////////////////////////////////////////////////////////////////////////////////////////
void PlotMillePede::DrawParamResult(bool addPlots)
{
  const Int_t layer = this->PrepareAdd(addPlots);

  const TString titleAdd = this->TitleAdd();
  UInt_t nPlot = 0;
  for (UInt_t iPar = 0; iPar < kNpar; ++iPar) { // 
    TString sel(Fixed(iPar,false));
    this->AddBasicSelection(sel);

    const TString toMu(this->ToMumMuRad(iPar));
    const TString finalMis(this->FinalMisAlignment(iPar) += toMu);
    const TString startMis(Parenth(MisParT() += Par(iPar)) += toMu);

    const TString hNameB(this->Unique(Form("before%d", iPar)) += 
			 Form("(101,-%f,%f)", fMaxDev, fMaxDev));
    TH1 *hBef = this->CreateHist(startMis, sel, hNameB);
    const TString hNameD(this->Unique(Form("end%d", iPar)) 
                         += Form("(%d,%f,%f)", hBef->GetNbinsX(), 
                                 hBef->GetXaxis()->GetXmin(), hBef->GetXaxis()->GetXmax()));
    TH1 *hEnd = this->CreateHist(finalMis, sel, hNameD);
    const TString hName2D(this->Unique(Form("vs%d", iPar)) += Form("(30,-%f,%f,30,-500,500)", fMaxDev, fMaxDev));
    TH1 *hVs = this->CreateHist(startMis + ":" + finalMis, sel, hName2D, "BOX");
    if (0. == hEnd->GetEntries()) continue;
    hEnd->SetTitle(DelName(iPar)+=titleAdd+";"+DelNameU(iPar)+=";#parameters");
    hVs->SetTitle(DelName(iPar)+=titleAdd+";" + DelNameU(iPar)+="(end);" + DelNameU(iPar) 
		  += "(start)");
    fHistManager->AddHist(hEnd, layer, "remaining misal."); //"diff. to misal.");
    fHistManager->AddHistSame(hBef, layer, nPlot, "misaligned");
    fHistManager->AddHist(hVs, layer+1);

    ++nPlot;
  }

  fHistManager->Draw();
}

////////////////////////////////////////////////////////////////////////////////////////////////////
void PlotMillePede::DrawPosResult(bool addPlots, const TString &selection)
{
  const Int_t layer = this->PrepareAdd(addPlots);
  TString sel(selection);
  this->AddBasicSelection(sel);

  const TString posNames[] = {"rphi", "r", "z", "phi", "x", "y"};
  
  const TString titleAdd = this->TitleAdd();
  UInt_t nPlot = 0;
  for (UInt_t iPos = 0; iPos < sizeof(posNames)/sizeof(posNames[0]); ++iPos) {
    const TString &posName = posNames[iPos];

    const TString toMu(this->ToMumMuRad(posName));
    const TString hNameB(this->Unique(posName + "Before")
			 += Form("(101,-%f,%f)", fMaxDev, fMaxDev));
    const TString misPos(Parenth(DeltaPos(posName, MisPosT())) += toMu);
    TH1 *hBef = this->CreateHist(misPos, sel, hNameB);
    if (0. == hBef->GetEntries()) {
      delete hBef;
      continue;
    }
    TString hNameD(this->Unique(posName + "End"));
    this->CopyAddBinning(hNameD, hBef);
    const TString endPos(Parenth(DeltaPos(posName, PosT())) += toMu);
    TH1 *hEnd = this->CreateHist(endPos, sel, hNameD);
    const TString hName2D(this->Unique(posName + "Vs(30,-100,100,30,-500,500)"));
    TH1 *hVs = this->CreateHist2D(endPos, misPos, sel, hName2D, "BOX");

    hEnd->SetTitle(DelName(posName)+=titleAdd+";"+DelNameU(posName)+=";#alignables");
    hVs->SetTitle(DelName(posName)+=titleAdd+";" + DelNameU(posName)+="(end);" 
		  + DelNameU(posName) += "(start)");

    fHistManager->AddHist(hVs, layer);
    fHistManager->AddHist(hEnd, layer + 1, "remaining misal.");
    fHistManager->AddHistSame(hBef, layer +1, nPlot, "misaligned");
    
    ++nPlot;
  }

  fHistManager->Draw();
}

////////////////////////////////////////////////////////////////////////////////////////////////////
void PlotMillePede::DrawMisVsLocation(bool addPlots, const TString &sel, Option_t *option)
{
  const Int_t layer = this->PrepareAdd(addPlots);

  TString opt(option);
  opt.ToLower();
  int vsEuler = -1;
  if (opt.Contains("vse0")) vsEuler = 0; // also vs euler angle, definition a)...
  else if (opt.Contains("vse1")) vsEuler = 1; //... or b)
  // profile of starting misalignment? (But not for euler stuff...)
  const bool addStartMis = (opt.Contains("mis") ? true : false);
  const bool addFixed = opt.Contains("withfixed");

  const TString titleAdd = this->TitleAdd();
  UInt_t nPlot = 0;
  for (UInt_t iPar = 0; iPar < kNpar; ++iPar) { // 
    TString fixSel(sel);
    if (!addFixed) {
      if (fixSel.Length()) fixSel.Prepend(Fixed(iPar, false) += AndL());
      else fixSel = Fixed(iPar, false);
    }
    this->AddBasicSelection(fixSel);

    const TString misPar(this->FinalMisAlignment(iPar) += ToMumMuRad(iPar));
    const TString startMisPar(MisParT() += Par(iPar) += ToMumMuRad(iPar));

    const TString nDpr(this->Unique(Form("misParR%d", iPar))); //  (100, -500, 500)
    TH2 *hMisParR = this->CreateHist2D(RPos(OrgPosT()), misPar, fixSel, nDpr, "BOX");
    TProfile *hProfParR = this->CreateHistProf(RPos(OrgPosT()), misPar, fixSel, "prof" + nDpr);
    if (0. == hMisParR->GetEntries()) continue;
    //hMisParR->RebinX(2);
    //    TH2 *hMisParStartR = this->CreateHist2D(RPos(OrgPosT()), startMisPar, fixSel, "start"+nDpr, "BOX");
    TProfile *hProfParStartR = !addStartMis ? 0 :
      this->CreateHistProf(RPos(OrgPosT()), startMisPar, fixSel, "profStart" + nDpr);

    const TString nDpz(this->Unique(Form("misParZ%d", iPar))); //  (100, -500, 500)
    TH2 *hMisParZ = this->CreateHist2D(OrgPosT() += ZPos(), misPar, fixSel, nDpz, "BOX");
    TProfile *hProfParZ = this->CreateHistProf(OrgPosT() += ZPos(), misPar, fixSel, "prof" + nDpz);
    //hMisParZ->RebinX(2);
    TProfile *hProfParStartZ = !addStartMis ? 0 :
      this->CreateHistProf(OrgPosT() += ZPos(), startMisPar, fixSel, "profStart" + nDpz);

    const TString nDpp(this->Unique(Form("misParPhi%d", iPar))); //  (100, -500, 500)
    TH2 *hMisParPhi = this->CreateHist2D(Phi(OrgPosT()), misPar, fixSel, nDpp, "BOX");
    TProfile *hProfParPhi = this->CreateHistProf(Phi(OrgPosT()), misPar, fixSel, "prof" + nDpp);
    //hMisParPhi->RebinX(2);
    TProfile *hProfParStartPhi = !addStartMis ? 0 :
      this->CreateHistProf(Phi(OrgPosT()), startMisPar, fixSel, "profStart" + nDpp);

//     const TString nDpth(this->Unique(Form("misParTheta%d", iPar))); //  (100, -500, 500)
//     TH2 *hMisParTheta = this->CreateHist2D(Theta(OrgPosT()), misPar, fixSel, nDpth, "BOX");
//     TProfile *hProfParTheta = this->CreateHistProf(Theta(OrgPosT()), misPar, fixSel, "prof"+nDpth);
//     //hMisParTheta->RebinX(2);
//     TProfile *hProfParStartTheta = !addStartMis ? 0 :
//       this->CreateHistProf(Theta(OrgPosT()), startMisPar, fixSel, "profStart" + nDpth);

    // euler angles with 0
    const TString nDpa0(this->Unique(Form("misParAlpha%d%d", vsEuler, iPar))); //  (100, -500, 500)
    TH2 *hMisParAl0 = 0;
    if (vsEuler >=0) hMisParAl0 = 
      this->CreateHist2D(Alpha(OrgPosT(), (vsEuler==0)), misPar, fixSel, nDpa0, "BOX");
    TProfile *hProfParAl0 = 0;
    if (vsEuler >=0) hProfParAl0 = 
      this->CreateHistProf(Alpha(OrgPosT(), (vsEuler==0)), misPar, fixSel, "prof" + nDpa0);
    //hMisParAl0->RebinX(2);

    const TString nDpb0(this->Unique(Form("misParBeta%d%d", vsEuler, iPar))); //  (100, -500, 500)
    TH2 *hMisParBet0 = 0;
    if (vsEuler >=0) hMisParBet0 = 
      this->CreateHist2D(Beta(OrgPosT(), (vsEuler==0)), misPar, fixSel, nDpb0, "BOX");
    TProfile *hProfParBet0 = 0;
    if (vsEuler >=0) hProfParBet0 = 
      this->CreateHistProf(Beta(OrgPosT(), (vsEuler==0)), misPar, fixSel, "prof" + nDpb0);
    //hMisParBet0->RebinX(2);

    const TString nDpg0(this->Unique(Form("misParGamma%d%d", vsEuler, iPar))); //  (100, -500, 500)
    TH2 *hMisParGam0 = 0;
    if (vsEuler >=0) hMisParGam0 =
      this->CreateHist2D(Gamma(OrgPosT(), (vsEuler==0)), misPar, fixSel, nDpg0, "BOX");
    TProfile *hProfParGam0 = 0;
    if (vsEuler >=0) hProfParGam0 = 
      this->CreateHistProf(Gamma(OrgPosT(), (vsEuler==0)), misPar, fixSel, "prof" + nDpg0);
    //hMisParGam0->RebinX(2);

    hMisParR->SetTitle(DelName(iPar) += " vs. r" + titleAdd + ";r[cm];" + DelNameU(iPar));
    // hMisParStartR->SetTitle(DelName(iPar) += " vs. r (start);r[cm];" + DelNameU(iPar));
    hMisParZ->SetTitle(DelName(iPar) += " vs. z" + titleAdd + ";z[cm];" + DelNameU(iPar));
    hMisParPhi->SetTitle(DelName(iPar) += + " vs. #phi" + titleAdd + ";#phi;" + DelNameU(iPar));
//     hMisParTheta->SetTitle(DelName(iPar) += " vs. #theta" + titleAdd + ";#theta;" + DelNameU(iPar));
    if (hMisParAl0)
      hMisParAl0->SetTitle(DelName(iPar) += Form(" vs. euler #alpha^{%d};#alpha^{%d};",
						  vsEuler, vsEuler) + DelNameU(iPar));
    if (hMisParBet0)
      hMisParBet0->SetTitle(DelName(iPar) += Form(" vs. euler #beta^{%d};#beta^{%d};",
						   vsEuler, vsEuler) + DelNameU(iPar));
    if (hMisParGam0)
      hMisParGam0->SetTitle(DelName(iPar) += Form(" vs. euler #gamma^{%d};#gamma^{%d};",
						   vsEuler, vsEuler) + DelNameU(iPar));
    hProfParR->SetTitle("<" + DelName(iPar) += "> vs. r" + titleAdd + ";r[cm];" + DelNameU(iPar));
    hProfParZ->SetTitle("<" + DelName(iPar) += "> vs. z" + titleAdd + ";z[cm];" + DelNameU(iPar));
    hProfParPhi->SetTitle("<" + DelName(iPar) += "> vs. #phi" + titleAdd + ";#phi;" + DelNameU(iPar));
//     hProfParTheta->SetTitle("<" + DelName(iPar) += "> vs. #theta" + titleAdd + ";#theta;" + DelNameU(iPar));
    if (hProfParAl0)
      hProfParAl0->SetTitle("<" + DelName(iPar) += Form("> vs. euler #alpha^{%d};#alpha^{%d};",
							vsEuler, vsEuler) + DelNameU(iPar));
    if (hProfParBet0)
      hProfParBet0->SetTitle("<" + DelName(iPar) += Form("> vs. euler #beta^{%d};#beta^{%d};",
							 vsEuler, vsEuler) + DelNameU(iPar));
    if (hProfParGam0)
      hProfParGam0->SetTitle("<" + DelName(iPar) += Form("> vs. euler #gamma^{%d};#gamma^{%d};",
							 vsEuler, vsEuler) + DelNameU(iPar));
    if (addStartMis) {
      hProfParStartR->SetTitle("<" + DelName(iPar) += "> vs. r (start);r[cm];" + DelNameU(iPar));
      hProfParStartZ->SetTitle("<" + DelName(iPar) += "> vs. z;z[cm];" + DelNameU(iPar));
      hProfParStartPhi->SetTitle("<" + DelName(iPar) += "> vs. #phi;#phi;" + DelNameU(iPar));
//       hProfParStartTheta->SetTitle("<" + DelName(iPar) += "> vs. #theta;#theta;" + DelNameU(iPar));
    }

    fHistManager->AddHist(hMisParR, layer+nPlot);//, "diff. to misal.");
    fHistManager->AddHist(hProfParR, layer+nPlot);//, "diff. to misal.");
    if (addStartMis) fHistManager->AddHist(hProfParStartR, layer+nPlot);//, "diff. to misal.");

    fHistManager->AddHist(hMisParZ, layer+nPlot);//, "misaligned");
    fHistManager->AddHist(hProfParZ, layer+nPlot);//, "misaligned");
    if (addStartMis) fHistManager->AddHist(hProfParStartZ, layer+nPlot);//, "misaligned");

    fHistManager->AddHist(hMisParPhi, layer+nPlot);//, "misaligned");
    fHistManager->AddHist(hProfParPhi, layer+nPlot);//, "misaligned");
    if (addStartMis) fHistManager->AddHist(hProfParStartPhi, layer+nPlot);//, "misaligned");

//     fHistManager->AddHist(hMisParTheta, layer+nPlot);//, "misaligned");
//     fHistManager->AddHist(hProfParTheta, layer+nPlot);//, "misaligned");
//     if (addStartMis) fHistManager->AddHist(hProfParStartTheta, layer+nPlot);//, "misaligned");

    if (hMisParAl0) fHistManager->AddHist(hMisParAl0, layer+nPlot+1);
    if (hProfParAl0) fHistManager->AddHist(hProfParAl0, layer+nPlot+1);

    if (hMisParBet0) fHistManager->AddHist(hMisParBet0, layer+nPlot+1);
    if (hProfParBet0) fHistManager->AddHist(hProfParBet0, layer+nPlot+1);
	            
    if (hMisParGam0) fHistManager->AddHist(hMisParGam0, layer+nPlot+1);
    if (hProfParGam0) fHistManager->AddHist(hProfParGam0, layer+nPlot+1);

    fHistManager->SetNumHistsXY((addStartMis ? 3 : 2), 4, layer+nPlot);
    if (vsEuler >= 0) {
      fHistManager->SetNumHistsXY(2, 3, layer+nPlot+1);
      ++nPlot;
    }
    ++nPlot;
  }

  fHistManager->Draw();
}

////////////////////////////////////////////////////////////////////////////////////////////////////
void PlotMillePede::DrawPosMisVsLocation(bool addPlots, const TString &selection, Option_t *option)
{
  const Int_t layer = this->PrepareAdd(addPlots);

  TString opt(option); opt.ToLower();
  const bool addStart = (opt.Contains("start") ? true : false);// profile of starting misalignment?

  TString sel(selection);
  this->AddBasicSelection(sel);

  const TString posNames[] = {"rphi", "r", "z", "phi", "x", "y"};

  const TString titleAdd = this->TitleAdd();
  UInt_t nPlot = 0;
  for (UInt_t iPos = 0; iPos < sizeof(posNames)/sizeof(posNames[0]); ++iPos) {
    const TString &posName = posNames[iPos];

    const TString toMu(this->ToMumMuRad(posName));
    const TString startPos(Parenth(DeltaPos(posName, MisPosT())) += toMu);
    const TString endPos(Parenth(DeltaPos(posName, PosT())) += toMu);

    const TString nDpr(this->Unique(posName + "MisPosR")); //  (100, -500, 500)
    TH2 *hEndPosR = this->CreateHist2D(RPos(OrgPosT()), endPos, sel, nDpr, "BOX");
    if (0. == hEndPosR->GetEntries()) {
      delete hEndPosR;
      continue;
    }
    TProfile *hProfPosR = this->CreateHistProf(RPos(OrgPosT()), endPos, sel, "prof" + nDpr);
    TProfile *hProfPosStartR = !addStart ? 0 :
      this->CreateHistProf(RPos(OrgPosT()), startPos, sel, "profStart" + nDpr);

    const TString nDpz(this->Unique(posName + "MisPosZ")); //  (100, -500, 500)
    TH2 *hEndPosZ = this->CreateHist2D(OrgPosT() += ZPos(), endPos, sel, nDpz, "BOX");
    TProfile *hProfPosZ = this->CreateHistProf(OrgPosT() += ZPos(), endPos, sel, "prof" + nDpz);
    TProfile *hProfPosStartZ = !addStart ? 0 :
      this->CreateHistProf(OrgPosT() += ZPos(), startPos, sel, "profStart" + nDpz);

    const TString nDpp(this->Unique(posName + "MisPosPhi")); //  (100, -500, 500)
    TH2 *hEndPosPhi = this->CreateHist2D(Phi(OrgPosT()), endPos, sel, nDpp, "BOX");
    TProfile *hProfPosPhi = this->CreateHistProf(Phi(OrgPosT()), endPos, sel, "prof" + nDpp);
    TProfile *hProfPosStartPhi = !addStart ? 0 :
      this->CreateHistProf(Phi(OrgPosT()), startPos, sel, "profStart" + nDpp);

    const TString nDpy(this->Unique(posName + "MisPosY")); //  (100, -500, 500)
    TH2 *hEndPosY = this->CreateHist2D(OrgPosT()+=YPos(), endPos, sel, nDpy, "BOX");
    TProfile *hProfPosY = this->CreateHistProf(OrgPosT()+=YPos(), endPos, sel, "prof" + nDpy);
    TProfile *hProfPosStartY = !addStart ? 0 :
      this->CreateHistProf(OrgPosT()+=YPos(), startPos, sel, "profStart" + nDpy);

    hEndPosR->SetTitle(DelName(posName) += " vs. r" + titleAdd + ";r[cm];" + DelNameU(posName));
    hEndPosZ->SetTitle(DelName(posName) += " vs. z" + titleAdd + ";z[cm];" + DelNameU(posName));
    hEndPosPhi->SetTitle(DelName(posName) += " vs. #phi" + titleAdd + ";#phi;" + DelNameU(posName));
    hEndPosY->SetTitle(DelName(posName) += " vs. y" + titleAdd + ";y[cm];" + DelNameU(posName));
    hProfPosR->SetTitle("<" + DelName(posName) += "> vs. r" + titleAdd + ";r[cm];" + DelNameU(posName));
    hProfPosZ->SetTitle("<" + DelName(posName) += "> vs. z" + titleAdd + ";z[cm];" + DelNameU(posName));
    hProfPosPhi->SetTitle("<" + DelName(posName) += "> vs. #phi" + titleAdd + ";#phi;" + DelNameU(posName));
    hProfPosY->SetTitle("<" + DelName(posName) += "> vs. y" + titleAdd + ";y[cm];" + DelNameU(posName));
    if (addStart) {
      hProfPosStartR->SetTitle("<" + DelName(posName)+="> vs. r (start)" + titleAdd + ";r[cm];"+DelNameU(posName));
      hProfPosStartZ->SetTitle("<" + DelName(posName) += "> vs. z" + titleAdd + ";z[cm];" + DelNameU(posName));
      hProfPosStartPhi->SetTitle("<" + DelName(posName) += "> vs. #phi" + titleAdd + ";#phi;" + DelNameU(posName));
      hProfPosStartY->SetTitle("<" + DelName(posName) += "> vs. y" + titleAdd + ";y[cm];" + DelNameU(posName));
    }

    fHistManager->AddHist(hEndPosR, layer+nPlot);//, "diff. to misal.");
    fHistManager->AddHist(hProfPosR, layer+nPlot);//, "diff. to misal.");
    if (addStart) fHistManager->AddHist(hProfPosStartR, layer+nPlot);//, "diff. to misal.");

    fHistManager->AddHist(hEndPosZ, layer+nPlot);//, "misaligned");
    fHistManager->AddHist(hProfPosZ, layer+nPlot);//, "misaligned");
    if (addStart) fHistManager->AddHist(hProfPosStartZ, layer+nPlot);//, "misaligned");

    fHistManager->AddHist(hEndPosPhi, layer+nPlot);//, "misaligned");
    fHistManager->AddHist(hProfPosPhi, layer+nPlot);//, "misaligned");
    if (addStart) fHistManager->AddHist(hProfPosStartPhi, layer+nPlot);//, "misaligned");

    fHistManager->AddHist(hEndPosY, layer+nPlot);//, "misaligned");
    fHistManager->AddHist(hProfPosY, layer+nPlot);//, "misaligned");
    if (addStart) fHistManager->AddHist(hProfPosStartY, layer+nPlot);//, "misaligned");

    fHistManager->SetNumHistsXY((addStart ? 3 : 2), 4, layer+nPlot);

    ++nPlot;
  }

  fHistManager->Draw();
}

////////////////////////////////////////////////////////////////////////////////////////////////////
void PlotMillePede::DrawLabelDiffAbove(UInt_t iPar, float minDiff, bool addPlots)
{
  const Int_t layer = this->PrepareAdd(addPlots);
  const TString titleAdd = this->TitleAdd();
  const TString misPar(this->FinalMisAlignment(iPar) += ToMumMuRad(iPar));

  TString fixSel(Fixed(iPar, false) += AndL() += Abs(misPar)//+=Div()+=ParSi(iPar)) 
                 += Form(">%f", minDiff));
  this->AddBasicSelection(fixSel);

  const TString name(this->Unique(Form("labelBigMis%d(100000,0,100000)", iPar)));

  TH1 *hLabel = this->CreateHist(MpT() += "Label", fixSel, name);
  //  this->GetMainTree()->Scan("Id:" + MpT() += "Label:" + misPar, fixSel);

  if (0. == hLabel->GetEntries()) return;
  
  hLabel->SetTitle("Label, " + Abs(DelNameU(iPar)) += Form(">%f", minDiff) + titleAdd + ";label");

  fHistManager->AddHist(hLabel, layer);

  fHistManager->Draw();
}

////////////////////////////////////////////////////////////////////////////////////////////////////
void PlotMillePede::DrawGlobCorr(bool addPlots, const TString &sel, Option_t *option,
				 Float_t min, Float_t max)
{
  // options:
  // "nal": no axis limit
  const TString opt(option);
  const bool nal = opt.Contains("nal", TString::kIgnoreCase);
  const Int_t layer = this->PrepareAdd(addPlots);
  const TString titleAdd = this->TitleAdd();

  UInt_t nPlot = 0;
  for (UInt_t iPar = 0; iPar < kNpar; ++iPar) { // 
    TString aSel(sel);
    if (aSel.Length()) aSel.Prepend(Fixed(iPar, false) += AndL());
    else aSel = Fixed(iPar, false);
    if (opt.Contains("valid", TString::kIgnoreCase)) {
      if (aSel.Length()) aSel.Prepend(Valid(iPar) += AndL());
      else aSel = Valid(iPar);
    }

    this->AddBasicSelection(aSel);

    const TString limits(nal ? "" : Form("(100,%f,%f)", min, max)); 
    const TString hName(this->Unique(Form("globCor%d", iPar)) += limits);
    TH1 *h = this->CreateHist(Cor(iPar), aSel, hName);
    if (0. == h->GetEntries()) continue;

    const TString limits2dR(nal ? "" : Form("(110,0,110, 100,%f,%f)", min, max));
    const TString hNameR(this->Unique(Form("rGlobCor%d", iPar)) += limits2dR);
    TH1 *hR = this->CreateHist2D(RPos(OrgPosT()), Cor(iPar), aSel, hNameR, "BOX");

    const TString limits2dZ(nal ? "" : Form("(110,-275,275,100,%f,%f)", min, max));
    const TString hNameZ(this->Unique(Form("zGlobCor%d", iPar)) += limits2dZ);
    TH1 *hZ = this->CreateHist2D(OrgPosT() += ZPos(), Cor(iPar), aSel, hNameZ, "BOX");

    const TString limits2dPhi(nal ? "" : Form("(100,-3.15,3.15,100,%f,%f)", min, max));
    const TString hNamePhi(this->Unique(Form("phiGlobCor%d", iPar)) += limits2dPhi);
    TH1 *hPhi=this->CreateHist2D(Phi(OrgPosT()), Cor(iPar), aSel, hNamePhi, "BOX");

    h->SetTitle(this->DelName(iPar) += titleAdd + ";Global Correlation;#parameters");
    hR->SetTitle(this->DelName(iPar) += titleAdd + ";r[cm];Global Correlation");
    hZ->SetTitle(this->DelName(iPar) += titleAdd + ";z[cm];Global Correlation");
    hPhi->SetTitle(this->DelName(iPar) += titleAdd + ";#phi;Global Correlation");
    fHistManager->AddHist(h, layer);
    fHistManager->AddHist(hR, layer+nPlot+1);
    fHistManager->AddHist(hZ, layer+nPlot+1);
    fHistManager->AddHist(hPhi, layer+nPlot+1);
    ++nPlot;
  }

  fHistManager->Draw();
}

////////////////////////////////////////////////////////////////////////////////////////////////////
void PlotMillePede::DrawPull(Option_t *option)
{
  const TString opt(option);
  const Int_t layer = this->PrepareAdd(opt.Contains("add", TString::kIgnoreCase));

  const TString titleAdd = this->TitleAdd();
  UInt_t nPlot = 0;
  for (UInt_t iPar = 0; iPar < kNpar; ++iPar) {
    const TString misPar(this->FinalMisAlignment(iPar));

    TString sel(Fixed(iPar,false));
    if (opt.Contains("valid", TString::kIgnoreCase)) {
      sel += AndL() += Valid(iPar);
    }
    this->AddBasicSelection(sel);

    const TString hNameS(this->Unique(Form("sigma%d", iPar)));
    TH1 *hSi = this->CreateHist(ParSi(iPar)+= ToMumMuRad(iPar), sel, hNameS);
    const TString hNameP(this->Unique(Form("pull%d", iPar)) 
			 += (opt.Contains("nolimit", TString::kIgnoreCase) ? "" : "(100,-6,6)"));
    TH1 *hPull = this->CreateHist(misPar + Div() += ParSi(iPar), sel, hNameP);
    if (0. == hPull->GetEntries()) continue;

    hPull->SetTitle("pull " + DelName(iPar) += titleAdd + ";#Delta/" 
		    + Fun("#sigma", DelName(iPar)) += ";#parameters");
    hPull->Fit("gaus", "Q0L"); // "0": do not directly draw, "Q": quiet, "L" likelihood for bin=0 treatment
    hPull->GetFunction("gaus")->ResetBit(TF1::kNotDraw);
    hSi->SetTitle(DelName(iPar) += titleAdd + ";" 
		  + Fun("#sigma", DelName(iPar)) += Unit(iPar) += ";#parameters");

    fHistManager->AddHist(hPull, layer); //, 0, "diff. to misal.");
    fHistManager->AddHist(hSi, layer+1); //, 0, "diff. to misal.");
    ++nPlot;
  }

  fHistManager->Draw();
}

////////////////////////////////////////////////////////////////////////////////////////////////////
void PlotMillePede::DrawErrorVsHit(bool addPlots, const TString &sel)
{
  const Int_t layer = this->PrepareAdd(addPlots);
  const TString titleAdd = this->TitleAdd();

  TString andSel(sel);
  if (andSel.Length()) andSel.Prepend(AndL());

  UInt_t nPlot = 0;
  for (UInt_t iPar = 0; iPar < kNpar; ++iPar) { // 
    TString aSel(sel);
    if (aSel.Length()) aSel.Prepend(Fixed(iPar, false) += AndL());
    else aSel = Fixed(iPar, false);
    this->AddBasicSelection(aSel);

    const TString toMum(this->ToMumMuRad(iPar));
    const TString nMiPaHit(this->Unique(Form("mis%dVsHit", iPar)));
    const TString misPar(this->FinalMisAlignment(iPar) += toMum);

    TH2 *hDParVsHit = this->CreateHist2D(misPar, HitsX(), aSel, nMiPaHit, "BOX");
    if (0. == hDParVsHit->GetEntries()) continue;

    const TString nSiHit(this->Unique(Form("sigmaVsHit%d", iPar)));
    TH2 *hSiVsHit = this->CreateHist2D(ParSi(iPar) += toMum, HitsX(),
                                       ParSiOk(iPar) += AndL() += aSel, nSiHit, "BOX");

    const TString nSiDiHit(this->Unique(Form("sigmaByHit%d", iPar)));
    TH1 *hSiByHit = this->CreateHist((ParSi(iPar) += toMum) += Div() += Sqrt(HitsX()),
                                     ParSiOk(iPar) += AndL() += aSel, nSiDiHit);

    hDParVsHit->SetTitle(DelName(iPar) += ": remaining misalign vs. N_{hit,x}" + titleAdd + ";" 
			 + Fun("#Delta", DelName(iPar))+=Unit(iPar)+=";N_{hit,x}"); 
    hSiVsHit->SetTitle(Fun("#sigma", DelName(iPar)) += " vs. N_{hit,x}" + titleAdd + ";" +
                       Fun("#sigma", DelName(iPar))+=Unit(iPar)+=";N_{hit,x}");
    hSiByHit->SetTitle(Fun("#sigma", DelName(iPar)) += Div() += Sqrt("N_{hit,x}") += titleAdd 
		       + ";" + Fun("#sigma", DelName(iPar)) += Div() += Sqrt("N_{hit,x}") +=
                       Unit(iPar) += ";#parameters");

    fHistManager->AddHist(hDParVsHit, layer);
    // check that we have sigma determined by pede:
    if (hSiVsHit->GetEntries()) fHistManager->AddHist(hSiVsHit, layer+1);
    if (hSiByHit->GetEntries()) fHistManager->AddHist(hSiByHit, layer+2);

    ++nPlot;
  }

  fHistManager->Draw();
}

////////////////////////////////////////////////////////////////////////////////////////////////////
void PlotMillePede::DrawHitMaps(bool addPlots, bool inclFullFixed)
{

  const Int_t layer = this->PrepareAdd(addPlots);
  TString sel(inclFullFixed ? "" : AnyFreePar().Data());
  this->AddBasicSelection(sel);
  const TString titleAdd = this->TitleAdd();

  Option_t *drawOpt = "BOX";

  TH2 *hRx = this->CreateHist2D(RPos(OrgPosT()), HitsX(), sel, this->Unique("xHitsR"), drawOpt);
  TProfile *hRxProf = this->CreateHistProf(RPos(OrgPosT()), HitsX(), sel,
					   this->Unique("profxHitsR"));
    //hRx->ProfileX("profHitsXR");
  TH2 *hRy = this->CreateHist2D(RPos(OrgPosT()), HitsY(), sel, this->Unique("yHitsR"), drawOpt);
  TProfile *hRyProf = this->CreateHistProf(RPos(OrgPosT()), HitsY(), sel,
                                           this->Unique("profyHitsR"));
    //hRy->ProfileX("profHitsYR");
  
  TH2 *hZx = this->CreateHist2D(OrgPosT()+=ZPos(), HitsX(), sel, this->Unique("xHitsZ"), drawOpt);
  TProfile *hZxProf = this->CreateHistProf(OrgPosT()+=ZPos(), HitsX(), sel,
                                           this->Unique("profxHitsZ"));
    //hZx->ProfileX("profHitsXZ");
  TH2 *hZy = this->CreateHist2D(OrgPosT()+=ZPos(), HitsY(), sel, this->Unique("yHitsZ"), drawOpt);
  TProfile *hZyProf = this->CreateHistProf(OrgPosT()+=ZPos(), HitsY(), sel,
                                           this->Unique("profyHitsZ"));
    //hZy->ProfileX("profHitsYZ");
  
  TH2 *hPhiX = this->CreateHist2D(Phi(OrgPosT()), HitsX(), sel, this->Unique("xHitsPhi"), drawOpt);
  TProfile *hPhixProf = this->CreateHistProf(Phi(OrgPosT()), HitsX(), sel,
                                             this->Unique("profxHitsPhi"));
    //hPhiX->ProfileX("profHitsXPhi");
  TH2 *hPhiY = this->CreateHist2D(Phi(OrgPosT()), HitsY(), sel, this->Unique("yHitsPhi"), drawOpt);
  TProfile *hPhiyProf = this->CreateHistProf(Phi(OrgPosT()), HitsY(), sel,
                                             this->Unique("profyHitsPhi"));
    //hPhiY->ProfileX("profHitsYPhi");

  TString selHitY(HitsY());
  if (!sel.IsNull()) selHitY += AndL() += sel;
  TH1 *hNhitXlog = this->CreateHist(Fun("TMath::Log10", HitsX()), // ignore 0 entries for log
                                    sel+AndL()+=Parenth(HitsX()+=">0"), this->Unique("NhitXlog"));
  TH1 *hNhitX = this->CreateHist(HitsX(), sel, this->Unique("NhitX"));
  TH1 *hNhitY = this->CreateHist(HitsY(), sel, this->Unique("NhitY"));
  TH2 *hNhitXvsY = this->CreateHist2D(HitsX(), HitsY(), sel, this->Unique("NhitXvsY"), drawOpt);
  TH1 *hNhitDiffXy = this->CreateHist(HitsX()+=Min()+=HitsY(), selHitY, this->Unique("NhitDiffXy"));
  TH1 *hNhitDiffXyVsR = this->CreateHist2D(RPos(OrgPosT()), HitsX()+=Min()+=HitsY(), 
                                           selHitY, this->Unique("NhitDiffXyVsR"), drawOpt);
  
  //hits= weight: i.e. sum!
  drawOpt = "COLZ";
  if (!sel.IsNull()) sel = Parenth(sel).Prepend(Mal());
  TH2 *hTotPhiRx = this->CreateHist2D(OrgPosT()+=XPos(), OrgPosT()+=YPos(), HitsX()+=sel,
                                      this->Unique("totXvsXY"), drawOpt);//+="(100,-110,110,100,110,110"
  TH2 *hTotPhiRy = this->CreateHist2D(OrgPosT()+=XPos(), OrgPosT()+=YPos(), HitsY()+=sel,
                                      this->Unique("totYvsXY"), drawOpt);
  TH2 *hTotRzX = this->CreateHist2D(OrgPosT()+=ZPos(), RPos(OrgPosT()), HitsX()+=sel,
                                    this->Unique("totXvsZR"), drawOpt);
  TH2 *hTotRzY = this->CreateHist2D(OrgPosT()+=ZPos(), RPos(OrgPosT()), HitsY()+=sel,
                                    this->Unique("totYvsZR"), drawOpt);


  hRx->SetTitle("#hits_{x} vs. r" + titleAdd + ";r[cm];N_{hit,x}");
  hRxProf->SetTitle("<#hits_{x}> vs. r" + titleAdd + ";r[cm];N_{hit,x}");
  hRy->SetTitle("#hits_{y} vs. r" + titleAdd + ";r[cm];N_{hit,y}");
  hRyProf->SetTitle("<#hits_{y}> vs. r" + titleAdd + ";r[cm];N_{hit,y}");
  hZx->SetTitle("#hits_{x} vs. z" + titleAdd + ";z[cm];N_{hit,x}");
  hZxProf->SetTitle("<#hits_{x}> vs. z" + titleAdd + ";z[cm];N_{hit,x}");
  hZy->SetTitle("#hits_{y} vs. z" + titleAdd + ";z[cm];N_{hit,y}");
  hZyProf->SetTitle("<#hits_{y}> vs. z" + titleAdd + ";z[cm];N_{hit,y}");
  hPhiX->SetTitle("#hits_{x} vs. #phi" + titleAdd + ";#phi;N_{hit,x}");
  hPhixProf->SetTitle("<#hits_{x}> vs. #phi" + titleAdd + ";#phi;N_{hit,x}");
  hPhiY->SetTitle("#hits_{y} vs. #phi" + titleAdd + ";#phi;N_{hit,y}");
  hPhiyProf->SetTitle("<#hits_{y}> vs. #phi" + titleAdd + ";#phi;N_{hit,y}");

  hNhitXlog->SetTitle("#hits_{x}: log_{10}" + titleAdd + ";log_{10}(N_{hit,x})");
  hNhitX->SetTitle("#hits_{x}" + titleAdd + ";N_{hit,x}");
  hNhitY->SetTitle("#hits_{y}" + titleAdd + ";N_{hit,y}");
  hNhitXvsY->SetTitle("#hits_{x} vs. #hits_{y}" + titleAdd + ";N_{hit,x};N_{hit,y}");
  hNhitDiffXy->SetTitle("#hits_{x} - #hits_{y} (#hits_{y}#neq0)" + titleAdd
			+ ";N_{hit,x}-N_{hit,y};#alignables");
  hNhitDiffXyVsR->SetTitle("#hits_{x} - #hits_{y} vs. r (#hits_{y}#neq0)" + titleAdd
			   + ";r[cm];N_{hit,x}-N_{hit,y}");

  hTotPhiRx->SetTitle("#hits_{x}" + titleAdd + ";x [cm];y [cm]");
  hTotPhiRy->SetTitle("#hits_{y}" + titleAdd + ";x [cm];y [cm]");
  hTotRzX->SetTitle("#hits_{x}" + titleAdd + ";z [cm];r [cm]");
  hTotRzY->SetTitle("#hits_{y}" + titleAdd + ";z [cm];r [cm]");

  const bool addYhists = (hNhitY->GetMean() || hNhitY->GetRMS()); 

  fHistManager->AddHist(hRx, layer);
  fHistManager->AddHist(hRxProf, layer+1);
  if (addYhists) {
    fHistManager->AddHist(hRy, layer);
    fHistManager->AddHist(hRyProf, layer+1);
  }
  fHistManager->AddHist(hZx, layer);
  fHistManager->AddHist(hZxProf, layer+1);
  if (addYhists) {
    fHistManager->AddHist(hZy, layer);
    fHistManager->AddHist(hZyProf, layer+1);
  }
  fHistManager->AddHist(hPhiX, layer);
  fHistManager->AddHist(hPhixProf, layer+1);
  if (addYhists) {
    fHistManager->AddHist(hPhiY, layer);
    fHistManager->AddHist(hPhiyProf, layer+1);
  }
  
  fHistManager->AddHist(hNhitXlog, layer+2);
  fHistManager->AddHist(hNhitX, layer+2);
  fHistManager->AddHist(hNhitY, layer+2); // add always to show that no y-hit
  if (addYhists) {
    fHistManager->AddHist(hNhitXvsY, layer+2);
    fHistManager->AddHist(hNhitDiffXy, layer+2);
    fHistManager->AddHist(hNhitDiffXyVsR, layer+2);
  }

  fHistManager->AddHist(hTotPhiRx, layer+3);
  if (addYhists) fHistManager->AddHist(hTotPhiRy, layer+3);
  fHistManager->AddHist(hTotRzX, layer+3);
  if (addYhists) if (hNhitY->GetEntries()) fHistManager->AddHist(hTotRzY, layer+3);

  fHistManager->Draw();
}

////////////////////////////////////////////////////////////////////////////////////////////////////
void PlotMillePede::DrawBigPullLabel(float minPull, bool addPlots)
{
  const Int_t layer = this->PrepareAdd(addPlots);

  //  UInt_t nPlot = 0;
  for (UInt_t iPar = 0; iPar < kNpar; ++iPar) { 

    const TString misPar(this->FinalMisAlignment(iPar));

    const TString hNameL(Form("bigPullLabel%d(100000,0,100000)", iPar));
    const TString isBigPull(Abs(misPar + Div() += ParSi(iPar)) += Form(" > %f", minPull));
    TString sel(isBigPull + AndL() += Fixed(iPar, false));
    this->AddBasicSelection(sel);

    TH1 *hLabel = this->CreateHist(Label(iPar), sel, hNameL);
    if (0. == hLabel->GetEntries()) continue;

    hLabel->SetTitle((DelName(iPar) += Form(": |pull| > %f", minPull)) += ";Label");
    fHistManager->AddHist(hLabel, layer);
    //    ++nPlot;
  }

  fHistManager->Draw();
}

////////////////////////////////////////////////////////////////////////////////////////////////////
void PlotMillePede::DrawBigPullPos(float minPull, bool addPlots)
{
  const Int_t layer = this->PrepareAdd(addPlots);
  const TString titleAdd = this->TitleAdd();

  UInt_t nPlot = 0;
  for (UInt_t iPar = 0; iPar < kNpar; ++iPar) { 
    const TString misPar(this->FinalMisAlignment(iPar));

    const TString isBigPull(Abs(misPar + Div() += ParSi(iPar)) += Form(" > %f", minPull));
    TString sel(isBigPull + AndL() += Fixed(iPar, false));
    this->AddBasicSelection(sel);

    const TString hNameR(this->Unique(Form("bigPullR%d", iPar)));

    TH1 *hR = this->CreateHist(RPos(OrgPosT()), sel, hNameR);
    if (0. == hR->GetEntries()) continue;
    const TString hNameZ(this->Unique(Form("bigPullZ%d", iPar)));
    TH1 *hZ = this->CreateHist(OrgPosT()+=ZPos(), sel, hNameZ);
    const TString hNameP(this->Unique(Form("bigPullP%d", iPar)));
    TH1 *hPhi = this->CreateHist(Phi(OrgPosT()), sel, hNameP);

    hR->SetTitle((DelName(iPar) += Form(": |pull| > %f", minPull)) += titleAdd + ";r[cm]");
    hZ->SetTitle((DelName(iPar) += Form(": |pull| > %f", minPull)) += titleAdd + ";z[cm]");
    hPhi->SetTitle((DelName(iPar) += Form(": |pull| > %f", minPull)) += titleAdd + ";#phi");
    fHistManager->AddHist(hR, layer + nPlot);
    fHistManager->AddHist(hZ, layer + nPlot);
    fHistManager->AddHist(hPhi, layer + nPlot);
    ++nPlot;
  }

  fHistManager->Draw();
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// void PlotMillePede::DrawShiftsImprovement()
// {
//
//   fHistManager->Clear();
//
//   Int_t nPlot = 0;
//   for (UInt_t iPar = kLocX; iPar <= kLocZ; ++iPar) { // 
//     const TString hNameB(Form("before%d(100, -500, 500)", iPar));//
//     const TString hNameA(Form("after%d(100, -500, 500)", iPar));//(100, -50, 50)
//     const TString hNameAs(Form("shouldafter%d(100, -500, 500)", iPar));//(100, -50, 50)
//     TH1 *hBefore = this->CreateHist(MisRelPosT()+=Pos(iPar) += "*10000",
// 				    "!" + Fixed(iPar), hNameB);
//     TH1 *hAfter = this->CreateHist(RelPosT()+=Pos(iPar) += "*10000",
// 				   "!" + Fixed(iPar), hNameA);
//     TH1 *hShouldAfter = this->CreateHist(Parenth(MisRelPosT()+=Pos(iPar)+=Plu()
//                                                  +=ParT()+=Par(iPar)) += "*10000",
//                                          "!" + Fixed(iPar), hNameAs);
//     //    if (0. == hAfter->GetEntries()) continue;
//     if (0. == hBefore->GetEntries()) continue;
//     fHistManager->AddHist(hBefore, 0, "Before");
//     fHistManager->AddHist(hAfter, 1, "After");
//     fHistManager->AddHistSame(hShouldAfter, 1, nPlot, "ShouldAfter");
//     fHistManager->AddHist(hBefore, 2, "Before");
//     fHistManager->AddHistSame(hAfter, 2, nPlot, "After");
//     ++nPlot;
//   }
//
//   fHistManager->Draw();
// }

////////////////////////////////////////////////////////////////////////////////////////////////////
void PlotMillePede::DrawSubDetId(bool addPlots)
{
  const Int_t layer = this->PrepareAdd(addPlots);
  const TString titleAdd = this->TitleAdd();

  const TString nameAll(this->Unique("subDetId"));
  const TString nameAct(this->Unique("subDetIdActive"));

  TString sel;
  this->AddBasicSelection(sel);
  TH1 *hAll = this->CreateHist(SubDetId(), sel, nameAll);

  if (!sel.IsNull()) sel = Parenth(sel) += AndL();
  sel += AnyFreePar();
  TH1 *hAct = this->CreateHist(SubDetId(), sel, nameAct);

  if (hAll->GetEntries()) {
    hAll->SetTitle("subDetId;ID(subdet)");
    fHistManager->AddHist(hAll, layer);
  }
  if (hAct->GetEntries()) {
    hAct->SetTitle("subDetId (any free param)" + titleAdd + ";ID(subdet)");
    fHistManager->AddHist(hAct, layer);
  }

  fHistManager->Draw();
}

///////////////////////////////////////////////////////////////////////////////////////////////////
void PlotMillePede::DrawXyArrow(Double_t factor, Option_t *option)
{
  const TString opt(option);
  const Int_t layer = this->PrepareAdd(opt.Contains("add", TString::kIgnoreCase));

  // prepare selection
  TString sel;
  this->AddBasicSelection(sel);
  // Draw command for populating tree->GetV1/2() - resulting hist is not of interest!
  delete this->Draw(OrgPosT() += XPos() += ":" + OrgPosT() += YPos(), sel, "", "goff");

  TTree *tree = this->GetMainTree();
  const Long64_t size = tree->GetSelectedRows();

  if (size == 0) return; // nothing survived...

  // get min/max x/y for frame
  Double_t maxX = tree->GetV1()[TMath::LocMax(size, tree->GetV1())];
  maxX *= (maxX < 0 ? 0.9 : 1.1);
  Double_t minX = tree->GetV1()[TMath::LocMin(size, tree->GetV1())];
  minX *= (minX > 0 ? 0.9 : 1.1);
  Double_t maxY = tree->GetV2()[TMath::LocMax(size, tree->GetV2())];
  maxY *= (maxY < 0 ? 0.9 : 1.1);
  Double_t minY = tree->GetV2()[TMath::LocMin(size, tree->GetV2())];
  minY *= (minY > 0 ? 0.9 : 1.1);
  TH1 *hFrame = new TH2F(this->Unique("frame"),
			 Form("scale %g%s;x [cm];y [cm]", factor, this->TitleAdd().Data()),
			 10, minX, maxX, 10, minY, maxY);
  hFrame->SetOption("AXIS");
  hFrame->SetEntries(size); // entries shows number of plotted arrows
  fHistManager->AddHist(hFrame, layer);

  // copy arrays from TTree:
  const std::vector<double> xs(tree->GetV1(), tree->GetV1() + size);
  const std::vector<double> ys(tree->GetV2(), tree->GetV2() + size);

  // Now draw for deltas (even GetV3()!) - again return value irrelevant
  delete this->Draw(DeltaPos("x", PosT()) += ":" + DeltaPos("y", PosT()) 
		    += ":" + DeltaPos("z", PosT()), 
		    sel, "", "goff");
  // copy delta x's and y's
  const std::vector<double> deltaXs(tree->GetV1(), tree->GetV1() + size);
  const std::vector<double> deltaYs(tree->GetV2(), tree->GetV2() + size);

  if (opt.Contains("zcirc", TString::kIgnoreCase)) { // circles for z to be drawn
    // get delta z from tree
    const std::vector<double> deltaZs(tree->GetV3(), tree->GetV3() + size);
    const Double_t rootFactor = TMath::Sqrt(TMath::Abs(factor)); //area grows ^2...
    // add z positions via coloured circles
    for (unsigned int i = 0; i < size; ++i) {
      if (deltaZs[i] == 0.) continue;
      TEllipse *circ = new TEllipse(xs[i], ys[i], TMath::Abs(rootFactor*deltaZs[i]));
      // circ->SetLineStyle(0); // no line at border
      if (deltaZs[i] < 0.) circ->SetFillColor(kRed);
      else circ->SetFillColor(kGreen);
      fHistManager->AddObject(circ, layer, 0);
    }
  }

  // xy-arrows on top
  TArrow::SetDefaultAngle(30.);
  for (unsigned int i = 0; i < size; ++i) {
    TArrow *arr = new TArrow(xs[i], ys[i], xs[i] + factor*deltaXs[i],
			     ys[i] + factor*deltaYs[i], 0.01, "|>");
    fHistManager->AddObject(arr, layer, 0);
  }

  fHistManager->Draw();
}

////////////////////////////////////////////////////////////////////////////////////////////////////
void PlotMillePede::ScanSelection(const char *sel, const char *addColumns)
{
  TString realSel;
  if (sel) {
    realSel = sel;
  } else {
    this->AddBasicSelection(realSel);
    const TString titleAdd(this->TitleAdd());
    if (titleAdd.Length()) std::cout << titleAdd << std::endl;
  }

  const TString mpPar(MpT() += Par());// += this->ToMumMuRad(iPar));
  //  this->GetMainTree()->Scan("Id:Pos:" + mpPar += Form(":HitsX:Sigma[%u]:Label", iPar), sel);
  TString scan("Id:Pos:" + mpPar += ":HitsX:Sigma:Label");
  if (addColumns) scan += addColumns;
  this->GetMainTree()->Scan(scan, sel);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
void PlotMillePede::ScanPedeParAbove(UInt_t iPar, float absMinInMuMm)
{
  const TString mpPar(MpT() += Par(iPar));// += this->ToMumMuRad(iPar));
  TString sel(Fixed(iPar, false) += AndL() += Abs(mpPar) += Form(">%f", absMinInMuMm));
  this->AddBasicSelection(sel);
  const TString titleAdd(this->TitleAdd());
  if (titleAdd.Length()) std::cout << titleAdd << std::endl;
  
  this->ScanSelection(sel);
  //    this->GetMainTree()->Scan("Id:Pos:" + mpPar += Form(":HitsX:Sigma[%u]:Label", iPar), sel);

}

////////////////////////////////////////////////////////////////////////////////////////////////////
void PlotMillePede::DrawCheck()
{

  fHistManager->Clear();

  const TString allTreeNames[] = {OrgPosT(), MisPosT(), MisRelPosT(), RelPosT(), PosT(),
                                  MisParT(), ParT(), MpT()};

  const unsigned int nTree = sizeof(allTreeNames) / sizeof(allTreeNames[0]);
  for (unsigned int i = 0; i < nTree; ++i) {
    TH1 *hId = this->CreateHist((PosT()+="Id - ")+allTreeNames[i]+"Id", "");
    TH1 *hObjId = this->CreateHist((PosT()+="ObjId - ")+allTreeNames[i]+"ObjId", "");
    fHistManager->AddHist(hId, i);
    fHistManager->AddHist(hObjId, i);
  }

  fHistManager->Draw();
}

////////////////////////////////////////////////////////////////////////////////////////////////////
Int_t PlotMillePede::PrepareAdd(bool addPlots)
{
  if (addPlots) {
    return fHistManager->GetNumLayers();
  } else {
    fHistManager->Clear();
    return 0;
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
TString PlotMillePede::Unique(const char *name) const
{
  if (!gROOT->FindObject(name)) return name;

  UInt_t i = 1;
  while (gROOT->FindObject(Form("%s_%u", name, i))) ++i;

  return Form("%s_%u", name, i);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
void PlotMillePede::AddBasicSelection(TString &sel) const
{
  // hierarchy level if selected
  if (fHieraLevel >= 0) {
    if (sel.IsNull())             sel = HieraLev(fHieraLevel);
    else sel = Parenth(sel) += AndL() += HieraLev(fHieraLevel);
  }

  // selected subdets ony (all if none selected)
  TString subDetSel; // first create or of subdets
  for (Int_t iSub = 0; iSub < fSubDetIds.GetSize(); ++iSub) {
    const TString newSubDetSel(Parenth(SubDetId()) += Form("==%d", fSubDetIds[iSub]));
    if (subDetSel.IsNull())             subDetSel = newSubDetSel;
    else subDetSel = Parenth(subDetSel) += OrL() += newSubDetSel;
  }
  if (!subDetSel.IsNull()) {
    if (sel.IsNull())              sel = Parenth(subDetSel);
    else sel = Parenth(sel) += AndL() += Parenth(subDetSel); 
  }
  
  // alignbale type (rod, det, petal...) if selected
  if (fAlignableTypeId >= 0) {
    const TString alignableTypeSel(Parenth(AlignableTypeId() += Form("==%d", fAlignableTypeId)));
    if (sel.IsNull()) sel = alignableTypeSel;
    else sel = Parenth(sel) += AndL() += alignableTypeSel;
  }

  if (fAdditionalSel.Length()) {
    if (sel.IsNull()) sel = fAdditionalSel;
    else sel = Parenth(sel) += AndL() += fAdditionalSel;
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
Float_t PlotMillePede::SetMaxDev(Float_t maxDev)
{
  // set x-axis range for result plots
  const Float_t devOld = fMaxDev;
  fMaxDev = maxDev;
  return devOld;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
void PlotMillePede::SetSubDetId(Int_t subDetId)
{
  // select a single subdet, 1-6 are TPB, TPE, TIB, TID, TOB, TEC, -1 means: take all
  if (subDetId == -1) {
    fSubDetIds.Set(0);
  } else {
    fSubDetIds.Set(1, &subDetId); // length 1, value of subDetId
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
void PlotMillePede::AddSubDetId(Int_t subDetId)
{
  // add subdet to selection, 1-6 are TPB, TPE, TIB, TID, TOB, TEC
  const Int_t last = fSubDetIds.GetSize();
  fSubDetIds.Set(last+1); // enlarge by one
  fSubDetIds[last] = subDetId;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
TString PlotMillePede::TitleAdd() const
{
  // add something to title that describes general selection
  TString result;
  for (Int_t i = 0; i < fSubDetIds.GetSize(); ++i) {
    if (result.Length()) result += " + ";
    switch (fSubDetIds[i]) {
    case 1:
      result += "BPIX"; break;
    case 2:
      result += "FPIX"; break;
    case 3:
      result += "TIB"; break;
    case 4:
      result += "TID"; break;
    case 5:
      result += "TOB"; break;
    case 6:
      result += "TEC"; break;
    default:
      // ::Warning("PlotMillePede::SubDetTitleAdd", "unknown subDetId %d", fSubDetIds[i]);
      result += "unknown subDet";
    }
  }

  if (fAlignableTypeId >= 0) {
    if (result.Length()) result += ", ";
    result += Form("type %d", fAlignableTypeId);
  }

  if (fHieraLevel != 0) {
    if (result.Length()) result += ", ";
    if (fHieraLevel < 0) result += Form("all hierar. levels");
    else                 result += Form("hier. level %d", fHieraLevel);
  }

  if (fAdditionalSelTitle.Length()) {
    if (result.Length()) result += ", ";
    result += fAdditionalSelTitle;
  }

  if (fTitle.Length()) {
    if (result.Length()) result.Prepend(fTitle + ", ");
    else result.Prepend(fTitle);
  }

  if (result.Length()) result.Prepend(": ");  

  return result;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
Int_t PlotMillePede::SetHieraLevel(Int_t hieraLevel)
{ // select hierarchical level (-1: all)
  const Int_t oldLevel = fHieraLevel;
  fHieraLevel = hieraLevel;
  return oldLevel;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
Int_t PlotMillePede::SetAlignableTypeId(Int_t alignableTypeId)
{ // select ali type id, i.e. rod, det, petal etc. (-1: take all)
  const Int_t oldTypeId = fAlignableTypeId;
  fAlignableTypeId = alignableTypeId;
  return oldTypeId;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
void PlotMillePede::AddAdditionalSel(const char *selection)
{
  if (!selection || selection[0] == '\0') {
    ::Warning("PlotMillePede::AddAdditionalSel", "Ignore empty selection.");
  } else {
    if (fAdditionalSel.Length()) fAdditionalSel = Parenth(fAdditionalSel) += AndL();
    // Add to title for hists as well :
    if (fAdditionalSelTitle.Length()) fAdditionalSelTitle += ", ";
    const TString sel(selection);
    if (sel == "StripDoubleOr1D") {
      fAdditionalSel += "(Id&3)==0";
      fAdditionalSelTitle += "Double sided or 1D layer/ring";
    } else if (sel == "StripRphi") {
      fAdditionalSel += "(Id&3)==2";
      fAdditionalSelTitle += "R#phi";
    } else if (sel == "StripStereo"){
      fAdditionalSel += "(Id&3)==1";
      fAdditionalSelTitle += "Stereo";
    } else { // genericaly add
      fAdditionalSel += selection;
      fAdditionalSelTitle += selection;
    }
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
void PlotMillePede::AddAdditionalSel(const TString &xyzrPhiNhit, Float_t min, Float_t max)
{
  const TString oldTitle = fAdditionalSelTitle; // backup
  if (xyzrPhiNhit == "Nhit") {
    this->AddAdditionalSel(HitsX() += Form(">%f && ", min) + HitsX() += Form("<%f", max));
  } else {
    this->AddAdditionalSel(OrgPos(xyzrPhiNhit) += Form(">%f && ", min) 
			   + OrgPos(xyzrPhiNhit) += Form("<%f", max));
  }
  // add to title in readable format
  fAdditionalSelTitle = oldTitle; // first remove what was added in unreadable format...
  if (fAdditionalSelTitle.Length()) fAdditionalSelTitle += ", ";
  fAdditionalSelTitle += Form("%g < %s < %g", min, xyzrPhiNhit.Data(), max);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
TString PlotMillePede::FinalMisAlignment(UInt_t iPar) const
{
  return (fUseDiff ? DiffPar(ParT(),MisParT(),iPar) : Parenth(ParT()+=Par(iPar)));
}

////////////////////////////////////////////////////////////////////////////////////////////////////
void PlotMillePede::CopyAddBinning(TString &name, const TH1 *h) const
{
  if (!h) return;
  
  name += Form("(%d,%f,%f", h->GetNbinsX(), h->GetXaxis()->GetXmin(), h->GetXaxis()->GetXmax());
  if (h->GetDimension() > 1) {
    name += Form(",%d,%f,%f", h->GetNbinsY(), h->GetYaxis()->GetXmin(), h->GetYaxis()->GetXmax());
  }
  if (h->GetDimension() > 2) {
    name += Form(",%d,%f,%f", h->GetNbinsZ(), h->GetZaxis()->GetXmin(), h->GetZaxis()->GetXmax());
  }

  name += ')';
}
