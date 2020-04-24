// Original Author: Gero Flucke
// last change    : $Date: 2013/03/07 11:22:10 $
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
#include <TPaveText.h>

#include <TError.h>
#include <TROOT.h>
#include <TCanvas.h>

#include <iostream>

#include "GFUtils/GFHistManager.h"

////////////////////////////////////////////////////////////////////////////////////////////////////
PlotMillePede::PlotMillePede(const char *fileName, Int_t iov, Int_t hieraLevel, bool useDiff)
  : MillePedeTrees(fileName, iov), fHistManager(new GFHistManager), fHieraLevel(hieraLevel),
    fUseDiff(useDiff), fSubDetIds(), fAlignableTypeId(-1),
    fMaxDevUp(500.), fMaxDevDown(-500.), fNbins(101)
{
  fHistManager->SetLegendX1Y1X2Y2(0.14, 0.7, 0.45, 0.9);
}
////////////////////////////////////////////////////////////////////////////////////////////////////
PlotMillePede::PlotMillePede(const char *fileName, Int_t iov, Int_t hieraLevel, const char *treeNameAdd)
  : MillePedeTrees(fileName, iov, treeNameAdd),
    fHistManager(new GFHistManager), fHieraLevel(hieraLevel),
    fUseDiff(false), fSubDetIds(), fAlignableTypeId(-1),
    fMaxDevUp(500.), fMaxDevDown(-500.), fNbins(101)
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
  if (o.Contains("r", TString::kIgnoreCase)) this->DrawParamResult("add");
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
void PlotMillePede::DrawPedeParam(Option_t *option, unsigned int nNonRigidParam)
{
  const bool addParVsPar = TString(option).Contains("vs", TString::kIgnoreCase);
  const Int_t layer = this->PrepareAdd(TString(option).Contains("add", TString::kIgnoreCase));
  const unsigned int nPar = kNpar + nNonRigidParam;
  const TString titleAdd = this->TitleAdd();
  UInt_t nPlot = 0;
  TObjArray primitivesParVsPar;
  for (UInt_t iPar = 0; iPar < nPar; ++iPar) { // 
    TString parSel(Fixed(iPar, false) += AndL() += Valid(iPar));
    this->AddBasicSelection(parSel);
    const TString toMum(this->ToMumMuRadPede(iPar));
    const TString hName(this->Unique(Form("pedePar%d", iPar))); // 
    TH1 *h = this->CreateHist(Parenth(MpT() += Par(iPar)) += toMum, parSel, hName);

    if (0. == h->GetEntries()) continue;

    TObjArray histsParVsPar;
    if (addParVsPar) {// parameters vs each other
      for (UInt_t iPar2 = iPar + 1; iPar2 < nPar; ++iPar2) {
	const TString hNameVs(this->Unique(Form("pedePar%d_%d", iPar, iPar2))); // 
	const TString toMum2(this->ToMumMuRadPede(iPar2));
	TH2 *hVs = this->CreateHist2D(Parenth(MpT() += Par(iPar)) += toMum,
				      Parenth(MpT() += Par(iPar2)) += toMum2, // Valid(iPar2??)
				      Parenth(parSel) += AndL() += Fixed(iPar2, false), hNameVs, "BOX");
	if (0. == hVs->GetEntries()) continue;// delete hVs;
	else {
	  hVs->SetTitle("pede: " + NamePede(iPar2) += " vs. " + NamePede(iPar) += titleAdd + ";"
			+ NamePede(iPar) += UnitPede(iPar) += ";" + NamePede(iPar2) += UnitPede(iPar2));
	  histsParVsPar.Add(hVs);
	  // add info about correlation factor
	  TPaveText *pave = new TPaveText(.15,.15,.5,.25, "NDC"); 
	  pave->SetBorderSize(1);
	  pave->AddText(Form("#rho = %.3f", hVs->GetCorrelationFactor()));
	  primitivesParVsPar.Add(pave);
	}
      }
    }

    // parameters and errors
    const TString hNameBySi(this->Unique(Form("pedeParBySi%d", iPar))); // 
    TH1 *hBySi = this->CreateHist(Parenth(MpT() += Par(iPar)) += Div() += ParSi(iPar),
                                  parSel + AndL() += ParSiOk(iPar), hNameBySi);
    TH1 *hBySiInv = 0;
    TH2 *hSiVsPar = 0;
    if (hBySi->GetEntries() == 0.) {
      delete hBySi; hBySi = 0;
    } else {
      const TString hNameBySiInv(this->Unique(Form("pedeParBySiInv%d", iPar)) += "(100,-20,20)"); // 
      hBySiInv = this->CreateHist(ParSi(iPar) += Div() += Parenth(MpT() += Par(iPar)),
                                  parSel + AndL() += ParSiOk(iPar), hNameBySiInv);
      const TString hNameSiVsPar(this->Unique(Form("pedeParVsSi%d", iPar))); // 
      hSiVsPar = this->CreateHist2D(Parenth(MpT() += Par(iPar)) += toMum, ParSi(iPar) += toMum,
				    parSel + AndL() += ParSiOk(iPar), hNameSiVsPar, "BOX");
    }
    
    // parameters vs hits
    const TString hNameH(this->Unique(Form("pedeParVsHits%d", iPar))); // 
    TH2 *hHits = this->CreateHist2D(HitsX(), Parenth(MpT()+=Par(iPar)) += toMum, parSel,
				    hNameH, "BOX");

    // parameters vs global correlation
    const TString hNameG(this->Unique(Form("pedeParVsGlob%d", iPar))); // 
    TH2 *hGlobCor = this->CreateHist2D(Cor(iPar), Parenth(MpT()+=Par(iPar)) += toMum,
				       parSel +  AndL() += Cor(iPar) += ">-0.1999", hNameG, "BOX");
    if (!hGlobCor->GetEntries()) {
      delete hGlobCor; hGlobCor = 0;
    }

    h->SetTitle("determined pede " + NamePede(iPar) += titleAdd + ";"
		+ NamePede(iPar) += UnitPede(iPar) += ";#alignables");
    hHits->SetTitle("determined pede " + NamePede(iPar) += titleAdd + " vs #n(hit_{x});N_{hit,x};"
		+ NamePede(iPar) += UnitPede(iPar));
    if (hGlobCor) hGlobCor->SetTitle("determined pede " + NamePede(iPar) += titleAdd + 
				     " vs glob. corr;Global Correlation;" + NamePede(iPar)
				     += UnitPede(iPar));
    fHistManager->AddHist(h, layer);
    fHistManager->AddHist(hHits, layer+1);
    if (addParVsPar) fHistManager->AddHists(&histsParVsPar, layer+2);
    if (hGlobCor) fHistManager->AddHist(hGlobCor, layer+(addParVsPar ? 3 : 2));

    if (hBySi) {
      const TString namI(NamePede(iPar));
      hBySi->SetTitle("pede: " + namI + "/#sigma_{" + namI + "}" + titleAdd + ";"
                      + namI + Div() += Fun("#sigma", namI) += ";#alignables");
      fHistManager->AddHist(hBySi, layer+2+(hGlobCor ? 1 : 0)+addParVsPar);
      hBySiInv->SetTitle("pede: #sigma_{" + namI + "}/" + namI + titleAdd + ";"
                         + Fun("#sigma", namI) += Div() += namI + ";#alignables");
      fHistManager->AddHist(hBySiInv, layer+3+(hGlobCor ? 1 : 0)+addParVsPar);
      hSiVsPar->SetTitle("pede: #sigma_{" + namI + " } vs " + namI + titleAdd + ";"
                         + namI + UnitPede(iPar) + ";" + Fun("#sigma", namI) += UnitPede(iPar));
      fHistManager->AddHist(hSiVsPar, layer+4+(hGlobCor ? 1 : 0)+addParVsPar);
    }
    ++nPlot;
  }
  
  if (addParVsPar) {
    for (Int_t i = 0; i < primitivesParVsPar.GetEntries(); ++i) {
      fHistManager->AddObject(primitivesParVsPar[i], layer+2, i);
    }
  }

  fHistManager->Draw();
}

////////////////////////////////////////////////////////////////////////////////////////////////////
void PlotMillePede::DrawPedeParamVsLocation(Option_t *option, unsigned int nNonRigidParam)
{
  const Int_t layer = this->PrepareAdd(TString(option).Contains("add", TString::kIgnoreCase));
  const TString titleAdd = this->TitleAdd();

  const unsigned int nPar = kNpar + nNonRigidParam;
  UInt_t nPlot = 0;
  for (UInt_t iPar = 0; iPar < nPar; ++iPar) { // 
    TString parSel(Fixed(iPar, false) += AndL() += Valid(iPar));
    this->AddBasicSelection(parSel);
    const TString toMum(this->ToMumMuRadPede(iPar));
    const TString pedePar(Parenth(MpT() += Par(iPar)) += toMum);

    const TString nDpz(this->Unique(Form("pedeParZ%d", iPar)));
    TH2 *hPedeParZ = this->CreateHist2D(OrgPosT() += ZPos(), pedePar,
					parSel, nDpz, "BOX");

    if (0. == hPedeParZ->GetEntries()) continue;

    const TString nDpr(this->Unique(Form("pedeParR%d", iPar)));
    TH2 *hPedeParR = this->CreateHist2D(RPos(OrgPosT()), pedePar,
					parSel, nDpr, "BOX");

    const TString nDpp(this->Unique(Form("pedeParPhi%d", iPar)));
    TH2 *hPedeParPhi = this->CreateHist2D(Phi(OrgPosT()), pedePar,
					  parSel, nDpp, "BOX");

//     const TString nDpx(this->Unique(Form("pedeParX%d", iPar)));
//     TH2 *hPedeParX = this->CreateHist2D(OrgPosT() += XPos(), pedePar,
// 					parSel, nDpx, "BOX");

    const TString nDpy(this->Unique(Form("pedeParY%d", iPar)));
    TH2 *hPedeParY = this->CreateHist2D(OrgPosT() += YPos(), pedePar,
					parSel, nDpy, "BOX");
    // theta?
    const TString title("determined pede " + NamePede(iPar) += " vs. %s"
			+ titleAdd + ";%s;" + NamePede(iPar) += UnitPede(iPar));
    hPedeParZ->SetTitle(Form(title.Data(), "z", "z [cm]"));
    hPedeParR->SetTitle(Form(title.Data(), "r", "r [cm]"));
    hPedeParPhi->SetTitle(Form(title.Data(), "#phi", "#phi"));
//     hPedeParX->SetTitle(Form(title.Data(), "x", "x [cm]"));
    hPedeParY->SetTitle(Form(title.Data(), "y", "y [cm]"));

    fHistManager->AddHist(hPedeParR, layer+nPlot);
    fHistManager->AddHist(hPedeParZ, layer+nPlot);
    fHistManager->AddHist(hPedeParPhi, layer+nPlot);
//     fHistManager->AddHist(hPedeParX, layer+nPlot);
    fHistManager->AddHist(hPedeParY, layer+nPlot);
//     fHistManager->SetNumHistsXY(3, 2, layer+nPlot);

    ++nPlot;
  }

  fHistManager->Draw();
}

// ////////////////////////////////////////////////////////////////////////////////////////////////////
// void PlotMillePede::DrawTwoSurfaceDeltas(Option_t *option)
// {
//   const Int_t layer = this->PrepareAdd(TString(option).Contains("add", TString::kIgnoreCase));
//   const TString titleAdd = this->TitleAdd();
//
//   // TEC 7
//   const double ySplit = .6;//;//really: 0.6;
//   const double vLenH  = 20.4715/2.;
//   const double uLenH  = 8.7961/2.;
//   const double vLenH1 = (vLenH + ySplit)/2.; 
//   const double vLenH2 = (vLenH - ySplit)/2.; 
//   // derived constants
//   const double yMean1 = -vLenH + vLenH1;// y of alpha1 rotation axis in module frame
//   const double yMean2 =  vLenH - vLenH2;// y of alpha2 rotation axis in module frame
//   const double gammaScale1 = vLenH1 + uLenH;
//   const double gammaScale2 = vLenH2 + uLenH;
//   // pede parameters
//   const TString pedeU1(MpT() += Par(0));
//   const TString pedeU2(MpT() += Par(9));
//   const TString pedeW1(MpT() += Par(2));
//   const TString pedeW2(MpT() += Par(11));
//   const TString pedeUslope1(MpT() += Par(3));
//   const TString pedeUslope2(MpT() += Par(12));
//   const TString pedeVslope1(MpT() += Par(4));
//   const TString pedeVslope2(MpT() += Par(13));
//   const TString pedeRotZ1(MpT() += Par(5));
//   const TString pedeRotZ2(MpT() += Par(14));
//   // derived sensor/module orientations
//   const TString alpha1(pedeVslope1 + Div() += Flt(vLenH1));
//   const TString alpha2(pedeVslope2 + Div() += Flt(vLenH2));
//   const TString gamma1(pedeRotZ1 + Div() += Flt(gammaScale1));
//   const TString gamma2(pedeRotZ2 + Div() += Flt(gammaScale2));
//   const TString moduleGamma(Parenth(gamma1 + Plu() += gamma2) += Div() += Flt(2.));
//
//   //Delta u = (u1 + gamma*yMean1 - (u2 - gamma*yMean2))/2;
//   const TString deltaU(Parenth(Parenth(pedeU1 + Plu() += moduleGamma + Mal() += Flt(yMean1))
// 			       += Min() +=
// 			       Parenth(pedeU2 + Plu() += moduleGamma + Mal() += Flt(yMean2)))
// 		       += Div() += Flt(2.));
//
//   TString parSelDu(Valid(0) += AndL() += Valid(5) += AndL() += Valid(9) += AndL() += Valid(14)
// 		   += AndL() += Fixed(0, false) += AndL() += Fixed(5, false)
// 		   += AndL() += Fixed(5, false) += AndL() += Fixed(14, false));
//   this->AddBasicSelection(parSelDu);
//   TH1 *hTmp = this->CreateHist(deltaU + this->ToMumMuRad(0), parSelDu, "utemp");
//   const TString hNameU(this->Unique("h2BowDeltaU") 
// 		       += Form("(101,%f,%f)", hTmp->GetMean()-fMaxDev, hTmp->GetMean()+fMaxDev));
//   delete hTmp;
//   TH1 *hDeltaU = this->CreateHist(deltaU + this->ToMumMuRad(0), parSelDu, hNameU);
//   hDeltaU->SetTitle("TwoBowed #deltau" + titleAdd + ";#deltau [#mum]");
//   fHistManager->AddHist(hDeltaU, layer);
//
//   //Delta w = (w1 - alpha*yMean1 - (w2 - alpha*yMean1))/2
//   const TString deltaW(Parenth(Parenth(pedeW1 + Min() += alpha1 + Mal() += Flt(yMean1))
// 			       += Min() +=
// 			       Parenth(pedeW2 + Min() += alpha2 + Mal() += Flt(yMean2)))
// 		       += Div() += Flt(2.));
//   TString parSelDw(Valid(2) += AndL() += Valid(4) += AndL() += Valid(11) += AndL() += Valid(13)
// 		   += AndL() += Fixed( 2, false) += AndL() += Fixed( 4, false)
// 		   += AndL() += Fixed(11, false) += AndL() += Fixed(13, false));
//   this->AddBasicSelection(parSelDw);
//   hTmp = this->CreateHist(deltaW + this->ToMumMuRad(0), parSelDw, "wtemp");
//   const TString hNameW(this->Unique("h2BowDeltaW") 
// 		       += Form("(101,%f,%f)", hTmp->GetMean()-fMaxDev, hTmp->GetMean()+fMaxDev));
//   delete hTmp;
//   TH1 *hDeltaW = this->CreateHist(deltaW + this->ToMumMuRad(0), parSelDw, hNameW);
//   hDeltaW->SetTitle("TwoBowed #deltaw" + titleAdd + ";#deltaw [#mum]");
//   fHistManager->AddHist(hDeltaW, layer);
//
//   //Delta alpha = (alpha1 - alpha2)/2
//   const TString deltaA(Parenth(alpha1 + Min() += alpha2) += Div() += Flt(2.));
//   TString parSelDa(Valid(4) += AndL() += Valid(13) += AndL() +=
// 		   Fixed(4, false) += AndL() += Fixed(13, false));
//   this->AddBasicSelection(parSelDa);
//   hTmp = this->CreateHist(deltaA + this->ToMumMuRad(3), parSelDa, "atemp");
//   const TString hNameA(this->Unique("h2BowDeltaA")
// 		       += Form("(101,%f,%f)", hTmp->GetMean()-fMaxDev, hTmp->GetMean()+fMaxDev));
//   delete hTmp;
//   TH1 *hDeltaA = this->CreateHist(deltaA + this->ToMumMuRad(3), parSelDa, hNameA);
//   hDeltaA->SetTitle("TwoBowed #delta#alpha" + titleAdd + ";#delta#alpha [#murad]");
//   fHistManager->AddHist(hDeltaA, layer);
//
//   //Delta beta = (beta1 - beta2)/2
//   const TString deltaB(Parenth(pedeUslope1 + Min() += pedeUslope2) + Div() += Flt(-2.*uLenH));
//   TString parSelDb(Valid(3) += AndL() += Valid(12) += AndL() +=
// 		   Fixed(3, false) += AndL() += Fixed(12, false));
//   this->AddBasicSelection(parSelDb);
//   hTmp = this->CreateHist(deltaB + this->ToMumMuRad(3), parSelDb, "btemp");
//   const TString hNameB(this->Unique("h2BowDeltaB")
// 		       += Form("(101,%f,%f)", hTmp->GetMean()-fMaxDev, hTmp->GetMean()+fMaxDev));
//   delete hTmp;
//   TH1 *hDeltaB = this->CreateHist(deltaB + this->ToMumMuRad(3), parSelDb, hNameB);
//   hDeltaB->SetTitle("TwoBowed #delta#beta" + titleAdd + ";#delta#beta [#murad]");
//   fHistManager->AddHist(hDeltaB, layer);
//
//   //Delta gamma = (gamma1 - gamma2)/2
//   const TString deltaG(Parenth(gamma1 + Min() += gamma2) += Div() += Flt(2.));
//   TString parSelDg(Valid(5) += AndL() += Valid(14) += AndL() +=
// 		   Fixed(5, false) += AndL() += Fixed(14, false));
//   this->AddBasicSelection(parSelDg);
//   hTmp = this->CreateHist(deltaG + this->ToMumMuRad(3), parSelDg, "gtemp");
//   const TString hNameG(this->Unique("h2BowDeltaG")
// 		       += Form("(101,%f,%f)", hTmp->GetMean()-fMaxDev, hTmp->GetMean()+fMaxDev));
//   delete hTmp;
//   TH1 *hDeltaG = this->CreateHist(deltaG + this->ToMumMuRad(3), parSelDg, hNameG);
//   hDeltaG->SetTitle("TwoBowed #delta#gamma" + titleAdd + ";#delta#gamma [#murad]");
//   fHistManager->AddHist(hDeltaG, layer);
//
//   TLine *lU = new TLine(-100., 0., -100., hDeltaU->GetMaximum() * 1.05);
//   lU->SetLineColor(kRed);
//   lU->SetLineWidth(3);
//   fHistManager->AddObject(lU, layer, 0);
//
//   TLine *lW = new TLine(-150., 0., -150., hDeltaW->GetMaximum() * 1.05);
//   lU->TAttLine::Copy(*lW);
//   fHistManager->AddObject(lW, layer, 1);
//
//   TLine *lA = new TLine(-150., 0., -150., hDeltaA->GetMaximum() * 1.05);
//   lU->TAttLine::Copy(*lA);
//   fHistManager->AddObject(lA, layer, 2);
//
//   TLine *lB = new TLine(-300., 0., -300., hDeltaB->GetMaximum() * 1.05);
//   lU->TAttLine::Copy(*lB);
//   fHistManager->AddObject(lB, layer, 3);
//
//   TLine *lG = new TLine(-250., 0., -250., hDeltaG->GetMaximum() * 1.05);
//   lU->TAttLine::Copy(*lG);
//   fHistManager->AddObject(lG, layer, 4);
//
//   fHistManager->Draw();
// }

////////////////////////////////////////////////////////////////////////////////////////////////////
void PlotMillePede::DrawSurfaceDeformations(const TString &whichOne, Option_t *option,
					    unsigned int maxNumPars, unsigned int firstPar)
{
  const Int_t layer = this->PrepareAdd(TString(option).Contains("add", TString::kIgnoreCase));
  const TString titleAdd = this->TitleAdd();
  const bool noLimit = TString(option).Contains("nolimit", TString::kIgnoreCase);

  TString parSel(Valid(0) += AndL() += Fixed(0, false)); // HACK: if u1 determination is fine
  if (TString(option).Contains("all", TString::kIgnoreCase)) parSel = "";
  this->AddBasicSelection(parSel);

  TObjArray whichOnes;
  if (whichOne.Contains("result", TString::kIgnoreCase)) whichOnes.Add(new TObjString("result"));
  if (whichOne.Contains("start",  TString::kIgnoreCase)) whichOnes.Add(new TObjString("start"));
  if (whichOne.Contains("diff",   TString::kIgnoreCase)) whichOnes.Add(new TObjString("diff"));

  for (Int_t wi = 0; wi < whichOnes.GetEntriesFast(); ++wi) {
    unsigned int nPlot = 0;
    for (unsigned int i = firstPar; i < maxNumPars; ++i) {
      TString hName(this->Unique(Form("hSurf%s%u", whichOnes[wi]->GetName(),i)));
      if (!noLimit) hName += Form("(%d,%f,%f)", fNbins, fMaxDevDown, fMaxDevUp);
      TH1 *h = this->CreateHist(DeformValue(i, whichOnes[wi]->GetName()) += this->ToMumMuRadSurfDef(i),
				parSel + AndL() += Parenth(NumDeformValues(whichOnes[wi]->GetName())),
				hName);
      
      if (!h || 0. == h->GetEntries()) continue;
      h->SetTitle(Form("SurfaceDeformation %s ", whichOnes[wi]->GetName())
		  + NameSurfDef(i) += titleAdd + ";"
		  + NameSurfDef(i) += UnitSurfDef(i));
      fHistManager->AddHistSame(h, layer, nPlot, whichOnes[wi]->GetName());
      //fHistManager->AddHistSame(hBef, layer, nPlot, "misaligned");
      ++nPlot;
    }
  }

  whichOnes.Delete();
  const bool old = fHistManager->SameWithStats(true);
  fHistManager->Draw();
  fHistManager->SameWithStats(old);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
void PlotMillePede::DrawSurfaceDeformationsVsLocation(const TString &whichOne,
						      Option_t *option,
						      unsigned int maxNumPar,
						      unsigned int firstPar)
{
  const Int_t layer = this->PrepareAdd(TString(option).Contains("add", TString::kIgnoreCase));
  const TString titleAdd = this->TitleAdd();

  // TObjArray whichOnes;
  // if (whichOne.Contains("result", TString::kIgnoreCase)) whichOnes.Add(new TObjString("result"));
  // if (whichOne.Contains("start",  TString::kIgnoreCase)) whichOnes.Add(new TObjString("start"));
  // if (whichOne.Contains("diff",   TString::kIgnoreCase)) whichOnes.Add(new TObjString("diff"));
  // for (Int_t wi = 0; wi < whichOnes.GetEntriesFast(); ++wi) {
  UInt_t nPlot = 0;
  for (UInt_t iPar = firstPar; iPar <= maxNumPar; ++iPar) { // 
    TString parSel(Valid(0) += AndL() += Fixed(0, false)); // HACK: if u1 determination is fine
    if (TString(option).Contains("all", TString::kIgnoreCase)) parSel = "";
    this->AddBasicSelection(parSel);

    //TString hNameR(this->Unique(Form("hSurfR%s%u", whichOnes[wi]->GetName(),i)));
    TString hNameR(this->Unique(Form("hSurfR%u", iPar)));

    TH1 *hR = this->CreateHist2D(RPos(OrgPosT()),
				 DeformValue(iPar, whichOne) += this->ToMumMuRadSurfDef(iPar),
				 parSel + AndL() += Parenth(NumDeformValues(whichOne)),
				 hNameR, "BOX");

    if (!hR || 0. == hR->GetEntries()) continue;

    TString hNameZ(this->Unique(Form("hSurfZ%u", iPar)));

    TH1 *hZ = this->CreateHist2D(OrgPosT() += ZPos(),
				 DeformValue(iPar, whichOne) += this->ToMumMuRadSurfDef(iPar),
				 parSel + AndL() += Parenth(NumDeformValues(whichOne)),
				 hNameZ, "BOX");


    const TString title("Surface deformation " + NameSurfDef(iPar) += " vs. %s"
			+ titleAdd + ";%s;" + NameSurfDef(iPar) += UnitSurfDef(iPar));
    hR->SetTitle(Form(title.Data(), "r", "r [cm]"));
    hZ->SetTitle(Form(title.Data(), "z", "z [cm]"));

    fHistManager->AddHist(hR, layer+nPlot);
    fHistManager->AddHist(hZ, layer+nPlot);
    ++nPlot;
  }

  fHistManager->Draw();
}

////////////////////////////////////////////////////////////////////////////////////////////////////
void PlotMillePede::DrawSurfaceDeformationsLayer(Option_t *option, const unsigned int firstDetLayer,
						 const unsigned int lastDetLayer,
						 const TString &whichOne, unsigned int maxNumPars)
{
  const TString opt(option);
  const Int_t firstLayer = this->PrepareAdd(opt.Contains("add", TString::kIgnoreCase));
  const bool noLimit = opt.Contains("nolimit", TString::kIgnoreCase);
  const bool spread = opt.Contains("spread", TString::kIgnoreCase);
  const bool verbose = opt.Contains("verbose", TString::kIgnoreCase);
  const bool verbose2 = opt.Contains("verboseByParam", TString::kIgnoreCase);

  this->SetDetLayerCuts(0, false); // just to generate warnings if needed!
  // loop on deformation parameters
  unsigned int iParUsed = 0;
  for (unsigned int iPar = 0; iPar < maxNumPars; ++iPar) {
    // create a hist to store the averages per layer
    const unsigned int numDetLayers = lastDetLayer - firstDetLayer + 1;
    TH1 *layerHist = new TH1F(this->Unique("hSurfAll" + whichOne += iPar),
			      "Average deformations " + NameSurfDef(iPar)
			      += ";;#LT" + (NameSurfDef(iPar) += "#GT") += UnitSurfDef(iPar),
			      numDetLayers, 0, numDetLayers);
    TH1 *layerHistWithSpread = 
      (spread ? static_cast<TH1*>(layerHist->Clone(Form("%s_spread", layerHist->GetName()))) : 0);
    if (spread) layerHistWithSpread->SetTitle(Form("%s (err is spread)", layerHist->GetTitle()));

    TH1 *layerHistRms = new TH1F(this->Unique("hSurfAllRms" + whichOne += iPar),
			      "RMS of deformations " + NameSurfDef(iPar)
			      += ";;RMS(" + (NameSurfDef(iPar) += ")") += UnitSurfDef(iPar),
			      numDetLayers, 0, numDetLayers);


    // loop on layers (i.e. subdet layers/rings)
    unsigned int iDetLayerUsed = 0;
    for (unsigned int iDetLayer = firstDetLayer; iDetLayer <= lastDetLayer; ++iDetLayer) {
      if (!this->SetDetLayerCuts(iDetLayer, true)) continue; // layer cuts implemented?
      TString sel; //(Valid(0) += AndL() += Fixed(0, false)); // HACK: if u1 determination is fine
      this->AddBasicSelection(sel); // append the cuts set
      // histo name with or without predefined limits:
      const TString hName(this->Unique(Form("hSurf%s%u_%u", whichOne.Data(), iPar, iDetLayer))
			  += (noLimit ? "" : Form("(%d,%f,%f)", fNbins, fMaxDevDown, fMaxDevUp)));
      // cut away values identical to zero:
      TH1 *h = this->CreateHist(DeformValue(iPar, whichOne) += this->ToMumMuRadSurfDef(iPar),
				(sel + AndL()) += Parenth(DeformValue(iPar, whichOne) += "!= 0."),
				hName);

      if (!h || 0. == h->GetEntries()) continue; // did something survive cuts?
      ++iDetLayerUsed;
      layerHist->GetXaxis()->SetBinLabel(iDetLayerUsed, this->DetLayerLabel(iDetLayer));
      layerHist->SetBinContent(iDetLayerUsed, h->GetMean());
      layerHist->SetBinError(iDetLayerUsed, h->GetMeanError()); //GetRMS());
      layerHistRms->GetXaxis()->SetBinLabel(iDetLayerUsed, this->DetLayerLabel(iDetLayer));
      layerHistRms->SetBinContent(iDetLayerUsed, h->GetRMS());
      if (spread) layerHistWithSpread->SetBinContent(iDetLayerUsed, h->GetMean());
      if (spread) layerHistWithSpread->SetBinError(iDetLayerUsed, h->GetRMS());
      if (verbose) {
	h->SetTitle(("SurfaceDeformation " + whichOne) += " "
		    + NameSurfDef(iPar) += this->TitleAdd() + ";"
		    + NameSurfDef(iPar) += UnitSurfDef(iPar));
	if (verbose2) fHistManager->AddHist(h, firstLayer + 2 + iPar);
	else          fHistManager->AddHist(h, firstLayer + 2 + iDetLayerUsed);
      } else delete h;
    }
    layerHist->LabelsDeflate(); // adjust to avoid empty bins
    if (spread) layerHistWithSpread->LabelsDeflate(); // dito
    layerHistRms->LabelsDeflate();        // dito
    if (layerHist->GetEntries()) {
      fHistManager->AddHistSame(layerHist, firstLayer, iParUsed,
				(spread ? "error: #sigma(mean)" : ""));
      if (spread) fHistManager->AddHistSame(layerHistWithSpread, firstLayer, iParUsed,
					    "error: RMS");
      fHistManager->AddHist(layerHistRms, firstLayer + 1);

      ++iParUsed;
    } else {
      delete layerHist; // v^{delta} usually is empty...
      delete layerHistRms;
    }
  }

  fHistManager->Draw();
  // now remove cuts implicitely set by SetDetLayerCuts(..):
  this->SetSubDetId(-1);
  this->ClearAdditionalSel();
}

////////////////////////////////////////////////////////////////////////////////////////////////////
bool PlotMillePede::SetDetLayerCuts(unsigned int detLayer, bool silent)
{
  if (!silent) {
    // warn if setting subdet/r/z below overwrites any general settings
    if (fAdditionalSelTitle.Length()) {
      ::Warning("PlotMillePede::SetDetLayerCuts",
		"Clearing selection '%s'!", fAdditionalSelTitle.Data());
    }
    if (fSubDetIds.GetSize()) {
      ::Warning("PlotMillePede::SetDetLayerCuts", "Possibly changing subdet selection!");
    }
  }
  this->ClearAdditionalSel();

  switch (detLayer) {
  case 0: // BPIX L1
    this->SetSubDetId(1);
    this->AddAdditionalSel("r", 0., 5.5);
    return true;
  case 1: // BPIX L2
    this->SetSubDetId(1);
    this->AddAdditionalSel("r", 5.5, 8.5);
    return true;
  case 2: // BPIX L3
    this->SetSubDetId(1);
    this->AddAdditionalSel("r", 8.5, 12.);
    return true;
    // FPIX not implemented
  case 3:
  case 4:
    this->SetSubDetId(2);
    return false;
    break;

  case 5: // TIB L1 rphi
    this->SetSubDetId(3);
    this->AddAdditionalSel("r", 20., 30.);
    this->AddAdditionalSel("StripRphi");
    return true;
  case 6: // TIB L1 stereo
    this->SetSubDetId(3);
    this->AddAdditionalSel("r", 20., 30.);
    this->AddAdditionalSel("StripStereo");
    return true;
  case 7: // TIB L2 rphi
    this->SetSubDetId(3);
    this->AddAdditionalSel("r", 30., 38.);
    this->AddAdditionalSel("StripRphi");
    return true;
  case 8: // TIB L2 stereo
    this->SetSubDetId(3);
    this->AddAdditionalSel("r", 30., 38.);
    this->AddAdditionalSel("StripStereo");
    return true;
  case 9: // TIB L3
    this->SetSubDetId(3);
    this->AddAdditionalSel("r", 38., 46.);
    return true;
  case 10: // TIB L4
    this->SetSubDetId(3);
    this->AddAdditionalSel("r", 46., 55.);
    return true;

  case 11: // TID R1 rphi
    this->SetSubDetId(4);
    this->AddAdditionalSel("r", 20., 33.);
    this->AddAdditionalSel("StripRphi");
    return true;
  case 12: // TID R1 stereo
    this->SetSubDetId(4);
    this->AddAdditionalSel("r", 20., 33.);
    this->AddAdditionalSel("StripStereo");
    return true;
  case 13: // TID R2 rphi
    this->SetSubDetId(4);
    this->AddAdditionalSel("r", 33., 41.);
    this->AddAdditionalSel("StripRphi");
    return true;
  case 14: //"TID R2 S";
    this->SetSubDetId(4);
    this->AddAdditionalSel("r", 33., 41.);
    this->AddAdditionalSel("StripStereo");
    return true;
  case 15: //"TID R3";
    this->SetSubDetId(4);
    this->AddAdditionalSel("r", 41., 50.);
    return true;

    // TID followed by TEC and not TOB to be able to call
    // DrawSurfaceDeformationsLayer("", n1, n2) for layers with single
    // (n1=0 and n2=21) and double (n1=22 and n2=33) sensor modules separately
  case 16:  // TEC R1 #phi
    this->SetSubDetId(6);
    this->AddAdditionalSel("r", 20., 33.);
    this->AddAdditionalSel("StripRphi");
    return true;
  case 17: // TEC R1 S
    this->SetSubDetId(6);
    this->AddAdditionalSel("r", 20., 33.);
    this->AddAdditionalSel("StripStereo");
    return true;
  case 18: // TEC R2 #phi
    this->SetSubDetId(6);
    this->AddAdditionalSel("r", 33., 41.);
    this->AddAdditionalSel("StripRphi");
    return true;
  case 19: //"TEC R2 S";
    this->SetSubDetId(6);
    this->AddAdditionalSel("r", 33., 41.);
    this->AddAdditionalSel("StripStereo");
    return true;
  case 20: //"TEC R3";
    this->SetSubDetId(6);
    this->AddAdditionalSel("r", 41., 50.);
    return true;
  case 21: //"TEC R4";
    this->SetSubDetId(6);
    this->AddAdditionalSel("r", 50., 60.);
    return true;
  case 22: //"TEC R5 R";
    this->SetSubDetId(6);
    this->AddAdditionalSel("r", 60., 75.);
    this->AddAdditionalSel("StripRphi");
    return true;
  case 23: //"TEC R5 S";
    this->SetSubDetId(6);
    this->AddAdditionalSel("r", 60., 75.);
    this->AddAdditionalSel("StripStereo");
    return true;
  case 24: //"TEC R6";
    this->SetSubDetId(6);
    this->AddAdditionalSel("r", 75., 90.);
    return true;
  case 25: //"TEC R7";
    this->SetSubDetId(6);
    this->AddAdditionalSel("r", 90., 120.);
    return true;

  case 26: // TOB L1 R
    this->SetSubDetId(5);
    this->AddAdditionalSel("r", 50., 65.);
    this->AddAdditionalSel("StripRphi");
    return true;
  case 27: // TOB L1 S
    this->SetSubDetId(5);
    this->AddAdditionalSel("r", 50., 65.);
    this->AddAdditionalSel("StripStereo");
    return true;
  case 28: // TOB L2 R
    this->SetSubDetId(5);
    this->AddAdditionalSel("r", 65., 73.);
    this->AddAdditionalSel("StripRphi");
    return true;
  case 29: // TOB L2 S
    this->SetSubDetId(5);
    this->AddAdditionalSel("r", 65., 73.);
    this->AddAdditionalSel("StripStereo");
    return true;
  case 30: // TOB L3
    this->SetSubDetId(5);
    this->AddAdditionalSel("r", 73., 82.5);
    return true;
  case 31: // TOB L4
    this->SetSubDetId(5);
    this->AddAdditionalSel("r", 82.5, 92.);
    return true;
  case 32:// TOB L5
    this->SetSubDetId(5);
    this->AddAdditionalSel("r", 92., 102.);
    return true;
  case 33:// TOB L6
    this->SetSubDetId(5);
    this->AddAdditionalSel("r", 102., 120.);
    return true;
  }

  return false;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
TString PlotMillePede::DetLayerLabel(unsigned int detLayer) const
{
  switch (detLayer) {
  case 0: return "BPIX L1";
  case 1: return "BPIX L2";
  case 2: return "BPIX L3";
    //   case 3: return "FPIX";
    //   case 4: return "FPIX";
    
  case 5: return "TIB L1#phi";//"TIB L1R";
  case 6: return "TIB L1s";   //"TIB L1S";
  case 7: return "TIB L2#phi";//"TIB L2R";
  case 8: return "TIB L2s";   //"TIB L2S";
  case 9: return "TIB L3";
  case 10: return "TIB L4";

  case 11: return "TID R1#phi";//"TID R1R";
  case 12: return "TID R1s";   //"TID R1S";
  case 13: return "TID R2#phi";//"TID R2R";
  case 14: return "TID R2s";   //"TID R2S";
  case 15: return "TID R3";

  case 16: return "TEC R1#phi";//"TEC R1R";
  case 17: return "TEC R1s";   //"TEC R1S";
  case 18: return "TEC R2#phi";//"TEC R2R";
  case 19: return "TEC R2s";   //"TEC R2S";
  case 20: return "TEC R3";
  case 21: return "TEC R4";
  case 22: return "TEC R5#phi";//"TEC R5R";
  case 23: return "TEC R5s";   //"TEC R5S";
  case 24: return "TEC R6";
  case 25: return "TEC R7";

  case 26: return "TOB L1#phi";//"TOB L1R";
  case 27: return "TOB L1s";   //"TOB L1S";
  case 28: return "TOB L2#phi";//"TOB L2R";
  case 29: return "TOB L2s";   //"TOB L2S";
  case 30: return "TOB L3";
  case 31: return "TOB L4";
  case 32: return "TOB L5";
  case 33: return "TOB L6";
  }

  return Form("unknown DetLayer %u", detLayer);
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

  const TString posNames[] = {"phi", "r", "x", "y", "z"};
  for (UInt_t iPos = 0; iPos < sizeof(posNames)/sizeof(posNames[0]); ++iPos) {
    const TString &pos = posNames[iPos];
    TH1 *h = this->CreateHist(OrgPos(pos), aSel, this->Unique("org_"+pos));
    h->SetTitle("original position " + Name(pos) + titleAdd + ";"
		+ Name(pos) + ";#alignables"); 
    fHistManager->AddHist(h, layer);
  }

  fHistManager->Draw();
}

////////////////////////////////////////////////////////////////////////////////////////////////////
void PlotMillePede::DrawParamResult(Option_t *option)
{
  const TString opt(option);
  const Int_t layer = this->PrepareAdd(opt.Contains("add", TString::kIgnoreCase));

  const TString titleAdd = this->TitleAdd();
  UInt_t nPlot = 0;
  for (UInt_t iPar = 0; iPar < kNpar; ++iPar) { // 
    TString sel(opt.Contains("withfixed", TString::kIgnoreCase) ? "" : Fixed(iPar,false).Data());
    this->AddBasicSelection(sel);

    const TString toMu(this->ToMumMuRad(iPar));
    const TString finalMis(this->FinalMisAlignment(iPar) += toMu);
    const TString startMis(Parenth(MisParT() += Par(iPar)) += toMu);

    const TString hNameB(this->Unique(Form("before%d", iPar)) += 
			 Form("(%d,%f,%f)", fNbins, fMaxDevDown, fMaxDevUp));
    TH1 *hBef = this->CreateHist(startMis, sel, hNameB);
    const TString hNameD(this->Unique(Form("end%d", iPar)) 
                         += Form("(%d,%f,%f)", hBef->GetNbinsX(), 
                                 hBef->GetXaxis()->GetXmin(), hBef->GetXaxis()->GetXmax()));
    TH1 *hEnd = this->CreateHist(finalMis, sel, hNameD);
    const TString hName2D(this->Unique(Form("vs%d", iPar)) += Form("(30,%f,%f,30,-500,500)", fMaxDevDown, fMaxDevUp));
    TH1 *hVs = this->CreateHist(startMis + ":" + finalMis, sel, hName2D, "BOX");
    if (0. == hEnd->GetEntries()) continue;
    hEnd->SetTitle(DelName(iPar)+=titleAdd+";"+DelNameU(iPar)+=";#parameters");
    hVs->SetTitle(DelName(iPar)+=titleAdd+";" + DelNameU(iPar)+="(end);" + DelNameU(iPar) 
		  += "(start)");
    if (this->GetTitle().Length() != 0) {
      fHistManager->AddHist(hEnd, layer, this->GetTitle());
    } else {
      fHistManager->AddHist(hEnd, layer, "remaining misal.");
    }
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
			 += Form("(%d,%f,%f)", fNbins, fMaxDevDown, fMaxDevUp));
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
    if (this->GetTitle().Length() != 0) {
      fHistManager->AddHist(hEnd, layer + 1, this->GetTitle());
    } else {
      fHistManager->AddHist(hEnd, layer + 1, "remaining misal.");
    }

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
    hProfParR->SetTitle("#LT" + DelName(iPar) += "#GT vs. r" + titleAdd + ";r[cm];" + DelNameU(iPar));
    hProfParZ->SetTitle("#LT" + DelName(iPar) += "#GT vs. z" + titleAdd + ";z[cm];" + DelNameU(iPar));
    hProfParPhi->SetTitle("#LT" + DelName(iPar) += "#GT vs. #phi" + titleAdd + ";#phi;" + DelNameU(iPar));
//     hProfParTheta->SetTitle("#LT" + DelName(iPar) += "#GT vs. #theta" + titleAdd + ";#theta;" + DelNameU(iPar));
    if (hProfParAl0)
      hProfParAl0->SetTitle("#LT" + DelName(iPar) += Form("#GT vs. euler #alpha^{%d};#alpha^{%d};",
							vsEuler, vsEuler) + DelNameU(iPar));
    if (hProfParBet0)
      hProfParBet0->SetTitle("#LT" + DelName(iPar) += Form("#GT vs. euler #beta^{%d};#beta^{%d};",
							 vsEuler, vsEuler) + DelNameU(iPar));
    if (hProfParGam0)
      hProfParGam0->SetTitle("#LT" + DelName(iPar) += Form("#GT vs. euler #gamma^{%d};#gamma^{%d};",
							 vsEuler, vsEuler) + DelNameU(iPar));
    if (addStartMis) {
      hProfParStartR->SetTitle("#LT" + DelName(iPar) += "#GT vs. r (start);r[cm];" + DelNameU(iPar));
      hProfParStartZ->SetTitle("#LT" + DelName(iPar) += "#GT vs. z (start);z[cm];" + DelNameU(iPar));
      hProfParStartPhi->SetTitle("#LT" + DelName(iPar) += "#GT vs. #phi (start);#phi;" + DelNameU(iPar));
//       hProfParStartTheta->SetTitle("#LT" + DelName(iPar) += "#GT vs. #theta;#theta;" + DelNameU(iPar));
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

  const TString posNames[] = {"rphi", "r", "z", "x", "y", "phi"};

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
    hProfPosR->SetTitle("#LT" + DelName(posName) += "#GT vs. r" + titleAdd + ";r[cm];" + DelNameU(posName));
    hProfPosZ->SetTitle("#LT" + DelName(posName) += "#GT vs. z" + titleAdd + ";z[cm];" + DelNameU(posName));
    hProfPosPhi->SetTitle("#LT" + DelName(posName) += "#GT vs. #phi" + titleAdd + ";#phi;" + DelNameU(posName));
    hProfPosY->SetTitle("#LT" + DelName(posName) += "#GT vs. y" + titleAdd + ";y[cm];" + DelNameU(posName));
    if (addStart) {
      hProfPosStartR->SetTitle("#LT" + DelName(posName)+="#GT vs. r (start)" + titleAdd + ";r[cm];"+DelNameU(posName));
      hProfPosStartZ->SetTitle("#LT" + DelName(posName) += "#GT vs. z" + titleAdd + ";z[cm];" + DelNameU(posName));
      hProfPosStartPhi->SetTitle("#LT" + DelName(posName) += "#GT vs. #phi" + titleAdd + ";#phi;" + DelNameU(posName));
      hProfPosStartY->SetTitle("#LT" + DelName(posName) += "#GT vs. y" + titleAdd + ";y[cm];" + DelNameU(posName));
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
  hRxProf->SetTitle("#LT#hits_{x}> vs. r" + titleAdd + ";r[cm];N_{hit,x}");
  hRy->SetTitle("#hits_{y} vs. r" + titleAdd + ";r[cm];N_{hit,y}");
  hRyProf->SetTitle("#LT#hits_{y}> vs. r" + titleAdd + ";r[cm];N_{hit,y}");
  hZx->SetTitle("#hits_{x} vs. z" + titleAdd + ";z[cm];N_{hit,x}");
  hZxProf->SetTitle("#LT#hits_{x}> vs. z" + titleAdd + ";z[cm];N_{hit,x}");
  hZy->SetTitle("#hits_{y} vs. z" + titleAdd + ";z[cm];N_{hit,y}");
  hZyProf->SetTitle("#LT#hits_{y}> vs. z" + titleAdd + ";z[cm];N_{hit,y}");
  hPhiX->SetTitle("#hits_{x} vs. #phi" + titleAdd + ";#phi;N_{hit,x}");
  hPhixProf->SetTitle("#LT#hits_{x}> vs. #phi" + titleAdd + ";#phi;N_{hit,x}");
  hPhiY->SetTitle("#hits_{y} vs. #phi" + titleAdd + ";#phi;N_{hit,y}");
  hPhiyProf->SetTitle("#LT#hits_{y}> vs. #phi" + titleAdd + ";#phi;N_{hit,y}");

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
    hAll->SetTitle("subDetId" + titleAdd + ";ID(subdet)");
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
    if (titleAdd.Length()) {
      std::cout << "Active selection: " << titleAdd << std::endl;
    }
  }

  const TString mpPar(MpT() += Par());// += this->ToMumMuRad(iPar));
  //  this->GetMainTree()->Scan("Id:Pos:" + mpPar += Form(":HitsX:Sigma[%u]:Label", iPar), sel);
  TString scan("Id:Pos:" + mpPar += ":HitsX:Sigma:Label");
  if (addColumns) scan += addColumns;
  this->GetMainTree()->Scan(scan, realSel);
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

  const TString allTreeNames[] = {OrgPosT(), MisPosT(), PosT(),
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
void PlotMillePede::SetMaxDev(Float_t maxDev)
{
  // set symmetric x-axis range for result plots (around 0)
  fMaxDevUp   =  TMath::Abs(maxDev);
  fMaxDevDown = -TMath::Abs(maxDev);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
void PlotMillePede::SetMaxDev(Float_t maxDevDown, Float_t maxDevUp)
{
  // set x-axis range for result plots
  if (maxDevUp < maxDevDown) {
    ::Error("PlotMillePede::SetMaxDev",
            "Upper limit %f smaller than lower limit %f => Swap them!", maxDevUp, maxDevDown);     
    fMaxDevUp   = maxDevDown;
    fMaxDevDown = maxDevUp;
  } else {
    fMaxDevUp   = maxDevUp;
    fMaxDevDown = maxDevDown;
  }
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
void PlotMillePede::SetSubDetIds(Int_t id1, Int_t id2, Int_t id3, Int_t id4, Int_t id5)
{
  this->SetSubDetId(id1);
  const Int_t ids[] = {id2, id3, id4, id5};
  for (unsigned int i = 0; i < sizeof(ids)/sizeof(ids[0]); ++i) {
    if (ids[i] > 0) this->AddSubDetId(ids[i]);
  }
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
    //result += Form("type %d", fAlignableTypeId);
    result += this->AlignableObjIdString(fAlignableTypeId);
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
TString PlotMillePede::AlignableObjIdString(Int_t objId) const
{
  switch (objId) { // see StructureType.h in CMSSW
  case 1: return "DetUnit";
  case 2: return "Det";
    //
  case 5: return "BPIXLayer";
  case 6: return "BPIXHalfBarrel";
    //
  case 11: return "FPIXHalfDisk";
    //
  case 15: return "TIBString";
    //
  case 19: return "TIBHalfBarrel";
  case 20: return "TIBBarrel";
    //
  case 25: return "TIDEndcap";
    //
  case 29: return "TOBHalfBarrel";
  case 30: return "TOBBarrel";
    //
  case 36: return "TECEndcap";
  default: 
    ::Error("PlotMillePede::AlignableObjIdString",
            "Missing implementation for ObjId %d, see "
	    "Alignment/CommonAlignment/interface/StructureType.h",
	    objId);
    return Form("alignable obj id %d", objId);
  }
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
    // stereo/rphi etc. selections:
    if (sel == "StripDoubleOr1D") {
      fAdditionalSel += "(Id&3)==0";
      fAdditionalSelTitle += "Double sided or 1D layer/ring";
    } else if (sel == "StripRphi") {
      fAdditionalSel += "(Id&3)==2";
      fAdditionalSelTitle += "R#phi";
    } else if (sel == "StripStereo"){
      fAdditionalSel += "(Id&3)==1";
      fAdditionalSelTitle += "Stereo";
    // anti stereo/rphi etc. selections:
    } else if (sel == "NotStripDoubleOr1D") {
      fAdditionalSel += "(Id&3)!=0";
      fAdditionalSelTitle += "!(Double sided or 1D layer/ring)";
    } else if (sel == "NotStripRphi") {
      fAdditionalSel += "(Id&3)!=2";
      fAdditionalSelTitle += "!R#phi";
    } else if (sel == "NotStripStereo"){
      fAdditionalSel += "(Id&3)!=1";
      fAdditionalSelTitle += "!Stereo";
      // genericaly add
    } else {
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
    this->AddAdditionalSel(HitsX() += Form(">=%f && ", min) + HitsX() += Form("<%f", max));
  } else {
    this->AddAdditionalSel(OrgPos(xyzrPhiNhit) += Form(">=%f && ", min) 
			   + OrgPos(xyzrPhiNhit) += Form("<%f", max));
  }
  // add to title in readable format
  fAdditionalSelTitle = oldTitle; // first remove what was added in unreadable format...
  if (fAdditionalSelTitle.Length()) fAdditionalSelTitle += ", ";
  fAdditionalSelTitle += Form("%g #leq %s < %g", min, xyzrPhiNhit.Data(), max);
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

////////////////////////////////////////////////////////////////////////////////////////////////////
void PlotMillePede::SetOutName(const TString& name) {
  fHistManager->SetCanvasName(name);
}
