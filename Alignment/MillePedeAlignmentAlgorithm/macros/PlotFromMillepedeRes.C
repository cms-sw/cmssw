#include "TString.h"
#include "TFile.h"
#include "TTree.h"
#include "TH1F.h"
#include "THStack.h"
#include "TCanvas.h"
#include "TLegend.h"
#include "TStyle.h"
#include "TPad.h"
#include "TGraphErrors.h"
#include <stdio.h>
#include <iostream>
#include <iomanip>

enum {PXB,PXF,TIB,TID,TOB,TEC};
TString subdLabels[6]={"PXB","PXF","TIB","TID","TOB","TEC"};
enum {PULLS, NHITS, PARS, PARSwithERRvsLABEL};

//###########################################################

TString StrPlotType(int plotType){
  TString str;
  if (plotType==PULLS) return "PULLS";
  if (plotType==NHITS) return "NHITS";
  if (plotType==PARS) return "PARS";
  if (plotType==PARSwithERRvsLABEL) return "PARSwithERRvsLABEL";
  return "UNKNOWN";
}//end of StrPlotType

//###########################################################

TString StrPar(int parIndLocal){
  TString str;
  std::cout<<"parIndLocal="<<parIndLocal<<std::endl;
  switch (parIndLocal){
    case 1: str="u"; break;
    case 2: str="v"; break;
    case 3: str="w"; break;
    case 4: str="alpha"; break;
    case 5: str="beta"; break;
    case 6: str="gamma"; break;
    case 7: str="def1"; break;
    case 8: str="def2"; break;
    case 9: str="def3"; break;
    default: str="UNKNOWN"; break;
  }// end of switch
  return str;
}//end of StrPar

//###########################################################

TString StrCutSubd(int isubd){
  if (isubd==PXB)  return "label>61 && label<8781";//TPB (PXB)
  if (isubd==PXF)  return "label>17541 && label<34561";//TPE (PXF)
  if (isubd==TIB)  return "label>37021 && label<79041";//TIB
  if (isubd==TID)  return "label>121061 && label<132721";//TID
  if (isubd==TOB)  return "label>144401 && label<214301";//TOB
  if (isubd==TEC)  return "label>284201 && label<380121";//TEC
  return "UNKNOWN";
}//end of StrCutSubd

//###########################################################

void PlotParValVsLabelWithErr(TFile* f, TTree* tr, TString strMillepedeRes, TString strOutdir)
{
 
  f->cd();
  TString canvName="c_";
  canvName+=strMillepedeRes;
  canvName+="_";
  canvName+=StrPlotType(PARSwithERRvsLABEL);
  canvName.ReplaceAll(".res","");

  TCanvas* canv = new TCanvas(canvName,canvName,900,600);
  canv->Divide(3,2);

  for (int ind=1; ind<=6; ind++){
    canv->cd(ind);
    TPad* pad = (TPad*)canv->GetPad(ind);
    TString strCut="((label%20-1)%9+1)==";
    strCut+=ind;
    int n = tr->Draw("label%700000:10000*parVal:10000*parErr:0.01*(label%700000)",strCut,"goff");
    TGraphErrors *gr = new TGraphErrors(n,tr->GetV1(),tr->GetV2(),tr->GetV4(),tr->GetV3());
    gr->SetMarkerStyle(20);
    gr->SetLineWidth(2);
    for (int i=0; i<n; i++){
      std::cout<<tr->GetV1()[i]<<" "<<tr->GetV2()[i]<<"+-"<<tr->GetV3()[i]<<std::endl;
    }
    gr->SetTitle(StrPar(ind)+TString(", 10000*(par+-err)"));
    gr->GetXaxis()->SetTitle("label%700000");
    gr->Draw("AP");
  }// end of loop over ind
  canvName+=".png";
  TString saveName=strOutdir+canvName;
  canv->SaveAs(saveName);
  saveName.ReplaceAll(".png",".pdf");
  canv->SaveAs(saveName);

}// end of PlotParValVsLabelWithErr

//###########################################################

void PlotHistsNhitsPerModule(TFile* f, TTree* tr, TString strMillepedeRes, TString strOutdir)
{
  TString canvName="c_";
  canvName+=strMillepedeRes;
  canvName+="_";
  canvName+=StrPlotType(NHITS);
  canvName.ReplaceAll(".res","");


  //enum {PXB,PXF,TIB,TID,TOB,TEC};
  int colors[6]={1,2,3,4,6,7};
//  TString labels[6]={"PXB","PXF","TIB","TID","TOB","TEC"};

  f->cd();
  TCanvas* canv = new TCanvas(canvName,canvName,600,600);
  canv->SetLogx();
  canv->SetLogy();

  for (int ind=1; ind<=1; ind++){
    TString strHist = "hNhits_";
    strHist+=StrPar(ind);
    TString strCut="label<700000 && ((label%20-1)%9+1)==";
    strCut+=ind;
    TStyle style; 
    style.SetTitleFontSize(0.2);
    THStack *hSt = new THStack("hNhits","# of derivatives (~tracks or hits) per module");
    TLegend *leg = new TLegend(0.75,0.65,0.95,0.95);
    for (int inv=0; inv<6; inv++){
      std::cout<<"- - - - - -"<<std::endl;
      std::cout<<subdLabels[inv]<<":"<<std::endl;
      std::cout<<StrCutSubd(inv)<<": "<<tr->GetEntries(StrCutSubd(inv))<<" parameters"<<std::endl;
      TString strHist1=strHist;
      strHist1+=ind;
      strHist1+=inv;
      TH1F* hValInt = new TH1F(strHist1,strHist1,300,10,15000);  
      TString strCut1 = strCut+TString(" && ")+StrCutSubd(inv);
      tr->Draw(TString("Nhits>>")+strHist1,strCut1,"goff");
      std::cout<<"# hits = "<<(int)hValInt->GetMean()<<"+-"<<(int)hValInt->GetRMS()<<std::endl;
      hValInt->SetLineColor(1);
      hValInt->SetFillColor(colors[inv]);
      hValInt->SetLineWidth(2);
      hSt->Add(hValInt);
      leg->AddEntry(hValInt,subdLabels[inv],"f");
      leg->SetFillColor(0);
    }
    hSt->Draw();
    leg->Draw("same");
    
  }//end of loop over ind

  canvName+=".png";
  TString saveName=strOutdir+canvName;
  canv->SaveAs(saveName);
  saveName.ReplaceAll(".png",".pdf");
  canv->SaveAs(saveName);
}//end of PlotHistsNhitsPerModule

//###########################################################

void PlotPullsDistr(TFile* f, TTree* tr, TString strMillepedeRes, TString strOutdir)
{

  TString canvName="c_";
  canvName+=strMillepedeRes;
  canvName+="_";
  canvName+=StrPlotType(PULLS);
  canvName.ReplaceAll(".res","");

  f->cd();
  TCanvas* canv = new TCanvas(canvName,canvName,600,600);
//  tr->Draw("parVal","((parN%20-1)%9+1)==2 && (parVal>-0.4 && parVal<0.4)");// v
  canv->Divide(3,3);

  for (int parInd=1; parInd<=9; parInd++){
    canv->cd(parInd);
    TString strCut="((label%20-1)%9+1)==";
    strCut+=parInd;
    strCut+=" && parErr!=0 && parVal!=0 && label<700000";
    TString hName="hPulls_";
    hName+= StrPar(parInd);
    TH1F* h = new TH1F(hName,hName,100,-3,3);
    TString strDraw="parVal/(1.41*parErr)>>";
    strDraw+=hName;
    tr->Draw(strDraw,strCut,"goff");
    h->Draw("EP");
  }// end of loop over parInd

  canvName+=".png";
  TString saveName=strOutdir+canvName;
  canv->SaveAs(saveName);
  saveName.ReplaceAll(".png",".pdf");
  canv->SaveAs(saveName);
}// end of PlotPullsDistr

//###########################################################

void PlotParsDistr(TFile* f, TTree* tr, TString strMillepedeRes, TString strOutdir)
{

  for (int isubd=PXB; isubd<=TEC; isubd++){

    TString canvName="c_";
    canvName+=strMillepedeRes;
    canvName+="_";
    canvName+=StrPlotType(PARS);
    canvName+="_";
    canvName+=subdLabels[isubd];
    canvName.ReplaceAll(".res","");

    f->cd();
    TCanvas* canv = new TCanvas(canvName,canvName,600,600);
    canv->Divide(3,3);

    for (int parInd=1; parInd<=9; parInd++){
      canv->cd(parInd);
      TString strCut="((label%20-1)%9+1)==";
      strCut+=parInd;
      strCut+=" && label<700000 && ";
      strCut+=StrCutSubd(isubd);
      TString hName="hPars_";
      hName+=subdLabels[isubd];
      hName+="_";
      hName+= StrPar(parInd);
      
      TTree* trCut = tr->CopyTree(strCut);
      float up = trCut->GetMaximum("parVal");
      float low = trCut->GetMinimum("parVal");
      std::cout<<"low="<<low<<", up="<<up<<", nent="<<trCut->GetEntries()<<std::endl;
      TH1F* h = new TH1F(hName,hName,100,10000*low,10000*up);
      TString strDraw="10000*parVal>>";
      strDraw+=hName;
      trCut->Draw(strDraw,strCut,"goff");
      h->SetMarkerStyle(2);
      h->Draw("EP");
    }// end of loop over parInd

    canvName+=".png";
    TString saveName=strOutdir+canvName;
    canv->SaveAs(saveName);
    saveName.ReplaceAll(".png",".pdf");
    canv->SaveAs(saveName);

  }//end of loop over isubd

}// end of PlotParsDistr

//###########################################################

void PlotFromMillepedeRes(TString strMillepedeRes, TString strOutdir, TString strVars, int plotType)
{
  // strPar = "u", "v", "w", "alpha", "beta", "gamma", "def1", "def2", "def3"
  TFile* f = new TFile(strOutdir+TString("fOut.root"),"recreate");
  TTree* tr = new TTree("tr","tr");
  tr->ReadFile(strMillepedeRes,strVars);

  if (plotType==PARSwithERRvsLABEL)
    PlotParValVsLabelWithErr(f, tr, strMillepedeRes, strOutdir);

  if (plotType==NHITS)
    PlotHistsNhitsPerModule(f, tr, strMillepedeRes, strOutdir);

  if (plotType==PULLS)
    PlotPullsDistr(f, tr, strMillepedeRes, strOutdir);

  if (plotType==PARS)
    PlotParsDistr(f, tr, strMillepedeRes, strOutdir);

}// end of PlotPars
