#include "TH1F.h"
#include "TCanvas.h"
#include "TGraphErrors.h"
#include "Utilities.h"
#include "DrawingDefinitions.h"
////////////////////////////////////////////////////////////
void DrawSingleCorrection(double eta, double ptMin, double ptMax)
{
  TGraphErrors *gL[10];
  TGraph *gErrL[10];
  vector<string> vT = parseLevels(CORRECTION_TAGS);
  TGraphAsymmErrors *gTot;
  TGraph *gErrHigh, *gErrLow;
  TString ss;
  char name[100];
  gTot = plotVsPt(eta,ptMin,ptMax);
  gErrHigh = getError(gTot,"HIGH");
  gErrLow = getError(gTot,"LOW");
  TPaveText *pave = new TPaveText(0.3,0.6,0.8,0.9,"NDC");
  pave->SetBorderSize(0);
  pave->SetFillColor(0);
  pave->SetTextAlign(12);
  pave->SetTextFont(42);
  pave->SetTextColor(4);
  for(unsigned int i=0;i<vT.size();i++)
    pave->AddText(vT[i].c_str());
  if (!SINGLE_UNCERTAINTY_TAG.empty())
    pave->AddText(SINGLE_UNCERTAINTY_TAG.c_str());
  sprintf(name,"#eta = %1.1f",eta);
  TText *t1 = pave->AddText(name);
  t1->SetTextAlign(22);
  t1->SetTextSize(0.05);
  if ((int)CORRECTION_TAGS.find("L1")>=0)
    {
      sprintf(name,"m_{jet} = %1.1f GeV",JetMASS);
      TText *t2 = pave->AddText(name);
      t2->SetTextAlign(22);
      t2->SetTextSize(0.05);
    } 
  ////////////////////////////////////////////////////////////
  TCanvas *c = new TCanvas("Correction","Correction");
  gPad->SetGridx();
  gPad->SetGridy();
  setStyle(gTot,4,3001,1,1,"","Uncorrected jet p_{T} (GeV)","Correction factor");
  if ((int)CORRECTION_TAGS.find("L1")>=0)
    {
      gTot->SetMaximum(1.1); 
      gTot->SetMinimum(0.9);
    }
  gTot->Draw("AL3");
  pave->Draw();
  
  if (SINGLE_UNCERTAINTY_TAG.empty()) break;
  ////////////////////////////////////////////////////////////
  TCanvas *c1 = new TCanvas("RelativeUncertainty","RelativeUncertainty");
  gPad->SetGridx();
  gPad->SetGridy();
  scale(gErrHigh,100.);
  scale(gErrLow,100.);
  gErrHigh->SetMaximum(1.7*FindMaximum(gErrHigh));
  setStyle(gErrHigh,1,1001,1,1,"","Uncorrected jet p_{T} (GeV)","Fractional uncertainty [%]");
  setStyle(gErrLow,4,1001,1,1,"","Uncorrected jet p_{T} (GeV)","Fractional uncertainty [%]");
  gErrHigh->Draw("AL");
  gErrLow->Draw("sameL");
  TLegend *leg = new TLegend(0.6,0.3,0.8,0.5);
  leg->AddEntry(gErrHigh,"High uncertainty","L");
  leg->AddEntry(gErrLow,"Low uncertainty","L");
  leg->SetFillColor(0);
  leg->SetLineColor(0);
  leg->SetTextFont(42); 
  leg->Draw();
  pave->Draw();
}  
////////////////////////////////////////////////////////////
void DrawSingleCorrection(double pt)
{
  TGraphErrors *gL[10];
  TGraph *gErrL[10];
  vector<string> vT = parseLevels(CORRECTION_TAGS);
  TGraphAsymmErrors *gTot;
  TGraph *gErrHigh, *gErrLow;
  char name[100];
  gTot = plotVsEta(pt); 
  gErrHigh = getError(gTot,"HIGH");
  gErrLow = getError(gTot,"LOW");
  
  TPaveText *pave = new TPaveText(0.3,0.6,0.8,0.9,"NDC");
  pave->SetBorderSize(0);
  pave->SetFillColor(0);
  pave->SetTextAlign(12);
  pave->SetTextFont(42);
  pave->SetTextColor(4);
  for(unsigned int i=0;i<vT.size();i++)
    pave->AddText(vT[i].c_str());
  if (!SINGLE_UNCERTAINTY_TAG.empty())
    pave->AddText(SINGLE_UNCERTAINTY_TAG.c_str());
  sprintf(name,"p_{T} = %d GeV",pt);
  TText *t1 = pave->AddText(name);
  t1->SetTextAlign(22);
  t1->SetTextSize(0.05);
  if ((int)CORRECTION_TAGS.find("L1")>=0)
    {
      sprintf(name,"m_{jet} = %1.1f GeV",JetMASS);
      TText *t2 = pave->AddText(name);
      t2->SetTextAlign(22);
      t2->SetTextSize(0.05);
    }
  ////////////////////////////////////////////////////////////
  TCanvas *c = new TCanvas("Correction","Correction");
  gPad->SetGridx();
  gPad->SetGridy();
  setStyle(gTot,4,3001,1,1,"","jet #eta","Correction factor");
  gTot->SetMaximum(1.3*FindMaximum(gTot));
  if ((int)CORRECTION_TAGS.find("L1")>=0)
    {
      gTot->SetMaximum(1.1); 
      gTot->SetMinimum(0.9);
    }
  gTot->Draw("AL3");
  gTot->Draw("AL3");
  pave->Draw();

  if (SINGLE_UNCERTAINTY_TAG.empty()) break;
  ////////////////////////////////////////////////////////////
  TCanvas *c1 = new TCanvas("RelativeUncertainty","RelativeUncertainty");
  gPad->SetGridx();
  gPad->SetGridy();
  scale(gErrHigh,100.);
  scale(gErrLow,100.);
  gErrHigh->SetMaximum(1.7*FindMaximum(gErrHigh));
  setStyle(gErrHigh,1,1001,1,1,"","jet #eta","Fractional uncertainty [%]");
  setStyle(gErrLow,4,1001,1,1,"","jet #eta","Fractional uncertainty [%]");
  gErrHigh->Draw("AL");
  gErrLow->Draw("sameL");
  TLegend *leg = new TLegend(0.6,0.3,0.8,0.5);
  leg->AddEntry(gErrHigh,"High uncertainty","L");
  leg->AddEntry(gErrLow,"Low uncertainty","L");
  leg->SetFillColor(0);
  leg->SetLineColor(0);
  leg->SetTextFont(42); 
  leg->Draw();
  pave->Draw();
}  
////////////////////////////////////////////////////////////
TGraphAsymmErrors* plotVsPt(double eta, double ptMin, double ptMax)
{
  CombinedJetCorrector *JetCorrector = new CombinedJetCorrector(LEVELS,CORRECTION_TAGS);
  JetCorrectionUncertainty *JetUnc;
  if (!SINGLE_UNCERTAINTY_TAG.empty())
    {
      string tmp = "../../../CondFormats/JetMETObjects/data/"+SINGLE_UNCERTAINTY_TAG+".txt";
      JetUnc = new JetCorrectionUncertainty(tmp);
    }
  int i(0),N(100);
  double scale,pt,ratio,energy;
  double x[1000],y[1000],eyU[1000],eyD[1000],ex[1000];
  double max = TMath::Min(ptMax,5000/cosh(eta));
  ratio = pow(max/ptMin,1./N);
  pt = ptMin;
  while (pt<=max)
    {    
      energy = sqrt(pow(JetMASS,2)+pow(pt*cosh(eta),2)); 
      scale = JetCorrector->getCorrection(pt,eta,energy);
      x[i] = pt;
      y[i] = scale;
      ex[i] = 0.;
      if (!SINGLE_UNCERTAINTY_TAG.empty())
        {
          eyU[i] = y[i]*JetUnc->uncertaintyPtEta(pt,eta,"UP");
          eyD[i] = y[i]*JetUnc->uncertaintyPtEta(pt,eta,"DOWN"); 
        }
      else
        {
          eyU[i] = 0;
          eyD[i] = 0; 
        }  
      pt*=ratio; 
      i++;
    }  
  TGraphAsymmErrors *g = new TGraphAsymmErrors(i,x,y,ex,ex,eyU,eyD);
  return g;
}
////////////////////////////////////////////////////////////
TGraphAsymmErrors* plotVsEta(double pt)
{
  CombinedJetCorrector *JetCorrector = new CombinedJetCorrector(LEVELS,CORRECTION_TAGS);
  JetCorrectionUncertainty *JetUnc;
  if (!SINGLE_UNCERTAINTY_TAG.empty())
    {
      string tmp = "../../../CondFormats/JetMETObjects/data/"+SINGLE_UNCERTAINTY_TAG+".txt";
      JetUnc = new JetCorrectionUncertainty(tmp);
    }
  int i(0),N(200);
  double scale,dx,eta,energy;
  double x[1000],y[1000],eyU[1000],eyD[1000],ex[1000];
  double max = TMath::Min(4.99,TMath::ACosH(0.5*CM_ENERGY/pt));
  dx = 2*max/N;
  eta = -max;
  while (eta<=max)
    {     
      energy = sqrt(pow(JetMASS,2)+pow(pt*cosh(eta),2));
      scale = JetCorrector->getCorrection(pt,eta,energy);
      x[i] = eta;
      y[i] = scale; 
      ex[i] = 0.;
      if (!SINGLE_UNCERTAINTY_TAG.empty())
        {
          eyU[i] = y[i]*JetUnc->uncertaintyPtEta(pt,eta,"UP");
          eyD[i] = y[i]*JetUnc->uncertaintyPtEta(pt,eta,"DOWN"); 
        }
      else
        {
          eyU[i] = 0;
          eyD[i] = 0; 
        } 
      eta+=dx; 
      i++;
    }  
  TGraphAsymmErrors *g = new TGraphAsymmErrors(i,x,y,ex,ex,eyU,eyD);
  return g;
}
