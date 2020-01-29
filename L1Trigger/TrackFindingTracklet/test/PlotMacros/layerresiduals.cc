#include "TMath.h"
#include "TRint.h"
#include "TROOT.h"
#include "TStyle.h"
#include "TLorentzVector.h"
#include "TCanvas.h"
#include "TH1.h"
#include "TGaxis.h"
#include <fstream>
#include <iostream>
#include "TMath.h"

class PlotResiduals {

public:
  PlotResiduals(int layer, int seedindex){
    layer_=layer;
    seedindex_=seedindex;
    double drphimax=0.2;
    double dzmax=15.0;
    if ((layer_>4)&&(seedindex_==1)) drphimax=1.0;
    if (seedindex_==1) dzmax=10.0;
    string name="r*phi residual in layer "+std::to_string(layer)+", "+seedname(seedindex)
      +" seed, pt<3 GeV";
    hist16l_ = new TH1F(name.c_str(),name.c_str(),50,-drphimax,drphimax);
    name="r*iphi residual in layer "+std::to_string(layer)+", "+seedname(seedindex)
      +" seed, pt<3 GeV";
    hist116l_ = new TH1F(name.c_str(),name.c_str(),50,-drphimax,drphimax);
    name="r*phi residual in layer "+std::to_string(layer)+", "+seedname(seedindex)
      +" seed, 3<pt<8 GeV";
    hist16m_ = new TH1F(name.c_str(),name.c_str(),50,-drphimax,drphimax);
    name="r*iphi residual in layer "+std::to_string(layer)+", "+seedname(seedindex)
      +" seed, 3<pt<8 GeV";
    hist116m_ = new TH1F(name.c_str(),name.c_str(),50,-drphimax,drphimax);
    name="r*phi residual in layer "+std::to_string(layer)+", "+seedname(seedindex)
      +" seed, pt>8 GeV";
    hist16h_ = new TH1F(name.c_str(),name.c_str(),50,-drphimax,drphimax);
    name="r*iphi residual in layer "+std::to_string(layer)+", "+seedname(seedindex)
      +" seed, pt>8 GeV";
    hist116h_ = new TH1F(name.c_str(),name.c_str(),50,-drphimax,drphimax);
    
    name="z residual in layer "+std::to_string(layer)+", "+seedname(seedindex);
    hist16_ = new TH1F(name.c_str(),name.c_str(),40,-dzmax,dzmax);
    name="iz residual in layer "+std::to_string(layer)+", "+seedname(seedindex);
    hist116_ = new TH1F(name.c_str(),name.c_str(),40,-dzmax,dzmax);
}

  string seedname(int seedindex) {

    if (seedindex==0) return "L1L2";
    if (seedindex==1) return "L3L4";
    if (seedindex==2) return "L5L6";
    if (seedindex==3) return "D1D2";
    if (seedindex==4) return "D3D4";
    if (seedindex==5) return "L1D1";
    if (seedindex==6) return "L2D1";

    return "Unkown seedindex";
    
  }
  
  void addResid(int layer, int seedindex, double pt, double idphi, double dphi, double dphicut, double idz, double dz, double dzcut){

    bool lowpt=(pt<3.0);
    bool highpt=(pt>8.0);
    bool medpt=(!lowpt)&&(!highpt);
    
    if (layer==layer_&&seedindex==seedindex_) {
      dphicut_=dphicut;
      dzcut_=dzcut;
      if (lowpt) {
	hist116l_->Fill(idphi);
	hist16l_->Fill(dphi);
      }
      if (medpt) {
	hist116m_->Fill(idphi);
	hist16m_->Fill(dphi);
      }
      if (highpt) {
	hist116h_->Fill(idphi);
	hist16h_->Fill(dphi);
      }
      hist116_->Fill(idz);
      hist16_->Fill(dz);

    }


  }

  void Draw(TCanvas* c){

    cout << "layer seed phicut : "<<layer_<<" "<<seedindex_<<" "<<dphicut_<<endl;
    
    c->cd(1);
    hist16l_->SetLineColor(kBlue);
    hist16l_->Draw();
    hist116l_->SetLineColor(kRed);
    hist116l_->Draw("Same");
    TLine* ll1 = new TLine(-dphicut_,0,-dphicut_,0.5*hist116l_->GetMaximum());
    ll1->Draw();
    TLine* ll2 = new TLine(dphicut_,0,dphicut_,0.5*hist116l_->GetMaximum());
    ll2->Draw();

    c->cd(2);
    hist16m_->SetLineColor(kBlue);
    hist16m_->Draw();
    hist116m_->SetLineColor(kRed);
    hist116m_->Draw("Same");
    TLine* lm1 = new TLine(-dphicut_,0,-dphicut_,0.5*hist116m_->GetMaximum());
    lm1->Draw();
    TLine* lm2 = new TLine(dphicut_,0,dphicut_,0.5*hist116m_->GetMaximum());
    lm2->Draw();

    c->cd(3);
    hist16h_->SetLineColor(kBlue);
    hist16h_->Draw();
    hist116h_->SetLineColor(kRed);
    hist116h_->Draw("Same");
    TLine* lh1 = new TLine(-dphicut_,0,-dphicut_,0.5*hist116h_->GetMaximum());
    lh1->Draw();
    TLine* lh2 = new TLine(dphicut_,0,dphicut_,0.5*hist116h_->GetMaximum());
    lh2->Draw();

    c->cd(4);
    hist16_->SetLineColor(kBlue);
    hist16_->Draw();
    hist116_->SetLineColor(kRed);
    hist116_->Draw("Same");
    TLine* l1 = new TLine(-dzcut_,0,-dzcut_,0.5*hist116_->GetMaximum());
    l1->Draw();
    TLine* l2 = new TLine(dzcut_,0,dzcut_,0.5*hist116_->GetMaximum());
    l2->Draw();

  }
  
private:

  int layer_;
  int seedindex_;
  double dphicut_;
  double dzcut_;
  
  TH1 *hist16l_;
  TH1 *hist116l_;
  TH1 *hist16m_;
  TH1 *hist116m_;
  TH1 *hist16h_;
  TH1 *hist116h_;
  TH1 *hist16_;
  TH1 *hist116_;


};


void layerresiduals(){
//
// To see the output of this macro, click here.

//

#include "TMath.h"

gROOT->Reset();

gROOT->SetStyle("Plain");

gStyle->SetCanvasColor(kWhite);

gStyle->SetCanvasBorderMode(0);     // turn off canvas borders
gStyle->SetPadBorderMode(0);
gStyle->SetOptStat(0);
gStyle->SetOptTitle(1);

  // For publishing:
  gStyle->SetLineWidth(1);
  gStyle->SetTextSize(1.1);
  gStyle->SetLabelSize(0.06,"xy");
  gStyle->SetTitleSize(0.06,"xy");
  gStyle->SetTitleOffset(1.2,"x");
  gStyle->SetTitleOffset(1.0,"y");
  gStyle->SetPadTopMargin(0.1);
  gStyle->SetPadRightMargin(0.1);
  gStyle->SetPadBottomMargin(0.16);
  gStyle->SetPadLeftMargin(0.12);




 TCanvas* c1 = new TCanvas("c1","Track performance",200,10,700,800);
 c1->Divide(2,2);
 c1->SetFillColor(0);
 c1->SetGrid();


 PlotResiduals Resid_L3_L1L2(3,0);
 PlotResiduals Resid_L4_L1L2(4,0);
 PlotResiduals Resid_L5_L1L2(5,0);
 PlotResiduals Resid_L6_L1L2(6,0);

 PlotResiduals Resid_L1_L3L4(1,1);
 PlotResiduals Resid_L2_L3L4(2,1);
 PlotResiduals Resid_L5_L3L4(5,1);
 PlotResiduals Resid_L6_L3L4(6,1);

 PlotResiduals Resid_L1_L5L6(1,2);
 PlotResiduals Resid_L2_L5L6(2,2);
 PlotResiduals Resid_L3_L5L6(3,2);
 PlotResiduals Resid_L4_L5L6(4,2);

 PlotResiduals Resid_L1_D1D2(1,3);
 PlotResiduals Resid_L1_D3D4(1,4);
 PlotResiduals Resid_L1_L2D1(1,6);
 
 PlotResiduals Resid_L2_D1D2(2,3);


 ifstream in("layerresiduals.txt");

 int count=0;

 while (in.good()) {

   double layer,seedindex,pt,idphi,dphi,dphicut,idz,dz,dzcut;
   
   in>>layer>>seedindex>>pt>>idphi>>dphi>>dphicut>>idz>>dz>>dzcut;

   if (!in.good()) continue;

   Resid_L3_L1L2.addResid(layer, seedindex, pt, idphi, dphi, dphicut, idz, dz, dzcut);
   Resid_L4_L1L2.addResid(layer, seedindex, pt, idphi, dphi, dphicut, idz, dz, dzcut);
   Resid_L5_L1L2.addResid(layer, seedindex, pt, idphi, dphi, dphicut, idz, dz, dzcut);
   Resid_L6_L1L2.addResid(layer, seedindex, pt, idphi, dphi, dphicut, idz, dz, dzcut);

   Resid_L1_L3L4.addResid(layer, seedindex, pt, idphi, dphi, dphicut, idz, dz, dzcut);
   Resid_L2_L3L4.addResid(layer, seedindex, pt, idphi, dphi, dphicut, idz, dz, dzcut);
   Resid_L5_L3L4.addResid(layer, seedindex, pt, idphi, dphi, dphicut, idz, dz, dzcut);
   Resid_L6_L3L4.addResid(layer, seedindex, pt, idphi, dphi, dphicut, idz, dz, dzcut);

   Resid_L1_L5L6.addResid(layer, seedindex, pt, idphi, dphi, dphicut, idz, dz, dzcut);
   Resid_L2_L5L6.addResid(layer, seedindex, pt, idphi, dphi, dphicut, idz, dz, dzcut);
   Resid_L3_L5L6.addResid(layer, seedindex, pt, idphi, dphi, dphicut, idz, dz, dzcut);
   Resid_L4_L5L6.addResid(layer, seedindex, pt, idphi, dphi, dphicut, idz, dz, dzcut);
   
   Resid_L1_D1D2.addResid(layer, seedindex, pt, idphi, dphi, dphicut, idz, dz, dzcut);
   Resid_L1_D3D4.addResid(layer, seedindex, pt, idphi, dphi, dphicut, idz, dz, dzcut);
   Resid_L1_L2D1.addResid(layer, seedindex, pt, idphi, dphi, dphicut, idz, dz, dzcut);
 
   Resid_L2_D1D2.addResid(layer, seedindex, pt, idphi, dphi, dphicut, idz, dz, dzcut);

   
   count++;

 }

//cout << "Processed: "<<count<<" events"<<endl;

 Resid_L1_L3L4.Draw(c1);
 c1->Print("layerresiduals.pdf(","pdf");
 Resid_L1_L5L6.Draw(c1);
 c1->Print("layerresiduals.pdf","pdf");
 Resid_L1_D1D2.Draw(c1);
 c1->Print("layerresiduals.pdf","pdf");
 Resid_L1_D3D4.Draw(c1);
 c1->Print("layerresiduals.pdf","pdf");
 Resid_L1_L2D1.Draw(c1);
 c1->Print("layerresiduals.pdf","pdf");
 Resid_L2_L3L4.Draw(c1);
 c1->Print("layerresiduals.pdf","pdf");
 Resid_L2_L5L6.Draw(c1);
 c1->Print("layerresiduals.pdf","pdf");
 Resid_L2_D1D2.Draw(c1);
 c1->Print("layerresiduals.pdf","pdf");
 Resid_L3_L1L2.Draw(c1);
 c1->Print("layerresiduals.pdf","pdf");
 Resid_L3_L5L6.Draw(c1);
 c1->Print("layerresiduals.pdf","pdf");
 Resid_L4_L1L2.Draw(c1);
 c1->Print("layerresiduals.pdf","pdf");
 Resid_L4_L5L6.Draw(c1);
 c1->Print("layerresiduals.pdf","pdf");
 Resid_L5_L1L2.Draw(c1);
 c1->Print("layerresiduals.pdf","pdf");
 Resid_L5_L3L4.Draw(c1);
 c1->Print("layerresiduals.pdf","pdf");
 Resid_L6_L1L2.Draw(c1);
 c1->Print("layerresiduals.pdf","pdf");
 Resid_L6_L3L4.Draw(c1);
 c1->Print("layerresiduals.pdf)","pdf");

 




}

