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
  PlotResiduals(int disk, int isPS,  int seedindex){
    disk_=disk;
    isPS_=isPS;
    seedindex_=seedindex;
    double drphimax=1.0;
    double drmax=10.0;
    if (isPS==1) drmax=5.0;
    string name="r*phi residual in D"+std::to_string(disk)+ps2s(isPS)+", "+seedname(seedindex)+" seed, pt<3 GeV";
    hist16l_ = new TH1F(name.c_str(),name.c_str(),50,-drphimax,drphimax);
    name="r*iphi residual in D"+std::to_string(disk)+ps2s(isPS)+", "+seedname(seedindex)+" seed, pt<3 GeV";
    hist116l_ = new TH1F(name.c_str(),name.c_str(),50,-drphimax,drphimax);
    name="r*phi residual in D"+std::to_string(disk)+ps2s(isPS)+", "+seedname(seedindex)+" seed, 3<pt<8 GeV";
    hist16m_ = new TH1F(name.c_str(),name.c_str(),50,-drphimax,drphimax);
    name="r*iphi residual in D"+std::to_string(disk)+ps2s(isPS)+", "+seedname(seedindex)+" seed, 3<pt<8 GeV";
    hist116m_ = new TH1F(name.c_str(),name.c_str(),50,-drphimax,drphimax);
    name="r*phi residual in D"+std::to_string(disk)+ps2s(isPS)+", "+seedname(seedindex)+" seed, pt>8 GeV";
    hist16h_ = new TH1F(name.c_str(),name.c_str(),50,-drphimax,drphimax);
    name="r*iphi residual in D"+std::to_string(disk)+ps2s(isPS)+", "+seedname(seedindex)+" seed, pt>8 GeV";
    hist116h_ = new TH1F(name.c_str(),name.c_str(),50,-drphimax,drphimax);
    
    name="r residual in D"+std::to_string(disk)+ps2s(isPS)+", "+seedname(seedindex);
    hist16_ = new TH1F(name.c_str(),name.c_str(),40,-drmax,drmax);
    name="ir residual in D"+std::to_string(disk)+ps2s(isPS)+", "+seedname(seedindex);
    hist116_ = new TH1F(name.c_str(),name.c_str(),40,-drmax,drmax);
}

  int addResid(int disk, int isPS, int seedindex, double pt, double idphi, double dphi, double dphicut, double idr, double dr, double drcut){

    bool lowpt=(pt<3.0);
    bool highpt=(pt>8.0);
    bool medpt=(!lowpt)&&(!highpt);
    
    if (disk==disk_&&seedindex==seedindex_&&isPS==isPS_) {
      dphicut_=dphicut;
      drcut_=drcut;
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
      hist116_->Fill(idr);
      hist16_->Fill(dr);

      return 1;
    }

    return 0;

  }

  void Draw(TCanvas* c){

    cout << "disk seedindex phicut : "<<disk_<<" "<<seedindex_<<" "<<dphicut_<<endl;
    
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
    TLine* l1 = new TLine(-drcut_,0,-drcut_,0.5*hist116_->GetMaximum());
    l1->Draw();
    TLine* l2 = new TLine(drcut_,0,drcut_,0.5*hist116_->GetMaximum());
    l2->Draw();

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

  string ps2s(int isPS){

    if (isPS) return "PS";

    return "2S";

  }
  
private:

  int disk_;
  int isPS_;
  int seedindex_;
  double dphicut_;
  double drcut_;
  
  TH1 *hist16l_;
  TH1 *hist116l_;
  TH1 *hist16m_;
  TH1 *hist116m_;
  TH1 *hist16h_;
  TH1 *hist116h_;
  TH1 *hist16_;
  TH1 *hist116_;


};


void diskresiduals(){
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

 PlotResiduals Resid_D1PS_L1L2(1,1,0);
 PlotResiduals Resid_D12S_L1L2(1,0,0);
 PlotResiduals Resid_D12S_L3L4(1,0,1);
 PlotResiduals Resid_D1PS_D3D4(1,1,4);
		
 PlotResiduals Resid_D2PS_L1L2(2,1,0);
 PlotResiduals Resid_D22S_L1L2(2,0,0);
 PlotResiduals Resid_D22S_L3L4(2,0,1);
 PlotResiduals Resid_D2PS_D3D4(2,1,4);
 PlotResiduals Resid_D2PS_L1D1(2,1,5);
 PlotResiduals Resid_D2PS_L2D1(2,1,6);
 PlotResiduals Resid_D22S_L2D1(2,0,6);
		
 PlotResiduals Resid_D3PS_L1L2(3,1,0);
 PlotResiduals Resid_D32S_L1L2(3,0,0);
 PlotResiduals Resid_D3PS_D1D2(3,1,3);
 PlotResiduals Resid_D32S_D1D2(3,0,3);
 PlotResiduals Resid_D3PS_L1D1(3,1,5);
 PlotResiduals Resid_D3PS_L2D1(3,1,6);
 PlotResiduals Resid_D32S_L2D1(3,0,6);
		
 PlotResiduals Resid_D42S_L1L2(4,0,0);
 PlotResiduals Resid_D4PS_D1D2(4,1,3);
 PlotResiduals Resid_D42S_D1D2(4,0,3);
 PlotResiduals Resid_D4PS_L1D1(4,1,5);
 PlotResiduals Resid_D42S_L1D1(4,0,5);
 PlotResiduals Resid_D42S_L2D1(4,0,6);
		
 PlotResiduals Resid_D5PS_D1D2(5,1,3);
 PlotResiduals Resid_D52S_D1D2(5,0,3);
 PlotResiduals Resid_D5PS_D3D4(5,1,4);
 PlotResiduals Resid_D52S_D3D4(5,0,4);
 PlotResiduals Resid_D5PS_L1D1(5,1,5);
 PlotResiduals Resid_D52S_L1D1(5,0,5);
 

 ifstream in("diskresiduals.txt");

 int count=0;

 while (in.good()) {

   double disk,isPS,seedlayer,seeddisk,pt,idphi,dphi,dphicut,idr,dr,drcut;
   
   in>>disk>>isPS>>seedlayer>>seeddisk>>pt>>idphi>>dphi>>dphicut>>idr>>dr>>drcut;

   if (!in.good()) continue;
   
   int seedindex=-1;

   if (seedlayer==1&&seeddisk==0) seedindex=0;  //L1L2
   if (seedlayer==3&&seeddisk==0) seedindex=1;  //L3L4
   if (seedlayer==5&&seeddisk==0) seedindex=2;  //L5L6
   if (seedlayer==0&&seeddisk==1) seedindex=3;  //D1D2
   if (seedlayer==0&&seeddisk==3) seedindex=4;  //D3D4
   if (seedlayer==1&&seeddisk==1) seedindex=5;  //L1D1
   if (seedlayer==2&&seeddisk==1) seedindex=6;  //L2D1


   int added=0;
   
   added+=Resid_D1PS_L1L2.addResid(disk, isPS, seedindex, pt, idphi, dphi, dphicut, idr, dr, drcut);
   added+=Resid_D12S_L1L2.addResid(disk, isPS, seedindex, pt, idphi, dphi, dphicut, idr, dr, drcut);
   added+=Resid_D12S_L3L4.addResid(disk, isPS, seedindex, pt, idphi, dphi, dphicut, idr, dr, drcut);
   added+=Resid_D1PS_D3D4.addResid(disk, isPS, seedindex, pt, idphi, dphi, dphicut, idr, dr, drcut);
		
   added+=Resid_D2PS_L1L2.addResid(disk, isPS, seedindex, pt, idphi, dphi, dphicut, idr, dr, drcut);
   added+=Resid_D22S_L1L2.addResid(disk, isPS, seedindex, pt, idphi, dphi, dphicut, idr, dr, drcut);
   added+=Resid_D22S_L3L4.addResid(disk, isPS, seedindex, pt, idphi, dphi, dphicut, idr, dr, drcut);
   added+=Resid_D2PS_D3D4.addResid(disk, isPS, seedindex, pt, idphi, dphi, dphicut, idr, dr, drcut);
   added+=Resid_D2PS_L1D1.addResid(disk, isPS, seedindex, pt, idphi, dphi, dphicut, idr, dr, drcut);
   added+=Resid_D2PS_L2D1.addResid(disk, isPS, seedindex, pt, idphi, dphi, dphicut, idr, dr, drcut);
   added+=Resid_D22S_L2D1.addResid(disk, isPS, seedindex, pt, idphi, dphi, dphicut, idr, dr, drcut);
		
   added+=Resid_D3PS_L1L2.addResid(disk, isPS, seedindex, pt, idphi, dphi, dphicut, idr, dr, drcut);
   added+=Resid_D32S_L1L2.addResid(disk, isPS, seedindex, pt, idphi, dphi, dphicut, idr, dr, drcut);
   added+=Resid_D3PS_D1D2.addResid(disk, isPS, seedindex, pt, idphi, dphi, dphicut, idr, dr, drcut);
   added+=Resid_D32S_D1D2.addResid(disk, isPS, seedindex, pt, idphi, dphi, dphicut, idr, dr, drcut);
   added+=Resid_D3PS_L1D1.addResid(disk, isPS, seedindex, pt, idphi, dphi, dphicut, idr, dr, drcut);
   added+=Resid_D3PS_L2D1.addResid(disk, isPS, seedindex, pt, idphi, dphi, dphicut, idr, dr, drcut);
   added+=Resid_D32S_L2D1.addResid(disk, isPS, seedindex, pt, idphi, dphi, dphicut, idr, dr, drcut);
		
   added+=Resid_D42S_L1L2.addResid(disk, isPS, seedindex, pt, idphi, dphi, dphicut, idr, dr, drcut);
   added+=Resid_D4PS_D1D2.addResid(disk, isPS, seedindex, pt, idphi, dphi, dphicut, idr, dr, drcut);
   added+=Resid_D42S_D1D2.addResid(disk, isPS, seedindex, pt, idphi, dphi, dphicut, idr, dr, drcut);
   added+=Resid_D4PS_L1D1.addResid(disk, isPS, seedindex, pt, idphi, dphi, dphicut, idr, dr, drcut);
   added+=Resid_D42S_L1D1.addResid(disk, isPS, seedindex, pt, idphi, dphi, dphicut, idr, dr, drcut);
   added+=Resid_D42S_L2D1.addResid(disk, isPS, seedindex, pt, idphi, dphi, dphicut, idr, dr, drcut);   
		
   added+=Resid_D5PS_D1D2.addResid(disk, isPS, seedindex, pt, idphi, dphi, dphicut, idr, dr, drcut);
   added+=Resid_D52S_D1D2.addResid(disk, isPS, seedindex, pt, idphi, dphi, dphicut, idr, dr, drcut);
   added+=Resid_D5PS_D3D4.addResid(disk, isPS, seedindex, pt, idphi, dphi, dphicut, idr, dr, drcut);
   added+=Resid_D52S_D3D4.addResid(disk, isPS, seedindex, pt, idphi, dphi, dphicut, idr, dr, drcut);
   added+=Resid_D5PS_L1D1.addResid(disk, isPS, seedindex, pt, idphi, dphi, dphicut, idr, dr, drcut);
   added+=Resid_D52S_L1D1.addResid(disk, isPS, seedindex, pt, idphi, dphi, dphicut, idr, dr, drcut);

   if (added!=1) {
     cout << "Added = "<<added<<" : disk isPS seedindex "<<disk<<" "<<isPS<<" "<<seedindex<<endl;
   }
   
   count++;

 }

//cout << "Processed: "<<count<<" events"<<endl;

 Resid_D1PS_L1L2.Draw(c1);
 c1->Print("diskresiduals.pdf(","pdf");
 Resid_D12S_L1L2.Draw(c1);
 c1->Print("diskresiduals.pdf(","pdf");
 Resid_D12S_L3L4.Draw(c1);
 c1->Print("diskresiduals.pdf","pdf");
 Resid_D1PS_D3D4.Draw(c1);
 c1->Print("diskresiduals.pdf","pdf");
 
 Resid_D2PS_L1L2.Draw(c1);
 c1->Print("diskresiduals.pdf","pdf");
 Resid_D22S_L1L2.Draw(c1);
 c1->Print("diskresiduals.pdf","pdf");
 Resid_D22S_L3L4.Draw(c1);
 c1->Print("diskresiduals.pdf","pdf");
 Resid_D2PS_D3D4.Draw(c1);
 c1->Print("diskresiduals.pdf","pdf");
 Resid_D2PS_L1D1.Draw(c1);
 c1->Print("diskresiduals.pdf","pdf");
 Resid_D2PS_L2D1.Draw(c1);
 c1->Print("diskresiduals.pdf","pdf");
 Resid_D22S_L2D1.Draw(c1);
 c1->Print("diskresiduals.pdf","pdf");
 
 Resid_D3PS_L1L2.Draw(c1);
 c1->Print("diskresiduals.pdf","pdf");
 Resid_D32S_L1L2.Draw(c1);
 c1->Print("diskresiduals.pdf","pdf");
 Resid_D3PS_D1D2.Draw(c1);
 c1->Print("diskresiduals.pdf","pdf");
 Resid_D32S_D1D2.Draw(c1);
 c1->Print("diskresiduals.pdf","pdf");
 Resid_D3PS_L1D1.Draw(c1);
 c1->Print("diskresiduals.pdf","pdf");
 Resid_D3PS_L2D1.Draw(c1);
 c1->Print("diskresiduals.pdf","pdf");
 Resid_D32S_L2D1.Draw(c1);
 c1->Print("diskresiduals.pdf","pdf");
 
 Resid_D42S_L1L2.Draw(c1);
 c1->Print("diskresiduals.pdf","pdf");
 Resid_D4PS_D1D2.Draw(c1);
 c1->Print("diskresiduals.pdf","pdf");
 Resid_D42S_D1D2.Draw(c1);
 c1->Print("diskresiduals.pdf","pdf");
 Resid_D4PS_L1D1.Draw(c1);
 c1->Print("diskresiduals.pdf","pdf");
 Resid_D42S_L1D1.Draw(c1);
 c1->Print("diskresiduals.pdf","pdf");
 Resid_D42S_L2D1.Draw(c1);
 c1->Print("diskresiduals.pdf","pdf"); 
 
 Resid_D5PS_D1D2.Draw(c1);
 c1->Print("diskresiduals.pdf","pdf");
 Resid_D52S_D1D2.Draw(c1);
 c1->Print("diskresiduals.pdf","pdf");
 Resid_D5PS_D3D4.Draw(c1);
 c1->Print("diskresiduals.pdf","pdf");
 Resid_D52S_D3D4.Draw(c1);
 c1->Print("diskresiduals.pdf","pdf");
 Resid_D5PS_L1D1.Draw(c1);
 c1->Print("diskresiduals.pdf","pdf");
 Resid_D52S_L1D1.Draw(c1);
 c1->Print("diskresiduals.pdf)","pdf");

}

