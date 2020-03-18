#include "TMath.h"
#include "TRint.h"
#include "TROOT.h"
#include "TStyle.h"
#include "TLorentzVector.h"
#include "TCanvas.h"
#include "TH1.h"
#include "TF1.h"
#include "TLatex.h"
#include "TLegend.h"
#include "TGaxis.h"
#include <fstream>
#include <iostream>
#include "TMath.h"
#include <string>
#include <map>

void diskmatch(){
//
// To see the output of this macro, click here.

//


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




 TCanvas* c1 = new TCanvas("c1","Track performance",250,50,600,700);
 c1->Divide(2,2);
 c1->SetFillColor(0);
 c1->SetGrid();

 TCanvas* c2 = new TCanvas("c2","Track performance",250,50,600,700);
 c2->Divide(2,2);
 c2->SetFillColor(0);
 c2->SetGrid();

 TCanvas* c3 = new TCanvas("c3","Track performance",250,50,600,700);
 c3->Divide(2,2);
 c3->SetFillColor(0);
 c3->SetGrid();

 TCanvas* c4 = new TCanvas("c4","Track performance",250,50,600,700);
 c4->Divide(2,2);
 c4->SetFillColor(0);
 c4->SetGrid();

 double rdphi=1.0;
 double drps=2.0;
 double dr2s=10.0;

 TH1 *hist121 = new TH1F("h121","L1D1 or L2D1 to D2 (PS)",50,-rdphi,rdphi);
 TH1 *hist122 = new TH1F("h122","L1D1 or L2D1 to D2 (PS)",50,-drps,drps);
 TH1 *hist123 = new TH1F("h123","L1D1 or L2D1 to D2 (2S)",50,-rdphi,rdphi);
 TH1 *hist124 = new TH1F("h124","L1D1 or L2D1 to D2 (2S)",50,-dr2s,dr2s);

 TH1 *hist131 = new TH1F("h131","L1D1 or L2D1 to D3 (PS)",50,-rdphi,rdphi);
 TH1 *hist132 = new TH1F("h132","L1D1 or L2D1 to D3 (PS)",50,-drps,drps);
 TH1 *hist133 = new TH1F("h133","L1D1 or L2D1 to D3 (2S)",50,-rdphi,rdphi);
 TH1 *hist134 = new TH1F("h134","L1D1 or L2D1 to D3 (2S)",50,-dr2s,dr2s);

 TH1 *hist141 = new TH1F("h141","L1D1 or L2D1 to D4 (PS)",50,-rdphi,rdphi);
 TH1 *hist142 = new TH1F("h142","L1D1 or L2D1 to D4 (PS)",50,-drps,drps);
 TH1 *hist143 = new TH1F("h143","L1D1 or L2D1 to D4 (2S)",50,-rdphi,rdphi);
 TH1 *hist144 = new TH1F("h144","L1D1 or L2D1 to D4 (2S)",50,-dr2s,dr2s);

 TH1 *hist151 = new TH1F("h151","L1D1 or L2D1 to D5 (PS)",50,-rdphi,rdphi);
 TH1 *hist152 = new TH1F("h152","L1D1 or L2D1 to D5 (PS)",50,-drps,drps);
 TH1 *hist153 = new TH1F("h153","L1D1 or L2D1 to D5 (2S)",50,-rdphi,rdphi);
 TH1 *hist154 = new TH1F("h154","L1D1 or L2D1 to D5 (2S)",50,-dr2s,dr2s);



 TH1 *hist221 = new TH1F("h221","L1D1 or L2D1 to D2 (PS)",50,-rdphi,rdphi);
 TH1 *hist222 = new TH1F("h222","L1D1 or L2D1 to D2 (PS)",50,-drps,drps);
 TH1 *hist223 = new TH1F("h223","L1D1 or L2D1 to D2 (2S)",50,-rdphi,rdphi);
 TH1 *hist224 = new TH1F("h224","L1D1 or L2D1 to D2 (2S)",50,-dr2s,dr2s);

 TH1 *hist231 = new TH1F("h231","L1D1 or L2D1 to D3 (PS)",50,-rdphi,rdphi);
 TH1 *hist232 = new TH1F("h232","L1D1 or L2D1 to D3 (PS)",50,-drps,drps);
 TH1 *hist233 = new TH1F("h233","L1D1 or L2D1 to D3 (2S)",50,-rdphi,rdphi);
 TH1 *hist234 = new TH1F("h234","L1D1 or L2D1 to D3 (2S)",50,-dr2s,dr2s);

 TH1 *hist241 = new TH1F("h241","L1D1 or L2D1 to D4 (PS)",50,-rdphi,rdphi);
 TH1 *hist242 = new TH1F("h242","L1D1 or L2D1 to D4 (PS)",50,-drps,drps);
 TH1 *hist243 = new TH1F("h243","L1D1 or L2D1 to D4 (2S)",50,-rdphi,rdphi);
 TH1 *hist244 = new TH1F("h244","L1D1 or L2D1 to D4 (2S)",50,-dr2s,dr2s);

 TH1 *hist251 = new TH1F("h251","L1D1 or L2D1 to D5 (PS)",50,-rdphi,rdphi);
 TH1 *hist252 = new TH1F("h252","L1D1 or L2D1 to D5 (PS)",50,-drps,drps);
 TH1 *hist253 = new TH1F("h253","L1D1 or L2D1 to D5 (2S)",50,-rdphi,rdphi);
 TH1 *hist254 = new TH1F("h254","L1D1 or L2D1 to D5 (2S)",50,-dr2s,dr2s);



 ifstream in("diskmatch1.txt");

 int count=0;

 while (in.good()){

   int disk,seeddisk;
   double phiproj,rproj,dphi,dr;
   double iphiproj,irproj,idphi,idr;

   in >>disk>>phiproj>>rproj>>dphi>>dr>>iphiproj>>irproj>>idphi>>idr;

   //if (layer!=1) continue;
   //if (nmatch<2) continue;

   
   if (disk==2&&rproj<60.0){
     if (fabs(dr)<0.2) hist121->Fill(rproj*dphi);
     if (fabs(rproj*dphi)<0.15) hist122->Fill(dr);
     if (fabs(dr)<0.2) hist221->Fill(rproj*idphi);
     if (fabs(rproj*dphi)<0.15) hist222->Fill(irproj-rproj);
   }
   if (disk==3&&rproj>60.0){
     if (fabs(dr)<3.0) hist123->Fill(rproj*dphi);
     if (fabs(rproj*dphi)<0.15) hist124->Fill(dr);
     if (fabs(dr)<3.0) hist223->Fill(rproj*idphi);
     if (fabs(rproj*dphi)<0.15) hist224->Fill(irproj-rproj);
   }

   if (disk==3&&rproj<60.0){
     if (fabs(dr)<0.2) hist131->Fill(rproj*dphi);
     if (fabs(rproj*dphi)<0.15) hist132->Fill(dr);
     if (fabs(dr)<0.2) hist231->Fill(rproj*idphi);
     if (fabs(rproj*dphi)<0.15) hist232->Fill(irproj-rproj);
   }
   if (disk==3&&rproj>60.0){
     if (fabs(dr)<3.0) hist133->Fill(rproj*dphi);
     if (fabs(rproj*dphi)<0.15) hist134->Fill(dr);
     if (fabs(dr)<3.0) hist233->Fill(rproj*idphi);
     if (fabs(rproj*dphi)<0.15) hist234->Fill(irproj-rproj);
   }

   if (disk==4&&rproj<60.0){
     if (fabs(dr)<0.2) hist141->Fill(rproj*dphi);
     if (fabs(rproj*dphi)<0.20) hist142->Fill(dr);
     if (fabs(dr)<0.2) hist241->Fill(rproj*idphi);
     if (fabs(rproj*dphi)<0.20) hist242->Fill(irproj-rproj);
   }
   if (disk==4&&rproj>60.0){
     if (fabs(dr)<3.0) hist143->Fill(rproj*dphi);
     if (fabs(rproj*dphi)<0.25) hist144->Fill(dr);
     if (fabs(dr)<3.0) hist243->Fill(rproj*idphi);
     if (fabs(rproj*dphi)<0.25) hist244->Fill(irproj-rproj);
   }

   if (disk==5&&rproj<60.0){
     if (fabs(dr)<0.2) hist151->Fill(rproj*dphi);
     if (fabs(rproj*dphi)<0.25) hist152->Fill(dr);
     if (fabs(dr)<0.2) hist251->Fill(rproj*idphi);
     if (fabs(rproj*dphi)<0.25) hist252->Fill(irproj-rproj);
   }
   if (disk==5&&rproj>60.0){
     //cout << "rdphi " << rproj*dphi <<" "<<rproj*idphi<<endl;
     if (fabs(dr)<3.0) hist153->Fill(rproj*dphi);
     if (fabs(rproj*dphi)<0.25) hist154->Fill(dr);
     if (fabs(dr)<3.0) hist253->Fill(rproj*idphi);
     if (fabs(rproj*dphi)<0.25) hist254->Fill(irproj-rproj);
   }

   count++;

 }

 cout << "Processed: "<<count<<" events"<<endl;

 c1->cd(1);
 hist121->GetXaxis()->SetTitle("r#Delta#phi [cm]");
 hist121->Draw();
 hist221->SetLineColor(kBlue);
 hist221->Draw("same");

 c1->cd(2);
 hist122->GetXaxis()->SetTitle("#Deltar [cm]");
 hist122->Draw();
 hist222->SetLineColor(kBlue);
 hist222->Draw("same");

 c1->cd(3);
 hist123->GetXaxis()->SetTitle("r#Delta#phi [cm]");
 hist123->Draw();
 hist223->SetLineColor(kBlue);
 hist223->Draw("same");

 c1->cd(4);
 hist124->GetXaxis()->SetTitle("#Deltar [cm]");
 hist124->Draw();
 hist224->SetLineColor(kBlue);
 hist224->Draw("same");

 c1->Print("diskmatch1.pdf");

 c2->cd(1);
 hist131->GetXaxis()->SetTitle("r#Delta#phi [cm]");
 hist131->Draw();
 hist231->SetLineColor(kBlue);
 hist231->Draw("same");

 c2->cd(2);
 hist132->GetXaxis()->SetTitle("#Deltar [cm]");
 hist132->Draw();
 hist232->SetLineColor(kBlue);
 hist232->Draw("same");

 c2->cd(3);
 hist133->GetXaxis()->SetTitle("r#Delta#phi [cm]");
 hist133->Draw();
 hist233->SetLineColor(kBlue);
 hist233->Draw("same");

 c2->cd(4);
 hist134->GetXaxis()->SetTitle("#Deltar [cm]");
 hist134->Draw();
 hist234->SetLineColor(kBlue);
 hist234->Draw("same");

 c2->Print("diskmatch2.pdf");

 c3->cd(1);
 hist141->GetXaxis()->SetTitle("r#Delta#phi [cm]");
 hist141->Draw();
 hist241->SetLineColor(kBlue);
 hist241->Draw("same");

 c3->cd(2);
 hist142->GetXaxis()->SetTitle("#Deltar [cm]");
 hist142->Draw();
 hist242->SetLineColor(kBlue);
 hist242->Draw("same");

 c3->cd(3);
 hist143->GetXaxis()->SetTitle("r#Delta#phi [cm]");
 hist143->Draw();
 hist243->SetLineColor(kBlue);
 hist243->Draw("same");

 c3->cd(4);
 hist144->GetXaxis()->SetTitle("#Deltar [cm]");
 hist144->Draw();
 hist244->SetLineColor(kBlue);
 hist244->Draw("same");

 c3->Print("diskmatch3.pdf");

 c4->cd(1);
 hist151->GetXaxis()->SetTitle("r#Delta#phi [cm]");
 hist151->Draw();
 hist251->SetLineColor(kBlue);
 hist251->Draw("same");

 c4->cd(2);
 hist152->GetXaxis()->SetTitle("#Deltar [cm]");
 hist152->Draw();
 hist252->SetLineColor(kBlue);
 hist252->Draw("same");

 c4->cd(3);
 hist153->GetXaxis()->SetTitle("r#Delta#phi [cm]");
 hist153->Draw();
 hist253->SetLineColor(kBlue);
 hist253->Draw("same");

 c4->cd(4);
 hist154->GetXaxis()->SetTitle("#Deltar [cm]");
 hist154->Draw();
 hist254->SetLineColor(kBlue);
 hist254->Draw("same");

 c4->Print("diskmatch4.pdf");

 
}

