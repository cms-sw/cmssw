{
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


  // For publishing:
  gStyle->SetLineWidth(1.5);
  gStyle->SetTextSize(1.1);
  gStyle->SetLabelSize(0.06,"xy");
  gStyle->SetTitleSize(0.06,"xy");
  gStyle->SetTitleOffset(1.2,"x");
  gStyle->SetTitleOffset(1.0,"y");
  gStyle->SetPadTopMargin(0.1);
  gStyle->SetPadRightMargin(0.1);
  gStyle->SetPadBottomMargin(0.16);
  gStyle->SetPadLeftMargin(0.12);


 c1 = new TCanvas("c1","Track performance",200,10,1000,1100);
 c1->Divide(2,2);
 c1->SetFillColor(0);
 c1_4->SetGridy(10);
 c1_3->SetGridy(10);


 TH1 *hist00 = new TH1F("h00","MinvDt",20,-1.0,1.0);
 TH1 *hist10 = new TH1F("h10","MinvDt",20,-1.0,1.0);
 TH1 *hist20 = new TH1F("h20","MinvDt",20,-1.0,1.0);
 TH1 *hist30 = new TH1F("h30","MinvDt",20,-1.0,1.0);


 ifstream in("alphadep.txt");


 double u,v;
 double value;
 
 in >> u >> v ;

 while (in.good()) {

   cout << "u, v = "<<u<<" "<<v<<endl;

   for (i=0;i<4;i++) {
     for (j=0;j<10;j++) {
       in >> value;
       if (fabs(v)<0.07) {
	 if (i==0&&j==7) h00->Fill(u,value);
	 if (i==1&&j==7) h10->Fill(u,value);
       }
       if (fabs(v)>0.93) {
	 if (i==0&&j==7) h20->Fill(u,value);
	 if (i==1&&j==7) h30->Fill(u,value);
       }
     }
   }

  in >> u >> v ;

 }

 c1->cd(1);
 h00->Draw();
 c1->cd(2);
 h10->Draw();
 c1->cd(3);
 h20->Draw();
 c1->cd(4);
 h30->Draw();

}

