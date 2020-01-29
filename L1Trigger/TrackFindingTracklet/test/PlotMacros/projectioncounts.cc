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
gStyle->SetOptStat(1111);
gStyle->SetOptTitle(1);

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




 c1 = new TCanvas("c1","Track performance",200,10,700,800);
 c1->Divide(2,2);
 c1->SetFillColor(0);
 c1->SetGrid();

 c2 = new TCanvas("c2","Track performance",200,10,700,800);
 c2->Divide(2,3);
 c2->SetFillColor(0);
 c2->SetGrid();

 c3 = new TCanvas("c3","Track performance",200,10,700,800);
 c3->Divide(2,3);
 c3->SetFillColor(0);
 c3->SetGrid();

 c4 = new TCanvas("c4","Track performance",200,10,700,800);
 c4->Divide(2,3);
 c4->SetFillColor(0);
 c4->SetGrid();


 c5 = new TCanvas("c5","Track performance",200,10,700,800);
 c5->Divide(2,2);
 c5->SetFillColor(0);
 c5->SetGrid();

 
 TH1 *hist01 = new TH1F("h01","Projections from region in L1L2 to L3 in sector",30,0.0,30.0);
 TH1 *hist02 = new TH1F("h02","Projections from region in L1L2 to L4 in sector",30,0.0,30.0);
 TH1 *hist03 = new TH1F("h03","Projections from region in L1L2 to L5 in sector",30,0.0,30.0);
 TH1 *hist04 = new TH1F("h04","Projections from region in L1L2 to L6 in sector",30,0.0,30.0);

 TH1 *hist11 = new TH1F("h11","Projections from full L1L2 to L3 in neighbor sector",30,0.0,30.0);
 TH1 *hist12 = new TH1F("h12","Projections from full L1L2 to L4 in neighbor sector",30,0.0,30.0);
 TH1 *hist13 = new TH1F("h13","Projections from full L1L2 to L5 in neighbor sector",30,0.0,30.0);
 TH1 *hist14 = new TH1F("h14","Projections from full L1L2 to L6 in neighbor sector",30,0.0,30.0);
 TH1 *hist15 = new TH1F("h15","Projections from full L1L2 to neighbor sector",30,0.0,100.0);

 TH1 *hist21 = new TH1F("h21","Projections from full L3L4 to L1 in neighbor sector",30,0.0,30.0);
 TH1 *hist22 = new TH1F("h22","Projections from full L3L4 to L2 in neighbor sector",30,0.0,30.0);
 TH1 *hist23 = new TH1F("h23","Projections from full L3L4 to L5 in neighbor sector",30,0.0,30.0);
 TH1 *hist24 = new TH1F("h24","Projections from full L3L4 to L6 in neighbor sector",30,0.0,30.0);
 TH1 *hist25 = new TH1F("h25","Projections from full L3L4 to neighbor sector",30,0.0,100.0);


 TH1 *hist31 = new TH1F("h31","Projections from full L5L6 to L1 in neighbor sector",30,0.0,30.0);
 TH1 *hist32 = new TH1F("h32","Projections from full L5L6 to L2 in neighbor sector",30,0.0,30.0);
 TH1 *hist33 = new TH1F("h33","Projections from full L5L6 to L3 in neighbor sector",30,0.0,30.0);
 TH1 *hist34 = new TH1F("h34","Projections from full L5L6 to L4 in neighbor sector",30,0.0,30.0);
 TH1 *hist35 = new TH1F("h35","Projections from full L5L6 to neighbor sector",30,0.0,100.0);




 ifstream in("projectioncounts.txt");

 int count=0;

 int oldneighbor=0;
 
 int sum_L1L2_L3=0;
 int sum_L1L2_L4=0;
 int sum_L1L2_L5=0;
 int sum_L1L2_L6=0;
 int sum_L1L2_tot=0;

 int sum_L3L4_L1=0;
 int sum_L3L4_L2=0;
 int sum_L3L4_L5=0;
 int sum_L3L4_L6=0;
 int sum_L3L4_tot=0;

 int sum_L5L6_L1=0;
 int sum_L5L6_L2=0;
 int sum_L5L6_L3=0;
 int sum_L5L6_L4=0;   
 int sum_L5L6_tot=0;

 while (in.good()){

   int neighbor,seedlayer,innerregion,outerregion,projlay,nproj;
  
   in >>neighbor>>seedlayer>>innerregion>>outerregion>>projlay>>nproj;

   if (!in.good()) continue;

   if (neighbor!=oldneighbor) {
     if (oldneighbor!=0) {
       hist11->Fill(sum_L1L2_L3);
       hist12->Fill(sum_L1L2_L4);
       hist13->Fill(sum_L1L2_L5);
       hist14->Fill(sum_L1L2_L6);
       hist15->Fill(sum_L1L2_tot);

       hist21->Fill(sum_L3L4_L1);
       hist22->Fill(sum_L3L4_L2);
       hist23->Fill(sum_L3L4_L5);
       hist24->Fill(sum_L3L4_L6);
       hist25->Fill(sum_L3L4_tot);

       hist31->Fill(sum_L5L6_L1);
       hist32->Fill(sum_L5L6_L2);
       hist33->Fill(sum_L5L6_L3);
       hist34->Fill(sum_L5L6_L4);
       hist35->Fill(sum_L5L6_tot);
     }
     sum_L1L2_L3=0;
     sum_L1L2_L4=0;
     sum_L1L2_L5=0;
     sum_L1L2_L6=0;
     sum_L1L2_tot=0;

     sum_L3L4_L1=0;
     sum_L3L4_L2=0;
     sum_L3L4_L5=0;
     sum_L3L4_L6=0;
     sum_L3L4_tot=0;

     sum_L5L6_L1=0;
     sum_L5L6_L2=0;
     sum_L5L6_L3=0;
     sum_L5L6_L4=0;
     sum_L5L6_tot=0;
   }

   //cout <<layer<<" "<<r<<endl;

   oldneighbor=neighbor;

   if (neighbor!=0&&seedlayer==1) {
     if (projlay==3) {
       hist01->Fill(nproj);
       sum_L1L2_L3+=nproj;
     }
     if (projlay==4) {
       hist02->Fill(nproj);
       sum_L1L2_L4+=nproj;
     }
     if (projlay==5) {
       hist03->Fill(nproj);
       sum_L1L2_L5+=nproj;
     }
     if (projlay==6) {
       hist04->Fill(nproj);
       sum_L1L2_L6+=nproj;
     }
     sum_L1L2_tot+=nproj;     
   }

   if (neighbor!=0&&seedlayer==3) {
     if (projlay==1) {
       sum_L3L4_L1+=nproj;
     }
     if (projlay==2) {
       sum_L3L4_L2+=nproj;
     }
     if (projlay==5) {
       sum_L3L4_L5+=nproj;
     }
     if (projlay==6) {
       sum_L3L4_L6+=nproj;
     }
     sum_L3L4_tot+=nproj;     
   }

   if (neighbor!=0&&seedlayer==5) {
     if (projlay==1) {
       sum_L5L6_L1+=nproj;
     }
     if (projlay==2) {
       sum_L5L6_L2+=nproj;
     }
     if (projlay==3) {
       sum_L5L6_L3+=nproj;
     }
     if (projlay==4) {
       sum_L5L6_L4+=nproj;
     }
     sum_L5L6_tot+=nproj;     
   }


   count++;

 }

 hist11->Fill(sum_L1L2_L3);
 hist12->Fill(sum_L1L2_L4);
 hist13->Fill(sum_L1L2_L5);
 hist14->Fill(sum_L1L2_L6);
 hist15->Fill(sum_L1L2_tot);

 hist21->Fill(sum_L3L4_L1);
 hist22->Fill(sum_L3L4_L2);
 hist23->Fill(sum_L3L4_L5);
 hist24->Fill(sum_L3L4_L6);
 hist25->Fill(sum_L3L4_tot);

 hist31->Fill(sum_L5L6_L1);
 hist32->Fill(sum_L5L6_L2);
 hist33->Fill(sum_L5L6_L3);
 hist34->Fill(sum_L5L6_L4);
 hist35->Fill(sum_L5L6_tot);
 
 
 cout << "Processed: "<<count<<" events"<<endl;
 
 c1->cd(1);
 gPad->SetLogy();
 h01->Draw();

 c1->cd(2);
 gPad->SetLogy();
 h02->Draw();

 c1->cd(3);
 gPad->SetLogy();
 h03->Draw();

 c1->cd(4);
 gPad->SetLogy();
 h04->Draw();


 c2->cd(1);
 gPad->SetLogy();
 h11->Draw();

 c2->cd(2);
 gPad->SetLogy();
 h12->Draw();

 c2->cd(3);
 gPad->SetLogy();
 h13->Draw();

 c2->cd(4);
 gPad->SetLogy();
 h14->Draw();

 c2->cd(5);
 gPad->SetLogy();
 h15->Draw();

 c2->Print("L1L2_projections.png");
 c2->Print("L1L2_projections.pdf");

 c3->cd(1);
 gPad->SetLogy();
 h21->Draw();

 c3->cd(2);
 gPad->SetLogy();
 h22->Draw();

 c3->cd(3);
 gPad->SetLogy();
 h23->Draw();

 c3->cd(4);
 gPad->SetLogy();
 h24->Draw();

 c3->cd(5);
 gPad->SetLogy();
 h25->Draw();

 c3->Print("L3L4_projections.png");
 c3->Print("L3L4_projections.pdf");


 c4->cd(1);
 gPad->SetLogy();
 h31->Draw();

 c4->cd(2);
 gPad->SetLogy();
 h32->Draw();

 c4->cd(3);
 gPad->SetLogy();
 h33->Draw();

 c4->cd(4);
 gPad->SetLogy();
 h34->Draw();

 c4->cd(5);
 gPad->SetLogy();
 h35->Draw();

 c4->Print("L5L6_projections.png");
 c4->Print("L5L6_projections.pdf");


 c5->cd(1);
 gPad->SetLogy();
 h15->GetXaxis()->SetTitle("Number of projections");
 h15->Draw();

 c5->cd(2);
 gPad->SetLogy();
 h25->GetXaxis()->SetTitle("Number of projections");
 h25->Draw();

 c5->cd(3);
 gPad->SetLogy();
 h35->GetXaxis()->SetTitle("Number of projections");
 h35->Draw();

 c5->Print("ProjectionSummary.png");
 c5->Print("ProjectionSummary.pdf");


}

