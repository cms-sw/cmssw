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
gStyle->SetOptTitle(0);

  // For publishing:
  gStyle->SetLineWidth(1.5);
  gStyle->SetTextSize(1.0);
  gStyle->SetLabelSize(0.06,"xy");
  gStyle->SetTitleSize(0.06,"xy");
  gStyle->SetTitleOffset(1.2,"x");
  gStyle->SetTitleOffset(1.0,"y");
  gStyle->SetPadTopMargin(0.1);
  gStyle->SetPadRightMargin(0.1);
  gStyle->SetPadBottomMargin(0.16);
  gStyle->SetPadLeftMargin(0.12);




 c1 = new TCanvas("c1","Track performance",200,50,600,700);
 c1->Divide(1,1);
 c1->SetFillColor(0);
 c1->SetGrid();

 c2 = new TCanvas("c2","Track performance",200,50,600,700);
 c2->Divide(2,4);
 c2->SetFillColor(0);
 c2->SetGrid();

 c3 = new TCanvas("c3","Track performance",200,50,600,700);
 c3->Divide(2,3);
 c3->SetFillColor(0);
 c3->SetGrid();

 c4 = new TCanvas("c4","Track performance",300,50,600,700);
 c4->Divide(2,3);
 c4->SetFillColor(0);
 c4->SetGrid();

 c5 = new TCanvas("c5","Track performance",400,50,600,700);
 c5->Divide(2,3);
 c5->SetFillColor(0);
 c5->SetGrid();

 c6 = new TCanvas("c6","Track performance",500,50,600,700);
 c6->Divide(2,3);
 c6->SetFillColor(0);
 c6->SetGrid();

 double maxstub=150.0;

 TH1 *hist10 = new TH1F("h10","Number of stubs in sector",80,0.0,800.0);
 TH1 *hist1 = new TH1F("h1","Number of stubs z1",50,0.0,maxstub);
 TH1 *hist2 = new TH1F("h2","Number of stubs z2",50,0.0,maxstub);
 TH1 *hist3 = new TH1F("h3","Number of stubs z3",50,0.0,maxstub);
 TH1 *hist4 = new TH1F("h4","Number of stubs z4",50,0.0,maxstub);
 TH1 *hist5 = new TH1F("h5","Number of stubs z5",50,0.0,maxstub);
 TH1 *hist6 = new TH1F("h6","Number of stubs z6",50,0.0,maxstub);
 TH1 *hist7 = new TH1F("h7","Number of stubs z7",50,0.0,maxstub);
 TH1 *hist8 = new TH1F("h8","Number of stubs z8",50,0.0,maxstub);

 TH1 *hist12 = new TH1F("h12","Number of stubs z1+z2",50,0.0,maxstub);
 TH1 *hist78 = new TH1F("h78","Number of stubs z7+z8",50,0.0,maxstub);

 TH1 *hist121 = new TH1F("h121","Number of stubs z12 D1",50,0.0,100.0);
 TH1 *hist122 = new TH1F("h122","Number of stubs z12 D2",50,0.0,100.0);
 TH1 *hist123 = new TH1F("h123","Number of stubs z12 D3",50,0.0,100.0);
 TH1 *hist124 = new TH1F("h124","Number of stubs z12 D4",50,0.0,100.0);
 TH1 *hist125 = new TH1F("h125","Number of stubs z12 D5",50,0.0,100.0);

 TH1 *hist131 = new TH1F("h131","Number of stubs z3 L1",50,0.0,100.0);
 TH1 *hist132 = new TH1F("h132","Number of stubs z3 L2",50,0.0,100.0);
 TH1 *hist133 = new TH1F("h133","Number of stubs z3 L3",50,0.0,100.0);
 TH1 *hist134 = new TH1F("h134","Number of stubs z3 L4",50,0.0,100.0);
 TH1 *hist135 = new TH1F("h135","Number of stubs z3 L5",50,0.0,100.0);
 TH1 *hist136 = new TH1F("h136","Number of stubs z3 L6",50,0.0,100.0);

 TH1 *hist141 = new TH1F("h141","Number of stubs z4 L1",50,0.0,100.0);
 TH1 *hist142 = new TH1F("h142","Number of stubs z4 L2",50,0.0,100.0);
 TH1 *hist143 = new TH1F("h143","Number of stubs z4 L3",50,0.0,100.0);
 TH1 *hist144 = new TH1F("h144","Number of stubs z4 L4",50,0.0,100.0);
 TH1 *hist145 = new TH1F("h145","Number of stubs z4 L5",50,0.0,100.0);
 TH1 *hist146 = new TH1F("h146","Number of stubs z4 L6",50,0.0,100.0);

 


 ifstream in("stubs.txt");

 int count=0;

 while (in.good()){

   int nstubs;
   int z11,z21,z31,z41,z51,z61,z71,z81;
   int z12,z22,z32,z42,z52,z62,z72,z82;
   int z13,z23,z33,z43,z53,z63,z73,z83;
   int z14,z24,z34,z44,z54,z64,z74,z84;
   int z15,z25,z35,z45,z55,z65,z75,z85;
   int z16,z26,z36,z46,z56,z66,z76,z86;
  

   in >>z11>>z12>>z13>>z14>>z15>>z16;
   in >>z21>>z22>>z23>>z24>>z25>>z26;
   in >>z31>>z32>>z33>>z34>>z35>>z36;
   in >>z41>>z42>>z43>>z44>>z45>>z46;
   in >>z51>>z52>>z53>>z54>>z55>>z56;
   in >>z61>>z62>>z63>>z64>>z65>>z66;
   in >>z71>>z72>>z73>>z74>>z75>>z76;
   in >>z81>>z82>>z83>>z84>>z85>>z86;

   if (!in.good()) continue;

   nstubs=z11+z21+z31+z41+z51+z61+z71+z81+
     z12+z22+z32+z42+z52+z62+z72+z82+
     z13+z23+z33+z43+z53+z63+z73+z83+
     z14+z24+z34+z44+z54+z64+z74+z84+
     z15+z25+z35+z45+z55+z65+z75+z85+
     z16+z26+z36+z46+z56+z66+z76+z86;

   hist10->Fill(nstubs);
   hist1->Fill(z11+z12+z13+z14+z15+z16);
   hist2->Fill(z21+z22+z23+z24+z25+z26);
   hist3->Fill(z31+z32+z33+z34+z35+z36);
   hist4->Fill(z41+z42+z43+z44+z45+z46);
   hist5->Fill(z51+z52+z53+z54+z55+z56);
   hist6->Fill(z61+z62+z63+z64+z65+z66);
   hist7->Fill(z71+z72+z73+z74+z75+z76);
   hist8->Fill(z81+z82+z83+z84+z85+z86);

   hist12->Fill(z11+z12+z13+z14+z15+z16+z21+z22+z23+z24+z25+z26);
   hist78->Fill(z71+z72+z73+z74+z75+z76+z81+z82+z83+z84+z85+z86);

   hist121->Fill(z11+z21);
   hist122->Fill(z12+z22);
   hist123->Fill(z13+z23);
   hist124->Fill(z14+z24);
   hist125->Fill(z15+z25);

   hist131->Fill(z31);
   hist132->Fill(z32);
   hist133->Fill(z33);
   hist134->Fill(z34);
   hist135->Fill(z35);
   hist136->Fill(z36);

   hist141->Fill(z41);
   hist142->Fill(z42);
   hist143->Fill(z43);
   hist144->Fill(z44);
   hist145->Fill(z45);
   hist146->Fill(z46);

   count++;

 }

 cout << "Processed: "<<count<<" events"<<endl;

 c1->cd(1);
 h10->GetYaxis()->SetTitle("Entries");
 h10->GetXaxis()->SetTitle("Stubs per sector");
 h10->Draw();

 c1->Print("stubs.png");
 c1->Print("stubs.pdf");

 c2->cd(1);
 h1->GetYaxis()->SetTitle("Entries");
 h1->GetXaxis()->SetTitle("Stubs per sector in region z1");
 h1->Draw();

 c2->cd(2);
 h2->GetYaxis()->SetTitle("Entries");
 h2->GetXaxis()->SetTitle("Stubs per sector in region z2");
 h2->Draw();

 c2->cd(3);
 h3->GetYaxis()->SetTitle("Entries");
 h3->GetXaxis()->SetTitle("Stubs per sector in region z3");
 h3->Draw();

 c2->cd(4);
 h4->GetYaxis()->SetTitle("Entries");
 h4->GetXaxis()->SetTitle("Stubs per sector in region z4");
 h4->Draw();

 c2->cd(5);
 h5->GetYaxis()->SetTitle("Entries");
 h5->GetXaxis()->SetTitle("Stubs per sector in region z5");
 h5->Draw();

 c2->cd(6);
 h6->GetYaxis()->SetTitle("Entries");
 h6->GetXaxis()->SetTitle("Stubs per sector in region z6");
 h6->Draw();

 c2->cd(7);
 h7->GetYaxis()->SetTitle("Entries");
 h7->GetXaxis()->SetTitle("Stubs per sector in region z7");
 h7->Draw();

 c2->cd(8);
 h8->GetYaxis()->SetTitle("Entries");
 h8->GetXaxis()->SetTitle("Stubs per sector in region z8");
 h8->Draw();

 c2->Print("stubsfedregion8.png");
 c2->Print("stubsfedregion8.pdf");

 c3->cd(1);
 h12->GetYaxis()->SetTitle("Entries");
 h12->GetXaxis()->SetTitle("Stubs per sector in region z1+z2");
 h12->Draw();

 c3->cd(2);
 h3->Draw();

 c3->cd(3);
 h4->Draw();

 c3->cd(4);
 h5->Draw();

 c3->cd(5);
 h6->Draw();

 c3->cd(6);
 h78->GetYaxis()->SetTitle("Entries");
 h78->GetXaxis()->SetTitle("Stubs per sector in region z7+z8");
 h78->Draw();

 c3->Print("stubsfedregion6.png");
 c3->Print("stubsfedregion6.pdf");

 c4->cd(1);
 h131->Draw();

 c4->cd(2);
 h132->Draw();

 c4->cd(3);
 h133->Draw();

 c4->cd(4);
 h134->Draw();

 c4->cd(5);
 h135->Draw();

 c4->cd(6);
 h136->Draw();

 c4->Print("stubsfedregion1.png");
 c4->Print("stubsfedregion1.pdf");

 c5->cd(1);
 h141->Draw();

 c5->cd(2);
 h142->Draw();

 c5->cd(3);
 h143->Draw();

 c5->cd(4);
 h144->Draw();

 c5->cd(5);
 h145->Draw();

 c5->cd(6);
 h146->Draw();

 c5->Print("stubsfedregion2.png");
 c5->Print("stubsfedregion2.pdf");

 c6->cd(1);
 h121->Draw();

 c6->cd(2);
 h122->Draw();

 c6->cd(3);
 h123->Draw();

 c6->cd(4);
 h124->Draw();

 c6->cd(5);
 h125->Draw();

 c5->Print("stubsfedregionfdisks.png");
 c5->Print("stubsfedregionfdisks.pdf");


}

