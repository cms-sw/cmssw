#include <iostream>
#include <fstream>
#include <iomanip>
#include <string>
#include <list>

#include <math.h>
#include <vector>

#include "Rtypes.h"
#include "TROOT.h"
#include "TRint.h"
#include "TObject.h"
#include "TFile.h"
// #include "TTree.h"
#include "TH1F.h"
#include "TCanvas.h"
#include "TApplication.h"
#include "TRefArray.h"
#include "TStyle.h"
#include "TGraph.h"
#include "TLegend.h"

void plotKine( std::string topology )
{

   gStyle->SetOptStat(0);
   
   std::string fnamePy6 = "GenJets_MG_Py6_" + topology + ".root";
   std::string fnamePy8 = "GenJets_MG_Py8_" + topology + ".root";
   
   TFile* f1 = new TFile( fnamePy6.c_str() );
   TFile* f2 = new TFile( fnamePy8.c_str() );
   
   TLegend* leg1 = new TLegend(0.6, 0.70, 0.9, 0.9);
   TLegend* leg2 = new TLegend(0.5, 0.70, 0.9, 0.9);
   TLegend* leg3 = new TLegend(0.5, 0.70, 0.9, 0.9);
   TLegend* leg4 = new TLegend(0.5, 0.70, 0.9, 0.9);
   
   // Py6
   //
   f1->cd("test");

   NJets->SetMarkerStyle(21);
   NJets->SetMarkerColor(kBlue);
   NJets->SetMarkerSize(0.9);

   NJetsAboveQCut->SetLineColor(kBlue);
   NJetsAboveQCut->SetLineWidth(2);
   
   LeadingJetPt->SetLineColor(kBlue);
   LeadingJetPt->SetLineWidth(2);
   Next2LeadingJetPt->SetLineColor(kBlue);
   Next2LeadingJetPt->SetLineWidth(2);
   LowestJetHt->SetLineColor(kBlue);
   LowestJetHt->SetLineWidth(2);     
     
   f2->cd("test");

   
   NJets->SetMarkerStyle(21);
   NJets->SetMarkerColor(kRed);
   NJets->SetMarkerSize(0.9);

   NJetsAboveQCut->SetLineColor(kRed);
   NJetsAboveQCut->SetLineWidth(2);

   LeadingJetPt->SetLineColor(kRed);
   LeadingJetPt->SetLineWidth(2);
   Next2LeadingJetPt->SetLineColor(kRed);
   Next2LeadingJetPt->SetLineWidth(2);
   LowestJetHt->SetLineColor(kRed);
   LowestJetHt->SetLineWidth(2);     

      
   TCanvas* myc1 = new TCanvas( "myc1", topology.c_str(), 800, 600 );
   myc1->Divide(2,1);

//   TPad* p2 = new TPad( "p2", "", 0.01, 0.34, 0.49, 0.66 );
//   TPad* p1 = new TPad( "p1", "", 0.01, 0.66, 0.99, 0.99 );
//   TPad* p3 = new TPad( "p3", "", 0.51, 0.34, 0.99, 0.66 );
//   TPad* p4 = new TPad( "p4", "", 0.01, 0.01, 0.49, 0.33 );
//   TPad* p5 = new TPad( "p5", "", 0.51, 0.01, 0.99, 0.33 );
   
//   p1->Draw();
//   p2->Draw();
//   p3->Draw();
//   p4->Draw();
//   p5->Draw();

//   p1->cd();

   myc1->cd(1);
   f1->cd("test");
   NJets->Draw("pe");
   leg1->AddEntry( NJets, "Pythia6", "p");
   f2->cd("test");
   NJets->Draw("pesame"); //pesame");
   leg1->AddEntry( NJets, "Pythia8", "p");
   leg1->Draw();
   
   float scale = 1.;
   
   myc1->cd(2);
   f1->cd("test");
//   scale = 1./NJets->Integral();
//   NJets->DrawNormalized("p", scale);
   leg2->SetTextSize(0.05);
   leg2->AddEntry( "", "Normalized", "" );
   leg2->AddEntry( NJets, "Pythia6", "p");
   f2->cd("test");
   scale = 1./NJets->Integral();
   NJets->DrawNormalized("p", scale);
   leg2->AddEntry( NJets, "Pythia8", "p");
   f1->cd("test");
   scale = 1./NJets->Integral();
   NJets->DrawNormalized("psame", scale);
   leg2->Draw();
      
   TCanvas* myc2 = new TCanvas( "myc2", topology.c_str(), 800, 600 );
   myc2->Divide(2,1);

   float scale1 = 1.;
   float scale2 = 1.;
   
   myc2->cd(1);
   f1->cd("test");
   NJetsAboveQCut->Draw(); 
   leg3->AddEntry( NJetsAboveQCut, "Pythia6" ); 
   f2->cd("test");
   NJetsAboveQCut->Draw("same"); 
   leg3->AddEntry( NJetsAboveQCut, "Pythia8" ); 
   //f1->cd("test");
   //NJetsAboveQCut->Draw("same"); 
   leg3->Draw();
   
   myc2->cd(2);
   f1->cd("test");
   //scale1 = 1./NJetsAboveQCut->Integral();
   //NJetsAboveQCut->DrawNormalized("", scale1);
   leg4->SetTextSize(0.05);
   leg4->AddEntry( "", "Normalized", "" );
   leg4->AddEntry( NJetsAboveQCut, "Pythia6" ); 
   f2->cd("test");
   scale2 = 1./NJetsAboveQCut->Integral();
   leg4->AddEntry( NJetsAboveQCut, "Pythia8" );
   NJetsAboveQCut->DrawNormalized("", scale2);
   f1->cd("test");
   scale1 = 1./NJetsAboveQCut->Integral();
   NJetsAboveQCut->DrawNormalized("same", scale1);
   leg4->Draw();
      
   TCanvas* myc3 = new TCanvas( "myc3", topology.c_str(), 1200, 600 );
   myc3->Divide(3,1);

//   p2->cd();

   myc3->cd(1);
   f2->cd("test");
   LeadingJetPt->DrawNormalized("",scale2);
   f1->cd("test");
   LeadingJetPt->DrawNormalized("same",scale1);
   leg4->Draw();
   
   myc3->cd(2);
   f2->cd("test");
   Next2LeadingJetPt->DrawNormalized("",scale2);
   f1->cd("test");
   Next2LeadingJetPt->DrawNormalized("same",scale1);
   leg4->Draw();
   
   myc3->cd(3);
   f2->cd("test");
   LowestJetHt->DrawNormalized("",scale2);
   f1->cd("test");
   LowestJetHt->DrawNormalized("same",scale1);
   leg4->Draw();
     
   myc3->cd();
   
   return;

}
