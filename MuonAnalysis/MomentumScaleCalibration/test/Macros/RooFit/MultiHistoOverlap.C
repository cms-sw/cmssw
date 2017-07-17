//-----------------------------------------//
// Version of the script which takes as input al list of filenames and labels
//root [0] .L MultiHistoOverlap.C++
//root [1] MultiHistoOverlap("BiasCheck_IDEAL.root=MC design,  BiasCheck_STARTUP.root=MC startup, BiasCheck_MP1073.root=Summer2011 Tk geometry",3)
//root [1] MultiHistoOverlap("BiasCheck_IDEAL.root=MC design,  BiasCheck_STARTUP.root=MC startup, BiasCheck_MP1073.root=Summer2011 Tk geometry, BiasCheck_MP0743.root=Tk Summer2011 (no mass constraint)",4)
//------------------------------------------//

#include <iostream>
#include <vector>
#include "Gtypes.h"
#include "TROOT.h"
#include "TStyle.h"
#include "TString.h"
#include "TPaveText.h"
#include "TFile.h"
#include "TH1.h"
#include "TH2.h"
#include "TLegend.h"
#include "TCanvas.h"
#include "TList.h"
#include "TMath.h"
#include "RooPlot.h"
#include "TAttMarker.h"

const Color_t colorlist[7]={kRed,kBlack,kBlack,kGreen,kMagenta,kCyan,kTeal};
const Int_t linestylelist[7]={1,1,1,1,1,1,1};
const Int_t stylelist[7]={1,1,1,1,1,1,1};
const Style_t markerstylelist[7]={kFullSquare,kFullCircle,kOpenCircle,kOpenTriangleUp,kOpenCircle,kOpenTriangleUp,kOpenCircle};
        
void MultiHistoOverlap(TString namesandlabels, Int_t nOfFiles, const TString& outDir="./"){

  gROOT->Reset();
  gROOT->ProcessLine(".L tdrstyle.C"); 
  gROOT->ProcessLine("setTDRStyle()");
 
 //  gSystem->Load("libRooFit");
 //  using namespace RooFit;
 // preamble
  TPaveText *cmsprel = new TPaveText(0.19, 0.95, 0.95, 0.99, "NDC");
  cmsprel->SetTextSize(0.03);
  cmsprel->SetTextFont(42);
  cmsprel->SetFillColor(0);
  cmsprel->SetBorderSize(0);
  cmsprel->SetMargin(0.01);
  cmsprel->SetTextAlign(12); // align left
  TString text = "CMS Preliminary 2011";
  cmsprel->AddText(0.0, 0.5,text);  
  TString text2 = "#sqrt{s} = 7 TeV  |#eta_{#mu}|<2.4";             
  cmsprel->AddText(0.8, 0.5, text2);


  TList* FileList  = new TList();  
  TList* LabelList = new TList();    
  TObjArray *nameandlabelpairs = namesandlabels.Tokenize(",");  
  for (Int_t i = 0; i < nameandlabelpairs->GetEntries(); ++i) {    
    TObjArray *aFileLegPair = TString(nameandlabelpairs->At(i)->GetName()).Tokenize("=");       
    if(aFileLegPair->GetEntries() == 2) {      
      FileList->Add( TFile::Open(aFileLegPair->At(0)->GetName())  ); 
      LabelList->Add( aFileLegPair->At(1) );    
    } else {      
      std::cout << "Please give file name and legend entry in the following form:\n" 		<< " filename1=legendentry1,filename2=legendentry2\n";          
    }  
  }
  

 Int_t NOfFiles =  FileList->GetSize();  
 if ( NOfFiles!=nOfFiles ){
   std::cout<<"&MSG-e: NOfFiles = "<<nOfFiles<<std::endl;  
   return;
 }  
 

 std::vector<TString> LegLabels;
 LegLabels.reserve(nOfFiles);    
 for(Int_t j=0; j < nOfFiles; j++) {       
   TObjString* legend = (TObjString*)LabelList->At(j);    
   LegLabels.push_back(legend->String());
   std::cout<<"LegLabels["<<j<<"]"<<LegLabels[j]<<std::endl;  
 }

 TLegend *leg=0; 

 TCanvas* c0 = new TCanvas("c0", "c0",50, 20, 800,600);
 TCanvas* c1 = new TCanvas("c1", "c1",50, 20, 800,600);
 TCanvas* c2 = new TCanvas("c2", "c2",50, 20, 800,600);
 TCanvas* c3 = new TCanvas("c3", "c3",50, 20, 800,600);
 TCanvas* c4 = new TCanvas("c4", "c4",50, 20, 800,600);
 TCanvas* c5 = new TCanvas("c5", "c5",50, 20, 1200,800);
 TCanvas* c6 = new TCanvas("c6", "c6",50, 20, 1200,800);

 TCanvas* c0s = new TCanvas("c0s", "c0s",50, 20, 800,600);
 TCanvas* c1s = new TCanvas("c1s", "c1s",50, 20, 800,600);
 TCanvas* c2s = new TCanvas("c2s", "c2s",50, 20, 800,600);
 TCanvas* c3s = new TCanvas("c3s", "c3s",50, 20, 800,600);

 TCanvas* cFit = new TCanvas("cFit", "cFit",50, 20, 1600, 800);


 //----------------- CANVAS C0 --------------//
 c0->SetFillColor(0);  
 c0->cd();

 leg = new TLegend(0.50,0.25,0.90,0.40);  
 leg->SetBorderSize(1);
 leg->SetFillColor(0);
 leg->SetTextFont(42);
 
// Mass VS muon phi plus -------------------------------
 TH1D *histoMassVsPhiPlus[nOfFiles];
 for(Int_t j=0; j < nOfFiles; j++) {     
   
   TFile *fin = (TFile*)FileList->At(j);    
   if (( histoMassVsPhiPlus[j] = (TH1D*)fin->Get("MassVsPhiPlus/allHistos/meanHisto"))){
     histoMassVsPhiPlus[j]->SetLineStyle(linestylelist[j]);
     histoMassVsPhiPlus[j]->SetMarkerColor(colorlist[j]);
     histoMassVsPhiPlus[j]->SetLineColor(colorlist[j]);
     histoMassVsPhiPlus[j]->SetMarkerStyle(markerstylelist[j]); 
     //     histoMassVsPhiPlus[j]->SetMarkerSize(0.75);
     if ( j == 0 ) {
       histoMassVsPhiPlus[j]->GetXaxis()->SetTitle("positive muon #phi (rad)");
       histoMassVsPhiPlus[j]->GetYaxis()->SetTitle("M_{#mu#mu} (GeV)");
       //       histoMassVsPhiPlus[j]->GetYaxis()->SetRangeUser(88.5,93.5);
       histoMassVsPhiPlus[j]->GetYaxis()->SetRangeUser(90.0,91.5);
       histoMassVsPhiPlus[j]->GetXaxis()->SetRangeUser(-3.14,3.14);
       histoMassVsPhiPlus[j]->Draw();
     } else {
       histoMassVsPhiPlus[j]->Draw("SAME");
     }
     leg->AddEntry(histoMassVsPhiPlus[j],LegLabels[j],"PL");  
   }
 }
 //cmsprel->Draw("same");
 leg->Draw("same");
 c0->SaveAs(outDir+"MassVsPhiPlus.png"); 


 //----------------- CANVAS C1 --------------//
 c1->SetFillColor(0);  
 c1->cd();

 leg = new TLegend(0.50,0.25,0.90,0.40);  
 leg->SetBorderSize(1);
 leg->SetFillColor(0);
 leg->SetTextFont(42);
 
// Mass VS muon eta plus -------------------------------
 TH1D *histoMassVsEtaPlus[nOfFiles];
 for(Int_t j=0; j < nOfFiles; j++) {     
   
   TFile *fin = (TFile*)FileList->At(j);    
   if (( histoMassVsEtaPlus[j] = (TH1D*)fin->Get("MassVsEtaPlus/allHistos/meanHisto"))){
     histoMassVsEtaPlus[j]->SetLineStyle(linestylelist[j]);
     histoMassVsEtaPlus[j]->SetMarkerColor(colorlist[j]);
     histoMassVsEtaPlus[j]->SetLineColor(colorlist[j]);
     histoMassVsEtaPlus[j]->SetMarkerStyle(markerstylelist[j]); 
     //     histoMassVsEtaPlus[j]->SetMarkerSize(0.75);
     if ( j == 0 ) {
       histoMassVsEtaPlus[j]->GetXaxis()->SetTitle("positive muon #eta");
       histoMassVsEtaPlus[j]->GetYaxis()->SetTitle("M_{#mu#mu} (GeV)");
       //       histoMassVsEtaPlus[j]->GetYaxis()->SetRangeUser(88.5,93.5);
       histoMassVsEtaPlus[j]->GetYaxis()->SetRangeUser(90.0,91.5);
       histoMassVsEtaPlus[j]->GetXaxis()->SetRangeUser(-2.41,2.41);
       histoMassVsEtaPlus[j]->Draw();
     } else {
       histoMassVsEtaPlus[j]->Draw("SAME");
     }
     leg->AddEntry(histoMassVsEtaPlus[j],LegLabels[j],"PL");  
   }
 }
 //cmsprel->Draw("same");
 leg->Draw("same");
 c1->SaveAs(outDir+"MassVsEtaPlus.png"); 


 //----------------- CANVAS C2 --------------//
 c2->SetFillColor(0);  
 c2->cd();

 leg = new TLegend(0.50,0.25,0.90,0.40);  
 leg->SetBorderSize(1);
 leg->SetFillColor(0);
 leg->SetTextFont(42);
 
// Mass VS muon eta plus - eta minus  -------------------------------
 TH1D *histoMassVsEtaPlusMinusDiff[nOfFiles];
 for(Int_t j=0; j < nOfFiles; j++) {     
   
   TFile *fin = (TFile*)FileList->At(j);    
   if (( histoMassVsEtaPlusMinusDiff[j] = (TH1D*)fin->Get("MassVsEtaPlusMinusDiff/allHistos/meanHisto"))){
     histoMassVsEtaPlusMinusDiff[j]->SetLineStyle(linestylelist[j]);
     histoMassVsEtaPlusMinusDiff[j]->SetMarkerColor(colorlist[j]);
     histoMassVsEtaPlusMinusDiff[j]->SetLineColor(colorlist[j]);
     histoMassVsEtaPlusMinusDiff[j]->SetMarkerStyle(markerstylelist[j]); 
     //     histoMassVsEtaPlusMinusDiff[j]->SetMarkerSize(0.75);
     if ( j == 0 ) {
       histoMassVsEtaPlusMinusDiff[j]->GetXaxis()->SetTitle("#eta pos. muon  #eta neg. muon");
       histoMassVsEtaPlusMinusDiff[j]->GetYaxis()->SetTitle("M_{#mu#mu} (GeV)");
       //       histoMassVsEtaPlusMinusDiff[j]->GetYaxis()->SetRangeUser(88.0,96.0);
       histoMassVsEtaPlusMinusDiff[j]->GetYaxis()->SetRangeUser(90.0,91.5);
       histoMassVsEtaPlusMinusDiff[j]->GetXaxis()->SetRangeUser(-3,3);
       histoMassVsEtaPlusMinusDiff[j]->Draw();
     } else {
       histoMassVsEtaPlusMinusDiff[j]->Draw("SAME");
     }
     leg->AddEntry(histoMassVsEtaPlusMinusDiff[j],LegLabels[j],"PL");  
   }
 }
 //cmsprel->Draw("same");
 leg->Draw("same");
 c2->SaveAs(outDir+"MassVsEtaPlusMinusDiff.png"); 

 //----------------- CANVAS C3 --------------//
 c3->SetFillColor(0);  
 c3->cd();

 leg = new TLegend(0.50,0.25,0.90,0.40);  
 leg->SetBorderSize(1);
 leg->SetFillColor(0);
 leg->SetTextFont(42);
 
// Mass VS muon phi minus -------------------------------
 TH1D *histoMassVsPhiMinus[nOfFiles];
 for(Int_t j=0; j < nOfFiles; j++) {     
   
   TFile *fin = (TFile*)FileList->At(j);    
   if (( histoMassVsPhiMinus[j] = (TH1D*)fin->Get("MassVsPhiMinus/allHistos/meanHisto"))){
     histoMassVsPhiMinus[j]->SetLineStyle(linestylelist[j]);
     histoMassVsPhiMinus[j]->SetMarkerColor(colorlist[j]);
     histoMassVsPhiMinus[j]->SetLineColor(colorlist[j]);
     histoMassVsPhiMinus[j]->SetMarkerStyle(markerstylelist[j]); 
     //     histoMassVsPhiMinus[j]->SetMarkerSize(0.75);
     if ( j == 0 ) {
       histoMassVsPhiMinus[j]->GetXaxis()->SetTitle("negative muon #phi (rad)");
       histoMassVsPhiMinus[j]->GetYaxis()->SetTitle("M_{#mu#mu} (GeV)");
       //       histoMassVsPhiMinus[j]->GetYaxis()->SetRangeUser(88.5,93.5);
       histoMassVsPhiMinus[j]->GetYaxis()->SetRangeUser(90.0,91.5);
       histoMassVsPhiMinus[j]->GetXaxis()->SetRangeUser(-3.14,3.14);
       histoMassVsPhiMinus[j]->Draw();
     } else {
       histoMassVsPhiMinus[j]->Draw("SAME");
     }
     leg->AddEntry(histoMassVsPhiMinus[j],LegLabels[j],"PL");  
   }
 }
 //cmsprel->Draw("same");
 leg->Draw("same");
 c3->SaveAs(outDir+"MassVsPhiMinus.png"); 


 //----------------- CANVAS C4 --------------//
 c4->SetFillColor(0);  
 c4->cd();

 leg = new TLegend(0.50,0.25,0.90,0.40);  
 leg->SetBorderSize(1);
 leg->SetFillColor(0);
 leg->SetTextFont(42);
 
// Mass VS muon eta minus -------------------------------
 TH1D *histoMassVsEtaMinus[nOfFiles];
 for(Int_t j=0; j < nOfFiles; j++) {     
   
   TFile *fin = (TFile*)FileList->At(j);    
   if (( histoMassVsEtaMinus[j] = (TH1D*)fin->Get("MassVsEtaMinus/allHistos/meanHisto"))){
     histoMassVsEtaMinus[j]->SetLineStyle(linestylelist[j]);
     histoMassVsEtaMinus[j]->SetMarkerColor(colorlist[j]);
     histoMassVsEtaMinus[j]->SetLineColor(colorlist[j]);
     histoMassVsEtaMinus[j]->SetMarkerStyle(markerstylelist[j]); 
     //     histoMassVsEtaMinus[j]->SetMarkerSize(0.75);
     if ( j == 0 ) {
       histoMassVsEtaMinus[j]->GetXaxis()->SetTitle("negative muon #eta");
       histoMassVsEtaMinus[j]->GetYaxis()->SetTitle("M_{#mu#mu} (GeV)");
       //       histoMassVsEtaMinus[j]->GetYaxis()->SetRangeUser(88.5,93.5);
       histoMassVsEtaMinus[j]->GetYaxis()->SetRangeUser(90.0,91.5);
       histoMassVsEtaMinus[j]->GetXaxis()->SetRangeUser(-2.41,2.41);
       histoMassVsEtaMinus[j]->Draw();
     } else {
       histoMassVsEtaMinus[j]->Draw("SAME");
     }
     leg->AddEntry(histoMassVsEtaMinus[j],LegLabels[j],"PL");  
   }
 }
 //cmsprel->Draw("same");
 leg->Draw("same");
 c4->SaveAs(outDir+"MassVsEtaMinus.png"); 

 //----------------- CANVAS C5 --------------//
 c5->SetFillColor(0);  
 c5->cd();

 leg = new TLegend(0.50,0.25,0.90,0.40);  
 leg->SetBorderSize(1);
 leg->SetFillColor(0);
 leg->SetTextFont(42);
 
 // Mass VS muon phi plus -------------------------------
 TH2D *histoMassVsEtaPhiPlus[nOfFiles];

 TStyle *newStyle;
 newStyle->SetPalette(1);
 // newStyle->SetOptTitle(1);

 Double_t zMin(82.);
 Double_t zMax(96.);
 for(Int_t j=0; j < nOfFiles; j++) {     
   
   TFile *fin = (TFile*)FileList->At(j);    
   if (( histoMassVsEtaPhiPlus[j] = (TH2D*)fin->Get("MassVsEtaPhiPlus/allHistos/meanHisto"))){
     if ( j == 0 ) {
       histoMassVsEtaPhiPlus[j]->SetTitle(LegLabels[j]);
       histoMassVsEtaPhiPlus[j]->GetXaxis()->SetTitle("positive muon #phi (rad)");
       histoMassVsEtaPhiPlus[j]->GetYaxis()->SetTitle("positive muon #eta");
       zMin = histoMassVsEtaPhiPlus[j]->GetMinimum();
       zMax = histoMassVsEtaPhiPlus[j]->GetMaximum();
       histoMassVsEtaPhiPlus[j]->Draw("COLZ");
       c5->SaveAs(outDir+"MassVsEtaPhiPlus_file0.png"); 
     } else {
       histoMassVsEtaPhiPlus[j]->SetTitle(LegLabels[j]);
       histoMassVsEtaPhiPlus[j]->SetMinimum(zMin);
       histoMassVsEtaPhiPlus[j]->SetMaximum(zMax);
       histoMassVsEtaPhiPlus[j]->Draw("COLZ");
       c5->SaveAs(outDir+"MassVsEtaPhiPlus_file"+(TString)Form("%d",(Int_t)j)+".png"); 
     }

   }
 }
 //cmsprel->Draw("same");

//  //----------------- CANVAS C6 --------------//
//  c6->SetFillColor(0);  
//  c6->cd();

//  leg = new TLegend(0.50,0.25,0.90,0.40);  
//  leg->SetBorderSize(1);
//  leg->SetFillColor(0);
//  leg->SetTextFont(42);
 
//  // Mass VS muon phi minus -------------------------------
//  TH2D *histoMassVsEtaPhiMinus[nOfFiles];

//  for(Int_t j=0; j < nOfFiles; j++) {     
   
//    TFile *fin = (TFile*)FileList->At(j);    
//    if (( histoMassVsEtaPhiMinus[j] = (TH2D*)fin->Get("MassVsEtaPhiMinus/allHistos/meanHisto"))){
//      if ( j == 0 ) {
//        histoMassVsEtaPhiMinus[j]->GetXaxis()->SetTitle("negative muon #phi (rad)");
//        histoMassVsEtaPhiMinus[j]->GetYaxis()->SetTitle("negative muon #eta");
//        zMin = histoMassVsEtaPhiMinus[j]->GetMinimum();
//        zMax = histoMassVsEtaPhiMinus[j]->GetMaximum();
//        histoMassVsEtaPhiMinus[j]->Draw();
//      } else {
//        histoMassVsEtaPhiMinus[j]->SetMinimum(zMin);
//        histoMassVsEtaPhiMinus[j]->SetMaximum(zMax);
//        histoMassVsEtaPhiMinus[j]->Draw("SAME");
//      }
//      leg->AddEntry(histoMassVsEtaPhiMinus[j],LegLabels[j],"PL");  
//    }
//  }
//  //cmsprel->Draw("same");
//  leg->Draw("same");
//  c6->SaveAs(outDir+"MassVsEtaPhiMinus.png"); 

// newStyle->SetOptTitle(0);

 const Color_t colorlist_resol[7]={kBlack,kGreen,kBlue,kMagenta,kCyan,kTeal,kRed};
 const Int_t linestylelist_resol[7]={1,1,1,1,1,1,1};
 const Int_t stylelist_resol[7]={1,1,1,1,1,1,1};
 const Style_t markerstylelist_resol[7]={kOpenCircle,kOpenTriangleUp,kOpenTriangleUp,kOpenCircle,kOpenTriangleUp,kOpenCircle,kOpenTriangleUp};


//  //----------------- CANVAS C0S --------------//
//  c0s->SetFillColor(0);  
//  c0s->cd();

//  leg = new TLegend(0.50,0.25,0.90,0.40);  
//  leg->SetBorderSize(1);
//  leg->SetFillColor(0);
//  leg->SetTextFont(42);

// // Sigma VS muon phi plus -------------------------------
//  TH1D *histoSigmaVsPhiPlus[nOfFiles];
//  for(Int_t j=0; j < nOfFiles; j++) {     
   
//    TFile *fin = (TFile*)FileList->At(j);    
//    if (( histoSigmaVsPhiPlus[j] = (TH1D*)fin->Get("MassVsPhiPlus/allHistos/sigmaHisto"))){
//      histoSigmaVsPhiPlus[j]->SetLineStyle(linestylelist_resol[j]);
//      histoSigmaVsPhiPlus[j]->SetMarkerColor(colorlist_resol[j]);
//      histoSigmaVsPhiPlus[j]->SetLineColor(colorlist_resol[j]);
//      histoSigmaVsPhiPlus[j]->SetMarkerStyle(markerstylelist_resol[j]); 
//      //     histoSigmaVsPhiPlus[j]->SetMarkerSize(0.75);
//      if ( j == 0 ) {
//        histoSigmaVsPhiPlus[j]->GetXaxis()->SetTitle("positive muon #phi (rad)");
//        histoSigmaVsPhiPlus[j]->GetYaxis()->SetTitle("#sigma(M_{#mu#mu}) (GeV)");
//        //       histoSigmaVsPhiPlus[j]->GetYaxis()->SetRangeUser(88.5,93.5);
//        histoSigmaVsPhiPlus[j]->GetYaxis()->SetRangeUser(0.,3.);
//        histoSigmaVsPhiPlus[j]->GetXaxis()->SetRangeUser(-3.14,3.14);
//        histoSigmaVsPhiPlus[j]->Draw();
//      } else {
//        histoSigmaVsPhiPlus[j]->Draw("SAME");
//      }
//      leg->AddEntry(histoSigmaVsPhiPlus[j],LegLabels[j],"PL");  
//    }
//  }
//  //cmsprel->Draw("same");
//  leg->Draw("same");
//  c0s->SaveAs(outDir+"SigmaVsPhiPlus.png"); 


 //----------------- CANVAS C1S --------------//
 c1s->SetFillColor(0);  
 c1s->cd();

 leg = new TLegend(0.50,0.25,0.90,0.40);  
 leg->SetBorderSize(1);
 leg->SetFillColor(0);
 leg->SetTextFont(42);

 
// Sigma VS muon eta plus -------------------------------
 TH1D *histoSigmaVsEtaPlus[nOfFiles];
 for(Int_t j=0; j < nOfFiles; j++) {     
   
   TFile *fin = (TFile*)FileList->At(j);    
   if (( histoSigmaVsEtaPlus[j] = (TH1D*)fin->Get("MassVsEtaPlus/allHistos/sigmaHisto"))){
     histoSigmaVsEtaPlus[j]->SetLineStyle(linestylelist_resol[j]);
     histoSigmaVsEtaPlus[j]->SetMarkerColor(colorlist_resol[j]);
     histoSigmaVsEtaPlus[j]->SetLineColor(colorlist_resol[j]);
     histoSigmaVsEtaPlus[j]->SetMarkerStyle(markerstylelist_resol[j]); 
     //     histoSigmaVsEtaPlus[j]->SetMarkerSize(0.75);
     if ( j == 0 ) {
       histoSigmaVsEtaPlus[j]->GetXaxis()->SetTitle("positive muon #eta");
       histoSigmaVsEtaPlus[j]->GetYaxis()->SetTitle("#sigma(M_{#mu#mu}) (GeV)");
       //       histoSigmaVsEtaPlus[j]->GetYaxis()->SetRangeUser(88.5,93.5);
       histoSigmaVsEtaPlus[j]->GetYaxis()->SetRangeUser(0.,3.);
       histoSigmaVsEtaPlus[j]->GetXaxis()->SetRangeUser(-2.41,2.41);
       histoSigmaVsEtaPlus[j]->Draw();
     } else {
       histoSigmaVsEtaPlus[j]->Draw("SAME");
     }
     leg->AddEntry(histoSigmaVsEtaPlus[j],LegLabels[j],"PL");  
   }
 }
 //cmsprel->Draw("same");
 leg->Draw("same");
 c1s->SaveAs(outDir+"SigmaVsEtaPlus.png"); 


//  //----------------- CANVAS C2S --------------//
//  c2s->SetFillColor(0);  
//  c2s->cd();

//  leg = new TLegend(0.50,0.25,0.90,0.40);  
//  leg->SetBorderSize(1);
//  leg->SetFillColor(0);
//  leg->SetTextFont(42);

// // Sigma VS muon eta plus - eta minus  -------------------------------
//  TH1D *histoSigmaVsEtaPlusMinusDiff[nOfFiles];
//  for(Int_t j=0; j < nOfFiles; j++) {     
   
//    TFile *fin = (TFile*)FileList->At(j);    
//    if (( histoSigmaVsEtaPlusMinusDiff[j] = (TH1D*)fin->Get("MassVsEtaPlusMinusDiff/allHistos/sigmaHisto"))){
//      histoSigmaVsEtaPlusMinusDiff[j]->SetLineStyle(linestylelist_resol[j]);
//      histoSigmaVsEtaPlusMinusDiff[j]->SetMarkerColor(colorlist_resol[j]);
//      histoSigmaVsEtaPlusMinusDiff[j]->SetLineColor(colorlist_resol[j]);
//      histoSigmaVsEtaPlusMinusDiff[j]->SetMarkerStyle(markerstylelist_resol[j]); 
//      //     histoSigmaVsEtaPlusMinusDiff[j]->SetMarkerSize(0.75);
//      if ( j == 0 ) {
//        histoSigmaVsEtaPlusMinusDiff[j]->GetXaxis()->SetTitle("#eta pos. muon - #eta neg. muon");
//        histoSigmaVsEtaPlusMinusDiff[j]->GetYaxis()->SetTitle("#sigma(M_{#mu#mu}) (GeV)");
//        //       histoSigmaVsEtaPlusMinusDiff[j]->GetYaxis()->SetRangeUser(88.0,96.0);
//        histoSigmaVsEtaPlusMinusDiff[j]->GetYaxis()->SetRangeUser(0.,3.);
//        //histoSigmaVsEtaPlusMinusDiff[j]->GetYaxis()->SetRangeUser(90.60,90.75);
//        histoSigmaVsEtaPlusMinusDiff[j]->GetXaxis()->SetRangeUser(-3.2,3.2);
//        histoSigmaVsEtaPlusMinusDiff[j]->Draw();
//      } else {
//        histoSigmaVsEtaPlusMinusDiff[j]->Draw("SAME");
//      }
//      leg->AddEntry(histoSigmaVsEtaPlusMinusDiff[j],LegLabels[j],"PL");  
//    }
//  }
//  //cmsprel->Draw("same");
//  leg->Draw("same");
//  c2s->SaveAs(outDir+"SigmaVsEtaPlusMinusDiff.png"); 


//  //----------------- CANVAS C3S --------------//
//  c3s->SetFillColor(0);  
//  c3s->cd();

//  leg = new TLegend(0.35,0.15,0.55,0.35);  
//  leg->SetBorderSize(1);
//  leg->SetFillColor(0);
//  leg->SetTextFont(42);
 
// // Sigma VS muon pT  -------------------------------
//  TH1D *histoSigmaVsPt[nOfFiles];
//  for(Int_t j=0; j < nOfFiles; j++) {     
   
//    TFile *fin = (TFile*)FileList->At(j);    
//    if (( histoSigmaVsPt[j] = (TH1D*)fin->Get("MassVsPt/allHistos/sigmaHisto"))){
//      histoSigmaVsPt[j]->SetLineStyle(linestylelist_resol[j]);
//      histoSigmaVsPt[j]->SetMarkerColor(colorlist_resol[j]);
//      histoSigmaVsPt[j]->SetLineColor(colorlist_resol[j]);
//      histoSigmaVsPt[j]->SetMarkerStyle(markerstylelist_resol[j]); 
//      //     histoSigmaVsPt[j]->SetMarkerSize(0.75);
//      if ( j == 0 ) {
//        histoSigmaVsPt[j]->GetXaxis()->SetTitle("muon p_T (GeV)");
//        histoSigmaVsPt[j]->GetYaxis()->SetTitle("#sigma(M_{#mu#mu}) (GeV)");
//        //       histoSigmaVsPt[j]->GetYaxis()->SetRangeUser(88.0,96.0);
//        histoSigmaVsPt[j]->GetYaxis()->SetRangeUser(0.,3.);
//        //histoSigmaVsPt[j]->GetYaxis()->SetRangeUser(90.60,90.75);
//        histoSigmaVsPt[j]->GetXaxis()->SetRangeUser(15.,105.);
//        histoSigmaVsPt[j]->Draw();
//      } else {
//        histoSigmaVsPt[j]->Draw("SAME");
//      }
//      leg->AddEntry(histoSigmaVsPt[j],LegLabels[j],"PL");  
//    }
//  }
//  //cmsprel->Draw("same");
//  leg->Draw("same");
//  c3s->SaveAs(outDir+"SigmaVsPt.png"); 

 //----------------- CANVAS CFIT --------------//
 cFit->SetFillColor(0);  
 cFit->cd();
 Float_t nN = TMath::Sqrt(nOfFiles);
 Int_t nX = (Int_t)nN;
 if ( nN-nX > 0.5 ) nX++;
 Int_t nY = (Int_t)(nOfFiles/nX);
 std::cout << nX << " ," << nY << std::endl;
 cFit->Divide(nOfFiles,1);
 
// Mass VS muon phi plus -------------------------------
 TFile *ZFitFile = new TFile("ZFitFile.root","RECREATE");
 RooPlot *histoLineShape[nOfFiles];
 for(Int_t j=0; j < nOfFiles; j++) {     
   
   TFile *fin = (TFile*)FileList->At(j);    
   if (( histoLineShape[j] = (RooPlot*)fin->Get("hRecBestResAllEvents_Mass_frame"))){
     std::cout<<"Writing fit histogrem file n. "<<j<<std::endl;
     histoLineShape[j]->Write();
     cFit->cd(j+1);
     histoLineShape[j]->SetTitle(LegLabels[j]);
     histoLineShape[j]->Draw();
     histoLineShape[j]->GetXaxis()->SetTitle("M_{#mu#mu} (GeV)");
//      TPaveText *cmsprel2 = new TPaveText(0.19, 0.95, 0.95, 0.99, "NDC");
//      cmsprel2->SetTextSize(0.03);
//      cmsprel2->SetTextFont(42);
//      cmsprel2->SetFillColor(0);
//      cmsprel2->SetBorderSize(0);
//      cmsprel2->SetMargin(0.01);
//      cmsprel2->SetTextAlign(12); // align left
//      cmsprel2->AddText(0.666666, 0.5, LegLabels[j]);

   }
 }
 ZFitFile->Close();
 // cmsprel2->Draw("same");
 cFit->SaveAs("ZFitFile.root");


 
 return; 
};
