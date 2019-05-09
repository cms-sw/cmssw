#include <iostream>
#include "Riostream.h"
#include <fstream>
#include <iomanip>
#include <vector>
#include <string>
#include <TEfficiency.h>
#include "TLegend.h"

//#define Maxselection 1
//#define Nhltpaths 442

void make_up(){
  gROOT->Reset();
  //  gStyle->SetOptStat(1111);
  gStyle->SetOptStat(0);
  gStyle->SetPalette(1);
  gStyle->SetPaintTextFormat("2.2f");
  
  TFile * theFile = new TFile("../test/dt_phase2.root");

  //Phase-1

  cout<<"creating canvas"<<endl;
  TCanvas * Ca0 = new TCanvas("Ca0","Ca0",1200,600);
  Ca0->cd();
 
  //TDC
  {
      TH1F * allTDChisto;
      allTDChisto  = (TH1F*) (theFile->Get("allTDChisto"));
      allTDChisto->SetXTitle("DTDigi tdc counts ");
      allTDChisto->SetYTitle("counts");
      allTDChisto->SetTitle("DTDigi tdc counts phase-1");
      allTDChisto->Draw();
      allTDChisto->SetFillColor(kBlack);
      Ca0->SaveAs("allTDChisto.png");
      Ca0->Clear();Ca0->Clear();

      TH1F * selected_chamber_TDChisto;
      selected_chamber_TDChisto  = (TH1F*) (theFile->Get("selected_chamber_TDChisto"));
      selected_chamber_TDChisto->SetXTitle("DTDigi tdc counts ");
      selected_chamber_TDChisto->SetYTitle("counts");
      selected_chamber_TDChisto->SetTitle("DTDigi tdc counts phase-1 selected chamber");
      selected_chamber_TDChisto->Draw();
      selected_chamber_TDChisto->SetFillColor(kRed);
      Ca0->SaveAs("selected_chamber_TDChisto.png");
      Ca0->Clear();Ca0->Clear();


  }
  //TIME
  {
      TH1F * allTIMEhisto;
      allTIMEhisto  = (TH1F*) (theFile->Get("allTIMEhisto"));
      allTIMEhisto->SetXTitle("DTDigi time (ns)");
      allTIMEhisto->SetYTitle("counts/ns");
      allTIMEhisto->SetTitle("DTDigi time phase-1");
      allTIMEhisto->Draw();
      allTIMEhisto->SetFillColor(kBlack);
      Ca0->SaveAs("allTIMEhisto.png");
      Ca0->Clear();Ca0->Clear();

      TH1F * selected_chamber_TIMEhisto;
      selected_chamber_TIMEhisto  = (TH1F*) (theFile->Get("selected_chamber_TIMEhisto"));
      selected_chamber_TIMEhisto->SetXTitle("DTDigi time (ns)");
      selected_chamber_TIMEhisto->SetYTitle("counts/ns");
      selected_chamber_TIMEhisto->SetTitle("DTDigi time phase-1 selected chamber");
      selected_chamber_TIMEhisto->Draw();
      selected_chamber_TIMEhisto->SetFillColor(kRed);
      Ca0->SaveAs("selected_chamber_TIMEhisto.png");
      Ca0->Clear();Ca0->Clear();
  }  

  //T0                                                                                                                                                                    
  {
      TH1F * allT0histo;
      allT0histo  = (TH1F*) (theFile->Get("allT0histo"));
      allT0histo->SetXTitle("segment t0 time (ns)");
      allT0histo->SetYTitle("counts");
      allT0histo->SetTitle("segment t0 time phase-1");
      allT0histo->Draw();
      allT0histo->SetFillColor(kBlack);
      Ca0->SaveAs("allT0histo.png");
      Ca0->Clear();Ca0->Clear();   


      /*
      TH1F * selected_chamber_T0histo;
      selected_chamber_T0histo  = (TH1F*) (theFile->Get("selected_chamber_T0histo"));
      selected_chamber_T0histo->SetXTitle("segment t0 time (ns)");
      selected_chamber_T0histo->SetYTitle("counts");
      selected_chamber_T0histo->SetTitle("segment t0 time phase-1");
      selected_chamber_T0histo->Draw();
      selected_chamber_T0histo->SetFillColor(kBlack);
      Ca0->SaveAs("selected_chamber_T0histo.png");
      Ca0->Clear();Ca0->Clear();
      */
  }


  //Phase-2

  TCanvas * Ca1 = new TCanvas("Ca1","Ca1",1200,600);
  Ca1->cd();


  //TDC
  {
      TH1F * allTDCPhase2histo;
      allTDCPhase2histo  = (TH1F*) (theFile->Get("allTDCPhase2histo"));
      allTDCPhase2histo->SetXTitle("DTDigi tdc counts phase2 ");
      allTDCPhase2histo->SetYTitle("counts");
      allTDCPhase2histo->SetTitle("DTDigi tdc counts phase-2");
      allTDCPhase2histo->Draw();
      //allTDCPhase2histo->SetFillColor(kBlack);
      Ca1->SaveAs("allTDCPhase2histo.png");
      Ca1->Clear(); Ca1->Clear();

      TH1F * selected_chamber_TDCPhase2histo;
      selected_chamber_TDCPhase2histo  = (TH1F*) (theFile->Get("selected_chamber_TDCPhase2histo"));
      selected_chamber_TDCPhase2histo->SetXTitle("DTDigi tdc counts phase2");
      selected_chamber_TDCPhase2histo->SetYTitle("counts");
      selected_chamber_TDCPhase2histo->SetTitle("DTDigi tdc counts phase-2 selected chamber");
      selected_chamber_TDCPhase2histo->Draw();
      //selected_chamber_TDCPhase2histo->SetFillColor(kRed);
      Ca1->SaveAs("selected_chamber_TDCPhase2histo.png");
      Ca1->Clear(); Ca1->Clear();


  }
  
  //TIME
  { 
      TH1F * allTIMEPhase2histo;
      allTIMEPhase2histo  = (TH1F*) (theFile->Get("allTIMEPhase2histo"));
      allTIMEPhase2histo->SetXTitle("DTDigi time phase2 (ns)");
      allTIMEPhase2histo->SetYTitle("counts/ns");
      allTIMEPhase2histo->SetTitle("DTDigi time phase-2");
      allTIMEPhase2histo->Draw();
      //allTIMEPhase2histo->SetFillColor(kBlack);
      Ca1->SaveAs("allTIMEPhase2histo.png");
      Ca1->Clear(); Ca1->Clear();
      
      TH1F * selected_chamber_TIMEPhase2histo;
      selected_chamber_TIMEPhase2histo  = (TH1F*) (theFile->Get("selected_chamber_TIMEPhase2histo"));
      selected_chamber_TIMEPhase2histo->SetXTitle("DTDigi time phase2 (ns)");
      selected_chamber_TIMEPhase2histo->SetYTitle("counts/ns");
      selected_chamber_TIMEPhase2histo->SetTitle("DTDigi time phase-2 selected chamber");
      selected_chamber_TIMEPhase2histo->Draw();
      //selected_chamber_TIMEPhase2histo->SetFillColor(kRed);
      Ca1->SaveAs("selected_chamber_TIMEPhase2histo.png");
      Ca1->Clear(); Ca1->Clear();
      
  }

  
  //T0                                                                                                                                                                  
  {
      TH1F * allT0Phase2histo;
      allT0Phase2histo  = (TH1F*) (theFile->Get("allT0Phase2histo"));
      allT0Phase2histo->SetXTitle("segment t0 time (ns)");
      allT0Phase2histo->SetYTitle("counts");
      allT0Phase2histo->SetTitle("segment t0 time phase-2");
      allT0Phase2histo->Draw();
      allT0Phase2histo->SetFillColor(kBlack);
      Ca1->SaveAs("allT0Phase2histo.png");
      Ca1->Clear();Ca1->Clear();

      //problem here
      TH1F * selected_chamber_T0Phase2histo;
      selected_chamber_T0Phase2histo  = (TH1F*) (theFile->Get("selected_chamber_T0Phase2histo"));
      selected_chamber_T0Phase2histo->SetXTitle("segment t0 time (ns)");
      selected_chamber_T0Phase2histo->SetYTitle("counts");
      selected_chamber_T0Phase2histo->SetTitle("segment t0 time phase-2");
      selected_chamber_T0Phase2histo->Draw();
      selected_chamber_T0Phase2histo->SetFillColor(kBlack);
      Ca1->SaveAs("selected_chamber_T0Phase2histo.png");
      Ca1->Clear();Ca1->Clear();
  }
  

  //2D
  /*
  TH2F * wirevslayer;
  wirevslayer  = (TH2F*) (theFile->Get("wirevslayer"));
  wirevslayer->SetXTitle("wire");
  wirevslayer->SetYTitle("L + (SL-1)*2 or SL1:1-4 SL3:5-8");
  wirevslayer->SetTitle("occupancy phi-layers vs wire selected chamber");
  wirevslayer->Draw("colz");
  Ca1->SaveAs("wirevslayer.png");
  Ca1->Clear(); Ca1->Clear();

  TH2F * wirevslayerzTDC;
  wirevslayer  = (TH2F*) (theFile->Get("wirevslayerzTDC"));
  wirevslayer->SetXTitle("wire -0.5 + digiTDC/1600.");
  wirevslayer->SetYTitle("L + (SL-1)*2 or SL1:1-4 SL3:5-8");
  wirevslayer->SetTitle("occupancy phi-layers vs wire selected chamber");
  wirevslayer->Draw("colz");
  Ca1->SaveAs("wirevslayerzTDC.png");
  Ca1->Clear(); Ca1->Clear();
  */

  //4D segments
  TH1F * selected_chamber_segment_x;
  selected_chamber_segment_x  = (TH1F*) (theFile->Get("selected_chamber_segment_x"));
  selected_chamber_segment_x->SetXTitle("4D segment x position (cm)");
  selected_chamber_segment_x->SetYTitle("counts");
  selected_chamber_segment_x->SetTitle("4D segment x position (cm)");
  selected_chamber_segment_x->SetFillColor(kBlack);
  selected_chamber_segment_x->Draw();
  Ca1->SaveAs("selected_chamber_segment_x.png");
  Ca1->Clear(); Ca1->Clear();

  TH1F * selected_chamber_segment_tanPhi;
  selected_chamber_segment_tanPhi  = (TH1F*) (theFile->Get("selected_chamber_segment_tanPhi"));
  selected_chamber_segment_tanPhi->SetXTitle("4D segment tan(#phi)");
  selected_chamber_segment_tanPhi->SetYTitle("counts");
  selected_chamber_segment_tanPhi->SetTitle("4D segment tan(#phi) (radians)");
  selected_chamber_segment_tanPhi->SetFillColor(kBlack);
  selected_chamber_segment_tanPhi->Draw();
  Ca1->SaveAs("selected_chamber_segment_tanPhi.png");
  Ca1->Clear(); Ca1->Clear();

  //TH1F * selected_chamber_segment_BX;
  //selected_chamber_segment_BX  = (TH1F*) (theFile->Get("selected_chamber_segment_BX"));
  //selected_chamber_segment_BX->SetXTitle("4D segment BX");
  //selected_chamber_segment_BX->SetYTitle("counts");
  //selected_chamber_segment_BX->SetTitle("4D segment BX");
  //selected_chamber_segment_BX->SetFillColor(kBlack);
  //selected_chamber_segment_BX->Draw();
  //Ca1->SaveAs("selected_chamber_segment_BX.png");
  //Ca1->Clear(); Ca1->Clear();


  //2D correlation JM/4D Segment

  TH2F * selected_chamber_segment_vs_jm_x;
  selected_chamber_segment_vs_jm_x  = (TH2F*) (theFile->Get("selected_chamber_segment_vs_jm_x"));
  selected_chamber_segment_vs_jm_x->SetXTitle("segment x position (cm)");
  selected_chamber_segment_vs_jm_x->SetYTitle("phase-2 L1 primitive x position (cm)");
  selected_chamber_segment_vs_jm_x->SetTitle("4D segment vs phase-2 L1 primitive x position (cm)");
  selected_chamber_segment_vs_jm_x->Draw("colz");
  Ca1->SaveAs("selected_chamber_segment_vs_jm_x.png");
  Ca1->Clear(); Ca1->Clear();

  
  TH2F * selected_chamber_segment_vs_jm_T0histo;
  selected_chamber_segment_vs_jm_T0histo  = (TH2F*) (theFile->Get("selected_chamber_segment_vs_jm_T0histo"));
  selected_chamber_segment_vs_jm_T0histo->SetXTitle("segment phase-2 t0");
  selected_chamber_segment_vs_jm_T0histo->SetYTitle("phase-2 L1 primitive t0");
  selected_chamber_segment_vs_jm_T0histo->SetTitle("4D segment vs phase-2 L1 primitive t0 (ns)");
  selected_chamber_segment_vs_jm_T0histo->Draw("colz");
  Ca1->SaveAs("selected_chamber_segment_vs_jm_T0histo.png");
  Ca1->Clear(); Ca1->Clear();
  
  
  TH2F * selected_chamber_segment_vs_jm_tanPhi;
  selected_chamber_segment_vs_jm_tanPhi  = (TH2F*) (theFile->Get("selected_chamber_segment_vs_jm_tanPhi"));
  selected_chamber_segment_vs_jm_tanPhi->SetXTitle("segment tan(#phi)");
  selected_chamber_segment_vs_jm_tanPhi->SetYTitle("phase-2 L1 primitive algo tan(#phi)");
  selected_chamber_segment_vs_jm_tanPhi->SetTitle("4D segment vs phase-2 L1 primitive tan(#phi)");
  selected_chamber_segment_vs_jm_tanPhi->Draw("colz");
  Ca1->SaveAs("selected_chamber_segment_vs_jm_tanPhi.png");
  Ca1->Clear(); Ca1->Clear();

  
  exit(0);

}
     
