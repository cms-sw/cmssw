// How to run: ./RemoteMonitoringIMPSM.cc.exe root_file1  root_file_ref Global
//
//
//
//

#include "LogEleMapdb.h"

#include <iostream>
#include <fstream>

#include "TH1.h"
#include "TH2.h"
#include "TCanvas.h"
#include "TROOT.h"
#include <TMath.h>
#include "TStyle.h"
#include "TSystem.h"
#include "TLegend.h"
#include "TText.h"
#include "TAxis.h"
#include "TFile.h"
#include "TLine.h"
#include "TGraph.h"

using namespace std;
int main(int argc, char* argv[]) {
  std::string dirnm = "Analyzer";
  gROOT->Reset();
  gROOT->SetStyle("Plain");
  gStyle->SetOptStat(0);
  gStyle->SetOptTitle(1);
  // ok change
  if (argc < 3)
    return 1;
  char fname[300];
  char refname[300];
  char runtypeC[300];
  sprintf(fname, "%s", argv[1]);
  sprintf(refname, "%s", argv[2]);
  sprintf(runtypeC, "%s", argv[3]);
  //               std::cout<<fname<<" "<<refname<<" "<<runtypeC<<std::endl;
  std::cout << " We are here to print fname refname runtypeC " << fname << " " << refname << " " << runtypeC
            << std::endl;
  // ok change

  //======================================================================
  // Connect the input files, parameters and get the 2-d histogram in memory
  string promt = (string)fname;
  string runtype = (string)runtypeC;
  string runnumber = "";
  for (unsigned int i = promt.size() - 11; i < promt.size() - 5; i++)
    runnumber += fname[i];

  TFile* hfile = new TFile(fname, "READ");
  hfile->ls();
  TDirectory* dir = (TDirectory*)hfile->FindObjectAny(dirnm.c_str());

  // with TfileService implementation, change everywhere below:     hfile->Get     to     dir->FindObjectAny

  //======================================================================
  // Prepare histograms and plot them to .png files

  // Phi-symmetry for Calibration Group:

  TCanvas* c1x0 = new TCanvas("c1x0", "c1x0", 300, 10, 800, 700);

  TCanvas* c1x1 = new TCanvas("c1x1", "c1x1", 100, 10, 600, 700);

  TCanvas* c2x1 = new TCanvas("c2x1", "c2x1", 200, 300, 1600, 800);

  TCanvas* c3x5 = new TCanvas("c3x5", "c3x5", 1000, 1500);
  //

  char* str = (char*)alloca(10000);

  // before upgrade 2017:
  // depth: HB depth1,2; HE depth1,2,3; HO depth4; HF depth1,2
  // 5 depthes:  0(empty),   1,2,3,4

  // upgrade 2017:
  // depth: HB depth1,2; HE depth1,2,3,4,5,6,7; HO depth4; HF depth1,2,3,4
  // 8 depthes:  0(empty),   1,2,3,4,5,6,7

  // upgrade 2021:
  // depth: HB depth1,2,3,4; HE depth1,2,3,4,5,6,7; HO depth4; HF depth1,2,3,4
  // 10 depthes:  0(empty),   1,2,3,4,5,6,7,8,9

  //  Int_t ALLDEPTH = 5;
  //  Int_t ALLDEPTH = 8;
  Int_t ALLDEPTH = 10;

  int k_min[5] = {0, 1, 1, 4, 1};  // minimum depth for each subdet
  //     int k_max[5]={0,2,3,4,2}; // maximum depth for each subdet
  //       int k_max[5]={0,2,7,4,4}; // maximum depth for each subdet
  int k_max[5] = {0, 4, 7, 4, 4};  // maximum depth for each subdet
  ///////////////////////////////////////////////////////////////////////////////////////////////////////////////

  const int nsub = 4;
  const int neta = 82;
  const int nphi = 72;
  int njeta = neta;
  int njphi = nphi;
  //const int ndepth = 7;
  int ndepth;
  /////////////   ///////////// //////////////////////////  /////////////  /////////////  /////////////  /////////////  /////////////  ////////////////////         Phi-symmetry for Calibration Group:
  /////////////  /////////////  //////////////////////////  /////////////  /////////////  /////////////  /////////////  ////////////////////         Phi-symmetry for Calibration Group:
  /////////////  /////////////  /////////////  /////////////  /////////////  /////////////  /////////////  ////////////////////         Phi-symmetry for Calibration Group:

  ////////////////////// Start   Recosignal  Start Recosignal  Start   Recosignal  Start   Recosignal  Start   Recosignal Start  Recosignal Start Recosignal Start Recosignal Start Recosignal Start Recosignal Start
  ////////////////////// Start   Recosignal  Start Recosignal  Start   Recosignal  Start   Recosignal  Start   Recosignal Start  Recosignal Start Recosignal Start Recosignal Start Recosignal Start Recosignal Start
  ////////////////////// Start   Recosignal  Start Recosignal  Start   Recosignal  Start   Recosignal  Start   Recosignal Start  Recosignal Start Recosignal Start Recosignal Start Recosignal Start Recosignal Start
  ////////////////////////////////////////////////////////////////////////////////////////////////////     Recosignal HB
  ////////////////////////////////////////////////////////////////////////////////////////////////////     Recosignal HB
  ////////////////////////////////////////////////////////////////////////////////////////////////////     Recosignal HB
  //  int k_max[5]={0,4,7,4,4}; // maximum depth for each subdet
  //ndepth = k_max[5];
  ndepth = 4;
  double arecosignalHB[ndepth][njeta][njphi];
  double recosignalvarianceHB[ndepth][njeta][njphi];
  //                                   RRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRR:   Recosignal HB  recSignalEnergy
  TH2F* recSignalEnergy1HB1 = (TH2F*)dir->FindObjectAny("h_recSignalEnergy1_HB1");
  TH2F* recSignalEnergy0HB1 = (TH2F*)dir->FindObjectAny("h_recSignalEnergy0_HB1");
  TH2F* recSignalEnergyHB1 = (TH2F*)recSignalEnergy1HB1->Clone("recSignalEnergyHB1");
  recSignalEnergyHB1->Divide(recSignalEnergy1HB1, recSignalEnergy0HB1, 1, 1, "B");
  TH2F* recSignalEnergy1HB2 = (TH2F*)dir->FindObjectAny("h_recSignalEnergy1_HB2");
  TH2F* recSignalEnergy0HB2 = (TH2F*)dir->FindObjectAny("h_recSignalEnergy0_HB2");
  TH2F* recSignalEnergyHB2 = (TH2F*)recSignalEnergy1HB2->Clone("recSignalEnergyHB2");
  recSignalEnergyHB2->Divide(recSignalEnergy1HB2, recSignalEnergy0HB2, 1, 1, "B");
  TH2F* recSignalEnergy1HB3 = (TH2F*)dir->FindObjectAny("h_recSignalEnergy1_HB3");
  TH2F* recSignalEnergy0HB3 = (TH2F*)dir->FindObjectAny("h_recSignalEnergy0_HB3");
  TH2F* recSignalEnergyHB3 = (TH2F*)recSignalEnergy1HB3->Clone("recSignalEnergyHB3");
  recSignalEnergyHB3->Divide(recSignalEnergy1HB3, recSignalEnergy0HB3, 1, 1, "B");
  TH2F* recSignalEnergy1HB4 = (TH2F*)dir->FindObjectAny("h_recSignalEnergy1_HB4");
  TH2F* recSignalEnergy0HB4 = (TH2F*)dir->FindObjectAny("h_recSignalEnergy0_HB4");
  TH2F* recSignalEnergyHB4 = (TH2F*)recSignalEnergy1HB4->Clone("recSignalEnergyHB4");
  recSignalEnergyHB4->Divide(recSignalEnergy1HB4, recSignalEnergy0HB4, 1, 1, "B");
  for (int jeta = 0; jeta < njeta; jeta++) {
    //====================================================================== PHI normalization & put R into massive arecosignalHB
    //preparation for PHI normalization:
    double sumrecosignalHB0 = 0;
    int nsumrecosignalHB0 = 0;
    double sumrecosignalHB1 = 0;
    int nsumrecosignalHB1 = 0;
    double sumrecosignalHB2 = 0;
    int nsumrecosignalHB2 = 0;
    double sumrecosignalHB3 = 0;
    int nsumrecosignalHB3 = 0;
    for (int jphi = 0; jphi < njphi; jphi++) {
      arecosignalHB[0][jeta][jphi] = recSignalEnergyHB1->GetBinContent(jeta + 1, jphi + 1);
      arecosignalHB[1][jeta][jphi] = recSignalEnergyHB2->GetBinContent(jeta + 1, jphi + 1);
      arecosignalHB[2][jeta][jphi] = recSignalEnergyHB3->GetBinContent(jeta + 1, jphi + 1);
      arecosignalHB[3][jeta][jphi] = recSignalEnergyHB4->GetBinContent(jeta + 1, jphi + 1);
      if (arecosignalHB[0][jeta][jphi] > 0.) {
        sumrecosignalHB0 += arecosignalHB[0][jeta][jphi];
        ++nsumrecosignalHB0;
      }
      if (arecosignalHB[1][jeta][jphi] > 0.) {
        sumrecosignalHB1 += arecosignalHB[1][jeta][jphi];
        ++nsumrecosignalHB1;
      }
      if (arecosignalHB[2][jeta][jphi] > 0.) {
        sumrecosignalHB2 += arecosignalHB[2][jeta][jphi];
        ++nsumrecosignalHB2;
      }
      if (arecosignalHB[3][jeta][jphi] > 0.) {
        sumrecosignalHB3 += arecosignalHB[3][jeta][jphi];
        ++nsumrecosignalHB3;
      }
    }  // phi
    // PHI normalization:
    for (int jphi = 0; jphi < njphi; jphi++) {
      if (arecosignalHB[0][jeta][jphi] > 0.)
        arecosignalHB[0][jeta][jphi] /= (sumrecosignalHB0 / nsumrecosignalHB0);
      if (arecosignalHB[1][jeta][jphi] > 0.)
        arecosignalHB[1][jeta][jphi] /= (sumrecosignalHB1 / nsumrecosignalHB1);
      if (arecosignalHB[2][jeta][jphi] > 0.)
        arecosignalHB[2][jeta][jphi] /= (sumrecosignalHB2 / nsumrecosignalHB2);
      if (arecosignalHB[3][jeta][jphi] > 0.)
        arecosignalHB[3][jeta][jphi] /= (sumrecosignalHB3 / nsumrecosignalHB3);
    }  // phi
  }    //eta
  //------------------------  2D-eta/phi-plot: R, averaged over depthfs
  //======================================================================
  //======================================================================
  //cout<<"      R2D-eta/phi-plot: R, averaged over depthfs *****" <<endl;
  c2x1->Clear();
  /////////////////
  c2x1->Divide(2, 1);
  c2x1->cd(1);
  TH2F* GefzRrecosignalHB42D = new TH2F("GefzRrecosignalHB42D", "", neta, -41., 41., nphi, 0., 72.);
  TH2F* GefzRrecosignalHB42D0 = new TH2F("GefzRrecosignalHB42D0", "", neta, -41., 41., nphi, 0., 72.);
  TH2F* GefzRrecosignalHB42DF = (TH2F*)GefzRrecosignalHB42D0->Clone("GefzRrecosignalHB42DF");
  for (int i = 0; i < ndepth; i++) {
    for (int jeta = 0; jeta < neta; jeta++) {
      for (int jphi = 0; jphi < nphi; jphi++) {
        double ccc1 = arecosignalHB[i][jeta][jphi];
        int k2plot = jeta - 41;
        int kkk = k2plot;  //if(k2plot >0 ) kkk=k2plot+1; //-41 +41 !=0
        if (ccc1 != 0.) {
          GefzRrecosignalHB42D->Fill(kkk, jphi, ccc1);
          GefzRrecosignalHB42D0->Fill(kkk, jphi, 1.);
        }
      }
    }
  }
  GefzRrecosignalHB42DF->Divide(GefzRrecosignalHB42D, GefzRrecosignalHB42D0, 1, 1, "B");  // average A
  gPad->SetGridy();
  gPad->SetGridx();  //      gPad->SetLogz();
  GefzRrecosignalHB42DF->SetXTitle("<R>_depth       #eta  \b");
  GefzRrecosignalHB42DF->SetYTitle("      #phi \b");
  GefzRrecosignalHB42DF->Draw("COLZ");

  c2x1->cd(2);
  TH1F* energyhitSignal_HB = (TH1F*)dir->FindObjectAny("h_energyhitSignal_HB");
  energyhitSignal_HB->SetMarkerStyle(20);
  energyhitSignal_HB->SetMarkerSize(0.4);
  energyhitSignal_HB->GetYaxis()->SetLabelSize(0.04);
  energyhitSignal_HB->SetXTitle("energyhitSignal_HB \b");
  energyhitSignal_HB->SetMarkerColor(2);
  energyhitSignal_HB->SetLineColor(0);
  gPad->SetGridy();
  gPad->SetGridx();
  energyhitSignal_HB->Draw("Error");

  /////////////////
  c2x1->Update();
  c2x1->Print("RrecosignalGeneralD2PhiSymmetryHB.png");
  c2x1->Clear();
  // clean-up
  if (GefzRrecosignalHB42D)
    delete GefzRrecosignalHB42D;
  if (GefzRrecosignalHB42D0)
    delete GefzRrecosignalHB42D0;
  if (GefzRrecosignalHB42DF)
    delete GefzRrecosignalHB42DF;
  //====================================================================== 1D plot: R vs phi , averaged over depthfs & eta
  //======================================================================
  //cout<<"      1D plot: R vs phi , averaged over depthfs & eta *****" <<endl;
  c1x1->Clear();
  /////////////////
  c1x1->Divide(1, 1);
  c1x1->cd(1);
  TH1F* GefzRrecosignalHB41D = new TH1F("GefzRrecosignalHB41D", "", nphi, 0., 72.);
  TH1F* GefzRrecosignalHB41D0 = new TH1F("GefzRrecosignalHB41D0", "", nphi, 0., 72.);
  TH1F* GefzRrecosignalHB41DF = (TH1F*)GefzRrecosignalHB41D0->Clone("GefzRrecosignalHB41DF");
  for (int jphi = 0; jphi < nphi; jphi++) {
    for (int jeta = 0; jeta < neta; jeta++) {
      for (int i = 0; i < ndepth; i++) {
        double ccc1 = arecosignalHB[i][jeta][jphi];
        if (ccc1 != 0.) {
          GefzRrecosignalHB41D->Fill(jphi, ccc1);
          GefzRrecosignalHB41D0->Fill(jphi, 1.);
        }
      }
    }
  }
  GefzRrecosignalHB41DF->Divide(
      GefzRrecosignalHB41D, GefzRrecosignalHB41D0, 1, 1, "B");  // R averaged over depthfs & eta
  GefzRrecosignalHB41D0->Sumw2();
  //    for (int jphi=1;jphi<73;jphi++) {GefzRrecosignalHB41DF->SetBinError(jphi,0.01);}
  gPad->SetGridy();
  gPad->SetGridx();  //      gPad->SetLogz();
  GefzRrecosignalHB41DF->SetMarkerStyle(20);
  GefzRrecosignalHB41DF->SetMarkerSize(1.4);
  GefzRrecosignalHB41DF->GetZaxis()->SetLabelSize(0.08);
  GefzRrecosignalHB41DF->SetXTitle("#phi  \b");
  GefzRrecosignalHB41DF->SetYTitle("  <R> \b");
  GefzRrecosignalHB41DF->SetZTitle("<R>_PHI  - AllDepthfs \b");
  GefzRrecosignalHB41DF->SetMarkerColor(4);
  GefzRrecosignalHB41DF->SetLineColor(
      4);  //  GefzRrecosignalHB41DF->SetMinimum(0.8);     //      GefzRrecosignalHB41DF->SetMaximum(1.000);
  GefzRrecosignalHB41DF->Draw("Error");
  /////////////////
  c1x1->Update();
  c1x1->Print("RrecosignalGeneralD1PhiSymmetryHB.png");
  c1x1->Clear();
  // clean-up
  if (GefzRrecosignalHB41D)
    delete GefzRrecosignalHB41D;
  if (GefzRrecosignalHB41D0)
    delete GefzRrecosignalHB41D0;
  if (GefzRrecosignalHB41DF)
    delete GefzRrecosignalHB41DF;
  //========================================================================================== 4
  //======================================================================
  //======================================================================1D plot: R vs phi , different eta,  depth=1
  //cout<<"      1D plot: R vs phi , different eta,  depth=1 *****" <<endl;
  c3x5->Clear();
  /////////////////
  c3x5->Divide(4, 4);
  c3x5->cd(1);
  int kcountHBpositivedirectionRecosignal1 = 1;
  TH1F* h2CeffHBpositivedirectionRecosignal1 = new TH1F("h2CeffHBpositivedirectionRecosignal1", "", nphi, 0., 72.);
  for (int jeta = 0; jeta < njeta; jeta++) {
    // positivedirectionRecosignal:
    if (jeta - 41 >= 0) {
      //	     for (int i=0;i<ndepth;i++) {
      // depth=1
      for (int i = 0; i < 1; i++) {
        TH1F* HBpositivedirectionRecosignal1 = (TH1F*)h2CeffHBpositivedirectionRecosignal1->Clone("twod1");
        float ccctest = 0;  // to avoid empty massive elements
        for (int jphi = 0; jphi < nphi; jphi++) {
          double ccc1 = arecosignalHB[i][jeta][jphi];
          if (ccc1 != 0.) {
            HBpositivedirectionRecosignal1->Fill(jphi, ccc1);
            ccctest = 1.;  //HBpositivedirectionRecosignal1->SetBinError(i,0.01);
          }
        }  // for jphi
        if (ccctest > 0.) {
          //	  cout<<"444        kcountHBpositivedirectionRecosignal1   =     "<<kcountHBpositivedirectionRecosignal1  <<"   jeta-41=     "<< jeta-41 <<endl;
          c3x5->cd(kcountHBpositivedirectionRecosignal1);
          HBpositivedirectionRecosignal1->SetMarkerStyle(20);
          HBpositivedirectionRecosignal1->SetMarkerSize(0.4);
          HBpositivedirectionRecosignal1->GetYaxis()->SetLabelSize(0.04);
          HBpositivedirectionRecosignal1->SetXTitle("HBpositivedirectionRecosignal1 \b");
          HBpositivedirectionRecosignal1->SetMarkerColor(2);
          HBpositivedirectionRecosignal1->SetLineColor(0);
          gPad->SetGridy();
          gPad->SetGridx();
          //	   gPad->SetLogy();
          if (kcountHBpositivedirectionRecosignal1 == 1)
            HBpositivedirectionRecosignal1->SetXTitle("R for HB+ jeta =  0; depth = 1 \b");
          if (kcountHBpositivedirectionRecosignal1 == 2)
            HBpositivedirectionRecosignal1->SetXTitle("R for HB+ jeta =  1; depth = 1 \b");
          if (kcountHBpositivedirectionRecosignal1 == 3)
            HBpositivedirectionRecosignal1->SetXTitle("R for HB+ jeta =  2; depth = 1 \b");
          if (kcountHBpositivedirectionRecosignal1 == 4)
            HBpositivedirectionRecosignal1->SetXTitle("R for HB+ jeta =  3; depth = 1 \b");
          if (kcountHBpositivedirectionRecosignal1 == 5)
            HBpositivedirectionRecosignal1->SetXTitle("R for HB+ jeta =  4; depth = 1 \b");
          if (kcountHBpositivedirectionRecosignal1 == 6)
            HBpositivedirectionRecosignal1->SetXTitle("R for HB+ jeta =  5; depth = 1 \b");
          if (kcountHBpositivedirectionRecosignal1 == 7)
            HBpositivedirectionRecosignal1->SetXTitle("R for HB+ jeta =  6; depth = 1 \b");
          if (kcountHBpositivedirectionRecosignal1 == 8)
            HBpositivedirectionRecosignal1->SetXTitle("R for HB+ jeta =  7; depth = 1 \b");
          if (kcountHBpositivedirectionRecosignal1 == 9)
            HBpositivedirectionRecosignal1->SetXTitle("R for HB+ jeta =  8; depth = 1 \b");
          if (kcountHBpositivedirectionRecosignal1 == 10)
            HBpositivedirectionRecosignal1->SetXTitle("R for HB+ jeta =  9; depth = 1 \b");
          if (kcountHBpositivedirectionRecosignal1 == 11)
            HBpositivedirectionRecosignal1->SetXTitle("R for HB+ jeta = 10; depth = 1 \b");
          if (kcountHBpositivedirectionRecosignal1 == 12)
            HBpositivedirectionRecosignal1->SetXTitle("R for HB+ jeta = 11; depth = 1 \b");
          if (kcountHBpositivedirectionRecosignal1 == 13)
            HBpositivedirectionRecosignal1->SetXTitle("R for HB+ jeta = 12; depth = 1 \b");
          if (kcountHBpositivedirectionRecosignal1 == 14)
            HBpositivedirectionRecosignal1->SetXTitle("R for HB+ jeta = 13; depth = 1 \b");
          if (kcountHBpositivedirectionRecosignal1 == 15)
            HBpositivedirectionRecosignal1->SetXTitle("R for HB+ jeta = 14; depth = 1 \b");
          if (kcountHBpositivedirectionRecosignal1 == 16)
            HBpositivedirectionRecosignal1->SetXTitle("R for HB+ jeta = 15; depth = 1 \b");
          HBpositivedirectionRecosignal1->Draw("Error");
          kcountHBpositivedirectionRecosignal1++;
          if (kcountHBpositivedirectionRecosignal1 > 16)
            break;  //
        }           //ccctest>0

      }  // for i
    }    //if(jeta-41 >= 0)
  }      //for jeta
  /////////////////
  c3x5->Update();
  c3x5->Print("RrecosignalPositiveDirectionhistD1PhiSymmetryDepth1HB.png");
  c3x5->Clear();
  // clean-up
  if (h2CeffHBpositivedirectionRecosignal1)
    delete h2CeffHBpositivedirectionRecosignal1;

  //========================================================================================== 5
  //======================================================================
  //======================================================================1D plot: R vs phi , different eta,  depth=2
  //  cout<<"      1D plot: R vs phi , different eta,  depth=2 *****" <<endl;
  c3x5->Clear();
  /////////////////
  c3x5->Divide(4, 4);
  c3x5->cd(1);
  int kcountHBpositivedirectionRecosignal2 = 1;
  TH1F* h2CeffHBpositivedirectionRecosignal2 = new TH1F("h2CeffHBpositivedirectionRecosignal2", "", nphi, 0., 72.);
  for (int jeta = 0; jeta < njeta; jeta++) {
    // positivedirectionRecosignal:
    if (jeta - 41 >= 0) {
      //	     for (int i=0;i<ndepth;i++) {
      // depth=2
      for (int i = 1; i < 2; i++) {
        TH1F* HBpositivedirectionRecosignal2 = (TH1F*)h2CeffHBpositivedirectionRecosignal2->Clone("twod1");
        float ccctest = 0;  // to avoid empty massive elements
        for (int jphi = 0; jphi < nphi; jphi++) {
          double ccc1 = arecosignalHB[i][jeta][jphi];
          if (ccc1 != 0.) {
            HBpositivedirectionRecosignal2->Fill(jphi, ccc1);
            ccctest = 1.;  //HBpositivedirectionRecosignal2->SetBinError(i,0.01);
          }
        }  // for jphi
        if (ccctest > 0.) {
          //cout<<"555        kcountHBpositivedirectionRecosignal2   =     "<<kcountHBpositivedirectionRecosignal2  <<"   jeta-41=     "<< jeta-41 <<endl;
          c3x5->cd(kcountHBpositivedirectionRecosignal2);
          HBpositivedirectionRecosignal2->SetMarkerStyle(20);
          HBpositivedirectionRecosignal2->SetMarkerSize(0.4);
          HBpositivedirectionRecosignal2->GetYaxis()->SetLabelSize(0.04);
          HBpositivedirectionRecosignal2->SetXTitle("HBpositivedirectionRecosignal2 \b");
          HBpositivedirectionRecosignal2->SetMarkerColor(2);
          HBpositivedirectionRecosignal2->SetLineColor(0);
          gPad->SetGridy();
          gPad->SetGridx();
          //	   gPad->SetLogy();
          if (kcountHBpositivedirectionRecosignal2 == 1)
            HBpositivedirectionRecosignal2->SetXTitle("R for HB+ jeta =  0; depth = 2 \b");
          if (kcountHBpositivedirectionRecosignal2 == 2)
            HBpositivedirectionRecosignal2->SetXTitle("R for HB+ jeta =  1; depth = 2 \b");
          if (kcountHBpositivedirectionRecosignal2 == 3)
            HBpositivedirectionRecosignal2->SetXTitle("R for HB+ jeta =  2; depth = 2 \b");
          if (kcountHBpositivedirectionRecosignal2 == 4)
            HBpositivedirectionRecosignal2->SetXTitle("R for HB+ jeta =  3; depth = 2 \b");
          if (kcountHBpositivedirectionRecosignal2 == 5)
            HBpositivedirectionRecosignal2->SetXTitle("R for HB+ jeta =  4; depth = 2 \b");
          if (kcountHBpositivedirectionRecosignal2 == 6)
            HBpositivedirectionRecosignal2->SetXTitle("R for HB+ jeta =  5; depth = 2 \b");
          if (kcountHBpositivedirectionRecosignal2 == 7)
            HBpositivedirectionRecosignal2->SetXTitle("R for HB+ jeta =  6; depth = 2 \b");
          if (kcountHBpositivedirectionRecosignal2 == 8)
            HBpositivedirectionRecosignal2->SetXTitle("R for HB+ jeta =  7; depth = 2 \b");
          if (kcountHBpositivedirectionRecosignal2 == 9)
            HBpositivedirectionRecosignal2->SetXTitle("R for HB+ jeta =  8; depth = 2 \b");
          if (kcountHBpositivedirectionRecosignal2 == 10)
            HBpositivedirectionRecosignal2->SetXTitle("R for HB+ jeta =  9; depth = 2 \b");
          if (kcountHBpositivedirectionRecosignal2 == 11)
            HBpositivedirectionRecosignal2->SetXTitle("R for HB+ jeta = 10; depth = 2 \b");
          if (kcountHBpositivedirectionRecosignal2 == 12)
            HBpositivedirectionRecosignal2->SetXTitle("R for HB+ jeta = 11; depth = 2 \b");
          if (kcountHBpositivedirectionRecosignal2 == 13)
            HBpositivedirectionRecosignal2->SetXTitle("R for HB+ jeta = 12; depth = 2 \b");
          if (kcountHBpositivedirectionRecosignal2 == 14)
            HBpositivedirectionRecosignal2->SetXTitle("R for HB+ jeta = 13; depth = 2 \b");
          if (kcountHBpositivedirectionRecosignal2 == 15)
            HBpositivedirectionRecosignal2->SetXTitle("R for HB+ jeta = 14; depth = 2 \b");
          if (kcountHBpositivedirectionRecosignal2 == 16)
            HBpositivedirectionRecosignal2->SetXTitle("R for HB+ jeta = 15; depth = 2 \b");
          HBpositivedirectionRecosignal2->Draw("Error");
          kcountHBpositivedirectionRecosignal2++;
          if (kcountHBpositivedirectionRecosignal2 > 16)
            break;  // 4x6 = 24
        }           //ccctest>0

      }  // for i
    }    //if(jeta-41 >= 0)
  }      //for jeta
  /////////////////
  c3x5->Update();
  c3x5->Print("RrecosignalPositiveDirectionhistD1PhiSymmetryDepth2HB.png");
  c3x5->Clear();
  // clean-up
  if (h2CeffHBpositivedirectionRecosignal2)
    delete h2CeffHBpositivedirectionRecosignal2;
  //========================================================================================== 6
  //======================================================================
  //======================================================================1D plot: R vs phi , different eta,  depth=3
  //cout<<"      1D plot: R vs phi , different eta,  depth=3 *****" <<endl;
  c3x5->Clear();
  /////////////////
  c3x5->Divide(4, 4);
  c3x5->cd(1);
  int kcountHBpositivedirectionRecosignal3 = 1;
  TH1F* h2CeffHBpositivedirectionRecosignal3 = new TH1F("h2CeffHBpositivedirectionRecosignal3", "", nphi, 0., 72.);
  for (int jeta = 0; jeta < njeta; jeta++) {
    // positivedirectionRecosignal:
    if (jeta - 41 >= 0) {
      //	     for (int i=0;i<ndepth;i++) {
      // depth=3
      for (int i = 2; i < 3; i++) {
        TH1F* HBpositivedirectionRecosignal3 = (TH1F*)h2CeffHBpositivedirectionRecosignal3->Clone("twod1");
        float ccctest = 0;  // to avoid empty massive elements
        for (int jphi = 0; jphi < nphi; jphi++) {
          double ccc1 = arecosignalHB[i][jeta][jphi];
          if (ccc1 != 0.) {
            HBpositivedirectionRecosignal3->Fill(jphi, ccc1);
            ccctest = 1.;  //HBpositivedirectionRecosignal3->SetBinError(i,0.01);
          }
        }  // for jphi
        if (ccctest > 0.) {
          //cout<<"666        kcountHBpositivedirectionRecosignal3   =     "<<kcountHBpositivedirectionRecosignal3  <<"   jeta-41=     "<< jeta-41 <<endl;
          c3x5->cd(kcountHBpositivedirectionRecosignal3);
          HBpositivedirectionRecosignal3->SetMarkerStyle(20);
          HBpositivedirectionRecosignal3->SetMarkerSize(0.4);
          HBpositivedirectionRecosignal3->GetYaxis()->SetLabelSize(0.04);
          HBpositivedirectionRecosignal3->SetXTitle("HBpositivedirectionRecosignal3 \b");
          HBpositivedirectionRecosignal3->SetMarkerColor(2);
          HBpositivedirectionRecosignal3->SetLineColor(0);
          gPad->SetGridy();
          gPad->SetGridx();
          //	   gPad->SetLogy();
          if (kcountHBpositivedirectionRecosignal3 == 1)
            HBpositivedirectionRecosignal3->SetXTitle("R for HB+ jeta =  0; depth = 3 \b");
          if (kcountHBpositivedirectionRecosignal3 == 2)
            HBpositivedirectionRecosignal3->SetXTitle("R for HB+ jeta =  1; depth = 3 \b");
          if (kcountHBpositivedirectionRecosignal3 == 3)
            HBpositivedirectionRecosignal3->SetXTitle("R for HB+ jeta =  2; depth = 3 \b");
          if (kcountHBpositivedirectionRecosignal3 == 4)
            HBpositivedirectionRecosignal3->SetXTitle("R for HB+ jeta =  3; depth = 3 \b");
          if (kcountHBpositivedirectionRecosignal3 == 5)
            HBpositivedirectionRecosignal3->SetXTitle("R for HB+ jeta =  4; depth = 3 \b");
          if (kcountHBpositivedirectionRecosignal3 == 6)
            HBpositivedirectionRecosignal3->SetXTitle("R for HB+ jeta =  5; depth = 3 \b");
          if (kcountHBpositivedirectionRecosignal3 == 7)
            HBpositivedirectionRecosignal3->SetXTitle("R for HB+ jeta =  6; depth = 3 \b");
          if (kcountHBpositivedirectionRecosignal3 == 8)
            HBpositivedirectionRecosignal3->SetXTitle("R for HB+ jeta =  7; depth = 3 \b");
          if (kcountHBpositivedirectionRecosignal3 == 9)
            HBpositivedirectionRecosignal3->SetXTitle("R for HB+ jeta =  8; depth = 3 \b");
          if (kcountHBpositivedirectionRecosignal3 == 10)
            HBpositivedirectionRecosignal3->SetXTitle("R for HB+ jeta =  9; depth = 3 \b");
          if (kcountHBpositivedirectionRecosignal3 == 11)
            HBpositivedirectionRecosignal3->SetXTitle("R for HB+ jeta =  0; depth = 3 \b");
          if (kcountHBpositivedirectionRecosignal3 == 12)
            HBpositivedirectionRecosignal3->SetXTitle("R for HB+ jeta = 11; depth = 3 \b");
          if (kcountHBpositivedirectionRecosignal3 == 13)
            HBpositivedirectionRecosignal3->SetXTitle("R for HB+ jeta = 12; depth = 3 \b");
          if (kcountHBpositivedirectionRecosignal3 == 14)
            HBpositivedirectionRecosignal3->SetXTitle("R for HB+ jeta = 13; depth = 3 \b");
          if (kcountHBpositivedirectionRecosignal3 == 15)
            HBpositivedirectionRecosignal3->SetXTitle("R for HB+ jeta = 14; depth = 3 \b");
          if (kcountHBpositivedirectionRecosignal3 == 16)
            HBpositivedirectionRecosignal3->SetXTitle("R for HB+ jeta = 15; depth = 3 \b");
          HBpositivedirectionRecosignal3->Draw("Error");
          kcountHBpositivedirectionRecosignal3++;
          if (kcountHBpositivedirectionRecosignal3 > 16)
            break;  // 4x6 = 24
        }           //ccctest>0

      }  // for i
    }    //if(jeta-41 >= 0)
  }      //for jeta
  /////////////////
  c3x5->Update();
  c3x5->Print("RrecosignalPositiveDirectionhistD1PhiSymmetryDepth3HB.png");
  c3x5->Clear();
  // clean-up
  if (h2CeffHBpositivedirectionRecosignal3)
    delete h2CeffHBpositivedirectionRecosignal3;
  //========================================================================================== 7
  //======================================================================
  //======================================================================1D plot: R vs phi , different eta,  depth=4
  //cout<<"      1D plot: R vs phi , different eta,  depth=4 *****" <<endl;
  c3x5->Clear();
  /////////////////
  c3x5->Divide(4, 4);
  c3x5->cd(1);
  int kcountHBpositivedirectionRecosignal4 = 1;
  TH1F* h2CeffHBpositivedirectionRecosignal4 = new TH1F("h2CeffHBpositivedirectionRecosignal4", "", nphi, 0., 72.);

  for (int jeta = 0; jeta < njeta; jeta++) {
    // positivedirectionRecosignal:
    if (jeta - 41 >= 0) {
      //	     for (int i=0;i<ndepth;i++) {
      // depth=4
      for (int i = 3; i < 4; i++) {
        TH1F* HBpositivedirectionRecosignal4 = (TH1F*)h2CeffHBpositivedirectionRecosignal4->Clone("twod1");

        float ccctest = 0;  // to avoid empty massive elements
        for (int jphi = 0; jphi < nphi; jphi++) {
          double ccc1 = arecosignalHB[i][jeta][jphi];
          if (ccc1 != 0.) {
            HBpositivedirectionRecosignal4->Fill(jphi, ccc1);
            ccctest = 1.;  //HBpositivedirectionRecosignal4->SetBinError(i,0.01);
          }
        }  // for jphi
        if (ccctest > 0.) {
          //cout<<"777        kcountHBpositivedirectionRecosignal4   =     "<<kcountHBpositivedirectionRecosignal4  <<"   jeta-41=     "<< jeta-41 <<endl;
          c3x5->cd(kcountHBpositivedirectionRecosignal4);
          HBpositivedirectionRecosignal4->SetMarkerStyle(20);
          HBpositivedirectionRecosignal4->SetMarkerSize(0.4);
          HBpositivedirectionRecosignal4->GetYaxis()->SetLabelSize(0.04);
          HBpositivedirectionRecosignal4->SetXTitle("HBpositivedirectionRecosignal4 \b");
          HBpositivedirectionRecosignal4->SetMarkerColor(2);
          HBpositivedirectionRecosignal4->SetLineColor(0);
          gPad->SetGridy();
          gPad->SetGridx();
          //	   gPad->SetLogy();
          if (kcountHBpositivedirectionRecosignal4 == 1)
            HBpositivedirectionRecosignal4->SetXTitle("R for HB+ jeta =  0; depth = 4 \b");
          if (kcountHBpositivedirectionRecosignal4 == 2)
            HBpositivedirectionRecosignal4->SetXTitle("R for HB+ jeta =  1; depth = 4 \b");
          if (kcountHBpositivedirectionRecosignal4 == 3)
            HBpositivedirectionRecosignal4->SetXTitle("R for HB+ jeta =  2; depth = 4 \b");
          if (kcountHBpositivedirectionRecosignal4 == 4)
            HBpositivedirectionRecosignal4->SetXTitle("R for HB+ jeta =  3; depth = 4 \b");
          if (kcountHBpositivedirectionRecosignal4 == 5)
            HBpositivedirectionRecosignal4->SetXTitle("R for HB+ jeta =  4; depth = 4 \b");
          if (kcountHBpositivedirectionRecosignal4 == 6)
            HBpositivedirectionRecosignal4->SetXTitle("R for HB+ jeta =  5; depth = 4 \b");
          if (kcountHBpositivedirectionRecosignal4 == 7)
            HBpositivedirectionRecosignal4->SetXTitle("R for HB+ jeta =  6; depth = 4 \b");
          if (kcountHBpositivedirectionRecosignal4 == 8)
            HBpositivedirectionRecosignal4->SetXTitle("R for HB+ jeta =  7; depth = 4 \b");
          if (kcountHBpositivedirectionRecosignal4 == 9)
            HBpositivedirectionRecosignal4->SetXTitle("R for HB+ jeta =  8; depth = 4 \b");
          if (kcountHBpositivedirectionRecosignal4 == 10)
            HBpositivedirectionRecosignal4->SetXTitle("R for HB+ jeta =  9; depth = 4 \b");
          if (kcountHBpositivedirectionRecosignal4 == 11)
            HBpositivedirectionRecosignal4->SetXTitle("R for HB+ jeta = 10; depth = 4 \b");
          if (kcountHBpositivedirectionRecosignal4 == 12)
            HBpositivedirectionRecosignal4->SetXTitle("R for HB+ jeta = 11; depth = 4 \b");
          if (kcountHBpositivedirectionRecosignal4 == 13)
            HBpositivedirectionRecosignal4->SetXTitle("R for HB+ jeta = 12; depth = 4 \b");
          if (kcountHBpositivedirectionRecosignal4 == 14)
            HBpositivedirectionRecosignal4->SetXTitle("R for HB+ jeta = 13; depth = 4 \b");
          if (kcountHBpositivedirectionRecosignal4 == 15)
            HBpositivedirectionRecosignal4->SetXTitle("R for HB+ jeta = 14; depth = 4 \b");
          if (kcountHBpositivedirectionRecosignal4 == 16)
            HBpositivedirectionRecosignal4->SetXTitle("R for HB+ jeta = 15; depth = 4 \b");
          HBpositivedirectionRecosignal4->Draw("Error");
          kcountHBpositivedirectionRecosignal4++;
          if (kcountHBpositivedirectionRecosignal4 > 16)
            break;  // 4x6 = 24
        }           //ccctest>0

      }  // for i
    }    //if(jeta-41 >= 0)
  }      //for jeta
  /////////////////
  c3x5->Update();
  c3x5->Print("RrecosignalPositiveDirectionhistD1PhiSymmetryDepth4HB.png");
  c3x5->Clear();
  // clean-up
  if (h2CeffHBpositivedirectionRecosignal4)
    delete h2CeffHBpositivedirectionRecosignal4;

  //========================================================================================== 1114
  //======================================================================
  //======================================================================1D plot: R vs phi , different eta,  depth=1
  //cout<<"      1D plot: R vs phi , different eta,  depth=1 *****" <<endl;
  c3x5->Clear();
  /////////////////
  c3x5->Divide(4, 4);
  c3x5->cd(1);
  int kcountHBnegativedirectionRecosignal1 = 1;
  TH1F* h2CeffHBnegativedirectionRecosignal1 = new TH1F("h2CeffHBnegativedirectionRecosignal1", "", nphi, 0., 72.);
  for (int jeta = 0; jeta < njeta; jeta++) {
    // negativedirectionRecosignal:
    if (jeta - 41 < 0) {
      //	     for (int i=0;i<ndepth;i++) {
      // depth=1
      for (int i = 0; i < 1; i++) {
        TH1F* HBnegativedirectionRecosignal1 = (TH1F*)h2CeffHBnegativedirectionRecosignal1->Clone("twod1");
        float ccctest = 0;  // to avoid empty massive elements
        for (int jphi = 0; jphi < nphi; jphi++) {
          double ccc1 = arecosignalHB[i][jeta][jphi];
          if (ccc1 != 0.) {
            HBnegativedirectionRecosignal1->Fill(jphi, ccc1);
            ccctest = 1.;  //HBnegativedirectionRecosignal1->SetBinError(i,0.01);
          }
        }  // for jphi
        if (ccctest > 0.) {
          //	  cout<<"444        kcountHBnegativedirectionRecosignal1   =     "<<kcountHBnegativedirectionRecosignal1  <<"   jeta-41=     "<< jeta-41 <<endl;
          c3x5->cd(kcountHBnegativedirectionRecosignal1);
          HBnegativedirectionRecosignal1->SetMarkerStyle(20);
          HBnegativedirectionRecosignal1->SetMarkerSize(0.4);
          HBnegativedirectionRecosignal1->GetYaxis()->SetLabelSize(0.04);
          HBnegativedirectionRecosignal1->SetXTitle("HBnegativedirectionRecosignal1 \b");
          HBnegativedirectionRecosignal1->SetMarkerColor(2);
          HBnegativedirectionRecosignal1->SetLineColor(0);
          gPad->SetGridy();
          gPad->SetGridx();
          //	   gPad->SetLogy();
          if (kcountHBnegativedirectionRecosignal1 == 1)
            HBnegativedirectionRecosignal1->SetXTitle("R for HB- jeta = -16; depth = 1 \b");
          if (kcountHBnegativedirectionRecosignal1 == 2)
            HBnegativedirectionRecosignal1->SetXTitle("R for HB- jeta = -15; depth = 1 \b");
          if (kcountHBnegativedirectionRecosignal1 == 3)
            HBnegativedirectionRecosignal1->SetXTitle("R for HB- jeta = -14; depth = 1 \b");
          if (kcountHBnegativedirectionRecosignal1 == 4)
            HBnegativedirectionRecosignal1->SetXTitle("R for HB- jeta = -13; depth = 1 \b");
          if (kcountHBnegativedirectionRecosignal1 == 5)
            HBnegativedirectionRecosignal1->SetXTitle("R for HB- jeta = -12; depth = 1 \b");
          if (kcountHBnegativedirectionRecosignal1 == 6)
            HBnegativedirectionRecosignal1->SetXTitle("R for HB- jeta = -11; depth = 1 \b");
          if (kcountHBnegativedirectionRecosignal1 == 7)
            HBnegativedirectionRecosignal1->SetXTitle("R for HB- jeta = -10; depth = 1 \b");
          if (kcountHBnegativedirectionRecosignal1 == 8)
            HBnegativedirectionRecosignal1->SetXTitle("R for HB- jeta =  -9; depth = 1 \b");
          if (kcountHBnegativedirectionRecosignal1 == 9)
            HBnegativedirectionRecosignal1->SetXTitle("R for HB- jeta =  -8; depth = 1 \b");
          if (kcountHBnegativedirectionRecosignal1 == 10)
            HBnegativedirectionRecosignal1->SetXTitle("R for HB- jeta =  -7; depth = 1 \b");
          if (kcountHBnegativedirectionRecosignal1 == 11)
            HBnegativedirectionRecosignal1->SetXTitle("R for HB- jeta =  -6; depth = 1 \b");
          if (kcountHBnegativedirectionRecosignal1 == 12)
            HBnegativedirectionRecosignal1->SetXTitle("R for HB- jeta =  -5; depth = 1 \b");
          if (kcountHBnegativedirectionRecosignal1 == 13)
            HBnegativedirectionRecosignal1->SetXTitle("R for HB- jeta =  -4; depth = 1 \b");
          if (kcountHBnegativedirectionRecosignal1 == 14)
            HBnegativedirectionRecosignal1->SetXTitle("R for HB- jeta =  -3; depth = 1 \b");
          if (kcountHBnegativedirectionRecosignal1 == 15)
            HBnegativedirectionRecosignal1->SetXTitle("R for HB- jeta =  -2; depth = 1 \b");
          if (kcountHBnegativedirectionRecosignal1 == 16)
            HBnegativedirectionRecosignal1->SetXTitle("R for HB- jeta =  -1; depth = 1 \b");
          HBnegativedirectionRecosignal1->Draw("Error");
          kcountHBnegativedirectionRecosignal1++;
          if (kcountHBnegativedirectionRecosignal1 > 16)
            break;  //
        }           //ccctest>0

      }  // for i
    }    //if(jeta-41 < 0 )
  }      //for jeta
  /////////////////
  c3x5->Update();
  c3x5->Print("RrecosignalNegativeDirectionhistD1PhiSymmetryDepth1HB.png");
  c3x5->Clear();
  // clean-up
  if (h2CeffHBnegativedirectionRecosignal1)
    delete h2CeffHBnegativedirectionRecosignal1;

  //========================================================================================== 1115
  //======================================================================
  //======================================================================1D plot: R vs phi , different eta,  depth=2
  //  cout<<"      1D plot: R vs phi , different eta,  depth=2 *****" <<endl;
  c3x5->Clear();
  /////////////////
  c3x5->Divide(4, 4);
  c3x5->cd(1);
  int kcountHBnegativedirectionRecosignal2 = 1;
  TH1F* h2CeffHBnegativedirectionRecosignal2 = new TH1F("h2CeffHBnegativedirectionRecosignal2", "", nphi, 0., 72.);
  for (int jeta = 0; jeta < njeta; jeta++) {
    // negativedirectionRecosignal:
    if (jeta - 41 < 0) {
      //	     for (int i=0;i<ndepth;i++) {
      // depth=2
      for (int i = 1; i < 2; i++) {
        TH1F* HBnegativedirectionRecosignal2 = (TH1F*)h2CeffHBnegativedirectionRecosignal2->Clone("twod1");
        float ccctest = 0;  // to avoid empty massive elements
        for (int jphi = 0; jphi < nphi; jphi++) {
          double ccc1 = arecosignalHB[i][jeta][jphi];
          if (ccc1 != 0.) {
            HBnegativedirectionRecosignal2->Fill(jphi, ccc1);
            ccctest = 1.;  //HBnegativedirectionRecosignal2->SetBinError(i,0.01);
          }
        }  // for jphi
        if (ccctest > 0.) {
          //cout<<"555        kcountHBnegativedirectionRecosignal2   =     "<<kcountHBnegativedirectionRecosignal2  <<"   jeta-41=     "<< jeta-41 <<endl;
          c3x5->cd(kcountHBnegativedirectionRecosignal2);
          HBnegativedirectionRecosignal2->SetMarkerStyle(20);
          HBnegativedirectionRecosignal2->SetMarkerSize(0.4);
          HBnegativedirectionRecosignal2->GetYaxis()->SetLabelSize(0.04);
          HBnegativedirectionRecosignal2->SetXTitle("HBnegativedirectionRecosignal2 \b");
          HBnegativedirectionRecosignal2->SetMarkerColor(2);
          HBnegativedirectionRecosignal2->SetLineColor(0);
          gPad->SetGridy();
          gPad->SetGridx();
          //	   gPad->SetLogy();
          if (kcountHBnegativedirectionRecosignal2 == 1)
            HBnegativedirectionRecosignal2->SetXTitle("R for HB- jeta = -16; depth = 2 \b");
          if (kcountHBnegativedirectionRecosignal2 == 2)
            HBnegativedirectionRecosignal2->SetXTitle("R for HB- jeta = -15; depth = 2 \b");
          if (kcountHBnegativedirectionRecosignal2 == 3)
            HBnegativedirectionRecosignal2->SetXTitle("R for HB- jeta = -14; depth = 2 \b");
          if (kcountHBnegativedirectionRecosignal2 == 4)
            HBnegativedirectionRecosignal2->SetXTitle("R for HB- jeta = -13; depth = 2 \b");
          if (kcountHBnegativedirectionRecosignal2 == 5)
            HBnegativedirectionRecosignal2->SetXTitle("R for HB- jeta = -12; depth = 2 \b");
          if (kcountHBnegativedirectionRecosignal2 == 6)
            HBnegativedirectionRecosignal2->SetXTitle("R for HB- jeta = -11; depth = 2 \b");
          if (kcountHBnegativedirectionRecosignal2 == 7)
            HBnegativedirectionRecosignal2->SetXTitle("R for HB- jeta = -10; depth = 2 \b");
          if (kcountHBnegativedirectionRecosignal2 == 8)
            HBnegativedirectionRecosignal2->SetXTitle("R for HB- jeta =  -9; depth = 2 \b");
          if (kcountHBnegativedirectionRecosignal2 == 9)
            HBnegativedirectionRecosignal2->SetXTitle("R for HB- jeta =  -8; depth = 2 \b");
          if (kcountHBnegativedirectionRecosignal2 == 10)
            HBnegativedirectionRecosignal2->SetXTitle("R for HB- jeta =  -7; depth = 2 \b");
          if (kcountHBnegativedirectionRecosignal2 == 11)
            HBnegativedirectionRecosignal2->SetXTitle("R for HB- jeta =  -6; depth = 2 \b");
          if (kcountHBnegativedirectionRecosignal2 == 12)
            HBnegativedirectionRecosignal2->SetXTitle("R for HB- jeta =  -5; depth = 2 \b");
          if (kcountHBnegativedirectionRecosignal2 == 13)
            HBnegativedirectionRecosignal2->SetXTitle("R for HB- jeta =  -4; depth = 2 \b");
          if (kcountHBnegativedirectionRecosignal2 == 14)
            HBnegativedirectionRecosignal2->SetXTitle("R for HB- jeta =  -3; depth = 2 \b");
          if (kcountHBnegativedirectionRecosignal2 == 15)
            HBnegativedirectionRecosignal2->SetXTitle("R for HB- jeta =  -2; depth = 2 \b");
          if (kcountHBnegativedirectionRecosignal2 == 16)
            HBnegativedirectionRecosignal2->SetXTitle("R for HB- jeta =  -1; depth = 2 \b");
          HBnegativedirectionRecosignal2->Draw("Error");
          kcountHBnegativedirectionRecosignal2++;
          if (kcountHBnegativedirectionRecosignal2 > 16)
            break;  // 4x6 = 24
        }           //ccctest>0

      }  // for i
    }    //if(jeta-41 < 0 )
  }      //for jeta
  /////////////////
  c3x5->Update();
  c3x5->Print("RrecosignalNegativeDirectionhistD1PhiSymmetryDepth2HB.png");
  c3x5->Clear();
  // clean-up
  if (h2CeffHBnegativedirectionRecosignal2)
    delete h2CeffHBnegativedirectionRecosignal2;
  //========================================================================================== 1116
  //======================================================================
  //======================================================================1D plot: R vs phi , different eta,  depth=3
  //cout<<"      1D plot: R vs phi , different eta,  depth=3 *****" <<endl;
  c3x5->Clear();
  /////////////////
  c3x5->Divide(4, 4);
  c3x5->cd(1);
  int kcountHBnegativedirectionRecosignal3 = 1;
  TH1F* h2CeffHBnegativedirectionRecosignal3 = new TH1F("h2CeffHBnegativedirectionRecosignal3", "", nphi, 0., 72.);
  for (int jeta = 0; jeta < njeta; jeta++) {
    // negativedirectionRecosignal:
    if (jeta - 41 < 0) {
      //	     for (int i=0;i<ndepth;i++) {
      // depth=3
      for (int i = 2; i < 3; i++) {
        TH1F* HBnegativedirectionRecosignal3 = (TH1F*)h2CeffHBnegativedirectionRecosignal3->Clone("twod1");
        float ccctest = 0;  // to avoid empty massive elements
        for (int jphi = 0; jphi < nphi; jphi++) {
          double ccc1 = arecosignalHB[i][jeta][jphi];
          if (ccc1 != 0.) {
            HBnegativedirectionRecosignal3->Fill(jphi, ccc1);
            ccctest = 1.;  //HBnegativedirectionRecosignal3->SetBinError(i,0.01);
          }
        }  // for jphi
        if (ccctest > 0.) {
          //cout<<"666        kcountHBnegativedirectionRecosignal3   =     "<<kcountHBnegativedirectionRecosignal3  <<"   jeta-41=     "<< jeta-41 <<endl;
          c3x5->cd(kcountHBnegativedirectionRecosignal3);
          HBnegativedirectionRecosignal3->SetMarkerStyle(20);
          HBnegativedirectionRecosignal3->SetMarkerSize(0.4);
          HBnegativedirectionRecosignal3->GetYaxis()->SetLabelSize(0.04);
          HBnegativedirectionRecosignal3->SetXTitle("HBnegativedirectionRecosignal3 \b");
          HBnegativedirectionRecosignal3->SetMarkerColor(2);
          HBnegativedirectionRecosignal3->SetLineColor(0);
          gPad->SetGridy();
          gPad->SetGridx();
          //	   gPad->SetLogy();
          if (kcountHBnegativedirectionRecosignal3 == 1)
            HBnegativedirectionRecosignal3->SetXTitle("R for HB- jeta = -16; depth = 3 \b");
          if (kcountHBnegativedirectionRecosignal3 == 2)
            HBnegativedirectionRecosignal3->SetXTitle("R for HB- jeta = -15; depth = 3 \b");
          if (kcountHBnegativedirectionRecosignal3 == 3)
            HBnegativedirectionRecosignal3->SetXTitle("R for HB- jeta = -14; depth = 3 \b");
          if (kcountHBnegativedirectionRecosignal3 == 4)
            HBnegativedirectionRecosignal3->SetXTitle("R for HB- jeta = -13; depth = 3 \b");
          if (kcountHBnegativedirectionRecosignal3 == 5)
            HBnegativedirectionRecosignal3->SetXTitle("R for HB- jeta = -12; depth = 3 \b");
          if (kcountHBnegativedirectionRecosignal3 == 6)
            HBnegativedirectionRecosignal3->SetXTitle("R for HB- jeta = -11; depth = 3 \b");
          if (kcountHBnegativedirectionRecosignal3 == 7)
            HBnegativedirectionRecosignal3->SetXTitle("R for HB- jeta = -10; depth = 3 \b");
          if (kcountHBnegativedirectionRecosignal3 == 8)
            HBnegativedirectionRecosignal3->SetXTitle("R for HB- jeta =  -9; depth = 3 \b");
          if (kcountHBnegativedirectionRecosignal3 == 9)
            HBnegativedirectionRecosignal3->SetXTitle("R for HB- jeta =  -8; depth = 3 \b");
          if (kcountHBnegativedirectionRecosignal3 == 10)
            HBnegativedirectionRecosignal3->SetXTitle("R for HB- jeta =  -7; depth = 3 \b");
          if (kcountHBnegativedirectionRecosignal3 == 11)
            HBnegativedirectionRecosignal3->SetXTitle("R for HB- jeta =  -6; depth = 3 \b");
          if (kcountHBnegativedirectionRecosignal3 == 12)
            HBnegativedirectionRecosignal3->SetXTitle("R for HB- jeta =  -5; depth = 3 \b");
          if (kcountHBnegativedirectionRecosignal3 == 13)
            HBnegativedirectionRecosignal3->SetXTitle("R for HB- jeta =  -4; depth = 3 \b");
          if (kcountHBnegativedirectionRecosignal3 == 14)
            HBnegativedirectionRecosignal3->SetXTitle("R for HB- jeta =  -3; depth = 3 \b");
          if (kcountHBnegativedirectionRecosignal3 == 15)
            HBnegativedirectionRecosignal3->SetXTitle("R for HB- jeta =  -2; depth = 3 \b");
          if (kcountHBnegativedirectionRecosignal3 == 16)
            HBnegativedirectionRecosignal3->SetXTitle("R for HB- jeta =  -1; depth = 3 \b");

          HBnegativedirectionRecosignal3->Draw("Error");
          kcountHBnegativedirectionRecosignal3++;
          if (kcountHBnegativedirectionRecosignal3 > 16)
            break;  // 4x6 = 24
        }           //ccctest>0

      }  // for i
    }    //if(jeta-41 < 0 )
  }      //for jeta
  /////////////////
  c3x5->Update();
  c3x5->Print("RrecosignalNegativeDirectionhistD1PhiSymmetryDepth3HB.png");
  c3x5->Clear();
  // clean-up
  if (h2CeffHBnegativedirectionRecosignal3)
    delete h2CeffHBnegativedirectionRecosignal3;
  //========================================================================================== 1117
  //======================================================================
  //======================================================================1D plot: R vs phi , different eta,  depth=4
  //cout<<"      1D plot: R vs phi , different eta,  depth=4 *****" <<endl;
  c3x5->Clear();
  /////////////////
  c3x5->Divide(4, 4);
  c3x5->cd(1);
  int kcountHBnegativedirectionRecosignal4 = 1;
  TH1F* h2CeffHBnegativedirectionRecosignal4 = new TH1F("h2CeffHBnegativedirectionRecosignal4", "", nphi, 0., 72.);

  for (int jeta = 0; jeta < njeta; jeta++) {
    // negativedirectionRecosignal:
    if (jeta - 41 < 0) {
      //	     for (int i=0;i<ndepth;i++) {
      // depth=4
      for (int i = 3; i < 4; i++) {
        TH1F* HBnegativedirectionRecosignal4 = (TH1F*)h2CeffHBnegativedirectionRecosignal4->Clone("twod1");

        float ccctest = 0;  // to avoid empty massive elements
        for (int jphi = 0; jphi < nphi; jphi++) {
          double ccc1 = arecosignalHB[i][jeta][jphi];
          if (ccc1 != 0.) {
            HBnegativedirectionRecosignal4->Fill(jphi, ccc1);
            ccctest = 1.;  //HBnegativedirectionRecosignal4->SetBinError(i,0.01);
          }
        }  // for jphi
        if (ccctest > 0.) {
          //cout<<"777        kcountHBnegativedirectionRecosignal4   =     "<<kcountHBnegativedirectionRecosignal4  <<"   jeta-41=     "<< jeta-41 <<endl;
          c3x5->cd(kcountHBnegativedirectionRecosignal4);
          HBnegativedirectionRecosignal4->SetMarkerStyle(20);
          HBnegativedirectionRecosignal4->SetMarkerSize(0.4);
          HBnegativedirectionRecosignal4->GetYaxis()->SetLabelSize(0.04);
          HBnegativedirectionRecosignal4->SetXTitle("HBnegativedirectionRecosignal4 \b");
          HBnegativedirectionRecosignal4->SetMarkerColor(2);
          HBnegativedirectionRecosignal4->SetLineColor(0);
          gPad->SetGridy();
          gPad->SetGridx();
          //	   gPad->SetLogy();
          if (kcountHBnegativedirectionRecosignal4 == 1)
            HBnegativedirectionRecosignal4->SetXTitle("R for HB- jeta = -16; depth = 4 \b");
          if (kcountHBnegativedirectionRecosignal4 == 2)
            HBnegativedirectionRecosignal4->SetXTitle("R for HB- jeta = -15; depth = 4 \b");
          if (kcountHBnegativedirectionRecosignal4 == 3)
            HBnegativedirectionRecosignal4->SetXTitle("R for HB- jeta = -14; depth = 4 \b");
          if (kcountHBnegativedirectionRecosignal4 == 4)
            HBnegativedirectionRecosignal4->SetXTitle("R for HB- jeta = -13; depth = 4 \b");
          if (kcountHBnegativedirectionRecosignal4 == 5)
            HBnegativedirectionRecosignal4->SetXTitle("R for HB- jeta = -12; depth = 4 \b");
          if (kcountHBnegativedirectionRecosignal4 == 6)
            HBnegativedirectionRecosignal4->SetXTitle("R for HB- jeta = -11; depth = 4 \b");
          if (kcountHBnegativedirectionRecosignal4 == 7)
            HBnegativedirectionRecosignal4->SetXTitle("R for HB- jeta = -10; depth = 4 \b");
          if (kcountHBnegativedirectionRecosignal4 == 8)
            HBnegativedirectionRecosignal4->SetXTitle("R for HB- jeta =  -9; depth = 4 \b");
          if (kcountHBnegativedirectionRecosignal4 == 9)
            HBnegativedirectionRecosignal4->SetXTitle("R for HB- jeta =  -8; depth = 4 \b");
          if (kcountHBnegativedirectionRecosignal4 == 10)
            HBnegativedirectionRecosignal4->SetXTitle("R for HB- jeta =  -7; depth = 4 \b");
          if (kcountHBnegativedirectionRecosignal4 == 11)
            HBnegativedirectionRecosignal4->SetXTitle("R for HB- jeta =  -6; depth = 4 \b");
          if (kcountHBnegativedirectionRecosignal4 == 12)
            HBnegativedirectionRecosignal4->SetXTitle("R for HB- jeta =  -5; depth = 4 \b");
          if (kcountHBnegativedirectionRecosignal4 == 13)
            HBnegativedirectionRecosignal4->SetXTitle("R for HB- jeta =  -4; depth = 4 \b");
          if (kcountHBnegativedirectionRecosignal4 == 14)
            HBnegativedirectionRecosignal4->SetXTitle("R for HB- jeta =  -3; depth = 4 \b");
          if (kcountHBnegativedirectionRecosignal4 == 15)
            HBnegativedirectionRecosignal4->SetXTitle("R for HB- jeta =  -2; depth = 4 \b");
          if (kcountHBnegativedirectionRecosignal4 == 16)
            HBnegativedirectionRecosignal4->SetXTitle("R for HB- jeta =  -1; depth = 4 \b");
          HBnegativedirectionRecosignal4->Draw("Error");
          kcountHBnegativedirectionRecosignal4++;
          if (kcountHBnegativedirectionRecosignal4 > 16)
            break;  // 4x6 = 24
        }           //ccctest>0

      }  // for i
    }    //if(jeta-41 < 0 )
  }      //for jeta
  /////////////////
  c3x5->Update();
  c3x5->Print("RrecosignalNegativeDirectionhistD1PhiSymmetryDepth4HB.png");
  c3x5->Clear();
  // clean-up
  if (h2CeffHBnegativedirectionRecosignal4)
    delete h2CeffHBnegativedirectionRecosignal4;

  //======================================================================================================================
  //======================================================================================================================
  //======================================================================================================================
  //                            DDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDD:

  //cout<<"    Start Vaiance: preparation  *****" <<endl;
  TH2F* recosignalVariance1HB1 = (TH2F*)dir->FindObjectAny("h_recSignalEnergy2_HB1");
  TH2F* recosignalVariance0HB1 = (TH2F*)dir->FindObjectAny("h_recSignalEnergy0_HB1");
  TH2F* recosignalVarianceHB1 = (TH2F*)recosignalVariance1HB1->Clone("recosignalVarianceHB1");
  recosignalVarianceHB1->Divide(recosignalVariance1HB1, recosignalVariance0HB1, 1, 1, "B");
  TH2F* recosignalVariance1HB2 = (TH2F*)dir->FindObjectAny("h_recSignalEnergy2_HB2");
  TH2F* recosignalVariance0HB2 = (TH2F*)dir->FindObjectAny("h_recSignalEnergy0_HB2");
  TH2F* recosignalVarianceHB2 = (TH2F*)recosignalVariance1HB2->Clone("recosignalVarianceHB2");
  recosignalVarianceHB2->Divide(recosignalVariance1HB2, recosignalVariance0HB2, 1, 1, "B");
  TH2F* recosignalVariance1HB3 = (TH2F*)dir->FindObjectAny("h_recSignalEnergy2_HB3");
  TH2F* recosignalVariance0HB3 = (TH2F*)dir->FindObjectAny("h_recSignalEnergy0_HB3");
  TH2F* recosignalVarianceHB3 = (TH2F*)recosignalVariance1HB3->Clone("recosignalVarianceHB3");
  recosignalVarianceHB3->Divide(recosignalVariance1HB3, recosignalVariance0HB3, 1, 1, "B");
  TH2F* recosignalVariance1HB4 = (TH2F*)dir->FindObjectAny("h_recSignalEnergy2_HB4");
  TH2F* recosignalVariance0HB4 = (TH2F*)dir->FindObjectAny("h_recSignalEnergy0_HB4");
  TH2F* recosignalVarianceHB4 = (TH2F*)recosignalVariance1HB4->Clone("recosignalVarianceHB4");
  recosignalVarianceHB4->Divide(recosignalVariance1HB4, recosignalVariance0HB4, 1, 1, "B");
  //cout<<"      Vaiance: preparation DONE *****" <<endl;
  //====================================================================== put Vaiance=Dispersia = Sig**2=<R**2> - (<R>)**2 into massive recosignalvarianceHB
  //                                                                                           = sum(R*R)/N - (sum(R)/N)**2
  for (int jeta = 0; jeta < njeta; jeta++) {
    //preparation for PHI normalization:
    double sumrecosignalHB0 = 0;
    int nsumrecosignalHB0 = 0;
    double sumrecosignalHB1 = 0;
    int nsumrecosignalHB1 = 0;
    double sumrecosignalHB2 = 0;
    int nsumrecosignalHB2 = 0;
    double sumrecosignalHB3 = 0;
    int nsumrecosignalHB3 = 0;
    for (int jphi = 0; jphi < njphi; jphi++) {
      recosignalvarianceHB[0][jeta][jphi] = recosignalVarianceHB1->GetBinContent(jeta + 1, jphi + 1);
      recosignalvarianceHB[1][jeta][jphi] = recosignalVarianceHB2->GetBinContent(jeta + 1, jphi + 1);
      recosignalvarianceHB[2][jeta][jphi] = recosignalVarianceHB3->GetBinContent(jeta + 1, jphi + 1);
      recosignalvarianceHB[3][jeta][jphi] = recosignalVarianceHB4->GetBinContent(jeta + 1, jphi + 1);
      if (recosignalvarianceHB[0][jeta][jphi] > 0.) {
        sumrecosignalHB0 += recosignalvarianceHB[0][jeta][jphi];
        ++nsumrecosignalHB0;
      }
      if (recosignalvarianceHB[1][jeta][jphi] > 0.) {
        sumrecosignalHB1 += recosignalvarianceHB[1][jeta][jphi];
        ++nsumrecosignalHB1;
      }
      if (recosignalvarianceHB[2][jeta][jphi] > 0.) {
        sumrecosignalHB2 += recosignalvarianceHB[2][jeta][jphi];
        ++nsumrecosignalHB2;
      }
      if (recosignalvarianceHB[3][jeta][jphi] > 0.) {
        sumrecosignalHB3 += recosignalvarianceHB[3][jeta][jphi];
        ++nsumrecosignalHB3;
      }
    }  // phi
    // PHI normalization :
    for (int jphi = 0; jphi < njphi; jphi++) {
      if (recosignalvarianceHB[0][jeta][jphi] > 0.)
        recosignalvarianceHB[0][jeta][jphi] /= (sumrecosignalHB0 / nsumrecosignalHB0);
      if (recosignalvarianceHB[1][jeta][jphi] > 0.)
        recosignalvarianceHB[1][jeta][jphi] /= (sumrecosignalHB1 / nsumrecosignalHB1);
      if (recosignalvarianceHB[2][jeta][jphi] > 0.)
        recosignalvarianceHB[2][jeta][jphi] /= (sumrecosignalHB2 / nsumrecosignalHB2);
      if (recosignalvarianceHB[3][jeta][jphi] > 0.)
        recosignalvarianceHB[3][jeta][jphi] /= (sumrecosignalHB3 / nsumrecosignalHB3);
    }  // phi
    //       recosignalvarianceHB (D)           = sum(R*R)/N - (sum(R)/N)**2
    for (int jphi = 0; jphi < njphi; jphi++) {
      //	   cout<<"12 12 12   jeta=     "<< jeta <<"   jphi   =     "<<jphi  <<endl;
      recosignalvarianceHB[0][jeta][jphi] -= arecosignalHB[0][jeta][jphi] * arecosignalHB[0][jeta][jphi];
      recosignalvarianceHB[0][jeta][jphi] = fabs(recosignalvarianceHB[0][jeta][jphi]);
      recosignalvarianceHB[1][jeta][jphi] -= arecosignalHB[1][jeta][jphi] * arecosignalHB[1][jeta][jphi];
      recosignalvarianceHB[1][jeta][jphi] = fabs(recosignalvarianceHB[1][jeta][jphi]);
      recosignalvarianceHB[2][jeta][jphi] -= arecosignalHB[2][jeta][jphi] * arecosignalHB[2][jeta][jphi];
      recosignalvarianceHB[2][jeta][jphi] = fabs(recosignalvarianceHB[2][jeta][jphi]);
      recosignalvarianceHB[3][jeta][jphi] -= arecosignalHB[3][jeta][jphi] * arecosignalHB[3][jeta][jphi];
      recosignalvarianceHB[3][jeta][jphi] = fabs(recosignalvarianceHB[3][jeta][jphi]);
    }
  }
  //cout<<"      Vaiance: DONE*****" <<endl;
  //------------------------  2D-eta/phi-plot: D, averaged over depthfs
  //======================================================================
  //======================================================================
  //cout<<"      R2D-eta/phi-plot: D, averaged over depthfs *****" <<endl;
  c1x1->Clear();
  /////////////////
  c1x0->Divide(1, 1);
  c1x0->cd(1);
  TH2F* DefzDrecosignalHB42D = new TH2F("DefzDrecosignalHB42D", "", neta, -41., 41., nphi, 0., 72.);
  TH2F* DefzDrecosignalHB42D0 = new TH2F("DefzDrecosignalHB42D0", "", neta, -41., 41., nphi, 0., 72.);
  TH2F* DefzDrecosignalHB42DF = (TH2F*)DefzDrecosignalHB42D0->Clone("DefzDrecosignalHB42DF");
  for (int i = 0; i < ndepth; i++) {
    for (int jeta = 0; jeta < neta; jeta++) {
      for (int jphi = 0; jphi < nphi; jphi++) {
        double ccc1 = recosignalvarianceHB[i][jeta][jphi];
        int k2plot = jeta - 41;
        int kkk = k2plot;  //if(k2plot >0   kkk=k2plot+1; //-41 +41 !=0
        if (arecosignalHB[i][jeta][jphi] > 0.) {
          DefzDrecosignalHB42D->Fill(kkk, jphi, ccc1);
          DefzDrecosignalHB42D0->Fill(kkk, jphi, 1.);
        }
      }
    }
  }
  DefzDrecosignalHB42DF->Divide(DefzDrecosignalHB42D, DefzDrecosignalHB42D0, 1, 1, "B");  // average A
  //    DefzDrecosignalHB1->Sumw2();
  gPad->SetGridy();
  gPad->SetGridx();  //      gPad->SetLogz();
  DefzDrecosignalHB42DF->SetMarkerStyle(20);
  DefzDrecosignalHB42DF->SetMarkerSize(0.4);
  DefzDrecosignalHB42DF->GetZaxis()->SetLabelSize(0.08);
  DefzDrecosignalHB42DF->SetXTitle("<D>_depth       #eta  \b");
  DefzDrecosignalHB42DF->SetYTitle("      #phi \b");
  DefzDrecosignalHB42DF->SetZTitle("<D>_depth \b");
  DefzDrecosignalHB42DF->SetMarkerColor(2);
  DefzDrecosignalHB42DF->SetLineColor(
      0);  //      DefzDrecosignalHB42DF->SetMaximum(1.000);  //      DefzDrecosignalHB42DF->SetMinimum(1.0);
  DefzDrecosignalHB42DF->Draw("COLZ");
  /////////////////
  c1x0->Update();
  c1x0->Print("DrecosignalGeneralD2PhiSymmetryHB.png");
  c1x0->Clear();
  // clean-up
  if (DefzDrecosignalHB42D)
    delete DefzDrecosignalHB42D;
  if (DefzDrecosignalHB42D0)
    delete DefzDrecosignalHB42D0;
  if (DefzDrecosignalHB42DF)
    delete DefzDrecosignalHB42DF;
  //====================================================================== 1D plot: D vs phi , averaged over depthfs & eta
  //======================================================================
  //cout<<"      1D plot: D vs phi , averaged over depthfs & eta *****" <<endl;
  c1x1->Clear();
  /////////////////
  c1x1->Divide(1, 1);
  c1x1->cd(1);
  TH1F* DefzDrecosignalHB41D = new TH1F("DefzDrecosignalHB41D", "", nphi, 0., 72.);
  TH1F* DefzDrecosignalHB41D0 = new TH1F("DefzDrecosignalHB41D0", "", nphi, 0., 72.);
  TH1F* DefzDrecosignalHB41DF = (TH1F*)DefzDrecosignalHB41D0->Clone("DefzDrecosignalHB41DF");

  for (int jphi = 0; jphi < nphi; jphi++) {
    for (int jeta = 0; jeta < neta; jeta++) {
      for (int i = 0; i < ndepth; i++) {
        double ccc1 = recosignalvarianceHB[i][jeta][jphi];
        if (arecosignalHB[i][jeta][jphi] > 0.) {
          DefzDrecosignalHB41D->Fill(jphi, ccc1);
          DefzDrecosignalHB41D0->Fill(jphi, 1.);
        }
      }
    }
  }
  //     DefzDrecosignalHB41D->Sumw2();DefzDrecosignalHB41D0->Sumw2();

  DefzDrecosignalHB41DF->Divide(
      DefzDrecosignalHB41D, DefzDrecosignalHB41D0, 1, 1, "B");  // R averaged over depthfs & eta
  DefzDrecosignalHB41D0->Sumw2();
  //    for (int jphi=1;jphi<73;jphi++) {DefzDrecosignalHB41DF->SetBinError(jphi,0.01);}
  gPad->SetGridy();
  gPad->SetGridx();  //      gPad->SetLogz();
  DefzDrecosignalHB41DF->SetMarkerStyle(20);
  DefzDrecosignalHB41DF->SetMarkerSize(1.4);
  DefzDrecosignalHB41DF->GetZaxis()->SetLabelSize(0.08);
  DefzDrecosignalHB41DF->SetXTitle("#phi  \b");
  DefzDrecosignalHB41DF->SetYTitle("  <D> \b");
  DefzDrecosignalHB41DF->SetZTitle("<D>_PHI  - AllDepthfs \b");
  DefzDrecosignalHB41DF->SetMarkerColor(4);
  DefzDrecosignalHB41DF->SetLineColor(
      4);  //  DefzDrecosignalHB41DF->SetMinimum(0.8);     DefzDrecosignalHB41DF->SetMinimum(-0.015);
  DefzDrecosignalHB41DF->Draw("Error");
  /////////////////
  c1x1->Update();
  c1x1->Print("DrecosignalGeneralD1PhiSymmetryHB.png");
  c1x1->Clear();
  // clean-up
  if (DefzDrecosignalHB41D)
    delete DefzDrecosignalHB41D;
  if (DefzDrecosignalHB41D0)
    delete DefzDrecosignalHB41D0;
  if (DefzDrecosignalHB41DF)
    delete DefzDrecosignalHB41DF;

  //========================================================================================== 14
  //======================================================================
  //======================================================================1D plot: D vs phi , different eta,  depth=1
  //cout<<"      1D plot: D vs phi , different eta,  depth=1 *****" <<endl;
  c3x5->Clear();
  /////////////////
  c3x5->Divide(4, 4);
  c3x5->cd(1);
  int kcountHBpositivedirectionRecosignalD1 = 1;
  TH1F* h2CeffHBpositivedirectionRecosignalD1 = new TH1F("h2CeffHBpositivedirectionRecosignalD1", "", nphi, 0., 72.);

  for (int jeta = 0; jeta < njeta; jeta++) {
    // positivedirectionRecosignalD:
    if (jeta - 41 >= 0) {
      //	     for (int i=0;i<ndepth;i++) {
      // depth=1
      for (int i = 0; i < 1; i++) {
        TH1F* HBpositivedirectionRecosignalD1 = (TH1F*)h2CeffHBpositivedirectionRecosignalD1->Clone("twod1");

        float ccctest = 0;  // to avoid empty massive elements
        for (int jphi = 0; jphi < nphi; jphi++) {
          double ccc1 = recosignalvarianceHB[i][jeta][jphi];
          if (arecosignalHB[i][jeta][jphi] > 0.) {
            HBpositivedirectionRecosignalD1->Fill(jphi, ccc1);
            ccctest = 1.;  //HBpositivedirectionRecosignalD1->SetBinError(i,0.01);
          }
        }  // for jphi
        if (ccctest > 0.) {
          //cout<<"1414       kcountHBpositivedirectionRecosignalD1   =     "<<kcountHBpositivedirectionRecosignalD1  <<"   jeta-41=     "<< jeta-41 <<endl;
          c3x5->cd(kcountHBpositivedirectionRecosignalD1);
          HBpositivedirectionRecosignalD1->SetMarkerStyle(20);
          HBpositivedirectionRecosignalD1->SetMarkerSize(0.4);
          HBpositivedirectionRecosignalD1->GetYaxis()->SetLabelSize(0.04);
          HBpositivedirectionRecosignalD1->SetXTitle("HBpositivedirectionRecosignalD1 \b");
          HBpositivedirectionRecosignalD1->SetMarkerColor(2);
          HBpositivedirectionRecosignalD1->SetLineColor(0);
          gPad->SetGridy();
          gPad->SetGridx();
          //	   gPad->SetLogy();
          if (kcountHBpositivedirectionRecosignalD1 == 1)
            HBpositivedirectionRecosignalD1->SetXTitle("D for HB+ jeta =  0; depth = 1 \b");
          if (kcountHBpositivedirectionRecosignalD1 == 2)
            HBpositivedirectionRecosignalD1->SetXTitle("D for HB+ jeta =  1; depth = 1 \b");
          if (kcountHBpositivedirectionRecosignalD1 == 3)
            HBpositivedirectionRecosignalD1->SetXTitle("D for HB+ jeta =  2; depth = 1 \b");
          if (kcountHBpositivedirectionRecosignalD1 == 4)
            HBpositivedirectionRecosignalD1->SetXTitle("D for HB+ jeta =  3; depth = 1 \b");
          if (kcountHBpositivedirectionRecosignalD1 == 5)
            HBpositivedirectionRecosignalD1->SetXTitle("D for HB+ jeta =  4; depth = 1 \b");
          if (kcountHBpositivedirectionRecosignalD1 == 6)
            HBpositivedirectionRecosignalD1->SetXTitle("D for HB+ jeta =  5; depth = 1 \b");
          if (kcountHBpositivedirectionRecosignalD1 == 7)
            HBpositivedirectionRecosignalD1->SetXTitle("D for HB+ jeta =  6; depth = 1 \b");
          if (kcountHBpositivedirectionRecosignalD1 == 8)
            HBpositivedirectionRecosignalD1->SetXTitle("D for HB+ jeta =  7; depth = 1 \b");
          if (kcountHBpositivedirectionRecosignalD1 == 9)
            HBpositivedirectionRecosignalD1->SetXTitle("D for HB+ jeta =  8; depth = 1 \b");
          if (kcountHBpositivedirectionRecosignalD1 == 10)
            HBpositivedirectionRecosignalD1->SetXTitle("D for HB+ jeta =  9; depth = 1 \b");
          if (kcountHBpositivedirectionRecosignalD1 == 11)
            HBpositivedirectionRecosignalD1->SetXTitle("D for HB+ jeta = 10; depth = 1 \b");
          if (kcountHBpositivedirectionRecosignalD1 == 12)
            HBpositivedirectionRecosignalD1->SetXTitle("D for HB+ jeta = 11; depth = 1 \b");
          if (kcountHBpositivedirectionRecosignalD1 == 13)
            HBpositivedirectionRecosignalD1->SetXTitle("D for HB+ jeta = 12; depth = 1 \b");
          if (kcountHBpositivedirectionRecosignalD1 == 14)
            HBpositivedirectionRecosignalD1->SetXTitle("D for HB+ jeta = 13; depth = 1 \b");
          if (kcountHBpositivedirectionRecosignalD1 == 15)
            HBpositivedirectionRecosignalD1->SetXTitle("D for HB+ jeta = 14; depth = 1 \b");
          if (kcountHBpositivedirectionRecosignalD1 == 16)
            HBpositivedirectionRecosignalD1->SetXTitle("D for HB+ jeta = 15; depth = 1 \b");
          HBpositivedirectionRecosignalD1->Draw("Error");
          kcountHBpositivedirectionRecosignalD1++;
          if (kcountHBpositivedirectionRecosignalD1 > 16)
            break;  // 4x6 = 24
        }           //ccctest>0

      }  // for i
    }    //if(jeta-41 >= 0)
  }      //for jeta
  /////////////////
  c3x5->Update();
  c3x5->Print("DrecosignalPositiveDirectionhistD1PhiSymmetryDepth1HB.png");
  c3x5->Clear();
  // clean-up
  if (h2CeffHBpositivedirectionRecosignalD1)
    delete h2CeffHBpositivedirectionRecosignalD1;
  //========================================================================================== 15
  //======================================================================
  //======================================================================1D plot: D vs phi , different eta,  depth=2
  //cout<<"      1D plot: D vs phi , different eta,  depth=2 *****" <<endl;
  c3x5->Clear();
  c3x5->Divide(4, 4);
  c3x5->cd(1);
  int kcountHBpositivedirectionRecosignalD2 = 1;
  TH1F* h2CeffHBpositivedirectionRecosignalD2 = new TH1F("h2CeffHBpositivedirectionRecosignalD2", "", nphi, 0., 72.);

  for (int jeta = 0; jeta < njeta; jeta++) {
    // positivedirectionRecosignalD:
    if (jeta - 41 >= 0) {
      //	     for (int i=0;i<ndepth;i++) {
      // depth=2
      for (int i = 1; i < 2; i++) {
        TH1F* HBpositivedirectionRecosignalD2 = (TH1F*)h2CeffHBpositivedirectionRecosignalD2->Clone("twod1");

        float ccctest = 0;  // to avoid empty massive elements
        for (int jphi = 0; jphi < nphi; jphi++) {
          double ccc1 = recosignalvarianceHB[i][jeta][jphi];
          if (arecosignalHB[i][jeta][jphi] > 0.) {
            HBpositivedirectionRecosignalD2->Fill(jphi, ccc1);
            ccctest = 1.;  //HBpositivedirectionRecosignalD2->SetBinError(i,0.01);
          }
        }  // for jphi
        if (ccctest > 0.) {
          //cout<<"1515       kcountHBpositivedirectionRecosignalD2   =     "<<kcountHBpositivedirectionRecosignalD2  <<"   jeta-41=     "<< jeta-41 <<endl;
          c3x5->cd(kcountHBpositivedirectionRecosignalD2);
          HBpositivedirectionRecosignalD2->SetMarkerStyle(20);
          HBpositivedirectionRecosignalD2->SetMarkerSize(0.4);
          HBpositivedirectionRecosignalD2->GetYaxis()->SetLabelSize(0.04);
          HBpositivedirectionRecosignalD2->SetXTitle("HBpositivedirectionRecosignalD2 \b");
          HBpositivedirectionRecosignalD2->SetMarkerColor(2);
          HBpositivedirectionRecosignalD2->SetLineColor(0);
          gPad->SetGridy();
          gPad->SetGridx();
          //	   gPad->SetLogy();
          if (kcountHBpositivedirectionRecosignalD2 == 1)
            HBpositivedirectionRecosignalD2->SetXTitle("D for HB+ jeta =  0; depth = 2 \b");
          if (kcountHBpositivedirectionRecosignalD2 == 2)
            HBpositivedirectionRecosignalD2->SetXTitle("D for HB+ jeta =  1; depth = 2 \b");
          if (kcountHBpositivedirectionRecosignalD2 == 3)
            HBpositivedirectionRecosignalD2->SetXTitle("D for HB+ jeta =  2; depth = 2 \b");
          if (kcountHBpositivedirectionRecosignalD2 == 4)
            HBpositivedirectionRecosignalD2->SetXTitle("D for HB+ jeta =  3; depth = 2 \b");
          if (kcountHBpositivedirectionRecosignalD2 == 5)
            HBpositivedirectionRecosignalD2->SetXTitle("D for HB+ jeta =  4; depth = 2 \b");
          if (kcountHBpositivedirectionRecosignalD2 == 6)
            HBpositivedirectionRecosignalD2->SetXTitle("D for HB+ jeta =  5; depth = 2 \b");
          if (kcountHBpositivedirectionRecosignalD2 == 7)
            HBpositivedirectionRecosignalD2->SetXTitle("D for HB+ jeta =  6; depth = 2 \b");
          if (kcountHBpositivedirectionRecosignalD2 == 8)
            HBpositivedirectionRecosignalD2->SetXTitle("D for HB+ jeta =  7; depth = 2 \b");
          if (kcountHBpositivedirectionRecosignalD2 == 9)
            HBpositivedirectionRecosignalD2->SetXTitle("D for HB+ jeta =  8; depth = 2 \b");
          if (kcountHBpositivedirectionRecosignalD2 == 10)
            HBpositivedirectionRecosignalD2->SetXTitle("D for HB+ jeta =  9; depth = 2 \b");
          if (kcountHBpositivedirectionRecosignalD2 == 11)
            HBpositivedirectionRecosignalD2->SetXTitle("D for HB+ jeta = 10; depth = 2 \b");
          if (kcountHBpositivedirectionRecosignalD2 == 12)
            HBpositivedirectionRecosignalD2->SetXTitle("D for HB+ jeta = 11; depth = 2 \b");
          if (kcountHBpositivedirectionRecosignalD2 == 13)
            HBpositivedirectionRecosignalD2->SetXTitle("D for HB+ jeta = 12; depth = 2 \b");
          if (kcountHBpositivedirectionRecosignalD2 == 14)
            HBpositivedirectionRecosignalD2->SetXTitle("D for HB+ jeta = 13; depth = 2 \b");
          if (kcountHBpositivedirectionRecosignalD2 == 15)
            HBpositivedirectionRecosignalD2->SetXTitle("D for HB+ jeta = 14; depth = 2 \b");
          if (kcountHBpositivedirectionRecosignalD2 == 16)
            HBpositivedirectionRecosignalD2->SetXTitle("D for HB+ jeta = 15; depth = 2 \b");
          HBpositivedirectionRecosignalD2->Draw("Error");
          kcountHBpositivedirectionRecosignalD2++;
          if (kcountHBpositivedirectionRecosignalD2 > 16)
            break;  // 4x6 = 24
        }           //ccctest>0

      }  // for i
    }    //if(jeta-41 >= 0)
  }      //for jeta
  /////////////////
  c3x5->Update();
  c3x5->Print("DrecosignalPositiveDirectionhistD1PhiSymmetryDepth2HB.png");
  c3x5->Clear();
  // clean-up
  if (h2CeffHBpositivedirectionRecosignalD2)
    delete h2CeffHBpositivedirectionRecosignalD2;
  //========================================================================================== 16
  //======================================================================
  //======================================================================1D plot: D vs phi , different eta,  depth=3
  //cout<<"      1D plot: D vs phi , different eta,  depth=3 *****" <<endl;
  c3x5->Clear();
  c3x5->Divide(4, 4);
  c3x5->cd(1);
  int kcountHBpositivedirectionRecosignalD3 = 1;
  TH1F* h2CeffHBpositivedirectionRecosignalD3 = new TH1F("h2CeffHBpositivedirectionRecosignalD3", "", nphi, 0., 72.);

  for (int jeta = 0; jeta < njeta; jeta++) {
    // positivedirectionRecosignalD:
    if (jeta - 41 >= 0) {
      //	     for (int i=0;i<ndepth;i++) {
      // depth=3
      for (int i = 2; i < 3; i++) {
        TH1F* HBpositivedirectionRecosignalD3 = (TH1F*)h2CeffHBpositivedirectionRecosignalD3->Clone("twod1");

        float ccctest = 0;  // to avoid empty massive elements
        for (int jphi = 0; jphi < nphi; jphi++) {
          double ccc1 = recosignalvarianceHB[i][jeta][jphi];
          if (arecosignalHB[i][jeta][jphi] > 0.) {
            HBpositivedirectionRecosignalD3->Fill(jphi, ccc1);
            ccctest = 1.;  //HBpositivedirectionRecosignalD3->SetBinError(i,0.01);
          }
        }  // for jphi
        if (ccctest > 0.) {
          //cout<<"1616       kcountHBpositivedirectionRecosignalD3   =     "<<kcountHBpositivedirectionRecosignalD3  <<"   jeta-41=     "<< jeta-41 <<endl;
          c3x5->cd(kcountHBpositivedirectionRecosignalD3);
          HBpositivedirectionRecosignalD3->SetMarkerStyle(20);
          HBpositivedirectionRecosignalD3->SetMarkerSize(0.4);
          HBpositivedirectionRecosignalD3->GetYaxis()->SetLabelSize(0.04);
          HBpositivedirectionRecosignalD3->SetXTitle("HBpositivedirectionRecosignalD3 \b");
          HBpositivedirectionRecosignalD3->SetMarkerColor(2);
          HBpositivedirectionRecosignalD3->SetLineColor(0);
          gPad->SetGridy();
          gPad->SetGridx();
          //	   gPad->SetLogy();
          if (kcountHBpositivedirectionRecosignalD3 == 1)
            HBpositivedirectionRecosignalD3->SetXTitle("D for HB+ jeta =  0; depth = 3 \b");
          if (kcountHBpositivedirectionRecosignalD3 == 2)
            HBpositivedirectionRecosignalD3->SetXTitle("D for HB+ jeta =  1; depth = 3 \b");
          if (kcountHBpositivedirectionRecosignalD3 == 3)
            HBpositivedirectionRecosignalD3->SetXTitle("D for HB+ jeta =  2; depth = 3 \b");
          if (kcountHBpositivedirectionRecosignalD3 == 4)
            HBpositivedirectionRecosignalD3->SetXTitle("D for HB+ jeta =  3; depth = 3 \b");
          if (kcountHBpositivedirectionRecosignalD3 == 5)
            HBpositivedirectionRecosignalD3->SetXTitle("D for HB+ jeta =  4; depth = 3 \b");
          if (kcountHBpositivedirectionRecosignalD3 == 6)
            HBpositivedirectionRecosignalD3->SetXTitle("D for HB+ jeta =  5; depth = 3 \b");
          if (kcountHBpositivedirectionRecosignalD3 == 7)
            HBpositivedirectionRecosignalD3->SetXTitle("D for HB+ jeta =  6; depth = 3 \b");
          if (kcountHBpositivedirectionRecosignalD3 == 8)
            HBpositivedirectionRecosignalD3->SetXTitle("D for HB+ jeta =  7; depth = 3 \b");
          if (kcountHBpositivedirectionRecosignalD3 == 9)
            HBpositivedirectionRecosignalD3->SetXTitle("D for HB+ jeta =  8; depth = 3 \b");
          if (kcountHBpositivedirectionRecosignalD3 == 10)
            HBpositivedirectionRecosignalD3->SetXTitle("D for HB+ jeta =  9; depth = 3 \b");
          if (kcountHBpositivedirectionRecosignalD3 == 11)
            HBpositivedirectionRecosignalD3->SetXTitle("D for HB+ jeta = 10; depth = 3 \b");
          if (kcountHBpositivedirectionRecosignalD3 == 12)
            HBpositivedirectionRecosignalD3->SetXTitle("D for HB+ jeta = 11; depth = 3 \b");
          if (kcountHBpositivedirectionRecosignalD3 == 13)
            HBpositivedirectionRecosignalD3->SetXTitle("D for HB+ jeta = 12; depth = 3 \b");
          if (kcountHBpositivedirectionRecosignalD3 == 14)
            HBpositivedirectionRecosignalD3->SetXTitle("D for HB+ jeta = 13; depth = 3 \b");
          if (kcountHBpositivedirectionRecosignalD3 == 15)
            HBpositivedirectionRecosignalD3->SetXTitle("D for HB+ jeta = 14; depth = 3 \b");
          if (kcountHBpositivedirectionRecosignalD3 == 16)
            HBpositivedirectionRecosignalD3->SetXTitle("D for HB+ jeta = 15; depth = 3 \b");
          HBpositivedirectionRecosignalD3->Draw("Error");
          kcountHBpositivedirectionRecosignalD3++;
          if (kcountHBpositivedirectionRecosignalD3 > 16)
            break;  // 4x6 = 24
        }           //ccctest>0

      }  // for i
    }    //if(jeta-41 >= 0)
  }      //for jeta
  /////////////////
  c3x5->Update();
  c3x5->Print("DrecosignalPositiveDirectionhistD1PhiSymmetryDepth3HB.png");
  c3x5->Clear();
  // clean-up
  if (h2CeffHBpositivedirectionRecosignalD3)
    delete h2CeffHBpositivedirectionRecosignalD3;
  //========================================================================================== 17
  //======================================================================
  //======================================================================1D plot: D vs phi , different eta,  depth=4
  //cout<<"      1D plot: D vs phi , different eta,  depth=4 *****" <<endl;
  c3x5->Clear();
  c3x5->Divide(4, 4);
  c3x5->cd(1);
  int kcountHBpositivedirectionRecosignalD4 = 1;
  TH1F* h2CeffHBpositivedirectionRecosignalD4 = new TH1F("h2CeffHBpositivedirectionRecosignalD4", "", nphi, 0., 72.);

  for (int jeta = 0; jeta < njeta; jeta++) {
    // positivedirectionRecosignalD:
    if (jeta - 41 >= 0) {
      //	     for (int i=0;i<ndepth;i++) {
      // depth=4
      for (int i = 3; i < 4; i++) {
        TH1F* HBpositivedirectionRecosignalD4 = (TH1F*)h2CeffHBpositivedirectionRecosignalD4->Clone("twod1");

        float ccctest = 0;  // to avoid empty massive elements
        for (int jphi = 0; jphi < nphi; jphi++) {
          double ccc1 = recosignalvarianceHB[i][jeta][jphi];
          if (arecosignalHB[i][jeta][jphi] > 0.) {
            HBpositivedirectionRecosignalD4->Fill(jphi, ccc1);
            ccctest = 1.;  //HBpositivedirectionRecosignalD4->SetBinError(i,0.01);
          }
        }  // for jphi
        if (ccctest > 0.) {
          //cout<<"1717       kcountHBpositivedirectionRecosignalD4   =     "<<kcountHBpositivedirectionRecosignalD4  <<"   jeta-41=     "<< jeta-41 <<endl;
          c3x5->cd(kcountHBpositivedirectionRecosignalD4);
          HBpositivedirectionRecosignalD4->SetMarkerStyle(20);
          HBpositivedirectionRecosignalD4->SetMarkerSize(0.4);
          HBpositivedirectionRecosignalD4->GetYaxis()->SetLabelSize(0.04);
          HBpositivedirectionRecosignalD4->SetXTitle("HBpositivedirectionRecosignalD4 \b");
          HBpositivedirectionRecosignalD4->SetMarkerColor(2);
          HBpositivedirectionRecosignalD4->SetLineColor(0);
          gPad->SetGridy();
          gPad->SetGridx();
          //	   gPad->SetLogy();
          if (kcountHBpositivedirectionRecosignalD4 == 1)
            HBpositivedirectionRecosignalD4->SetXTitle("D for HB+ jeta =  0; depth = 4 \b");
          if (kcountHBpositivedirectionRecosignalD4 == 2)
            HBpositivedirectionRecosignalD4->SetXTitle("D for HB+ jeta =  1; depth = 4 \b");
          if (kcountHBpositivedirectionRecosignalD4 == 3)
            HBpositivedirectionRecosignalD4->SetXTitle("D for HB+ jeta =  2; depth = 4 \b");
          if (kcountHBpositivedirectionRecosignalD4 == 4)
            HBpositivedirectionRecosignalD4->SetXTitle("D for HB+ jeta =  3; depth = 4 \b");
          if (kcountHBpositivedirectionRecosignalD4 == 5)
            HBpositivedirectionRecosignalD4->SetXTitle("D for HB+ jeta =  4; depth = 4 \b");
          if (kcountHBpositivedirectionRecosignalD4 == 6)
            HBpositivedirectionRecosignalD4->SetXTitle("D for HB+ jeta =  5; depth = 4 \b");
          if (kcountHBpositivedirectionRecosignalD4 == 7)
            HBpositivedirectionRecosignalD4->SetXTitle("D for HB+ jeta =  6; depth = 4 \b");
          if (kcountHBpositivedirectionRecosignalD4 == 8)
            HBpositivedirectionRecosignalD4->SetXTitle("D for HB+ jeta =  7; depth = 4 \b");
          if (kcountHBpositivedirectionRecosignalD4 == 9)
            HBpositivedirectionRecosignalD4->SetXTitle("D for HB+ jeta =  8; depth = 4 \b");
          if (kcountHBpositivedirectionRecosignalD4 == 10)
            HBpositivedirectionRecosignalD4->SetXTitle("D for HB+ jeta =  9; depth = 4 \b");
          if (kcountHBpositivedirectionRecosignalD4 == 11)
            HBpositivedirectionRecosignalD4->SetXTitle("D for HB+ jeta = 10; depth = 4 \b");
          if (kcountHBpositivedirectionRecosignalD4 == 12)
            HBpositivedirectionRecosignalD4->SetXTitle("D for HB+ jeta = 11; depth = 4 \b");
          if (kcountHBpositivedirectionRecosignalD4 == 13)
            HBpositivedirectionRecosignalD4->SetXTitle("D for HB+ jeta = 12; depth = 4 \b");
          if (kcountHBpositivedirectionRecosignalD4 == 14)
            HBpositivedirectionRecosignalD4->SetXTitle("D for HB+ jeta = 13; depth = 4 \b");
          if (kcountHBpositivedirectionRecosignalD4 == 15)
            HBpositivedirectionRecosignalD4->SetXTitle("D for HB+ jeta = 14; depth = 4 \b");
          if (kcountHBpositivedirectionRecosignalD4 == 16)
            HBpositivedirectionRecosignalD4->SetXTitle("D for HB+ jeta = 15; depth = 4 \b");
          HBpositivedirectionRecosignalD4->Draw("Error");
          kcountHBpositivedirectionRecosignalD4++;
          if (kcountHBpositivedirectionRecosignalD4 > 16)
            break;  // 4x6 = 24
        }           //ccctest>0

      }  // for i
    }    //if(jeta-41 >= 0)
  }      //for jeta
  /////////////////
  c3x5->Update();
  c3x5->Print("DrecosignalPositiveDirectionhistD1PhiSymmetryDepth4HB.png");
  c3x5->Clear();
  // clean-up
  if (h2CeffHBpositivedirectionRecosignalD4)
    delete h2CeffHBpositivedirectionRecosignalD4;

  //========================================================================================== 22214
  //======================================================================
  //======================================================================1D plot: D vs phi , different eta,  depth=1
  //cout<<"      1D plot: D vs phi , different eta,  depth=1 *****" <<endl;
  c3x5->Clear();
  /////////////////
  c3x5->Divide(4, 4);
  c3x5->cd(1);
  int kcountHBnegativedirectionRecosignalD1 = 1;
  TH1F* h2CeffHBnegativedirectionRecosignalD1 = new TH1F("h2CeffHBnegativedirectionRecosignalD1", "", nphi, 0., 72.);

  for (int jeta = 0; jeta < njeta; jeta++) {
    // negativedirectionRecosignalD:
    if (jeta - 41 < 0) {
      //	     for (int i=0;i<ndepth;i++) {
      // depth=1
      for (int i = 0; i < 1; i++) {
        TH1F* HBnegativedirectionRecosignalD1 = (TH1F*)h2CeffHBnegativedirectionRecosignalD1->Clone("twod1");

        float ccctest = 0;  // to avoid empty massive elements
        for (int jphi = 0; jphi < nphi; jphi++) {
          double ccc1 = recosignalvarianceHB[i][jeta][jphi];
          if (arecosignalHB[i][jeta][jphi] > 0.) {
            HBnegativedirectionRecosignalD1->Fill(jphi, ccc1);
            ccctest = 1.;  //HBnegativedirectionRecosignalD1->SetBinError(i,0.01);
          }
        }  // for jphi
        if (ccctest > 0.) {
          //cout<<"1414       kcountHBnegativedirectionRecosignalD1   =     "<<kcountHBnegativedirectionRecosignalD1  <<"   jeta-41=     "<< jeta-41 <<endl;
          c3x5->cd(kcountHBnegativedirectionRecosignalD1);
          HBnegativedirectionRecosignalD1->SetMarkerStyle(20);
          HBnegativedirectionRecosignalD1->SetMarkerSize(0.4);
          HBnegativedirectionRecosignalD1->GetYaxis()->SetLabelSize(0.04);
          HBnegativedirectionRecosignalD1->SetXTitle("HBnegativedirectionRecosignalD1 \b");
          HBnegativedirectionRecosignalD1->SetMarkerColor(2);
          HBnegativedirectionRecosignalD1->SetLineColor(0);
          gPad->SetGridy();
          gPad->SetGridx();
          //	   gPad->SetLogy();
          if (kcountHBnegativedirectionRecosignalD1 == 1)
            HBnegativedirectionRecosignalD1->SetXTitle("D for HB- jeta = -16; depth = 1 \b");
          if (kcountHBnegativedirectionRecosignalD1 == 2)
            HBnegativedirectionRecosignalD1->SetXTitle("D for HB- jeta = -15; depth = 1 \b");
          if (kcountHBnegativedirectionRecosignalD1 == 3)
            HBnegativedirectionRecosignalD1->SetXTitle("D for HB- jeta = -14; depth = 1 \b");
          if (kcountHBnegativedirectionRecosignalD1 == 4)
            HBnegativedirectionRecosignalD1->SetXTitle("D for HB- jeta = -13; depth = 1 \b");
          if (kcountHBnegativedirectionRecosignalD1 == 5)
            HBnegativedirectionRecosignalD1->SetXTitle("D for HB- jeta = -12; depth = 1 \b");
          if (kcountHBnegativedirectionRecosignalD1 == 6)
            HBnegativedirectionRecosignalD1->SetXTitle("D for HB- jeta = -11; depth = 1 \b");
          if (kcountHBnegativedirectionRecosignalD1 == 7)
            HBnegativedirectionRecosignalD1->SetXTitle("D for HB- jeta = -10; depth = 1 \b");
          if (kcountHBnegativedirectionRecosignalD1 == 8)
            HBnegativedirectionRecosignalD1->SetXTitle("D for HB- jeta =  -9; depth = 1 \b");
          if (kcountHBnegativedirectionRecosignalD1 == 9)
            HBnegativedirectionRecosignalD1->SetXTitle("D for HB- jeta =  -8; depth = 1 \b");
          if (kcountHBnegativedirectionRecosignalD1 == 10)
            HBnegativedirectionRecosignalD1->SetXTitle("D for HB- jeta =  -7; depth = 1 \b");
          if (kcountHBnegativedirectionRecosignalD1 == 11)
            HBnegativedirectionRecosignalD1->SetXTitle("D for HB- jeta =  -6; depth = 1 \b");
          if (kcountHBnegativedirectionRecosignalD1 == 12)
            HBnegativedirectionRecosignalD1->SetXTitle("D for HB- jeta =  -5; depth = 1 \b");
          if (kcountHBnegativedirectionRecosignalD1 == 13)
            HBnegativedirectionRecosignalD1->SetXTitle("D for HB- jeta =  -4; depth = 1 \b");
          if (kcountHBnegativedirectionRecosignalD1 == 14)
            HBnegativedirectionRecosignalD1->SetXTitle("D for HB- jeta =  -3; depth = 1 \b");
          if (kcountHBnegativedirectionRecosignalD1 == 15)
            HBnegativedirectionRecosignalD1->SetXTitle("D for HB- jeta =  -2; depth = 1 \b");
          if (kcountHBnegativedirectionRecosignalD1 == 16)
            HBnegativedirectionRecosignalD1->SetXTitle("D for HB- jeta =  -1; depth = 1 \b");
          HBnegativedirectionRecosignalD1->Draw("Error");
          kcountHBnegativedirectionRecosignalD1++;
          if (kcountHBnegativedirectionRecosignalD1 > 16)
            break;  // 4x6 = 24
        }           //ccctest>0

      }  // for i
    }    //if(jeta-41 < 0)
  }      //for jeta
  /////////////////
  c3x5->Update();
  c3x5->Print("DrecosignalNegativeDirectionhistD1PhiSymmetryDepth1HB.png");
  c3x5->Clear();
  // clean-up
  if (h2CeffHBnegativedirectionRecosignalD1)
    delete h2CeffHBnegativedirectionRecosignalD1;
  //========================================================================================== 22215
  //======================================================================
  //======================================================================1D plot: D vs phi , different eta,  depth=2
  //cout<<"      1D plot: D vs phi , different eta,  depth=2 *****" <<endl;
  c3x5->Clear();
  c3x5->Divide(4, 4);
  c3x5->cd(1);
  int kcountHBnegativedirectionRecosignalD2 = 1;
  TH1F* h2CeffHBnegativedirectionRecosignalD2 = new TH1F("h2CeffHBnegativedirectionRecosignalD2", "", nphi, 0., 72.);

  for (int jeta = 0; jeta < njeta; jeta++) {
    // negativedirectionRecosignalD:
    if (jeta - 41 < 0) {
      //	     for (int i=0;i<ndepth;i++) {
      // depth=2
      for (int i = 1; i < 2; i++) {
        TH1F* HBnegativedirectionRecosignalD2 = (TH1F*)h2CeffHBnegativedirectionRecosignalD2->Clone("twod1");

        float ccctest = 0;  // to avoid empty massive elements
        for (int jphi = 0; jphi < nphi; jphi++) {
          double ccc1 = recosignalvarianceHB[i][jeta][jphi];
          if (arecosignalHB[i][jeta][jphi] > 0.) {
            HBnegativedirectionRecosignalD2->Fill(jphi, ccc1);
            ccctest = 1.;  //HBnegativedirectionRecosignalD2->SetBinError(i,0.01);
          }
        }  // for jphi
        if (ccctest > 0.) {
          //cout<<"1515       kcountHBnegativedirectionRecosignalD2   =     "<<kcountHBnegativedirectionRecosignalD2  <<"   jeta-41=     "<< jeta-41 <<endl;
          c3x5->cd(kcountHBnegativedirectionRecosignalD2);
          HBnegativedirectionRecosignalD2->SetMarkerStyle(20);
          HBnegativedirectionRecosignalD2->SetMarkerSize(0.4);
          HBnegativedirectionRecosignalD2->GetYaxis()->SetLabelSize(0.04);
          HBnegativedirectionRecosignalD2->SetXTitle("HBnegativedirectionRecosignalD2 \b");
          HBnegativedirectionRecosignalD2->SetMarkerColor(2);
          HBnegativedirectionRecosignalD2->SetLineColor(0);
          gPad->SetGridy();
          gPad->SetGridx();
          //	   gPad->SetLogy();
          if (kcountHBnegativedirectionRecosignalD2 == 1)
            HBnegativedirectionRecosignalD2->SetXTitle("D for HB- jeta =-16; depth = 2 \b");
          if (kcountHBnegativedirectionRecosignalD2 == 2)
            HBnegativedirectionRecosignalD2->SetXTitle("D for HB- jeta =-15; depth = 2 \b");
          if (kcountHBnegativedirectionRecosignalD2 == 3)
            HBnegativedirectionRecosignalD2->SetXTitle("D for HB- jeta =-14; depth = 2 \b");
          if (kcountHBnegativedirectionRecosignalD2 == 4)
            HBnegativedirectionRecosignalD2->SetXTitle("D for HB- jeta =-13; depth = 2 \b");
          if (kcountHBnegativedirectionRecosignalD2 == 5)
            HBnegativedirectionRecosignalD2->SetXTitle("D for HB- jeta =-12; depth = 2 \b");
          if (kcountHBnegativedirectionRecosignalD2 == 6)
            HBnegativedirectionRecosignalD2->SetXTitle("D for HB- jeta =-11; depth = 2 \b");
          if (kcountHBnegativedirectionRecosignalD2 == 7)
            HBnegativedirectionRecosignalD2->SetXTitle("D for HB- jeta =-10; depth = 2 \b");
          if (kcountHBnegativedirectionRecosignalD2 == 8)
            HBnegativedirectionRecosignalD2->SetXTitle("D for HB- jeta =-9 ; depth = 2 \b");
          if (kcountHBnegativedirectionRecosignalD2 == 9)
            HBnegativedirectionRecosignalD2->SetXTitle("D for HB- jeta =-8 ; depth = 2 \b");
          if (kcountHBnegativedirectionRecosignalD2 == 10)
            HBnegativedirectionRecosignalD2->SetXTitle("D for HB- jeta =-7 ; depth = 2 \b");
          if (kcountHBnegativedirectionRecosignalD2 == 11)
            HBnegativedirectionRecosignalD2->SetXTitle("D for HB- jeta =-6 ; depth = 2 \b");
          if (kcountHBnegativedirectionRecosignalD2 == 12)
            HBnegativedirectionRecosignalD2->SetXTitle("D for HB- jeta =-5 ; depth = 2 \b");
          if (kcountHBnegativedirectionRecosignalD2 == 13)
            HBnegativedirectionRecosignalD2->SetXTitle("D for HB- jeta =-4 ; depth = 2 \b");
          if (kcountHBnegativedirectionRecosignalD2 == 14)
            HBnegativedirectionRecosignalD2->SetXTitle("D for HB- jeta =-3 ; depth = 2 \b");
          if (kcountHBnegativedirectionRecosignalD2 == 15)
            HBnegativedirectionRecosignalD2->SetXTitle("D for HB- jeta =-2 ; depth = 2 \b");
          if (kcountHBnegativedirectionRecosignalD2 == 16)
            HBnegativedirectionRecosignalD2->SetXTitle("D for HB- jeta =-1 ; depth = 2 \b");
          HBnegativedirectionRecosignalD2->Draw("Error");
          kcountHBnegativedirectionRecosignalD2++;
          if (kcountHBnegativedirectionRecosignalD2 > 16)
            break;  // 4x6 = 24
        }           //ccctest>0

      }  // for i
    }    //if(jeta-41 < 0)
  }      //for jeta
  /////////////////
  c3x5->Update();
  c3x5->Print("DrecosignalNegativeDirectionhistD1PhiSymmetryDepth2HB.png");
  c3x5->Clear();
  // clean-up
  if (h2CeffHBnegativedirectionRecosignalD2)
    delete h2CeffHBnegativedirectionRecosignalD2;
  //========================================================================================== 22216
  //======================================================================
  //======================================================================1D plot: D vs phi , different eta,  depth=3
  //cout<<"      1D plot: D vs phi , different eta,  depth=3 *****" <<endl;
  c3x5->Clear();
  c3x5->Divide(4, 4);
  c3x5->cd(1);
  int kcountHBnegativedirectionRecosignalD3 = 1;
  TH1F* h2CeffHBnegativedirectionRecosignalD3 = new TH1F("h2CeffHBnegativedirectionRecosignalD3", "", nphi, 0., 72.);

  for (int jeta = 0; jeta < njeta; jeta++) {
    // negativedirectionRecosignalD:
    if (jeta - 41 < 0) {
      //	     for (int i=0;i<ndepth;i++) {
      // depth=3
      for (int i = 2; i < 3; i++) {
        TH1F* HBnegativedirectionRecosignalD3 = (TH1F*)h2CeffHBnegativedirectionRecosignalD3->Clone("twod1");

        float ccctest = 0;  // to avoid empty massive elements
        for (int jphi = 0; jphi < nphi; jphi++) {
          double ccc1 = recosignalvarianceHB[i][jeta][jphi];
          if (arecosignalHB[i][jeta][jphi] > 0.) {
            HBnegativedirectionRecosignalD3->Fill(jphi, ccc1);
            ccctest = 1.;  //HBnegativedirectionRecosignalD3->SetBinError(i,0.01);
          }
        }  // for jphi
        if (ccctest > 0.) {
          //cout<<"1616       kcountHBnegativedirectionRecosignalD3   =     "<<kcountHBnegativedirectionRecosignalD3  <<"   jeta-41=     "<< jeta-41 <<endl;
          c3x5->cd(kcountHBnegativedirectionRecosignalD3);
          HBnegativedirectionRecosignalD3->SetMarkerStyle(20);
          HBnegativedirectionRecosignalD3->SetMarkerSize(0.4);
          HBnegativedirectionRecosignalD3->GetYaxis()->SetLabelSize(0.04);
          HBnegativedirectionRecosignalD3->SetXTitle("HBnegativedirectionRecosignalD3 \b");
          HBnegativedirectionRecosignalD3->SetMarkerColor(2);
          HBnegativedirectionRecosignalD3->SetLineColor(0);
          gPad->SetGridy();
          gPad->SetGridx();
          //	   gPad->SetLogy();
          if (kcountHBnegativedirectionRecosignalD3 == 1)
            HBnegativedirectionRecosignalD3->SetXTitle("D for HB- jeta =-16; depth = 3 \b");
          if (kcountHBnegativedirectionRecosignalD3 == 2)
            HBnegativedirectionRecosignalD3->SetXTitle("D for HB- jeta =-15; depth = 3 \b");
          if (kcountHBnegativedirectionRecosignalD3 == 3)
            HBnegativedirectionRecosignalD3->SetXTitle("D for HB- jeta =-14; depth = 3 \b");
          if (kcountHBnegativedirectionRecosignalD3 == 4)
            HBnegativedirectionRecosignalD3->SetXTitle("D for HB- jeta =-13; depth = 3 \b");
          if (kcountHBnegativedirectionRecosignalD3 == 5)
            HBnegativedirectionRecosignalD3->SetXTitle("D for HB- jeta =-12; depth = 3 \b");
          if (kcountHBnegativedirectionRecosignalD3 == 6)
            HBnegativedirectionRecosignalD3->SetXTitle("D for HB- jeta =-11; depth = 3 \b");
          if (kcountHBnegativedirectionRecosignalD3 == 7)
            HBnegativedirectionRecosignalD3->SetXTitle("D for HB- jeta =-10; depth = 3 \b");
          if (kcountHBnegativedirectionRecosignalD3 == 8)
            HBnegativedirectionRecosignalD3->SetXTitle("D for HB- jeta =-9 ; depth = 3 \b");
          if (kcountHBnegativedirectionRecosignalD3 == 9)
            HBnegativedirectionRecosignalD3->SetXTitle("D for HB- jeta =-8 ; depth = 3 \b");
          if (kcountHBnegativedirectionRecosignalD3 == 10)
            HBnegativedirectionRecosignalD3->SetXTitle("D for HB- jeta =-7 ; depth = 3 \b");
          if (kcountHBnegativedirectionRecosignalD3 == 11)
            HBnegativedirectionRecosignalD3->SetXTitle("D for HB- jeta =-6 ; depth = 3 \b");
          if (kcountHBnegativedirectionRecosignalD3 == 12)
            HBnegativedirectionRecosignalD3->SetXTitle("D for HB- jeta =-5 ; depth = 3 \b");
          if (kcountHBnegativedirectionRecosignalD3 == 13)
            HBnegativedirectionRecosignalD3->SetXTitle("D for HB- jeta =-4 ; depth = 3 \b");
          if (kcountHBnegativedirectionRecosignalD3 == 14)
            HBnegativedirectionRecosignalD3->SetXTitle("D for HB- jeta =-3 ; depth = 3 \b");
          if (kcountHBnegativedirectionRecosignalD3 == 15)
            HBnegativedirectionRecosignalD3->SetXTitle("D for HB- jeta =-2 ; depth = 3 \b");
          if (kcountHBnegativedirectionRecosignalD3 == 16)
            HBnegativedirectionRecosignalD3->SetXTitle("D for HB- jeta =-1 ; depth = 3 \b");
          HBnegativedirectionRecosignalD3->Draw("Error");
          kcountHBnegativedirectionRecosignalD3++;
          if (kcountHBnegativedirectionRecosignalD3 > 16)
            break;  // 4x6 = 24
        }           //ccctest>0

      }  // for i
    }    //if(jeta-41 < 0)
  }      //for jeta
  /////////////////
  c3x5->Update();
  c3x5->Print("DrecosignalNegativeDirectionhistD1PhiSymmetryDepth3HB.png");
  c3x5->Clear();
  // clean-up
  if (h2CeffHBnegativedirectionRecosignalD3)
    delete h2CeffHBnegativedirectionRecosignalD3;
  //========================================================================================== 22217
  //======================================================================
  //======================================================================1D plot: D vs phi , different eta,  depth=4
  //cout<<"      1D plot: D vs phi , different eta,  depth=4 *****" <<endl;
  c3x5->Clear();
  c3x5->Divide(4, 4);
  c3x5->cd(1);
  int kcountHBnegativedirectionRecosignalD4 = 1;
  TH1F* h2CeffHBnegativedirectionRecosignalD4 = new TH1F("h2CeffHBnegativedirectionRecosignalD4", "", nphi, 0., 72.);

  for (int jeta = 0; jeta < njeta; jeta++) {
    // negativedirectionRecosignalD:
    if (jeta - 41 < 0) {
      //	     for (int i=0;i<ndepth;i++) {
      // depth=4
      for (int i = 3; i < 4; i++) {
        TH1F* HBnegativedirectionRecosignalD4 = (TH1F*)h2CeffHBnegativedirectionRecosignalD4->Clone("twod1");

        float ccctest = 0;  // to avoid empty massive elements
        for (int jphi = 0; jphi < nphi; jphi++) {
          double ccc1 = recosignalvarianceHB[i][jeta][jphi];
          if (arecosignalHB[i][jeta][jphi] > 0.) {
            HBnegativedirectionRecosignalD4->Fill(jphi, ccc1);
            ccctest = 1.;  //HBnegativedirectionRecosignalD4->SetBinError(i,0.01);
          }
        }  // for jphi
        if (ccctest > 0.) {
          //cout<<"1717       kcountHBnegativedirectionRecosignalD4   =     "<<kcountHBnegativedirectionRecosignalD4  <<"   jeta-41=     "<< jeta-41 <<endl;
          c3x5->cd(kcountHBnegativedirectionRecosignalD4);
          HBnegativedirectionRecosignalD4->SetMarkerStyle(20);
          HBnegativedirectionRecosignalD4->SetMarkerSize(0.4);
          HBnegativedirectionRecosignalD4->GetYaxis()->SetLabelSize(0.04);
          HBnegativedirectionRecosignalD4->SetXTitle("HBnegativedirectionRecosignalD4 \b");
          HBnegativedirectionRecosignalD4->SetMarkerColor(2);
          HBnegativedirectionRecosignalD4->SetLineColor(0);
          gPad->SetGridy();
          gPad->SetGridx();
          //	   gPad->SetLogy();
          if (kcountHBnegativedirectionRecosignalD4 == 1)
            HBnegativedirectionRecosignalD4->SetXTitle("D for HB- jeta =-16; depth = 4 \b");
          if (kcountHBnegativedirectionRecosignalD4 == 2)
            HBnegativedirectionRecosignalD4->SetXTitle("D for HB- jeta =-15; depth = 4 \b");
          if (kcountHBnegativedirectionRecosignalD4 == 3)
            HBnegativedirectionRecosignalD4->SetXTitle("D for HB- jeta =-14; depth = 4 \b");
          if (kcountHBnegativedirectionRecosignalD4 == 4)
            HBnegativedirectionRecosignalD4->SetXTitle("D for HB- jeta =-13; depth = 4 \b");
          if (kcountHBnegativedirectionRecosignalD4 == 5)
            HBnegativedirectionRecosignalD4->SetXTitle("D for HB- jeta =-12; depth = 4 \b");
          if (kcountHBnegativedirectionRecosignalD4 == 6)
            HBnegativedirectionRecosignalD4->SetXTitle("D for HB- jeta =-11; depth = 4 \b");
          if (kcountHBnegativedirectionRecosignalD4 == 7)
            HBnegativedirectionRecosignalD4->SetXTitle("D for HB- jeta =-10; depth = 4 \b");
          if (kcountHBnegativedirectionRecosignalD4 == 8)
            HBnegativedirectionRecosignalD4->SetXTitle("D for HB- jeta =-9 ; depth = 4 \b");
          if (kcountHBnegativedirectionRecosignalD4 == 9)
            HBnegativedirectionRecosignalD4->SetXTitle("D for HB- jeta =-8 ; depth = 4 \b");
          if (kcountHBnegativedirectionRecosignalD4 == 10)
            HBnegativedirectionRecosignalD4->SetXTitle("D for HB- jeta =-7 ; depth = 4 \b");
          if (kcountHBnegativedirectionRecosignalD4 == 11)
            HBnegativedirectionRecosignalD4->SetXTitle("D for HB- jeta =-6 ; depth = 4 \b");
          if (kcountHBnegativedirectionRecosignalD4 == 12)
            HBnegativedirectionRecosignalD4->SetXTitle("D for HB- jeta =-5 ; depth = 4 \b");
          if (kcountHBnegativedirectionRecosignalD4 == 13)
            HBnegativedirectionRecosignalD4->SetXTitle("D for HB- jeta =-4 ; depth = 4 \b");
          if (kcountHBnegativedirectionRecosignalD4 == 14)
            HBnegativedirectionRecosignalD4->SetXTitle("D for HB- jeta =-3 ; depth = 4 \b");
          if (kcountHBnegativedirectionRecosignalD4 == 15)
            HBnegativedirectionRecosignalD4->SetXTitle("D for HB- jeta =-2 ; depth = 4 \b");
          if (kcountHBnegativedirectionRecosignalD4 == 16)
            HBnegativedirectionRecosignalD4->SetXTitle("D for HB- jeta =-1 ; depth = 4 \b");
          HBnegativedirectionRecosignalD4->Draw("Error");
          kcountHBnegativedirectionRecosignalD4++;
          if (kcountHBnegativedirectionRecosignalD4 > 16)
            break;  // 4x6 = 24
        }           //ccctest>0

      }  // for i
    }    //if(jeta-41 < 0)
  }      //for jeta
  /////////////////
  c3x5->Update();
  c3x5->Print("DrecosignalNegativeDirectionhistD1PhiSymmetryDepth4HB.png");
  c3x5->Clear();
  // clean-up
  if (h2CeffHBnegativedirectionRecosignalD4)
    delete h2CeffHBnegativedirectionRecosignalD4;

  //=====================================================================       END of Recosignal HB for phi-symmetry
  //=====================================================================       END of Recosignal HB for phi-symmetry
  //=====================================================================       END of Recosignal HB for phi-symmetry

  ////////////////////////////////////////////////////////////////////////////////////////////////////         Phi-symmetry for Calibration Group:    Recosignal HE
  ////////////////////////////////////////////////////////////////////////////////////////////////////         Phi-symmetry for Calibration Group:    Recosignal HE
  ////////////////////////////////////////////////////////////////////////////////////////////////////         Phi-symmetry for Calibration Group:    Recosignal HE
  //  int k_max[5]={0,4,7,4,4}; // maximum depth for each subdet
  //ndepth = k_max[3];
  ndepth = 7;
  //  const int ndepth = 7;
  double arecosignalhe[ndepth][njeta][njphi];
  double recosignalvariancehe[ndepth][njeta][njphi];
  //                                   RRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRR:   Recosignal HE
  TH2F* recSignalEnergy1HE1 = (TH2F*)dir->FindObjectAny("h_recSignalEnergy1_HE1");
  TH2F* recSignalEnergy0HE1 = (TH2F*)dir->FindObjectAny("h_recSignalEnergy0_HE1");
  TH2F* recSignalEnergyHE1 = (TH2F*)recSignalEnergy1HE1->Clone("recSignalEnergyHE1");
  recSignalEnergyHE1->Divide(recSignalEnergy1HE1, recSignalEnergy0HE1, 1, 1, "B");
  TH2F* recSignalEnergy1HE2 = (TH2F*)dir->FindObjectAny("h_recSignalEnergy1_HE2");
  TH2F* recSignalEnergy0HE2 = (TH2F*)dir->FindObjectAny("h_recSignalEnergy0_HE2");
  TH2F* recSignalEnergyHE2 = (TH2F*)recSignalEnergy1HE2->Clone("recSignalEnergyHE2");
  recSignalEnergyHE2->Divide(recSignalEnergy1HE2, recSignalEnergy0HE2, 1, 1, "B");
  TH2F* recSignalEnergy1HE3 = (TH2F*)dir->FindObjectAny("h_recSignalEnergy1_HE3");
  TH2F* recSignalEnergy0HE3 = (TH2F*)dir->FindObjectAny("h_recSignalEnergy0_HE3");
  TH2F* recSignalEnergyHE3 = (TH2F*)recSignalEnergy1HE3->Clone("recSignalEnergyHE3");
  recSignalEnergyHE3->Divide(recSignalEnergy1HE3, recSignalEnergy0HE3, 1, 1, "B");
  TH2F* recSignalEnergy1HE4 = (TH2F*)dir->FindObjectAny("h_recSignalEnergy1_HE4");
  TH2F* recSignalEnergy0HE4 = (TH2F*)dir->FindObjectAny("h_recSignalEnergy0_HE4");
  TH2F* recSignalEnergyHE4 = (TH2F*)recSignalEnergy1HE4->Clone("recSignalEnergyHE4");
  recSignalEnergyHE4->Divide(recSignalEnergy1HE4, recSignalEnergy0HE4, 1, 1, "B");
  TH2F* recSignalEnergy1HE5 = (TH2F*)dir->FindObjectAny("h_recSignalEnergy1_HE5");
  TH2F* recSignalEnergy0HE5 = (TH2F*)dir->FindObjectAny("h_recSignalEnergy0_HE5");
  TH2F* recSignalEnergyHE5 = (TH2F*)recSignalEnergy1HE5->Clone("recSignalEnergyHE5");
  recSignalEnergyHE5->Divide(recSignalEnergy1HE5, recSignalEnergy0HE5, 1, 1, "B");
  TH2F* recSignalEnergy1HE6 = (TH2F*)dir->FindObjectAny("h_recSignalEnergy1_HE6");
  TH2F* recSignalEnergy0HE6 = (TH2F*)dir->FindObjectAny("h_recSignalEnergy0_HE6");
  TH2F* recSignalEnergyHE6 = (TH2F*)recSignalEnergy1HE6->Clone("recSignalEnergyHE6");
  recSignalEnergyHE6->Divide(recSignalEnergy1HE6, recSignalEnergy0HE6, 1, 1, "B");
  TH2F* recSignalEnergy1HE7 = (TH2F*)dir->FindObjectAny("h_recSignalEnergy1_HE7");
  TH2F* recSignalEnergy0HE7 = (TH2F*)dir->FindObjectAny("h_recSignalEnergy0_HE7");
  TH2F* recSignalEnergyHE7 = (TH2F*)recSignalEnergy1HE7->Clone("recSignalEnergyHE7");
  recSignalEnergyHE7->Divide(recSignalEnergy1HE7, recSignalEnergy0HE7, 1, 1, "B");
  for (int jeta = 0; jeta < njeta; jeta++) {
    //====================================================================== PHI normalization & put R into massive arecosignalhe
    //preparation for PHI normalization:
    double sumrecosignalHE0 = 0;
    int nsumrecosignalHE0 = 0;
    double sumrecosignalHE1 = 0;
    int nsumrecosignalHE1 = 0;
    double sumrecosignalHE2 = 0;
    int nsumrecosignalHE2 = 0;
    double sumrecosignalHE3 = 0;
    int nsumrecosignalHE3 = 0;
    double sumrecosignalHE4 = 0;
    int nsumrecosignalHE4 = 0;
    double sumrecosignalHE5 = 0;
    int nsumrecosignalHE5 = 0;
    double sumrecosignalHE6 = 0;
    int nsumrecosignalHE6 = 0;
    for (int jphi = 0; jphi < njphi; jphi++) {
      arecosignalhe[0][jeta][jphi] = recSignalEnergyHE1->GetBinContent(jeta + 1, jphi + 1);
      arecosignalhe[1][jeta][jphi] = recSignalEnergyHE2->GetBinContent(jeta + 1, jphi + 1);
      arecosignalhe[2][jeta][jphi] = recSignalEnergyHE3->GetBinContent(jeta + 1, jphi + 1);
      arecosignalhe[3][jeta][jphi] = recSignalEnergyHE4->GetBinContent(jeta + 1, jphi + 1);
      arecosignalhe[4][jeta][jphi] = recSignalEnergyHE5->GetBinContent(jeta + 1, jphi + 1);
      arecosignalhe[5][jeta][jphi] = recSignalEnergyHE6->GetBinContent(jeta + 1, jphi + 1);
      arecosignalhe[6][jeta][jphi] = recSignalEnergyHE7->GetBinContent(jeta + 1, jphi + 1);
      if (arecosignalhe[0][jeta][jphi] > 0.) {
        sumrecosignalHE0 += arecosignalhe[0][jeta][jphi];
        ++nsumrecosignalHE0;
      }
      if (arecosignalhe[1][jeta][jphi] > 0.) {
        sumrecosignalHE1 += arecosignalhe[1][jeta][jphi];
        ++nsumrecosignalHE1;
      }
      if (arecosignalhe[2][jeta][jphi] > 0.) {
        sumrecosignalHE2 += arecosignalhe[2][jeta][jphi];
        ++nsumrecosignalHE2;
      }
      if (arecosignalhe[3][jeta][jphi] > 0.) {
        sumrecosignalHE3 += arecosignalhe[3][jeta][jphi];
        ++nsumrecosignalHE3;
      }
      if (arecosignalhe[4][jeta][jphi] > 0.) {
        sumrecosignalHE4 += arecosignalhe[4][jeta][jphi];
        ++nsumrecosignalHE4;
      }
      if (arecosignalhe[5][jeta][jphi] > 0.) {
        sumrecosignalHE5 += arecosignalhe[5][jeta][jphi];
        ++nsumrecosignalHE5;
      }
      if (arecosignalhe[6][jeta][jphi] > 0.) {
        sumrecosignalHE6 += arecosignalhe[6][jeta][jphi];
        ++nsumrecosignalHE6;
      }
    }  // phi
    // PHI normalization:
    for (int jphi = 0; jphi < njphi; jphi++) {
      if (arecosignalhe[0][jeta][jphi] > 0.)
        arecosignalhe[0][jeta][jphi] /= (sumrecosignalHE0 / nsumrecosignalHE0);
      if (arecosignalhe[1][jeta][jphi] > 0.)
        arecosignalhe[1][jeta][jphi] /= (sumrecosignalHE1 / nsumrecosignalHE1);
      if (arecosignalhe[2][jeta][jphi] > 0.)
        arecosignalhe[2][jeta][jphi] /= (sumrecosignalHE2 / nsumrecosignalHE2);
      if (arecosignalhe[3][jeta][jphi] > 0.)
        arecosignalhe[3][jeta][jphi] /= (sumrecosignalHE3 / nsumrecosignalHE3);
      if (arecosignalhe[4][jeta][jphi] > 0.)
        arecosignalhe[4][jeta][jphi] /= (sumrecosignalHE4 / nsumrecosignalHE4);
      if (arecosignalhe[5][jeta][jphi] > 0.)
        arecosignalhe[5][jeta][jphi] /= (sumrecosignalHE5 / nsumrecosignalHE5);
      if (arecosignalhe[6][jeta][jphi] > 0.)
        arecosignalhe[6][jeta][jphi] /= (sumrecosignalHE6 / nsumrecosignalHE6);
    }  // phi
  }    //eta
  //------------------------  2D-eta/phi-plot: R, averaged over depthes
  //======================================================================
  //======================================================================
  //cout<<"      R2D-eta/phi-plot: R, averaged over depthes *****" <<endl;
  c2x1->Clear();
  /////////////////
  c2x1->Divide(2, 1);
  c2x1->cd(1);
  TH2F* GefzRrecosignalHE42D = new TH2F("GefzRrecosignalHE42D", "", neta, -41., 41., nphi, 0., 72.);
  TH2F* GefzRrecosignalHE42D0 = new TH2F("GefzRrecosignalHE42D0", "", neta, -41., 41., nphi, 0., 72.);
  TH2F* GefzRrecosignalHE42DF = (TH2F*)GefzRrecosignalHE42D0->Clone("GefzRrecosignalHE42DF");
  for (int i = 0; i < ndepth; i++) {
    for (int jeta = 0; jeta < neta; jeta++) {
      for (int jphi = 0; jphi < nphi; jphi++) {
        double ccc1 = arecosignalhe[i][jeta][jphi];
        int k2plot = jeta - 41;
        int kkk = k2plot;  //if(k2plot >0 ) kkk=k2plot+1; //-41 +41 !=0
        if (ccc1 != 0.) {
          GefzRrecosignalHE42D->Fill(kkk, jphi, ccc1);
          GefzRrecosignalHE42D0->Fill(kkk, jphi, 1.);
        }
      }
    }
  }
  GefzRrecosignalHE42DF->Divide(GefzRrecosignalHE42D, GefzRrecosignalHE42D0, 1, 1, "B");  // average A
  gPad->SetGridy();
  gPad->SetGridx();  //      gPad->SetLogz();
  GefzRrecosignalHE42DF->SetXTitle("<R>_depth       #eta  \b");
  GefzRrecosignalHE42DF->SetYTitle("      #phi \b");
  GefzRrecosignalHE42DF->Draw("COLZ");

  c2x1->cd(2);
  TH1F* energyhitSignal_HE = (TH1F*)dir->FindObjectAny("h_energyhitSignal_HE");
  energyhitSignal_HE->SetMarkerStyle(20);
  energyhitSignal_HE->SetMarkerSize(0.4);
  energyhitSignal_HE->GetYaxis()->SetLabelSize(0.04);
  energyhitSignal_HE->SetXTitle("energyhitSignal_HE \b");
  energyhitSignal_HE->SetMarkerColor(2);
  energyhitSignal_HE->SetLineColor(0);
  gPad->SetGridy();
  gPad->SetGridx();
  energyhitSignal_HE->Draw("Error");

  /////////////////
  c2x1->Update();
  c2x1->Print("RrecosignalGeneralD2PhiSymmetryHE.png");
  c2x1->Clear();
  // clean-up
  if (GefzRrecosignalHE42D)
    delete GefzRrecosignalHE42D;
  if (GefzRrecosignalHE42D0)
    delete GefzRrecosignalHE42D0;
  if (GefzRrecosignalHE42DF)
    delete GefzRrecosignalHE42DF;
  //====================================================================== 1D plot: R vs phi , averaged over depthes & eta
  //======================================================================
  //cout<<"      1D plot: R vs phi , averaged over depthes & eta *****" <<endl;
  c1x1->Clear();
  /////////////////
  c1x1->Divide(1, 1);
  c1x1->cd(1);
  TH1F* GefzRrecosignalHE41D = new TH1F("GefzRrecosignalHE41D", "", nphi, 0., 72.);
  TH1F* GefzRrecosignalHE41D0 = new TH1F("GefzRrecosignalHE41D0", "", nphi, 0., 72.);
  TH1F* GefzRrecosignalHE41DF = (TH1F*)GefzRrecosignalHE41D0->Clone("GefzRrecosignalHE41DF");
  for (int jphi = 0; jphi < nphi; jphi++) {
    for (int jeta = 0; jeta < neta; jeta++) {
      for (int i = 0; i < ndepth; i++) {
        double ccc1 = arecosignalhe[i][jeta][jphi];
        if (ccc1 != 0.) {
          GefzRrecosignalHE41D->Fill(jphi, ccc1);
          GefzRrecosignalHE41D0->Fill(jphi, 1.);
        }
      }
    }
  }
  GefzRrecosignalHE41DF->Divide(
      GefzRrecosignalHE41D, GefzRrecosignalHE41D0, 1, 1, "B");  // R averaged over depthes & eta
  GefzRrecosignalHE41D0->Sumw2();
  //    for (int jphi=1;jphi<73;jphi++) {GefzRrecosignalHE41DF->SetBinError(jphi,0.01);}
  gPad->SetGridy();
  gPad->SetGridx();  //      gPad->SetLogz();
  GefzRrecosignalHE41DF->SetMarkerStyle(20);
  GefzRrecosignalHE41DF->SetMarkerSize(1.4);
  GefzRrecosignalHE41DF->GetZaxis()->SetLabelSize(0.08);
  GefzRrecosignalHE41DF->SetXTitle("#phi  \b");
  GefzRrecosignalHE41DF->SetYTitle("  <R> \b");
  GefzRrecosignalHE41DF->SetZTitle("<R>_PHI  - AllDepthes \b");
  GefzRrecosignalHE41DF->SetMarkerColor(4);
  GefzRrecosignalHE41DF->SetLineColor(
      4);  //  GefzRrecosignalHE41DF->SetMinimum(0.8);     //      GefzRrecosignalHE41DF->SetMaximum(1.000);
  GefzRrecosignalHE41DF->Draw("Error");
  /////////////////
  c1x1->Update();
  c1x1->Print("RrecosignalGeneralD1PhiSymmetryHE.png");
  c1x1->Clear();
  // clean-up
  if (GefzRrecosignalHE41D)
    delete GefzRrecosignalHE41D;
  if (GefzRrecosignalHE41D0)
    delete GefzRrecosignalHE41D0;
  if (GefzRrecosignalHE41DF)
    delete GefzRrecosignalHE41DF;

  //========================================================================================== 4
  //======================================================================
  //======================================================================1D plot: R vs phi , different eta,  depth=1
  //cout<<"      1D plot: R vs phi , different eta,  depth=1 *****" <<endl;
  c3x5->Clear();
  /////////////////
  c3x5->Divide(3, 5);
  c3x5->cd(1);
  int kcountHEpositivedirectionRecosignal1 = 1;
  TH1F* h2CeffHEpositivedirectionRecosignal1 = new TH1F("h2CeffHEpositivedirectionRecosignal1", "", nphi, 0., 72.);
  for (int jeta = 0; jeta < njeta; jeta++) {
    // positivedirectionRecosignal:
    if (jeta - 41 >= 0) {
      //	     for (int i=0;i<ndepth;i++) {
      // depth=1
      for (int i = 0; i < 1; i++) {
        TH1F* HEpositivedirectionRecosignal1 = (TH1F*)h2CeffHEpositivedirectionRecosignal1->Clone("twod1");
        float ccctest = 0;  // to avoid empty massive elements
        for (int jphi = 0; jphi < nphi; jphi++) {
          double ccc1 = arecosignalhe[i][jeta][jphi];
          if (ccc1 != 0.) {
            HEpositivedirectionRecosignal1->Fill(jphi, ccc1);
            ccctest = 1.;  //HEpositivedirectionRecosignal1->SetBinError(i,0.01);
          }
        }  // for jphi
        if (ccctest > 0.) {
          //	  cout<<"444        kcountHEpositivedirectionRecosignal1   =     "<<kcountHEpositivedirectionRecosignal1  <<"   jeta-41=     "<< jeta-41 <<endl;
          c3x5->cd(kcountHEpositivedirectionRecosignal1);
          HEpositivedirectionRecosignal1->SetMarkerStyle(20);
          HEpositivedirectionRecosignal1->SetMarkerSize(0.4);
          HEpositivedirectionRecosignal1->GetYaxis()->SetLabelSize(0.04);
          HEpositivedirectionRecosignal1->SetXTitle("HEpositivedirectionRecosignal1 \b");
          HEpositivedirectionRecosignal1->SetMarkerColor(2);
          HEpositivedirectionRecosignal1->SetLineColor(0);
          gPad->SetGridy();
          gPad->SetGridx();
          //	   gPad->SetLogy();
          if (kcountHEpositivedirectionRecosignal1 == 1)
            HEpositivedirectionRecosignal1->SetXTitle("R for HE+ jeta = 17; depth = 1 \b");
          if (kcountHEpositivedirectionRecosignal1 == 2)
            HEpositivedirectionRecosignal1->SetXTitle("R for HE+ jeta = 18; depth = 1 \b");
          if (kcountHEpositivedirectionRecosignal1 == 3)
            HEpositivedirectionRecosignal1->SetXTitle("R for HE+ jeta = 19; depth = 1 \b");
          if (kcountHEpositivedirectionRecosignal1 == 4)
            HEpositivedirectionRecosignal1->SetXTitle("R for HE+ jeta = 20; depth = 1 \b");
          if (kcountHEpositivedirectionRecosignal1 == 5)
            HEpositivedirectionRecosignal1->SetXTitle("R for HE+ jeta = 21; depth = 1 \b");
          if (kcountHEpositivedirectionRecosignal1 == 6)
            HEpositivedirectionRecosignal1->SetXTitle("R for HE+ jeta = 22; depth = 1 \b");
          if (kcountHEpositivedirectionRecosignal1 == 7)
            HEpositivedirectionRecosignal1->SetXTitle("R for HE+ jeta = 23; depth = 1 \b");
          if (kcountHEpositivedirectionRecosignal1 == 8)
            HEpositivedirectionRecosignal1->SetXTitle("R for HE+ jeta = 24; depth = 1 \b");
          if (kcountHEpositivedirectionRecosignal1 == 9)
            HEpositivedirectionRecosignal1->SetXTitle("R for HE+ jeta = 25; depth = 1 \b");
          if (kcountHEpositivedirectionRecosignal1 == 10)
            HEpositivedirectionRecosignal1->SetXTitle("R for HE+ jeta = 26; depth = 1 \b");
          if (kcountHEpositivedirectionRecosignal1 == 11)
            HEpositivedirectionRecosignal1->SetXTitle("R for HE+ jeta = 27; depth = 1 \b");
          if (kcountHEpositivedirectionRecosignal1 == 12)
            HEpositivedirectionRecosignal1->SetXTitle("R for HE+ jeta = 28; depth = 1 \b");
          HEpositivedirectionRecosignal1->Draw("Error");
          kcountHEpositivedirectionRecosignal1++;
          if (kcountHEpositivedirectionRecosignal1 > 12)
            break;  // 4x6 = 24
        }           //ccctest>0

      }  // for i
    }    //if(jeta-41 >= 0)
  }      //for jeta
  /////////////////
  c3x5->Update();
  c3x5->Print("RrecosignalPositiveDirectionhistD1PhiSymmetryDepth1HE.png");
  c3x5->Clear();
  // clean-up
  if (h2CeffHEpositivedirectionRecosignal1)
    delete h2CeffHEpositivedirectionRecosignal1;

  //========================================================================================== 5
  //======================================================================
  //======================================================================1D plot: R vs phi , different eta,  depth=2
  //  cout<<"      1D plot: R vs phi , different eta,  depth=2 *****" <<endl;
  c3x5->Clear();
  /////////////////
  c3x5->Divide(3, 5);
  c3x5->cd(1);
  int kcountHEpositivedirectionRecosignal2 = 1;
  TH1F* h2CeffHEpositivedirectionRecosignal2 = new TH1F("h2CeffHEpositivedirectionRecosignal2", "", nphi, 0., 72.);
  for (int jeta = 0; jeta < njeta; jeta++) {
    // positivedirectionRecosignal:
    if (jeta - 41 >= 0) {
      //	     for (int i=0;i<ndepth;i++) {
      // depth=2
      for (int i = 1; i < 2; i++) {
        TH1F* HEpositivedirectionRecosignal2 = (TH1F*)h2CeffHEpositivedirectionRecosignal2->Clone("twod1");
        float ccctest = 0;  // to avoid empty massive elements
        for (int jphi = 0; jphi < nphi; jphi++) {
          double ccc1 = arecosignalhe[i][jeta][jphi];
          if (ccc1 != 0.) {
            HEpositivedirectionRecosignal2->Fill(jphi, ccc1);
            ccctest = 1.;  //HEpositivedirectionRecosignal2->SetBinError(i,0.01);
          }
        }  // for jphi
        if (ccctest > 0.) {
          //cout<<"555        kcountHEpositivedirectionRecosignal2   =     "<<kcountHEpositivedirectionRecosignal2  <<"   jeta-41=     "<< jeta-41 <<endl;
          c3x5->cd(kcountHEpositivedirectionRecosignal2);
          HEpositivedirectionRecosignal2->SetMarkerStyle(20);
          HEpositivedirectionRecosignal2->SetMarkerSize(0.4);
          HEpositivedirectionRecosignal2->GetYaxis()->SetLabelSize(0.04);
          HEpositivedirectionRecosignal2->SetXTitle("HEpositivedirectionRecosignal2 \b");
          HEpositivedirectionRecosignal2->SetMarkerColor(2);
          HEpositivedirectionRecosignal2->SetLineColor(0);
          gPad->SetGridy();
          gPad->SetGridx();
          //	   gPad->SetLogy();
          if (kcountHEpositivedirectionRecosignal2 == 1)
            HEpositivedirectionRecosignal2->SetXTitle("R for HE+ jeta = 16; depth = 2 \b");
          if (kcountHEpositivedirectionRecosignal2 == 2)
            HEpositivedirectionRecosignal2->SetXTitle("R for HE+ jeta = 17; depth = 2 \b");
          if (kcountHEpositivedirectionRecosignal2 == 3)
            HEpositivedirectionRecosignal2->SetXTitle("R for HE+ jeta = 18; depth = 2 \b");
          if (kcountHEpositivedirectionRecosignal2 == 4)
            HEpositivedirectionRecosignal2->SetXTitle("R for HE+ jeta = 19; depth = 2 \b");
          if (kcountHEpositivedirectionRecosignal2 == 5)
            HEpositivedirectionRecosignal2->SetXTitle("R for HE+ jeta = 20; depth = 2 \b");
          if (kcountHEpositivedirectionRecosignal2 == 6)
            HEpositivedirectionRecosignal2->SetXTitle("R for HE+ jeta = 21; depth = 2 \b");
          if (kcountHEpositivedirectionRecosignal2 == 7)
            HEpositivedirectionRecosignal2->SetXTitle("R for HE+ jeta = 22; depth = 2 \b");
          if (kcountHEpositivedirectionRecosignal2 == 8)
            HEpositivedirectionRecosignal2->SetXTitle("R for HE+ jeta = 23; depth = 2 \b");
          if (kcountHEpositivedirectionRecosignal2 == 9)
            HEpositivedirectionRecosignal2->SetXTitle("R for HE+ jeta = 24; depth = 2 \b");
          if (kcountHEpositivedirectionRecosignal2 == 10)
            HEpositivedirectionRecosignal2->SetXTitle("R for HE+ jeta = 25; depth = 2 \b");
          if (kcountHEpositivedirectionRecosignal2 == 11)
            HEpositivedirectionRecosignal2->SetXTitle("R for HE+ jeta = 26; depth = 2 \b");
          if (kcountHEpositivedirectionRecosignal2 == 12)
            HEpositivedirectionRecosignal2->SetXTitle("R for HE+ jeta = 27; depth = 2 \b");
          if (kcountHEpositivedirectionRecosignal2 == 13)
            HEpositivedirectionRecosignal2->SetXTitle("R for HE+ jeta = 28; depth = 2 \b");
          HEpositivedirectionRecosignal2->Draw("Error");
          kcountHEpositivedirectionRecosignal2++;
          if (kcountHEpositivedirectionRecosignal2 > 13)
            break;  // 4x6 = 24
        }           //ccctest>0

      }  // for i
    }    //if(jeta-41 >= 0)
  }      //for jeta
  /////////////////
  c3x5->Update();
  c3x5->Print("RrecosignalPositiveDirectionhistD1PhiSymmetryDepth2HE.png");
  c3x5->Clear();
  // clean-up
  if (h2CeffHEpositivedirectionRecosignal2)
    delete h2CeffHEpositivedirectionRecosignal2;
  //========================================================================================== 6
  //======================================================================
  //======================================================================1D plot: R vs phi , different eta,  depth=3
  //cout<<"      1D plot: R vs phi , different eta,  depth=3 *****" <<endl;
  c3x5->Clear();
  /////////////////
  c3x5->Divide(3, 5);
  c3x5->cd(1);
  int kcountHEpositivedirectionRecosignal3 = 1;
  TH1F* h2CeffHEpositivedirectionRecosignal3 = new TH1F("h2CeffHEpositivedirectionRecosignal3", "", nphi, 0., 72.);
  for (int jeta = 0; jeta < njeta; jeta++) {
    // positivedirectionRecosignal:
    if (jeta - 41 >= 0) {
      //	     for (int i=0;i<ndepth;i++) {
      // depth=3
      for (int i = 2; i < 3; i++) {
        TH1F* HEpositivedirectionRecosignal3 = (TH1F*)h2CeffHEpositivedirectionRecosignal3->Clone("twod1");
        float ccctest = 0;  // to avoid empty massive elements
        for (int jphi = 0; jphi < nphi; jphi++) {
          double ccc1 = arecosignalhe[i][jeta][jphi];
          if (ccc1 != 0.) {
            HEpositivedirectionRecosignal3->Fill(jphi, ccc1);
            ccctest = 1.;  //HEpositivedirectionRecosignal3->SetBinError(i,0.01);
          }
        }  // for jphi
        if (ccctest > 0.) {
          //cout<<"666        kcountHEpositivedirectionRecosignal3   =     "<<kcountHEpositivedirectionRecosignal3  <<"   jeta-41=     "<< jeta-41 <<endl;
          c3x5->cd(kcountHEpositivedirectionRecosignal3);
          HEpositivedirectionRecosignal3->SetMarkerStyle(20);
          HEpositivedirectionRecosignal3->SetMarkerSize(0.4);
          HEpositivedirectionRecosignal3->GetYaxis()->SetLabelSize(0.04);
          HEpositivedirectionRecosignal3->SetXTitle("HEpositivedirectionRecosignal3 \b");
          HEpositivedirectionRecosignal3->SetMarkerColor(2);
          HEpositivedirectionRecosignal3->SetLineColor(0);
          gPad->SetGridy();
          gPad->SetGridx();
          //	   gPad->SetLogy();
          if (kcountHEpositivedirectionRecosignal3 == 1)
            HEpositivedirectionRecosignal3->SetXTitle("R for HE+ jeta = 16; depth = 3 \b");
          if (kcountHEpositivedirectionRecosignal3 == 2)
            HEpositivedirectionRecosignal3->SetXTitle("R for HE+ jeta = 17; depth = 3 \b");
          if (kcountHEpositivedirectionRecosignal3 == 3)
            HEpositivedirectionRecosignal3->SetXTitle("R for HE+ jeta = 18; depth = 3 \b");
          if (kcountHEpositivedirectionRecosignal3 == 4)
            HEpositivedirectionRecosignal3->SetXTitle("R for HE+ jeta = 19; depth = 3 \b");
          if (kcountHEpositivedirectionRecosignal3 == 5)
            HEpositivedirectionRecosignal3->SetXTitle("R for HE+ jeta = 20; depth = 3 \b");
          if (kcountHEpositivedirectionRecosignal3 == 6)
            HEpositivedirectionRecosignal3->SetXTitle("R for HE+ jeta = 21; depth = 3 \b");
          if (kcountHEpositivedirectionRecosignal3 == 7)
            HEpositivedirectionRecosignal3->SetXTitle("R for HE+ jeta = 22; depth = 3 \b");
          if (kcountHEpositivedirectionRecosignal3 == 8)
            HEpositivedirectionRecosignal3->SetXTitle("R for HE+ jeta = 23; depth = 3 \b");
          if (kcountHEpositivedirectionRecosignal3 == 9)
            HEpositivedirectionRecosignal3->SetXTitle("R for HE+ jeta = 24; depth = 3 \b");
          if (kcountHEpositivedirectionRecosignal3 == 10)
            HEpositivedirectionRecosignal3->SetXTitle("R for HE+ jeta = 25; depth = 3 \b");
          if (kcountHEpositivedirectionRecosignal3 == 11)
            HEpositivedirectionRecosignal3->SetXTitle("R for HE+ jeta = 26; depth = 3 \b");
          if (kcountHEpositivedirectionRecosignal3 == 12)
            HEpositivedirectionRecosignal3->SetXTitle("R for HE+ jeta = 27; depth = 3 \b");
          if (kcountHEpositivedirectionRecosignal3 == 13)
            HEpositivedirectionRecosignal3->SetXTitle("R for HE+ jeta = 28; depth = 3 \b");
          HEpositivedirectionRecosignal3->Draw("Error");
          kcountHEpositivedirectionRecosignal3++;
          if (kcountHEpositivedirectionRecosignal3 > 13)
            break;  // 4x6 = 24
        }           //ccctest>0

      }  // for i
    }    //if(jeta-41 >= 0)
  }      //for jeta
  /////////////////
  c3x5->Update();
  c3x5->Print("RrecosignalPositiveDirectionhistD1PhiSymmetryDepth3HE.png");
  c3x5->Clear();
  // clean-up
  if (h2CeffHEpositivedirectionRecosignal3)
    delete h2CeffHEpositivedirectionRecosignal3;
  //========================================================================================== 7
  //======================================================================
  //======================================================================1D plot: R vs phi , different eta,  depth=4
  //cout<<"      1D plot: R vs phi , different eta,  depth=4 *****" <<endl;
  c3x5->Clear();
  /////////////////
  c3x5->Divide(3, 5);
  c3x5->cd(1);
  int kcountHEpositivedirectionRecosignal4 = 1;
  TH1F* h2CeffHEpositivedirectionRecosignal4 = new TH1F("h2CeffHEpositivedirectionRecosignal4", "", nphi, 0., 72.);

  for (int jeta = 0; jeta < njeta; jeta++) {
    // positivedirectionRecosignal:
    if (jeta - 41 >= 0) {
      //	     for (int i=0;i<ndepth;i++) {
      // depth=4
      for (int i = 3; i < 4; i++) {
        TH1F* HEpositivedirectionRecosignal4 = (TH1F*)h2CeffHEpositivedirectionRecosignal4->Clone("twod1");

        float ccctest = 0;  // to avoid empty massive elements
        for (int jphi = 0; jphi < nphi; jphi++) {
          double ccc1 = arecosignalhe[i][jeta][jphi];
          if (ccc1 != 0.) {
            HEpositivedirectionRecosignal4->Fill(jphi, ccc1);
            ccctest = 1.;  //HEpositivedirectionRecosignal4->SetBinError(i,0.01);
          }
        }  // for jphi
        if (ccctest > 0.) {
          //cout<<"777        kcountHEpositivedirectionRecosignal4   =     "<<kcountHEpositivedirectionRecosignal4  <<"   jeta-41=     "<< jeta-41 <<endl;
          c3x5->cd(kcountHEpositivedirectionRecosignal4);
          HEpositivedirectionRecosignal4->SetMarkerStyle(20);
          HEpositivedirectionRecosignal4->SetMarkerSize(0.4);
          HEpositivedirectionRecosignal4->GetYaxis()->SetLabelSize(0.04);
          HEpositivedirectionRecosignal4->SetXTitle("HEpositivedirectionRecosignal4 \b");
          HEpositivedirectionRecosignal4->SetMarkerColor(2);
          HEpositivedirectionRecosignal4->SetLineColor(0);
          gPad->SetGridy();
          gPad->SetGridx();
          //	   gPad->SetLogy();
          if (kcountHEpositivedirectionRecosignal4 == 1)
            HEpositivedirectionRecosignal4->SetXTitle("R for HE+ jeta = 15; depth = 4 \b");
          if (kcountHEpositivedirectionRecosignal4 == 2)
            HEpositivedirectionRecosignal4->SetXTitle("R for HE+ jeta = 17; depth = 4 \b");
          if (kcountHEpositivedirectionRecosignal4 == 3)
            HEpositivedirectionRecosignal4->SetXTitle("R for HE+ jeta = 18; depth = 4 \b");
          if (kcountHEpositivedirectionRecosignal4 == 4)
            HEpositivedirectionRecosignal4->SetXTitle("R for HE+ jeta = 19; depth = 4 \b");
          if (kcountHEpositivedirectionRecosignal4 == 5)
            HEpositivedirectionRecosignal4->SetXTitle("R for HE+ jeta = 20; depth = 4 \b");
          if (kcountHEpositivedirectionRecosignal4 == 6)
            HEpositivedirectionRecosignal4->SetXTitle("R for HE+ jeta = 21; depth = 4 \b");
          if (kcountHEpositivedirectionRecosignal4 == 7)
            HEpositivedirectionRecosignal4->SetXTitle("R for HE+ jeta = 22; depth = 4 \b");
          if (kcountHEpositivedirectionRecosignal4 == 8)
            HEpositivedirectionRecosignal4->SetXTitle("R for HE+ jeta = 23; depth = 4 \b");
          if (kcountHEpositivedirectionRecosignal4 == 9)
            HEpositivedirectionRecosignal4->SetXTitle("R for HE+ jeta = 24; depth = 4 \b");
          if (kcountHEpositivedirectionRecosignal4 == 10)
            HEpositivedirectionRecosignal4->SetXTitle("R for HE+ jeta = 25; depth = 4 \b");
          if (kcountHEpositivedirectionRecosignal4 == 11)
            HEpositivedirectionRecosignal4->SetXTitle("R for HE+ jeta = 26; depth = 4 \b");
          if (kcountHEpositivedirectionRecosignal4 == 12)
            HEpositivedirectionRecosignal4->SetXTitle("R for HE+ jeta = 27; depth = 4 \b");
          HEpositivedirectionRecosignal4->Draw("Error");
          kcountHEpositivedirectionRecosignal4++;
          if (kcountHEpositivedirectionRecosignal4 > 12)
            break;  // 4x6 = 24
        }           //ccctest>0

      }  // for i
    }    //if(jeta-41 >= 0)
  }      //for jeta
  /////////////////
  c3x5->Update();
  c3x5->Print("RrecosignalPositiveDirectionhistD1PhiSymmetryDepth4HE.png");
  c3x5->Clear();
  // clean-up
  if (h2CeffHEpositivedirectionRecosignal4)
    delete h2CeffHEpositivedirectionRecosignal4;
  //========================================================================================== 8
  //======================================================================
  //======================================================================1D plot: R vs phi , different eta,  depth=5
  //cout<<"      1D plot: R vs phi , different eta,  depth=5 *****" <<endl;
  c3x5->Clear();
  /////////////////
  c3x5->Divide(3, 5);
  c3x5->cd(1);
  int kcountHEpositivedirectionRecosignal5 = 1;
  TH1F* h2CeffHEpositivedirectionRecosignal5 = new TH1F("h2CeffHEpositivedirectionRecosignal5", "", nphi, 0., 72.);

  for (int jeta = 0; jeta < njeta; jeta++) {
    // positivedirectionRecosignal:
    if (jeta - 41 >= 0) {
      //	     for (int i=0;i<ndepth;i++) {
      // depth=5
      for (int i = 4; i < 5; i++) {
        TH1F* HEpositivedirectionRecosignal5 = (TH1F*)h2CeffHEpositivedirectionRecosignal5->Clone("twod1");

        float ccctest = 0;  // to avoid empty massive elements
        for (int jphi = 0; jphi < nphi; jphi++) {
          //	       cout<<"888  initial      kcountHEpositivedirectionRecosignal5   =     "<<kcountHEpositivedirectionRecosignal5  <<"   jeta-41=     "<< jeta-41 <<"   jphi=     "<< jphi <<"   arecosignalhe[i][jeta][jphi]=     "<< arecosignalhe[i][jeta][jphi] <<"  depth=     "<< i <<endl;

          double ccc1 = arecosignalhe[i][jeta][jphi];
          if (ccc1 != 0.) {
            HEpositivedirectionRecosignal5->Fill(jphi, ccc1);
            ccctest = 1.;  //HEpositivedirectionRecosignal5->SetBinError(i,0.01);
          }
        }  // for jphi
        if (ccctest > 0.) {
          //cout<<"888        kcountHEpositivedirectionRecosignal5   =     "<<kcountHEpositivedirectionRecosignal5  <<"   jeta-41=     "<< jeta-41 <<endl;
          c3x5->cd(kcountHEpositivedirectionRecosignal5);
          HEpositivedirectionRecosignal5->SetMarkerStyle(20);
          HEpositivedirectionRecosignal5->SetMarkerSize(0.4);
          HEpositivedirectionRecosignal5->GetYaxis()->SetLabelSize(0.04);
          HEpositivedirectionRecosignal5->SetXTitle("HEpositivedirectionRecosignal5 \b");
          HEpositivedirectionRecosignal5->SetMarkerColor(2);
          HEpositivedirectionRecosignal5->SetLineColor(0);
          gPad->SetGridy();
          gPad->SetGridx();
          //	   gPad->SetLogy();
          if (kcountHEpositivedirectionRecosignal5 == 1)
            HEpositivedirectionRecosignal5->SetXTitle("R for HE+ jeta = 17; depth = 5 \b");
          if (kcountHEpositivedirectionRecosignal5 == 2)
            HEpositivedirectionRecosignal5->SetXTitle("R for HE+ jeta = 18; depth = 5 \b");
          if (kcountHEpositivedirectionRecosignal5 == 3)
            HEpositivedirectionRecosignal5->SetXTitle("R for HE+ jeta = 19; depth = 5 \b");
          if (kcountHEpositivedirectionRecosignal5 == 4)
            HEpositivedirectionRecosignal5->SetXTitle("R for HE+ jeta = 20; depth = 5 \b");
          if (kcountHEpositivedirectionRecosignal5 == 5)
            HEpositivedirectionRecosignal5->SetXTitle("R for HE+ jeta = 21; depth = 5 \b");
          if (kcountHEpositivedirectionRecosignal5 == 6)
            HEpositivedirectionRecosignal5->SetXTitle("R for HE+ jeta = 22; depth = 5 \b");
          if (kcountHEpositivedirectionRecosignal5 == 7)
            HEpositivedirectionRecosignal5->SetXTitle("R for HE+ jeta = 23; depth = 5 \b");
          if (kcountHEpositivedirectionRecosignal5 == 8)
            HEpositivedirectionRecosignal5->SetXTitle("R for HE+ jeta = 24; depth = 5 \b");
          if (kcountHEpositivedirectionRecosignal5 == 9)
            HEpositivedirectionRecosignal5->SetXTitle("R for HE+ jeta = 25; depth = 5 \b");
          if (kcountHEpositivedirectionRecosignal5 == 10)
            HEpositivedirectionRecosignal5->SetXTitle("R for HE+ jeta = 26; depth = 5 \b");
          if (kcountHEpositivedirectionRecosignal5 == 11)
            HEpositivedirectionRecosignal5->SetXTitle("R for HE+ jeta = 27; depth = 5 \b");
          HEpositivedirectionRecosignal5->Draw("Error");
          kcountHEpositivedirectionRecosignal5++;
          if (kcountHEpositivedirectionRecosignal5 > 11)
            break;  // 4x6 = 24
        }           //ccctest>0

      }  // for i
    }    //if(jeta-41 >= 0)
  }      //for jeta
  /////////////////
  c3x5->Update();
  c3x5->Print("RrecosignalPositiveDirectionhistD1PhiSymmetryDepth5HE.png");
  c3x5->Clear();
  // clean-up
  if (h2CeffHEpositivedirectionRecosignal5)
    delete h2CeffHEpositivedirectionRecosignal5;
  //========================================================================================== 9
  //======================================================================
  //======================================================================1D plot: R vs phi , different eta,  depth=6
  //cout<<"      1D plot: R vs phi , different eta,  depth=6 *****" <<endl;
  c3x5->Clear();
  /////////////////
  c3x5->Divide(3, 5);
  c3x5->cd(1);
  int kcountHEpositivedirectionRecosignal6 = 1;
  TH1F* h2CeffHEpositivedirectionRecosignal6 = new TH1F("h2CeffHEpositivedirectionRecosignal6", "", nphi, 0., 72.);

  for (int jeta = 0; jeta < njeta; jeta++) {
    // positivedirectionRecosignal:
    if (jeta - 41 >= 0) {
      //	     for (int i=0;i<ndepth;i++) {
      // depth=6
      for (int i = 5; i < 6; i++) {
        TH1F* HEpositivedirectionRecosignal6 = (TH1F*)h2CeffHEpositivedirectionRecosignal6->Clone("twod1");

        float ccctest = 0;  // to avoid empty massive elements
        for (int jphi = 0; jphi < nphi; jphi++) {
          double ccc1 = arecosignalhe[i][jeta][jphi];
          if (ccc1 != 0.) {
            HEpositivedirectionRecosignal6->Fill(jphi, ccc1);
            ccctest = 1.;  //HEpositivedirectionRecosignal6->SetBinError(i,0.01);
          }
        }  // for jphi
        if (ccctest > 0.) {
          //cout<<"999        kcountHEpositivedirectionRecosignal6   =     "<<kcountHEpositivedirectionRecosignal6  <<"   jeta-41=     "<< jeta-41 <<endl;
          c3x5->cd(kcountHEpositivedirectionRecosignal6);
          HEpositivedirectionRecosignal6->SetMarkerStyle(20);
          HEpositivedirectionRecosignal6->SetMarkerSize(0.4);
          HEpositivedirectionRecosignal6->GetYaxis()->SetLabelSize(0.04);
          HEpositivedirectionRecosignal6->SetXTitle("HEpositivedirectionRecosignal6 \b");
          HEpositivedirectionRecosignal6->SetMarkerColor(2);
          HEpositivedirectionRecosignal6->SetLineColor(0);
          gPad->SetGridy();
          gPad->SetGridx();
          //	   gPad->SetLogy();
          if (kcountHEpositivedirectionRecosignal6 == 1)
            HEpositivedirectionRecosignal6->SetXTitle("R for HE+ jeta = 18; depth = 6 \b");
          if (kcountHEpositivedirectionRecosignal6 == 2)
            HEpositivedirectionRecosignal6->SetXTitle("R for HE+ jeta = 19; depth = 6 \b");
          if (kcountHEpositivedirectionRecosignal6 == 3)
            HEpositivedirectionRecosignal6->SetXTitle("R for HE+ jeta = 20; depth = 6 \b");
          if (kcountHEpositivedirectionRecosignal6 == 4)
            HEpositivedirectionRecosignal6->SetXTitle("R for HE+ jeta = 21; depth = 6 \b");
          if (kcountHEpositivedirectionRecosignal6 == 5)
            HEpositivedirectionRecosignal6->SetXTitle("R for HE+ jeta = 22; depth = 6 \b");
          if (kcountHEpositivedirectionRecosignal6 == 6)
            HEpositivedirectionRecosignal6->SetXTitle("R for HE+ jeta = 23; depth = 6 \b");
          if (kcountHEpositivedirectionRecosignal6 == 7)
            HEpositivedirectionRecosignal6->SetXTitle("R for HE+ jeta = 24; depth = 6 \b");
          if (kcountHEpositivedirectionRecosignal6 == 8)
            HEpositivedirectionRecosignal6->SetXTitle("R for HE+ jeta = 25; depth = 6 \b");
          if (kcountHEpositivedirectionRecosignal6 == 9)
            HEpositivedirectionRecosignal6->SetXTitle("R for HE+ jeta = 26; depth = 6 \b");
          if (kcountHEpositivedirectionRecosignal6 == 10)
            HEpositivedirectionRecosignal6->SetXTitle("R for HE+ jeta = 27; depth = 6 \b");
          HEpositivedirectionRecosignal6->Draw("Error");
          kcountHEpositivedirectionRecosignal6++;
          if (kcountHEpositivedirectionRecosignal6 > 10)
            break;  // 4x6 = 24
        }           //ccctest>0

      }  // for i
    }    //if(jeta-41 >= 0)
  }      //for jeta
  /////////////////
  c3x5->Update();
  c3x5->Print("RrecosignalPositiveDirectionhistD1PhiSymmetryDepth6HE.png");
  c3x5->Clear();
  // clean-up
  if (h2CeffHEpositivedirectionRecosignal6)
    delete h2CeffHEpositivedirectionRecosignal6;
  //========================================================================================== 10
  //======================================================================
  //======================================================================1D plot: R vs phi , different eta,  depth=7
  //cout<<"      1D plot: R vs phi , different eta,  depth=7 *****" <<endl;
  c3x5->Clear();
  /////////////////
  c3x5->Divide(3, 5);
  c3x5->cd(1);
  int kcountHEpositivedirectionRecosignal7 = 1;
  TH1F* h2CeffHEpositivedirectionRecosignal7 = new TH1F("h2CeffHEpositivedirectionRecosignal7", "", nphi, 0., 72.);

  for (int jeta = 0; jeta < njeta; jeta++) {
    // positivedirectionRecosignal:
    if (jeta - 41 >= 0) {
      //	     for (int i=0;i<ndepth;i++) {
      // depth=7
      for (int i = 6; i < 7; i++) {
        TH1F* HEpositivedirectionRecosignal7 = (TH1F*)h2CeffHEpositivedirectionRecosignal7->Clone("twod1");

        float ccctest = 0;  // to avoid empty massive elements
        for (int jphi = 0; jphi < nphi; jphi++) {
          double ccc1 = arecosignalhe[i][jeta][jphi];
          if (ccc1 != 0.) {
            HEpositivedirectionRecosignal7->Fill(jphi, ccc1);
            ccctest = 1.;  //HEpositivedirectionRecosignal7->SetBinError(i,0.01);
          }
        }  // for jphi
        if (ccctest > 0.) {
          //cout<<"1010       kcountHEpositivedirectionRecosignal7   =     "<<kcountHEpositivedirectionRecosignal7  <<"   jeta-41=     "<< jeta-41 <<endl;
          c3x5->cd(kcountHEpositivedirectionRecosignal7);
          HEpositivedirectionRecosignal7->SetMarkerStyle(20);
          HEpositivedirectionRecosignal7->SetMarkerSize(0.4);
          HEpositivedirectionRecosignal7->GetYaxis()->SetLabelSize(0.04);
          HEpositivedirectionRecosignal7->SetXTitle("HEpositivedirectionRecosignal7 \b");
          HEpositivedirectionRecosignal7->SetMarkerColor(2);
          HEpositivedirectionRecosignal7->SetLineColor(0);
          gPad->SetGridy();
          gPad->SetGridx();
          //	   gPad->SetLogy();
          if (kcountHEpositivedirectionRecosignal7 == 1)
            HEpositivedirectionRecosignal7->SetXTitle("R for HE+ jeta = 25; depth = 7 \b");
          if (kcountHEpositivedirectionRecosignal7 == 2)
            HEpositivedirectionRecosignal7->SetXTitle("R for HE+ jeta = 26; depth = 7 \b");
          if (kcountHEpositivedirectionRecosignal7 == 3)
            HEpositivedirectionRecosignal7->SetXTitle("R for HE+ jeta = 27; depth = 7 \b");
          HEpositivedirectionRecosignal7->Draw("Error");
          kcountHEpositivedirectionRecosignal7++;
          if (kcountHEpositivedirectionRecosignal7 > 3)
            break;  //
        }           //ccctest>0

      }  // for i
    }    //if(jeta-41 >= 0)
  }      //for jeta
  /////////////////
  c3x5->Update();
  c3x5->Print("RrecosignalPositiveDirectionhistD1PhiSymmetryDepth7HE.png");
  c3x5->Clear();
  // clean-up
  if (h2CeffHEpositivedirectionRecosignal7)
    delete h2CeffHEpositivedirectionRecosignal7;

  //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  //========================================================================================== 1114
  //======================================================================
  //======================================================================1D plot: R vs phi , different eta,  depth=1
  //cout<<"      1D plot: R vs phi , different eta,  depth=1 *****" <<endl;
  c3x5->Clear();
  /////////////////
  c3x5->Divide(3, 5);
  c3x5->cd(1);
  int kcountHEnegativedirectionRecosignal1 = 1;
  TH1F* h2CeffHEnegativedirectionRecosignal1 = new TH1F("h2CeffHEnegativedirectionRecosignal1", "", nphi, 0., 72.);
  for (int jeta = 0; jeta < njeta; jeta++) {
    // negativedirectionRecosignal:
    if (jeta - 41 < 0) {
      //	     for (int i=0;i<ndepth;i++) {
      // depth=1
      for (int i = 0; i < 1; i++) {
        TH1F* HEnegativedirectionRecosignal1 = (TH1F*)h2CeffHEnegativedirectionRecosignal1->Clone("twod1");
        float ccctest = 0;  // to avoid empty massive elements
        for (int jphi = 0; jphi < nphi; jphi++) {
          double ccc1 = arecosignalhe[i][jeta][jphi];
          if (ccc1 != 0.) {
            HEnegativedirectionRecosignal1->Fill(jphi, ccc1);
            ccctest = 1.;  //HEnegativedirectionRecosignal1->SetBinError(i,0.01);
          }
        }  // for jphi
        if (ccctest > 0.) {
          //	  cout<<"444        kcountHEnegativedirectionRecosignal1   =     "<<kcountHEnegativedirectionRecosignal1  <<"   jeta-41=     "<< jeta-41 <<endl;
          c3x5->cd(kcountHEnegativedirectionRecosignal1);
          HEnegativedirectionRecosignal1->SetMarkerStyle(20);
          HEnegativedirectionRecosignal1->SetMarkerSize(0.4);
          HEnegativedirectionRecosignal1->GetYaxis()->SetLabelSize(0.04);
          HEnegativedirectionRecosignal1->SetXTitle("HEnegativedirectionRecosignal1 \b");
          HEnegativedirectionRecosignal1->SetMarkerColor(2);
          HEnegativedirectionRecosignal1->SetLineColor(0);
          gPad->SetGridy();
          gPad->SetGridx();
          //	   gPad->SetLogy();
          if (kcountHEnegativedirectionRecosignal1 == 1)
            HEnegativedirectionRecosignal1->SetXTitle("R for HE- jeta =-29; depth = 1 \b");
          if (kcountHEnegativedirectionRecosignal1 == 2)
            HEnegativedirectionRecosignal1->SetXTitle("R for HE- jeta =-28; depth = 1 \b");
          if (kcountHEnegativedirectionRecosignal1 == 3)
            HEnegativedirectionRecosignal1->SetXTitle("R for HE- jeta =-27; depth = 1 \b");
          if (kcountHEnegativedirectionRecosignal1 == 4)
            HEnegativedirectionRecosignal1->SetXTitle("R for HE- jeta =-26; depth = 1 \b");
          if (kcountHEnegativedirectionRecosignal1 == 5)
            HEnegativedirectionRecosignal1->SetXTitle("R for HE- jeta =-25; depth = 1 \b");
          if (kcountHEnegativedirectionRecosignal1 == 6)
            HEnegativedirectionRecosignal1->SetXTitle("R for HE- jeta =-24; depth = 1 \b");
          if (kcountHEnegativedirectionRecosignal1 == 7)
            HEnegativedirectionRecosignal1->SetXTitle("R for HE- jeta =-23; depth = 1 \b");
          if (kcountHEnegativedirectionRecosignal1 == 8)
            HEnegativedirectionRecosignal1->SetXTitle("R for HE- jeta =-22; depth = 1 \b");
          if (kcountHEnegativedirectionRecosignal1 == 9)
            HEnegativedirectionRecosignal1->SetXTitle("R for HE- jeta =-21; depth = 1 \b");
          if (kcountHEnegativedirectionRecosignal1 == 10)
            HEnegativedirectionRecosignal1->SetXTitle("R for HE- jeta =-20; depth = 1 \b");
          if (kcountHEnegativedirectionRecosignal1 == 11)
            HEnegativedirectionRecosignal1->SetXTitle("R for HE- jeta =-19; depth = 1 \b");
          if (kcountHEnegativedirectionRecosignal1 == 12)
            HEnegativedirectionRecosignal1->SetXTitle("R for HE- jeta =-18; depth = 1 \b");
          HEnegativedirectionRecosignal1->Draw("Error");
          kcountHEnegativedirectionRecosignal1++;
          if (kcountHEnegativedirectionRecosignal1 > 12)
            break;  // 4x6 = 24
        }           //ccctest>0

      }  // for i
    }    //if(jeta-41 < 0)
  }      //for jeta
  /////////////////
  c3x5->Update();
  c3x5->Print("RrecosignalNegativeDirectionhistD1PhiSymmetryDepth1HE.png");
  c3x5->Clear();
  // clean-up
  if (h2CeffHEnegativedirectionRecosignal1)
    delete h2CeffHEnegativedirectionRecosignal1;

  //========================================================================================== 1115
  //======================================================================
  //======================================================================1D plot: R vs phi , different eta,  depth=2
  //  cout<<"      1D plot: R vs phi , different eta,  depth=2 *****" <<endl;
  c3x5->Clear();
  /////////////////
  c3x5->Divide(3, 5);
  c3x5->cd(1);
  int kcountHEnegativedirectionRecosignal2 = 1;
  TH1F* h2CeffHEnegativedirectionRecosignal2 = new TH1F("h2CeffHEnegativedirectionRecosignal2", "", nphi, 0., 72.);
  for (int jeta = 0; jeta < njeta; jeta++) {
    // negativedirectionRecosignal:
    if (jeta - 41 < 0) {
      //	     for (int i=0;i<ndepth;i++) {
      // depth=2
      for (int i = 1; i < 2; i++) {
        TH1F* HEnegativedirectionRecosignal2 = (TH1F*)h2CeffHEnegativedirectionRecosignal2->Clone("twod1");
        float ccctest = 0;  // to avoid empty massive elements
        for (int jphi = 0; jphi < nphi; jphi++) {
          double ccc1 = arecosignalhe[i][jeta][jphi];
          if (ccc1 != 0.) {
            HEnegativedirectionRecosignal2->Fill(jphi, ccc1);
            ccctest = 1.;  //HEnegativedirectionRecosignal2->SetBinError(i,0.01);
          }
        }  // for jphi
        if (ccctest > 0.) {
          //cout<<"555        kcountHEnegativedirectionRecosignal2   =     "<<kcountHEnegativedirectionRecosignal2  <<"   jeta-41=     "<< jeta-41 <<endl;
          c3x5->cd(kcountHEnegativedirectionRecosignal2);
          HEnegativedirectionRecosignal2->SetMarkerStyle(20);
          HEnegativedirectionRecosignal2->SetMarkerSize(0.4);
          HEnegativedirectionRecosignal2->GetYaxis()->SetLabelSize(0.04);
          HEnegativedirectionRecosignal2->SetXTitle("HEnegativedirectionRecosignal2 \b");
          HEnegativedirectionRecosignal2->SetMarkerColor(2);
          HEnegativedirectionRecosignal2->SetLineColor(0);
          gPad->SetGridy();
          gPad->SetGridx();
          //	   gPad->SetLogy();
          if (kcountHEnegativedirectionRecosignal2 == 1)
            HEnegativedirectionRecosignal2->SetXTitle("R for HE- jeta =-29; depth = 2 \b");
          if (kcountHEnegativedirectionRecosignal2 == 2)
            HEnegativedirectionRecosignal2->SetXTitle("R for HE- jeta =-28; depth = 2 \b");
          if (kcountHEnegativedirectionRecosignal2 == 3)
            HEnegativedirectionRecosignal2->SetXTitle("R for HE- jeta =-27; depth = 2 \b");
          if (kcountHEnegativedirectionRecosignal2 == 4)
            HEnegativedirectionRecosignal2->SetXTitle("R for HE- jeta =-26; depth = 2 \b");
          if (kcountHEnegativedirectionRecosignal2 == 5)
            HEnegativedirectionRecosignal2->SetXTitle("R for HE- jeta =-25; depth = 2 \b");
          if (kcountHEnegativedirectionRecosignal2 == 6)
            HEnegativedirectionRecosignal2->SetXTitle("R for HE- jeta =-24; depth = 2 \b");
          if (kcountHEnegativedirectionRecosignal2 == 7)
            HEnegativedirectionRecosignal2->SetXTitle("R for HE- jeta =-23; depth = 2 \b");
          if (kcountHEnegativedirectionRecosignal2 == 8)
            HEnegativedirectionRecosignal2->SetXTitle("R for HE- jeta =-22; depth = 2 \b");
          if (kcountHEnegativedirectionRecosignal2 == 9)
            HEnegativedirectionRecosignal2->SetXTitle("R for HE- jeta =-21; depth = 2 \b");
          if (kcountHEnegativedirectionRecosignal2 == 10)
            HEnegativedirectionRecosignal2->SetXTitle("R for HE- jeta =-20; depth = 2 \b");
          if (kcountHEnegativedirectionRecosignal2 == 11)
            HEnegativedirectionRecosignal2->SetXTitle("R for HE- jeta =-19; depth = 2 \b");
          if (kcountHEnegativedirectionRecosignal2 == 12)
            HEnegativedirectionRecosignal2->SetXTitle("R for HE- jeta =-18; depth = 2 \b");
          if (kcountHEnegativedirectionRecosignal2 == 13)
            HEnegativedirectionRecosignal2->SetXTitle("R for HE- jeta =-17; depth = 2 \b");
          HEnegativedirectionRecosignal2->Draw("Error");
          kcountHEnegativedirectionRecosignal2++;
          if (kcountHEnegativedirectionRecosignal2 > 13)
            break;  // 4x6 = 24
        }           //ccctest>0

      }  // for i
    }    //if(jeta-41 < 0)
  }      //for jeta
  /////////////////
  c3x5->Update();
  c3x5->Print("RrecosignalNegativeDirectionhistD1PhiSymmetryDepth2HE.png");
  c3x5->Clear();
  // clean-up
  if (h2CeffHEnegativedirectionRecosignal2)
    delete h2CeffHEnegativedirectionRecosignal2;
  //========================================================================================== 1116
  //======================================================================
  //======================================================================1D plot: R vs phi , different eta,  depth=3
  //cout<<"      1D plot: R vs phi , different eta,  depth=3 *****" <<endl;
  c3x5->Clear();
  /////////////////
  c3x5->Divide(3, 5);
  c3x5->cd(1);
  int kcountHEnegativedirectionRecosignal3 = 1;
  TH1F* h2CeffHEnegativedirectionRecosignal3 = new TH1F("h2CeffHEnegativedirectionRecosignal3", "", nphi, 0., 72.);
  for (int jeta = 0; jeta < njeta; jeta++) {
    // negativedirectionRecosignal:
    if (jeta - 41 < 0) {
      //	     for (int i=0;i<ndepth;i++) {
      // depth=3
      for (int i = 2; i < 3; i++) {
        TH1F* HEnegativedirectionRecosignal3 = (TH1F*)h2CeffHEnegativedirectionRecosignal3->Clone("twod1");
        float ccctest = 0;  // to avoid empty massive elements
        for (int jphi = 0; jphi < nphi; jphi++) {
          double ccc1 = arecosignalhe[i][jeta][jphi];
          if (ccc1 != 0.) {
            HEnegativedirectionRecosignal3->Fill(jphi, ccc1);
            ccctest = 1.;  //HEnegativedirectionRecosignal3->SetBinError(i,0.01);
          }
        }  // for jphi
        if (ccctest > 0.) {
          //cout<<"666        kcountHEnegativedirectionRecosignal3   =     "<<kcountHEnegativedirectionRecosignal3  <<"   jeta-41=     "<< jeta-41 <<endl;
          c3x5->cd(kcountHEnegativedirectionRecosignal3);
          HEnegativedirectionRecosignal3->SetMarkerStyle(20);
          HEnegativedirectionRecosignal3->SetMarkerSize(0.4);
          HEnegativedirectionRecosignal3->GetYaxis()->SetLabelSize(0.04);
          HEnegativedirectionRecosignal3->SetXTitle("HEnegativedirectionRecosignal3 \b");
          HEnegativedirectionRecosignal3->SetMarkerColor(2);
          HEnegativedirectionRecosignal3->SetLineColor(0);
          gPad->SetGridy();
          gPad->SetGridx();
          //	   gPad->SetLogy();
          if (kcountHEnegativedirectionRecosignal3 == 1)
            HEnegativedirectionRecosignal3->SetXTitle("R for HE- jeta =-29; depth = 3 \b");
          if (kcountHEnegativedirectionRecosignal3 == 2)
            HEnegativedirectionRecosignal3->SetXTitle("R for HE- jeta =-28; depth = 3 \b");
          if (kcountHEnegativedirectionRecosignal3 == 3)
            HEnegativedirectionRecosignal3->SetXTitle("R for HE- jeta =-27; depth = 3 \b");
          if (kcountHEnegativedirectionRecosignal3 == 4)
            HEnegativedirectionRecosignal3->SetXTitle("R for HE- jeta =-26; depth = 3 \b");
          if (kcountHEnegativedirectionRecosignal3 == 5)
            HEnegativedirectionRecosignal3->SetXTitle("R for HE- jeta =-25; depth = 3 \b");
          if (kcountHEnegativedirectionRecosignal3 == 6)
            HEnegativedirectionRecosignal3->SetXTitle("R for HE- jeta =-24; depth = 3 \b");
          if (kcountHEnegativedirectionRecosignal3 == 7)
            HEnegativedirectionRecosignal3->SetXTitle("R for HE- jeta =-23; depth = 3 \b");
          if (kcountHEnegativedirectionRecosignal3 == 8)
            HEnegativedirectionRecosignal3->SetXTitle("R for HE- jeta =-22; depth = 3 \b");
          if (kcountHEnegativedirectionRecosignal3 == 9)
            HEnegativedirectionRecosignal3->SetXTitle("R for HE- jeta =-21; depth = 3 \b");
          if (kcountHEnegativedirectionRecosignal3 == 10)
            HEnegativedirectionRecosignal3->SetXTitle("R for HE- jeta =-20; depth = 3 \b");
          if (kcountHEnegativedirectionRecosignal3 == 11)
            HEnegativedirectionRecosignal3->SetXTitle("R for HE- jeta =-19; depth = 3 \b");
          if (kcountHEnegativedirectionRecosignal3 == 12)
            HEnegativedirectionRecosignal3->SetXTitle("R for HE- jeta =-18; depth = 3 \b");
          if (kcountHEnegativedirectionRecosignal3 == 13)
            HEnegativedirectionRecosignal3->SetXTitle("R for HE- jeta =-17; depth = 3 \b");
          HEnegativedirectionRecosignal3->Draw("Error");
          kcountHEnegativedirectionRecosignal3++;
          if (kcountHEnegativedirectionRecosignal3 > 13)
            break;  // 4x6 = 24
        }           //ccctest>0

      }  // for i
    }    //if(jeta-41 < 0)
  }      //for jeta
  /////////////////
  c3x5->Update();
  c3x5->Print("RrecosignalNegativeDirectionhistD1PhiSymmetryDepth3HE.png");
  c3x5->Clear();
  // clean-up
  if (h2CeffHEnegativedirectionRecosignal3)
    delete h2CeffHEnegativedirectionRecosignal3;
  //========================================================================================== 1117
  //======================================================================
  //======================================================================1D plot: R vs phi , different eta,  depth=4
  //cout<<"      1D plot: R vs phi , different eta,  depth=4 *****" <<endl;
  c3x5->Clear();
  /////////////////
  c3x5->Divide(3, 5);
  c3x5->cd(1);
  int kcountHEnegativedirectionRecosignal4 = 1;
  TH1F* h2CeffHEnegativedirectionRecosignal4 = new TH1F("h2CeffHEnegativedirectionRecosignal4", "", nphi, 0., 72.);

  for (int jeta = 0; jeta < njeta; jeta++) {
    // negativedirectionRecosignal:
    if (jeta - 41 < 0) {
      //	     for (int i=0;i<ndepth;i++) {
      // depth=4
      for (int i = 3; i < 4; i++) {
        TH1F* HEnegativedirectionRecosignal4 = (TH1F*)h2CeffHEnegativedirectionRecosignal4->Clone("twod1");

        float ccctest = 0;  // to avoid empty massive elements
        for (int jphi = 0; jphi < nphi; jphi++) {
          double ccc1 = arecosignalhe[i][jeta][jphi];
          if (ccc1 != 0.) {
            HEnegativedirectionRecosignal4->Fill(jphi, ccc1);
            ccctest = 1.;  //HEnegativedirectionRecosignal4->SetBinError(i,0.01);
          }
        }  // for jphi
        if (ccctest > 0.) {
          //cout<<"777        kcountHEnegativedirectionRecosignal4   =     "<<kcountHEnegativedirectionRecosignal4  <<"   jeta-41=     "<< jeta-41 <<endl;
          c3x5->cd(kcountHEnegativedirectionRecosignal4);
          HEnegativedirectionRecosignal4->SetMarkerStyle(20);
          HEnegativedirectionRecosignal4->SetMarkerSize(0.4);
          HEnegativedirectionRecosignal4->GetYaxis()->SetLabelSize(0.04);
          HEnegativedirectionRecosignal4->SetXTitle("HEnegativedirectionRecosignal4 \b");
          HEnegativedirectionRecosignal4->SetMarkerColor(2);
          HEnegativedirectionRecosignal4->SetLineColor(0);
          gPad->SetGridy();
          gPad->SetGridx();
          //	   gPad->SetLogy();
          if (kcountHEnegativedirectionRecosignal4 == 1)
            HEnegativedirectionRecosignal4->SetXTitle("R for HE- jeta =-28; depth = 4 \b");
          if (kcountHEnegativedirectionRecosignal4 == 2)
            HEnegativedirectionRecosignal4->SetXTitle("R for HE- jeta =-27; depth = 4 \b");
          if (kcountHEnegativedirectionRecosignal4 == 3)
            HEnegativedirectionRecosignal4->SetXTitle("R for HE- jeta =-26; depth = 4 \b");
          if (kcountHEnegativedirectionRecosignal4 == 4)
            HEnegativedirectionRecosignal4->SetXTitle("R for HE- jeta =-25; depth = 4 \b");
          if (kcountHEnegativedirectionRecosignal4 == 5)
            HEnegativedirectionRecosignal4->SetXTitle("R for HE- jeta =-24; depth = 4 \b");
          if (kcountHEnegativedirectionRecosignal4 == 6)
            HEnegativedirectionRecosignal4->SetXTitle("R for HE- jeta =-23; depth = 4 \b");
          if (kcountHEnegativedirectionRecosignal4 == 7)
            HEnegativedirectionRecosignal4->SetXTitle("R for HE- jeta =-22; depth = 4 \b");
          if (kcountHEnegativedirectionRecosignal4 == 8)
            HEnegativedirectionRecosignal4->SetXTitle("R for HE- jeta =-21; depth = 4 \b");
          if (kcountHEnegativedirectionRecosignal4 == 9)
            HEnegativedirectionRecosignal4->SetXTitle("R for HE- jeta =-20; depth = 4 \b");
          if (kcountHEnegativedirectionRecosignal4 == 10)
            HEnegativedirectionRecosignal4->SetXTitle("R for HE- jeta =-19; depth = 4 \b");
          if (kcountHEnegativedirectionRecosignal4 == 11)
            HEnegativedirectionRecosignal4->SetXTitle("R for HE- jeta =-18; depth = 4 \b");
          if (kcountHEnegativedirectionRecosignal4 == 12)
            HEnegativedirectionRecosignal4->SetXTitle("R for HE- jeta =-16; depth = 4 \b");
          HEnegativedirectionRecosignal4->Draw("Error");
          kcountHEnegativedirectionRecosignal4++;
          if (kcountHEnegativedirectionRecosignal4 > 12)
            break;  // 4x6 = 24
        }           //ccctest>0

      }  // for i
    }    //if(jeta-41 < 0)
  }      //for jeta
  /////////////////
  c3x5->Update();
  c3x5->Print("RrecosignalNegativeDirectionhistD1PhiSymmetryDepth4HE.png");
  c3x5->Clear();
  // clean-up
  if (h2CeffHEnegativedirectionRecosignal4)
    delete h2CeffHEnegativedirectionRecosignal4;
  //========================================================================================== 1118
  //======================================================================
  //======================================================================1D plot: R vs phi , different eta,  depth=5
  //cout<<"      1D plot: R vs phi , different eta,  depth=5 *****" <<endl;
  c3x5->Clear();
  /////////////////
  c3x5->Divide(3, 5);
  c3x5->cd(1);
  int kcountHEnegativedirectionRecosignal5 = 1;
  TH1F* h2CeffHEnegativedirectionRecosignal5 = new TH1F("h2CeffHEnegativedirectionRecosignal5", "", nphi, 0., 72.);

  for (int jeta = 0; jeta < njeta; jeta++) {
    // negativedirectionRecosignal:
    if (jeta - 41 < 0) {
      //	     for (int i=0;i<ndepth;i++) {
      // depth=5
      for (int i = 4; i < 5; i++) {
        TH1F* HEnegativedirectionRecosignal5 = (TH1F*)h2CeffHEnegativedirectionRecosignal5->Clone("twod1");

        float ccctest = 0;  // to avoid empty massive elements
        for (int jphi = 0; jphi < nphi; jphi++) {
          //	       cout<<"888  initial      kcountHEnegativedirectionRecosignal5   =     "<<kcountHEnegativedirectionRecosignal5  <<"   jeta-41=     "<< jeta-41 <<"   jphi=     "<< jphi <<"   arecosignalhe[i][jeta][jphi]=     "<< arecosignalhe[i][jeta][jphi] <<"  depth=     "<< i <<endl;

          double ccc1 = arecosignalhe[i][jeta][jphi];
          if (ccc1 != 0.) {
            HEnegativedirectionRecosignal5->Fill(jphi, ccc1);
            ccctest = 1.;  //HEnegativedirectionRecosignal5->SetBinError(i,0.01);
          }
        }  // for jphi
        if (ccctest > 0.) {
          //cout<<"888        kcountHEnegativedirectionRecosignal5   =     "<<kcountHEnegativedirectionRecosignal5  <<"   jeta-41=     "<< jeta-41 <<endl;
          c3x5->cd(kcountHEnegativedirectionRecosignal5);
          HEnegativedirectionRecosignal5->SetMarkerStyle(20);
          HEnegativedirectionRecosignal5->SetMarkerSize(0.4);
          HEnegativedirectionRecosignal5->GetYaxis()->SetLabelSize(0.04);
          HEnegativedirectionRecosignal5->SetXTitle("HEnegativedirectionRecosignal5 \b");
          HEnegativedirectionRecosignal5->SetMarkerColor(2);
          HEnegativedirectionRecosignal5->SetLineColor(0);
          gPad->SetGridy();
          gPad->SetGridx();
          //	   gPad->SetLogy();
          if (kcountHEnegativedirectionRecosignal5 == 1)
            HEnegativedirectionRecosignal5->SetXTitle("R for HE- jeta =-28; depth = 5 \b");
          if (kcountHEnegativedirectionRecosignal5 == 2)
            HEnegativedirectionRecosignal5->SetXTitle("R for HE- jeta =-27; depth = 5 \b");
          if (kcountHEnegativedirectionRecosignal5 == 3)
            HEnegativedirectionRecosignal5->SetXTitle("R for HE- jeta =-26; depth = 5 \b");
          if (kcountHEnegativedirectionRecosignal5 == 4)
            HEnegativedirectionRecosignal5->SetXTitle("R for HE- jeta =-25; depth = 5 \b");
          if (kcountHEnegativedirectionRecosignal5 == 5)
            HEnegativedirectionRecosignal5->SetXTitle("R for HE- jeta =-24; depth = 5 \b");
          if (kcountHEnegativedirectionRecosignal5 == 6)
            HEnegativedirectionRecosignal5->SetXTitle("R for HE- jeta =-23; depth = 5 \b");
          if (kcountHEnegativedirectionRecosignal5 == 7)
            HEnegativedirectionRecosignal5->SetXTitle("R for HE- jeta =-22; depth = 5 \b");
          if (kcountHEnegativedirectionRecosignal5 == 8)
            HEnegativedirectionRecosignal5->SetXTitle("R for HE- jeta =-21; depth = 5 \b");
          if (kcountHEnegativedirectionRecosignal5 == 9)
            HEnegativedirectionRecosignal5->SetXTitle("R for HE- jeta =-20; depth = 5 \b");
          if (kcountHEnegativedirectionRecosignal5 == 10)
            HEnegativedirectionRecosignal5->SetXTitle("R for HE- jeta =-19; depth = 5 \b");
          if (kcountHEnegativedirectionRecosignal5 == 11)
            HEnegativedirectionRecosignal5->SetXTitle("R for HE- jeta =-18; depth = 5 \b");
          HEnegativedirectionRecosignal5->Draw("Error");
          kcountHEnegativedirectionRecosignal5++;
          if (kcountHEnegativedirectionRecosignal5 > 11)
            break;  // 4x6 = 24
        }           //ccctest>0

      }  // for i
    }    //if(jeta-41 < 0)
  }      //for jeta
  /////////////////
  c3x5->Update();
  c3x5->Print("RrecosignalNegativeDirectionhistD1PhiSymmetryDepth5HE.png");
  c3x5->Clear();
  // clean-up
  if (h2CeffHEnegativedirectionRecosignal5)
    delete h2CeffHEnegativedirectionRecosignal5;
  //========================================================================================== 1119
  //======================================================================
  //======================================================================1D plot: R vs phi , different eta,  depth=6
  //cout<<"      1D plot: R vs phi , different eta,  depth=6 *****" <<endl;
  c3x5->Clear();
  /////////////////
  c3x5->Divide(3, 5);
  c3x5->cd(1);
  int kcountHEnegativedirectionRecosignal6 = 1;
  TH1F* h2CeffHEnegativedirectionRecosignal6 = new TH1F("h2CeffHEnegativedirectionRecosignal6", "", nphi, 0., 72.);

  for (int jeta = 0; jeta < njeta; jeta++) {
    // negativedirectionRecosignal:
    if (jeta - 41 < 0) {
      //	     for (int i=0;i<ndepth;i++) {
      // depth=6
      for (int i = 5; i < 6; i++) {
        TH1F* HEnegativedirectionRecosignal6 = (TH1F*)h2CeffHEnegativedirectionRecosignal6->Clone("twod1");

        float ccctest = 0;  // to avoid empty massive elements
        for (int jphi = 0; jphi < nphi; jphi++) {
          double ccc1 = arecosignalhe[i][jeta][jphi];
          if (ccc1 != 0.) {
            HEnegativedirectionRecosignal6->Fill(jphi, ccc1);
            ccctest = 1.;  //HEnegativedirectionRecosignal6->SetBinError(i,0.01);
          }
        }  // for jphi
        if (ccctest > 0.) {
          //cout<<"999        kcountHEnegativedirectionRecosignal6   =     "<<kcountHEnegativedirectionRecosignal6  <<"   jeta-41=     "<< jeta-41 <<endl;
          c3x5->cd(kcountHEnegativedirectionRecosignal6);
          HEnegativedirectionRecosignal6->SetMarkerStyle(20);
          HEnegativedirectionRecosignal6->SetMarkerSize(0.4);
          HEnegativedirectionRecosignal6->GetYaxis()->SetLabelSize(0.04);
          HEnegativedirectionRecosignal6->SetXTitle("HEnegativedirectionRecosignal6 \b");
          HEnegativedirectionRecosignal6->SetMarkerColor(2);
          HEnegativedirectionRecosignal6->SetLineColor(0);
          gPad->SetGridy();
          gPad->SetGridx();
          //	   gPad->SetLogy();
          if (kcountHEnegativedirectionRecosignal6 == 1)
            HEnegativedirectionRecosignal6->SetXTitle("R for HE- jeta =-28; depth = 6 \b");
          if (kcountHEnegativedirectionRecosignal6 == 2)
            HEnegativedirectionRecosignal6->SetXTitle("R for HE- jeta =-27; depth = 6 \b");
          if (kcountHEnegativedirectionRecosignal6 == 3)
            HEnegativedirectionRecosignal6->SetXTitle("R for HE- jeta =-26; depth = 6 \b");
          if (kcountHEnegativedirectionRecosignal6 == 4)
            HEnegativedirectionRecosignal6->SetXTitle("R for HE- jeta =-25; depth = 6 \b");
          if (kcountHEnegativedirectionRecosignal6 == 5)
            HEnegativedirectionRecosignal6->SetXTitle("R for HE- jeta =-24; depth = 6 \b");
          if (kcountHEnegativedirectionRecosignal6 == 6)
            HEnegativedirectionRecosignal6->SetXTitle("R for HE- jeta =-23; depth = 6 \b");
          if (kcountHEnegativedirectionRecosignal6 == 7)
            HEnegativedirectionRecosignal6->SetXTitle("R for HE- jeta =-22; depth = 6 \b");
          if (kcountHEnegativedirectionRecosignal6 == 8)
            HEnegativedirectionRecosignal6->SetXTitle("R for HE- jeta =-21; depth = 6 \b");
          if (kcountHEnegativedirectionRecosignal6 == 9)
            HEnegativedirectionRecosignal6->SetXTitle("R for HE- jeta =-20; depth = 6 \b");
          if (kcountHEnegativedirectionRecosignal6 == 10)
            HEnegativedirectionRecosignal6->SetXTitle("R for HE- jeta =-19; depth = 6 \b");
          HEnegativedirectionRecosignal6->Draw("Error");
          kcountHEnegativedirectionRecosignal6++;
          if (kcountHEnegativedirectionRecosignal6 > 10)
            break;  // 4x6 = 24
        }           //ccctest>0

      }  // for i
    }    //if(jeta-41 < 0)
  }      //for jeta
  /////////////////
  c3x5->Update();
  c3x5->Print("RrecosignalNegativeDirectionhistD1PhiSymmetryDepth6HE.png");
  c3x5->Clear();
  // clean-up
  if (h2CeffHEnegativedirectionRecosignal6)
    delete h2CeffHEnegativedirectionRecosignal6;
  //========================================================================================== 11110
  //======================================================================
  //======================================================================1D plot: R vs phi , different eta,  depth=7
  //cout<<"      1D plot: R vs phi , different eta,  depth=7 *****" <<endl;
  c3x5->Clear();
  /////////////////
  c3x5->Divide(3, 5);
  c3x5->cd(1);
  int kcountHEnegativedirectionRecosignal7 = 1;
  TH1F* h2CeffHEnegativedirectionRecosignal7 = new TH1F("h2CeffHEnegativedirectionRecosignal7", "", nphi, 0., 72.);

  for (int jeta = 0; jeta < njeta; jeta++) {
    // negativedirectionRecosignal:
    if (jeta - 41 < 0) {
      //	     for (int i=0;i<ndepth;i++) {
      // depth=7
      for (int i = 6; i < 7; i++) {
        TH1F* HEnegativedirectionRecosignal7 = (TH1F*)h2CeffHEnegativedirectionRecosignal7->Clone("twod1");

        float ccctest = 0;  // to avoid empty massive elements
        for (int jphi = 0; jphi < nphi; jphi++) {
          double ccc1 = arecosignalhe[i][jeta][jphi];
          if (ccc1 != 0.) {
            HEnegativedirectionRecosignal7->Fill(jphi, ccc1);
            ccctest = 1.;  //HEnegativedirectionRecosignal7->SetBinError(i,0.01);
          }
        }  // for jphi
        if (ccctest > 0.) {
          //cout<<"1010       kcountHEnegativedirectionRecosignal7   =     "<<kcountHEnegativedirectionRecosignal7  <<"   jeta-41=     "<< jeta-41 <<endl;
          c3x5->cd(kcountHEnegativedirectionRecosignal7);
          HEnegativedirectionRecosignal7->SetMarkerStyle(20);
          HEnegativedirectionRecosignal7->SetMarkerSize(0.4);
          HEnegativedirectionRecosignal7->GetYaxis()->SetLabelSize(0.04);
          HEnegativedirectionRecosignal7->SetXTitle("HEnegativedirectionRecosignal7 \b");
          HEnegativedirectionRecosignal7->SetMarkerColor(2);
          HEnegativedirectionRecosignal7->SetLineColor(0);
          gPad->SetGridy();
          gPad->SetGridx();
          //	   gPad->SetLogy();
          if (kcountHEnegativedirectionRecosignal7 == 1)
            HEnegativedirectionRecosignal7->SetXTitle("R for HE- jeta =-28; depth = 7 \b");
          if (kcountHEnegativedirectionRecosignal7 == 2)
            HEnegativedirectionRecosignal7->SetXTitle("R for HE- jeta =-27; depth = 7 \b");
          if (kcountHEnegativedirectionRecosignal7 == 3)
            HEnegativedirectionRecosignal7->SetXTitle("R for HE- jeta =-26; depth = 7 \b");
          HEnegativedirectionRecosignal7->Draw("Error");
          kcountHEnegativedirectionRecosignal7++;
          if (kcountHEnegativedirectionRecosignal7 > 3)
            break;  // 4x6 = 24
        }           //ccctest>0

      }  // for i
    }    //if(jeta-41 < 0)
  }      //for jeta
  /////////////////
  c3x5->Update();
  c3x5->Print("RrecosignalNegativeDirectionhistD1PhiSymmetryDepth7HE.png");
  c3x5->Clear();
  // clean-up
  if (h2CeffHEnegativedirectionRecosignal7)
    delete h2CeffHEnegativedirectionRecosignal7;

  //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

  //                            DDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDD:

  //cout<<"    Start Vaiance: preparation  *****" <<endl;
  TH2F* recosignalVariance1HE1 = (TH2F*)dir->FindObjectAny("h_recSignalEnergy2_HE1");
  TH2F* recosignalVariance0HE1 = (TH2F*)dir->FindObjectAny("h_recSignalEnergy0_HE1");
  TH2F* recosignalVarianceHE1 = (TH2F*)recosignalVariance1HE1->Clone("recosignalVarianceHE1");
  recosignalVarianceHE1->Divide(recosignalVariance1HE1, recosignalVariance0HE1, 1, 1, "B");
  TH2F* recosignalVariance1HE2 = (TH2F*)dir->FindObjectAny("h_recSignalEnergy2_HE2");
  TH2F* recosignalVariance0HE2 = (TH2F*)dir->FindObjectAny("h_recSignalEnergy0_HE2");
  TH2F* recosignalVarianceHE2 = (TH2F*)recosignalVariance1HE2->Clone("recosignalVarianceHE2");
  recosignalVarianceHE2->Divide(recosignalVariance1HE2, recosignalVariance0HE2, 1, 1, "B");
  TH2F* recosignalVariance1HE3 = (TH2F*)dir->FindObjectAny("h_recSignalEnergy2_HE3");
  TH2F* recosignalVariance0HE3 = (TH2F*)dir->FindObjectAny("h_recSignalEnergy0_HE3");
  TH2F* recosignalVarianceHE3 = (TH2F*)recosignalVariance1HE3->Clone("recosignalVarianceHE3");
  recosignalVarianceHE3->Divide(recosignalVariance1HE3, recosignalVariance0HE3, 1, 1, "B");
  TH2F* recosignalVariance1HE4 = (TH2F*)dir->FindObjectAny("h_recSignalEnergy2_HE4");
  TH2F* recosignalVariance0HE4 = (TH2F*)dir->FindObjectAny("h_recSignalEnergy0_HE4");
  TH2F* recosignalVarianceHE4 = (TH2F*)recosignalVariance1HE4->Clone("recosignalVarianceHE4");
  recosignalVarianceHE4->Divide(recosignalVariance1HE4, recosignalVariance0HE4, 1, 1, "B");
  TH2F* recosignalVariance1HE5 = (TH2F*)dir->FindObjectAny("h_recSignalEnergy2_HE5");
  TH2F* recosignalVariance0HE5 = (TH2F*)dir->FindObjectAny("h_recSignalEnergy0_HE5");
  TH2F* recosignalVarianceHE5 = (TH2F*)recosignalVariance1HE5->Clone("recosignalVarianceHE5");
  recosignalVarianceHE5->Divide(recosignalVariance1HE5, recosignalVariance0HE5, 1, 1, "B");
  TH2F* recosignalVariance1HE6 = (TH2F*)dir->FindObjectAny("h_recSignalEnergy2_HE6");
  TH2F* recosignalVariance0HE6 = (TH2F*)dir->FindObjectAny("h_recSignalEnergy0_HE6");
  TH2F* recosignalVarianceHE6 = (TH2F*)recosignalVariance1HE6->Clone("recosignalVarianceHE6");
  recosignalVarianceHE6->Divide(recosignalVariance1HE6, recosignalVariance0HE6, 1, 1, "B");
  TH2F* recosignalVariance1HE7 = (TH2F*)dir->FindObjectAny("h_recSignalEnergy2_HE7");
  TH2F* recosignalVariance0HE7 = (TH2F*)dir->FindObjectAny("h_recSignalEnergy0_HE7");
  TH2F* recosignalVarianceHE7 = (TH2F*)recosignalVariance1HE7->Clone("recosignalVarianceHE7");
  recosignalVarianceHE7->Divide(recosignalVariance1HE7, recosignalVariance0HE7, 1, 1, "B");
  //cout<<"      Vaiance: preparation DONE *****" <<endl;
  //====================================================================== put Vaiance=Dispersia = Sig**2=<R**2> - (<R>)**2 into massive recosignalvariancehe
  //                                                                                           = sum(R*R)/N - (sum(R)/N)**2
  for (int jeta = 0; jeta < njeta; jeta++) {
    //preparation for PHI normalization:
    double sumrecosignalHE0 = 0;
    int nsumrecosignalHE0 = 0;
    double sumrecosignalHE1 = 0;
    int nsumrecosignalHE1 = 0;
    double sumrecosignalHE2 = 0;
    int nsumrecosignalHE2 = 0;
    double sumrecosignalHE3 = 0;
    int nsumrecosignalHE3 = 0;
    double sumrecosignalHE4 = 0;
    int nsumrecosignalHE4 = 0;
    double sumrecosignalHE5 = 0;
    int nsumrecosignalHE5 = 0;
    double sumrecosignalHE6 = 0;
    int nsumrecosignalHE6 = 0;
    for (int jphi = 0; jphi < njphi; jphi++) {
      recosignalvariancehe[0][jeta][jphi] = recosignalVarianceHE1->GetBinContent(jeta + 1, jphi + 1);
      recosignalvariancehe[1][jeta][jphi] = recosignalVarianceHE2->GetBinContent(jeta + 1, jphi + 1);
      recosignalvariancehe[2][jeta][jphi] = recosignalVarianceHE3->GetBinContent(jeta + 1, jphi + 1);
      recosignalvariancehe[3][jeta][jphi] = recosignalVarianceHE4->GetBinContent(jeta + 1, jphi + 1);
      recosignalvariancehe[4][jeta][jphi] = recosignalVarianceHE5->GetBinContent(jeta + 1, jphi + 1);
      recosignalvariancehe[5][jeta][jphi] = recosignalVarianceHE6->GetBinContent(jeta + 1, jphi + 1);
      recosignalvariancehe[6][jeta][jphi] = recosignalVarianceHE7->GetBinContent(jeta + 1, jphi + 1);
      if (recosignalvariancehe[0][jeta][jphi] > 0.) {
        sumrecosignalHE0 += recosignalvariancehe[0][jeta][jphi];
        ++nsumrecosignalHE0;
      }
      if (recosignalvariancehe[1][jeta][jphi] > 0.) {
        sumrecosignalHE1 += recosignalvariancehe[1][jeta][jphi];
        ++nsumrecosignalHE1;
      }
      if (recosignalvariancehe[2][jeta][jphi] > 0.) {
        sumrecosignalHE2 += recosignalvariancehe[2][jeta][jphi];
        ++nsumrecosignalHE2;
      }
      if (recosignalvariancehe[3][jeta][jphi] > 0.) {
        sumrecosignalHE3 += recosignalvariancehe[3][jeta][jphi];
        ++nsumrecosignalHE3;
      }
      if (recosignalvariancehe[4][jeta][jphi] > 0.) {
        sumrecosignalHE4 += recosignalvariancehe[4][jeta][jphi];
        ++nsumrecosignalHE4;
      }
      if (recosignalvariancehe[5][jeta][jphi] > 0.) {
        sumrecosignalHE5 += recosignalvariancehe[5][jeta][jphi];
        ++nsumrecosignalHE5;
      }
      if (recosignalvariancehe[6][jeta][jphi] > 0.) {
        sumrecosignalHE6 += recosignalvariancehe[6][jeta][jphi];
        ++nsumrecosignalHE6;
      }
    }  // phi
    // PHI normalization :
    for (int jphi = 0; jphi < njphi; jphi++) {
      if (recosignalvariancehe[0][jeta][jphi] > 0.)
        recosignalvariancehe[0][jeta][jphi] /= (sumrecosignalHE0 / nsumrecosignalHE0);
      if (recosignalvariancehe[1][jeta][jphi] > 0.)
        recosignalvariancehe[1][jeta][jphi] /= (sumrecosignalHE1 / nsumrecosignalHE1);
      if (recosignalvariancehe[2][jeta][jphi] > 0.)
        recosignalvariancehe[2][jeta][jphi] /= (sumrecosignalHE2 / nsumrecosignalHE2);
      if (recosignalvariancehe[3][jeta][jphi] > 0.)
        recosignalvariancehe[3][jeta][jphi] /= (sumrecosignalHE3 / nsumrecosignalHE3);
      if (recosignalvariancehe[4][jeta][jphi] > 0.)
        recosignalvariancehe[4][jeta][jphi] /= (sumrecosignalHE4 / nsumrecosignalHE4);
      if (recosignalvariancehe[5][jeta][jphi] > 0.)
        recosignalvariancehe[5][jeta][jphi] /= (sumrecosignalHE5 / nsumrecosignalHE5);
      if (recosignalvariancehe[6][jeta][jphi] > 0.)
        recosignalvariancehe[6][jeta][jphi] /= (sumrecosignalHE6 / nsumrecosignalHE6);
    }  // phi
    //       recosignalvariancehe (D)           = sum(R*R)/N - (sum(R)/N)**2
    for (int jphi = 0; jphi < njphi; jphi++) {
      //	   cout<<"12 12 12   jeta=     "<< jeta <<"   jphi   =     "<<jphi  <<endl;
      recosignalvariancehe[0][jeta][jphi] -= arecosignalhe[0][jeta][jphi] * arecosignalhe[0][jeta][jphi];
      recosignalvariancehe[0][jeta][jphi] = fabs(recosignalvariancehe[0][jeta][jphi]);
      recosignalvariancehe[1][jeta][jphi] -= arecosignalhe[1][jeta][jphi] * arecosignalhe[1][jeta][jphi];
      recosignalvariancehe[1][jeta][jphi] = fabs(recosignalvariancehe[1][jeta][jphi]);
      recosignalvariancehe[2][jeta][jphi] -= arecosignalhe[2][jeta][jphi] * arecosignalhe[2][jeta][jphi];
      recosignalvariancehe[2][jeta][jphi] = fabs(recosignalvariancehe[2][jeta][jphi]);
      recosignalvariancehe[3][jeta][jphi] -= arecosignalhe[3][jeta][jphi] * arecosignalhe[3][jeta][jphi];
      recosignalvariancehe[3][jeta][jphi] = fabs(recosignalvariancehe[3][jeta][jphi]);
      recosignalvariancehe[4][jeta][jphi] -= arecosignalhe[4][jeta][jphi] * arecosignalhe[4][jeta][jphi];
      recosignalvariancehe[4][jeta][jphi] = fabs(recosignalvariancehe[4][jeta][jphi]);
      recosignalvariancehe[5][jeta][jphi] -= arecosignalhe[5][jeta][jphi] * arecosignalhe[5][jeta][jphi];
      recosignalvariancehe[5][jeta][jphi] = fabs(recosignalvariancehe[5][jeta][jphi]);
      recosignalvariancehe[6][jeta][jphi] -= arecosignalhe[6][jeta][jphi] * arecosignalhe[6][jeta][jphi];
      recosignalvariancehe[6][jeta][jphi] = fabs(recosignalvariancehe[6][jeta][jphi]);
    }
  }
  //cout<<"      Vaiance: DONE*****" <<endl;
  //------------------------  2D-eta/phi-plot: D, averaged over depthes
  //======================================================================
  //======================================================================
  //cout<<"      R2D-eta/phi-plot: D, averaged over depthes *****" <<endl;
  c1x1->Clear();
  /////////////////
  c1x0->Divide(1, 1);
  c1x0->cd(1);
  TH2F* DefzDrecosignalHE42D = new TH2F("DefzDrecosignalHE42D", "", neta, -41., 41., nphi, 0., 72.);
  TH2F* DefzDrecosignalHE42D0 = new TH2F("DefzDrecosignalHE42D0", "", neta, -41., 41., nphi, 0., 72.);
  TH2F* DefzDrecosignalHE42DF = (TH2F*)DefzDrecosignalHE42D0->Clone("DefzDrecosignalHE42DF");
  for (int i = 0; i < ndepth; i++) {
    for (int jeta = 0; jeta < neta; jeta++) {
      for (int jphi = 0; jphi < nphi; jphi++) {
        double ccc1 = recosignalvariancehe[i][jeta][jphi];
        int k2plot = jeta - 41;
        int kkk = k2plot;  //if(k2plot >0   kkk=k2plot+1; //-41 +41 !=0
        if (arecosignalhe[i][jeta][jphi] > 0.) {
          DefzDrecosignalHE42D->Fill(kkk, jphi, ccc1);
          DefzDrecosignalHE42D0->Fill(kkk, jphi, 1.);
        }
      }
    }
  }
  DefzDrecosignalHE42DF->Divide(DefzDrecosignalHE42D, DefzDrecosignalHE42D0, 1, 1, "B");  // average A
  //    DefzDrecosignalHE1->Sumw2();
  gPad->SetGridy();
  gPad->SetGridx();  //      gPad->SetLogz();
  DefzDrecosignalHE42DF->SetMarkerStyle(20);
  DefzDrecosignalHE42DF->SetMarkerSize(0.4);
  DefzDrecosignalHE42DF->GetZaxis()->SetLabelSize(0.08);
  DefzDrecosignalHE42DF->SetXTitle("<D>_depth       #eta  \b");
  DefzDrecosignalHE42DF->SetYTitle("      #phi \b");
  DefzDrecosignalHE42DF->SetZTitle("<D>_depth \b");
  DefzDrecosignalHE42DF->SetMarkerColor(2);
  DefzDrecosignalHE42DF->SetLineColor(
      0);  //      DefzDrecosignalHE42DF->SetMaximum(1.000);  //      DefzDrecosignalHE42DF->SetMinimum(1.0);
  DefzDrecosignalHE42DF->Draw("COLZ");
  /////////////////
  c1x0->Update();
  c1x0->Print("DrecosignalGeneralD2PhiSymmetryHE.png");
  c1x0->Clear();
  // clean-up
  if (DefzDrecosignalHE42D)
    delete DefzDrecosignalHE42D;
  if (DefzDrecosignalHE42D0)
    delete DefzDrecosignalHE42D0;
  if (DefzDrecosignalHE42DF)
    delete DefzDrecosignalHE42DF;
  //====================================================================== 1D plot: D vs phi , averaged over depthes & eta
  //======================================================================
  //cout<<"      1D plot: D vs phi , averaged over depthes & eta *****" <<endl;
  c1x1->Clear();
  /////////////////
  c1x1->Divide(1, 1);
  c1x1->cd(1);
  TH1F* DefzDrecosignalHE41D = new TH1F("DefzDrecosignalHE41D", "", nphi, 0., 72.);
  TH1F* DefzDrecosignalHE41D0 = new TH1F("DefzDrecosignalHE41D0", "", nphi, 0., 72.);
  TH1F* DefzDrecosignalHE41DF = (TH1F*)DefzDrecosignalHE41D0->Clone("DefzDrecosignalHE41DF");

  for (int jphi = 0; jphi < nphi; jphi++) {
    for (int jeta = 0; jeta < neta; jeta++) {
      for (int i = 0; i < ndepth; i++) {
        double ccc1 = recosignalvariancehe[i][jeta][jphi];
        if (arecosignalhe[i][jeta][jphi] > 0.) {
          DefzDrecosignalHE41D->Fill(jphi, ccc1);
          DefzDrecosignalHE41D0->Fill(jphi, 1.);
        }
      }
    }
  }
  //     DefzDrecosignalHE41D->Sumw2();DefzDrecosignalHE41D0->Sumw2();

  DefzDrecosignalHE41DF->Divide(
      DefzDrecosignalHE41D, DefzDrecosignalHE41D0, 1, 1, "B");  // R averaged over depthes & eta
  DefzDrecosignalHE41D0->Sumw2();
  //    for (int jphi=1;jphi<73;jphi++) {DefzDrecosignalHE41DF->SetBinError(jphi,0.01);}
  gPad->SetGridy();
  gPad->SetGridx();  //      gPad->SetLogz();
  DefzDrecosignalHE41DF->SetMarkerStyle(20);
  DefzDrecosignalHE41DF->SetMarkerSize(1.4);
  DefzDrecosignalHE41DF->GetZaxis()->SetLabelSize(0.08);
  DefzDrecosignalHE41DF->SetXTitle("#phi  \b");
  DefzDrecosignalHE41DF->SetYTitle("  <D> \b");
  DefzDrecosignalHE41DF->SetZTitle("<D>_PHI  - AllDepthes \b");
  DefzDrecosignalHE41DF->SetMarkerColor(4);
  DefzDrecosignalHE41DF->SetLineColor(
      4);  // DefzDrecosignalHE41DF->SetMinimum(0.8);     DefzDrecosignalHE41DF->SetMinimum(-0.015);
  DefzDrecosignalHE41DF->Draw("Error");
  /////////////////
  c1x1->Update();
  c1x1->Print("DrecosignalGeneralD1PhiSymmetryHE.png");
  c1x1->Clear();
  // clean-up
  if (DefzDrecosignalHE41D)
    delete DefzDrecosignalHE41D;
  if (DefzDrecosignalHE41D0)
    delete DefzDrecosignalHE41D0;
  if (DefzDrecosignalHE41DF)
    delete DefzDrecosignalHE41DF;
  //========================================================================================== 14
  //======================================================================
  //======================================================================1D plot: D vs phi , different eta,  depth=1
  //cout<<"      1D plot: D vs phi , different eta,  depth=1 *****" <<endl;
  c3x5->Clear();
  /////////////////
  c3x5->Divide(3, 5);
  c3x5->cd(1);
  int kcountHEpositivedirectionRecosignalD1 = 1;
  TH1F* h2CeffHEpositivedirectionRecosignalD1 = new TH1F("h2CeffHEpositivedirectionRecosignalD1", "", nphi, 0., 72.);

  for (int jeta = 0; jeta < njeta; jeta++) {
    // positivedirectionRecosignalD:
    if (jeta - 41 >= 0) {
      //	     for (int i=0;i<ndepth;i++) {
      // depth=1
      for (int i = 0; i < 1; i++) {
        TH1F* HEpositivedirectionRecosignalD1 = (TH1F*)h2CeffHEpositivedirectionRecosignalD1->Clone("twod1");

        float ccctest = 0;  // to avoid empty massive elements
        for (int jphi = 0; jphi < nphi; jphi++) {
          double ccc1 = recosignalvariancehe[i][jeta][jphi];
          if (arecosignalhe[i][jeta][jphi] > 0.) {
            HEpositivedirectionRecosignalD1->Fill(jphi, ccc1);
            ccctest = 1.;  //HEpositivedirectionRecosignalD1->SetBinError(i,0.01);
          }
        }  // for jphi
        if (ccctest > 0.) {
          //cout<<"1414       kcountHEpositivedirectionRecosignalD1   =     "<<kcountHEpositivedirectionRecosignalD1  <<"   jeta-41=     "<< jeta-41 <<endl;
          c3x5->cd(kcountHEpositivedirectionRecosignalD1);
          HEpositivedirectionRecosignalD1->SetMarkerStyle(20);
          HEpositivedirectionRecosignalD1->SetMarkerSize(0.4);
          HEpositivedirectionRecosignalD1->GetYaxis()->SetLabelSize(0.04);
          HEpositivedirectionRecosignalD1->SetXTitle("HEpositivedirectionRecosignalD1 \b");
          HEpositivedirectionRecosignalD1->SetMarkerColor(2);
          HEpositivedirectionRecosignalD1->SetLineColor(0);
          gPad->SetGridy();
          gPad->SetGridx();
          //	   gPad->SetLogy();
          if (kcountHEpositivedirectionRecosignalD1 == 1)
            HEpositivedirectionRecosignalD1->SetXTitle("D for HE+ jeta = 17; depth = 1 \b");
          if (kcountHEpositivedirectionRecosignalD1 == 2)
            HEpositivedirectionRecosignalD1->SetXTitle("D for HE+ jeta = 18; depth = 1 \b");
          if (kcountHEpositivedirectionRecosignalD1 == 3)
            HEpositivedirectionRecosignalD1->SetXTitle("D for HE+ jeta = 19; depth = 1 \b");
          if (kcountHEpositivedirectionRecosignalD1 == 4)
            HEpositivedirectionRecosignalD1->SetXTitle("D for HE+ jeta = 20; depth = 1 \b");
          if (kcountHEpositivedirectionRecosignalD1 == 5)
            HEpositivedirectionRecosignalD1->SetXTitle("D for HE+ jeta = 21; depth = 1 \b");
          if (kcountHEpositivedirectionRecosignalD1 == 6)
            HEpositivedirectionRecosignalD1->SetXTitle("D for HE+ jeta = 22; depth = 1 \b");
          if (kcountHEpositivedirectionRecosignalD1 == 7)
            HEpositivedirectionRecosignalD1->SetXTitle("D for HE+ jeta = 23; depth = 1 \b");
          if (kcountHEpositivedirectionRecosignalD1 == 8)
            HEpositivedirectionRecosignalD1->SetXTitle("D for HE+ jeta = 24; depth = 1 \b");
          if (kcountHEpositivedirectionRecosignalD1 == 9)
            HEpositivedirectionRecosignalD1->SetXTitle("D for HE+ jeta = 25; depth = 1 \b");
          if (kcountHEpositivedirectionRecosignalD1 == 10)
            HEpositivedirectionRecosignalD1->SetXTitle("D for HE+ jeta = 26; depth = 1 \b");
          if (kcountHEpositivedirectionRecosignalD1 == 11)
            HEpositivedirectionRecosignalD1->SetXTitle("D for HE+ jeta = 27; depth = 1 \b");
          if (kcountHEpositivedirectionRecosignalD1 == 12)
            HEpositivedirectionRecosignalD1->SetXTitle("D for HE+ jeta = 28; depth = 1 \b");
          HEpositivedirectionRecosignalD1->Draw("Error");
          kcountHEpositivedirectionRecosignalD1++;
          if (kcountHEpositivedirectionRecosignalD1 > 12)
            break;  // 4x6 = 24
        }           //ccctest>0

      }  // for i
    }    //if(jeta-41 >= 0)
  }      //for jeta
  /////////////////
  c3x5->Update();
  c3x5->Print("DrecosignalPositiveDirectionhistD1PhiSymmetryDepth1HE.png");
  c3x5->Clear();
  // clean-up
  if (h2CeffHEpositivedirectionRecosignalD1)
    delete h2CeffHEpositivedirectionRecosignalD1;
  //========================================================================================== 15
  //======================================================================
  //======================================================================1D plot: D vs phi , different eta,  depth=2
  //cout<<"      1D plot: D vs phi , different eta,  depth=2 *****" <<endl;
  c3x5->Clear();
  c3x5->Divide(3, 5);
  c3x5->cd(1);
  int kcountHEpositivedirectionRecosignalD2 = 1;
  TH1F* h2CeffHEpositivedirectionRecosignalD2 = new TH1F("h2CeffHEpositivedirectionRecosignalD2", "", nphi, 0., 72.);

  for (int jeta = 0; jeta < njeta; jeta++) {
    // positivedirectionRecosignalD:
    if (jeta - 41 >= 0) {
      //	     for (int i=0;i<ndepth;i++) {
      // depth=2
      for (int i = 1; i < 2; i++) {
        TH1F* HEpositivedirectionRecosignalD2 = (TH1F*)h2CeffHEpositivedirectionRecosignalD2->Clone("twod1");

        float ccctest = 0;  // to avoid empty massive elements
        for (int jphi = 0; jphi < nphi; jphi++) {
          double ccc1 = recosignalvariancehe[i][jeta][jphi];
          if (arecosignalhe[i][jeta][jphi] > 0.) {
            HEpositivedirectionRecosignalD2->Fill(jphi, ccc1);
            ccctest = 1.;  //HEpositivedirectionRecosignalD2->SetBinError(i,0.01);
          }
        }  // for jphi
        if (ccctest > 0.) {
          //cout<<"1515       kcountHEpositivedirectionRecosignalD2   =     "<<kcountHEpositivedirectionRecosignalD2  <<"   jeta-41=     "<< jeta-41 <<endl;
          c3x5->cd(kcountHEpositivedirectionRecosignalD2);
          HEpositivedirectionRecosignalD2->SetMarkerStyle(20);
          HEpositivedirectionRecosignalD2->SetMarkerSize(0.4);
          HEpositivedirectionRecosignalD2->GetYaxis()->SetLabelSize(0.04);
          HEpositivedirectionRecosignalD2->SetXTitle("HEpositivedirectionRecosignalD2 \b");
          HEpositivedirectionRecosignalD2->SetMarkerColor(2);
          HEpositivedirectionRecosignalD2->SetLineColor(0);
          gPad->SetGridy();
          gPad->SetGridx();
          //	   gPad->SetLogy();
          if (kcountHEpositivedirectionRecosignalD2 == 1)
            HEpositivedirectionRecosignalD2->SetXTitle("D for HE+ jeta = 16; depth = 2 \b");
          if (kcountHEpositivedirectionRecosignalD2 == 2)
            HEpositivedirectionRecosignalD2->SetXTitle("D for HE+ jeta = 17; depth = 2 \b");
          if (kcountHEpositivedirectionRecosignalD2 == 3)
            HEpositivedirectionRecosignalD2->SetXTitle("D for HE+ jeta = 18; depth = 2 \b");
          if (kcountHEpositivedirectionRecosignalD2 == 4)
            HEpositivedirectionRecosignalD2->SetXTitle("D for HE+ jeta = 19; depth = 2 \b");
          if (kcountHEpositivedirectionRecosignalD2 == 5)
            HEpositivedirectionRecosignalD2->SetXTitle("D for HE+ jeta = 20; depth = 2 \b");
          if (kcountHEpositivedirectionRecosignalD2 == 6)
            HEpositivedirectionRecosignalD2->SetXTitle("D for HE+ jeta = 21; depth = 2 \b");
          if (kcountHEpositivedirectionRecosignalD2 == 7)
            HEpositivedirectionRecosignalD2->SetXTitle("D for HE+ jeta = 22; depth = 2 \b");
          if (kcountHEpositivedirectionRecosignalD2 == 8)
            HEpositivedirectionRecosignalD2->SetXTitle("D for HE+ jeta = 23; depth = 2 \b");
          if (kcountHEpositivedirectionRecosignalD2 == 9)
            HEpositivedirectionRecosignalD2->SetXTitle("D for HE+ jeta = 24; depth = 2 \b");
          if (kcountHEpositivedirectionRecosignalD2 == 10)
            HEpositivedirectionRecosignalD2->SetXTitle("D for HE+ jeta = 25; depth = 2 \b");
          if (kcountHEpositivedirectionRecosignalD2 == 11)
            HEpositivedirectionRecosignalD2->SetXTitle("D for HE+ jeta = 26; depth = 2 \b");
          if (kcountHEpositivedirectionRecosignalD2 == 12)
            HEpositivedirectionRecosignalD2->SetXTitle("D for HE+ jeta = 27; depth = 2 \b");
          if (kcountHEpositivedirectionRecosignalD2 == 13)
            HEpositivedirectionRecosignalD2->SetXTitle("D for HE+ jeta = 28; depth = 2 \b");
          HEpositivedirectionRecosignalD2->Draw("Error");
          kcountHEpositivedirectionRecosignalD2++;
          if (kcountHEpositivedirectionRecosignalD2 > 13)
            break;  // 4x6 = 24
        }           //ccctest>0

      }  // for i
    }    //if(jeta-41 >= 0)
  }      //for jeta
  /////////////////
  c3x5->Update();
  c3x5->Print("DrecosignalPositiveDirectionhistD1PhiSymmetryDepth2HE.png");
  c3x5->Clear();
  // clean-up
  if (h2CeffHEpositivedirectionRecosignalD2)
    delete h2CeffHEpositivedirectionRecosignalD2;
  //========================================================================================== 16
  //======================================================================
  //======================================================================1D plot: D vs phi , different eta,  depth=3
  //cout<<"      1D plot: D vs phi , different eta,  depth=3 *****" <<endl;
  c3x5->Clear();
  c3x5->Divide(3, 5);
  c3x5->cd(1);
  int kcountHEpositivedirectionRecosignalD3 = 1;
  TH1F* h2CeffHEpositivedirectionRecosignalD3 = new TH1F("h2CeffHEpositivedirectionRecosignalD3", "", nphi, 0., 72.);

  for (int jeta = 0; jeta < njeta; jeta++) {
    // positivedirectionRecosignalD:
    if (jeta - 41 >= 0) {
      //	     for (int i=0;i<ndepth;i++) {
      // depth=3
      for (int i = 2; i < 3; i++) {
        TH1F* HEpositivedirectionRecosignalD3 = (TH1F*)h2CeffHEpositivedirectionRecosignalD3->Clone("twod1");

        float ccctest = 0;  // to avoid empty massive elements
        for (int jphi = 0; jphi < nphi; jphi++) {
          double ccc1 = recosignalvariancehe[i][jeta][jphi];
          if (arecosignalhe[i][jeta][jphi] > 0.) {
            HEpositivedirectionRecosignalD3->Fill(jphi, ccc1);
            ccctest = 1.;  //HEpositivedirectionRecosignalD3->SetBinError(i,0.01);
          }
        }  // for jphi
        if (ccctest > 0.) {
          //cout<<"1616       kcountHEpositivedirectionRecosignalD3   =     "<<kcountHEpositivedirectionRecosignalD3  <<"   jeta-41=     "<< jeta-41 <<endl;
          c3x5->cd(kcountHEpositivedirectionRecosignalD3);
          HEpositivedirectionRecosignalD3->SetMarkerStyle(20);
          HEpositivedirectionRecosignalD3->SetMarkerSize(0.4);
          HEpositivedirectionRecosignalD3->GetYaxis()->SetLabelSize(0.04);
          HEpositivedirectionRecosignalD3->SetXTitle("HEpositivedirectionRecosignalD3 \b");
          HEpositivedirectionRecosignalD3->SetMarkerColor(2);
          HEpositivedirectionRecosignalD3->SetLineColor(0);
          gPad->SetGridy();
          gPad->SetGridx();
          //	   gPad->SetLogy();
          if (kcountHEpositivedirectionRecosignalD3 == 1)
            HEpositivedirectionRecosignalD3->SetXTitle("D for HE+ jeta = 16; depth = 3 \b");
          if (kcountHEpositivedirectionRecosignalD3 == 2)
            HEpositivedirectionRecosignalD3->SetXTitle("D for HE+ jeta = 17; depth = 3 \b");
          if (kcountHEpositivedirectionRecosignalD3 == 3)
            HEpositivedirectionRecosignalD3->SetXTitle("D for HE+ jeta = 18; depth = 3 \b");
          if (kcountHEpositivedirectionRecosignalD3 == 4)
            HEpositivedirectionRecosignalD3->SetXTitle("D for HE+ jeta = 19; depth = 3 \b");
          if (kcountHEpositivedirectionRecosignalD3 == 5)
            HEpositivedirectionRecosignalD3->SetXTitle("D for HE+ jeta = 20; depth = 3 \b");
          if (kcountHEpositivedirectionRecosignalD3 == 6)
            HEpositivedirectionRecosignalD3->SetXTitle("D for HE+ jeta = 21; depth = 3 \b");
          if (kcountHEpositivedirectionRecosignalD3 == 7)
            HEpositivedirectionRecosignalD3->SetXTitle("D for HE+ jeta = 22; depth = 3 \b");
          if (kcountHEpositivedirectionRecosignalD3 == 8)
            HEpositivedirectionRecosignalD3->SetXTitle("D for HE+ jeta = 23; depth = 3 \b");
          if (kcountHEpositivedirectionRecosignalD3 == 9)
            HEpositivedirectionRecosignalD3->SetXTitle("D for HE+ jeta = 24; depth = 3 \b");
          if (kcountHEpositivedirectionRecosignalD3 == 10)
            HEpositivedirectionRecosignalD3->SetXTitle("D for HE+ jeta = 25; depth = 3 \b");
          if (kcountHEpositivedirectionRecosignalD3 == 11)
            HEpositivedirectionRecosignalD3->SetXTitle("D for HE+ jeta = 26; depth = 3 \b");
          if (kcountHEpositivedirectionRecosignalD3 == 12)
            HEpositivedirectionRecosignalD3->SetXTitle("D for HE+ jeta = 27; depth = 3 \b");
          if (kcountHEpositivedirectionRecosignalD3 == 13)
            HEpositivedirectionRecosignalD3->SetXTitle("D for HE+ jeta = 28; depth = 3 \b");
          HEpositivedirectionRecosignalD3->Draw("Error");
          kcountHEpositivedirectionRecosignalD3++;
          if (kcountHEpositivedirectionRecosignalD3 > 13)
            break;  // 4x6 = 24
        }           //ccctest>0

      }  // for i
    }    //if(jeta-41 >= 0)
  }      //for jeta
  /////////////////
  c3x5->Update();
  c3x5->Print("DrecosignalPositiveDirectionhistD1PhiSymmetryDepth3HE.png");
  c3x5->Clear();
  // clean-up
  if (h2CeffHEpositivedirectionRecosignalD3)
    delete h2CeffHEpositivedirectionRecosignalD3;
  //========================================================================================== 17
  //======================================================================
  //======================================================================1D plot: D vs phi , different eta,  depth=4
  //cout<<"      1D plot: D vs phi , different eta,  depth=4 *****" <<endl;
  c3x5->Clear();
  c3x5->Divide(3, 5);
  c3x5->cd(1);
  int kcountHEpositivedirectionRecosignalD4 = 1;
  TH1F* h2CeffHEpositivedirectionRecosignalD4 = new TH1F("h2CeffHEpositivedirectionRecosignalD4", "", nphi, 0., 72.);

  for (int jeta = 0; jeta < njeta; jeta++) {
    // positivedirectionRecosignalD:
    if (jeta - 41 >= 0) {
      //	     for (int i=0;i<ndepth;i++) {
      // depth=4
      for (int i = 3; i < 4; i++) {
        TH1F* HEpositivedirectionRecosignalD4 = (TH1F*)h2CeffHEpositivedirectionRecosignalD4->Clone("twod1");

        float ccctest = 0;  // to avoid empty massive elements
        for (int jphi = 0; jphi < nphi; jphi++) {
          double ccc1 = recosignalvariancehe[i][jeta][jphi];
          if (arecosignalhe[i][jeta][jphi] > 0.) {
            HEpositivedirectionRecosignalD4->Fill(jphi, ccc1);
            ccctest = 1.;  //HEpositivedirectionRecosignalD4->SetBinError(i,0.01);
          }
        }  // for jphi
        if (ccctest > 0.) {
          //cout<<"1717       kcountHEpositivedirectionRecosignalD4   =     "<<kcountHEpositivedirectionRecosignalD4  <<"   jeta-41=     "<< jeta-41 <<endl;
          c3x5->cd(kcountHEpositivedirectionRecosignalD4);
          HEpositivedirectionRecosignalD4->SetMarkerStyle(20);
          HEpositivedirectionRecosignalD4->SetMarkerSize(0.4);
          HEpositivedirectionRecosignalD4->GetYaxis()->SetLabelSize(0.04);
          HEpositivedirectionRecosignalD4->SetXTitle("HEpositivedirectionRecosignalD4 \b");
          HEpositivedirectionRecosignalD4->SetMarkerColor(2);
          HEpositivedirectionRecosignalD4->SetLineColor(0);
          gPad->SetGridy();
          gPad->SetGridx();
          //	   gPad->SetLogy();
          if (kcountHEpositivedirectionRecosignalD4 == 1)
            HEpositivedirectionRecosignalD4->SetXTitle("D for HE+ jeta = 15; depth = 4 \b");
          if (kcountHEpositivedirectionRecosignalD4 == 2)
            HEpositivedirectionRecosignalD4->SetXTitle("D for HE+ jeta = 17; depth = 4 \b");
          if (kcountHEpositivedirectionRecosignalD4 == 3)
            HEpositivedirectionRecosignalD4->SetXTitle("D for HE+ jeta = 18; depth = 4 \b");
          if (kcountHEpositivedirectionRecosignalD4 == 4)
            HEpositivedirectionRecosignalD4->SetXTitle("D for HE+ jeta = 19; depth = 4 \b");
          if (kcountHEpositivedirectionRecosignalD4 == 5)
            HEpositivedirectionRecosignalD4->SetXTitle("D for HE+ jeta = 20; depth = 4 \b");
          if (kcountHEpositivedirectionRecosignalD4 == 6)
            HEpositivedirectionRecosignalD4->SetXTitle("D for HE+ jeta = 21; depth = 4 \b");
          if (kcountHEpositivedirectionRecosignalD4 == 7)
            HEpositivedirectionRecosignalD4->SetXTitle("D for HE+ jeta = 22; depth = 4 \b");
          if (kcountHEpositivedirectionRecosignalD4 == 8)
            HEpositivedirectionRecosignalD4->SetXTitle("D for HE+ jeta = 23; depth = 4 \b");
          if (kcountHEpositivedirectionRecosignalD4 == 9)
            HEpositivedirectionRecosignalD4->SetXTitle("D for HE+ jeta = 24; depth = 4 \b");
          if (kcountHEpositivedirectionRecosignalD4 == 10)
            HEpositivedirectionRecosignalD4->SetXTitle("D for HE+ jeta = 25; depth = 4 \b");
          if (kcountHEpositivedirectionRecosignalD4 == 11)
            HEpositivedirectionRecosignalD4->SetXTitle("D for HE+ jeta = 26; depth = 4 \b");
          if (kcountHEpositivedirectionRecosignalD4 == 12)
            HEpositivedirectionRecosignalD4->SetXTitle("D for HE+ jeta = 27; depth = 4 \b");
          HEpositivedirectionRecosignalD4->Draw("Error");
          kcountHEpositivedirectionRecosignalD4++;
          if (kcountHEpositivedirectionRecosignalD4 > 12)
            break;  // 4x6 = 24
        }           //ccctest>0

      }  // for i
    }    //if(jeta-41 >= 0)
  }      //for jeta
  /////////////////
  c3x5->Update();
  c3x5->Print("DrecosignalPositiveDirectionhistD1PhiSymmetryDepth4HE.png");
  c3x5->Clear();
  // clean-up
  if (h2CeffHEpositivedirectionRecosignalD4)
    delete h2CeffHEpositivedirectionRecosignalD4;
  //========================================================================================== 18
  //======================================================================
  //======================================================================1D plot: D vs phi , different eta,  depth=5
  //cout<<"      1D plot: D vs phi , different eta,  depth=5 *****" <<endl;
  c3x5->Clear();
  c3x5->Divide(3, 5);
  c3x5->cd(1);
  int kcountHEpositivedirectionRecosignalD5 = 1;
  TH1F* h2CeffHEpositivedirectionRecosignalD5 = new TH1F("h2CeffHEpositivedirectionRecosignalD5", "", nphi, 0., 72.);

  for (int jeta = 0; jeta < njeta; jeta++) {
    // positivedirectionRecosignalD:
    if (jeta - 41 >= 0) {
      //	     for (int i=0;i<ndepth;i++) {
      // depth=5
      for (int i = 4; i < 5; i++) {
        TH1F* HEpositivedirectionRecosignalD5 = (TH1F*)h2CeffHEpositivedirectionRecosignalD5->Clone("twod1");

        float ccctest = 0;  // to avoid empty massive elements
        for (int jphi = 0; jphi < nphi; jphi++) {
          double ccc1 = recosignalvariancehe[i][jeta][jphi];
          if (arecosignalhe[i][jeta][jphi] > 0.) {
            HEpositivedirectionRecosignalD5->Fill(jphi, ccc1);
            ccctest = 1.;  //HEpositivedirectionRecosignalD5->SetBinError(i,0.01);
          }
        }  // for jphi
        if (ccctest > 0.) {
          //cout<<"1818       kcountHEpositivedirectionRecosignalD5   =     "<<kcountHEpositivedirectionRecosignalD5  <<"   jeta-41=     "<< jeta-41 <<endl;
          c3x5->cd(kcountHEpositivedirectionRecosignalD5);
          HEpositivedirectionRecosignalD5->SetMarkerStyle(20);
          HEpositivedirectionRecosignalD5->SetMarkerSize(0.4);
          HEpositivedirectionRecosignalD5->GetYaxis()->SetLabelSize(0.04);
          HEpositivedirectionRecosignalD5->SetXTitle("HEpositivedirectionRecosignalD5 \b");
          HEpositivedirectionRecosignalD5->SetMarkerColor(2);
          HEpositivedirectionRecosignalD5->SetLineColor(0);
          gPad->SetGridy();
          gPad->SetGridx();
          //	   gPad->SetLogy();
          if (kcountHEpositivedirectionRecosignalD5 == 1)
            HEpositivedirectionRecosignalD5->SetXTitle("D for HE+ jeta = 17; depth = 5 \b");
          if (kcountHEpositivedirectionRecosignalD5 == 2)
            HEpositivedirectionRecosignalD5->SetXTitle("D for HE+ jeta = 18; depth = 5 \b");
          if (kcountHEpositivedirectionRecosignalD5 == 3)
            HEpositivedirectionRecosignalD5->SetXTitle("D for HE+ jeta = 19; depth = 5 \b");
          if (kcountHEpositivedirectionRecosignalD5 == 4)
            HEpositivedirectionRecosignalD5->SetXTitle("D for HE+ jeta = 20; depth = 5 \b");
          if (kcountHEpositivedirectionRecosignalD5 == 5)
            HEpositivedirectionRecosignalD5->SetXTitle("D for HE+ jeta = 21; depth = 5 \b");
          if (kcountHEpositivedirectionRecosignalD5 == 6)
            HEpositivedirectionRecosignalD5->SetXTitle("D for HE+ jeta = 22; depth = 5 \b");
          if (kcountHEpositivedirectionRecosignalD5 == 7)
            HEpositivedirectionRecosignalD5->SetXTitle("D for HE+ jeta = 23; depth = 5 \b");
          if (kcountHEpositivedirectionRecosignalD5 == 8)
            HEpositivedirectionRecosignalD5->SetXTitle("D for HE+ jeta = 24; depth = 5 \b");
          if (kcountHEpositivedirectionRecosignalD5 == 9)
            HEpositivedirectionRecosignalD5->SetXTitle("D for HE+ jeta = 25; depth = 5 \b");
          if (kcountHEpositivedirectionRecosignalD5 == 10)
            HEpositivedirectionRecosignalD5->SetXTitle("D for HE+ jeta = 26; depth = 5 \b");
          if (kcountHEpositivedirectionRecosignalD5 == 11)
            HEpositivedirectionRecosignalD5->SetXTitle("D for HE+ jeta = 27; depth = 5 \b");
          HEpositivedirectionRecosignalD5->Draw("Error");
          kcountHEpositivedirectionRecosignalD5++;
          if (kcountHEpositivedirectionRecosignalD5 > 11)
            break;  // 4x6 = 24
        }           //ccctest>0

      }  // for i
    }    //if(jeta-41 >= 0)
  }      //for jeta
  /////////////////
  c3x5->Update();
  c3x5->Print("DrecosignalPositiveDirectionhistD1PhiSymmetryDepth5HE.png");
  c3x5->Clear();
  // clean-up
  if (h2CeffHEpositivedirectionRecosignalD5)
    delete h2CeffHEpositivedirectionRecosignalD5;
  //========================================================================================== 19
  //======================================================================
  //======================================================================1D plot: D vs phi , different eta,  depth=6
  //cout<<"      1D plot: D vs phi , different eta,  depth=6 *****" <<endl;
  c3x5->Clear();
  c3x5->Divide(3, 5);
  c3x5->cd(1);
  int kcountHEpositivedirectionRecosignalD6 = 1;
  TH1F* h2CeffHEpositivedirectionRecosignalD6 = new TH1F("h2CeffHEpositivedirectionRecosignalD6", "", nphi, 0., 72.);

  for (int jeta = 0; jeta < njeta; jeta++) {
    // positivedirectionRecosignalD:
    if (jeta - 41 >= 0) {
      //	     for (int i=0;i<ndepth;i++) {
      // depth=6
      for (int i = 5; i < 6; i++) {
        TH1F* HEpositivedirectionRecosignalD6 = (TH1F*)h2CeffHEpositivedirectionRecosignalD6->Clone("twod1");

        float ccctest = 0;  // to avoid empty massive elements
        for (int jphi = 0; jphi < nphi; jphi++) {
          double ccc1 = recosignalvariancehe[i][jeta][jphi];
          if (arecosignalhe[i][jeta][jphi] > 0.) {
            HEpositivedirectionRecosignalD6->Fill(jphi, ccc1);
            ccctest = 1.;  //HEpositivedirectionRecosignalD6->SetBinError(i,0.01);
          }
        }  // for jphi
        if (ccctest > 0.) {
          //cout<<"1919       kcountHEpositivedirectionRecosignalD6   =     "<<kcountHEpositivedirectionRecosignalD6  <<"   jeta-41=     "<< jeta-41 <<endl;
          c3x5->cd(kcountHEpositivedirectionRecosignalD6);
          HEpositivedirectionRecosignalD6->SetMarkerStyle(20);
          HEpositivedirectionRecosignalD6->SetMarkerSize(0.4);
          HEpositivedirectionRecosignalD6->GetYaxis()->SetLabelSize(0.04);
          HEpositivedirectionRecosignalD6->SetXTitle("HEpositivedirectionRecosignalD6 \b");
          HEpositivedirectionRecosignalD6->SetMarkerColor(2);
          HEpositivedirectionRecosignalD6->SetLineColor(0);
          gPad->SetGridy();
          gPad->SetGridx();
          //	   gPad->SetLogy();
          if (kcountHEpositivedirectionRecosignalD6 == 1)
            HEpositivedirectionRecosignalD6->SetXTitle("D for HE+ jeta = 18; depth = 6 \b");
          if (kcountHEpositivedirectionRecosignalD6 == 2)
            HEpositivedirectionRecosignalD6->SetXTitle("D for HE+ jeta = 19; depth = 6 \b");
          if (kcountHEpositivedirectionRecosignalD6 == 3)
            HEpositivedirectionRecosignalD6->SetXTitle("D for HE+ jeta = 20; depth = 6 \b");
          if (kcountHEpositivedirectionRecosignalD6 == 4)
            HEpositivedirectionRecosignalD6->SetXTitle("D for HE+ jeta = 21; depth = 6 \b");
          if (kcountHEpositivedirectionRecosignalD6 == 5)
            HEpositivedirectionRecosignalD6->SetXTitle("D for HE+ jeta = 22; depth = 6 \b");
          if (kcountHEpositivedirectionRecosignalD6 == 6)
            HEpositivedirectionRecosignalD6->SetXTitle("D for HE+ jeta = 23; depth = 6 \b");
          if (kcountHEpositivedirectionRecosignalD6 == 7)
            HEpositivedirectionRecosignalD6->SetXTitle("D for HE+ jeta = 24; depth = 6 \b");
          if (kcountHEpositivedirectionRecosignalD6 == 8)
            HEpositivedirectionRecosignalD6->SetXTitle("D for HE+ jeta = 25; depth = 6 \b");
          if (kcountHEpositivedirectionRecosignalD6 == 9)
            HEpositivedirectionRecosignalD6->SetXTitle("D for HE+ jeta = 26; depth = 6 \b");
          if (kcountHEpositivedirectionRecosignalD6 == 10)
            HEpositivedirectionRecosignalD6->SetXTitle("D for HE+ jeta = 27; depth = 6 \b");
          HEpositivedirectionRecosignalD6->Draw("Error");
          kcountHEpositivedirectionRecosignalD6++;
          if (kcountHEpositivedirectionRecosignalD6 > 10)
            break;  // 4x6 = 24
        }           //ccctest>0

      }  // for i
    }    //if(jeta-41 >= 0)
  }      //for jeta
  /////////////////
  c3x5->Update();
  c3x5->Print("DrecosignalPositiveDirectionhistD1PhiSymmetryDepth6HE.png");
  c3x5->Clear();
  // clean-up
  if (h2CeffHEpositivedirectionRecosignalD6)
    delete h2CeffHEpositivedirectionRecosignalD6;
  //========================================================================================== 20
  //======================================================================
  //======================================================================1D plot: D vs phi , different eta,  depth=7
  //cout<<"      1D plot: D vs phi , different eta,  depth=7 *****" <<endl;
  c3x5->Clear();
  c3x5->Divide(3, 5);
  c3x5->cd(1);
  int kcountHEpositivedirectionRecosignalD7 = 1;
  TH1F* h2CeffHEpositivedirectionRecosignalD7 = new TH1F("h2CeffHEpositivedirectionRecosignalD7", "", nphi, 0., 72.);

  for (int jeta = 0; jeta < njeta; jeta++) {
    // positivedirectionRecosignalD:
    if (jeta - 41 >= 0) {
      //	     for (int i=0;i<ndepth;i++) {
      // depth=7
      for (int i = 6; i < 7; i++) {
        TH1F* HEpositivedirectionRecosignalD7 = (TH1F*)h2CeffHEpositivedirectionRecosignalD7->Clone("twod1");

        float ccctest = 0;  // to avoid empty massive elements
        for (int jphi = 0; jphi < nphi; jphi++) {
          double ccc1 = recosignalvariancehe[i][jeta][jphi];
          if (arecosignalhe[i][jeta][jphi] > 0.) {
            HEpositivedirectionRecosignalD7->Fill(jphi, ccc1);
            ccctest = 1.;  //HEpositivedirectionRecosignalD7->SetBinError(i,0.01);
          }
        }  // for jphi
        if (ccctest != 0.) {
          //cout<<"2020       kcountHEpositivedirectionRecosignalD7   =     "<<kcountHEpositivedirectionRecosignalD7  <<"   jeta-41=     "<< jeta-41 <<endl;
          c3x5->cd(kcountHEpositivedirectionRecosignalD7);
          HEpositivedirectionRecosignalD7->SetMarkerStyle(20);
          HEpositivedirectionRecosignalD7->SetMarkerSize(0.4);
          HEpositivedirectionRecosignalD7->GetYaxis()->SetLabelSize(0.04);
          HEpositivedirectionRecosignalD7->SetXTitle("HEpositivedirectionRecosignalD7 \b");
          HEpositivedirectionRecosignalD7->SetMarkerColor(2);
          HEpositivedirectionRecosignalD7->SetLineColor(0);
          gPad->SetGridy();
          gPad->SetGridx();
          //	   gPad->SetLogy();
          if (kcountHEpositivedirectionRecosignalD7 == 1)
            HEpositivedirectionRecosignalD7->SetXTitle("D for HE+ jeta = 25; depth = 7 \b");
          if (kcountHEpositivedirectionRecosignalD7 == 2)
            HEpositivedirectionRecosignalD7->SetXTitle("D for HE+ jeta = 26; depth = 7 \b");
          if (kcountHEpositivedirectionRecosignalD7 == 3)
            HEpositivedirectionRecosignalD7->SetXTitle("D for HE+ jeta = 27; depth = 7 \b");
          HEpositivedirectionRecosignalD7->Draw("Error");
          kcountHEpositivedirectionRecosignalD7++;
          if (kcountHEpositivedirectionRecosignalD7 > 3)
            break;  // 4x6 = 24
        }           //ccctest>0

      }  // for i
    }    //if(jeta-41 >= 0)
  }      //for jeta
  /////////////////
  c3x5->Update();
  c3x5->Print("DrecosignalPositiveDirectionhistD1PhiSymmetryDepth7HE.png");
  c3x5->Clear();
  // clean-up
  if (h2CeffHEpositivedirectionRecosignalD7)
    delete h2CeffHEpositivedirectionRecosignalD7;

  //========================================================================================== 22222214
  //======================================================================
  //======================================================================1D plot: D vs phi , different eta,  depth=1
  //cout<<"      1D plot: D vs phi , different eta,  depth=1 *****" <<endl;
  c3x5->Clear();
  /////////////////
  c3x5->Divide(3, 5);
  c3x5->cd(1);
  int kcountHEnegativedirectionRecosignalD1 = 1;
  TH1F* h2CeffHEnegativedirectionRecosignalD1 = new TH1F("h2CeffHEnegativedirectionRecosignalD1", "", nphi, 0., 72.);

  for (int jeta = 0; jeta < njeta; jeta++) {
    // negativedirectionRecosignalD:
    if (jeta - 41 < 0) {
      //	     for (int i=0;i<ndepth;i++) {
      // depth=1
      for (int i = 0; i < 1; i++) {
        TH1F* HEnegativedirectionRecosignalD1 = (TH1F*)h2CeffHEnegativedirectionRecosignalD1->Clone("twod1");

        float ccctest = 0;  // to avoid empty massive elements
        for (int jphi = 0; jphi < nphi; jphi++) {
          double ccc1 = recosignalvariancehe[i][jeta][jphi];
          if (arecosignalhe[i][jeta][jphi] > 0.) {
            HEnegativedirectionRecosignalD1->Fill(jphi, ccc1);
            ccctest = 1.;  //HEnegativedirectionRecosignalD1->SetBinError(i,0.01);
          }
        }  // for jphi
        if (ccctest > 0.) {
          //cout<<"1414       kcountHEnegativedirectionRecosignalD1   =     "<<kcountHEnegativedirectionRecosignalD1  <<"   jeta-41=     "<< jeta-41 <<endl;
          c3x5->cd(kcountHEnegativedirectionRecosignalD1);
          HEnegativedirectionRecosignalD1->SetMarkerStyle(20);
          HEnegativedirectionRecosignalD1->SetMarkerSize(0.4);
          HEnegativedirectionRecosignalD1->GetYaxis()->SetLabelSize(0.04);
          HEnegativedirectionRecosignalD1->SetXTitle("HEnegativedirectionRecosignalD1 \b");
          HEnegativedirectionRecosignalD1->SetMarkerColor(2);
          HEnegativedirectionRecosignalD1->SetLineColor(0);
          gPad->SetGridy();
          gPad->SetGridx();
          //	   gPad->SetLogy();
          if (kcountHEnegativedirectionRecosignalD1 == 1)
            HEnegativedirectionRecosignalD1->SetXTitle("D for HE- jeta =-29; depth = 1 \b");
          if (kcountHEnegativedirectionRecosignalD1 == 2)
            HEnegativedirectionRecosignalD1->SetXTitle("D for HE- jeta =-28; depth = 1 \b");
          if (kcountHEnegativedirectionRecosignalD1 == 3)
            HEnegativedirectionRecosignalD1->SetXTitle("D for HE- jeta =-27; depth = 1 \b");
          if (kcountHEnegativedirectionRecosignalD1 == 4)
            HEnegativedirectionRecosignalD1->SetXTitle("D for HE- jeta =-26; depth = 1 \b");
          if (kcountHEnegativedirectionRecosignalD1 == 5)
            HEnegativedirectionRecosignalD1->SetXTitle("D for HE- jeta =-25; depth = 1 \b");
          if (kcountHEnegativedirectionRecosignalD1 == 6)
            HEnegativedirectionRecosignalD1->SetXTitle("D for HE- jeta =-24; depth = 1 \b");
          if (kcountHEnegativedirectionRecosignalD1 == 7)
            HEnegativedirectionRecosignalD1->SetXTitle("D for HE- jeta =-23; depth = 1 \b");
          if (kcountHEnegativedirectionRecosignalD1 == 8)
            HEnegativedirectionRecosignalD1->SetXTitle("D for HE- jeta =-22; depth = 1 \b");
          if (kcountHEnegativedirectionRecosignalD1 == 9)
            HEnegativedirectionRecosignalD1->SetXTitle("D for HE- jeta =-21; depth = 1 \b");
          if (kcountHEnegativedirectionRecosignalD1 == 10)
            HEnegativedirectionRecosignalD1->SetXTitle("D for HE- jeta =-20; depth = 1 \b");
          if (kcountHEnegativedirectionRecosignalD1 == 11)
            HEnegativedirectionRecosignalD1->SetXTitle("D for HE- jeta =-19; depth = 1 \b");
          if (kcountHEnegativedirectionRecosignalD1 == 12)
            HEnegativedirectionRecosignalD1->SetXTitle("D for HE- jeta =-18; depth = 1 \b");
          HEnegativedirectionRecosignalD1->Draw("Error");
          kcountHEnegativedirectionRecosignalD1++;
          if (kcountHEnegativedirectionRecosignalD1 > 12)
            break;  // 4x6 = 24
        }           //ccctest>0

      }  // for i
    }    //if(jeta-41 < 0)
  }      //for jeta
  /////////////////
  c3x5->Update();
  c3x5->Print("DrecosignalNegativeDirectionhistD1PhiSymmetryDepth1HE.png");
  c3x5->Clear();
  // clean-up
  if (h2CeffHEnegativedirectionRecosignalD1)
    delete h2CeffHEnegativedirectionRecosignalD1;
  //========================================================================================== 22222215
  //======================================================================
  //======================================================================1D plot: D vs phi , different eta,  depth=2
  //cout<<"      1D plot: D vs phi , different eta,  depth=2 *****" <<endl;
  c3x5->Clear();
  c3x5->Divide(3, 5);
  c3x5->cd(1);
  int kcountHEnegativedirectionRecosignalD2 = 1;
  TH1F* h2CeffHEnegativedirectionRecosignalD2 = new TH1F("h2CeffHEnegativedirectionRecosignalD2", "", nphi, 0., 72.);

  for (int jeta = 0; jeta < njeta; jeta++) {
    // negativedirectionRecosignalD:
    if (jeta - 41 < 0) {
      //	     for (int i=0;i<ndepth;i++) {
      // depth=2
      for (int i = 1; i < 2; i++) {
        TH1F* HEnegativedirectionRecosignalD2 = (TH1F*)h2CeffHEnegativedirectionRecosignalD2->Clone("twod1");

        float ccctest = 0;  // to avoid empty massive elements
        for (int jphi = 0; jphi < nphi; jphi++) {
          double ccc1 = recosignalvariancehe[i][jeta][jphi];
          if (arecosignalhe[i][jeta][jphi] > 0.) {
            HEnegativedirectionRecosignalD2->Fill(jphi, ccc1);
            ccctest = 1.;  //HEnegativedirectionRecosignalD2->SetBinError(i,0.01);
          }
        }  // for jphi
        if (ccctest > 0.) {
          //cout<<"1515       kcountHEnegativedirectionRecosignalD2   =     "<<kcountHEnegativedirectionRecosignalD2  <<"   jeta-41=     "<< jeta-41 <<endl;
          c3x5->cd(kcountHEnegativedirectionRecosignalD2);
          HEnegativedirectionRecosignalD2->SetMarkerStyle(20);
          HEnegativedirectionRecosignalD2->SetMarkerSize(0.4);
          HEnegativedirectionRecosignalD2->GetYaxis()->SetLabelSize(0.04);
          HEnegativedirectionRecosignalD2->SetXTitle("HEnegativedirectionRecosignalD2 \b");
          HEnegativedirectionRecosignalD2->SetMarkerColor(2);
          HEnegativedirectionRecosignalD2->SetLineColor(0);
          gPad->SetGridy();
          gPad->SetGridx();
          //	   gPad->SetLogy();
          if (kcountHEnegativedirectionRecosignalD2 == 1)
            HEnegativedirectionRecosignalD2->SetXTitle("D for HE- jeta =-29; depth = 2 \b");
          if (kcountHEnegativedirectionRecosignalD2 == 2)
            HEnegativedirectionRecosignalD2->SetXTitle("D for HE- jeta =-28; depth = 2 \b");
          if (kcountHEnegativedirectionRecosignalD2 == 3)
            HEnegativedirectionRecosignalD2->SetXTitle("D for HE- jeta =-27; depth = 2 \b");
          if (kcountHEnegativedirectionRecosignalD2 == 4)
            HEnegativedirectionRecosignalD2->SetXTitle("D for HE- jeta =-26; depth = 2 \b");
          if (kcountHEnegativedirectionRecosignalD2 == 5)
            HEnegativedirectionRecosignalD2->SetXTitle("D for HE- jeta =-25; depth = 2 \b");
          if (kcountHEnegativedirectionRecosignalD2 == 6)
            HEnegativedirectionRecosignalD2->SetXTitle("D for HE- jeta =-24; depth = 2 \b");
          if (kcountHEnegativedirectionRecosignalD2 == 7)
            HEnegativedirectionRecosignalD2->SetXTitle("D for HE- jeta =-23; depth = 2 \b");
          if (kcountHEnegativedirectionRecosignalD2 == 8)
            HEnegativedirectionRecosignalD2->SetXTitle("D for HE- jeta =-22; depth = 2 \b");
          if (kcountHEnegativedirectionRecosignalD2 == 9)
            HEnegativedirectionRecosignalD2->SetXTitle("D for HE- jeta =-21; depth = 2 \b");
          if (kcountHEnegativedirectionRecosignalD2 == 10)
            HEnegativedirectionRecosignalD2->SetXTitle("D for HE- jeta =-20; depth = 2 \b");
          if (kcountHEnegativedirectionRecosignalD2 == 11)
            HEnegativedirectionRecosignalD2->SetXTitle("D for HE- jeta =-19; depth = 2 \b");
          if (kcountHEnegativedirectionRecosignalD2 == 12)
            HEnegativedirectionRecosignalD2->SetXTitle("D for HE- jeta =-18; depth = 2 \b");
          if (kcountHEnegativedirectionRecosignalD2 == 13)
            HEnegativedirectionRecosignalD2->SetXTitle("D for HE- jeta =-17; depth = 2 \b");
          HEnegativedirectionRecosignalD2->Draw("Error");
          kcountHEnegativedirectionRecosignalD2++;
          if (kcountHEnegativedirectionRecosignalD2 > 13)
            break;  // 4x6 = 24
        }           //ccctest>0

      }  // for i
    }    //if(jeta-41 < 0)
  }      //for jeta
  /////////////////
  c3x5->Update();
  c3x5->Print("DrecosignalNegativeDirectionhistD1PhiSymmetryDepth2HE.png");
  c3x5->Clear();
  // clean-up
  if (h2CeffHEnegativedirectionRecosignalD2)
    delete h2CeffHEnegativedirectionRecosignalD2;
  //========================================================================================== 22222216
  //======================================================================
  //======================================================================1D plot: D vs phi , different eta,  depth=3
  //cout<<"      1D plot: D vs phi , different eta,  depth=3 *****" <<endl;
  c3x5->Clear();
  c3x5->Divide(3, 5);
  c3x5->cd(1);
  int kcountHEnegativedirectionRecosignalD3 = 1;
  TH1F* h2CeffHEnegativedirectionRecosignalD3 = new TH1F("h2CeffHEnegativedirectionRecosignalD3", "", nphi, 0., 72.);

  for (int jeta = 0; jeta < njeta; jeta++) {
    // negativedirectionRecosignalD:
    if (jeta - 41 < 0) {
      //	     for (int i=0;i<ndepth;i++) {
      // depth=3
      for (int i = 2; i < 3; i++) {
        TH1F* HEnegativedirectionRecosignalD3 = (TH1F*)h2CeffHEnegativedirectionRecosignalD3->Clone("twod1");

        float ccctest = 0;  // to avoid empty massive elements
        for (int jphi = 0; jphi < nphi; jphi++) {
          double ccc1 = recosignalvariancehe[i][jeta][jphi];
          if (arecosignalhe[i][jeta][jphi] > 0.) {
            HEnegativedirectionRecosignalD3->Fill(jphi, ccc1);
            ccctest = 1.;  //HEnegativedirectionRecosignalD3->SetBinError(i,0.01);
          }
        }  // for jphi
        if (ccctest > 0.) {
          //cout<<"1616       kcountHEnegativedirectionRecosignalD3   =     "<<kcountHEnegativedirectionRecosignalD3  <<"   jeta-41=     "<< jeta-41 <<endl;
          c3x5->cd(kcountHEnegativedirectionRecosignalD3);
          HEnegativedirectionRecosignalD3->SetMarkerStyle(20);
          HEnegativedirectionRecosignalD3->SetMarkerSize(0.4);
          HEnegativedirectionRecosignalD3->GetYaxis()->SetLabelSize(0.04);
          HEnegativedirectionRecosignalD3->SetXTitle("HEnegativedirectionRecosignalD3 \b");
          HEnegativedirectionRecosignalD3->SetMarkerColor(2);
          HEnegativedirectionRecosignalD3->SetLineColor(0);
          gPad->SetGridy();
          gPad->SetGridx();
          //	   gPad->SetLogy();
          if (kcountHEnegativedirectionRecosignalD3 == 1)
            HEnegativedirectionRecosignalD3->SetXTitle("D for HE- jeta =-29; depth = 3 \b");
          if (kcountHEnegativedirectionRecosignalD3 == 2)
            HEnegativedirectionRecosignalD3->SetXTitle("D for HE- jeta =-28; depth = 3 \b");
          if (kcountHEnegativedirectionRecosignalD3 == 3)
            HEnegativedirectionRecosignalD3->SetXTitle("D for HE- jeta =-27; depth = 3 \b");
          if (kcountHEnegativedirectionRecosignalD3 == 4)
            HEnegativedirectionRecosignalD3->SetXTitle("D for HE- jeta =-26; depth = 3 \b");
          if (kcountHEnegativedirectionRecosignalD3 == 5)
            HEnegativedirectionRecosignalD3->SetXTitle("D for HE- jeta =-25; depth = 3 \b");
          if (kcountHEnegativedirectionRecosignalD3 == 6)
            HEnegativedirectionRecosignalD3->SetXTitle("D for HE- jeta =-24; depth = 3 \b");
          if (kcountHEnegativedirectionRecosignalD3 == 7)
            HEnegativedirectionRecosignalD3->SetXTitle("D for HE- jeta =-23; depth = 3 \b");
          if (kcountHEnegativedirectionRecosignalD3 == 8)
            HEnegativedirectionRecosignalD3->SetXTitle("D for HE- jeta =-22; depth = 3 \b");
          if (kcountHEnegativedirectionRecosignalD3 == 9)
            HEnegativedirectionRecosignalD3->SetXTitle("D for HE- jeta =-21; depth = 3 \b");
          if (kcountHEnegativedirectionRecosignalD3 == 10)
            HEnegativedirectionRecosignalD3->SetXTitle("D for HE- jeta =-20; depth = 3 \b");
          if (kcountHEnegativedirectionRecosignalD3 == 11)
            HEnegativedirectionRecosignalD3->SetXTitle("D for HE- jeta =-19; depth = 3 \b");
          if (kcountHEnegativedirectionRecosignalD3 == 12)
            HEnegativedirectionRecosignalD3->SetXTitle("D for HE- jeta =-18; depth = 3 \b");
          if (kcountHEnegativedirectionRecosignalD3 == 13)
            HEnegativedirectionRecosignalD3->SetXTitle("D for HE- jeta =-17; depth = 3 \b");
          HEnegativedirectionRecosignalD3->Draw("Error");
          kcountHEnegativedirectionRecosignalD3++;
          if (kcountHEnegativedirectionRecosignalD3 > 13)
            break;  // 4x6 = 24
        }           //ccctest>0

      }  // for i
    }    //if(jeta-41 < 0)
  }      //for jeta
  /////////////////
  c3x5->Update();
  c3x5->Print("DrecosignalNegativeDirectionhistD1PhiSymmetryDepth3HE.png");
  c3x5->Clear();
  // clean-up
  if (h2CeffHEnegativedirectionRecosignalD3)
    delete h2CeffHEnegativedirectionRecosignalD3;
  //========================================================================================== 22222217
  //======================================================================
  //======================================================================1D plot: D vs phi , different eta,  depth=4
  //cout<<"      1D plot: D vs phi , different eta,  depth=4 *****" <<endl;
  c3x5->Clear();
  c3x5->Divide(3, 5);
  c3x5->cd(1);
  int kcountHEnegativedirectionRecosignalD4 = 1;
  TH1F* h2CeffHEnegativedirectionRecosignalD4 = new TH1F("h2CeffHEnegativedirectionRecosignalD4", "", nphi, 0., 72.);

  for (int jeta = 0; jeta < njeta; jeta++) {
    // negativedirectionRecosignalD:
    if (jeta - 41 < 0) {
      //	     for (int i=0;i<ndepth;i++) {
      // depth=4
      for (int i = 3; i < 4; i++) {
        TH1F* HEnegativedirectionRecosignalD4 = (TH1F*)h2CeffHEnegativedirectionRecosignalD4->Clone("twod1");

        float ccctest = 0;  // to avoid empty massive elements
        for (int jphi = 0; jphi < nphi; jphi++) {
          double ccc1 = recosignalvariancehe[i][jeta][jphi];
          if (arecosignalhe[i][jeta][jphi] > 0.) {
            HEnegativedirectionRecosignalD4->Fill(jphi, ccc1);
            ccctest = 1.;  //HEnegativedirectionRecosignalD4->SetBinError(i,0.01);
          }
        }  // for jphi
        if (ccctest > 0.) {
          //cout<<"1717       kcountHEnegativedirectionRecosignalD4   =     "<<kcountHEnegativedirectionRecosignalD4  <<"   jeta-41=     "<< jeta-41 <<endl;
          c3x5->cd(kcountHEnegativedirectionRecosignalD4);
          HEnegativedirectionRecosignalD4->SetMarkerStyle(20);
          HEnegativedirectionRecosignalD4->SetMarkerSize(0.4);
          HEnegativedirectionRecosignalD4->GetYaxis()->SetLabelSize(0.04);
          HEnegativedirectionRecosignalD4->SetXTitle("HEnegativedirectionRecosignalD4 \b");
          HEnegativedirectionRecosignalD4->SetMarkerColor(2);
          HEnegativedirectionRecosignalD4->SetLineColor(0);
          gPad->SetGridy();
          gPad->SetGridx();
          //	   gPad->SetLogy();
          if (kcountHEnegativedirectionRecosignalD4 == 1)
            HEnegativedirectionRecosignalD4->SetXTitle("D for HE- jeta =-28; depth = 4 \b");
          if (kcountHEnegativedirectionRecosignalD4 == 2)
            HEnegativedirectionRecosignalD4->SetXTitle("D for HE- jeta =-27; depth = 4 \b");
          if (kcountHEnegativedirectionRecosignalD4 == 3)
            HEnegativedirectionRecosignalD4->SetXTitle("D for HE- jeta =-26; depth = 4 \b");
          if (kcountHEnegativedirectionRecosignalD4 == 4)
            HEnegativedirectionRecosignalD4->SetXTitle("D for HE- jeta =-25; depth = 4 \b");
          if (kcountHEnegativedirectionRecosignalD4 == 5)
            HEnegativedirectionRecosignalD4->SetXTitle("D for HE- jeta =-24; depth = 4 \b");
          if (kcountHEnegativedirectionRecosignalD4 == 6)
            HEnegativedirectionRecosignalD4->SetXTitle("D for HE- jeta =-23; depth = 4 \b");
          if (kcountHEnegativedirectionRecosignalD4 == 7)
            HEnegativedirectionRecosignalD4->SetXTitle("D for HE- jeta =-22; depth = 4 \b");
          if (kcountHEnegativedirectionRecosignalD4 == 8)
            HEnegativedirectionRecosignalD4->SetXTitle("D for HE- jeta =-21; depth = 4 \b");
          if (kcountHEnegativedirectionRecosignalD4 == 9)
            HEnegativedirectionRecosignalD4->SetXTitle("D for HE- jeta =-20; depth = 4 \b");
          if (kcountHEnegativedirectionRecosignalD4 == 10)
            HEnegativedirectionRecosignalD4->SetXTitle("D for HE- jeta =-19; depth = 4 \b");
          if (kcountHEnegativedirectionRecosignalD4 == 11)
            HEnegativedirectionRecosignalD4->SetXTitle("D for HE- jeta =-18; depth = 4 \b");
          if (kcountHEnegativedirectionRecosignalD4 == 12)
            HEnegativedirectionRecosignalD4->SetXTitle("D for HE- jeta =-16; depth = 4 \b");
          HEnegativedirectionRecosignalD4->Draw("Error");
          kcountHEnegativedirectionRecosignalD4++;
          if (kcountHEnegativedirectionRecosignalD4 > 12)
            break;  // 4x6 = 24
        }           //ccctest>0

      }  // for i
    }    //if(jeta-41 < 0)
  }      //for jeta
  /////////////////
  c3x5->Update();
  c3x5->Print("DrecosignalNegativeDirectionhistD1PhiSymmetryDepth4HE.png");
  c3x5->Clear();
  // clean-up
  if (h2CeffHEnegativedirectionRecosignalD4)
    delete h2CeffHEnegativedirectionRecosignalD4;
  //========================================================================================== 22222218
  //======================================================================
  //======================================================================1D plot: D vs phi , different eta,  depth=5
  //cout<<"      1D plot: D vs phi , different eta,  depth=5 *****" <<endl;
  c3x5->Clear();
  c3x5->Divide(3, 5);
  c3x5->cd(1);
  int kcountHEnegativedirectionRecosignalD5 = 1;
  TH1F* h2CeffHEnegativedirectionRecosignalD5 = new TH1F("h2CeffHEnegativedirectionRecosignalD5", "", nphi, 0., 72.);

  for (int jeta = 0; jeta < njeta; jeta++) {
    // negativedirectionRecosignalD:
    if (jeta - 41 < 0) {
      //	     for (int i=0;i<ndepth;i++) {
      // depth=5
      for (int i = 4; i < 5; i++) {
        TH1F* HEnegativedirectionRecosignalD5 = (TH1F*)h2CeffHEnegativedirectionRecosignalD5->Clone("twod1");

        float ccctest = 0;  // to avoid empty massive elements
        for (int jphi = 0; jphi < nphi; jphi++) {
          double ccc1 = recosignalvariancehe[i][jeta][jphi];
          if (arecosignalhe[i][jeta][jphi] > 0.) {
            HEnegativedirectionRecosignalD5->Fill(jphi, ccc1);
            ccctest = 1.;  //HEnegativedirectionRecosignalD5->SetBinError(i,0.01);
          }
        }  // for jphi
        if (ccctest > 0.) {
          //cout<<"1818       kcountHEnegativedirectionRecosignalD5   =     "<<kcountHEnegativedirectionRecosignalD5  <<"   jeta-41=     "<< jeta-41 <<endl;
          c3x5->cd(kcountHEnegativedirectionRecosignalD5);
          HEnegativedirectionRecosignalD5->SetMarkerStyle(20);
          HEnegativedirectionRecosignalD5->SetMarkerSize(0.4);
          HEnegativedirectionRecosignalD5->GetYaxis()->SetLabelSize(0.04);
          HEnegativedirectionRecosignalD5->SetXTitle("HEnegativedirectionRecosignalD5 \b");
          HEnegativedirectionRecosignalD5->SetMarkerColor(2);
          HEnegativedirectionRecosignalD5->SetLineColor(0);
          gPad->SetGridy();
          gPad->SetGridx();
          //	   gPad->SetLogy();
          if (kcountHEnegativedirectionRecosignalD5 == 1)
            HEnegativedirectionRecosignalD5->SetXTitle("D for HE- jeta =-28; depth = 5 \b");
          if (kcountHEnegativedirectionRecosignalD5 == 2)
            HEnegativedirectionRecosignalD5->SetXTitle("D for HE- jeta =-27; depth = 5 \b");
          if (kcountHEnegativedirectionRecosignalD5 == 3)
            HEnegativedirectionRecosignalD5->SetXTitle("D for HE- jeta =-26; depth = 5 \b");
          if (kcountHEnegativedirectionRecosignalD5 == 4)
            HEnegativedirectionRecosignalD5->SetXTitle("D for HE- jeta =-25; depth = 5 \b");
          if (kcountHEnegativedirectionRecosignalD5 == 5)
            HEnegativedirectionRecosignalD5->SetXTitle("D for HE- jeta =-24; depth = 5 \b");
          if (kcountHEnegativedirectionRecosignalD5 == 6)
            HEnegativedirectionRecosignalD5->SetXTitle("D for HE- jeta =-23; depth = 5 \b");
          if (kcountHEnegativedirectionRecosignalD5 == 7)
            HEnegativedirectionRecosignalD5->SetXTitle("D for HE- jeta =-22; depth = 5 \b");
          if (kcountHEnegativedirectionRecosignalD5 == 8)
            HEnegativedirectionRecosignalD5->SetXTitle("D for HE- jeta =-21; depth = 5 \b");
          if (kcountHEnegativedirectionRecosignalD5 == 9)
            HEnegativedirectionRecosignalD5->SetXTitle("D for HE- jeta =-20; depth = 5 \b");
          if (kcountHEnegativedirectionRecosignalD5 == 10)
            HEnegativedirectionRecosignalD5->SetXTitle("D for HE- jeta =-19; depth = 5 \b");
          if (kcountHEnegativedirectionRecosignalD5 == 11)
            HEnegativedirectionRecosignalD5->SetXTitle("D for HE- jeta =-18; depth = 5 \b");
          HEnegativedirectionRecosignalD5->Draw("Error");
          kcountHEnegativedirectionRecosignalD5++;
          if (kcountHEnegativedirectionRecosignalD5 > 11)
            break;  // 4x6 = 24
        }           //ccctest>0

      }  // for i
    }    //if(jeta-41 < 0)
  }      //for jeta
  /////////////////
  c3x5->Update();
  c3x5->Print("DrecosignalNegativeDirectionhistD1PhiSymmetryDepth5HE.png");
  c3x5->Clear();
  // clean-up
  if (h2CeffHEnegativedirectionRecosignalD5)
    delete h2CeffHEnegativedirectionRecosignalD5;
  //========================================================================================== 22222219
  //======================================================================
  //======================================================================1D plot: D vs phi , different eta,  depth=6
  //cout<<"      1D plot: D vs phi , different eta,  depth=6 *****" <<endl;
  c3x5->Clear();
  c3x5->Divide(3, 5);
  c3x5->cd(1);
  int kcountHEnegativedirectionRecosignalD6 = 1;
  TH1F* h2CeffHEnegativedirectionRecosignalD6 = new TH1F("h2CeffHEnegativedirectionRecosignalD6", "", nphi, 0., 72.);

  for (int jeta = 0; jeta < njeta; jeta++) {
    // negativedirectionRecosignalD:
    if (jeta - 41 < 0) {
      //	     for (int i=0;i<ndepth;i++) {
      // depth=6
      for (int i = 5; i < 6; i++) {
        TH1F* HEnegativedirectionRecosignalD6 = (TH1F*)h2CeffHEnegativedirectionRecosignalD6->Clone("twod1");

        float ccctest = 0;  // to avoid empty massive elements
        for (int jphi = 0; jphi < nphi; jphi++) {
          double ccc1 = recosignalvariancehe[i][jeta][jphi];
          if (arecosignalhe[i][jeta][jphi] > 0.) {
            HEnegativedirectionRecosignalD6->Fill(jphi, ccc1);
            ccctest = 1.;  //HEnegativedirectionRecosignalD6->SetBinError(i,0.01);
          }
        }  // for jphi
        if (ccctest > 0.) {
          //cout<<"1919       kcountHEnegativedirectionRecosignalD6   =     "<<kcountHEnegativedirectionRecosignalD6  <<"   jeta-41=     "<< jeta-41 <<endl;
          c3x5->cd(kcountHEnegativedirectionRecosignalD6);
          HEnegativedirectionRecosignalD6->SetMarkerStyle(20);
          HEnegativedirectionRecosignalD6->SetMarkerSize(0.4);
          HEnegativedirectionRecosignalD6->GetYaxis()->SetLabelSize(0.04);
          HEnegativedirectionRecosignalD6->SetXTitle("HEnegativedirectionRecosignalD6 \b");
          HEnegativedirectionRecosignalD6->SetMarkerColor(2);
          HEnegativedirectionRecosignalD6->SetLineColor(0);
          gPad->SetGridy();
          gPad->SetGridx();
          //	   gPad->SetLogy();
          if (kcountHEnegativedirectionRecosignalD6 == 1)
            HEnegativedirectionRecosignalD6->SetXTitle("D for HE- jeta =-28; depth = 6 \b");
          if (kcountHEnegativedirectionRecosignalD6 == 2)
            HEnegativedirectionRecosignalD6->SetXTitle("D for HE- jeta =-27; depth = 6 \b");
          if (kcountHEnegativedirectionRecosignalD6 == 3)
            HEnegativedirectionRecosignalD6->SetXTitle("D for HE- jeta =-26; depth = 6 \b");
          if (kcountHEnegativedirectionRecosignalD6 == 4)
            HEnegativedirectionRecosignalD6->SetXTitle("D for HE- jeta =-25; depth = 6 \b");
          if (kcountHEnegativedirectionRecosignalD6 == 5)
            HEnegativedirectionRecosignalD6->SetXTitle("D for HE- jeta =-24; depth = 6 \b");
          if (kcountHEnegativedirectionRecosignalD6 == 6)
            HEnegativedirectionRecosignalD6->SetXTitle("D for HE- jeta =-23; depth = 6 \b");
          if (kcountHEnegativedirectionRecosignalD6 == 7)
            HEnegativedirectionRecosignalD6->SetXTitle("D for HE- jeta =-22; depth = 6 \b");
          if (kcountHEnegativedirectionRecosignalD6 == 8)
            HEnegativedirectionRecosignalD6->SetXTitle("D for HE- jeta =-21; depth = 6 \b");
          if (kcountHEnegativedirectionRecosignalD6 == 9)
            HEnegativedirectionRecosignalD6->SetXTitle("D for HE- jeta =-20; depth = 6 \b");
          if (kcountHEnegativedirectionRecosignalD6 == 10)
            HEnegativedirectionRecosignalD6->SetXTitle("D for HE- jeta =-19; depth = 6 \b");
          HEnegativedirectionRecosignalD6->Draw("Error");
          kcountHEnegativedirectionRecosignalD6++;
          if (kcountHEnegativedirectionRecosignalD6 > 10)
            break;  // 4x6 = 24
        }           //ccctest>0

      }  // for i
    }    //if(jeta-41 < 0)
  }      //for jeta
  /////////////////
  c3x5->Update();
  c3x5->Print("DrecosignalNegativeDirectionhistD1PhiSymmetryDepth6HE.png");
  c3x5->Clear();
  // clean-up
  if (h2CeffHEnegativedirectionRecosignalD6)
    delete h2CeffHEnegativedirectionRecosignalD6;
  //========================================================================================== 22222220
  //======================================================================
  //======================================================================1D plot: D vs phi , different eta,  depth=7
  //cout<<"      1D plot: D vs phi , different eta,  depth=7 *****" <<endl;
  c3x5->Clear();
  c3x5->Divide(3, 5);
  c3x5->cd(1);
  int kcountHEnegativedirectionRecosignalD7 = 1;
  TH1F* h2CeffHEnegativedirectionRecosignalD7 = new TH1F("h2CeffHEnegativedirectionRecosignalD7", "", nphi, 0., 72.);

  for (int jeta = 0; jeta < njeta; jeta++) {
    // negativedirectionRecosignalD:
    if (jeta - 41 < 0) {
      //	     for (int i=0;i<ndepth;i++) {
      // depth=7
      for (int i = 6; i < 7; i++) {
        TH1F* HEnegativedirectionRecosignalD7 = (TH1F*)h2CeffHEnegativedirectionRecosignalD7->Clone("twod1");

        float ccctest = 0;  // to avoid empty massive elements
        for (int jphi = 0; jphi < nphi; jphi++) {
          double ccc1 = recosignalvariancehe[i][jeta][jphi];
          if (arecosignalhe[i][jeta][jphi] > 0.) {
            HEnegativedirectionRecosignalD7->Fill(jphi, ccc1);
            ccctest = 1.;  //HEnegativedirectionRecosignalD7->SetBinError(i,0.01);
          }
        }  // for jphi
        if (ccctest != 0.) {
          //cout<<"2020       kcountHEnegativedirectionRecosignalD7   =     "<<kcountHEnegativedirectionRecosignalD7  <<"   jeta-41=     "<< jeta-41 <<endl;
          c3x5->cd(kcountHEnegativedirectionRecosignalD7);
          HEnegativedirectionRecosignalD7->SetMarkerStyle(20);
          HEnegativedirectionRecosignalD7->SetMarkerSize(0.4);
          HEnegativedirectionRecosignalD7->GetYaxis()->SetLabelSize(0.04);
          HEnegativedirectionRecosignalD7->SetXTitle("HEnegativedirectionRecosignalD7 \b");
          HEnegativedirectionRecosignalD7->SetMarkerColor(2);
          HEnegativedirectionRecosignalD7->SetLineColor(0);
          gPad->SetGridy();
          gPad->SetGridx();
          //	   gPad->SetLogy();
          if (kcountHEnegativedirectionRecosignalD7 == 1)
            HEnegativedirectionRecosignalD7->SetXTitle("D for HE- jeta =-28; depth = 7 \b");
          if (kcountHEnegativedirectionRecosignalD7 == 2)
            HEnegativedirectionRecosignalD7->SetXTitle("D for HE- jeta =-27; depth = 7 \b");
          if (kcountHEnegativedirectionRecosignalD7 == 3)
            HEnegativedirectionRecosignalD7->SetXTitle("D for HE- jeta =-26; depth = 7 \b");
          HEnegativedirectionRecosignalD7->Draw("Error");
          kcountHEnegativedirectionRecosignalD7++;
          if (kcountHEnegativedirectionRecosignalD7 > 3)
            break;  // 4x6 = 24
        }           //ccctest>0

      }  // for i
    }    //if(jeta-41 < 0)
  }      //for jeta
  /////////////////
  c3x5->Update();
  c3x5->Print("DrecosignalNegativeDirectionhistD1PhiSymmetryDepth7HE.png");
  c3x5->Clear();
  // clean-up
  if (h2CeffHEnegativedirectionRecosignalD7)
    delete h2CeffHEnegativedirectionRecosignalD7;
  //=====================================================================       END of Recosignal HE for phi-symmetry
  //=====================================================================       END of Recosignal HE for phi-symmetry
  //=====================================================================       END of Recosignal HE for phi-symmetry
  ////////////////////////////////////////////////////////////////////////////////////////////////////         Phi-symmetry for Calibration Group:    Recosignal HF
  ////////////////////////////////////////////////////////////////////////////////////////////////////         Phi-symmetry for Calibration Group:    Recosignal HF
  ////////////////////////////////////////////////////////////////////////////////////////////////////         Phi-symmetry for Calibration Group:    Recosignal HF
  //  int k_max[5]={0,4,7,4,4}; // maximum depth for each subdet
  //ndepth = k_max[5];
  ndepth = 2;
  double arecosignalHF[ndepth][njeta][njphi];
  double recosignalvarianceHF[ndepth][njeta][njphi];
  //cout<<"111RRRRRRRRRRRRRRRRRRRRRRRRR      Recosignal HF" <<endl;
  //                                   RRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRR:   Recosignal HF
  TH2F* recSignalEnergy1HF1 = (TH2F*)dir->FindObjectAny("h_recSignalEnergy1_HF1");
  TH2F* recSignalEnergy0HF1 = (TH2F*)dir->FindObjectAny("h_recSignalEnergy0_HF1");
  TH2F* recSignalEnergyHF1 = (TH2F*)recSignalEnergy1HF1->Clone("recSignalEnergyHF1");
  recSignalEnergyHF1->Divide(recSignalEnergy1HF1, recSignalEnergy0HF1, 1, 1, "B");
  TH2F* recSignalEnergy1HF2 = (TH2F*)dir->FindObjectAny("h_recSignalEnergy1_HF2");
  TH2F* recSignalEnergy0HF2 = (TH2F*)dir->FindObjectAny("h_recSignalEnergy0_HF2");
  TH2F* recSignalEnergyHF2 = (TH2F*)recSignalEnergy1HF2->Clone("recSignalEnergyHF2");
  recSignalEnergyHF2->Divide(recSignalEnergy1HF2, recSignalEnergy0HF2, 1, 1, "B");
  //  cout<<"222RRRRRRRRRRRRRRRRRRRRRRRRR      Recosignal HF" <<endl;
  //====================================================================== PHI normalization & put R into massive arecosignalHF
  for (int jeta = 0; jeta < njeta; jeta++) {
    //preparation for PHI normalization:
    double sumrecosignalHF0 = 0;
    int nsumrecosignalHF0 = 0;
    double sumrecosignalHF1 = 0;
    int nsumrecosignalHF1 = 0;
    for (int jphi = 0; jphi < njphi; jphi++) {
      arecosignalHF[0][jeta][jphi] = recSignalEnergyHF1->GetBinContent(jeta + 1, jphi + 1);
      arecosignalHF[1][jeta][jphi] = recSignalEnergyHF2->GetBinContent(jeta + 1, jphi + 1);
      if (arecosignalHF[0][jeta][jphi] > 0.) {
        sumrecosignalHF0 += arecosignalHF[0][jeta][jphi];
        ++nsumrecosignalHF0;
      }
      if (arecosignalHF[1][jeta][jphi] > 0.) {
        sumrecosignalHF1 += arecosignalHF[1][jeta][jphi];
        ++nsumrecosignalHF1;
      }
    }  // phi
    // PHI normalization:
    for (int jphi = 0; jphi < njphi; jphi++) {
      if (arecosignalHF[0][jeta][jphi] > 0.)
        arecosignalHF[0][jeta][jphi] /= (sumrecosignalHF0 / nsumrecosignalHF0);
      if (arecosignalHF[1][jeta][jphi] > 0.)
        arecosignalHF[1][jeta][jphi] /= (sumrecosignalHF1 / nsumrecosignalHF1);
    }  // phi
  }    //eta
  //------------------------  2D-eta/phi-plot: R, averaged over depthfs
  //======================================================================
  //======================================================================
  // cout<<"      R2D-eta/phi-plot: R, averaged over depthfs *****" <<endl;
  c2x1->Clear();
  /////////////////
  c2x1->Divide(2, 1);
  c2x1->cd(1);
  TH2F* GefzRrecosignalHF42D = new TH2F("GefzRrecosignalHF42D", "", neta, -41., 41., nphi, 0., 72.);
  TH2F* GefzRrecosignalHF42D0 = new TH2F("GefzRrecosignalHF42D0", "", neta, -41., 41., nphi, 0., 72.);
  TH2F* GefzRrecosignalHF42DF = (TH2F*)GefzRrecosignalHF42D0->Clone("GefzRrecosignalHF42DF");
  for (int i = 0; i < ndepth; i++) {
    for (int jeta = 0; jeta < neta; jeta++) {
      for (int jphi = 0; jphi < nphi; jphi++) {
        double ccc1 = arecosignalHF[i][jeta][jphi];
        int k2plot = jeta - 41;
        int kkk = k2plot;  //if(k2plot >0 ) kkk=k2plot+1; //-41 +41 !=0
        if (ccc1 != 0.) {
          GefzRrecosignalHF42D->Fill(kkk, jphi, ccc1);
          GefzRrecosignalHF42D0->Fill(kkk, jphi, 1.);
        }
      }
    }
  }
  GefzRrecosignalHF42DF->Divide(GefzRrecosignalHF42D, GefzRrecosignalHF42D0, 1, 1, "B");  // average A
  gPad->SetGridy();
  gPad->SetGridx();  //      gPad->SetLogz();
  GefzRrecosignalHF42DF->SetXTitle("<R>_depth       #eta  \b");
  GefzRrecosignalHF42DF->SetYTitle("      #phi \b");
  GefzRrecosignalHF42DF->Draw("COLZ");

  c2x1->cd(2);
  TH1F* energyhitSignal_HF = (TH1F*)dir->FindObjectAny("h_energyhitSignal_HF");
  energyhitSignal_HF->SetMarkerStyle(20);
  energyhitSignal_HF->SetMarkerSize(0.4);
  energyhitSignal_HF->GetYaxis()->SetLabelSize(0.04);
  energyhitSignal_HF->SetXTitle("energyhitSignal_HF \b");
  energyhitSignal_HF->SetMarkerColor(2);
  energyhitSignal_HF->SetLineColor(0);
  gPad->SetGridy();
  gPad->SetGridx();
  energyhitSignal_HF->Draw("Error");

  /////////////////
  c2x1->Update();
  c2x1->Print("RrecosignalGeneralD2PhiSymmetryHF.png");
  c2x1->Clear();
  // clean-up
  if (GefzRrecosignalHF42D)
    delete GefzRrecosignalHF42D;
  if (GefzRrecosignalHF42D0)
    delete GefzRrecosignalHF42D0;
  if (GefzRrecosignalHF42DF)
    delete GefzRrecosignalHF42DF;
  //====================================================================== 1D plot: R vs phi , averaged over depthfs & eta
  //======================================================================
  //cout<<"      1D plot: R vs phi , averaged over depthfs & eta *****" <<endl;
  c1x1->Clear();
  /////////////////
  c1x1->Divide(1, 1);
  c1x1->cd(1);
  TH1F* GefzRrecosignalHF41D = new TH1F("GefzRrecosignalHF41D", "", nphi, 0., 72.);
  TH1F* GefzRrecosignalHF41D0 = new TH1F("GefzRrecosignalHF41D0", "", nphi, 0., 72.);
  TH1F* GefzRrecosignalHF41DF = (TH1F*)GefzRrecosignalHF41D0->Clone("GefzRrecosignalHF41DF");
  for (int jphi = 0; jphi < nphi; jphi++) {
    for (int jeta = 0; jeta < neta; jeta++) {
      for (int i = 0; i < ndepth; i++) {
        double ccc1 = arecosignalHF[i][jeta][jphi];
        if (ccc1 != 0.) {
          GefzRrecosignalHF41D->Fill(jphi, ccc1);
          GefzRrecosignalHF41D0->Fill(jphi, 1.);
        }
      }
    }
  }
  GefzRrecosignalHF41DF->Divide(
      GefzRrecosignalHF41D, GefzRrecosignalHF41D0, 1, 1, "B");  // R averaged over depthfs & eta
  GefzRrecosignalHF41D0->Sumw2();
  //    for (int jphi=1;jphi<73;jphi++) {GefzRrecosignalHF41DF->SetBinError(jphi,0.01);}
  gPad->SetGridy();
  gPad->SetGridx();  //      gPad->SetLogz();
  GefzRrecosignalHF41DF->SetMarkerStyle(20);
  GefzRrecosignalHF41DF->SetMarkerSize(1.4);
  GefzRrecosignalHF41DF->GetZaxis()->SetLabelSize(0.08);
  GefzRrecosignalHF41DF->SetXTitle("#phi  \b");
  GefzRrecosignalHF41DF->SetYTitle("  <R> \b");
  GefzRrecosignalHF41DF->SetZTitle("<R>_PHI  - AllDepthfs \b");
  GefzRrecosignalHF41DF->SetMarkerColor(4);
  GefzRrecosignalHF41DF->SetLineColor(
      4);  //  GefzRrecosignalHF41DF->SetMinimum(0.8);     //      GefzRrecosignalHF41DF->SetMaximum(1.000);
  GefzRrecosignalHF41DF->Draw("Error");
  /////////////////
  c1x1->Update();
  c1x1->Print("RrecosignalGeneralD1PhiSymmetryHF.png");
  c1x1->Clear();
  // clean-up
  if (GefzRrecosignalHF41D)
    delete GefzRrecosignalHF41D;
  if (GefzRrecosignalHF41D0)
    delete GefzRrecosignalHF41D0;
  if (GefzRrecosignalHF41DF)
    delete GefzRrecosignalHF41DF;
  //========================================================================================== 4
  //======================================================================
  //======================================================================1D plot: R vs phi , different eta,  depth=1
  //cout<<"      1D plot: R vs phi , different eta,  depth=1 *****" <<endl;
  c3x5->Clear();
  /////////////////
  c3x5->Divide(3, 5);
  c3x5->cd(1);
  int kcountHFpositivedirectionRecosignal1 = 1;
  TH1F* h2CeffHFpositivedirectionRecosignal1 = new TH1F("h2CeffHFpositivedirectionRecosignal1", "", nphi, 0., 72.);
  for (int jeta = 0; jeta < njeta; jeta++) {
    // positivedirectionRecosignal:
    if (jeta - 41 >= 0) {
      //	     for (int i=0;i<ndepth;i++) {
      // depth=1
      for (int i = 0; i < 1; i++) {
        TH1F* HFpositivedirectionRecosignal1 = (TH1F*)h2CeffHFpositivedirectionRecosignal1->Clone("twod1");
        float ccctest = 0;  // to avoid empty massive elements
        for (int jphi = 0; jphi < nphi; jphi++) {
          double ccc1 = arecosignalHF[i][jeta][jphi];
          if (ccc1 != 0.) {
            HFpositivedirectionRecosignal1->Fill(jphi, ccc1);
            ccctest = 1.;  //HFpositivedirectionRecosignal1->SetBinError(i,0.01);
          }
        }  // for jphi
        if (ccctest > 0.) {
          //	  cout<<"444        kcountHFpositivedirectionRecosignal1   =     "<<kcountHFpositivedirectionRecosignal1  <<"   jeta-41=     "<< jeta-41 <<endl;
          c3x5->cd(kcountHFpositivedirectionRecosignal1);
          HFpositivedirectionRecosignal1->SetMarkerStyle(20);
          HFpositivedirectionRecosignal1->SetMarkerSize(0.4);
          HFpositivedirectionRecosignal1->GetYaxis()->SetLabelSize(0.04);
          HFpositivedirectionRecosignal1->SetXTitle("HFpositivedirectionRecosignal1 \b");
          HFpositivedirectionRecosignal1->SetMarkerColor(2);
          HFpositivedirectionRecosignal1->SetLineColor(0);
          gPad->SetGridy();
          gPad->SetGridx();
          //	   gPad->SetLogy();
          if (kcountHFpositivedirectionRecosignal1 == 1)
            HFpositivedirectionRecosignal1->SetXTitle("R for HF+ jeta = 28; depth = 1 \b");
          if (kcountHFpositivedirectionRecosignal1 == 2)
            HFpositivedirectionRecosignal1->SetXTitle("R for HF+ jeta = 29; depth = 1 \b");
          if (kcountHFpositivedirectionRecosignal1 == 3)
            HFpositivedirectionRecosignal1->SetXTitle("R for HF+ jeta = 30; depth = 1 \b");
          if (kcountHFpositivedirectionRecosignal1 == 4)
            HFpositivedirectionRecosignal1->SetXTitle("R for HF+ jeta = 31; depth = 1 \b");
          if (kcountHFpositivedirectionRecosignal1 == 5)
            HFpositivedirectionRecosignal1->SetXTitle("R for HF+ jeta = 32; depth = 1 \b");
          if (kcountHFpositivedirectionRecosignal1 == 6)
            HFpositivedirectionRecosignal1->SetXTitle("R for HF+ jeta = 33; depth = 1 \b");
          if (kcountHFpositivedirectionRecosignal1 == 7)
            HFpositivedirectionRecosignal1->SetXTitle("R for HF+ jeta = 34; depth = 1 \b");
          if (kcountHFpositivedirectionRecosignal1 == 8)
            HFpositivedirectionRecosignal1->SetXTitle("R for HF+ jeta = 35; depth = 1 \b");
          if (kcountHFpositivedirectionRecosignal1 == 9)
            HFpositivedirectionRecosignal1->SetXTitle("R for HF+ jeta = 36; depth = 1 \b");
          if (kcountHFpositivedirectionRecosignal1 == 10)
            HFpositivedirectionRecosignal1->SetXTitle("R for HF+ jeta = 37; depth = 1 \b");
          if (kcountHFpositivedirectionRecosignal1 == 11)
            HFpositivedirectionRecosignal1->SetXTitle("R for HF+ jeta = 38; depth = 1 \b");
          if (kcountHFpositivedirectionRecosignal1 == 12)
            HFpositivedirectionRecosignal1->SetXTitle("R for HF+ jeta = 39; depth = 1 \b");
          if (kcountHFpositivedirectionRecosignal1 == 13)
            HFpositivedirectionRecosignal1->SetXTitle("R for HF+ jeta = 40; depth = 1 \b");
          HFpositivedirectionRecosignal1->Draw("Error");
          kcountHFpositivedirectionRecosignal1++;
          if (kcountHFpositivedirectionRecosignal1 > 13)
            break;  //
        }           //ccctest>0

      }  // for i
    }    //if(jeta-41 >= 0)
  }      //for jeta
  /////////////////
  c3x5->Update();
  c3x5->Print("RrecosignalPositiveDirectionhistD1PhiSymmetryDepth1HF.png");
  c3x5->Clear();
  // clean-up
  if (h2CeffHFpositivedirectionRecosignal1)
    delete h2CeffHFpositivedirectionRecosignal1;

  //========================================================================================== 5
  //======================================================================
  //======================================================================1D plot: R vs phi , different eta,  depth=2
  //  cout<<"      1D plot: R vs phi , different eta,  depth=2 *****" <<endl;
  c3x5->Clear();
  /////////////////
  c3x5->Divide(3, 5);
  c3x5->cd(1);
  int kcountHFpositivedirectionRecosignal2 = 1;
  TH1F* h2CeffHFpositivedirectionRecosignal2 = new TH1F("h2CeffHFpositivedirectionRecosignal2", "", nphi, 0., 72.);
  for (int jeta = 0; jeta < njeta; jeta++) {
    // positivedirectionRecosignal:
    if (jeta - 41 >= 0) {
      //	     for (int i=0;i<ndepth;i++) {
      // depth=2
      for (int i = 1; i < 2; i++) {
        TH1F* HFpositivedirectionRecosignal2 = (TH1F*)h2CeffHFpositivedirectionRecosignal2->Clone("twod1");
        float ccctest = 0;  // to avoid empty massive elements
        for (int jphi = 0; jphi < nphi; jphi++) {
          double ccc1 = arecosignalHF[i][jeta][jphi];
          if (ccc1 != 0.) {
            HFpositivedirectionRecosignal2->Fill(jphi, ccc1);
            ccctest = 1.;  //HFpositivedirectionRecosignal2->SetBinError(i,0.01);
          }
        }  // for jphi
        if (ccctest > 0.) {
          //cout<<"555        kcountHFpositivedirectionRecosignal2   =     "<<kcountHFpositivedirectionRecosignal2  <<"   jeta-41=     "<< jeta-41 <<endl;
          c3x5->cd(kcountHFpositivedirectionRecosignal2);
          HFpositivedirectionRecosignal2->SetMarkerStyle(20);
          HFpositivedirectionRecosignal2->SetMarkerSize(0.4);
          HFpositivedirectionRecosignal2->GetYaxis()->SetLabelSize(0.04);
          HFpositivedirectionRecosignal2->SetXTitle("HFpositivedirectionRecosignal2 \b");
          HFpositivedirectionRecosignal2->SetMarkerColor(2);
          HFpositivedirectionRecosignal2->SetLineColor(0);
          gPad->SetGridy();
          gPad->SetGridx();
          //	   gPad->SetLogy();
          if (kcountHFpositivedirectionRecosignal2 == 1)
            HFpositivedirectionRecosignal2->SetXTitle("R for HF+ jeta = 28; depth = 2 \b");
          if (kcountHFpositivedirectionRecosignal2 == 2)
            HFpositivedirectionRecosignal2->SetXTitle("R for HF+ jeta = 29; depth = 2 \b");
          if (kcountHFpositivedirectionRecosignal2 == 3)
            HFpositivedirectionRecosignal2->SetXTitle("R for HF+ jeta = 30; depth = 2 \b");
          if (kcountHFpositivedirectionRecosignal2 == 4)
            HFpositivedirectionRecosignal2->SetXTitle("R for HF+ jeta = 31; depth = 2 \b");
          if (kcountHFpositivedirectionRecosignal2 == 5)
            HFpositivedirectionRecosignal2->SetXTitle("R for HF+ jeta = 32; depth = 2 \b");
          if (kcountHFpositivedirectionRecosignal2 == 6)
            HFpositivedirectionRecosignal2->SetXTitle("R for HF+ jeta = 33; depth = 2 \b");
          if (kcountHFpositivedirectionRecosignal2 == 7)
            HFpositivedirectionRecosignal2->SetXTitle("R for HF+ jeta = 34; depth = 2 \b");
          if (kcountHFpositivedirectionRecosignal2 == 8)
            HFpositivedirectionRecosignal2->SetXTitle("R for HF+ jeta = 35; depth = 2 \b");
          if (kcountHFpositivedirectionRecosignal2 == 9)
            HFpositivedirectionRecosignal2->SetXTitle("R for HF+ jeta = 36; depth = 2 \b");
          if (kcountHFpositivedirectionRecosignal2 == 10)
            HFpositivedirectionRecosignal2->SetXTitle("R for HF+ jeta = 37; depth = 2 \b");
          if (kcountHFpositivedirectionRecosignal2 == 11)
            HFpositivedirectionRecosignal2->SetXTitle("R for HF+ jeta = 38; depth = 2 \b");
          if (kcountHFpositivedirectionRecosignal2 == 12)
            HFpositivedirectionRecosignal2->SetXTitle("R for HF+ jeta = 39; depth = 2 \b");
          if (kcountHFpositivedirectionRecosignal2 == 13)
            HFpositivedirectionRecosignal2->SetXTitle("R for HF+ jeta = 40; depth = 2 \b");
          HFpositivedirectionRecosignal2->Draw("Error");
          kcountHFpositivedirectionRecosignal2++;
          if (kcountHFpositivedirectionRecosignal2 > 13)
            break;  // 4x6 = 24
        }           //ccctest>0

      }  // for i
    }    //if(jeta-41 >= 0)
  }      //for jeta
  /////////////////
  c3x5->Update();
  c3x5->Print("RrecosignalPositiveDirectionhistD1PhiSymmetryDepth2HF.png");
  c3x5->Clear();
  // clean-up
  if (h2CeffHFpositivedirectionRecosignal2)
    delete h2CeffHFpositivedirectionRecosignal2;

  //========================================================================================== 1111114
  //======================================================================
  //======================================================================1D plot: R vs phi , different eta,  depth=1
  //cout<<"      1D plot: R vs phi , different eta,  depth=1 *****" <<endl;
  c3x5->Clear();
  /////////////////
  c3x5->Divide(3, 5);
  c3x5->cd(1);
  int kcountHFnegativedirectionRecosignal1 = 1;
  TH1F* h2CeffHFnegativedirectionRecosignal1 = new TH1F("h2CeffHFnegativedirectionRecosignal1", "", nphi, 0., 72.);
  for (int jeta = 0; jeta < njeta; jeta++) {
    // negativedirectionRecosignal:
    if (jeta - 41 < 0) {
      //	     for (int i=0;i<ndepth;i++) {
      // depth=1
      for (int i = 0; i < 1; i++) {
        TH1F* HFnegativedirectionRecosignal1 = (TH1F*)h2CeffHFnegativedirectionRecosignal1->Clone("twod1");
        float ccctest = 0;  // to avoid empty massive elements
        for (int jphi = 0; jphi < nphi; jphi++) {
          double ccc1 = arecosignalHF[i][jeta][jphi];
          if (ccc1 != 0.) {
            HFnegativedirectionRecosignal1->Fill(jphi, ccc1);
            ccctest = 1.;  //HFnegativedirectionRecosignal1->SetBinError(i,0.01);
          }
        }  // for jphi
        if (ccctest > 0.) {
          //	  cout<<"444        kcountHFnegativedirectionRecosignal1   =     "<<kcountHFnegativedirectionRecosignal1  <<"   jeta-41=     "<< jeta-41 <<endl;
          c3x5->cd(kcountHFnegativedirectionRecosignal1);
          HFnegativedirectionRecosignal1->SetMarkerStyle(20);
          HFnegativedirectionRecosignal1->SetMarkerSize(0.4);
          HFnegativedirectionRecosignal1->GetYaxis()->SetLabelSize(0.04);
          HFnegativedirectionRecosignal1->SetXTitle("HFnegativedirectionRecosignal1 \b");
          HFnegativedirectionRecosignal1->SetMarkerColor(2);
          HFnegativedirectionRecosignal1->SetLineColor(0);
          gPad->SetGridy();
          gPad->SetGridx();
          //	   gPad->SetLogy();
          if (kcountHFnegativedirectionRecosignal1 == 1)
            HFnegativedirectionRecosignal1->SetXTitle("R for HF- jeta =-41; depth = 1 \b");
          if (kcountHFnegativedirectionRecosignal1 == 2)
            HFnegativedirectionRecosignal1->SetXTitle("R for HF- jeta =-40; depth = 1 \b");
          if (kcountHFnegativedirectionRecosignal1 == 3)
            HFnegativedirectionRecosignal1->SetXTitle("R for HF- jeta =-39; depth = 1 \b");
          if (kcountHFnegativedirectionRecosignal1 == 4)
            HFnegativedirectionRecosignal1->SetXTitle("R for HF- jeta =-38; depth = 1 \b");
          if (kcountHFnegativedirectionRecosignal1 == 5)
            HFnegativedirectionRecosignal1->SetXTitle("R for HF- jeta =-37; depth = 1 \b");
          if (kcountHFnegativedirectionRecosignal1 == 6)
            HFnegativedirectionRecosignal1->SetXTitle("R for HF- jeta =-36; depth = 1 \b");
          if (kcountHFnegativedirectionRecosignal1 == 7)
            HFnegativedirectionRecosignal1->SetXTitle("R for HF- jeta =-35; depth = 1 \b");
          if (kcountHFnegativedirectionRecosignal1 == 8)
            HFnegativedirectionRecosignal1->SetXTitle("R for HF- jeta =-34; depth = 1 \b");
          if (kcountHFnegativedirectionRecosignal1 == 9)
            HFnegativedirectionRecosignal1->SetXTitle("R for HF- jeta =-33; depth = 1 \b");
          if (kcountHFnegativedirectionRecosignal1 == 10)
            HFnegativedirectionRecosignal1->SetXTitle("R for HF- jeta =-32; depth = 1 \b");
          if (kcountHFnegativedirectionRecosignal1 == 11)
            HFnegativedirectionRecosignal1->SetXTitle("R for HF- jeta =-31; depth = 1 \b");
          if (kcountHFnegativedirectionRecosignal1 == 12)
            HFnegativedirectionRecosignal1->SetXTitle("R for HF- jeta =-30; depth = 1 \b");
          if (kcountHFnegativedirectionRecosignal1 == 13)
            HFnegativedirectionRecosignal1->SetXTitle("R for HF- jeta =-29; depth = 1 \b");
          HFnegativedirectionRecosignal1->Draw("Error");
          kcountHFnegativedirectionRecosignal1++;
          if (kcountHFnegativedirectionRecosignal1 > 13)
            break;  //
        }           //ccctest>0

      }  // for i
    }    //if(jeta-41< 0)
  }      //for jeta
  /////////////////
  c3x5->Update();
  c3x5->Print("RrecosignalNegativeDirectionhistD1PhiSymmetryDepth1HF.png");
  c3x5->Clear();
  // clean-up
  if (h2CeffHFnegativedirectionRecosignal1)
    delete h2CeffHFnegativedirectionRecosignal1;

  //========================================================================================== 1111115
  //======================================================================
  //======================================================================1D plot: R vs phi , different eta,  depth=2
  //  cout<<"      1D plot: R vs phi , different eta,  depth=2 *****" <<endl;
  c3x5->Clear();
  /////////////////
  c3x5->Divide(3, 5);
  c3x5->cd(1);
  int kcountHFnegativedirectionRecosignal2 = 1;
  TH1F* h2CeffHFnegativedirectionRecosignal2 = new TH1F("h2CeffHFnegativedirectionRecosignal2", "", nphi, 0., 72.);
  for (int jeta = 0; jeta < njeta; jeta++) {
    // negativedirectionRecosignal:
    if (jeta - 41 < 0) {
      //	     for (int i=0;i<ndepth;i++) {
      // depth=2
      for (int i = 1; i < 2; i++) {
        TH1F* HFnegativedirectionRecosignal2 = (TH1F*)h2CeffHFnegativedirectionRecosignal2->Clone("twod1");
        float ccctest = 0;  // to avoid empty massive elements
        for (int jphi = 0; jphi < nphi; jphi++) {
          double ccc1 = arecosignalHF[i][jeta][jphi];
          if (ccc1 != 0.) {
            HFnegativedirectionRecosignal2->Fill(jphi, ccc1);
            ccctest = 1.;  //HFnegativedirectionRecosignal2->SetBinError(i,0.01);
          }
        }  // for jphi
        if (ccctest > 0.) {
          //cout<<"555        kcountHFnegativedirectionRecosignal2   =     "<<kcountHFnegativedirectionRecosignal2  <<"   jeta-41=     "<< jeta-41 <<endl;
          c3x5->cd(kcountHFnegativedirectionRecosignal2);
          HFnegativedirectionRecosignal2->SetMarkerStyle(20);
          HFnegativedirectionRecosignal2->SetMarkerSize(0.4);
          HFnegativedirectionRecosignal2->GetYaxis()->SetLabelSize(0.04);
          HFnegativedirectionRecosignal2->SetXTitle("HFnegativedirectionRecosignal2 \b");
          HFnegativedirectionRecosignal2->SetMarkerColor(2);
          HFnegativedirectionRecosignal2->SetLineColor(0);
          gPad->SetGridy();
          gPad->SetGridx();
          //	   gPad->SetLogy();
          if (kcountHFnegativedirectionRecosignal2 == 1)
            HFnegativedirectionRecosignal2->SetXTitle("R for HF- jeta =-41; depth = 2 \b");
          if (kcountHFnegativedirectionRecosignal2 == 2)
            HFnegativedirectionRecosignal2->SetXTitle("R for HF- jeta =-40; depth = 2 \b");
          if (kcountHFnegativedirectionRecosignal2 == 3)
            HFnegativedirectionRecosignal2->SetXTitle("R for HF- jeta =-39; depth = 2 \b");
          if (kcountHFnegativedirectionRecosignal2 == 4)
            HFnegativedirectionRecosignal2->SetXTitle("R for HF- jeta =-38; depth = 2 \b");
          if (kcountHFnegativedirectionRecosignal2 == 5)
            HFnegativedirectionRecosignal2->SetXTitle("R for HF- jeta =-37; depth = 2 \b");
          if (kcountHFnegativedirectionRecosignal2 == 6)
            HFnegativedirectionRecosignal2->SetXTitle("R for HF- jeta =-36; depth = 2 \b");
          if (kcountHFnegativedirectionRecosignal2 == 7)
            HFnegativedirectionRecosignal2->SetXTitle("R for HF- jeta =-35; depth = 2 \b");
          if (kcountHFnegativedirectionRecosignal2 == 8)
            HFnegativedirectionRecosignal2->SetXTitle("R for HF- jeta =-34; depth = 2 \b");
          if (kcountHFnegativedirectionRecosignal2 == 9)
            HFnegativedirectionRecosignal2->SetXTitle("R for HF- jeta =-33; depth = 2 \b");
          if (kcountHFnegativedirectionRecosignal2 == 10)
            HFnegativedirectionRecosignal2->SetXTitle("R for HF- jeta =-32; depth = 2 \b");
          if (kcountHFnegativedirectionRecosignal2 == 11)
            HFnegativedirectionRecosignal2->SetXTitle("R for HF- jeta =-31; depth = 2 \b");
          if (kcountHFnegativedirectionRecosignal2 == 12)
            HFnegativedirectionRecosignal2->SetXTitle("R for HF- jeta =-30; depth = 2 \b");
          if (kcountHFnegativedirectionRecosignal2 == 13)
            HFnegativedirectionRecosignal2->SetXTitle("R for HF- jeta =-20; depth = 2 \b");
          HFnegativedirectionRecosignal2->Draw("Error");
          kcountHFnegativedirectionRecosignal2++;
          if (kcountHFnegativedirectionRecosignal2 > 13)
            break;  // 4x6 = 24
        }           //ccctest>0

      }  // for i
    }    //if(jeta-41< 0)
  }      //for jeta
  /////////////////
  c3x5->Update();
  c3x5->Print("RrecosignalNegativeDirectionhistD1PhiSymmetryDepth2HF.png");
  c3x5->Clear();
  // clean-up
  if (h2CeffHFnegativedirectionRecosignal2)
    delete h2CeffHFnegativedirectionRecosignal2;

  //======================================================================================================================
  //======================================================================================================================
  //======================================================================================================================
  //                            DDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDD:

  //cout<<"    Start Vaiance: preparation  *****" <<endl;
  TH2F* recosignalVariance1HF1 = (TH2F*)dir->FindObjectAny("h_recSignalEnergy2_HF1");
  TH2F* recosignalVariance0HF1 = (TH2F*)dir->FindObjectAny("h_recSignalEnergy0_HF1");
  TH2F* recosignalVarianceHF1 = (TH2F*)recosignalVariance1HF1->Clone("recosignalVarianceHF1");
  recosignalVarianceHF1->Divide(recosignalVariance1HF1, recosignalVariance0HF1, 1, 1, "B");
  TH2F* recosignalVariance1HF2 = (TH2F*)dir->FindObjectAny("h_recSignalEnergy2_HF2");
  TH2F* recosignalVariance0HF2 = (TH2F*)dir->FindObjectAny("h_recSignalEnergy0_HF2");
  TH2F* recosignalVarianceHF2 = (TH2F*)recosignalVariance1HF2->Clone("recosignalVarianceHF2");
  recosignalVarianceHF2->Divide(recosignalVariance1HF2, recosignalVariance0HF2, 1, 1, "B");
  //cout<<"      Vaiance: preparation DONE *****" <<endl;
  //====================================================================== put Vaiance=Dispersia = Sig**2=<R**2> - (<R>)**2 into massive recosignalvarianceHF
  //                                                                                           = sum(R*R)/N - (sum(R)/N)**2
  for (int jeta = 0; jeta < njeta; jeta++) {
    //preparation for PHI normalization:
    double sumrecosignalHF0 = 0;
    int nsumrecosignalHF0 = 0;
    double sumrecosignalHF1 = 0;
    int nsumrecosignalHF1 = 0;
    for (int jphi = 0; jphi < njphi; jphi++) {
      recosignalvarianceHF[0][jeta][jphi] = recosignalVarianceHF1->GetBinContent(jeta + 1, jphi + 1);
      recosignalvarianceHF[1][jeta][jphi] = recosignalVarianceHF2->GetBinContent(jeta + 1, jphi + 1);
      if (recosignalvarianceHF[0][jeta][jphi] > 0.) {
        sumrecosignalHF0 += recosignalvarianceHF[0][jeta][jphi];
        ++nsumrecosignalHF0;
      }
      if (recosignalvarianceHF[1][jeta][jphi] > 0.) {
        sumrecosignalHF1 += recosignalvarianceHF[1][jeta][jphi];
        ++nsumrecosignalHF1;
      }
    }  // phi
    // PHI normalization :
    for (int jphi = 0; jphi < njphi; jphi++) {
      if (recosignalvarianceHF[0][jeta][jphi] > 0.)
        recosignalvarianceHF[0][jeta][jphi] /= (sumrecosignalHF0 / nsumrecosignalHF0);
      if (recosignalvarianceHF[1][jeta][jphi] > 0.)
        recosignalvarianceHF[1][jeta][jphi] /= (sumrecosignalHF1 / nsumrecosignalHF1);
    }  // phi
    //       recosignalvarianceHF (D)           = sum(R*R)/N - (sum(R)/N)**2
    for (int jphi = 0; jphi < njphi; jphi++) {
      //	   cout<<"12 12 12   jeta=     "<< jeta <<"   jphi   =     "<<jphi  <<endl;
      recosignalvarianceHF[0][jeta][jphi] -= arecosignalHF[0][jeta][jphi] * arecosignalHF[0][jeta][jphi];
      recosignalvarianceHF[0][jeta][jphi] = fabs(recosignalvarianceHF[0][jeta][jphi]);
      recosignalvarianceHF[1][jeta][jphi] -= arecosignalHF[1][jeta][jphi] * arecosignalHF[1][jeta][jphi];
      recosignalvarianceHF[1][jeta][jphi] = fabs(recosignalvarianceHF[1][jeta][jphi]);
    }
  }
  //cout<<"      Vaiance: DONE*****" <<endl;
  //------------------------  2D-eta/phi-plot: D, averaged over depthfs
  //======================================================================
  //======================================================================
  //cout<<"      R2D-eta/phi-plot: D, averaged over depthfs *****" <<endl;
  c1x1->Clear();
  /////////////////
  c1x0->Divide(1, 1);
  c1x0->cd(1);
  TH2F* DefzDrecosignalHF42D = new TH2F("DefzDrecosignalHF42D", "", neta, -41., 41., nphi, 0., 72.);
  TH2F* DefzDrecosignalHF42D0 = new TH2F("DefzDrecosignalHF42D0", "", neta, -41., 41., nphi, 0., 72.);
  TH2F* DefzDrecosignalHF42DF = (TH2F*)DefzDrecosignalHF42D0->Clone("DefzDrecosignalHF42DF");
  for (int i = 0; i < ndepth; i++) {
    for (int jeta = 0; jeta < neta; jeta++) {
      for (int jphi = 0; jphi < nphi; jphi++) {
        double ccc1 = recosignalvarianceHF[i][jeta][jphi];
        int k2plot = jeta - 41;
        int kkk = k2plot;  //if(k2plot >0   kkk=k2plot+1; //-41 +41 !=0
        if (arecosignalHF[i][jeta][jphi] > 0.) {
          DefzDrecosignalHF42D->Fill(kkk, jphi, ccc1);
          DefzDrecosignalHF42D0->Fill(kkk, jphi, 1.);
        }
      }
    }
  }
  DefzDrecosignalHF42DF->Divide(DefzDrecosignalHF42D, DefzDrecosignalHF42D0, 1, 1, "B");  // average A
  //    DefzDrecosignalHF1->Sumw2();
  gPad->SetGridy();
  gPad->SetGridx();  //      gPad->SetLogz();
  DefzDrecosignalHF42DF->SetMarkerStyle(20);
  DefzDrecosignalHF42DF->SetMarkerSize(0.4);
  DefzDrecosignalHF42DF->GetZaxis()->SetLabelSize(0.08);
  DefzDrecosignalHF42DF->SetXTitle("<D>_depth       #eta  \b");
  DefzDrecosignalHF42DF->SetYTitle("      #phi \b");
  DefzDrecosignalHF42DF->SetZTitle("<D>_depth \b");
  DefzDrecosignalHF42DF->SetMarkerColor(2);
  DefzDrecosignalHF42DF->SetLineColor(
      0);  //      DefzDrecosignalHF42DF->SetMaximum(1.000);  //      DefzDrecosignalHF42DF->SetMinimum(1.0);
  DefzDrecosignalHF42DF->Draw("COLZ");
  /////////////////
  c1x0->Update();
  c1x0->Print("DrecosignalGeneralD2PhiSymmetryHF.png");
  c1x0->Clear();
  // clean-up
  if (DefzDrecosignalHF42D)
    delete DefzDrecosignalHF42D;
  if (DefzDrecosignalHF42D0)
    delete DefzDrecosignalHF42D0;
  if (DefzDrecosignalHF42DF)
    delete DefzDrecosignalHF42DF;
  //====================================================================== 1D plot: D vs phi , averaged over depthfs & eta
  //======================================================================
  //cout<<"      1D plot: D vs phi , averaged over depthfs & eta *****" <<endl;
  c1x1->Clear();
  /////////////////
  c1x1->Divide(1, 1);
  c1x1->cd(1);
  TH1F* DefzDrecosignalHF41D = new TH1F("DefzDrecosignalHF41D", "", nphi, 0., 72.);
  TH1F* DefzDrecosignalHF41D0 = new TH1F("DefzDrecosignalHF41D0", "", nphi, 0., 72.);
  TH1F* DefzDrecosignalHF41DF = (TH1F*)DefzDrecosignalHF41D0->Clone("DefzDrecosignalHF41DF");

  for (int jphi = 0; jphi < nphi; jphi++) {
    for (int jeta = 0; jeta < neta; jeta++) {
      for (int i = 0; i < ndepth; i++) {
        double ccc1 = recosignalvarianceHF[i][jeta][jphi];
        if (arecosignalHF[i][jeta][jphi] > 0.) {
          DefzDrecosignalHF41D->Fill(jphi, ccc1);
          DefzDrecosignalHF41D0->Fill(jphi, 1.);
        }
      }
    }
  }
  //     DefzDrecosignalHF41D->Sumw2();DefzDrecosignalHF41D0->Sumw2();

  DefzDrecosignalHF41DF->Divide(
      DefzDrecosignalHF41D, DefzDrecosignalHF41D0, 1, 1, "B");  // R averaged over depthfs & eta
  DefzDrecosignalHF41D0->Sumw2();
  //    for (int jphi=1;jphi<73;jphi++) {DefzDrecosignalHF41DF->SetBinError(jphi,0.01);}
  gPad->SetGridy();
  gPad->SetGridx();  //      gPad->SetLogz();
  DefzDrecosignalHF41DF->SetMarkerStyle(20);
  DefzDrecosignalHF41DF->SetMarkerSize(1.4);
  DefzDrecosignalHF41DF->GetZaxis()->SetLabelSize(0.08);
  DefzDrecosignalHF41DF->SetXTitle("#phi  \b");
  DefzDrecosignalHF41DF->SetYTitle("  <D> \b");
  DefzDrecosignalHF41DF->SetZTitle("<D>_PHI  - AllDepthfs \b");
  DefzDrecosignalHF41DF->SetMarkerColor(4);
  DefzDrecosignalHF41DF->SetLineColor(
      4);  //  DefzDrecosignalHF41DF->SetMinimum(0.8);     DefzDrecosignalHF41DF->SetMinimum(-0.015);
  DefzDrecosignalHF41DF->Draw("Error");
  /////////////////
  c1x1->Update();
  c1x1->Print("DrecosignalGeneralD1PhiSymmetryHF.png");
  c1x1->Clear();
  // clean-up
  if (DefzDrecosignalHF41D)
    delete DefzDrecosignalHF41D;
  if (DefzDrecosignalHF41D0)
    delete DefzDrecosignalHF41D0;
  if (DefzDrecosignalHF41DF)
    delete DefzDrecosignalHF41DF;
  //========================================================================================== 14
  //======================================================================
  //======================================================================1D plot: D vs phi , different eta,  depth=1
  //cout<<"      1D plot: D vs phi , different eta,  depth=1 *****" <<endl;
  c3x5->Clear();
  /////////////////
  c3x5->Divide(3, 5);
  c3x5->cd(1);
  int kcountHFpositivedirectionRecosignalD1 = 1;
  TH1F* h2CeffHFpositivedirectionRecosignalD1 = new TH1F("h2CeffHFpositivedirectionRecosignalD1", "", nphi, 0., 72.);

  for (int jeta = 0; jeta < njeta; jeta++) {
    // positivedirectionRecosignalD:
    if (jeta - 41 >= 0) {
      //	     for (int i=0;i<ndepth;i++) {
      // depth=1
      for (int i = 0; i < 1; i++) {
        TH1F* HFpositivedirectionRecosignalD1 = (TH1F*)h2CeffHFpositivedirectionRecosignalD1->Clone("twod1");

        float ccctest = 0;  // to avoid empty massive elements
        for (int jphi = 0; jphi < nphi; jphi++) {
          double ccc1 = recosignalvarianceHF[i][jeta][jphi];
          if (arecosignalHF[i][jeta][jphi] > 0.) {
            HFpositivedirectionRecosignalD1->Fill(jphi, ccc1);
            ccctest = 1.;  //HFpositivedirectionRecosignalD1->SetBinError(i,0.01);
          }
        }  // for jphi
        if (ccctest > 0.) {
          //cout<<"1414       kcountHFpositivedirectionRecosignalD1   =     "<<kcountHFpositivedirectionRecosignalD1  <<"   jeta-41=     "<< jeta-41 <<endl;
          c3x5->cd(kcountHFpositivedirectionRecosignalD1);
          HFpositivedirectionRecosignalD1->SetMarkerStyle(20);
          HFpositivedirectionRecosignalD1->SetMarkerSize(0.4);
          HFpositivedirectionRecosignalD1->GetYaxis()->SetLabelSize(0.04);
          HFpositivedirectionRecosignalD1->SetXTitle("HFpositivedirectionRecosignalD1 \b");
          HFpositivedirectionRecosignalD1->SetMarkerColor(2);
          HFpositivedirectionRecosignalD1->SetLineColor(0);
          gPad->SetGridy();
          gPad->SetGridx();
          //	   gPad->SetLogy();
          if (kcountHFpositivedirectionRecosignalD1 == 1)
            HFpositivedirectionRecosignalD1->SetXTitle("D for HF+ jeta = 28; depth = 1 \b");
          if (kcountHFpositivedirectionRecosignalD1 == 2)
            HFpositivedirectionRecosignalD1->SetXTitle("D for HF+ jeta = 29; depth = 1 \b");
          if (kcountHFpositivedirectionRecosignalD1 == 3)
            HFpositivedirectionRecosignalD1->SetXTitle("D for HF+ jeta = 30; depth = 1 \b");
          if (kcountHFpositivedirectionRecosignalD1 == 4)
            HFpositivedirectionRecosignalD1->SetXTitle("D for HF+ jeta = 31; depth = 1 \b");
          if (kcountHFpositivedirectionRecosignalD1 == 5)
            HFpositivedirectionRecosignalD1->SetXTitle("D for HF+ jeta = 32; depth = 1 \b");
          if (kcountHFpositivedirectionRecosignalD1 == 6)
            HFpositivedirectionRecosignalD1->SetXTitle("D for HF+ jeta = 33; depth = 1 \b");
          if (kcountHFpositivedirectionRecosignalD1 == 7)
            HFpositivedirectionRecosignalD1->SetXTitle("D for HF+ jeta = 34; depth = 1 \b");
          if (kcountHFpositivedirectionRecosignalD1 == 8)
            HFpositivedirectionRecosignalD1->SetXTitle("D for HF+ jeta = 35; depth = 1 \b");
          if (kcountHFpositivedirectionRecosignalD1 == 9)
            HFpositivedirectionRecosignalD1->SetXTitle("D for HF+ jeta = 36; depth = 1 \b");
          if (kcountHFpositivedirectionRecosignalD1 == 10)
            HFpositivedirectionRecosignalD1->SetXTitle("D for HF+ jeta = 37; depth = 1 \b");
          if (kcountHFpositivedirectionRecosignalD1 == 11)
            HFpositivedirectionRecosignalD1->SetXTitle("D for HF+ jeta = 38; depth = 1 \b");
          if (kcountHFpositivedirectionRecosignalD1 == 12)
            HFpositivedirectionRecosignalD1->SetXTitle("D for HF+ jeta = 39; depth = 1 \b");
          if (kcountHFpositivedirectionRecosignalD1 == 13)
            HFpositivedirectionRecosignalD1->SetXTitle("D for HF+ jeta = 40; depth = 1 \b");
          HFpositivedirectionRecosignalD1->Draw("Error");
          kcountHFpositivedirectionRecosignalD1++;
          if (kcountHFpositivedirectionRecosignalD1 > 13)
            break;  // 4x6 = 24
        }           //ccctest>0

      }  // for i
    }    //if(jeta-41 >= 0)
  }      //for jeta
  /////////////////
  c3x5->Update();
  c3x5->Print("DrecosignalPositiveDirectionhistD1PhiSymmetryDepth1HF.png");
  c3x5->Clear();
  // clean-up
  if (h2CeffHFpositivedirectionRecosignalD1)
    delete h2CeffHFpositivedirectionRecosignalD1;
  //========================================================================================== 15
  //======================================================================
  //======================================================================1D plot: D vs phi , different eta,  depth=2
  //cout<<"      1D plot: D vs phi , different eta,  depth=2 *****" <<endl;
  c3x5->Clear();
  c3x5->Divide(3, 5);
  c3x5->cd(1);
  int kcountHFpositivedirectionRecosignalD2 = 1;
  TH1F* h2CeffHFpositivedirectionRecosignalD2 = new TH1F("h2CeffHFpositivedirectionRecosignalD2", "", nphi, 0., 72.);

  for (int jeta = 0; jeta < njeta; jeta++) {
    // positivedirectionRecosignalD:
    if (jeta - 41 >= 0) {
      //	     for (int i=0;i<ndepth;i++) {
      // depth=2
      for (int i = 1; i < 2; i++) {
        TH1F* HFpositivedirectionRecosignalD2 = (TH1F*)h2CeffHFpositivedirectionRecosignalD2->Clone("twod1");

        float ccctest = 0;  // to avoid empty massive elements
        for (int jphi = 0; jphi < nphi; jphi++) {
          double ccc1 = recosignalvarianceHF[i][jeta][jphi];
          if (arecosignalHF[i][jeta][jphi] > 0.) {
            HFpositivedirectionRecosignalD2->Fill(jphi, ccc1);
            ccctest = 1.;  //HFpositivedirectionRecosignalD2->SetBinError(i,0.01);
          }
        }  // for jphi
        if (ccctest > 0.) {
          //cout<<"1515       kcountHFpositivedirectionRecosignalD2   =     "<<kcountHFpositivedirectionRecosignalD2  <<"   jeta-41=     "<< jeta-41 <<endl;
          c3x5->cd(kcountHFpositivedirectionRecosignalD2);
          HFpositivedirectionRecosignalD2->SetMarkerStyle(20);
          HFpositivedirectionRecosignalD2->SetMarkerSize(0.4);
          HFpositivedirectionRecosignalD2->GetYaxis()->SetLabelSize(0.04);
          HFpositivedirectionRecosignalD2->SetXTitle("HFpositivedirectionRecosignalD2 \b");
          HFpositivedirectionRecosignalD2->SetMarkerColor(2);
          HFpositivedirectionRecosignalD2->SetLineColor(0);
          gPad->SetGridy();
          gPad->SetGridx();
          //	   gPad->SetLogy();
          if (kcountHFpositivedirectionRecosignalD2 == 1)
            HFpositivedirectionRecosignalD2->SetXTitle("D for HF+ jeta = 28; depth = 2 \b");
          if (kcountHFpositivedirectionRecosignalD2 == 2)
            HFpositivedirectionRecosignalD2->SetXTitle("D for HF+ jeta = 29; depth = 2 \b");
          if (kcountHFpositivedirectionRecosignalD2 == 3)
            HFpositivedirectionRecosignalD2->SetXTitle("D for HF+ jeta = 30; depth = 2 \b");
          if (kcountHFpositivedirectionRecosignalD2 == 4)
            HFpositivedirectionRecosignalD2->SetXTitle("D for HF+ jeta = 31; depth = 2 \b");
          if (kcountHFpositivedirectionRecosignalD2 == 5)
            HFpositivedirectionRecosignalD2->SetXTitle("D for HF+ jeta = 32; depth = 2 \b");
          if (kcountHFpositivedirectionRecosignalD2 == 6)
            HFpositivedirectionRecosignalD2->SetXTitle("D for HF+ jeta = 33; depth = 2 \b");
          if (kcountHFpositivedirectionRecosignalD2 == 7)
            HFpositivedirectionRecosignalD2->SetXTitle("D for HF+ jeta = 34; depth = 2 \b");
          if (kcountHFpositivedirectionRecosignalD2 == 8)
            HFpositivedirectionRecosignalD2->SetXTitle("D for HF+ jeta = 35; depth = 2 \b");
          if (kcountHFpositivedirectionRecosignalD2 == 9)
            HFpositivedirectionRecosignalD2->SetXTitle("D for HF+ jeta = 36; depth = 2 \b");
          if (kcountHFpositivedirectionRecosignalD2 == 10)
            HFpositivedirectionRecosignalD2->SetXTitle("D for HF+ jeta = 37; depth = 2 \b");
          if (kcountHFpositivedirectionRecosignalD2 == 11)
            HFpositivedirectionRecosignalD2->SetXTitle("D for HF+ jeta = 38; depth = 2 \b");
          if (kcountHFpositivedirectionRecosignalD2 == 12)
            HFpositivedirectionRecosignalD2->SetXTitle("D for HF+ jeta = 39; depth = 2 \b");
          if (kcountHFpositivedirectionRecosignalD2 == 13)
            HFpositivedirectionRecosignalD2->SetXTitle("D for HF+ jeta = 40; depth = 2 \b");
          HFpositivedirectionRecosignalD2->Draw("Error");
          kcountHFpositivedirectionRecosignalD2++;
          if (kcountHFpositivedirectionRecosignalD2 > 13)
            break;  // 4x6 = 24
        }           //ccctest>0

      }  // for i
    }    //if(jeta-41 >= 0)
  }      //for jeta
  /////////////////
  c3x5->Update();
  c3x5->Print("DrecosignalPositiveDirectionhistD1PhiSymmetryDepth2HF.png");
  c3x5->Clear();
  // clean-up
  if (h2CeffHFpositivedirectionRecosignalD2)
    delete h2CeffHFpositivedirectionRecosignalD2;
  //========================================================================================== 22222214
  //======================================================================
  //======================================================================1D plot: D vs phi , different eta,  depth=1
  //cout<<"      1D plot: D vs phi , different eta,  depth=1 *****" <<endl;
  c3x5->Clear();
  /////////////////
  c3x5->Divide(3, 5);
  c3x5->cd(1);
  int kcountHFnegativedirectionRecosignalD1 = 1;
  TH1F* h2CeffHFnegativedirectionRecosignalD1 = new TH1F("h2CeffHFnegativedirectionRecosignalD1", "", nphi, 0., 72.);

  for (int jeta = 0; jeta < njeta; jeta++) {
    // negativedirectionRecosignalD:
    if (jeta - 41 < 0) {
      //	     for (int i=0;i<ndepth;i++) {
      // depth=1
      for (int i = 0; i < 1; i++) {
        TH1F* HFnegativedirectionRecosignalD1 = (TH1F*)h2CeffHFnegativedirectionRecosignalD1->Clone("twod1");

        float ccctest = 0;  // to avoid empty massive elements
        for (int jphi = 0; jphi < nphi; jphi++) {
          double ccc1 = recosignalvarianceHF[i][jeta][jphi];
          if (arecosignalHF[i][jeta][jphi] > 0.) {
            HFnegativedirectionRecosignalD1->Fill(jphi, ccc1);
            ccctest = 1.;  //HFnegativedirectionRecosignalD1->SetBinError(i,0.01);
          }
        }  // for jphi
        if (ccctest > 0.) {
          //cout<<"1414       kcountHFnegativedirectionRecosignalD1   =     "<<kcountHFnegativedirectionRecosignalD1  <<"   jeta-41=     "<< jeta-41 <<endl;
          c3x5->cd(kcountHFnegativedirectionRecosignalD1);
          HFnegativedirectionRecosignalD1->SetMarkerStyle(20);
          HFnegativedirectionRecosignalD1->SetMarkerSize(0.4);
          HFnegativedirectionRecosignalD1->GetYaxis()->SetLabelSize(0.04);
          HFnegativedirectionRecosignalD1->SetXTitle("HFnegativedirectionRecosignalD1 \b");
          HFnegativedirectionRecosignalD1->SetMarkerColor(2);
          HFnegativedirectionRecosignalD1->SetLineColor(0);
          gPad->SetGridy();
          gPad->SetGridx();
          //	   gPad->SetLogy();
          if (kcountHFnegativedirectionRecosignalD1 == 1)
            HFnegativedirectionRecosignalD1->SetXTitle("D for HF- jeta =-41; depth = 1 \b");
          if (kcountHFnegativedirectionRecosignalD1 == 2)
            HFnegativedirectionRecosignalD1->SetXTitle("D for HF- jeta =-40; depth = 1 \b");
          if (kcountHFnegativedirectionRecosignalD1 == 3)
            HFnegativedirectionRecosignalD1->SetXTitle("D for HF- jeta =-39; depth = 1 \b");
          if (kcountHFnegativedirectionRecosignalD1 == 4)
            HFnegativedirectionRecosignalD1->SetXTitle("D for HF- jeta =-38; depth = 1 \b");
          if (kcountHFnegativedirectionRecosignalD1 == 5)
            HFnegativedirectionRecosignalD1->SetXTitle("D for HF- jeta =-37; depth = 1 \b");
          if (kcountHFnegativedirectionRecosignalD1 == 6)
            HFnegativedirectionRecosignalD1->SetXTitle("D for HF- jeta =-36; depth = 1 \b");
          if (kcountHFnegativedirectionRecosignalD1 == 7)
            HFnegativedirectionRecosignalD1->SetXTitle("D for HF- jeta =-35; depth = 1 \b");
          if (kcountHFnegativedirectionRecosignalD1 == 8)
            HFnegativedirectionRecosignalD1->SetXTitle("D for HF- jeta =-34; depth = 1 \b");
          if (kcountHFnegativedirectionRecosignalD1 == 9)
            HFnegativedirectionRecosignalD1->SetXTitle("D for HF- jeta =-33; depth = 1 \b");
          if (kcountHFnegativedirectionRecosignalD1 == 10)
            HFnegativedirectionRecosignalD1->SetXTitle("D for HF- jeta =-32; depth = 1 \b");
          if (kcountHFnegativedirectionRecosignalD1 == 11)
            HFnegativedirectionRecosignalD1->SetXTitle("D for HF- jeta =-31; depth = 1 \b");
          if (kcountHFnegativedirectionRecosignalD1 == 12)
            HFnegativedirectionRecosignalD1->SetXTitle("D for HF- jeta =-30; depth = 1 \b");
          if (kcountHFnegativedirectionRecosignalD1 == 13)
            HFnegativedirectionRecosignalD1->SetXTitle("D for HF- jeta =-29; depth = 1 \b");
          HFnegativedirectionRecosignalD1->Draw("Error");
          kcountHFnegativedirectionRecosignalD1++;
          if (kcountHFnegativedirectionRecosignalD1 > 13)
            break;  // 4x6 = 24
        }           //ccctest>0

      }  // for i
    }    //if(jeta-41< 0)
  }      //for jeta
  /////////////////
  c3x5->Update();
  c3x5->Print("DrecosignalNegativeDirectionhistD1PhiSymmetryDepth1HF.png");
  c3x5->Clear();
  // clean-up
  if (h2CeffHFnegativedirectionRecosignalD1)
    delete h2CeffHFnegativedirectionRecosignalD1;
  //========================================================================================== 22222215
  //======================================================================
  //======================================================================1D plot: D vs phi , different eta,  depth=2
  //cout<<"      1D plot: D vs phi , different eta,  depth=2 *****" <<endl;
  c3x5->Clear();
  c3x5->Divide(3, 5);
  c3x5->cd(1);
  int kcountHFnegativedirectionRecosignalD2 = 1;
  TH1F* h2CeffHFnegativedirectionRecosignalD2 = new TH1F("h2CeffHFnegativedirectionRecosignalD2", "", nphi, 0., 72.);

  for (int jeta = 0; jeta < njeta; jeta++) {
    // negativedirectionRecosignalD:
    if (jeta - 41 < 0) {
      //	     for (int i=0;i<ndepth;i++) {
      // depth=2
      for (int i = 1; i < 2; i++) {
        TH1F* HFnegativedirectionRecosignalD2 = (TH1F*)h2CeffHFnegativedirectionRecosignalD2->Clone("twod1");

        float ccctest = 0;  // to avoid empty massive elements
        for (int jphi = 0; jphi < nphi; jphi++) {
          double ccc1 = recosignalvarianceHF[i][jeta][jphi];
          if (arecosignalHF[i][jeta][jphi] > 0.) {
            HFnegativedirectionRecosignalD2->Fill(jphi, ccc1);
            ccctest = 1.;  //HFnegativedirectionRecosignalD2->SetBinError(i,0.01);
          }
        }  // for jphi
        if (ccctest > 0.) {
          //cout<<"1515       kcountHFnegativedirectionRecosignalD2   =     "<<kcountHFnegativedirectionRecosignalD2  <<"   jeta-41=     "<< jeta-41 <<endl;
          c3x5->cd(kcountHFnegativedirectionRecosignalD2);
          HFnegativedirectionRecosignalD2->SetMarkerStyle(20);
          HFnegativedirectionRecosignalD2->SetMarkerSize(0.4);
          HFnegativedirectionRecosignalD2->GetYaxis()->SetLabelSize(0.04);
          HFnegativedirectionRecosignalD2->SetXTitle("HFnegativedirectionRecosignalD2 \b");
          HFnegativedirectionRecosignalD2->SetMarkerColor(2);
          HFnegativedirectionRecosignalD2->SetLineColor(0);
          gPad->SetGridy();
          gPad->SetGridx();
          //	   gPad->SetLogy();
          if (kcountHFnegativedirectionRecosignalD2 == 1)
            HFnegativedirectionRecosignalD2->SetXTitle("D for HF- jeta =-41; depth = 2 \b");
          if (kcountHFnegativedirectionRecosignalD2 == 2)
            HFnegativedirectionRecosignalD2->SetXTitle("D for HF- jeta =-40; depth = 2 \b");
          if (kcountHFnegativedirectionRecosignalD2 == 3)
            HFnegativedirectionRecosignalD2->SetXTitle("D for HF- jeta =-39; depth = 2 \b");
          if (kcountHFnegativedirectionRecosignalD2 == 4)
            HFnegativedirectionRecosignalD2->SetXTitle("D for HF- jeta =-38; depth = 2 \b");
          if (kcountHFnegativedirectionRecosignalD2 == 5)
            HFnegativedirectionRecosignalD2->SetXTitle("D for HF- jeta =-37; depth = 2 \b");
          if (kcountHFnegativedirectionRecosignalD2 == 6)
            HFnegativedirectionRecosignalD2->SetXTitle("D for HF- jeta =-36; depth = 2 \b");
          if (kcountHFnegativedirectionRecosignalD2 == 7)
            HFnegativedirectionRecosignalD2->SetXTitle("D for HF- jeta =-35; depth = 2 \b");
          if (kcountHFnegativedirectionRecosignalD2 == 8)
            HFnegativedirectionRecosignalD2->SetXTitle("D for HF- jeta =-34; depth = 2 \b");
          if (kcountHFnegativedirectionRecosignalD2 == 9)
            HFnegativedirectionRecosignalD2->SetXTitle("D for HF- jeta =-33; depth = 2 \b");
          if (kcountHFnegativedirectionRecosignalD2 == 10)
            HFnegativedirectionRecosignalD2->SetXTitle("D for HF- jeta =-32; depth = 2 \b");
          if (kcountHFnegativedirectionRecosignalD2 == 11)
            HFnegativedirectionRecosignalD2->SetXTitle("D for HF- jeta =-31; depth = 2 \b");
          if (kcountHFnegativedirectionRecosignalD2 == 12)
            HFnegativedirectionRecosignalD2->SetXTitle("D for HF- jeta =-30; depth = 2 \b");
          if (kcountHFnegativedirectionRecosignalD2 == 13)
            HFnegativedirectionRecosignalD2->SetXTitle("D for HF- jeta =-29; depth = 2 \b");
          HFnegativedirectionRecosignalD2->Draw("Error");
          kcountHFnegativedirectionRecosignalD2++;
          if (kcountHFnegativedirectionRecosignalD2 > 13)
            break;  // 4x6 = 24
        }           //ccctest>0

      }  // for i
    }    //if(jeta-41< 0)
  }      //for jeta
  /////////////////
  c3x5->Update();
  c3x5->Print("DrecosignalNegativeDirectionhistD1PhiSymmetryDepth2HF.png");
  c3x5->Clear();
  // clean-up
  if (h2CeffHFnegativedirectionRecosignalD2)
    delete h2CeffHFnegativedirectionRecosignalD2;

  //=====================================================================       END of Recosignal HF for phi-symmetry
  //=====================================================================       END of Recosignal HF for phi-symmetry
  //=====================================================================       END of Recosignal HF for phi-symmetry
  //============================================================================================================       END of Recosignal for phi-symmetry
  //============================================================================================================       END of Recosignal for phi-symmetry
  //============================================================================================================       END of Recosignal for phi-symmetry

  ////////////////////// Start   Reconoise  Start Reconoise  Start   Reconoise  Start   Reconoise  Start   Reconoise Start  Reconoise Start Reconoise Start Reconoise Start Reconoise Start Reconoise Start
  ////////////////////// Start   Reconoise  Start Reconoise  Start   Reconoise  Start   Reconoise  Start   Reconoise Start  Reconoise Start Reconoise Start Reconoise Start Reconoise Start Reconoise Start
  ////////////////////// Start   Reconoise  Start Reconoise  Start   Reconoise  Start   Reconoise  Start   Reconoise Start  Reconoise Start Reconoise Start Reconoise Start Reconoise Start Reconoise Start
  ////////////////////////////////////////////////////////////////////////////////////////////////////     Reconoise HB
  ////////////////////////////////////////////////////////////////////////////////////////////////////     Reconoise HB
  ////////////////////////////////////////////////////////////////////////////////////////////////////     Reconoise HB
  //  int k_max[5]={0,4,7,4,4}; // maximum depth for each subdet
  //ndepth = k_max[5];
  ndepth = 4;
  double areconoiseHB[ndepth][njeta][njphi];
  double breconoiseHB[ndepth][njeta][njphi];
  double reconoisevarianceHB[ndepth][njeta][njphi];
  //                                   RRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRR:   Reconoise HB  recNoiseEnergy
  TH2F* recNoiseEnergy1HB1 = (TH2F*)dir->FindObjectAny("h_recNoiseEnergy1_HB1");
  TH2F* recNoiseEnergy0HB1 = (TH2F*)dir->FindObjectAny("h_recNoiseEnergy0_HB1");
  TH2F* recNoiseEnergyHB1 = (TH2F*)recNoiseEnergy1HB1->Clone("recNoiseEnergyHB1");
  recNoiseEnergyHB1->Divide(recNoiseEnergy1HB1, recNoiseEnergy0HB1, 1, 1, "B");
  TH2F* recNoiseEnergy1HB2 = (TH2F*)dir->FindObjectAny("h_recNoiseEnergy1_HB2");
  TH2F* recNoiseEnergy0HB2 = (TH2F*)dir->FindObjectAny("h_recNoiseEnergy0_HB2");
  TH2F* recNoiseEnergyHB2 = (TH2F*)recNoiseEnergy1HB2->Clone("recNoiseEnergyHB2");
  recNoiseEnergyHB2->Divide(recNoiseEnergy1HB2, recNoiseEnergy0HB2, 1, 1, "B");
  TH2F* recNoiseEnergy1HB3 = (TH2F*)dir->FindObjectAny("h_recNoiseEnergy1_HB3");
  TH2F* recNoiseEnergy0HB3 = (TH2F*)dir->FindObjectAny("h_recNoiseEnergy0_HB3");
  TH2F* recNoiseEnergyHB3 = (TH2F*)recNoiseEnergy1HB3->Clone("recNoiseEnergyHB3");
  recNoiseEnergyHB3->Divide(recNoiseEnergy1HB3, recNoiseEnergy0HB3, 1, 1, "B");
  TH2F* recNoiseEnergy1HB4 = (TH2F*)dir->FindObjectAny("h_recNoiseEnergy1_HB4");
  TH2F* recNoiseEnergy0HB4 = (TH2F*)dir->FindObjectAny("h_recNoiseEnergy0_HB4");
  TH2F* recNoiseEnergyHB4 = (TH2F*)recNoiseEnergy1HB4->Clone("recNoiseEnergyHB4");
  recNoiseEnergyHB4->Divide(recNoiseEnergy1HB4, recNoiseEnergy0HB4, 1, 1, "B");
  for (int jeta = 0; jeta < njeta; jeta++) {
    if ((jeta - 41 >= -16 && jeta - 41 <= -1) || (jeta - 41 >= 0 && jeta - 41 <= 15)) {
      //====================================================================== PHI normalization & put R into massive areconoiseHB
      //preparation for PHI normalization:
      double sumreconoiseHB0 = 0;
      int nsumreconoiseHB0 = 0;
      double sumreconoiseHB1 = 0;
      int nsumreconoiseHB1 = 0;
      double sumreconoiseHB2 = 0;
      int nsumreconoiseHB2 = 0;
      double sumreconoiseHB3 = 0;
      int nsumreconoiseHB3 = 0;
      for (int jphi = 0; jphi < njphi; jphi++) {
        areconoiseHB[0][jeta][jphi] = recNoiseEnergyHB1->GetBinContent(jeta + 1, jphi + 1);
        areconoiseHB[1][jeta][jphi] = recNoiseEnergyHB2->GetBinContent(jeta + 1, jphi + 1);
        areconoiseHB[2][jeta][jphi] = recNoiseEnergyHB3->GetBinContent(jeta + 1, jphi + 1);
        areconoiseHB[3][jeta][jphi] = recNoiseEnergyHB4->GetBinContent(jeta + 1, jphi + 1);
        breconoiseHB[0][jeta][jphi] = recNoiseEnergyHB1->GetBinContent(jeta + 1, jphi + 1);
        breconoiseHB[1][jeta][jphi] = recNoiseEnergyHB2->GetBinContent(jeta + 1, jphi + 1);
        breconoiseHB[2][jeta][jphi] = recNoiseEnergyHB3->GetBinContent(jeta + 1, jphi + 1);
        breconoiseHB[3][jeta][jphi] = recNoiseEnergyHB4->GetBinContent(jeta + 1, jphi + 1);
        if (areconoiseHB[0][jeta][jphi] != 0.) {
          sumreconoiseHB0 += areconoiseHB[0][jeta][jphi];
          ++nsumreconoiseHB0;
        }
        if (areconoiseHB[1][jeta][jphi] != 0.) {
          sumreconoiseHB1 += areconoiseHB[1][jeta][jphi];
          ++nsumreconoiseHB1;
        }
        if (areconoiseHB[2][jeta][jphi] != 0.) {
          sumreconoiseHB2 += areconoiseHB[2][jeta][jphi];
          ++nsumreconoiseHB2;
        }
        if (areconoiseHB[3][jeta][jphi] != 0.) {
          sumreconoiseHB3 += areconoiseHB[3][jeta][jphi];
          ++nsumreconoiseHB3;
        }
      }  // phi
      // PHI normalization:  DIF
      for (int jphi = 0; jphi < njphi; jphi++) {
        if (sumreconoiseHB0 != 0.)
          breconoiseHB[0][jeta][jphi] -= (sumreconoiseHB0 / nsumreconoiseHB0);
        if (sumreconoiseHB1 != 0.)
          breconoiseHB[1][jeta][jphi] -= (sumreconoiseHB1 / nsumreconoiseHB1);
        if (sumreconoiseHB2 != 0.)
          breconoiseHB[2][jeta][jphi] -= (sumreconoiseHB2 / nsumreconoiseHB2);
        if (sumreconoiseHB3 != 0.)
          breconoiseHB[3][jeta][jphi] -= (sumreconoiseHB3 / nsumreconoiseHB3);
      }  // phi
      // PHI normalization:  R
      for (int jphi = 0; jphi < njphi; jphi++) {
        if (areconoiseHB[0][jeta][jphi] != 0.)
          areconoiseHB[0][jeta][jphi] /= (sumreconoiseHB0 / nsumreconoiseHB0);
        if (areconoiseHB[1][jeta][jphi] != 0.)
          areconoiseHB[1][jeta][jphi] /= (sumreconoiseHB1 / nsumreconoiseHB1);
        if (areconoiseHB[2][jeta][jphi] != 0.)
          areconoiseHB[2][jeta][jphi] /= (sumreconoiseHB2 / nsumreconoiseHB2);
        if (areconoiseHB[3][jeta][jphi] != 0.)
          areconoiseHB[3][jeta][jphi] /= (sumreconoiseHB3 / nsumreconoiseHB3);
      }  // phi
    }    //if eta
  }      //eta
  //------------------------  2D-eta/phi-plot: R, averaged over depthfs
  //======================================================================
  //======================================================================
  //cout<<"      R2D-eta/phi-plot: R, averaged over depthfs *****" <<endl;
  c2x1->Clear();
  /////////////////
  c2x1->Divide(2, 1);
  c2x1->cd(1);
  TH2F* GefzRreconoiseHB42D = new TH2F("GefzRreconoiseHB42D", "", neta, -41., 41., nphi, 0., 72.);
  TH2F* GefzRreconoiseHB42D0 = new TH2F("GefzRreconoiseHB42D0", "", neta, -41., 41., nphi, 0., 72.);
  TH2F* GefzRreconoiseHB42DF = (TH2F*)GefzRreconoiseHB42D0->Clone("GefzRreconoiseHB42DF");
  for (int i = 0; i < ndepth; i++) {
    for (int jeta = 0; jeta < neta; jeta++) {
      if ((jeta - 41 >= -16 && jeta - 41 <= -1) || (jeta - 41 >= 0 && jeta - 41 <= 15)) {
        for (int jphi = 0; jphi < nphi; jphi++) {
          double ccc1 = areconoiseHB[i][jeta][jphi];
          int k2plot = jeta - 41;
          int kkk = k2plot;  //if(k2plot >0 ) kkk=k2plot+1; //-41 +41 !=0
          if (ccc1 != 0.) {
            GefzRreconoiseHB42D->Fill(kkk, jphi, ccc1);
            GefzRreconoiseHB42D0->Fill(kkk, jphi, 1.);
          }
        }
      }
    }
  }
  GefzRreconoiseHB42DF->Divide(GefzRreconoiseHB42D, GefzRreconoiseHB42D0, 1, 1, "B");  // average A
  gPad->SetGridy();
  gPad->SetGridx();  //      gPad->SetLogz();
  GefzRreconoiseHB42DF->SetXTitle("<R>_depth       #eta  \b");
  GefzRreconoiseHB42DF->SetYTitle("      #phi \b");
  GefzRreconoiseHB42DF->Draw("COLZ");

  c2x1->cd(2);
  TH1F* energyhitNoise_HB = (TH1F*)dir->FindObjectAny("h_energyhitNoise_HB");
  energyhitNoise_HB->SetMarkerStyle(20);
  energyhitNoise_HB->SetMarkerSize(0.4);
  energyhitNoise_HB->GetYaxis()->SetLabelSize(0.04);
  energyhitNoise_HB->SetXTitle("energyhitNoise_HB \b");
  energyhitNoise_HB->SetMarkerColor(2);
  energyhitNoise_HB->SetLineColor(0);
  gPad->SetGridy();
  gPad->SetGridx();
  energyhitNoise_HB->Draw("Error");

  /////////////////
  c2x1->Update();
  c2x1->Print("RreconoiseGeneralD2PhiSymmetryHB.png");
  c2x1->Clear();
  // clean-up
  if (GefzRreconoiseHB42D)
    delete GefzRreconoiseHB42D;
  if (GefzRreconoiseHB42D0)
    delete GefzRreconoiseHB42D0;
  if (GefzRreconoiseHB42DF)
    delete GefzRreconoiseHB42DF;
  //====================================================================== 1D plot: R vs phi , averaged over depthfs & eta
  //======================================================================
  //cout<<"      1D plot: R vs phi , averaged over depthfs & eta *****" <<endl;
  c1x1->Clear();
  /////////////////
  c1x1->Divide(1, 1);
  c1x1->cd(1);
  TH1F* GefzRreconoiseHB41D = new TH1F("GefzRreconoiseHB41D", "", nphi, 0., 72.);
  TH1F* GefzRreconoiseHB41D0 = new TH1F("GefzRreconoiseHB41D0", "", nphi, 0., 72.);
  TH1F* GefzRreconoiseHB41DF = (TH1F*)GefzRreconoiseHB41D0->Clone("GefzRreconoiseHB41DF");
  for (int jphi = 0; jphi < nphi; jphi++) {
    for (int jeta = 0; jeta < neta; jeta++) {
      if ((jeta - 41 >= -16 && jeta - 41 <= -1) || (jeta - 41 >= 0 && jeta - 41 <= 15)) {
        for (int i = 0; i < ndepth; i++) {
          double ccc1 = areconoiseHB[i][jeta][jphi];
          if (ccc1 != 0.) {
            GefzRreconoiseHB41D->Fill(jphi, ccc1);
            GefzRreconoiseHB41D0->Fill(jphi, 1.);
          }
        }
      }
    }
  }
  GefzRreconoiseHB41DF->Divide(GefzRreconoiseHB41D, GefzRreconoiseHB41D0, 1, 1, "B");  // R averaged over depthfs & eta
  GefzRreconoiseHB41D0->Sumw2();
  //    for (int jphi=1;jphi<73;jphi++) {GefzRreconoiseHB41DF->SetBinError(jphi,0.01);}
  gPad->SetGridy();
  gPad->SetGridx();  //      gPad->SetLogz();
  GefzRreconoiseHB41DF->SetMarkerStyle(20);
  GefzRreconoiseHB41DF->SetMarkerSize(1.4);
  GefzRreconoiseHB41DF->GetZaxis()->SetLabelSize(0.08);
  GefzRreconoiseHB41DF->SetXTitle("#phi  \b");
  GefzRreconoiseHB41DF->SetYTitle("  <R> \b");
  GefzRreconoiseHB41DF->SetZTitle("<R>_PHI  - AllDepthfs \b");
  GefzRreconoiseHB41DF->SetMarkerColor(4);
  GefzRreconoiseHB41DF->SetLineColor(
      4);  //GefzRreconoiseHB41DF->SetMinimum(0.8);     //      GefzRreconoiseHB41DF->SetMaximum(1.000);
  GefzRreconoiseHB41DF->Draw("Error");
  /////////////////
  c1x1->Update();
  c1x1->Print("RreconoiseGeneralD1PhiSymmetryHB.png");
  c1x1->Clear();
  // clean-up
  if (GefzRreconoiseHB41D)
    delete GefzRreconoiseHB41D;
  if (GefzRreconoiseHB41D0)
    delete GefzRreconoiseHB41D0;
  if (GefzRreconoiseHB41DF)
    delete GefzRreconoiseHB41DF;
  //========================================================================================== 4
  //======================================================================
  //======================================================================1D plot: R vs phi , different eta,  depth=1
  //cout<<"      1D plot: R vs phi , different eta,  depth=1 *****" <<endl;
  c3x5->Clear();
  /////////////////
  c3x5->Divide(4, 4);
  c3x5->cd(1);
  int kcountHBpositivedirectionReconoise1 = 1;
  TH1F* h2CeffHBpositivedirectionReconoise1 = new TH1F("h2CeffHBpositivedirectionReconoise1", "", nphi, 0., 72.);
  for (int jeta = 0; jeta < njeta; jeta++) {
    // positivedirectionReconoise:
    if (jeta - 41 >= 0 && jeta - 41 <= 15) {
      //	     for (int i=0;i<ndepth;i++) {
      // depth=1
      for (int i = 0; i < 1; i++) {
        TH1F* HBpositivedirectionReconoise1 = (TH1F*)h2CeffHBpositivedirectionReconoise1->Clone("twod1");
        float ccctest = 0;  // to avoid empty massive elements
        for (int jphi = 0; jphi < nphi; jphi++) {
          double ccc1 = areconoiseHB[i][jeta][jphi];
          if (ccc1 != 0.) {
            HBpositivedirectionReconoise1->Fill(jphi, ccc1);
            ccctest = 1.;  //HBpositivedirectionReconoise1->SetBinError(i,0.01);
          }
        }  // for jphi
        if (ccctest > 0.) {
          //	  cout<<"444        kcountHBpositivedirectionReconoise1   =     "<<kcountHBpositivedirectionReconoise1  <<"   jeta-41=     "<< jeta-41 <<endl;
          c3x5->cd(kcountHBpositivedirectionReconoise1);
          HBpositivedirectionReconoise1->SetMarkerStyle(20);
          HBpositivedirectionReconoise1->SetMarkerSize(0.4);
          HBpositivedirectionReconoise1->GetYaxis()->SetLabelSize(0.04);
          HBpositivedirectionReconoise1->SetXTitle("HBpositivedirectionReconoise1 \b");
          HBpositivedirectionReconoise1->SetMarkerColor(2);
          HBpositivedirectionReconoise1->SetLineColor(0);
          gPad->SetGridy();
          gPad->SetGridx();
          //	   gPad->SetLogy();
          if (kcountHBpositivedirectionReconoise1 == 1)
            HBpositivedirectionReconoise1->SetXTitle("R for HB+ jeta =  0; depth = 1 \b");
          if (kcountHBpositivedirectionReconoise1 == 2)
            HBpositivedirectionReconoise1->SetXTitle("R for HB+ jeta =  1; depth = 1 \b");
          if (kcountHBpositivedirectionReconoise1 == 3)
            HBpositivedirectionReconoise1->SetXTitle("R for HB+ jeta =  2; depth = 1 \b");
          if (kcountHBpositivedirectionReconoise1 == 4)
            HBpositivedirectionReconoise1->SetXTitle("R for HB+ jeta =  3; depth = 1 \b");
          if (kcountHBpositivedirectionReconoise1 == 5)
            HBpositivedirectionReconoise1->SetXTitle("R for HB+ jeta =  4; depth = 1 \b");
          if (kcountHBpositivedirectionReconoise1 == 6)
            HBpositivedirectionReconoise1->SetXTitle("R for HB+ jeta =  5; depth = 1 \b");
          if (kcountHBpositivedirectionReconoise1 == 7)
            HBpositivedirectionReconoise1->SetXTitle("R for HB+ jeta =  6; depth = 1 \b");
          if (kcountHBpositivedirectionReconoise1 == 8)
            HBpositivedirectionReconoise1->SetXTitle("R for HB+ jeta =  7; depth = 1 \b");
          if (kcountHBpositivedirectionReconoise1 == 9)
            HBpositivedirectionReconoise1->SetXTitle("R for HB+ jeta =  8; depth = 1 \b");
          if (kcountHBpositivedirectionReconoise1 == 10)
            HBpositivedirectionReconoise1->SetXTitle("R for HB+ jeta =  9; depth = 1 \b");
          if (kcountHBpositivedirectionReconoise1 == 11)
            HBpositivedirectionReconoise1->SetXTitle("R for HB+ jeta = 10; depth = 1 \b");
          if (kcountHBpositivedirectionReconoise1 == 12)
            HBpositivedirectionReconoise1->SetXTitle("R for HB+ jeta = 11; depth = 1 \b");
          if (kcountHBpositivedirectionReconoise1 == 13)
            HBpositivedirectionReconoise1->SetXTitle("R for HB+ jeta = 12; depth = 1 \b");
          if (kcountHBpositivedirectionReconoise1 == 14)
            HBpositivedirectionReconoise1->SetXTitle("R for HB+ jeta = 13; depth = 1 \b");
          if (kcountHBpositivedirectionReconoise1 == 15)
            HBpositivedirectionReconoise1->SetXTitle("R for HB+ jeta = 14; depth = 1 \b");
          if (kcountHBpositivedirectionReconoise1 == 16)
            HBpositivedirectionReconoise1->SetXTitle("R for HB+ jeta = 15; depth = 1 \b");
          HBpositivedirectionReconoise1->Draw("Error");
          kcountHBpositivedirectionReconoise1++;
          if (kcountHBpositivedirectionReconoise1 > 16)
            break;  //
        }           //ccctest>0

      }  // for i
    }    //if(jeta-41 >= 0)
  }      //for jeta
  /////////////////
  c3x5->Update();
  c3x5->Print("RreconoisePositiveDirectionhistD1PhiSymmetryDepth1HB.png");
  c3x5->Clear();
  // clean-up
  if (h2CeffHBpositivedirectionReconoise1)
    delete h2CeffHBpositivedirectionReconoise1;

  //========================================================================================== 5
  //======================================================================
  //======================================================================1D plot: R vs phi , different eta,  depth=2
  //  cout<<"      1D plot: R vs phi , different eta,  depth=2 *****" <<endl;
  c3x5->Clear();
  /////////////////
  c3x5->Divide(4, 4);
  c3x5->cd(1);
  int kcountHBpositivedirectionReconoise2 = 1;
  TH1F* h2CeffHBpositivedirectionReconoise2 = new TH1F("h2CeffHBpositivedirectionReconoise2", "", nphi, 0., 72.);
  for (int jeta = 0; jeta < njeta; jeta++) {
    // positivedirectionReconoise:
    if (jeta - 41 >= 0 && jeta - 41 <= 15) {
      //	     for (int i=0;i<ndepth;i++) {
      // depth=2
      for (int i = 1; i < 2; i++) {
        TH1F* HBpositivedirectionReconoise2 = (TH1F*)h2CeffHBpositivedirectionReconoise2->Clone("twod1");
        float ccctest = 0;  // to avoid empty massive elements
        for (int jphi = 0; jphi < nphi; jphi++) {
          double ccc1 = areconoiseHB[i][jeta][jphi];
          if (ccc1 != 0.) {
            HBpositivedirectionReconoise2->Fill(jphi, ccc1);
            ccctest = 1.;  //HBpositivedirectionReconoise2->SetBinError(i,0.01);
          }
        }  // for jphi
        if (ccctest > 0.) {
          //cout<<"555        kcountHBpositivedirectionReconoise2   =     "<<kcountHBpositivedirectionReconoise2  <<"   jeta-41=     "<< jeta-41 <<endl;
          c3x5->cd(kcountHBpositivedirectionReconoise2);
          HBpositivedirectionReconoise2->SetMarkerStyle(20);
          HBpositivedirectionReconoise2->SetMarkerSize(0.4);
          HBpositivedirectionReconoise2->GetYaxis()->SetLabelSize(0.04);
          HBpositivedirectionReconoise2->SetXTitle("HBpositivedirectionReconoise2 \b");
          HBpositivedirectionReconoise2->SetMarkerColor(2);
          HBpositivedirectionReconoise2->SetLineColor(0);
          gPad->SetGridy();
          gPad->SetGridx();
          //	   gPad->SetLogy();
          if (kcountHBpositivedirectionReconoise2 == 1)
            HBpositivedirectionReconoise2->SetXTitle("R for HB+ jeta =  0; depth = 2 \b");
          if (kcountHBpositivedirectionReconoise2 == 2)
            HBpositivedirectionReconoise2->SetXTitle("R for HB+ jeta =  1; depth = 2 \b");
          if (kcountHBpositivedirectionReconoise2 == 3)
            HBpositivedirectionReconoise2->SetXTitle("R for HB+ jeta =  2; depth = 2 \b");
          if (kcountHBpositivedirectionReconoise2 == 4)
            HBpositivedirectionReconoise2->SetXTitle("R for HB+ jeta =  3; depth = 2 \b");
          if (kcountHBpositivedirectionReconoise2 == 5)
            HBpositivedirectionReconoise2->SetXTitle("R for HB+ jeta =  4; depth = 2 \b");
          if (kcountHBpositivedirectionReconoise2 == 6)
            HBpositivedirectionReconoise2->SetXTitle("R for HB+ jeta =  5; depth = 2 \b");
          if (kcountHBpositivedirectionReconoise2 == 7)
            HBpositivedirectionReconoise2->SetXTitle("R for HB+ jeta =  6; depth = 2 \b");
          if (kcountHBpositivedirectionReconoise2 == 8)
            HBpositivedirectionReconoise2->SetXTitle("R for HB+ jeta =  7; depth = 2 \b");
          if (kcountHBpositivedirectionReconoise2 == 9)
            HBpositivedirectionReconoise2->SetXTitle("R for HB+ jeta =  8; depth = 2 \b");
          if (kcountHBpositivedirectionReconoise2 == 10)
            HBpositivedirectionReconoise2->SetXTitle("R for HB+ jeta =  9; depth = 2 \b");
          if (kcountHBpositivedirectionReconoise2 == 11)
            HBpositivedirectionReconoise2->SetXTitle("R for HB+ jeta = 10; depth = 2 \b");
          if (kcountHBpositivedirectionReconoise2 == 12)
            HBpositivedirectionReconoise2->SetXTitle("R for HB+ jeta = 11; depth = 2 \b");
          if (kcountHBpositivedirectionReconoise2 == 13)
            HBpositivedirectionReconoise2->SetXTitle("R for HB+ jeta = 12; depth = 2 \b");
          if (kcountHBpositivedirectionReconoise2 == 14)
            HBpositivedirectionReconoise2->SetXTitle("R for HB+ jeta = 13; depth = 2 \b");
          if (kcountHBpositivedirectionReconoise2 == 15)
            HBpositivedirectionReconoise2->SetXTitle("R for HB+ jeta = 14; depth = 2 \b");
          if (kcountHBpositivedirectionReconoise2 == 16)
            HBpositivedirectionReconoise2->SetXTitle("R for HB+ jeta = 15; depth = 2 \b");
          HBpositivedirectionReconoise2->Draw("Error");
          kcountHBpositivedirectionReconoise2++;
          if (kcountHBpositivedirectionReconoise2 > 16)
            break;  // 4x6 = 24
        }           //ccctest>0

      }  // for i
    }    //if(jeta-41 >= 0)
  }      //for jeta
  /////////////////
  c3x5->Update();
  c3x5->Print("RreconoisePositiveDirectionhistD1PhiSymmetryDepth2HB.png");
  c3x5->Clear();
  // clean-up
  if (h2CeffHBpositivedirectionReconoise2)
    delete h2CeffHBpositivedirectionReconoise2;
  //========================================================================================== 6
  //======================================================================
  //======================================================================1D plot: R vs phi , different eta,  depth=3
  //cout<<"      1D plot: R vs phi , different eta,  depth=3 *****" <<endl;
  c3x5->Clear();
  /////////////////
  c3x5->Divide(4, 4);
  c3x5->cd(1);
  int kcountHBpositivedirectionReconoise3 = 1;
  TH1F* h2CeffHBpositivedirectionReconoise3 = new TH1F("h2CeffHBpositivedirectionReconoise3", "", nphi, 0., 72.);
  for (int jeta = 0; jeta < njeta; jeta++) {
    // positivedirectionReconoise:
    if (jeta - 41 >= 0 && jeta - 41 <= 15) {
      //	     for (int i=0;i<ndepth;i++) {
      // depth=3
      for (int i = 2; i < 3; i++) {
        TH1F* HBpositivedirectionReconoise3 = (TH1F*)h2CeffHBpositivedirectionReconoise3->Clone("twod1");
        float ccctest = 0;  // to avoid empty massive elements
        for (int jphi = 0; jphi < nphi; jphi++) {
          double ccc1 = areconoiseHB[i][jeta][jphi];
          if (ccc1 != 0.) {
            HBpositivedirectionReconoise3->Fill(jphi, ccc1);
            ccctest = 1.;  //HBpositivedirectionReconoise3->SetBinError(i,0.01);
          }
        }  // for jphi
        if (ccctest > 0.) {
          //cout<<"666        kcountHBpositivedirectionReconoise3   =     "<<kcountHBpositivedirectionReconoise3  <<"   jeta-41=     "<< jeta-41 <<endl;
          c3x5->cd(kcountHBpositivedirectionReconoise3);
          HBpositivedirectionReconoise3->SetMarkerStyle(20);
          HBpositivedirectionReconoise3->SetMarkerSize(0.4);
          HBpositivedirectionReconoise3->GetYaxis()->SetLabelSize(0.04);
          HBpositivedirectionReconoise3->SetXTitle("HBpositivedirectionReconoise3 \b");
          HBpositivedirectionReconoise3->SetMarkerColor(2);
          HBpositivedirectionReconoise3->SetLineColor(0);
          gPad->SetGridy();
          gPad->SetGridx();
          //	   gPad->SetLogy();
          if (kcountHBpositivedirectionReconoise3 == 1)
            HBpositivedirectionReconoise3->SetXTitle("R for HB+ jeta =  0; depth = 3 \b");
          if (kcountHBpositivedirectionReconoise3 == 2)
            HBpositivedirectionReconoise3->SetXTitle("R for HB+ jeta =  1; depth = 3 \b");
          if (kcountHBpositivedirectionReconoise3 == 3)
            HBpositivedirectionReconoise3->SetXTitle("R for HB+ jeta =  2; depth = 3 \b");
          if (kcountHBpositivedirectionReconoise3 == 4)
            HBpositivedirectionReconoise3->SetXTitle("R for HB+ jeta =  3; depth = 3 \b");
          if (kcountHBpositivedirectionReconoise3 == 5)
            HBpositivedirectionReconoise3->SetXTitle("R for HB+ jeta =  4; depth = 3 \b");
          if (kcountHBpositivedirectionReconoise3 == 6)
            HBpositivedirectionReconoise3->SetXTitle("R for HB+ jeta =  5; depth = 3 \b");
          if (kcountHBpositivedirectionReconoise3 == 7)
            HBpositivedirectionReconoise3->SetXTitle("R for HB+ jeta =  6; depth = 3 \b");
          if (kcountHBpositivedirectionReconoise3 == 8)
            HBpositivedirectionReconoise3->SetXTitle("R for HB+ jeta =  7; depth = 3 \b");
          if (kcountHBpositivedirectionReconoise3 == 9)
            HBpositivedirectionReconoise3->SetXTitle("R for HB+ jeta =  8; depth = 3 \b");
          if (kcountHBpositivedirectionReconoise3 == 10)
            HBpositivedirectionReconoise3->SetXTitle("R for HB+ jeta =  9; depth = 3 \b");
          if (kcountHBpositivedirectionReconoise3 == 11)
            HBpositivedirectionReconoise3->SetXTitle("R for HB+ jeta =  0; depth = 3 \b");
          if (kcountHBpositivedirectionReconoise3 == 12)
            HBpositivedirectionReconoise3->SetXTitle("R for HB+ jeta = 11; depth = 3 \b");
          if (kcountHBpositivedirectionReconoise3 == 13)
            HBpositivedirectionReconoise3->SetXTitle("R for HB+ jeta = 12; depth = 3 \b");
          if (kcountHBpositivedirectionReconoise3 == 14)
            HBpositivedirectionReconoise3->SetXTitle("R for HB+ jeta = 13; depth = 3 \b");
          if (kcountHBpositivedirectionReconoise3 == 15)
            HBpositivedirectionReconoise3->SetXTitle("R for HB+ jeta = 14; depth = 3 \b");
          if (kcountHBpositivedirectionReconoise3 == 16)
            HBpositivedirectionReconoise3->SetXTitle("R for HB+ jeta = 15; depth = 3 \b");
          HBpositivedirectionReconoise3->Draw("Error");
          kcountHBpositivedirectionReconoise3++;
          if (kcountHBpositivedirectionReconoise3 > 16)
            break;  // 4x6 = 24
        }           //ccctest>0

      }  // for i
    }    //if(jeta-41 >= 0)
  }      //for jeta
  /////////////////
  c3x5->Update();
  c3x5->Print("RreconoisePositiveDirectionhistD1PhiSymmetryDepth3HB.png");
  c3x5->Clear();
  // clean-up
  if (h2CeffHBpositivedirectionReconoise3)
    delete h2CeffHBpositivedirectionReconoise3;
  //========================================================================================== 7
  //======================================================================
  //======================================================================1D plot: R vs phi , different eta,  depth=4
  //cout<<"      1D plot: R vs phi , different eta,  depth=4 *****" <<endl;
  c3x5->Clear();
  /////////////////
  c3x5->Divide(4, 4);
  c3x5->cd(1);
  int kcountHBpositivedirectionReconoise4 = 1;
  TH1F* h2CeffHBpositivedirectionReconoise4 = new TH1F("h2CeffHBpositivedirectionReconoise4", "", nphi, 0., 72.);

  for (int jeta = 0; jeta < njeta; jeta++) {
    // positivedirectionReconoise:
    if (jeta - 41 >= 0 && jeta - 41 <= 15) {
      //	     for (int i=0;i<ndepth;i++) {
      // depth=4
      for (int i = 3; i < 4; i++) {
        TH1F* HBpositivedirectionReconoise4 = (TH1F*)h2CeffHBpositivedirectionReconoise4->Clone("twod1");

        float ccctest = 0;  // to avoid empty massive elements
        for (int jphi = 0; jphi < nphi; jphi++) {
          double ccc1 = areconoiseHB[i][jeta][jphi];
          if (ccc1 != 0.) {
            HBpositivedirectionReconoise4->Fill(jphi, ccc1);
            ccctest = 1.;  //HBpositivedirectionReconoise4->SetBinError(i,0.01);
          }
        }  // for jphi
        if (ccctest > 0.) {
          //cout<<"777        kcountHBpositivedirectionReconoise4   =     "<<kcountHBpositivedirectionReconoise4  <<"   jeta-41=     "<< jeta-41 <<endl;
          c3x5->cd(kcountHBpositivedirectionReconoise4);
          HBpositivedirectionReconoise4->SetMarkerStyle(20);
          HBpositivedirectionReconoise4->SetMarkerSize(0.4);
          HBpositivedirectionReconoise4->GetYaxis()->SetLabelSize(0.04);
          HBpositivedirectionReconoise4->SetXTitle("HBpositivedirectionReconoise4 \b");
          HBpositivedirectionReconoise4->SetMarkerColor(2);
          HBpositivedirectionReconoise4->SetLineColor(0);
          gPad->SetGridy();
          gPad->SetGridx();
          //	   gPad->SetLogy();
          if (kcountHBpositivedirectionReconoise4 == 1)
            HBpositivedirectionReconoise4->SetXTitle("R for HB+ jeta =  0; depth = 4 \b");
          if (kcountHBpositivedirectionReconoise4 == 2)
            HBpositivedirectionReconoise4->SetXTitle("R for HB+ jeta =  1; depth = 4 \b");
          if (kcountHBpositivedirectionReconoise4 == 3)
            HBpositivedirectionReconoise4->SetXTitle("R for HB+ jeta =  2; depth = 4 \b");
          if (kcountHBpositivedirectionReconoise4 == 4)
            HBpositivedirectionReconoise4->SetXTitle("R for HB+ jeta =  3; depth = 4 \b");
          if (kcountHBpositivedirectionReconoise4 == 5)
            HBpositivedirectionReconoise4->SetXTitle("R for HB+ jeta =  4; depth = 4 \b");
          if (kcountHBpositivedirectionReconoise4 == 6)
            HBpositivedirectionReconoise4->SetXTitle("R for HB+ jeta =  5; depth = 4 \b");
          if (kcountHBpositivedirectionReconoise4 == 7)
            HBpositivedirectionReconoise4->SetXTitle("R for HB+ jeta =  6; depth = 4 \b");
          if (kcountHBpositivedirectionReconoise4 == 8)
            HBpositivedirectionReconoise4->SetXTitle("R for HB+ jeta =  7; depth = 4 \b");
          if (kcountHBpositivedirectionReconoise4 == 9)
            HBpositivedirectionReconoise4->SetXTitle("R for HB+ jeta =  8; depth = 4 \b");
          if (kcountHBpositivedirectionReconoise4 == 10)
            HBpositivedirectionReconoise4->SetXTitle("R for HB+ jeta =  9; depth = 4 \b");
          if (kcountHBpositivedirectionReconoise4 == 11)
            HBpositivedirectionReconoise4->SetXTitle("R for HB+ jeta = 10; depth = 4 \b");
          if (kcountHBpositivedirectionReconoise4 == 12)
            HBpositivedirectionReconoise4->SetXTitle("R for HB+ jeta = 11; depth = 4 \b");
          if (kcountHBpositivedirectionReconoise4 == 13)
            HBpositivedirectionReconoise4->SetXTitle("R for HB+ jeta = 12; depth = 4 \b");
          if (kcountHBpositivedirectionReconoise4 == 14)
            HBpositivedirectionReconoise4->SetXTitle("R for HB+ jeta = 13; depth = 4 \b");
          if (kcountHBpositivedirectionReconoise4 == 15)
            HBpositivedirectionReconoise4->SetXTitle("R for HB+ jeta = 14; depth = 4 \b");
          if (kcountHBpositivedirectionReconoise4 == 16)
            HBpositivedirectionReconoise4->SetXTitle("R for HB+ jeta = 15; depth = 4 \b");
          HBpositivedirectionReconoise4->Draw("Error");
          kcountHBpositivedirectionReconoise4++;
          if (kcountHBpositivedirectionReconoise4 > 16)
            break;  // 4x6 = 24
        }           //ccctest>0

      }  // for i
    }    //if(jeta-41 >= 0)
  }      //for jeta
  /////////////////
  c3x5->Update();
  c3x5->Print("RreconoisePositiveDirectionhistD1PhiSymmetryDepth4HB.png");
  c3x5->Clear();
  // clean-up
  if (h2CeffHBpositivedirectionReconoise4)
    delete h2CeffHBpositivedirectionReconoise4;

  //========================================================================================== 1114
  //======================================================================
  //======================================================================1D plot: R vs phi , different eta,  depth=1
  //cout<<"      1D plot: R vs phi , different eta,  depth=1 *****" <<endl;
  c3x5->Clear();
  /////////////////
  c3x5->Divide(4, 4);
  c3x5->cd(1);
  int kcountHBnegativedirectionReconoise1 = 1;
  TH1F* h2CeffHBnegativedirectionReconoise1 = new TH1F("h2CeffHBnegativedirectionReconoise1", "", nphi, 0., 72.);
  for (int jeta = 0; jeta < njeta; jeta++) {
    // negativedirectionReconoise:
    if (jeta - 41 >= -16 && jeta - 41 <= -1) {
      //	     for (int i=0;i<ndepth;i++) {
      // depth=1
      for (int i = 0; i < 1; i++) {
        TH1F* HBnegativedirectionReconoise1 = (TH1F*)h2CeffHBnegativedirectionReconoise1->Clone("twod1");
        float ccctest = 0;  // to avoid empty massive elements
        for (int jphi = 0; jphi < nphi; jphi++) {
          double ccc1 = areconoiseHB[i][jeta][jphi];
          if (ccc1 != 0.) {
            HBnegativedirectionReconoise1->Fill(jphi, ccc1);
            ccctest = 1.;  //HBnegativedirectionReconoise1->SetBinError(i,0.01);
          }
        }  // for jphi
        if (ccctest > 0.) {
          //	  cout<<"444        kcountHBnegativedirectionReconoise1   =     "<<kcountHBnegativedirectionReconoise1  <<"   jeta-41=     "<< jeta-41 <<endl;
          c3x5->cd(kcountHBnegativedirectionReconoise1);
          HBnegativedirectionReconoise1->SetMarkerStyle(20);
          HBnegativedirectionReconoise1->SetMarkerSize(0.4);
          HBnegativedirectionReconoise1->GetYaxis()->SetLabelSize(0.04);
          HBnegativedirectionReconoise1->SetXTitle("HBnegativedirectionReconoise1 \b");
          HBnegativedirectionReconoise1->SetMarkerColor(2);
          HBnegativedirectionReconoise1->SetLineColor(0);
          gPad->SetGridy();
          gPad->SetGridx();
          //	   gPad->SetLogy();
          if (kcountHBnegativedirectionReconoise1 == 1)
            HBnegativedirectionReconoise1->SetXTitle("R for HB- jeta = -16; depth = 1 \b");
          if (kcountHBnegativedirectionReconoise1 == 2)
            HBnegativedirectionReconoise1->SetXTitle("R for HB- jeta = -15; depth = 1 \b");
          if (kcountHBnegativedirectionReconoise1 == 3)
            HBnegativedirectionReconoise1->SetXTitle("R for HB- jeta = -14; depth = 1 \b");
          if (kcountHBnegativedirectionReconoise1 == 4)
            HBnegativedirectionReconoise1->SetXTitle("R for HB- jeta = -13; depth = 1 \b");
          if (kcountHBnegativedirectionReconoise1 == 5)
            HBnegativedirectionReconoise1->SetXTitle("R for HB- jeta = -12; depth = 1 \b");
          if (kcountHBnegativedirectionReconoise1 == 6)
            HBnegativedirectionReconoise1->SetXTitle("R for HB- jeta = -11; depth = 1 \b");
          if (kcountHBnegativedirectionReconoise1 == 7)
            HBnegativedirectionReconoise1->SetXTitle("R for HB- jeta = -10; depth = 1 \b");
          if (kcountHBnegativedirectionReconoise1 == 8)
            HBnegativedirectionReconoise1->SetXTitle("R for HB- jeta =  -9; depth = 1 \b");
          if (kcountHBnegativedirectionReconoise1 == 9)
            HBnegativedirectionReconoise1->SetXTitle("R for HB- jeta =  -8; depth = 1 \b");
          if (kcountHBnegativedirectionReconoise1 == 10)
            HBnegativedirectionReconoise1->SetXTitle("R for HB- jeta =  -7; depth = 1 \b");
          if (kcountHBnegativedirectionReconoise1 == 11)
            HBnegativedirectionReconoise1->SetXTitle("R for HB- jeta =  -6; depth = 1 \b");
          if (kcountHBnegativedirectionReconoise1 == 12)
            HBnegativedirectionReconoise1->SetXTitle("R for HB- jeta =  -5; depth = 1 \b");
          if (kcountHBnegativedirectionReconoise1 == 13)
            HBnegativedirectionReconoise1->SetXTitle("R for HB- jeta =  -4; depth = 1 \b");
          if (kcountHBnegativedirectionReconoise1 == 14)
            HBnegativedirectionReconoise1->SetXTitle("R for HB- jeta =  -3; depth = 1 \b");
          if (kcountHBnegativedirectionReconoise1 == 15)
            HBnegativedirectionReconoise1->SetXTitle("R for HB- jeta =  -2; depth = 1 \b");
          if (kcountHBnegativedirectionReconoise1 == 16)
            HBnegativedirectionReconoise1->SetXTitle("R for HB- jeta =  -1; depth = 1 \b");
          HBnegativedirectionReconoise1->Draw("Error");
          kcountHBnegativedirectionReconoise1++;
          if (kcountHBnegativedirectionReconoise1 > 16)
            break;  //
        }           //ccctest>0

      }  // for i
    }    //if(jeta-41 < 0 )
  }      //for jeta
  /////////////////
  c3x5->Update();
  c3x5->Print("RreconoiseNegativeDirectionhistD1PhiSymmetryDepth1HB.png");
  c3x5->Clear();
  // clean-up
  if (h2CeffHBnegativedirectionReconoise1)
    delete h2CeffHBnegativedirectionReconoise1;

  //========================================================================================== 1115
  //======================================================================
  //======================================================================1D plot: R vs phi , different eta,  depth=2
  //  cout<<"      1D plot: R vs phi , different eta,  depth=2 *****" <<endl;
  c3x5->Clear();
  /////////////////
  c3x5->Divide(4, 4);
  c3x5->cd(1);
  int kcountHBnegativedirectionReconoise2 = 1;
  TH1F* h2CeffHBnegativedirectionReconoise2 = new TH1F("h2CeffHBnegativedirectionReconoise2", "", nphi, 0., 72.);
  for (int jeta = 0; jeta < njeta; jeta++) {
    // negativedirectionReconoise:
    if (jeta - 41 >= -16 && jeta - 41 <= -1) {
      //	     for (int i=0;i<ndepth;i++) {
      // depth=2
      for (int i = 1; i < 2; i++) {
        TH1F* HBnegativedirectionReconoise2 = (TH1F*)h2CeffHBnegativedirectionReconoise2->Clone("twod1");
        float ccctest = 0;  // to avoid empty massive elements
        for (int jphi = 0; jphi < nphi; jphi++) {
          double ccc1 = areconoiseHB[i][jeta][jphi];
          if (ccc1 != 0.) {
            HBnegativedirectionReconoise2->Fill(jphi, ccc1);
            ccctest = 1.;  //HBnegativedirectionReconoise2->SetBinError(i,0.01);
          }
        }  // for jphi
        if (ccctest > 0.) {
          //cout<<"555        kcountHBnegativedirectionReconoise2   =     "<<kcountHBnegativedirectionReconoise2  <<"   jeta-41=     "<< jeta-41 <<endl;
          c3x5->cd(kcountHBnegativedirectionReconoise2);
          HBnegativedirectionReconoise2->SetMarkerStyle(20);
          HBnegativedirectionReconoise2->SetMarkerSize(0.4);
          HBnegativedirectionReconoise2->GetYaxis()->SetLabelSize(0.04);
          HBnegativedirectionReconoise2->SetXTitle("HBnegativedirectionReconoise2 \b");
          HBnegativedirectionReconoise2->SetMarkerColor(2);
          HBnegativedirectionReconoise2->SetLineColor(0);
          gPad->SetGridy();
          gPad->SetGridx();
          //	   gPad->SetLogy();
          if (kcountHBnegativedirectionReconoise2 == 1)
            HBnegativedirectionReconoise2->SetXTitle("R for HB- jeta = -16; depth = 2 \b");
          if (kcountHBnegativedirectionReconoise2 == 2)
            HBnegativedirectionReconoise2->SetXTitle("R for HB- jeta = -15; depth = 2 \b");
          if (kcountHBnegativedirectionReconoise2 == 3)
            HBnegativedirectionReconoise2->SetXTitle("R for HB- jeta = -14; depth = 2 \b");
          if (kcountHBnegativedirectionReconoise2 == 4)
            HBnegativedirectionReconoise2->SetXTitle("R for HB- jeta = -13; depth = 2 \b");
          if (kcountHBnegativedirectionReconoise2 == 5)
            HBnegativedirectionReconoise2->SetXTitle("R for HB- jeta = -12; depth = 2 \b");
          if (kcountHBnegativedirectionReconoise2 == 6)
            HBnegativedirectionReconoise2->SetXTitle("R for HB- jeta = -11; depth = 2 \b");
          if (kcountHBnegativedirectionReconoise2 == 7)
            HBnegativedirectionReconoise2->SetXTitle("R for HB- jeta = -10; depth = 2 \b");
          if (kcountHBnegativedirectionReconoise2 == 8)
            HBnegativedirectionReconoise2->SetXTitle("R for HB- jeta =  -9; depth = 2 \b");
          if (kcountHBnegativedirectionReconoise2 == 9)
            HBnegativedirectionReconoise2->SetXTitle("R for HB- jeta =  -8; depth = 2 \b");
          if (kcountHBnegativedirectionReconoise2 == 10)
            HBnegativedirectionReconoise2->SetXTitle("R for HB- jeta =  -7; depth = 2 \b");
          if (kcountHBnegativedirectionReconoise2 == 11)
            HBnegativedirectionReconoise2->SetXTitle("R for HB- jeta =  -6; depth = 2 \b");
          if (kcountHBnegativedirectionReconoise2 == 12)
            HBnegativedirectionReconoise2->SetXTitle("R for HB- jeta =  -5; depth = 2 \b");
          if (kcountHBnegativedirectionReconoise2 == 13)
            HBnegativedirectionReconoise2->SetXTitle("R for HB- jeta =  -4; depth = 2 \b");
          if (kcountHBnegativedirectionReconoise2 == 14)
            HBnegativedirectionReconoise2->SetXTitle("R for HB- jeta =  -3; depth = 2 \b");
          if (kcountHBnegativedirectionReconoise2 == 15)
            HBnegativedirectionReconoise2->SetXTitle("R for HB- jeta =  -2; depth = 2 \b");
          if (kcountHBnegativedirectionReconoise2 == 16)
            HBnegativedirectionReconoise2->SetXTitle("R for HB- jeta =  -1; depth = 2 \b");
          HBnegativedirectionReconoise2->Draw("Error");
          kcountHBnegativedirectionReconoise2++;
          if (kcountHBnegativedirectionReconoise2 > 16)
            break;  // 4x6 = 24
        }           //ccctest>0

      }  // for i
    }    //if(jeta-41 < 0 )
  }      //for jeta
  /////////////////
  c3x5->Update();
  c3x5->Print("RreconoiseNegativeDirectionhistD1PhiSymmetryDepth2HB.png");
  c3x5->Clear();
  // clean-up
  if (h2CeffHBnegativedirectionReconoise2)
    delete h2CeffHBnegativedirectionReconoise2;
  //========================================================================================== 1116
  //======================================================================
  //======================================================================1D plot: R vs phi , different eta,  depth=3
  //cout<<"      1D plot: R vs phi , different eta,  depth=3 *****" <<endl;
  c3x5->Clear();
  /////////////////
  c3x5->Divide(4, 4);
  c3x5->cd(1);
  int kcountHBnegativedirectionReconoise3 = 1;
  TH1F* h2CeffHBnegativedirectionReconoise3 = new TH1F("h2CeffHBnegativedirectionReconoise3", "", nphi, 0., 72.);
  for (int jeta = 0; jeta < njeta; jeta++) {
    // negativedirectionReconoise:
    if (jeta - 41 >= -16 && jeta - 41 <= -1) {
      //	     for (int i=0;i<ndepth;i++) {
      // depth=3
      for (int i = 2; i < 3; i++) {
        TH1F* HBnegativedirectionReconoise3 = (TH1F*)h2CeffHBnegativedirectionReconoise3->Clone("twod1");
        float ccctest = 0;  // to avoid empty massive elements
        for (int jphi = 0; jphi < nphi; jphi++) {
          double ccc1 = areconoiseHB[i][jeta][jphi];
          if (ccc1 != 0.) {
            HBnegativedirectionReconoise3->Fill(jphi, ccc1);
            ccctest = 1.;  //HBnegativedirectionReconoise3->SetBinError(i,0.01);
          }
        }  // for jphi
        if (ccctest > 0.) {
          //cout<<"666        kcountHBnegativedirectionReconoise3   =     "<<kcountHBnegativedirectionReconoise3  <<"   jeta-41=     "<< jeta-41 <<endl;
          c3x5->cd(kcountHBnegativedirectionReconoise3);
          HBnegativedirectionReconoise3->SetMarkerStyle(20);
          HBnegativedirectionReconoise3->SetMarkerSize(0.4);
          HBnegativedirectionReconoise3->GetYaxis()->SetLabelSize(0.04);
          HBnegativedirectionReconoise3->SetXTitle("HBnegativedirectionReconoise3 \b");
          HBnegativedirectionReconoise3->SetMarkerColor(2);
          HBnegativedirectionReconoise3->SetLineColor(0);
          gPad->SetGridy();
          gPad->SetGridx();
          //	   gPad->SetLogy();
          if (kcountHBnegativedirectionReconoise3 == 1)
            HBnegativedirectionReconoise3->SetXTitle("R for HB- jeta = -16; depth = 3 \b");
          if (kcountHBnegativedirectionReconoise3 == 2)
            HBnegativedirectionReconoise3->SetXTitle("R for HB- jeta = -15; depth = 3 \b");
          if (kcountHBnegativedirectionReconoise3 == 3)
            HBnegativedirectionReconoise3->SetXTitle("R for HB- jeta = -14; depth = 3 \b");
          if (kcountHBnegativedirectionReconoise3 == 4)
            HBnegativedirectionReconoise3->SetXTitle("R for HB- jeta = -13; depth = 3 \b");
          if (kcountHBnegativedirectionReconoise3 == 5)
            HBnegativedirectionReconoise3->SetXTitle("R for HB- jeta = -12; depth = 3 \b");
          if (kcountHBnegativedirectionReconoise3 == 6)
            HBnegativedirectionReconoise3->SetXTitle("R for HB- jeta = -11; depth = 3 \b");
          if (kcountHBnegativedirectionReconoise3 == 7)
            HBnegativedirectionReconoise3->SetXTitle("R for HB- jeta = -10; depth = 3 \b");
          if (kcountHBnegativedirectionReconoise3 == 8)
            HBnegativedirectionReconoise3->SetXTitle("R for HB- jeta =  -9; depth = 3 \b");
          if (kcountHBnegativedirectionReconoise3 == 9)
            HBnegativedirectionReconoise3->SetXTitle("R for HB- jeta =  -8; depth = 3 \b");
          if (kcountHBnegativedirectionReconoise3 == 10)
            HBnegativedirectionReconoise3->SetXTitle("R for HB- jeta =  -7; depth = 3 \b");
          if (kcountHBnegativedirectionReconoise3 == 11)
            HBnegativedirectionReconoise3->SetXTitle("R for HB- jeta =  -6; depth = 3 \b");
          if (kcountHBnegativedirectionReconoise3 == 12)
            HBnegativedirectionReconoise3->SetXTitle("R for HB- jeta =  -5; depth = 3 \b");
          if (kcountHBnegativedirectionReconoise3 == 13)
            HBnegativedirectionReconoise3->SetXTitle("R for HB- jeta =  -4; depth = 3 \b");
          if (kcountHBnegativedirectionReconoise3 == 14)
            HBnegativedirectionReconoise3->SetXTitle("R for HB- jeta =  -3; depth = 3 \b");
          if (kcountHBnegativedirectionReconoise3 == 15)
            HBnegativedirectionReconoise3->SetXTitle("R for HB- jeta =  -2; depth = 3 \b");
          if (kcountHBnegativedirectionReconoise3 == 16)
            HBnegativedirectionReconoise3->SetXTitle("R for HB- jeta =  -1; depth = 3 \b");

          HBnegativedirectionReconoise3->Draw("Error");
          kcountHBnegativedirectionReconoise3++;
          if (kcountHBnegativedirectionReconoise3 > 16)
            break;  // 4x6 = 24
        }           //ccctest>0

      }  // for i
    }    //if(jeta-41 < 0 )
  }      //for jeta
  /////////////////
  c3x5->Update();
  c3x5->Print("RreconoiseNegativeDirectionhistD1PhiSymmetryDepth3HB.png");
  c3x5->Clear();
  // clean-up
  if (h2CeffHBnegativedirectionReconoise3)
    delete h2CeffHBnegativedirectionReconoise3;
  //========================================================================================== 1117
  //======================================================================
  //======================================================================1D plot: R vs phi , different eta,  depth=4
  //cout<<"      1D plot: R vs phi , different eta,  depth=4 *****" <<endl;
  c3x5->Clear();
  /////////////////
  c3x5->Divide(4, 4);
  c3x5->cd(1);
  int kcountHBnegativedirectionReconoise4 = 1;
  TH1F* h2CeffHBnegativedirectionReconoise4 = new TH1F("h2CeffHBnegativedirectionReconoise4", "", nphi, 0., 72.);

  for (int jeta = 0; jeta < njeta; jeta++) {
    // negativedirectionReconoise:
    if (jeta - 41 >= -16 && jeta - 41 <= -1) {
      //	     for (int i=0;i<ndepth;i++) {
      // depth=4
      for (int i = 3; i < 4; i++) {
        TH1F* HBnegativedirectionReconoise4 = (TH1F*)h2CeffHBnegativedirectionReconoise4->Clone("twod1");

        float ccctest = 0;  // to avoid empty massive elements
        for (int jphi = 0; jphi < nphi; jphi++) {
          double ccc1 = areconoiseHB[i][jeta][jphi];
          if (ccc1 != 0.) {
            HBnegativedirectionReconoise4->Fill(jphi, ccc1);
            ccctest = 1.;  //HBnegativedirectionReconoise4->SetBinError(i,0.01);
          }
        }  // for jphi
        if (ccctest > 0.) {
          //cout<<"777        kcountHBnegativedirectionReconoise4   =     "<<kcountHBnegativedirectionReconoise4  <<"   jeta-41=     "<< jeta-41 <<endl;
          c3x5->cd(kcountHBnegativedirectionReconoise4);
          HBnegativedirectionReconoise4->SetMarkerStyle(20);
          HBnegativedirectionReconoise4->SetMarkerSize(0.4);
          HBnegativedirectionReconoise4->GetYaxis()->SetLabelSize(0.04);
          HBnegativedirectionReconoise4->SetXTitle("HBnegativedirectionReconoise4 \b");
          HBnegativedirectionReconoise4->SetMarkerColor(2);
          HBnegativedirectionReconoise4->SetLineColor(0);
          gPad->SetGridy();
          gPad->SetGridx();
          //	   gPad->SetLogy();
          if (kcountHBnegativedirectionReconoise4 == 1)
            HBnegativedirectionReconoise4->SetXTitle("R for HB- jeta = -16; depth = 4 \b");
          if (kcountHBnegativedirectionReconoise4 == 2)
            HBnegativedirectionReconoise4->SetXTitle("R for HB- jeta = -15; depth = 4 \b");
          if (kcountHBnegativedirectionReconoise4 == 3)
            HBnegativedirectionReconoise4->SetXTitle("R for HB- jeta = -14; depth = 4 \b");
          if (kcountHBnegativedirectionReconoise4 == 4)
            HBnegativedirectionReconoise4->SetXTitle("R for HB- jeta = -13; depth = 4 \b");
          if (kcountHBnegativedirectionReconoise4 == 5)
            HBnegativedirectionReconoise4->SetXTitle("R for HB- jeta = -12; depth = 4 \b");
          if (kcountHBnegativedirectionReconoise4 == 6)
            HBnegativedirectionReconoise4->SetXTitle("R for HB- jeta = -11; depth = 4 \b");
          if (kcountHBnegativedirectionReconoise4 == 7)
            HBnegativedirectionReconoise4->SetXTitle("R for HB- jeta = -10; depth = 4 \b");
          if (kcountHBnegativedirectionReconoise4 == 8)
            HBnegativedirectionReconoise4->SetXTitle("R for HB- jeta =  -9; depth = 4 \b");
          if (kcountHBnegativedirectionReconoise4 == 9)
            HBnegativedirectionReconoise4->SetXTitle("R for HB- jeta =  -8; depth = 4 \b");
          if (kcountHBnegativedirectionReconoise4 == 10)
            HBnegativedirectionReconoise4->SetXTitle("R for HB- jeta =  -7; depth = 4 \b");
          if (kcountHBnegativedirectionReconoise4 == 11)
            HBnegativedirectionReconoise4->SetXTitle("R for HB- jeta =  -6; depth = 4 \b");
          if (kcountHBnegativedirectionReconoise4 == 12)
            HBnegativedirectionReconoise4->SetXTitle("R for HB- jeta =  -5; depth = 4 \b");
          if (kcountHBnegativedirectionReconoise4 == 13)
            HBnegativedirectionReconoise4->SetXTitle("R for HB- jeta =  -4; depth = 4 \b");
          if (kcountHBnegativedirectionReconoise4 == 14)
            HBnegativedirectionReconoise4->SetXTitle("R for HB- jeta =  -3; depth = 4 \b");
          if (kcountHBnegativedirectionReconoise4 == 15)
            HBnegativedirectionReconoise4->SetXTitle("R for HB- jeta =  -2; depth = 4 \b");
          if (kcountHBnegativedirectionReconoise4 == 16)
            HBnegativedirectionReconoise4->SetXTitle("R for HB- jeta =  -1; depth = 4 \b");
          HBnegativedirectionReconoise4->Draw("Error");
          kcountHBnegativedirectionReconoise4++;
          if (kcountHBnegativedirectionReconoise4 > 16)
            break;  // 4x6 = 24
        }           //ccctest>0

      }  // for i
    }    //if(jeta-41 < 0 )
  }      //for jeta
  /////////////////
  c3x5->Update();
  c3x5->Print("RreconoiseNegativeDirectionhistD1PhiSymmetryDepth4HB.png");
  c3x5->Clear();
  // clean-up
  if (h2CeffHBnegativedirectionReconoise4)
    delete h2CeffHBnegativedirectionReconoise4;

  //======================================================================================================================
  //                                   DIFDIFDIFDIFDIFDIFDIFDIFDIFDIFDIFDIFDIFDIFDIFDIFDIFDIFDIFDIF:   Reconoise HE
  //======================================================================================================================
  //======================================================================
  //cout<<"      R2D-eta/phi-plot: DIF, averaged over depthfs *****" <<endl;
  c2x1->Clear();
  /////////////////
  c2x1->Divide(2, 1);
  c2x1->cd(1);
  TH2F* GefzDIFreconoiseHB42D = new TH2F("GefzDIFreconoiseHB42D", "", neta, -41., 41., nphi, 0., 72.);
  TH2F* GefzDIFreconoiseHB42D0 = new TH2F("GefzDIFreconoiseHB42D0", "", neta, -41., 41., nphi, 0., 72.);
  TH2F* GefzDIFreconoiseHB42DF = (TH2F*)GefzDIFreconoiseHB42D0->Clone("GefzDIFreconoiseHB42DF");
  for (int i = 0; i < ndepth; i++) {
    for (int jeta = 0; jeta < neta; jeta++) {
      if ((jeta - 41 >= -16 && jeta - 41 <= -1) || (jeta - 41 >= 0 && jeta - 41 <= 15)) {
        for (int jphi = 0; jphi < nphi; jphi++) {
          double ccc1 = breconoiseHB[i][jeta][jphi];
          int k2plot = jeta - 41;
          int kkk = k2plot;  //if(k2plot >0 ) kkk=k2plot+1; //-41 +41 !=0
          if (ccc1 != 0.) {
            GefzDIFreconoiseHB42D->Fill(kkk, jphi, ccc1);
            GefzDIFreconoiseHB42D0->Fill(kkk, jphi, 1.);
          }
        }
      }
    }
  }
  GefzDIFreconoiseHB42DF->Divide(GefzDIFreconoiseHB42D, GefzDIFreconoiseHB42D0, 1, 1, "B");  // average A
  gPad->SetGridy();
  gPad->SetGridx();  //      gPad->SetLogz();
  GefzDIFreconoiseHB42DF->SetXTitle("<DIF>_depth       #eta  \b");
  GefzDIFreconoiseHB42DF->SetYTitle("      #phi \b");
  GefzDIFreconoiseHB42DF->Draw("COLZ");

  //  c2x1->cd(2);
  //  TH1F *energyhitNoise_HB= (TH1F*)dir->FindObjectAny("h_energyhitNoise_HB");
  //  energyhitNoise_HB ->SetMarkerStyle(20);energyhitNoise_HB ->SetMarkerSize(0.4);energyhitNoise_HB ->GetYaxis()->SetLabelSize(0.04);energyhitNoise_HB ->SetXTitle("energyhitNoise_HB \b");energyhitNoise_HB ->SetMarkerColor(2);energyhitNoise_HB ->SetLineColor(0);gPad->SetGridy();gPad->SetGridx();energyhitNoise_HB ->Draw("Error");

  /////////////////
  c2x1->Update();
  c2x1->Print("DIFreconoiseGeneralD2PhiSymmetryHB.png");
  c2x1->Clear();
  // clean-up
  if (GefzDIFreconoiseHB42D)
    delete GefzDIFreconoiseHB42D;
  if (GefzDIFreconoiseHB42D0)
    delete GefzDIFreconoiseHB42D0;
  if (GefzDIFreconoiseHB42DF)
    delete GefzDIFreconoiseHB42DF;
  //====================================================================== 1D plot: DIF vs phi , averaged over depthfs & eta
  //======================================================================
  //cout<<"      1D plot: DIF vs phi , averaged over depthfs & eta *****" <<endl;
  c1x1->Clear();
  /////////////////
  c1x1->Divide(1, 1);
  c1x1->cd(1);
  TH1F* GefzDIFreconoiseHB41D = new TH1F("GefzDIFreconoiseHB41D", "", nphi, 0., 72.);
  TH1F* GefzDIFreconoiseHB41D0 = new TH1F("GefzDIFreconoiseHB41D0", "", nphi, 0., 72.);
  TH1F* GefzDIFreconoiseHB41DF = (TH1F*)GefzDIFreconoiseHB41D0->Clone("GefzDIFreconoiseHB41DF");
  for (int jphi = 0; jphi < nphi; jphi++) {
    for (int jeta = 0; jeta < neta; jeta++) {
      if ((jeta - 41 >= -16 && jeta - 41 <= -1) || (jeta - 41 >= 0 && jeta - 41 <= 15)) {
        for (int i = 0; i < ndepth; i++) {
          double ccc1 = breconoiseHB[i][jeta][jphi];
          if (ccc1 != 0.) {
            GefzDIFreconoiseHB41D->Fill(jphi, ccc1);
            GefzDIFreconoiseHB41D0->Fill(jphi, 1.);
          }
        }
      }
    }
  }
  GefzDIFreconoiseHB41DF->Divide(
      GefzDIFreconoiseHB41D, GefzDIFreconoiseHB41D0, 1, 1, "B");  // DIF averaged over depthfs & eta
  GefzDIFreconoiseHB41D0->Sumw2();
  //    for (int jphi=1;jphi<73;jphi++) {GefzDIFreconoiseHB41DF->SetBinError(jphi,0.01);}
  gPad->SetGridy();
  gPad->SetGridx();  //      gPad->SetLogz();
  GefzDIFreconoiseHB41DF->SetMarkerStyle(20);
  GefzDIFreconoiseHB41DF->SetMarkerSize(1.4);
  GefzDIFreconoiseHB41DF->GetZaxis()->SetLabelSize(0.08);
  GefzDIFreconoiseHB41DF->SetXTitle("#phi  \b");
  GefzDIFreconoiseHB41DF->SetYTitle("  <DIF> \b");
  GefzDIFreconoiseHB41DF->SetZTitle("<DIF>_PHI  - AllDepthfs \b");
  GefzDIFreconoiseHB41DF->SetMarkerColor(4);
  GefzDIFreconoiseHB41DF->SetLineColor(
      4);  //GefzDIFreconoiseHB41DF->SetMinimum(0.8);     //      GefzDIFreconoiseHB41DF->SetMaximum(1.000);
  GefzDIFreconoiseHB41DF->Draw("Error");
  /////////////////
  c1x1->Update();
  c1x1->Print("DIFreconoiseGeneralD1PhiSymmetryHB.png");
  c1x1->Clear();
  // clean-up
  if (GefzDIFreconoiseHB41D)
    delete GefzDIFreconoiseHB41D;
  if (GefzDIFreconoiseHB41D0)
    delete GefzDIFreconoiseHB41D0;
  if (GefzDIFreconoiseHB41DF)
    delete GefzDIFreconoiseHB41DF;
  //========================================================================================== 4
  //======================================================================
  //======================================================================1D plot: DIF vs phi , different eta,  depth=1
  //cout<<"      1D plot: DIF vs phi , different eta,  depth=1 *****" <<endl;
  c3x5->Clear();
  /////////////////
  c3x5->Divide(4, 4);
  c3x5->cd(1);
  int kcountHBpositivedirectionReconoiseDIF1 = 1;
  TH1F* h2CeffHBpositivedirectionReconoiseDIF1 = new TH1F("h2CeffHBpositivedirectionReconoiseDIF1", "", nphi, 0., 72.);
  for (int jeta = 0; jeta < njeta; jeta++) {
    // positivedirectionReconoiseDIF:
    if (jeta - 41 >= 0 && jeta - 41 <= 15) {
      //	     for (int i=0;i<ndepth;i++) {
      // depth=1
      for (int i = 0; i < 1; i++) {
        TH1F* HBpositivedirectionReconoiseDIF1 = (TH1F*)h2CeffHBpositivedirectionReconoiseDIF1->Clone("twod1");
        float ccctest = 0;  // to avoid empty massive elements
        for (int jphi = 0; jphi < nphi; jphi++) {
          double ccc1 = breconoiseHB[i][jeta][jphi];
          if (ccc1 != 0.) {
            HBpositivedirectionReconoiseDIF1->Fill(jphi, ccc1);
            ccctest = 1.;  //HBpositivedirectionReconoiseDIF1->SetBinError(i,0.01);
          }
        }  // for jphi
        if (ccctest > 0.) {
          //	  cout<<"444        kcountHBpositivedirectionReconoiseDIF1   =     "<<kcountHBpositivedirectionReconoiseDIF1  <<"   jeta-41=     "<< jeta-41 <<endl;
          c3x5->cd(kcountHBpositivedirectionReconoiseDIF1);
          HBpositivedirectionReconoiseDIF1->SetMarkerStyle(20);
          HBpositivedirectionReconoiseDIF1->SetMarkerSize(0.4);
          HBpositivedirectionReconoiseDIF1->GetYaxis()->SetLabelSize(0.04);
          HBpositivedirectionReconoiseDIF1->SetXTitle("HBpositivedirectionReconoiseDIF1 \b");
          HBpositivedirectionReconoiseDIF1->SetMarkerColor(2);
          HBpositivedirectionReconoiseDIF1->SetLineColor(0);
          gPad->SetGridy();
          gPad->SetGridx();
          //	   gPad->SetLogy();
          if (kcountHBpositivedirectionReconoiseDIF1 == 1)
            HBpositivedirectionReconoiseDIF1->SetXTitle("DIF for HB+ jeta =  0; depth = 1 \b");
          if (kcountHBpositivedirectionReconoiseDIF1 == 2)
            HBpositivedirectionReconoiseDIF1->SetXTitle("DIF for HB+ jeta =  1; depth = 1 \b");
          if (kcountHBpositivedirectionReconoiseDIF1 == 3)
            HBpositivedirectionReconoiseDIF1->SetXTitle("DIF for HB+ jeta =  2; depth = 1 \b");
          if (kcountHBpositivedirectionReconoiseDIF1 == 4)
            HBpositivedirectionReconoiseDIF1->SetXTitle("DIF for HB+ jeta =  3; depth = 1 \b");
          if (kcountHBpositivedirectionReconoiseDIF1 == 5)
            HBpositivedirectionReconoiseDIF1->SetXTitle("DIF for HB+ jeta =  4; depth = 1 \b");
          if (kcountHBpositivedirectionReconoiseDIF1 == 6)
            HBpositivedirectionReconoiseDIF1->SetXTitle("DIF for HB+ jeta =  5; depth = 1 \b");
          if (kcountHBpositivedirectionReconoiseDIF1 == 7)
            HBpositivedirectionReconoiseDIF1->SetXTitle("DIF for HB+ jeta =  6; depth = 1 \b");
          if (kcountHBpositivedirectionReconoiseDIF1 == 8)
            HBpositivedirectionReconoiseDIF1->SetXTitle("DIF for HB+ jeta =  7; depth = 1 \b");
          if (kcountHBpositivedirectionReconoiseDIF1 == 9)
            HBpositivedirectionReconoiseDIF1->SetXTitle("DIF for HB+ jeta =  8; depth = 1 \b");
          if (kcountHBpositivedirectionReconoiseDIF1 == 10)
            HBpositivedirectionReconoiseDIF1->SetXTitle("DIF for HB+ jeta =  9; depth = 1 \b");
          if (kcountHBpositivedirectionReconoiseDIF1 == 11)
            HBpositivedirectionReconoiseDIF1->SetXTitle("DIF for HB+ jeta = 10; depth = 1 \b");
          if (kcountHBpositivedirectionReconoiseDIF1 == 12)
            HBpositivedirectionReconoiseDIF1->SetXTitle("DIF for HB+ jeta = 11; depth = 1 \b");
          if (kcountHBpositivedirectionReconoiseDIF1 == 13)
            HBpositivedirectionReconoiseDIF1->SetXTitle("DIF for HB+ jeta = 12; depth = 1 \b");
          if (kcountHBpositivedirectionReconoiseDIF1 == 14)
            HBpositivedirectionReconoiseDIF1->SetXTitle("DIF for HB+ jeta = 13; depth = 1 \b");
          if (kcountHBpositivedirectionReconoiseDIF1 == 15)
            HBpositivedirectionReconoiseDIF1->SetXTitle("DIF for HB+ jeta = 14; depth = 1 \b");
          if (kcountHBpositivedirectionReconoiseDIF1 == 16)
            HBpositivedirectionReconoiseDIF1->SetXTitle("DIF for HB+ jeta = 15; depth = 1 \b");
          HBpositivedirectionReconoiseDIF1->Draw("Error");
          kcountHBpositivedirectionReconoiseDIF1++;
          if (kcountHBpositivedirectionReconoiseDIF1 > 16)
            break;  //
        }           //ccctest>0

      }  // for i
    }    //if(jeta-41 >= 0)
  }      //for jeta
  /////////////////
  c3x5->Update();
  c3x5->Print("DIFreconoisePositiveDirectionhistD1PhiSymmetryDepth1HB.png");
  c3x5->Clear();
  // clean-up
  if (h2CeffHBpositivedirectionReconoiseDIF1)
    delete h2CeffHBpositivedirectionReconoiseDIF1;

  //========================================================================================== 5
  //======================================================================
  //======================================================================1D plot: R vs phi , different eta,  depth=2
  //  cout<<"      1D plot: R vs phi , different eta,  depth=2 *****" <<endl;
  c3x5->Clear();
  /////////////////
  c3x5->Divide(4, 4);
  c3x5->cd(1);
  int kcountHBpositivedirectionReconoiseDIF2 = 1;
  TH1F* h2CeffHBpositivedirectionReconoiseDIF2 = new TH1F("h2CeffHBpositivedirectionReconoiseDIF2", "", nphi, 0., 72.);
  for (int jeta = 0; jeta < njeta; jeta++) {
    // positivedirectionReconoiseDIF:
    if (jeta - 41 >= 0 && jeta - 41 <= 15) {
      //	     for (int i=0;i<ndepth;i++) {
      // depth=2
      for (int i = 1; i < 2; i++) {
        TH1F* HBpositivedirectionReconoiseDIF2 = (TH1F*)h2CeffHBpositivedirectionReconoiseDIF2->Clone("twod1");
        float ccctest = 0;  // to avoid empty massive elements
        for (int jphi = 0; jphi < nphi; jphi++) {
          double ccc1 = breconoiseHB[i][jeta][jphi];
          if (ccc1 != 0.) {
            HBpositivedirectionReconoiseDIF2->Fill(jphi, ccc1);
            ccctest = 1.;  //HBpositivedirectionReconoiseDIF2->SetBinError(i,0.01);
          }
        }  // for jphi
        if (ccctest > 0.) {
          //cout<<"555        kcountHBpositivedirectionReconoiseDIF2   =     "<<kcountHBpositivedirectionReconoiseDIF2  <<"   jeta-41=     "<< jeta-41 <<endl;
          c3x5->cd(kcountHBpositivedirectionReconoiseDIF2);
          HBpositivedirectionReconoiseDIF2->SetMarkerStyle(20);
          HBpositivedirectionReconoiseDIF2->SetMarkerSize(0.4);
          HBpositivedirectionReconoiseDIF2->GetYaxis()->SetLabelSize(0.04);
          HBpositivedirectionReconoiseDIF2->SetXTitle("HBpositivedirectionReconoiseDIF2 \b");
          HBpositivedirectionReconoiseDIF2->SetMarkerColor(2);
          HBpositivedirectionReconoiseDIF2->SetLineColor(0);
          gPad->SetGridy();
          gPad->SetGridx();
          //	   gPad->SetLogy();
          if (kcountHBpositivedirectionReconoiseDIF2 == 1)
            HBpositivedirectionReconoiseDIF2->SetXTitle("DIF for HB+ jeta =  0; depth = 2 \b");
          if (kcountHBpositivedirectionReconoiseDIF2 == 2)
            HBpositivedirectionReconoiseDIF2->SetXTitle("DIF for HB+ jeta =  1; depth = 2 \b");
          if (kcountHBpositivedirectionReconoiseDIF2 == 3)
            HBpositivedirectionReconoiseDIF2->SetXTitle("DIF for HB+ jeta =  2; depth = 2 \b");
          if (kcountHBpositivedirectionReconoiseDIF2 == 4)
            HBpositivedirectionReconoiseDIF2->SetXTitle("DIF for HB+ jeta =  3; depth = 2 \b");
          if (kcountHBpositivedirectionReconoiseDIF2 == 5)
            HBpositivedirectionReconoiseDIF2->SetXTitle("DIF for HB+ jeta =  4; depth = 2 \b");
          if (kcountHBpositivedirectionReconoiseDIF2 == 6)
            HBpositivedirectionReconoiseDIF2->SetXTitle("DIF for HB+ jeta =  5; depth = 2 \b");
          if (kcountHBpositivedirectionReconoiseDIF2 == 7)
            HBpositivedirectionReconoiseDIF2->SetXTitle("DIF for HB+ jeta =  6; depth = 2 \b");
          if (kcountHBpositivedirectionReconoiseDIF2 == 8)
            HBpositivedirectionReconoiseDIF2->SetXTitle("DIF for HB+ jeta =  7; depth = 2 \b");
          if (kcountHBpositivedirectionReconoiseDIF2 == 9)
            HBpositivedirectionReconoiseDIF2->SetXTitle("DIF for HB+ jeta =  8; depth = 2 \b");
          if (kcountHBpositivedirectionReconoiseDIF2 == 10)
            HBpositivedirectionReconoiseDIF2->SetXTitle("DIF for HB+ jeta =  9; depth = 2 \b");
          if (kcountHBpositivedirectionReconoiseDIF2 == 11)
            HBpositivedirectionReconoiseDIF2->SetXTitle("DIF for HB+ jeta = 10; depth = 2 \b");
          if (kcountHBpositivedirectionReconoiseDIF2 == 12)
            HBpositivedirectionReconoiseDIF2->SetXTitle("DIF for HB+ jeta = 11; depth = 2 \b");
          if (kcountHBpositivedirectionReconoiseDIF2 == 13)
            HBpositivedirectionReconoiseDIF2->SetXTitle("DIF for HB+ jeta = 12; depth = 2 \b");
          if (kcountHBpositivedirectionReconoiseDIF2 == 14)
            HBpositivedirectionReconoiseDIF2->SetXTitle("DIF for HB+ jeta = 13; depth = 2 \b");
          if (kcountHBpositivedirectionReconoiseDIF2 == 15)
            HBpositivedirectionReconoiseDIF2->SetXTitle("DIF for HB+ jeta = 14; depth = 2 \b");
          if (kcountHBpositivedirectionReconoiseDIF2 == 16)
            HBpositivedirectionReconoiseDIF2->SetXTitle("DIF for HB+ jeta = 15; depth = 2 \b");
          HBpositivedirectionReconoiseDIF2->Draw("Error");
          kcountHBpositivedirectionReconoiseDIF2++;
          if (kcountHBpositivedirectionReconoiseDIF2 > 16)
            break;  // 4x6 = 24
        }           //ccctest>0

      }  // for i
    }    //if(jeta-41 >= 0)
  }      //for jeta
  /////////////////
  c3x5->Update();
  c3x5->Print("DIFreconoisePositiveDirectionhistD1PhiSymmetryDepth2HB.png");
  c3x5->Clear();
  // clean-up
  if (h2CeffHBpositivedirectionReconoiseDIF2)
    delete h2CeffHBpositivedirectionReconoiseDIF2;
  //========================================================================================== 6
  //======================================================================
  //======================================================================1D plot: R vs phi , different eta,  depth=3
  //cout<<"      1D plot: R vs phi , different eta,  depth=3 *****" <<endl;
  c3x5->Clear();
  /////////////////
  c3x5->Divide(4, 4);
  c3x5->cd(1);
  int kcountHBpositivedirectionReconoiseDIF3 = 1;
  TH1F* h2CeffHBpositivedirectionReconoiseDIF3 = new TH1F("h2CeffHBpositivedirectionReconoiseDIF3", "", nphi, 0., 72.);
  for (int jeta = 0; jeta < njeta; jeta++) {
    // positivedirectionReconoiseDIF:
    if (jeta - 41 >= 0 && jeta - 41 <= 15) {
      //	     for (int i=0;i<ndepth;i++) {
      // depth=3
      for (int i = 2; i < 3; i++) {
        TH1F* HBpositivedirectionReconoiseDIF3 = (TH1F*)h2CeffHBpositivedirectionReconoiseDIF3->Clone("twod1");
        float ccctest = 0;  // to avoid empty massive elements
        for (int jphi = 0; jphi < nphi; jphi++) {
          double ccc1 = breconoiseHB[i][jeta][jphi];
          if (ccc1 != 0.) {
            HBpositivedirectionReconoiseDIF3->Fill(jphi, ccc1);
            ccctest = 1.;  //HBpositivedirectionReconoiseDIF3->SetBinError(i,0.01);
          }
        }  // for jphi
        if (ccctest > 0.) {
          //cout<<"666        kcountHBpositivedirectionReconoiseDIF3   =     "<<kcountHBpositivedirectionReconoiseDIF3  <<"   jeta-41=     "<< jeta-41 <<endl;
          c3x5->cd(kcountHBpositivedirectionReconoiseDIF3);
          HBpositivedirectionReconoiseDIF3->SetMarkerStyle(20);
          HBpositivedirectionReconoiseDIF3->SetMarkerSize(0.4);
          HBpositivedirectionReconoiseDIF3->GetYaxis()->SetLabelSize(0.04);
          HBpositivedirectionReconoiseDIF3->SetXTitle("HBpositivedirectionReconoiseDIF3 \b");
          HBpositivedirectionReconoiseDIF3->SetMarkerColor(2);
          HBpositivedirectionReconoiseDIF3->SetLineColor(0);
          gPad->SetGridy();
          gPad->SetGridx();
          //	   gPad->SetLogy();
          if (kcountHBpositivedirectionReconoiseDIF3 == 1)
            HBpositivedirectionReconoiseDIF3->SetXTitle("DIF for HB+ jeta =  0; depth = 3 \b");
          if (kcountHBpositivedirectionReconoiseDIF3 == 2)
            HBpositivedirectionReconoiseDIF3->SetXTitle("DIF for HB+ jeta =  1; depth = 3 \b");
          if (kcountHBpositivedirectionReconoiseDIF3 == 3)
            HBpositivedirectionReconoiseDIF3->SetXTitle("DIF for HB+ jeta =  2; depth = 3 \b");
          if (kcountHBpositivedirectionReconoiseDIF3 == 4)
            HBpositivedirectionReconoiseDIF3->SetXTitle("DIF for HB+ jeta =  3; depth = 3 \b");
          if (kcountHBpositivedirectionReconoiseDIF3 == 5)
            HBpositivedirectionReconoiseDIF3->SetXTitle("DIF for HB+ jeta =  4; depth = 3 \b");
          if (kcountHBpositivedirectionReconoiseDIF3 == 6)
            HBpositivedirectionReconoiseDIF3->SetXTitle("DIF for HB+ jeta =  5; depth = 3 \b");
          if (kcountHBpositivedirectionReconoiseDIF3 == 7)
            HBpositivedirectionReconoiseDIF3->SetXTitle("DIF for HB+ jeta =  6; depth = 3 \b");
          if (kcountHBpositivedirectionReconoiseDIF3 == 8)
            HBpositivedirectionReconoiseDIF3->SetXTitle("DIF for HB+ jeta =  7; depth = 3 \b");
          if (kcountHBpositivedirectionReconoiseDIF3 == 9)
            HBpositivedirectionReconoiseDIF3->SetXTitle("DIF for HB+ jeta =  8; depth = 3 \b");
          if (kcountHBpositivedirectionReconoiseDIF3 == 10)
            HBpositivedirectionReconoiseDIF3->SetXTitle("DIF for HB+ jeta =  9; depth = 3 \b");
          if (kcountHBpositivedirectionReconoiseDIF3 == 11)
            HBpositivedirectionReconoiseDIF3->SetXTitle("DIF for HB+ jeta =  0; depth = 3 \b");
          if (kcountHBpositivedirectionReconoiseDIF3 == 12)
            HBpositivedirectionReconoiseDIF3->SetXTitle("DIF for HB+ jeta = 11; depth = 3 \b");
          if (kcountHBpositivedirectionReconoiseDIF3 == 13)
            HBpositivedirectionReconoiseDIF3->SetXTitle("DIF for HB+ jeta = 12; depth = 3 \b");
          if (kcountHBpositivedirectionReconoiseDIF3 == 14)
            HBpositivedirectionReconoiseDIF3->SetXTitle("DIF for HB+ jeta = 13; depth = 3 \b");
          if (kcountHBpositivedirectionReconoiseDIF3 == 15)
            HBpositivedirectionReconoiseDIF3->SetXTitle("DIF for HB+ jeta = 14; depth = 3 \b");
          if (kcountHBpositivedirectionReconoiseDIF3 == 16)
            HBpositivedirectionReconoiseDIF3->SetXTitle("DIF for HB+ jeta = 15; depth = 3 \b");
          HBpositivedirectionReconoiseDIF3->Draw("Error");
          kcountHBpositivedirectionReconoiseDIF3++;
          if (kcountHBpositivedirectionReconoiseDIF3 > 16)
            break;  // 4x6 = 24
        }           //ccctest>0

      }  // for i
    }    //if(jeta-41 >= 0)
  }      //for jeta
  /////////////////
  c3x5->Update();
  c3x5->Print("DIFreconoisePositiveDirectionhistD1PhiSymmetryDepth3HB.png");
  c3x5->Clear();
  // clean-up
  if (h2CeffHBpositivedirectionReconoiseDIF3)
    delete h2CeffHBpositivedirectionReconoiseDIF3;
  //========================================================================================== 7
  //======================================================================
  //======================================================================1D plot: R vs phi , different eta,  depth=4
  //cout<<"      1D plot: R vs phi , different eta,  depth=4 *****" <<endl;
  c3x5->Clear();
  /////////////////
  c3x5->Divide(4, 4);
  c3x5->cd(1);
  int kcountHBpositivedirectionReconoiseDIF4 = 1;
  TH1F* h2CeffHBpositivedirectionReconoiseDIF4 = new TH1F("h2CeffHBpositivedirectionReconoiseDIF4", "", nphi, 0., 72.);

  for (int jeta = 0; jeta < njeta; jeta++) {
    // positivedirectionReconoiseDIF:
    if (jeta - 41 >= 0 && jeta - 41 <= 15) {
      //	     for (int i=0;i<ndepth;i++) {
      // depth=4
      for (int i = 3; i < 4; i++) {
        TH1F* HBpositivedirectionReconoiseDIF4 = (TH1F*)h2CeffHBpositivedirectionReconoiseDIF4->Clone("twod1");

        float ccctest = 0;  // to avoid empty massive elements
        for (int jphi = 0; jphi < nphi; jphi++) {
          double ccc1 = breconoiseHB[i][jeta][jphi];
          if (ccc1 != 0.) {
            HBpositivedirectionReconoiseDIF4->Fill(jphi, ccc1);
            ccctest = 1.;  //HBpositivedirectionReconoiseDIF4->SetBinError(i,0.01);
          }
        }  // for jphi
        if (ccctest > 0.) {
          //cout<<"777        kcountHBpositivedirectionReconoiseDIF4   =     "<<kcountHBpositivedirectionReconoiseDIF4  <<"   jeta-41=     "<< jeta-41 <<endl;
          c3x5->cd(kcountHBpositivedirectionReconoiseDIF4);
          HBpositivedirectionReconoiseDIF4->SetMarkerStyle(20);
          HBpositivedirectionReconoiseDIF4->SetMarkerSize(0.4);
          HBpositivedirectionReconoiseDIF4->GetYaxis()->SetLabelSize(0.04);
          HBpositivedirectionReconoiseDIF4->SetXTitle("HBpositivedirectionReconoiseDIF4 \b");
          HBpositivedirectionReconoiseDIF4->SetMarkerColor(2);
          HBpositivedirectionReconoiseDIF4->SetLineColor(0);
          gPad->SetGridy();
          gPad->SetGridx();
          //	   gPad->SetLogy();
          if (kcountHBpositivedirectionReconoiseDIF4 == 1)
            HBpositivedirectionReconoiseDIF4->SetXTitle("DIF for HB+ jeta =  0; depth = 4 \b");
          if (kcountHBpositivedirectionReconoiseDIF4 == 2)
            HBpositivedirectionReconoiseDIF4->SetXTitle("DIF for HB+ jeta =  1; depth = 4 \b");
          if (kcountHBpositivedirectionReconoiseDIF4 == 3)
            HBpositivedirectionReconoiseDIF4->SetXTitle("DIF for HB+ jeta =  2; depth = 4 \b");
          if (kcountHBpositivedirectionReconoiseDIF4 == 4)
            HBpositivedirectionReconoiseDIF4->SetXTitle("DIF for HB+ jeta =  3; depth = 4 \b");
          if (kcountHBpositivedirectionReconoiseDIF4 == 5)
            HBpositivedirectionReconoiseDIF4->SetXTitle("DIF for HB+ jeta =  4; depth = 4 \b");
          if (kcountHBpositivedirectionReconoiseDIF4 == 6)
            HBpositivedirectionReconoiseDIF4->SetXTitle("DIF for HB+ jeta =  5; depth = 4 \b");
          if (kcountHBpositivedirectionReconoiseDIF4 == 7)
            HBpositivedirectionReconoiseDIF4->SetXTitle("DIF for HB+ jeta =  6; depth = 4 \b");
          if (kcountHBpositivedirectionReconoiseDIF4 == 8)
            HBpositivedirectionReconoiseDIF4->SetXTitle("DIF for HB+ jeta =  7; depth = 4 \b");
          if (kcountHBpositivedirectionReconoiseDIF4 == 9)
            HBpositivedirectionReconoiseDIF4->SetXTitle("DIF for HB+ jeta =  8; depth = 4 \b");
          if (kcountHBpositivedirectionReconoiseDIF4 == 10)
            HBpositivedirectionReconoiseDIF4->SetXTitle("DIF for HB+ jeta =  9; depth = 4 \b");
          if (kcountHBpositivedirectionReconoiseDIF4 == 11)
            HBpositivedirectionReconoiseDIF4->SetXTitle("DIF for HB+ jeta = 10; depth = 4 \b");
          if (kcountHBpositivedirectionReconoiseDIF4 == 12)
            HBpositivedirectionReconoiseDIF4->SetXTitle("DIF for HB+ jeta = 11; depth = 4 \b");
          if (kcountHBpositivedirectionReconoiseDIF4 == 13)
            HBpositivedirectionReconoiseDIF4->SetXTitle("DIF for HB+ jeta = 12; depth = 4 \b");
          if (kcountHBpositivedirectionReconoiseDIF4 == 14)
            HBpositivedirectionReconoiseDIF4->SetXTitle("DIF for HB+ jeta = 13; depth = 4 \b");
          if (kcountHBpositivedirectionReconoiseDIF4 == 15)
            HBpositivedirectionReconoiseDIF4->SetXTitle("DIF for HB+ jeta = 14; depth = 4 \b");
          if (kcountHBpositivedirectionReconoiseDIF4 == 16)
            HBpositivedirectionReconoiseDIF4->SetXTitle("DIF for HB+ jeta = 15; depth = 4 \b");
          HBpositivedirectionReconoiseDIF4->Draw("Error");
          kcountHBpositivedirectionReconoiseDIF4++;
          if (kcountHBpositivedirectionReconoiseDIF4 > 16)
            break;  // 4x6 = 24
        }           //ccctest>0

      }  // for i
    }    //if(jeta-41 >= 0)
  }      //for jeta
  /////////////////
  c3x5->Update();
  c3x5->Print("DIFreconoisePositiveDirectionhistD1PhiSymmetryDepth4HB.png");
  c3x5->Clear();
  // clean-up
  if (h2CeffHBpositivedirectionReconoiseDIF4)
    delete h2CeffHBpositivedirectionReconoiseDIF4;

  //========================================================================================== 1114
  //======================================================================
  //======================================================================1D plot: R vs phi , different eta,  depth=1
  //cout<<"      1D plot: R vs phi , different eta,  depth=1 *****" <<endl;
  c3x5->Clear();
  /////////////////
  c3x5->Divide(4, 4);
  c3x5->cd(1);
  int kcountHBnegativedirectionReconoiseDIF1 = 1;
  TH1F* h2CeffHBnegativedirectionReconoiseDIF1 = new TH1F("h2CeffHBnegativedirectionReconoiseDIF1", "", nphi, 0., 72.);
  for (int jeta = 0; jeta < njeta; jeta++) {
    // negativedirectionReconoiseDIF:
    if (jeta - 41 >= -16 && jeta - 41 <= -1) {
      //	     for (int i=0;i<ndepth;i++) {
      // depth=1
      for (int i = 0; i < 1; i++) {
        TH1F* HBnegativedirectionReconoiseDIF1 = (TH1F*)h2CeffHBnegativedirectionReconoiseDIF1->Clone("twod1");
        float ccctest = 0;  // to avoid empty massive elements
        for (int jphi = 0; jphi < nphi; jphi++) {
          double ccc1 = breconoiseHB[i][jeta][jphi];
          if (ccc1 != 0.) {
            HBnegativedirectionReconoiseDIF1->Fill(jphi, ccc1);
            ccctest = 1.;  //HBnegativedirectionReconoiseDIF1->SetBinError(i,0.01);
          }
        }  // for jphi
        if (ccctest > 0.) {
          //	  cout<<"444        kcountHBnegativedirectionReconoiseDIF1   =     "<<kcountHBnegativedirectionReconoiseDIF1  <<"   jeta-41=     "<< jeta-41 <<endl;
          c3x5->cd(kcountHBnegativedirectionReconoiseDIF1);
          HBnegativedirectionReconoiseDIF1->SetMarkerStyle(20);
          HBnegativedirectionReconoiseDIF1->SetMarkerSize(0.4);
          HBnegativedirectionReconoiseDIF1->GetYaxis()->SetLabelSize(0.04);
          HBnegativedirectionReconoiseDIF1->SetXTitle("HBnegativedirectionReconoiseDIF1 \b");
          HBnegativedirectionReconoiseDIF1->SetMarkerColor(2);
          HBnegativedirectionReconoiseDIF1->SetLineColor(0);
          gPad->SetGridy();
          gPad->SetGridx();
          //	   gPad->SetLogy();
          if (kcountHBnegativedirectionReconoiseDIF1 == 1)
            HBnegativedirectionReconoiseDIF1->SetXTitle("DIF for HB- jeta = -16; depth = 1 \b");
          if (kcountHBnegativedirectionReconoiseDIF1 == 2)
            HBnegativedirectionReconoiseDIF1->SetXTitle("DIF for HB- jeta = -15; depth = 1 \b");
          if (kcountHBnegativedirectionReconoiseDIF1 == 3)
            HBnegativedirectionReconoiseDIF1->SetXTitle("DIF for HB- jeta = -14; depth = 1 \b");
          if (kcountHBnegativedirectionReconoiseDIF1 == 4)
            HBnegativedirectionReconoiseDIF1->SetXTitle("DIF for HB- jeta = -13; depth = 1 \b");
          if (kcountHBnegativedirectionReconoiseDIF1 == 5)
            HBnegativedirectionReconoiseDIF1->SetXTitle("DIF for HB- jeta = -12; depth = 1 \b");
          if (kcountHBnegativedirectionReconoiseDIF1 == 6)
            HBnegativedirectionReconoiseDIF1->SetXTitle("DIF for HB- jeta = -11; depth = 1 \b");
          if (kcountHBnegativedirectionReconoiseDIF1 == 7)
            HBnegativedirectionReconoiseDIF1->SetXTitle("DIF for HB- jeta = -10; depth = 1 \b");
          if (kcountHBnegativedirectionReconoiseDIF1 == 8)
            HBnegativedirectionReconoiseDIF1->SetXTitle("DIF for HB- jeta =  -9; depth = 1 \b");
          if (kcountHBnegativedirectionReconoiseDIF1 == 9)
            HBnegativedirectionReconoiseDIF1->SetXTitle("DIF for HB- jeta =  -8; depth = 1 \b");
          if (kcountHBnegativedirectionReconoiseDIF1 == 10)
            HBnegativedirectionReconoiseDIF1->SetXTitle("DIF for HB- jeta =  -7; depth = 1 \b");
          if (kcountHBnegativedirectionReconoiseDIF1 == 11)
            HBnegativedirectionReconoiseDIF1->SetXTitle("DIF for HB- jeta =  -6; depth = 1 \b");
          if (kcountHBnegativedirectionReconoiseDIF1 == 12)
            HBnegativedirectionReconoiseDIF1->SetXTitle("DIF for HB- jeta =  -5; depth = 1 \b");
          if (kcountHBnegativedirectionReconoiseDIF1 == 13)
            HBnegativedirectionReconoiseDIF1->SetXTitle("DIF for HB- jeta =  -4; depth = 1 \b");
          if (kcountHBnegativedirectionReconoiseDIF1 == 14)
            HBnegativedirectionReconoiseDIF1->SetXTitle("DIF for HB- jeta =  -3; depth = 1 \b");
          if (kcountHBnegativedirectionReconoiseDIF1 == 15)
            HBnegativedirectionReconoiseDIF1->SetXTitle("DIF for HB- jeta =  -2; depth = 1 \b");
          if (kcountHBnegativedirectionReconoiseDIF1 == 16)
            HBnegativedirectionReconoiseDIF1->SetXTitle("DIF for HB- jeta =  -1; depth = 1 \b");
          HBnegativedirectionReconoiseDIF1->Draw("Error");
          kcountHBnegativedirectionReconoiseDIF1++;
          if (kcountHBnegativedirectionReconoiseDIF1 > 16)
            break;  //
        }           //ccctest>0

      }  // for i
    }    //if(jeta-41 < 0 )
  }      //for jeta
  /////////////////
  c3x5->Update();
  c3x5->Print("DIFreconoiseNegativeDirectionhistD1PhiSymmetryDepth1HB.png");
  c3x5->Clear();
  // clean-up
  if (h2CeffHBnegativedirectionReconoiseDIF1)
    delete h2CeffHBnegativedirectionReconoiseDIF1;

  //========================================================================================== 1115
  //======================================================================
  //======================================================================1D plot: R vs phi , different eta,  depth=2
  //  cout<<"      1D plot: R vs phi , different eta,  depth=2 *****" <<endl;
  c3x5->Clear();
  /////////////////
  c3x5->Divide(4, 4);
  c3x5->cd(1);
  int kcountHBnegativedirectionReconoiseDIF2 = 1;
  TH1F* h2CeffHBnegativedirectionReconoiseDIF2 = new TH1F("h2CeffHBnegativedirectionReconoiseDIF2", "", nphi, 0., 72.);
  for (int jeta = 0; jeta < njeta; jeta++) {
    // negativedirectionReconoiseDIF:
    if (jeta - 41 >= -16 && jeta - 41 <= -1) {
      //	     for (int i=0;i<ndepth;i++) {
      // depth=2
      for (int i = 1; i < 2; i++) {
        TH1F* HBnegativedirectionReconoiseDIF2 = (TH1F*)h2CeffHBnegativedirectionReconoiseDIF2->Clone("twod1");
        float ccctest = 0;  // to avoid empty massive elements
        for (int jphi = 0; jphi < nphi; jphi++) {
          double ccc1 = breconoiseHB[i][jeta][jphi];
          if (ccc1 != 0.) {
            HBnegativedirectionReconoiseDIF2->Fill(jphi, ccc1);
            ccctest = 1.;  //HBnegativedirectionReconoiseDIF2->SetBinError(i,0.01);
          }
        }  // for jphi
        if (ccctest > 0.) {
          //cout<<"555        kcountHBnegativedirectionReconoiseDIF2   =     "<<kcountHBnegativedirectionReconoiseDIF2  <<"   jeta-41=     "<< jeta-41 <<endl;
          c3x5->cd(kcountHBnegativedirectionReconoiseDIF2);
          HBnegativedirectionReconoiseDIF2->SetMarkerStyle(20);
          HBnegativedirectionReconoiseDIF2->SetMarkerSize(0.4);
          HBnegativedirectionReconoiseDIF2->GetYaxis()->SetLabelSize(0.04);
          HBnegativedirectionReconoiseDIF2->SetXTitle("HBnegativedirectionReconoiseDIF2 \b");
          HBnegativedirectionReconoiseDIF2->SetMarkerColor(2);
          HBnegativedirectionReconoiseDIF2->SetLineColor(0);
          gPad->SetGridy();
          gPad->SetGridx();
          //	   gPad->SetLogy();
          if (kcountHBnegativedirectionReconoiseDIF2 == 1)
            HBnegativedirectionReconoiseDIF2->SetXTitle("DIF for HB- jeta = -16; depth = 2 \b");
          if (kcountHBnegativedirectionReconoiseDIF2 == 2)
            HBnegativedirectionReconoiseDIF2->SetXTitle("DIF for HB- jeta = -15; depth = 2 \b");
          if (kcountHBnegativedirectionReconoiseDIF2 == 3)
            HBnegativedirectionReconoiseDIF2->SetXTitle("DIF for HB- jeta = -14; depth = 2 \b");
          if (kcountHBnegativedirectionReconoiseDIF2 == 4)
            HBnegativedirectionReconoiseDIF2->SetXTitle("DIF for HB- jeta = -13; depth = 2 \b");
          if (kcountHBnegativedirectionReconoiseDIF2 == 5)
            HBnegativedirectionReconoiseDIF2->SetXTitle("DIF for HB- jeta = -12; depth = 2 \b");
          if (kcountHBnegativedirectionReconoiseDIF2 == 6)
            HBnegativedirectionReconoiseDIF2->SetXTitle("DIF for HB- jeta = -11; depth = 2 \b");
          if (kcountHBnegativedirectionReconoiseDIF2 == 7)
            HBnegativedirectionReconoiseDIF2->SetXTitle("DIF for HB- jeta = -10; depth = 2 \b");
          if (kcountHBnegativedirectionReconoiseDIF2 == 8)
            HBnegativedirectionReconoiseDIF2->SetXTitle("DIF for HB- jeta =  -9; depth = 2 \b");
          if (kcountHBnegativedirectionReconoiseDIF2 == 9)
            HBnegativedirectionReconoiseDIF2->SetXTitle("DIF for HB- jeta =  -8; depth = 2 \b");
          if (kcountHBnegativedirectionReconoiseDIF2 == 10)
            HBnegativedirectionReconoiseDIF2->SetXTitle("DIF for HB- jeta =  -7; depth = 2 \b");
          if (kcountHBnegativedirectionReconoiseDIF2 == 11)
            HBnegativedirectionReconoiseDIF2->SetXTitle("DIF for HB- jeta =  -6; depth = 2 \b");
          if (kcountHBnegativedirectionReconoiseDIF2 == 12)
            HBnegativedirectionReconoiseDIF2->SetXTitle("DIF for HB- jeta =  -5; depth = 2 \b");
          if (kcountHBnegativedirectionReconoiseDIF2 == 13)
            HBnegativedirectionReconoiseDIF2->SetXTitle("DIF for HB- jeta =  -4; depth = 2 \b");
          if (kcountHBnegativedirectionReconoiseDIF2 == 14)
            HBnegativedirectionReconoiseDIF2->SetXTitle("DIF for HB- jeta =  -3; depth = 2 \b");
          if (kcountHBnegativedirectionReconoiseDIF2 == 15)
            HBnegativedirectionReconoiseDIF2->SetXTitle("DIF for HB- jeta =  -2; depth = 2 \b");
          if (kcountHBnegativedirectionReconoiseDIF2 == 16)
            HBnegativedirectionReconoiseDIF2->SetXTitle("DIF for HB- jeta =  -1; depth = 2 \b");
          HBnegativedirectionReconoiseDIF2->Draw("Error");
          kcountHBnegativedirectionReconoiseDIF2++;
          if (kcountHBnegativedirectionReconoiseDIF2 > 16)
            break;  // 4x6 = 24
        }           //ccctest>0

      }  // for i
    }    //if(jeta-41 < 0 )
  }      //for jeta
  /////////////////
  c3x5->Update();
  c3x5->Print("DIFreconoiseNegativeDirectionhistD1PhiSymmetryDepth2HB.png");
  c3x5->Clear();
  // clean-up
  if (h2CeffHBnegativedirectionReconoiseDIF2)
    delete h2CeffHBnegativedirectionReconoiseDIF2;
  //========================================================================================== 1116
  //======================================================================
  //======================================================================1D plot: R vs phi , different eta,  depth=3
  //cout<<"      1D plot: R vs phi , different eta,  depth=3 *****" <<endl;
  c3x5->Clear();
  /////////////////
  c3x5->Divide(4, 4);
  c3x5->cd(1);
  int kcountHBnegativedirectionReconoiseDIF3 = 1;
  TH1F* h2CeffHBnegativedirectionReconoiseDIF3 = new TH1F("h2CeffHBnegativedirectionReconoiseDIF3", "", nphi, 0., 72.);
  for (int jeta = 0; jeta < njeta; jeta++) {
    // negativedirectionReconoiseDIF:
    if (jeta - 41 >= -16 && jeta - 41 <= -1) {
      //	     for (int i=0;i<ndepth;i++) {
      // depth=3
      for (int i = 2; i < 3; i++) {
        TH1F* HBnegativedirectionReconoiseDIF3 = (TH1F*)h2CeffHBnegativedirectionReconoiseDIF3->Clone("twod1");
        float ccctest = 0;  // to avoid empty massive elements
        for (int jphi = 0; jphi < nphi; jphi++) {
          double ccc1 = breconoiseHB[i][jeta][jphi];
          if (ccc1 != 0.) {
            HBnegativedirectionReconoiseDIF3->Fill(jphi, ccc1);
            ccctest = 1.;  //HBnegativedirectionReconoiseDIF3->SetBinError(i,0.01);
          }
        }  // for jphi
        if (ccctest > 0.) {
          //cout<<"666        kcountHBnegativedirectionReconoiseDIF3   =     "<<kcountHBnegativedirectionReconoiseDIF3  <<"   jeta-41=     "<< jeta-41 <<endl;
          c3x5->cd(kcountHBnegativedirectionReconoiseDIF3);
          HBnegativedirectionReconoiseDIF3->SetMarkerStyle(20);
          HBnegativedirectionReconoiseDIF3->SetMarkerSize(0.4);
          HBnegativedirectionReconoiseDIF3->GetYaxis()->SetLabelSize(0.04);
          HBnegativedirectionReconoiseDIF3->SetXTitle("HBnegativedirectionReconoiseDIF3 \b");
          HBnegativedirectionReconoiseDIF3->SetMarkerColor(2);
          HBnegativedirectionReconoiseDIF3->SetLineColor(0);
          gPad->SetGridy();
          gPad->SetGridx();
          //	   gPad->SetLogy();
          if (kcountHBnegativedirectionReconoiseDIF3 == 1)
            HBnegativedirectionReconoiseDIF3->SetXTitle("DIF for HB- jeta = -16; depth = 3 \b");
          if (kcountHBnegativedirectionReconoiseDIF3 == 2)
            HBnegativedirectionReconoiseDIF3->SetXTitle("DIF for HB- jeta = -15; depth = 3 \b");
          if (kcountHBnegativedirectionReconoiseDIF3 == 3)
            HBnegativedirectionReconoiseDIF3->SetXTitle("DIF for HB- jeta = -14; depth = 3 \b");
          if (kcountHBnegativedirectionReconoiseDIF3 == 4)
            HBnegativedirectionReconoiseDIF3->SetXTitle("DIF for HB- jeta = -13; depth = 3 \b");
          if (kcountHBnegativedirectionReconoiseDIF3 == 5)
            HBnegativedirectionReconoiseDIF3->SetXTitle("DIF for HB- jeta = -12; depth = 3 \b");
          if (kcountHBnegativedirectionReconoiseDIF3 == 6)
            HBnegativedirectionReconoiseDIF3->SetXTitle("DIF for HB- jeta = -11; depth = 3 \b");
          if (kcountHBnegativedirectionReconoiseDIF3 == 7)
            HBnegativedirectionReconoiseDIF3->SetXTitle("DIF for HB- jeta = -10; depth = 3 \b");
          if (kcountHBnegativedirectionReconoiseDIF3 == 8)
            HBnegativedirectionReconoiseDIF3->SetXTitle("DIF for HB- jeta =  -9; depth = 3 \b");
          if (kcountHBnegativedirectionReconoiseDIF3 == 9)
            HBnegativedirectionReconoiseDIF3->SetXTitle("DIF for HB- jeta =  -8; depth = 3 \b");
          if (kcountHBnegativedirectionReconoiseDIF3 == 10)
            HBnegativedirectionReconoiseDIF3->SetXTitle("DIF for HB- jeta =  -7; depth = 3 \b");
          if (kcountHBnegativedirectionReconoiseDIF3 == 11)
            HBnegativedirectionReconoiseDIF3->SetXTitle("DIF for HB- jeta =  -6; depth = 3 \b");
          if (kcountHBnegativedirectionReconoiseDIF3 == 12)
            HBnegativedirectionReconoiseDIF3->SetXTitle("DIF for HB- jeta =  -5; depth = 3 \b");
          if (kcountHBnegativedirectionReconoiseDIF3 == 13)
            HBnegativedirectionReconoiseDIF3->SetXTitle("DIF for HB- jeta =  -4; depth = 3 \b");
          if (kcountHBnegativedirectionReconoiseDIF3 == 14)
            HBnegativedirectionReconoiseDIF3->SetXTitle("DIF for HB- jeta =  -3; depth = 3 \b");
          if (kcountHBnegativedirectionReconoiseDIF3 == 15)
            HBnegativedirectionReconoiseDIF3->SetXTitle("DIF for HB- jeta =  -2; depth = 3 \b");
          if (kcountHBnegativedirectionReconoiseDIF3 == 16)
            HBnegativedirectionReconoiseDIF3->SetXTitle("DIF for HB- jeta =  -1; depth = 3 \b");

          HBnegativedirectionReconoiseDIF3->Draw("Error");
          kcountHBnegativedirectionReconoiseDIF3++;
          if (kcountHBnegativedirectionReconoiseDIF3 > 16)
            break;  // 4x6 = 24
        }           //ccctest>0

      }  // for i
    }    //if(jeta-41 < 0 )
  }      //for jeta
  /////////////////
  c3x5->Update();
  c3x5->Print("DIFreconoiseNegativeDirectionhistD1PhiSymmetryDepth3HB.png");
  c3x5->Clear();
  // clean-up
  if (h2CeffHBnegativedirectionReconoiseDIF3)
    delete h2CeffHBnegativedirectionReconoiseDIF3;
  //========================================================================================== 1117
  //======================================================================
  //======================================================================1D plot: R vs phi , different eta,  depth=4
  //cout<<"      1D plot: R vs phi , different eta,  depth=4 *****" <<endl;
  c3x5->Clear();
  /////////////////
  c3x5->Divide(4, 4);
  c3x5->cd(1);
  int kcountHBnegativedirectionReconoiseDIF4 = 1;
  TH1F* h2CeffHBnegativedirectionReconoiseDIF4 = new TH1F("h2CeffHBnegativedirectionReconoiseDIF4", "", nphi, 0., 72.);

  for (int jeta = 0; jeta < njeta; jeta++) {
    // negativedirectionReconoiseDIF:
    if (jeta - 41 >= -16 && jeta - 41 <= -1) {
      //	     for (int i=0;i<ndepth;i++) {
      // depth=4
      for (int i = 3; i < 4; i++) {
        TH1F* HBnegativedirectionReconoiseDIF4 = (TH1F*)h2CeffHBnegativedirectionReconoiseDIF4->Clone("twod1");

        float ccctest = 0;  // to avoid empty massive elements
        for (int jphi = 0; jphi < nphi; jphi++) {
          double ccc1 = breconoiseHB[i][jeta][jphi];
          if (ccc1 != 0.) {
            HBnegativedirectionReconoiseDIF4->Fill(jphi, ccc1);
            ccctest = 1.;  //HBnegativedirectionReconoiseDIF4->SetBinError(i,0.01);
          }
        }  // for jphi
        if (ccctest > 0.) {
          //cout<<"777        kcountHBnegativedirectionReconoiseDIF4   =     "<<kcountHBnegativedirectionReconoiseDIF4  <<"   jeta-41=     "<< jeta-41 <<endl;
          c3x5->cd(kcountHBnegativedirectionReconoiseDIF4);
          HBnegativedirectionReconoiseDIF4->SetMarkerStyle(20);
          HBnegativedirectionReconoiseDIF4->SetMarkerSize(0.4);
          HBnegativedirectionReconoiseDIF4->GetYaxis()->SetLabelSize(0.04);
          HBnegativedirectionReconoiseDIF4->SetXTitle("HBnegativedirectionReconoiseDIF4 \b");
          HBnegativedirectionReconoiseDIF4->SetMarkerColor(2);
          HBnegativedirectionReconoiseDIF4->SetLineColor(0);
          gPad->SetGridy();
          gPad->SetGridx();
          //	   gPad->SetLogy();
          if (kcountHBnegativedirectionReconoiseDIF4 == 1)
            HBnegativedirectionReconoiseDIF4->SetXTitle("DIF for HB- jeta = -16; depth = 4 \b");
          if (kcountHBnegativedirectionReconoiseDIF4 == 2)
            HBnegativedirectionReconoiseDIF4->SetXTitle("DIF for HB- jeta = -15; depth = 4 \b");
          if (kcountHBnegativedirectionReconoiseDIF4 == 3)
            HBnegativedirectionReconoiseDIF4->SetXTitle("DIF for HB- jeta = -14; depth = 4 \b");
          if (kcountHBnegativedirectionReconoiseDIF4 == 4)
            HBnegativedirectionReconoiseDIF4->SetXTitle("DIF for HB- jeta = -13; depth = 4 \b");
          if (kcountHBnegativedirectionReconoiseDIF4 == 5)
            HBnegativedirectionReconoiseDIF4->SetXTitle("DIF for HB- jeta = -12; depth = 4 \b");
          if (kcountHBnegativedirectionReconoiseDIF4 == 6)
            HBnegativedirectionReconoiseDIF4->SetXTitle("DIF for HB- jeta = -11; depth = 4 \b");
          if (kcountHBnegativedirectionReconoiseDIF4 == 7)
            HBnegativedirectionReconoiseDIF4->SetXTitle("DIF for HB- jeta = -10; depth = 4 \b");
          if (kcountHBnegativedirectionReconoiseDIF4 == 8)
            HBnegativedirectionReconoiseDIF4->SetXTitle("DIF for HB- jeta =  -9; depth = 4 \b");
          if (kcountHBnegativedirectionReconoiseDIF4 == 9)
            HBnegativedirectionReconoiseDIF4->SetXTitle("DIF for HB- jeta =  -8; depth = 4 \b");
          if (kcountHBnegativedirectionReconoiseDIF4 == 10)
            HBnegativedirectionReconoiseDIF4->SetXTitle("DIF for HB- jeta =  -7; depth = 4 \b");
          if (kcountHBnegativedirectionReconoiseDIF4 == 11)
            HBnegativedirectionReconoiseDIF4->SetXTitle("DIF for HB- jeta =  -6; depth = 4 \b");
          if (kcountHBnegativedirectionReconoiseDIF4 == 12)
            HBnegativedirectionReconoiseDIF4->SetXTitle("DIF for HB- jeta =  -5; depth = 4 \b");
          if (kcountHBnegativedirectionReconoiseDIF4 == 13)
            HBnegativedirectionReconoiseDIF4->SetXTitle("DIF for HB- jeta =  -4; depth = 4 \b");
          if (kcountHBnegativedirectionReconoiseDIF4 == 14)
            HBnegativedirectionReconoiseDIF4->SetXTitle("DIF for HB- jeta =  -3; depth = 4 \b");
          if (kcountHBnegativedirectionReconoiseDIF4 == 15)
            HBnegativedirectionReconoiseDIF4->SetXTitle("DIF for HB- jeta =  -2; depth = 4 \b");
          if (kcountHBnegativedirectionReconoiseDIF4 == 16)
            HBnegativedirectionReconoiseDIF4->SetXTitle("DIF for HB- jeta =  -1; depth = 4 \b");
          HBnegativedirectionReconoiseDIF4->Draw("Error");
          kcountHBnegativedirectionReconoiseDIF4++;
          if (kcountHBnegativedirectionReconoiseDIF4 > 16)
            break;  // 4x6 = 24
        }           //ccctest>0

      }  // for i
    }    //if(jeta-41 < 0 )
  }      //for jeta
  /////////////////
  c3x5->Update();
  c3x5->Print("DIFreconoiseNegativeDirectionhistD1PhiSymmetryDepth4HB.png");
  c3x5->Clear();
  // clean-up
  if (h2CeffHBnegativedirectionReconoiseDIF4)
    delete h2CeffHBnegativedirectionReconoiseDIF4;

  //======================================================================================================================
  //======================================================================================================================
  //======================================================================================================================
  //                            DDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDD:

  //cout<<"    Start Vaiance: preparation  *****" <<endl;
  TH2F* reconoiseVariance1HB1 = (TH2F*)dir->FindObjectAny("h_recNoiseEnergy2_HB1");
  TH2F* reconoiseVariance0HB1 = (TH2F*)dir->FindObjectAny("h_recNoiseEnergy0_HB1");
  TH2F* reconoiseVarianceHB1 = (TH2F*)reconoiseVariance1HB1->Clone("reconoiseVarianceHB1");
  reconoiseVarianceHB1->Divide(reconoiseVariance1HB1, reconoiseVariance0HB1, 1, 1, "B");
  TH2F* reconoiseVariance1HB2 = (TH2F*)dir->FindObjectAny("h_recNoiseEnergy2_HB2");
  TH2F* reconoiseVariance0HB2 = (TH2F*)dir->FindObjectAny("h_recNoiseEnergy0_HB2");
  TH2F* reconoiseVarianceHB2 = (TH2F*)reconoiseVariance1HB2->Clone("reconoiseVarianceHB2");
  reconoiseVarianceHB2->Divide(reconoiseVariance1HB2, reconoiseVariance0HB2, 1, 1, "B");
  TH2F* reconoiseVariance1HB3 = (TH2F*)dir->FindObjectAny("h_recNoiseEnergy2_HB3");
  TH2F* reconoiseVariance0HB3 = (TH2F*)dir->FindObjectAny("h_recNoiseEnergy0_HB3");
  TH2F* reconoiseVarianceHB3 = (TH2F*)reconoiseVariance1HB3->Clone("reconoiseVarianceHB3");
  reconoiseVarianceHB3->Divide(reconoiseVariance1HB3, reconoiseVariance0HB3, 1, 1, "B");
  TH2F* reconoiseVariance1HB4 = (TH2F*)dir->FindObjectAny("h_recNoiseEnergy2_HB4");
  TH2F* reconoiseVariance0HB4 = (TH2F*)dir->FindObjectAny("h_recNoiseEnergy0_HB4");
  TH2F* reconoiseVarianceHB4 = (TH2F*)reconoiseVariance1HB4->Clone("reconoiseVarianceHB4");
  reconoiseVarianceHB4->Divide(reconoiseVariance1HB4, reconoiseVariance0HB4, 1, 1, "B");
  //cout<<"      Vaiance: preparation DONE *****" <<endl;
  //====================================================================== put Vaiance=Dispersia = Sig**2=<R**2> - (<R>)**2 into massive reconoisevarianceHB
  //                                                                                           = sum(R*R)/N - (sum(R)/N)**2
  for (int jeta = 0; jeta < njeta; jeta++) {
    if ((jeta - 41 >= -16 && jeta - 41 <= -1) || (jeta - 41 >= 0 && jeta - 41 <= 15)) {
      //preparation for PHI normalization:
      double sumreconoiseHB0 = 0;
      int nsumreconoiseHB0 = 0;
      double sumreconoiseHB1 = 0;
      int nsumreconoiseHB1 = 0;
      double sumreconoiseHB2 = 0;
      int nsumreconoiseHB2 = 0;
      double sumreconoiseHB3 = 0;
      int nsumreconoiseHB3 = 0;
      for (int jphi = 0; jphi < njphi; jphi++) {
        reconoisevarianceHB[0][jeta][jphi] = reconoiseVarianceHB1->GetBinContent(jeta + 1, jphi + 1);
        reconoisevarianceHB[1][jeta][jphi] = reconoiseVarianceHB2->GetBinContent(jeta + 1, jphi + 1);
        reconoisevarianceHB[2][jeta][jphi] = reconoiseVarianceHB3->GetBinContent(jeta + 1, jphi + 1);
        reconoisevarianceHB[3][jeta][jphi] = reconoiseVarianceHB4->GetBinContent(jeta + 1, jphi + 1);
        if (reconoisevarianceHB[0][jeta][jphi] != 0.) {
          sumreconoiseHB0 += reconoisevarianceHB[0][jeta][jphi];
          ++nsumreconoiseHB0;
        }
        if (reconoisevarianceHB[1][jeta][jphi] != 0.) {
          sumreconoiseHB1 += reconoisevarianceHB[1][jeta][jphi];
          ++nsumreconoiseHB1;
        }
        if (reconoisevarianceHB[2][jeta][jphi] != 0.) {
          sumreconoiseHB2 += reconoisevarianceHB[2][jeta][jphi];
          ++nsumreconoiseHB2;
        }
        if (reconoisevarianceHB[3][jeta][jphi] != 0.) {
          sumreconoiseHB3 += reconoisevarianceHB[3][jeta][jphi];
          ++nsumreconoiseHB3;
        }
      }  // phi
      // PHI normalization :
      for (int jphi = 0; jphi < njphi; jphi++) {
        if (sumreconoiseHB0 != 0.)
          reconoisevarianceHB[0][jeta][jphi] /= (sumreconoiseHB0 / nsumreconoiseHB0);
        if (sumreconoiseHB1 != 0.)
          reconoisevarianceHB[1][jeta][jphi] /= (sumreconoiseHB1 / nsumreconoiseHB1);
        if (sumreconoiseHB2 != 0.)
          reconoisevarianceHB[2][jeta][jphi] /= (sumreconoiseHB2 / nsumreconoiseHB2);
        if (sumreconoiseHB3 != 0.)
          reconoisevarianceHB[3][jeta][jphi] /= (sumreconoiseHB3 / nsumreconoiseHB3);
      }  // phi
      //       reconoisevarianceHB (D)           = sum(R*R)/N - (sum(R)/N)**2
      for (int jphi = 0; jphi < njphi; jphi++) {
        //	   cout<<"12 12 12   jeta=     "<< jeta <<"   jphi   =     "<<jphi  <<endl;
        reconoisevarianceHB[0][jeta][jphi] -= areconoiseHB[0][jeta][jphi] * areconoiseHB[0][jeta][jphi];
        reconoisevarianceHB[0][jeta][jphi] = fabs(reconoisevarianceHB[0][jeta][jphi]);
        reconoisevarianceHB[1][jeta][jphi] -= areconoiseHB[1][jeta][jphi] * areconoiseHB[1][jeta][jphi];
        reconoisevarianceHB[1][jeta][jphi] = fabs(reconoisevarianceHB[1][jeta][jphi]);
        reconoisevarianceHB[2][jeta][jphi] -= areconoiseHB[2][jeta][jphi] * areconoiseHB[2][jeta][jphi];
        reconoisevarianceHB[2][jeta][jphi] = fabs(reconoisevarianceHB[2][jeta][jphi]);
        reconoisevarianceHB[3][jeta][jphi] -= areconoiseHB[3][jeta][jphi] * areconoiseHB[3][jeta][jphi];
        reconoisevarianceHB[3][jeta][jphi] = fabs(reconoisevarianceHB[3][jeta][jphi]);
      }
    }
  }
  //cout<<"      Vaiance: DONE*****" <<endl;
  //------------------------  2D-eta/phi-plot: D, averaged over depthfs
  //======================================================================
  //======================================================================
  //cout<<"      R2D-eta/phi-plot: D, averaged over depthfs *****" <<endl;
  c1x1->Clear();
  /////////////////
  c1x0->Divide(1, 1);
  c1x0->cd(1);
  TH2F* DefzDreconoiseHB42D = new TH2F("DefzDreconoiseHB42D", "", neta, -41., 41., nphi, 0., 72.);
  TH2F* DefzDreconoiseHB42D0 = new TH2F("DefzDreconoiseHB42D0", "", neta, -41., 41., nphi, 0., 72.);
  TH2F* DefzDreconoiseHB42DF = (TH2F*)DefzDreconoiseHB42D0->Clone("DefzDreconoiseHB42DF");
  for (int i = 0; i < ndepth; i++) {
    for (int jeta = 0; jeta < neta; jeta++) {
      if ((jeta - 41 >= -16 && jeta - 41 <= -1) || (jeta - 41 >= 0 && jeta - 41 <= 15)) {
        for (int jphi = 0; jphi < nphi; jphi++) {
          double ccc1 = reconoisevarianceHB[i][jeta][jphi];
          int k2plot = jeta - 41;
          int kkk = k2plot;  //if(k2plot >0   kkk=k2plot+1; //-41 +41 !=0
          if (areconoiseHB[i][jeta][jphi] > 0.) {
            DefzDreconoiseHB42D->Fill(kkk, jphi, ccc1);
            DefzDreconoiseHB42D0->Fill(kkk, jphi, 1.);
          }
        }
      }
    }
  }
  DefzDreconoiseHB42DF->Divide(DefzDreconoiseHB42D, DefzDreconoiseHB42D0, 1, 1, "B");  // average A
  //    DefzDreconoiseHB1->Sumw2();
  gPad->SetGridy();
  gPad->SetGridx();  //      gPad->SetLogz();
  DefzDreconoiseHB42DF->SetMarkerStyle(20);
  DefzDreconoiseHB42DF->SetMarkerSize(0.4);
  DefzDreconoiseHB42DF->GetZaxis()->SetLabelSize(0.08);
  DefzDreconoiseHB42DF->SetXTitle("<D>_depth       #eta  \b");
  DefzDreconoiseHB42DF->SetYTitle("      #phi \b");
  DefzDreconoiseHB42DF->SetZTitle("<D>_depth \b");
  DefzDreconoiseHB42DF->SetMarkerColor(2);
  DefzDreconoiseHB42DF->SetLineColor(
      0);  //      DefzDreconoiseHB42DF->SetMaximum(1.000);  //      DefzDreconoiseHB42DF->SetMinimum(1.0);
  DefzDreconoiseHB42DF->Draw("COLZ");
  /////////////////
  c1x0->Update();
  c1x0->Print("DreconoiseGeneralD2PhiSymmetryHB.png");
  c1x0->Clear();
  // clean-up
  if (DefzDreconoiseHB42D)
    delete DefzDreconoiseHB42D;
  if (DefzDreconoiseHB42D0)
    delete DefzDreconoiseHB42D0;
  if (DefzDreconoiseHB42DF)
    delete DefzDreconoiseHB42DF;
  //====================================================================== 1D plot: D vs phi , averaged over depthfs & eta
  //======================================================================
  //cout<<"      1D plot: D vs phi , averaged over depthfs & eta *****" <<endl;
  c1x1->Clear();
  /////////////////
  c1x1->Divide(1, 1);
  c1x1->cd(1);
  TH1F* DefzDreconoiseHB41D = new TH1F("DefzDreconoiseHB41D", "", nphi, 0., 72.);
  TH1F* DefzDreconoiseHB41D0 = new TH1F("DefzDreconoiseHB41D0", "", nphi, 0., 72.);
  TH1F* DefzDreconoiseHB41DF = (TH1F*)DefzDreconoiseHB41D0->Clone("DefzDreconoiseHB41DF");

  for (int jphi = 0; jphi < nphi; jphi++) {
    for (int jeta = 0; jeta < neta; jeta++) {
      if ((jeta - 41 >= -16 && jeta - 41 <= -1) || (jeta - 41 >= 0 && jeta - 41 <= 15)) {
        for (int i = 0; i < ndepth; i++) {
          double ccc1 = reconoisevarianceHB[i][jeta][jphi];
          if (areconoiseHB[i][jeta][jphi] > 0.) {
            DefzDreconoiseHB41D->Fill(jphi, ccc1);
            DefzDreconoiseHB41D0->Fill(jphi, 1.);
          }
        }
      }
    }
  }
  //     DefzDreconoiseHB41D->Sumw2();DefzDreconoiseHB41D0->Sumw2();

  DefzDreconoiseHB41DF->Divide(DefzDreconoiseHB41D, DefzDreconoiseHB41D0, 1, 1, "B");  // R averaged over depthfs & eta
  DefzDreconoiseHB41D0->Sumw2();
  //    for (int jphi=1;jphi<73;jphi++) {DefzDreconoiseHB41DF->SetBinError(jphi,0.01);}
  gPad->SetGridy();
  gPad->SetGridx();  //      gPad->SetLogz();
  DefzDreconoiseHB41DF->SetMarkerStyle(20);
  DefzDreconoiseHB41DF->SetMarkerSize(1.4);
  DefzDreconoiseHB41DF->GetZaxis()->SetLabelSize(0.08);
  DefzDreconoiseHB41DF->SetXTitle("#phi  \b");
  DefzDreconoiseHB41DF->SetYTitle("  <D> \b");
  DefzDreconoiseHB41DF->SetZTitle("<D>_PHI  - AllDepthfs \b");
  DefzDreconoiseHB41DF->SetMarkerColor(4);
  DefzDreconoiseHB41DF->SetLineColor(
      4);  //DefzDreconoiseHB41DF->SetMinimum(0.8);     DefzDreconoiseHB41DF->SetMinimum(-0.015);
  DefzDreconoiseHB41DF->Draw("Error");
  /////////////////
  c1x1->Update();
  c1x1->Print("DreconoiseGeneralD1PhiSymmetryHB.png");
  c1x1->Clear();
  // clean-up
  if (DefzDreconoiseHB41D)
    delete DefzDreconoiseHB41D;
  if (DefzDreconoiseHB41D0)
    delete DefzDreconoiseHB41D0;
  if (DefzDreconoiseHB41DF)
    delete DefzDreconoiseHB41DF;

  //========================================================================================== 14
  //======================================================================
  //======================================================================1D plot: D vs phi , different eta,  depth=1
  //cout<<"      1D plot: D vs phi , different eta,  depth=1 *****" <<endl;
  c3x5->Clear();
  /////////////////
  c3x5->Divide(4, 4);
  c3x5->cd(1);
  int kcountHBpositivedirectionReconoiseD1 = 1;
  TH1F* h2CeffHBpositivedirectionReconoiseD1 = new TH1F("h2CeffHBpositivedirectionReconoiseD1", "", nphi, 0., 72.);

  for (int jeta = 0; jeta < njeta; jeta++) {
    // positivedirectionReconoiseD:
    if (jeta - 41 >= 0 && jeta - 41 <= 15) {
      //	     for (int i=0;i<ndepth;i++) {
      // depth=1
      for (int i = 0; i < 1; i++) {
        TH1F* HBpositivedirectionReconoiseD1 = (TH1F*)h2CeffHBpositivedirectionReconoiseD1->Clone("twod1");

        float ccctest = 0;  // to avoid empty massive elements
        for (int jphi = 0; jphi < nphi; jphi++) {
          double ccc1 = reconoisevarianceHB[i][jeta][jphi];
          if (areconoiseHB[i][jeta][jphi] > 0.) {
            HBpositivedirectionReconoiseD1->Fill(jphi, ccc1);
            ccctest = 1.;  //HBpositivedirectionReconoiseD1->SetBinError(i,0.01);
          }
        }  // for jphi
        if (ccctest > 0.) {
          //cout<<"1414       kcountHBpositivedirectionReconoiseD1   =     "<<kcountHBpositivedirectionReconoiseD1  <<"   jeta-41=     "<< jeta-41 <<endl;
          c3x5->cd(kcountHBpositivedirectionReconoiseD1);
          HBpositivedirectionReconoiseD1->SetMarkerStyle(20);
          HBpositivedirectionReconoiseD1->SetMarkerSize(0.4);
          HBpositivedirectionReconoiseD1->GetYaxis()->SetLabelSize(0.04);
          HBpositivedirectionReconoiseD1->SetXTitle("HBpositivedirectionReconoiseD1 \b");
          HBpositivedirectionReconoiseD1->SetMarkerColor(2);
          HBpositivedirectionReconoiseD1->SetLineColor(0);
          gPad->SetGridy();
          gPad->SetGridx();
          //	   gPad->SetLogy();
          if (kcountHBpositivedirectionReconoiseD1 == 1)
            HBpositivedirectionReconoiseD1->SetXTitle("D for HB+ jeta =  0; depth = 1 \b");
          if (kcountHBpositivedirectionReconoiseD1 == 2)
            HBpositivedirectionReconoiseD1->SetXTitle("D for HB+ jeta =  1; depth = 1 \b");
          if (kcountHBpositivedirectionReconoiseD1 == 3)
            HBpositivedirectionReconoiseD1->SetXTitle("D for HB+ jeta =  2; depth = 1 \b");
          if (kcountHBpositivedirectionReconoiseD1 == 4)
            HBpositivedirectionReconoiseD1->SetXTitle("D for HB+ jeta =  3; depth = 1 \b");
          if (kcountHBpositivedirectionReconoiseD1 == 5)
            HBpositivedirectionReconoiseD1->SetXTitle("D for HB+ jeta =  4; depth = 1 \b");
          if (kcountHBpositivedirectionReconoiseD1 == 6)
            HBpositivedirectionReconoiseD1->SetXTitle("D for HB+ jeta =  5; depth = 1 \b");
          if (kcountHBpositivedirectionReconoiseD1 == 7)
            HBpositivedirectionReconoiseD1->SetXTitle("D for HB+ jeta =  6; depth = 1 \b");
          if (kcountHBpositivedirectionReconoiseD1 == 8)
            HBpositivedirectionReconoiseD1->SetXTitle("D for HB+ jeta =  7; depth = 1 \b");
          if (kcountHBpositivedirectionReconoiseD1 == 9)
            HBpositivedirectionReconoiseD1->SetXTitle("D for HB+ jeta =  8; depth = 1 \b");
          if (kcountHBpositivedirectionReconoiseD1 == 10)
            HBpositivedirectionReconoiseD1->SetXTitle("D for HB+ jeta =  9; depth = 1 \b");
          if (kcountHBpositivedirectionReconoiseD1 == 11)
            HBpositivedirectionReconoiseD1->SetXTitle("D for HB+ jeta = 10; depth = 1 \b");
          if (kcountHBpositivedirectionReconoiseD1 == 12)
            HBpositivedirectionReconoiseD1->SetXTitle("D for HB+ jeta = 11; depth = 1 \b");
          if (kcountHBpositivedirectionReconoiseD1 == 13)
            HBpositivedirectionReconoiseD1->SetXTitle("D for HB+ jeta = 12; depth = 1 \b");
          if (kcountHBpositivedirectionReconoiseD1 == 14)
            HBpositivedirectionReconoiseD1->SetXTitle("D for HB+ jeta = 13; depth = 1 \b");
          if (kcountHBpositivedirectionReconoiseD1 == 15)
            HBpositivedirectionReconoiseD1->SetXTitle("D for HB+ jeta = 14; depth = 1 \b");
          if (kcountHBpositivedirectionReconoiseD1 == 16)
            HBpositivedirectionReconoiseD1->SetXTitle("D for HB+ jeta = 15; depth = 1 \b");
          HBpositivedirectionReconoiseD1->Draw("Error");
          kcountHBpositivedirectionReconoiseD1++;
          if (kcountHBpositivedirectionReconoiseD1 > 16)
            break;  // 4x6 = 24
        }           //ccctest>0

      }  // for i
    }    //if(jeta-41 >= 0)
  }      //for jeta
  /////////////////
  c3x5->Update();
  c3x5->Print("DreconoisePositiveDirectionhistD1PhiSymmetryDepth1HB.png");
  c3x5->Clear();
  // clean-up
  if (h2CeffHBpositivedirectionReconoiseD1)
    delete h2CeffHBpositivedirectionReconoiseD1;
  //========================================================================================== 15
  //======================================================================
  //======================================================================1D plot: D vs phi , different eta,  depth=2
  //cout<<"      1D plot: D vs phi , different eta,  depth=2 *****" <<endl;
  c3x5->Clear();
  c3x5->Divide(4, 4);
  c3x5->cd(1);
  int kcountHBpositivedirectionReconoiseD2 = 1;
  TH1F* h2CeffHBpositivedirectionReconoiseD2 = new TH1F("h2CeffHBpositivedirectionReconoiseD2", "", nphi, 0., 72.);

  for (int jeta = 0; jeta < njeta; jeta++) {
    // positivedirectionReconoiseD:
    if (jeta - 41 >= 0 && jeta - 41 <= 15) {
      //	     for (int i=0;i<ndepth;i++) {
      // depth=2
      for (int i = 1; i < 2; i++) {
        TH1F* HBpositivedirectionReconoiseD2 = (TH1F*)h2CeffHBpositivedirectionReconoiseD2->Clone("twod1");

        float ccctest = 0;  // to avoid empty massive elements
        for (int jphi = 0; jphi < nphi; jphi++) {
          double ccc1 = reconoisevarianceHB[i][jeta][jphi];
          if (areconoiseHB[i][jeta][jphi] > 0.) {
            HBpositivedirectionReconoiseD2->Fill(jphi, ccc1);
            ccctest = 1.;  //HBpositivedirectionReconoiseD2->SetBinError(i,0.01);
          }
        }  // for jphi
        if (ccctest > 0.) {
          //cout<<"1515       kcountHBpositivedirectionReconoiseD2   =     "<<kcountHBpositivedirectionReconoiseD2  <<"   jeta-41=     "<< jeta-41 <<endl;
          c3x5->cd(kcountHBpositivedirectionReconoiseD2);
          HBpositivedirectionReconoiseD2->SetMarkerStyle(20);
          HBpositivedirectionReconoiseD2->SetMarkerSize(0.4);
          HBpositivedirectionReconoiseD2->GetYaxis()->SetLabelSize(0.04);
          HBpositivedirectionReconoiseD2->SetXTitle("HBpositivedirectionReconoiseD2 \b");
          HBpositivedirectionReconoiseD2->SetMarkerColor(2);
          HBpositivedirectionReconoiseD2->SetLineColor(0);
          gPad->SetGridy();
          gPad->SetGridx();
          //	   gPad->SetLogy();
          if (kcountHBpositivedirectionReconoiseD2 == 1)
            HBpositivedirectionReconoiseD2->SetXTitle("D for HB+ jeta =  0; depth = 2 \b");
          if (kcountHBpositivedirectionReconoiseD2 == 2)
            HBpositivedirectionReconoiseD2->SetXTitle("D for HB+ jeta =  1; depth = 2 \b");
          if (kcountHBpositivedirectionReconoiseD2 == 3)
            HBpositivedirectionReconoiseD2->SetXTitle("D for HB+ jeta =  2; depth = 2 \b");
          if (kcountHBpositivedirectionReconoiseD2 == 4)
            HBpositivedirectionReconoiseD2->SetXTitle("D for HB+ jeta =  3; depth = 2 \b");
          if (kcountHBpositivedirectionReconoiseD2 == 5)
            HBpositivedirectionReconoiseD2->SetXTitle("D for HB+ jeta =  4; depth = 2 \b");
          if (kcountHBpositivedirectionReconoiseD2 == 6)
            HBpositivedirectionReconoiseD2->SetXTitle("D for HB+ jeta =  5; depth = 2 \b");
          if (kcountHBpositivedirectionReconoiseD2 == 7)
            HBpositivedirectionReconoiseD2->SetXTitle("D for HB+ jeta =  6; depth = 2 \b");
          if (kcountHBpositivedirectionReconoiseD2 == 8)
            HBpositivedirectionReconoiseD2->SetXTitle("D for HB+ jeta =  7; depth = 2 \b");
          if (kcountHBpositivedirectionReconoiseD2 == 9)
            HBpositivedirectionReconoiseD2->SetXTitle("D for HB+ jeta =  8; depth = 2 \b");
          if (kcountHBpositivedirectionReconoiseD2 == 10)
            HBpositivedirectionReconoiseD2->SetXTitle("D for HB+ jeta =  9; depth = 2 \b");
          if (kcountHBpositivedirectionReconoiseD2 == 11)
            HBpositivedirectionReconoiseD2->SetXTitle("D for HB+ jeta = 10; depth = 2 \b");
          if (kcountHBpositivedirectionReconoiseD2 == 12)
            HBpositivedirectionReconoiseD2->SetXTitle("D for HB+ jeta = 11; depth = 2 \b");
          if (kcountHBpositivedirectionReconoiseD2 == 13)
            HBpositivedirectionReconoiseD2->SetXTitle("D for HB+ jeta = 12; depth = 2 \b");
          if (kcountHBpositivedirectionReconoiseD2 == 14)
            HBpositivedirectionReconoiseD2->SetXTitle("D for HB+ jeta = 13; depth = 2 \b");
          if (kcountHBpositivedirectionReconoiseD2 == 15)
            HBpositivedirectionReconoiseD2->SetXTitle("D for HB+ jeta = 14; depth = 2 \b");
          if (kcountHBpositivedirectionReconoiseD2 == 16)
            HBpositivedirectionReconoiseD2->SetXTitle("D for HB+ jeta = 15; depth = 2 \b");
          HBpositivedirectionReconoiseD2->Draw("Error");
          kcountHBpositivedirectionReconoiseD2++;
          if (kcountHBpositivedirectionReconoiseD2 > 16)
            break;  // 4x6 = 24
        }           //ccctest>0

      }  // for i
    }    //if(jeta-41 >= 0)
  }      //for jeta
  /////////////////
  c3x5->Update();
  c3x5->Print("DreconoisePositiveDirectionhistD1PhiSymmetryDepth2HB.png");
  c3x5->Clear();
  // clean-up
  if (h2CeffHBpositivedirectionReconoiseD2)
    delete h2CeffHBpositivedirectionReconoiseD2;
  //========================================================================================== 16
  //======================================================================
  //======================================================================1D plot: D vs phi , different eta,  depth=3
  //cout<<"      1D plot: D vs phi , different eta,  depth=3 *****" <<endl;
  c3x5->Clear();
  c3x5->Divide(4, 4);
  c3x5->cd(1);
  int kcountHBpositivedirectionReconoiseD3 = 1;
  TH1F* h2CeffHBpositivedirectionReconoiseD3 = new TH1F("h2CeffHBpositivedirectionReconoiseD3", "", nphi, 0., 72.);

  for (int jeta = 0; jeta < njeta; jeta++) {
    // positivedirectionReconoiseD:
    if (jeta - 41 >= 0 && jeta - 41 <= 15) {
      //	     for (int i=0;i<ndepth;i++) {
      // depth=3
      for (int i = 2; i < 3; i++) {
        TH1F* HBpositivedirectionReconoiseD3 = (TH1F*)h2CeffHBpositivedirectionReconoiseD3->Clone("twod1");

        float ccctest = 0;  // to avoid empty massive elements
        for (int jphi = 0; jphi < nphi; jphi++) {
          double ccc1 = reconoisevarianceHB[i][jeta][jphi];
          if (areconoiseHB[i][jeta][jphi] > 0.) {
            HBpositivedirectionReconoiseD3->Fill(jphi, ccc1);
            ccctest = 1.;  //HBpositivedirectionReconoiseD3->SetBinError(i,0.01);
          }
        }  // for jphi
        if (ccctest > 0.) {
          //cout<<"1616       kcountHBpositivedirectionReconoiseD3   =     "<<kcountHBpositivedirectionReconoiseD3  <<"   jeta-41=     "<< jeta-41 <<endl;
          c3x5->cd(kcountHBpositivedirectionReconoiseD3);
          HBpositivedirectionReconoiseD3->SetMarkerStyle(20);
          HBpositivedirectionReconoiseD3->SetMarkerSize(0.4);
          HBpositivedirectionReconoiseD3->GetYaxis()->SetLabelSize(0.04);
          HBpositivedirectionReconoiseD3->SetXTitle("HBpositivedirectionReconoiseD3 \b");
          HBpositivedirectionReconoiseD3->SetMarkerColor(2);
          HBpositivedirectionReconoiseD3->SetLineColor(0);
          gPad->SetGridy();
          gPad->SetGridx();
          //	   gPad->SetLogy();
          if (kcountHBpositivedirectionReconoiseD3 == 1)
            HBpositivedirectionReconoiseD3->SetXTitle("D for HB+ jeta =  0; depth = 3 \b");
          if (kcountHBpositivedirectionReconoiseD3 == 2)
            HBpositivedirectionReconoiseD3->SetXTitle("D for HB+ jeta =  1; depth = 3 \b");
          if (kcountHBpositivedirectionReconoiseD3 == 3)
            HBpositivedirectionReconoiseD3->SetXTitle("D for HB+ jeta =  2; depth = 3 \b");
          if (kcountHBpositivedirectionReconoiseD3 == 4)
            HBpositivedirectionReconoiseD3->SetXTitle("D for HB+ jeta =  3; depth = 3 \b");
          if (kcountHBpositivedirectionReconoiseD3 == 5)
            HBpositivedirectionReconoiseD3->SetXTitle("D for HB+ jeta =  4; depth = 3 \b");
          if (kcountHBpositivedirectionReconoiseD3 == 6)
            HBpositivedirectionReconoiseD3->SetXTitle("D for HB+ jeta =  5; depth = 3 \b");
          if (kcountHBpositivedirectionReconoiseD3 == 7)
            HBpositivedirectionReconoiseD3->SetXTitle("D for HB+ jeta =  6; depth = 3 \b");
          if (kcountHBpositivedirectionReconoiseD3 == 8)
            HBpositivedirectionReconoiseD3->SetXTitle("D for HB+ jeta =  7; depth = 3 \b");
          if (kcountHBpositivedirectionReconoiseD3 == 9)
            HBpositivedirectionReconoiseD3->SetXTitle("D for HB+ jeta =  8; depth = 3 \b");
          if (kcountHBpositivedirectionReconoiseD3 == 10)
            HBpositivedirectionReconoiseD3->SetXTitle("D for HB+ jeta =  9; depth = 3 \b");
          if (kcountHBpositivedirectionReconoiseD3 == 11)
            HBpositivedirectionReconoiseD3->SetXTitle("D for HB+ jeta = 10; depth = 3 \b");
          if (kcountHBpositivedirectionReconoiseD3 == 12)
            HBpositivedirectionReconoiseD3->SetXTitle("D for HB+ jeta = 11; depth = 3 \b");
          if (kcountHBpositivedirectionReconoiseD3 == 13)
            HBpositivedirectionReconoiseD3->SetXTitle("D for HB+ jeta = 12; depth = 3 \b");
          if (kcountHBpositivedirectionReconoiseD3 == 14)
            HBpositivedirectionReconoiseD3->SetXTitle("D for HB+ jeta = 13; depth = 3 \b");
          if (kcountHBpositivedirectionReconoiseD3 == 15)
            HBpositivedirectionReconoiseD3->SetXTitle("D for HB+ jeta = 14; depth = 3 \b");
          if (kcountHBpositivedirectionReconoiseD3 == 16)
            HBpositivedirectionReconoiseD3->SetXTitle("D for HB+ jeta = 15; depth = 3 \b");
          HBpositivedirectionReconoiseD3->Draw("Error");
          kcountHBpositivedirectionReconoiseD3++;
          if (kcountHBpositivedirectionReconoiseD3 > 16)
            break;  // 4x6 = 24
        }           //ccctest>0

      }  // for i
    }    //if(jeta-41 >= 0)
  }      //for jeta
  /////////////////
  c3x5->Update();
  c3x5->Print("DreconoisePositiveDirectionhistD1PhiSymmetryDepth3HB.png");
  c3x5->Clear();
  // clean-up
  if (h2CeffHBpositivedirectionReconoiseD3)
    delete h2CeffHBpositivedirectionReconoiseD3;
  //========================================================================================== 17
  //======================================================================
  //======================================================================1D plot: D vs phi , different eta,  depth=4
  //cout<<"      1D plot: D vs phi , different eta,  depth=4 *****" <<endl;
  c3x5->Clear();
  c3x5->Divide(4, 4);
  c3x5->cd(1);
  int kcountHBpositivedirectionReconoiseD4 = 1;
  TH1F* h2CeffHBpositivedirectionReconoiseD4 = new TH1F("h2CeffHBpositivedirectionReconoiseD4", "", nphi, 0., 72.);

  for (int jeta = 0; jeta < njeta; jeta++) {
    // positivedirectionReconoiseD:
    if (jeta - 41 >= 0 && jeta - 41 <= 15) {
      //	     for (int i=0;i<ndepth;i++) {
      // depth=4
      for (int i = 3; i < 4; i++) {
        TH1F* HBpositivedirectionReconoiseD4 = (TH1F*)h2CeffHBpositivedirectionReconoiseD4->Clone("twod1");

        float ccctest = 0;  // to avoid empty massive elements
        for (int jphi = 0; jphi < nphi; jphi++) {
          double ccc1 = reconoisevarianceHB[i][jeta][jphi];
          if (areconoiseHB[i][jeta][jphi] > 0.) {
            HBpositivedirectionReconoiseD4->Fill(jphi, ccc1);
            ccctest = 1.;  //HBpositivedirectionReconoiseD4->SetBinError(i,0.01);
          }
        }  // for jphi
        if (ccctest > 0.) {
          //cout<<"1717       kcountHBpositivedirectionReconoiseD4   =     "<<kcountHBpositivedirectionReconoiseD4  <<"   jeta-41=     "<< jeta-41 <<endl;
          c3x5->cd(kcountHBpositivedirectionReconoiseD4);
          HBpositivedirectionReconoiseD4->SetMarkerStyle(20);
          HBpositivedirectionReconoiseD4->SetMarkerSize(0.4);
          HBpositivedirectionReconoiseD4->GetYaxis()->SetLabelSize(0.04);
          HBpositivedirectionReconoiseD4->SetXTitle("HBpositivedirectionReconoiseD4 \b");
          HBpositivedirectionReconoiseD4->SetMarkerColor(2);
          HBpositivedirectionReconoiseD4->SetLineColor(0);
          gPad->SetGridy();
          gPad->SetGridx();
          //	   gPad->SetLogy();
          if (kcountHBpositivedirectionReconoiseD4 == 1)
            HBpositivedirectionReconoiseD4->SetXTitle("D for HB+ jeta =  0; depth = 4 \b");
          if (kcountHBpositivedirectionReconoiseD4 == 2)
            HBpositivedirectionReconoiseD4->SetXTitle("D for HB+ jeta =  1; depth = 4 \b");
          if (kcountHBpositivedirectionReconoiseD4 == 3)
            HBpositivedirectionReconoiseD4->SetXTitle("D for HB+ jeta =  2; depth = 4 \b");
          if (kcountHBpositivedirectionReconoiseD4 == 4)
            HBpositivedirectionReconoiseD4->SetXTitle("D for HB+ jeta =  3; depth = 4 \b");
          if (kcountHBpositivedirectionReconoiseD4 == 5)
            HBpositivedirectionReconoiseD4->SetXTitle("D for HB+ jeta =  4; depth = 4 \b");
          if (kcountHBpositivedirectionReconoiseD4 == 6)
            HBpositivedirectionReconoiseD4->SetXTitle("D for HB+ jeta =  5; depth = 4 \b");
          if (kcountHBpositivedirectionReconoiseD4 == 7)
            HBpositivedirectionReconoiseD4->SetXTitle("D for HB+ jeta =  6; depth = 4 \b");
          if (kcountHBpositivedirectionReconoiseD4 == 8)
            HBpositivedirectionReconoiseD4->SetXTitle("D for HB+ jeta =  7; depth = 4 \b");
          if (kcountHBpositivedirectionReconoiseD4 == 9)
            HBpositivedirectionReconoiseD4->SetXTitle("D for HB+ jeta =  8; depth = 4 \b");
          if (kcountHBpositivedirectionReconoiseD4 == 10)
            HBpositivedirectionReconoiseD4->SetXTitle("D for HB+ jeta =  9; depth = 4 \b");
          if (kcountHBpositivedirectionReconoiseD4 == 11)
            HBpositivedirectionReconoiseD4->SetXTitle("D for HB+ jeta = 10; depth = 4 \b");
          if (kcountHBpositivedirectionReconoiseD4 == 12)
            HBpositivedirectionReconoiseD4->SetXTitle("D for HB+ jeta = 11; depth = 4 \b");
          if (kcountHBpositivedirectionReconoiseD4 == 13)
            HBpositivedirectionReconoiseD4->SetXTitle("D for HB+ jeta = 12; depth = 4 \b");
          if (kcountHBpositivedirectionReconoiseD4 == 14)
            HBpositivedirectionReconoiseD4->SetXTitle("D for HB+ jeta = 13; depth = 4 \b");
          if (kcountHBpositivedirectionReconoiseD4 == 15)
            HBpositivedirectionReconoiseD4->SetXTitle("D for HB+ jeta = 14; depth = 4 \b");
          if (kcountHBpositivedirectionReconoiseD4 == 16)
            HBpositivedirectionReconoiseD4->SetXTitle("D for HB+ jeta = 15; depth = 4 \b");
          HBpositivedirectionReconoiseD4->Draw("Error");
          kcountHBpositivedirectionReconoiseD4++;
          if (kcountHBpositivedirectionReconoiseD4 > 16)
            break;  // 4x6 = 24
        }           //ccctest>0

      }  // for i
    }    //if(jeta-41 >= 0)
  }      //for jeta
  /////////////////
  c3x5->Update();
  c3x5->Print("DreconoisePositiveDirectionhistD1PhiSymmetryDepth4HB.png");
  c3x5->Clear();
  // clean-up
  if (h2CeffHBpositivedirectionReconoiseD4)
    delete h2CeffHBpositivedirectionReconoiseD4;

  //========================================================================================== 22214
  //======================================================================
  //======================================================================1D plot: D vs phi , different eta,  depth=1
  //cout<<"      1D plot: D vs phi , different eta,  depth=1 *****" <<endl;
  c3x5->Clear();
  /////////////////
  c3x5->Divide(4, 4);
  c3x5->cd(1);
  int kcountHBnegativedirectionReconoiseD1 = 1;
  TH1F* h2CeffHBnegativedirectionReconoiseD1 = new TH1F("h2CeffHBnegativedirectionReconoiseD1", "", nphi, 0., 72.);

  for (int jeta = 0; jeta < njeta; jeta++) {
    // negativedirectionReconoiseD:
    if (jeta - 41 >= -16 && jeta - 41 <= -1) {
      //	     for (int i=0;i<ndepth;i++) {
      // depth=1
      for (int i = 0; i < 1; i++) {
        TH1F* HBnegativedirectionReconoiseD1 = (TH1F*)h2CeffHBnegativedirectionReconoiseD1->Clone("twod1");

        float ccctest = 0;  // to avoid empty massive elements
        for (int jphi = 0; jphi < nphi; jphi++) {
          double ccc1 = reconoisevarianceHB[i][jeta][jphi];
          if (areconoiseHB[i][jeta][jphi] > 0.) {
            HBnegativedirectionReconoiseD1->Fill(jphi, ccc1);
            ccctest = 1.;  //HBnegativedirectionReconoiseD1->SetBinError(i,0.01);
          }
        }  // for jphi
        if (ccctest > 0.) {
          //cout<<"1414       kcountHBnegativedirectionReconoiseD1   =     "<<kcountHBnegativedirectionReconoiseD1  <<"   jeta-41=     "<< jeta-41 <<endl;
          c3x5->cd(kcountHBnegativedirectionReconoiseD1);
          HBnegativedirectionReconoiseD1->SetMarkerStyle(20);
          HBnegativedirectionReconoiseD1->SetMarkerSize(0.4);
          HBnegativedirectionReconoiseD1->GetYaxis()->SetLabelSize(0.04);
          HBnegativedirectionReconoiseD1->SetXTitle("HBnegativedirectionReconoiseD1 \b");
          HBnegativedirectionReconoiseD1->SetMarkerColor(2);
          HBnegativedirectionReconoiseD1->SetLineColor(0);
          gPad->SetGridy();
          gPad->SetGridx();
          //	   gPad->SetLogy();
          if (kcountHBnegativedirectionReconoiseD1 == 1)
            HBnegativedirectionReconoiseD1->SetXTitle("D for HB- jeta =-16; depth = 1 \b");
          if (kcountHBnegativedirectionReconoiseD1 == 2)
            HBnegativedirectionReconoiseD1->SetXTitle("D for HB- jeta =-15; depth = 1 \b");
          if (kcountHBnegativedirectionReconoiseD1 == 3)
            HBnegativedirectionReconoiseD1->SetXTitle("D for HB- jeta =-14; depth = 1 \b");
          if (kcountHBnegativedirectionReconoiseD1 == 4)
            HBnegativedirectionReconoiseD1->SetXTitle("D for HB- jeta =-13; depth = 1 \b");
          if (kcountHBnegativedirectionReconoiseD1 == 5)
            HBnegativedirectionReconoiseD1->SetXTitle("D for HB- jeta =-12; depth = 1 \b");
          if (kcountHBnegativedirectionReconoiseD1 == 6)
            HBnegativedirectionReconoiseD1->SetXTitle("D for HB- jeta =-11; depth = 1 \b");
          if (kcountHBnegativedirectionReconoiseD1 == 7)
            HBnegativedirectionReconoiseD1->SetXTitle("D for HB- jeta =-10; depth = 1 \b");
          if (kcountHBnegativedirectionReconoiseD1 == 8)
            HBnegativedirectionReconoiseD1->SetXTitle("D for HB- jeta =-9; depth = 1 \b");
          if (kcountHBnegativedirectionReconoiseD1 == 9)
            HBnegativedirectionReconoiseD1->SetXTitle("D for HB- jeta =-8; depth = 1 \b");
          if (kcountHBnegativedirectionReconoiseD1 == 10)
            HBnegativedirectionReconoiseD1->SetXTitle("D for HB- jeta =-7; depth = 1 \b");
          if (kcountHBnegativedirectionReconoiseD1 == 11)
            HBnegativedirectionReconoiseD1->SetXTitle("D for HB- jeta =-6; depth = 1 \b");
          if (kcountHBnegativedirectionReconoiseD1 == 12)
            HBnegativedirectionReconoiseD1->SetXTitle("D for HB- jeta =-5; depth = 1 \b");
          if (kcountHBnegativedirectionReconoiseD1 == 13)
            HBnegativedirectionReconoiseD1->SetXTitle("D for HB- jeta =-4; depth = 1 \b");
          if (kcountHBnegativedirectionReconoiseD1 == 14)
            HBnegativedirectionReconoiseD1->SetXTitle("D for HB- jeta =-3; depth = 1 \b");
          if (kcountHBnegativedirectionReconoiseD1 == 15)
            HBnegativedirectionReconoiseD1->SetXTitle("D for HB- jeta =-2; depth = 1 \b");
          if (kcountHBnegativedirectionReconoiseD1 == 16)
            HBnegativedirectionReconoiseD1->SetXTitle("D for HB- jeta =-1; depth = 1 \b");
          HBnegativedirectionReconoiseD1->Draw("Error");
          kcountHBnegativedirectionReconoiseD1++;
          if (kcountHBnegativedirectionReconoiseD1 > 16)
            break;  // 4x6 = 24
        }           //ccctest>0

      }  // for i
    }    //if(jeta-41 < 0)
  }      //for jeta
  /////////////////
  c3x5->Update();
  c3x5->Print("DreconoiseNegativeDirectionhistD1PhiSymmetryDepth1HB.png");
  c3x5->Clear();
  // clean-up
  if (h2CeffHBnegativedirectionReconoiseD1)
    delete h2CeffHBnegativedirectionReconoiseD1;
  //========================================================================================== 22215
  //======================================================================
  //======================================================================1D plot: D vs phi , different eta,  depth=2
  //cout<<"      1D plot: D vs phi , different eta,  depth=2 *****" <<endl;
  c3x5->Clear();
  c3x5->Divide(4, 4);
  c3x5->cd(1);
  int kcountHBnegativedirectionReconoiseD2 = 1;
  TH1F* h2CeffHBnegativedirectionReconoiseD2 = new TH1F("h2CeffHBnegativedirectionReconoiseD2", "", nphi, 0., 72.);

  for (int jeta = 0; jeta < njeta; jeta++) {
    // negativedirectionReconoiseD:
    if (jeta - 41 >= -16 && jeta - 41 <= -1) {
      //	     for (int i=0;i<ndepth;i++) {
      // depth=2
      for (int i = 1; i < 2; i++) {
        TH1F* HBnegativedirectionReconoiseD2 = (TH1F*)h2CeffHBnegativedirectionReconoiseD2->Clone("twod1");

        float ccctest = 0;  // to avoid empty massive elements
        for (int jphi = 0; jphi < nphi; jphi++) {
          double ccc1 = reconoisevarianceHB[i][jeta][jphi];
          if (areconoiseHB[i][jeta][jphi] > 0.) {
            HBnegativedirectionReconoiseD2->Fill(jphi, ccc1);
            ccctest = 1.;  //HBnegativedirectionReconoiseD2->SetBinError(i,0.01);
          }
        }  // for jphi
        if (ccctest > 0.) {
          //cout<<"1515       kcountHBnegativedirectionReconoiseD2   =     "<<kcountHBnegativedirectionReconoiseD2  <<"   jeta-41=     "<< jeta-41 <<endl;
          c3x5->cd(kcountHBnegativedirectionReconoiseD2);
          HBnegativedirectionReconoiseD2->SetMarkerStyle(20);
          HBnegativedirectionReconoiseD2->SetMarkerSize(0.4);
          HBnegativedirectionReconoiseD2->GetYaxis()->SetLabelSize(0.04);
          HBnegativedirectionReconoiseD2->SetXTitle("HBnegativedirectionReconoiseD2 \b");
          HBnegativedirectionReconoiseD2->SetMarkerColor(2);
          HBnegativedirectionReconoiseD2->SetLineColor(0);
          gPad->SetGridy();
          gPad->SetGridx();
          //	   gPad->SetLogy();
          if (kcountHBnegativedirectionReconoiseD2 == 1)
            HBnegativedirectionReconoiseD2->SetXTitle("D for HB- jeta =-16; depth = 2 \b");
          if (kcountHBnegativedirectionReconoiseD2 == 2)
            HBnegativedirectionReconoiseD2->SetXTitle("D for HB- jeta =-15; depth = 2 \b");
          if (kcountHBnegativedirectionReconoiseD2 == 3)
            HBnegativedirectionReconoiseD2->SetXTitle("D for HB- jeta =-14; depth = 2 \b");
          if (kcountHBnegativedirectionReconoiseD2 == 4)
            HBnegativedirectionReconoiseD2->SetXTitle("D for HB- jeta =-13; depth = 2 \b");
          if (kcountHBnegativedirectionReconoiseD2 == 5)
            HBnegativedirectionReconoiseD2->SetXTitle("D for HB- jeta =-12; depth = 2 \b");
          if (kcountHBnegativedirectionReconoiseD2 == 6)
            HBnegativedirectionReconoiseD2->SetXTitle("D for HB- jeta =-11; depth = 2 \b");
          if (kcountHBnegativedirectionReconoiseD2 == 7)
            HBnegativedirectionReconoiseD2->SetXTitle("D for HB- jeta =-10; depth = 2 \b");
          if (kcountHBnegativedirectionReconoiseD2 == 8)
            HBnegativedirectionReconoiseD2->SetXTitle("D for HB- jeta =-9; depth = 2 \b");
          if (kcountHBnegativedirectionReconoiseD2 == 9)
            HBnegativedirectionReconoiseD2->SetXTitle("D for HB- jeta =-8; depth = 2 \b");
          if (kcountHBnegativedirectionReconoiseD2 == 10)
            HBnegativedirectionReconoiseD2->SetXTitle("D for HB- jeta =-7; depth = 2 \b");
          if (kcountHBnegativedirectionReconoiseD2 == 11)
            HBnegativedirectionReconoiseD2->SetXTitle("D for HB- jeta =-6; depth = 2 \b");
          if (kcountHBnegativedirectionReconoiseD2 == 12)
            HBnegativedirectionReconoiseD2->SetXTitle("D for HB- jeta =-5; depth = 2 \b");
          if (kcountHBnegativedirectionReconoiseD2 == 13)
            HBnegativedirectionReconoiseD2->SetXTitle("D for HB- jeta =-4; depth = 2 \b");
          if (kcountHBnegativedirectionReconoiseD2 == 14)
            HBnegativedirectionReconoiseD2->SetXTitle("D for HB- jeta =-3; depth = 2 \b");
          if (kcountHBnegativedirectionReconoiseD2 == 15)
            HBnegativedirectionReconoiseD2->SetXTitle("D for HB- jeta =-2; depth = 2 \b");
          if (kcountHBnegativedirectionReconoiseD2 == 16)
            HBnegativedirectionReconoiseD2->SetXTitle("D for HB- jeta =-1; depth = 2 \b");
          HBnegativedirectionReconoiseD2->Draw("Error");
          kcountHBnegativedirectionReconoiseD2++;
          if (kcountHBnegativedirectionReconoiseD2 > 16)
            break;  // 4x6 = 24
        }           //ccctest>0

      }  // for i
    }    //if(jeta-41 < 0)
  }      //for jeta
  /////////////////
  c3x5->Update();
  c3x5->Print("DreconoiseNegativeDirectionhistD1PhiSymmetryDepth2HB.png");
  c3x5->Clear();
  // clean-up
  if (h2CeffHBnegativedirectionReconoiseD2)
    delete h2CeffHBnegativedirectionReconoiseD2;
  //========================================================================================== 22216
  //======================================================================
  //======================================================================1D plot: D vs phi , different eta,  depth=3
  //cout<<"      1D plot: D vs phi , different eta,  depth=3 *****" <<endl;
  c3x5->Clear();
  c3x5->Divide(4, 4);
  c3x5->cd(1);
  int kcountHBnegativedirectionReconoiseD3 = 1;
  TH1F* h2CeffHBnegativedirectionReconoiseD3 = new TH1F("h2CeffHBnegativedirectionReconoiseD3", "", nphi, 0., 72.);

  for (int jeta = 0; jeta < njeta; jeta++) {
    // negativedirectionReconoiseD:
    if (jeta - 41 >= -16 && jeta - 41 <= -1) {
      //	     for (int i=0;i<ndepth;i++) {
      // depth=3
      for (int i = 2; i < 3; i++) {
        TH1F* HBnegativedirectionReconoiseD3 = (TH1F*)h2CeffHBnegativedirectionReconoiseD3->Clone("twod1");

        float ccctest = 0;  // to avoid empty massive elements
        for (int jphi = 0; jphi < nphi; jphi++) {
          double ccc1 = reconoisevarianceHB[i][jeta][jphi];
          if (areconoiseHB[i][jeta][jphi] > 0.) {
            HBnegativedirectionReconoiseD3->Fill(jphi, ccc1);
            ccctest = 1.;  //HBnegativedirectionReconoiseD3->SetBinError(i,0.01);
          }
        }  // for jphi
        if (ccctest > 0.) {
          //cout<<"1616       kcountHBnegativedirectionReconoiseD3   =     "<<kcountHBnegativedirectionReconoiseD3  <<"   jeta-41=     "<< jeta-41 <<endl;
          c3x5->cd(kcountHBnegativedirectionReconoiseD3);
          HBnegativedirectionReconoiseD3->SetMarkerStyle(20);
          HBnegativedirectionReconoiseD3->SetMarkerSize(0.4);
          HBnegativedirectionReconoiseD3->GetYaxis()->SetLabelSize(0.04);
          HBnegativedirectionReconoiseD3->SetXTitle("HBnegativedirectionReconoiseD3 \b");
          HBnegativedirectionReconoiseD3->SetMarkerColor(2);
          HBnegativedirectionReconoiseD3->SetLineColor(0);
          gPad->SetGridy();
          gPad->SetGridx();
          //	   gPad->SetLogy();
          if (kcountHBnegativedirectionReconoiseD3 == 1)
            HBnegativedirectionReconoiseD3->SetXTitle("D for HB- jeta =-16; depth = 3 \b");
          if (kcountHBnegativedirectionReconoiseD3 == 2)
            HBnegativedirectionReconoiseD3->SetXTitle("D for HB- jeta =-15; depth = 3 \b");
          if (kcountHBnegativedirectionReconoiseD3 == 3)
            HBnegativedirectionReconoiseD3->SetXTitle("D for HB- jeta =-14; depth = 3 \b");
          if (kcountHBnegativedirectionReconoiseD3 == 4)
            HBnegativedirectionReconoiseD3->SetXTitle("D for HB- jeta =-13; depth = 3 \b");
          if (kcountHBnegativedirectionReconoiseD3 == 5)
            HBnegativedirectionReconoiseD3->SetXTitle("D for HB- jeta =-12; depth = 3 \b");
          if (kcountHBnegativedirectionReconoiseD3 == 6)
            HBnegativedirectionReconoiseD3->SetXTitle("D for HB- jeta =-11; depth = 3 \b");
          if (kcountHBnegativedirectionReconoiseD3 == 7)
            HBnegativedirectionReconoiseD3->SetXTitle("D for HB- jeta =-10; depth = 3 \b");
          if (kcountHBnegativedirectionReconoiseD3 == 8)
            HBnegativedirectionReconoiseD3->SetXTitle("D for HB- jeta =-9; depth = 3 \b");
          if (kcountHBnegativedirectionReconoiseD3 == 9)
            HBnegativedirectionReconoiseD3->SetXTitle("D for HB- jeta =-8; depth = 3 \b");
          if (kcountHBnegativedirectionReconoiseD3 == 10)
            HBnegativedirectionReconoiseD3->SetXTitle("D for HB- jeta =-7; depth = 3 \b");
          if (kcountHBnegativedirectionReconoiseD3 == 11)
            HBnegativedirectionReconoiseD3->SetXTitle("D for HB- jeta =-6; depth = 3 \b");
          if (kcountHBnegativedirectionReconoiseD3 == 12)
            HBnegativedirectionReconoiseD3->SetXTitle("D for HB- jeta =-5; depth = 3 \b");
          if (kcountHBnegativedirectionReconoiseD3 == 13)
            HBnegativedirectionReconoiseD3->SetXTitle("D for HB- jeta =-4; depth = 3 \b");
          if (kcountHBnegativedirectionReconoiseD3 == 14)
            HBnegativedirectionReconoiseD3->SetXTitle("D for HB- jeta =-3; depth = 3 \b");
          if (kcountHBnegativedirectionReconoiseD3 == 15)
            HBnegativedirectionReconoiseD3->SetXTitle("D for HB- jeta =-2; depth = 3 \b");
          if (kcountHBnegativedirectionReconoiseD3 == 16)
            HBnegativedirectionReconoiseD3->SetXTitle("D for HB- jeta =-1; depth = 3 \b");
          HBnegativedirectionReconoiseD3->Draw("Error");
          kcountHBnegativedirectionReconoiseD3++;
          if (kcountHBnegativedirectionReconoiseD3 > 16)
            break;  // 4x6 = 24
        }           //ccctest>0

      }  // for i
    }    //if(jeta-41 < 0)
  }      //for jeta
  /////////////////
  c3x5->Update();
  c3x5->Print("DreconoiseNegativeDirectionhistD1PhiSymmetryDepth3HB.png");
  c3x5->Clear();
  // clean-up
  if (h2CeffHBnegativedirectionReconoiseD3)
    delete h2CeffHBnegativedirectionReconoiseD3;
  //========================================================================================== 22217
  //======================================================================
  //======================================================================1D plot: D vs phi , different eta,  depth=4
  //cout<<"      1D plot: D vs phi , different eta,  depth=4 *****" <<endl;
  c3x5->Clear();
  c3x5->Divide(4, 4);
  c3x5->cd(1);
  int kcountHBnegativedirectionReconoiseD4 = 1;
  TH1F* h2CeffHBnegativedirectionReconoiseD4 = new TH1F("h2CeffHBnegativedirectionReconoiseD4", "", nphi, 0., 72.);

  for (int jeta = 0; jeta < njeta; jeta++) {
    // negativedirectionReconoiseD:
    if (jeta - 41 >= -16 && jeta - 41 <= -1) {
      //	     for (int i=0;i<ndepth;i++) {
      // depth=4
      for (int i = 3; i < 4; i++) {
        TH1F* HBnegativedirectionReconoiseD4 = (TH1F*)h2CeffHBnegativedirectionReconoiseD4->Clone("twod1");

        float ccctest = 0;  // to avoid empty massive elements
        for (int jphi = 0; jphi < nphi; jphi++) {
          double ccc1 = reconoisevarianceHB[i][jeta][jphi];
          if (areconoiseHB[i][jeta][jphi] > 0.) {
            HBnegativedirectionReconoiseD4->Fill(jphi, ccc1);
            ccctest = 1.;  //HBnegativedirectionReconoiseD4->SetBinError(i,0.01);
          }
        }  // for jphi
        if (ccctest > 0.) {
          //cout<<"1717       kcountHBnegativedirectionReconoiseD4   =     "<<kcountHBnegativedirectionReconoiseD4  <<"   jeta-41=     "<< jeta-41 <<endl;
          c3x5->cd(kcountHBnegativedirectionReconoiseD4);
          HBnegativedirectionReconoiseD4->SetMarkerStyle(20);
          HBnegativedirectionReconoiseD4->SetMarkerSize(0.4);
          HBnegativedirectionReconoiseD4->GetYaxis()->SetLabelSize(0.04);
          HBnegativedirectionReconoiseD4->SetXTitle("HBnegativedirectionReconoiseD4 \b");
          HBnegativedirectionReconoiseD4->SetMarkerColor(2);
          HBnegativedirectionReconoiseD4->SetLineColor(0);
          gPad->SetGridy();
          gPad->SetGridx();
          //	   gPad->SetLogy();
          if (kcountHBnegativedirectionReconoiseD4 == 1)
            HBnegativedirectionReconoiseD4->SetXTitle("D for HB- jeta =-16; depth = 4 \b");
          if (kcountHBnegativedirectionReconoiseD4 == 2)
            HBnegativedirectionReconoiseD4->SetXTitle("D for HB- jeta =-15; depth = 4 \b");
          if (kcountHBnegativedirectionReconoiseD4 == 3)
            HBnegativedirectionReconoiseD4->SetXTitle("D for HB- jeta =-14; depth = 4 \b");
          if (kcountHBnegativedirectionReconoiseD4 == 4)
            HBnegativedirectionReconoiseD4->SetXTitle("D for HB- jeta =-13; depth = 4 \b");
          if (kcountHBnegativedirectionReconoiseD4 == 5)
            HBnegativedirectionReconoiseD4->SetXTitle("D for HB- jeta =-12; depth = 4 \b");
          if (kcountHBnegativedirectionReconoiseD4 == 6)
            HBnegativedirectionReconoiseD4->SetXTitle("D for HB- jeta =-11; depth = 4 \b");
          if (kcountHBnegativedirectionReconoiseD4 == 7)
            HBnegativedirectionReconoiseD4->SetXTitle("D for HB- jeta =-10; depth = 4 \b");
          if (kcountHBnegativedirectionReconoiseD4 == 8)
            HBnegativedirectionReconoiseD4->SetXTitle("D for HB- jeta =-9; depth = 4 \b");
          if (kcountHBnegativedirectionReconoiseD4 == 9)
            HBnegativedirectionReconoiseD4->SetXTitle("D for HB- jeta =-8; depth = 4 \b");
          if (kcountHBnegativedirectionReconoiseD4 == 10)
            HBnegativedirectionReconoiseD4->SetXTitle("D for HB- jeta =-7; depth = 4 \b");
          if (kcountHBnegativedirectionReconoiseD4 == 11)
            HBnegativedirectionReconoiseD4->SetXTitle("D for HB- jeta =-6; depth = 4 \b");
          if (kcountHBnegativedirectionReconoiseD4 == 12)
            HBnegativedirectionReconoiseD4->SetXTitle("D for HB- jeta =-5; depth = 4 \b");
          if (kcountHBnegativedirectionReconoiseD4 == 13)
            HBnegativedirectionReconoiseD4->SetXTitle("D for HB- jeta =-4; depth = 4 \b");
          if (kcountHBnegativedirectionReconoiseD4 == 14)
            HBnegativedirectionReconoiseD4->SetXTitle("D for HB- jeta =-3; depth = 4 \b");
          if (kcountHBnegativedirectionReconoiseD4 == 15)
            HBnegativedirectionReconoiseD4->SetXTitle("D for HB- jeta =-2; depth = 4 \b");
          if (kcountHBnegativedirectionReconoiseD4 == 16)
            HBnegativedirectionReconoiseD4->SetXTitle("D for HB- jeta =-1; depth = 4 \b");
          HBnegativedirectionReconoiseD4->Draw("Error");
          kcountHBnegativedirectionReconoiseD4++;
          if (kcountHBnegativedirectionReconoiseD4 > 16)
            break;  // 4x6 = 24
        }           //ccctest>0

      }  // for i
    }    //if(jeta-41 < 0)
  }      //for jeta
  /////////////////
  c3x5->Update();
  c3x5->Print("DreconoiseNegativeDirectionhistD1PhiSymmetryDepth4HB.png");
  c3x5->Clear();
  // clean-up
  if (h2CeffHBnegativedirectionReconoiseD4)
    delete h2CeffHBnegativedirectionReconoiseD4;

  //=====================================================================       END of Reconoise HB for phi-symmetry
  //=====================================================================       END of Reconoise HB for phi-symmetry
  //=====================================================================       END of Reconoise HB for phi-symmetry

  ////////////////////////////////////////////////////////////////////////////////////////////////////         Phi-symmetry for Calibration Group:    Reconoise HE
  ////////////////////////////////////////////////////////////////////////////////////////////////////         Phi-symmetry for Calibration Group:    Reconoise HE
  ////////////////////////////////////////////////////////////////////////////////////////////////////         Phi-symmetry for Calibration Group:    Reconoise HE
  //  int k_max[5]={0,4,7,4,4}; // maximum depth for each subdet
  //ndepth = k_max[3];
  ndepth = 7;
  //  const int ndepth = 7;
  double areconoisehe[ndepth][njeta][njphi];
  double breconoisehe[ndepth][njeta][njphi];
  double reconoisevariancehe[ndepth][njeta][njphi];
  ////////////////////////////////////////////////////////////////////////////////////////////////////////////
  TH2F* recNoiseEnergy1HE1 = (TH2F*)dir->FindObjectAny("h_recNoiseEnergy1_HE1");
  TH2F* recNoiseEnergy0HE1 = (TH2F*)dir->FindObjectAny("h_recNoiseEnergy0_HE1");
  TH2F* recNoiseEnergyHE1 = (TH2F*)recNoiseEnergy1HE1->Clone("recNoiseEnergyHE1");
  recNoiseEnergyHE1->Divide(recNoiseEnergy1HE1, recNoiseEnergy0HE1, 1, 1, "B");
  TH2F* recNoiseEnergy1HE2 = (TH2F*)dir->FindObjectAny("h_recNoiseEnergy1_HE2");
  TH2F* recNoiseEnergy0HE2 = (TH2F*)dir->FindObjectAny("h_recNoiseEnergy0_HE2");
  TH2F* recNoiseEnergyHE2 = (TH2F*)recNoiseEnergy1HE2->Clone("recNoiseEnergyHE2");
  recNoiseEnergyHE2->Divide(recNoiseEnergy1HE2, recNoiseEnergy0HE2, 1, 1, "B");
  TH2F* recNoiseEnergy1HE3 = (TH2F*)dir->FindObjectAny("h_recNoiseEnergy1_HE3");
  TH2F* recNoiseEnergy0HE3 = (TH2F*)dir->FindObjectAny("h_recNoiseEnergy0_HE3");
  TH2F* recNoiseEnergyHE3 = (TH2F*)recNoiseEnergy1HE3->Clone("recNoiseEnergyHE3");
  recNoiseEnergyHE3->Divide(recNoiseEnergy1HE3, recNoiseEnergy0HE3, 1, 1, "B");
  TH2F* recNoiseEnergy1HE4 = (TH2F*)dir->FindObjectAny("h_recNoiseEnergy1_HE4");
  TH2F* recNoiseEnergy0HE4 = (TH2F*)dir->FindObjectAny("h_recNoiseEnergy0_HE4");
  TH2F* recNoiseEnergyHE4 = (TH2F*)recNoiseEnergy1HE4->Clone("recNoiseEnergyHE4");
  recNoiseEnergyHE4->Divide(recNoiseEnergy1HE4, recNoiseEnergy0HE4, 1, 1, "B");
  TH2F* recNoiseEnergy1HE5 = (TH2F*)dir->FindObjectAny("h_recNoiseEnergy1_HE5");
  TH2F* recNoiseEnergy0HE5 = (TH2F*)dir->FindObjectAny("h_recNoiseEnergy0_HE5");
  TH2F* recNoiseEnergyHE5 = (TH2F*)recNoiseEnergy1HE5->Clone("recNoiseEnergyHE5");
  recNoiseEnergyHE5->Divide(recNoiseEnergy1HE5, recNoiseEnergy0HE5, 1, 1, "B");
  TH2F* recNoiseEnergy1HE6 = (TH2F*)dir->FindObjectAny("h_recNoiseEnergy1_HE6");
  TH2F* recNoiseEnergy0HE6 = (TH2F*)dir->FindObjectAny("h_recNoiseEnergy0_HE6");
  TH2F* recNoiseEnergyHE6 = (TH2F*)recNoiseEnergy1HE6->Clone("recNoiseEnergyHE6");
  recNoiseEnergyHE6->Divide(recNoiseEnergy1HE6, recNoiseEnergy0HE6, 1, 1, "B");
  TH2F* recNoiseEnergy1HE7 = (TH2F*)dir->FindObjectAny("h_recNoiseEnergy1_HE7");
  TH2F* recNoiseEnergy0HE7 = (TH2F*)dir->FindObjectAny("h_recNoiseEnergy0_HE7");
  TH2F* recNoiseEnergyHE7 = (TH2F*)recNoiseEnergy1HE7->Clone("recNoiseEnergyHE7");
  recNoiseEnergyHE7->Divide(recNoiseEnergy1HE7, recNoiseEnergy0HE7, 1, 1, "B");
  for (int jeta = 0; jeta < njeta; jeta++) {
    if ((jeta - 41 >= -29 && jeta - 41 <= -16) || (jeta - 41 >= 15 && jeta - 41 <= 28)) {
      //====================================================================== PHI normalization & put R into massive areconoisehe
      //preparation for PHI normalization:
      double sumreconoiseHE0 = 0;
      int nsumreconoiseHE0 = 0;
      double sumreconoiseHE1 = 0;
      int nsumreconoiseHE1 = 0;
      double sumreconoiseHE2 = 0;
      int nsumreconoiseHE2 = 0;
      double sumreconoiseHE3 = 0;
      int nsumreconoiseHE3 = 0;
      double sumreconoiseHE4 = 0;
      int nsumreconoiseHE4 = 0;
      double sumreconoiseHE5 = 0;
      int nsumreconoiseHE5 = 0;
      double sumreconoiseHE6 = 0;
      int nsumreconoiseHE6 = 0;
      for (int jphi = 0; jphi < njphi; jphi++) {
        areconoisehe[0][jeta][jphi] = recNoiseEnergyHE1->GetBinContent(jeta + 1, jphi + 1);
        areconoisehe[1][jeta][jphi] = recNoiseEnergyHE2->GetBinContent(jeta + 1, jphi + 1);
        areconoisehe[2][jeta][jphi] = recNoiseEnergyHE3->GetBinContent(jeta + 1, jphi + 1);
        areconoisehe[3][jeta][jphi] = recNoiseEnergyHE4->GetBinContent(jeta + 1, jphi + 1);
        areconoisehe[4][jeta][jphi] = recNoiseEnergyHE5->GetBinContent(jeta + 1, jphi + 1);
        areconoisehe[5][jeta][jphi] = recNoiseEnergyHE6->GetBinContent(jeta + 1, jphi + 1);
        areconoisehe[6][jeta][jphi] = recNoiseEnergyHE7->GetBinContent(jeta + 1, jphi + 1);

        breconoisehe[0][jeta][jphi] = recNoiseEnergyHE1->GetBinContent(jeta + 1, jphi + 1);
        breconoisehe[1][jeta][jphi] = recNoiseEnergyHE2->GetBinContent(jeta + 1, jphi + 1);
        breconoisehe[2][jeta][jphi] = recNoiseEnergyHE3->GetBinContent(jeta + 1, jphi + 1);
        breconoisehe[3][jeta][jphi] = recNoiseEnergyHE4->GetBinContent(jeta + 1, jphi + 1);
        breconoisehe[4][jeta][jphi] = recNoiseEnergyHE5->GetBinContent(jeta + 1, jphi + 1);
        breconoisehe[5][jeta][jphi] = recNoiseEnergyHE6->GetBinContent(jeta + 1, jphi + 1);
        breconoisehe[6][jeta][jphi] = recNoiseEnergyHE7->GetBinContent(jeta + 1, jphi + 1);

        if (areconoisehe[0][jeta][jphi] != 0.) {
          sumreconoiseHE0 += areconoisehe[0][jeta][jphi];
          ++nsumreconoiseHE0;
        }
        if (areconoisehe[1][jeta][jphi] != 0.) {
          sumreconoiseHE1 += areconoisehe[1][jeta][jphi];
          ++nsumreconoiseHE1;
        }
        if (areconoisehe[2][jeta][jphi] != 0.) {
          sumreconoiseHE2 += areconoisehe[2][jeta][jphi];
          ++nsumreconoiseHE2;
        }
        if (areconoisehe[3][jeta][jphi] != 0.) {
          sumreconoiseHE3 += areconoisehe[3][jeta][jphi];
          ++nsumreconoiseHE3;
        }
        if (areconoisehe[4][jeta][jphi] != 0.) {
          sumreconoiseHE4 += areconoisehe[4][jeta][jphi];
          ++nsumreconoiseHE4;
        }
        if (areconoisehe[5][jeta][jphi] != 0.) {
          sumreconoiseHE5 += areconoisehe[5][jeta][jphi];
          ++nsumreconoiseHE5;
        }
        if (areconoisehe[6][jeta][jphi] != 0.) {
          sumreconoiseHE6 += areconoisehe[6][jeta][jphi];
          ++nsumreconoiseHE6;
        }
      }  // phi

      // PHI normalization for DIF:
      for (int jphi = 0; jphi < njphi; jphi++) {
        if (sumreconoiseHE0 != 0.)
          breconoisehe[0][jeta][jphi] -= (sumreconoiseHE0 / nsumreconoiseHE0);
        if (sumreconoiseHE1 != 0.)
          breconoisehe[1][jeta][jphi] -= (sumreconoiseHE1 / nsumreconoiseHE1);
        if (sumreconoiseHE2 != 0.)
          breconoisehe[2][jeta][jphi] -= (sumreconoiseHE2 / nsumreconoiseHE2);
        if (sumreconoiseHE3 != 0.)
          breconoisehe[3][jeta][jphi] -= (sumreconoiseHE3 / nsumreconoiseHE3);
        if (sumreconoiseHE4 != 0.)
          breconoisehe[4][jeta][jphi] -= (sumreconoiseHE4 / nsumreconoiseHE4);
        if (sumreconoiseHE5 != 0.)
          breconoisehe[5][jeta][jphi] -= (sumreconoiseHE5 / nsumreconoiseHE5);
        if (sumreconoiseHE6 != 0.)
          breconoisehe[6][jeta][jphi] -= (sumreconoiseHE6 / nsumreconoiseHE6);
      }  // phi

      // PHI normalization for R:
      for (int jphi = 0; jphi < njphi; jphi++) {
        if (sumreconoiseHE0 != 0.)
          areconoisehe[0][jeta][jphi] /= (sumreconoiseHE0 / nsumreconoiseHE0);
        if (sumreconoiseHE1 != 0.)
          areconoisehe[1][jeta][jphi] /= (sumreconoiseHE1 / nsumreconoiseHE1);
        if (sumreconoiseHE2 != 0.)
          areconoisehe[2][jeta][jphi] /= (sumreconoiseHE2 / nsumreconoiseHE2);
        if (sumreconoiseHE3 != 0.)
          areconoisehe[3][jeta][jphi] /= (sumreconoiseHE3 / nsumreconoiseHE3);
        if (sumreconoiseHE4 != 0.)
          areconoisehe[4][jeta][jphi] /= (sumreconoiseHE4 / nsumreconoiseHE4);
        if (sumreconoiseHE5 != 0.)
          areconoisehe[5][jeta][jphi] /= (sumreconoiseHE5 / nsumreconoiseHE5);
        if (sumreconoiseHE6 != 0.)
          areconoisehe[6][jeta][jphi] /= (sumreconoiseHE6 / nsumreconoiseHE6);
      }  // phi
    }    //if( (jeta-41 >=
  }      //eta
  //------------------------  2D-eta/phi-plot: R, averaged over depthes
  //======================================================================
  //                                   RRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRR:   Reconoise HE
  //======================================================================
  c2x1->Clear();
  /////////////////
  c2x1->Divide(2, 1);
  c2x1->cd(1);
  TH2F* GefzRreconoiseHE42D = new TH2F("GefzRreconoiseHE42D", "", neta, -41., 41., nphi, 0., 72.);
  TH2F* GefzRreconoiseHE42D0 = new TH2F("GefzRreconoiseHE42D0", "", neta, -41., 41., nphi, 0., 72.);
  TH2F* GefzRreconoiseHE42DF = (TH2F*)GefzRreconoiseHE42D0->Clone("GefzRreconoiseHE42DF");
  for (int i = 0; i < ndepth; i++) {
    for (int jeta = 0; jeta < neta; jeta++) {
      if ((jeta - 41 >= -29 && jeta - 41 <= -16) || (jeta - 41 >= 15 && jeta - 41 <= 28)) {
        for (int jphi = 0; jphi < nphi; jphi++) {
          double ccc1 = areconoisehe[i][jeta][jphi];
          int k2plot = jeta - 41;
          int kkk = k2plot;  //if(k2plot >0 ) kkk=k2plot+1; //-41 +41 !=0
          if (ccc1 != 0.) {
            GefzRreconoiseHE42D->Fill(kkk, jphi, ccc1);
            GefzRreconoiseHE42D0->Fill(kkk, jphi, 1.);
          }
        }
      }
    }
  }
  GefzRreconoiseHE42DF->Divide(GefzRreconoiseHE42D, GefzRreconoiseHE42D0, 1, 1, "B");  // average A
  gPad->SetGridy();
  gPad->SetGridx();  //      gPad->SetLogz();
  GefzRreconoiseHE42DF->SetXTitle("<R>_depth       #eta  \b");
  GefzRreconoiseHE42DF->SetYTitle("      #phi \b");
  GefzRreconoiseHE42DF->Draw("COLZ");

  c2x1->cd(2);
  TH1F* energyhitNoise_HE = (TH1F*)dir->FindObjectAny("h_energyhitNoise_HE");
  energyhitNoise_HE->SetMarkerStyle(20);
  energyhitNoise_HE->SetMarkerSize(0.4);
  energyhitNoise_HE->GetYaxis()->SetLabelSize(0.04);
  energyhitNoise_HE->SetXTitle("energyhitNoise_HE \b");
  energyhitNoise_HE->SetMarkerColor(2);
  energyhitNoise_HE->SetLineColor(0);
  gPad->SetGridy();
  gPad->SetGridx();
  energyhitNoise_HE->Draw("Error");

  /////////////////
  c2x1->Update();
  c2x1->Print("RreconoiseGeneralD2PhiSymmetryHE.png");
  c2x1->Clear();
  // clean-up
  if (GefzRreconoiseHE42D)
    delete GefzRreconoiseHE42D;
  if (GefzRreconoiseHE42D0)
    delete GefzRreconoiseHE42D0;
  if (GefzRreconoiseHE42DF)
    delete GefzRreconoiseHE42DF;
  //====================================================================== 1D plot: R vs phi , averaged over depthes & eta
  //======================================================================
  //cout<<"      1D plot: R vs phi , averaged over depthes & eta *****" <<endl;
  c1x1->Clear();
  /////////////////
  c1x1->Divide(1, 1);
  c1x1->cd(1);
  TH1F* GefzRreconoiseHE41D = new TH1F("GefzRreconoiseHE41D", "", nphi, 0., 72.);
  TH1F* GefzRreconoiseHE41D0 = new TH1F("GefzRreconoiseHE41D0", "", nphi, 0., 72.);
  TH1F* GefzRreconoiseHE41DF = (TH1F*)GefzRreconoiseHE41D0->Clone("GefzRreconoiseHE41DF");
  for (int jphi = 0; jphi < nphi; jphi++) {
    for (int jeta = 0; jeta < neta; jeta++) {
      if ((jeta - 41 >= -29 && jeta - 41 <= -16) || (jeta - 41 >= 15 && jeta - 41 <= 28)) {
        for (int i = 0; i < ndepth; i++) {
          double ccc1 = areconoisehe[i][jeta][jphi];
          if (ccc1 != 0.) {
            GefzRreconoiseHE41D->Fill(jphi, ccc1);
            GefzRreconoiseHE41D0->Fill(jphi, 1.);
          }
        }
      }
    }
  }
  GefzRreconoiseHE41DF->Divide(GefzRreconoiseHE41D, GefzRreconoiseHE41D0, 1, 1, "B");  // R averaged over depthes & eta
  GefzRreconoiseHE41D0->Sumw2();
  //    for (int jphi=1;jphi<73;jphi++) {GefzRreconoiseHE41DF->SetBinError(jphi,0.01);}
  gPad->SetGridy();
  gPad->SetGridx();  //      gPad->SetLogz();
  GefzRreconoiseHE41DF->SetMarkerStyle(20);
  GefzRreconoiseHE41DF->SetMarkerSize(1.4);
  GefzRreconoiseHE41DF->GetZaxis()->SetLabelSize(0.08);
  GefzRreconoiseHE41DF->SetXTitle("#phi  \b");
  GefzRreconoiseHE41DF->SetYTitle("  <R> \b");
  GefzRreconoiseHE41DF->SetZTitle("<R>_PHI  - AllDepthes \b");
  GefzRreconoiseHE41DF->SetMarkerColor(4);
  GefzRreconoiseHE41DF->SetLineColor(
      4);  // GefzRreconoiseHE41DF->SetMinimum(0.8);     //      GefzRreconoiseHE41DF->SetMaximum(1.000);
  GefzRreconoiseHE41DF->Draw("Error");
  /////////////////
  c1x1->Update();
  c1x1->Print("RreconoiseGeneralD1PhiSymmetryHE.png");
  c1x1->Clear();
  // clean-up
  if (GefzRreconoiseHE41D)
    delete GefzRreconoiseHE41D;
  if (GefzRreconoiseHE41D0)
    delete GefzRreconoiseHE41D0;
  if (GefzRreconoiseHE41DF)
    delete GefzRreconoiseHE41DF;

  //========================================================================================== 4
  //======================================================================
  //======================================================================1D plot: R vs phi , different eta,  depth=1
  //cout<<"      1D plot: R vs phi , different eta,  depth=1 *****" <<endl;
  c3x5->Clear();
  /////////////////
  c3x5->Divide(3, 5);
  c3x5->cd(1);
  int kcountHEpositivedirectionReconoise1 = 1;
  TH1F* h2CeffHEpositivedirectionReconoise1 = new TH1F("h2CeffHEpositivedirectionReconoise1", "", nphi, 0., 72.);
  for (int jeta = 0; jeta < njeta; jeta++) {
    // positivedirectionReconoise:
    if (jeta - 41 >= 15 && jeta - 41 <= 28) {
      //	     for (int i=0;i<ndepth;i++) {
      // depth=1
      for (int i = 0; i < 1; i++) {
        TH1F* HEpositivedirectionReconoise1 = (TH1F*)h2CeffHEpositivedirectionReconoise1->Clone("twod1");
        float ccctest = 0;  // to avoid empty massive elements
        for (int jphi = 0; jphi < nphi; jphi++) {
          double ccc1 = areconoisehe[i][jeta][jphi];
          if (ccc1 != 0.) {
            HEpositivedirectionReconoise1->Fill(jphi, ccc1);
            ccctest = 1.;  //HEpositivedirectionReconoise1->SetBinError(i,0.01);
          }
        }  // for jphi
        if (ccctest > 0.) {
          //	  cout<<"444        kcountHEpositivedirectionReconoise1   =     "<<kcountHEpositivedirectionReconoise1  <<"   jeta-41=     "<< jeta-41 <<endl;
          c3x5->cd(kcountHEpositivedirectionReconoise1);
          HEpositivedirectionReconoise1->SetMarkerStyle(20);
          HEpositivedirectionReconoise1->SetMarkerSize(0.4);
          HEpositivedirectionReconoise1->GetYaxis()->SetLabelSize(0.04);
          HEpositivedirectionReconoise1->SetXTitle("HEpositivedirectionReconoise1 \b");
          HEpositivedirectionReconoise1->SetMarkerColor(2);
          HEpositivedirectionReconoise1->SetLineColor(0);
          gPad->SetGridy();
          gPad->SetGridx();
          //	   gPad->SetLogy();
          if (kcountHEpositivedirectionReconoise1 == 1)
            HEpositivedirectionReconoise1->SetXTitle("R for HE+ jeta = 17; depth = 1 \b");
          if (kcountHEpositivedirectionReconoise1 == 2)
            HEpositivedirectionReconoise1->SetXTitle("R for HE+ jeta = 18; depth = 1 \b");
          if (kcountHEpositivedirectionReconoise1 == 3)
            HEpositivedirectionReconoise1->SetXTitle("R for HE+ jeta = 19; depth = 1 \b");
          if (kcountHEpositivedirectionReconoise1 == 4)
            HEpositivedirectionReconoise1->SetXTitle("R for HE+ jeta = 20; depth = 1 \b");
          if (kcountHEpositivedirectionReconoise1 == 5)
            HEpositivedirectionReconoise1->SetXTitle("R for HE+ jeta = 21; depth = 1 \b");
          if (kcountHEpositivedirectionReconoise1 == 6)
            HEpositivedirectionReconoise1->SetXTitle("R for HE+ jeta = 22; depth = 1 \b");
          if (kcountHEpositivedirectionReconoise1 == 7)
            HEpositivedirectionReconoise1->SetXTitle("R for HE+ jeta = 23; depth = 1 \b");
          if (kcountHEpositivedirectionReconoise1 == 8)
            HEpositivedirectionReconoise1->SetXTitle("R for HE+ jeta = 24; depth = 1 \b");
          if (kcountHEpositivedirectionReconoise1 == 9)
            HEpositivedirectionReconoise1->SetXTitle("R for HE+ jeta = 25; depth = 1 \b");
          if (kcountHEpositivedirectionReconoise1 == 10)
            HEpositivedirectionReconoise1->SetXTitle("R for HE+ jeta = 26; depth = 1 \b");
          if (kcountHEpositivedirectionReconoise1 == 11)
            HEpositivedirectionReconoise1->SetXTitle("R for HE+ jeta = 27; depth = 1 \b");
          if (kcountHEpositivedirectionReconoise1 == 12)
            HEpositivedirectionReconoise1->SetXTitle("R for HE+ jeta = 28; depth = 1 \b");
          HEpositivedirectionReconoise1->Draw("Error");
          kcountHEpositivedirectionReconoise1++;
          if (kcountHEpositivedirectionReconoise1 > 12)
            break;  // 4x6 = 24
        }           //ccctest>0

      }  // for i
    }    //if(jeta-41 >= 15 && jeta-41 <= 28
  }      //for jeta
  /////////////////
  c3x5->Update();
  c3x5->Print("RreconoisePositiveDirectionhistD1PhiSymmetryDepth1HE.png");
  c3x5->Clear();
  // clean-up
  if (h2CeffHEpositivedirectionReconoise1)
    delete h2CeffHEpositivedirectionReconoise1;

  //========================================================================================== 5
  //======================================================================
  //======================================================================1D plot: R vs phi , different eta,  depth=2
  //  cout<<"      1D plot: R vs phi , different eta,  depth=2 *****" <<endl;
  c3x5->Clear();
  /////////////////
  c3x5->Divide(3, 5);
  c3x5->cd(1);
  int kcountHEpositivedirectionReconoise2 = 1;
  TH1F* h2CeffHEpositivedirectionReconoise2 = new TH1F("h2CeffHEpositivedirectionReconoise2", "", nphi, 0., 72.);
  for (int jeta = 0; jeta < njeta; jeta++) {
    // positivedirectionReconoise:
    if (jeta - 41 >= 15 && jeta - 41 <= 28) {
      //	     for (int i=0;i<ndepth;i++) {
      // depth=2
      for (int i = 1; i < 2; i++) {
        TH1F* HEpositivedirectionReconoise2 = (TH1F*)h2CeffHEpositivedirectionReconoise2->Clone("twod1");
        float ccctest = 0;  // to avoid empty massive elements
        for (int jphi = 0; jphi < nphi; jphi++) {
          double ccc1 = areconoisehe[i][jeta][jphi];
          if (ccc1 != 0.) {
            HEpositivedirectionReconoise2->Fill(jphi, ccc1);
            ccctest = 1.;  //HEpositivedirectionReconoise2->SetBinError(i,0.01);
          }
        }  // for jphi
        if (ccctest > 0.) {
          //cout<<"555        kcountHEpositivedirectionReconoise2   =     "<<kcountHEpositivedirectionReconoise2  <<"   jeta-41=     "<< jeta-41 <<endl;
          c3x5->cd(kcountHEpositivedirectionReconoise2);
          HEpositivedirectionReconoise2->SetMarkerStyle(20);
          HEpositivedirectionReconoise2->SetMarkerSize(0.4);
          HEpositivedirectionReconoise2->GetYaxis()->SetLabelSize(0.04);
          HEpositivedirectionReconoise2->SetXTitle("HEpositivedirectionReconoise2 \b");
          HEpositivedirectionReconoise2->SetMarkerColor(2);
          HEpositivedirectionReconoise2->SetLineColor(0);
          gPad->SetGridy();
          gPad->SetGridx();
          //	   gPad->SetLogy();
          if (kcountHEpositivedirectionReconoise2 == 1)
            HEpositivedirectionReconoise2->SetXTitle("R for HE+ jeta = 16; depth = 2 \b");
          if (kcountHEpositivedirectionReconoise2 == 2)
            HEpositivedirectionReconoise2->SetXTitle("R for HE+ jeta = 17; depth = 2 \b");
          if (kcountHEpositivedirectionReconoise2 == 3)
            HEpositivedirectionReconoise2->SetXTitle("R for HE+ jeta = 18; depth = 2 \b");
          if (kcountHEpositivedirectionReconoise2 == 4)
            HEpositivedirectionReconoise2->SetXTitle("R for HE+ jeta = 19; depth = 2 \b");
          if (kcountHEpositivedirectionReconoise2 == 5)
            HEpositivedirectionReconoise2->SetXTitle("R for HE+ jeta = 20; depth = 2 \b");
          if (kcountHEpositivedirectionReconoise2 == 6)
            HEpositivedirectionReconoise2->SetXTitle("R for HE+ jeta = 21; depth = 2 \b");
          if (kcountHEpositivedirectionReconoise2 == 7)
            HEpositivedirectionReconoise2->SetXTitle("R for HE+ jeta = 22; depth = 2 \b");
          if (kcountHEpositivedirectionReconoise2 == 8)
            HEpositivedirectionReconoise2->SetXTitle("R for HE+ jeta = 23; depth = 2 \b");
          if (kcountHEpositivedirectionReconoise2 == 9)
            HEpositivedirectionReconoise2->SetXTitle("R for HE+ jeta = 24; depth = 2 \b");
          if (kcountHEpositivedirectionReconoise2 == 10)
            HEpositivedirectionReconoise2->SetXTitle("R for HE+ jeta = 25; depth = 2 \b");
          if (kcountHEpositivedirectionReconoise2 == 11)
            HEpositivedirectionReconoise2->SetXTitle("R for HE+ jeta = 26; depth = 2 \b");
          if (kcountHEpositivedirectionReconoise2 == 12)
            HEpositivedirectionReconoise2->SetXTitle("R for HE+ jeta = 27; depth = 2 \b");
          if (kcountHEpositivedirectionReconoise2 == 13)
            HEpositivedirectionReconoise2->SetXTitle("R for HE+ jeta = 28; depth = 2 \b");
          HEpositivedirectionReconoise2->Draw("Error");
          kcountHEpositivedirectionReconoise2++;
          if (kcountHEpositivedirectionReconoise2 > 13)
            break;  // 4x6 = 24
        }           //ccctest>0

      }  // for i
    }    //if(jeta-41
  }      //for jeta
  /////////////////
  c3x5->Update();
  c3x5->Print("RreconoisePositiveDirectionhistD1PhiSymmetryDepth2HE.png");
  c3x5->Clear();
  // clean-up
  if (h2CeffHEpositivedirectionReconoise2)
    delete h2CeffHEpositivedirectionReconoise2;
  //========================================================================================== 6
  //======================================================================
  //======================================================================1D plot: R vs phi , different eta,  depth=3
  //cout<<"      1D plot: R vs phi , different eta,  depth=3 *****" <<endl;
  c3x5->Clear();
  /////////////////
  c3x5->Divide(3, 5);
  c3x5->cd(1);
  int kcountHEpositivedirectionReconoise3 = 1;
  TH1F* h2CeffHEpositivedirectionReconoise3 = new TH1F("h2CeffHEpositivedirectionReconoise3", "", nphi, 0., 72.);
  for (int jeta = 0; jeta < njeta; jeta++) {
    // positivedirectionReconoise:
    if (jeta - 41 >= 15 && jeta - 41 <= 28) {
      //	     for (int i=0;i<ndepth;i++) {
      // depth=3
      for (int i = 2; i < 3; i++) {
        TH1F* HEpositivedirectionReconoise3 = (TH1F*)h2CeffHEpositivedirectionReconoise3->Clone("twod1");
        float ccctest = 0;  // to avoid empty massive elements
        for (int jphi = 0; jphi < nphi; jphi++) {
          double ccc1 = areconoisehe[i][jeta][jphi];
          if (ccc1 != 0.) {
            HEpositivedirectionReconoise3->Fill(jphi, ccc1);
            ccctest = 1.;  //HEpositivedirectionReconoise3->SetBinError(i,0.01);
          }
        }  // for jphi
        if (ccctest > 0.) {
          //cout<<"666        kcountHEpositivedirectionReconoise3   =     "<<kcountHEpositivedirectionReconoise3  <<"   jeta-41=     "<< jeta-41 <<endl;
          c3x5->cd(kcountHEpositivedirectionReconoise3);
          HEpositivedirectionReconoise3->SetMarkerStyle(20);
          HEpositivedirectionReconoise3->SetMarkerSize(0.4);
          HEpositivedirectionReconoise3->GetYaxis()->SetLabelSize(0.04);
          HEpositivedirectionReconoise3->SetXTitle("HEpositivedirectionReconoise3 \b");
          HEpositivedirectionReconoise3->SetMarkerColor(2);
          HEpositivedirectionReconoise3->SetLineColor(0);
          gPad->SetGridy();
          gPad->SetGridx();
          //	   gPad->SetLogy();
          if (kcountHEpositivedirectionReconoise3 == 1)
            HEpositivedirectionReconoise3->SetXTitle("R for HE+ jeta = 16; depth = 3 \b");
          if (kcountHEpositivedirectionReconoise3 == 2)
            HEpositivedirectionReconoise3->SetXTitle("R for HE+ jeta = 17; depth = 3 \b");
          if (kcountHEpositivedirectionReconoise3 == 3)
            HEpositivedirectionReconoise3->SetXTitle("R for HE+ jeta = 18; depth = 3 \b");
          if (kcountHEpositivedirectionReconoise3 == 4)
            HEpositivedirectionReconoise3->SetXTitle("R for HE+ jeta = 19; depth = 3 \b");
          if (kcountHEpositivedirectionReconoise3 == 5)
            HEpositivedirectionReconoise3->SetXTitle("R for HE+ jeta = 20; depth = 3 \b");
          if (kcountHEpositivedirectionReconoise3 == 6)
            HEpositivedirectionReconoise3->SetXTitle("R for HE+ jeta = 21; depth = 3 \b");
          if (kcountHEpositivedirectionReconoise3 == 7)
            HEpositivedirectionReconoise3->SetXTitle("R for HE+ jeta = 22; depth = 3 \b");
          if (kcountHEpositivedirectionReconoise3 == 8)
            HEpositivedirectionReconoise3->SetXTitle("R for HE+ jeta = 23; depth = 3 \b");
          if (kcountHEpositivedirectionReconoise3 == 9)
            HEpositivedirectionReconoise3->SetXTitle("R for HE+ jeta = 24; depth = 3 \b");
          if (kcountHEpositivedirectionReconoise3 == 10)
            HEpositivedirectionReconoise3->SetXTitle("R for HE+ jeta = 25; depth = 3 \b");
          if (kcountHEpositivedirectionReconoise3 == 11)
            HEpositivedirectionReconoise3->SetXTitle("R for HE+ jeta = 26; depth = 3 \b");
          if (kcountHEpositivedirectionReconoise3 == 12)
            HEpositivedirectionReconoise3->SetXTitle("R for HE+ jeta = 27; depth = 3 \b");
          if (kcountHEpositivedirectionReconoise3 == 13)
            HEpositivedirectionReconoise3->SetXTitle("R for HE+ jeta = 28; depth = 3 \b");
          HEpositivedirectionReconoise3->Draw("Error");
          kcountHEpositivedirectionReconoise3++;
          if (kcountHEpositivedirectionReconoise3 > 13)
            break;  // 4x6 = 24
        }           //ccctest>0

      }  // for i
    }    //if(jeta-41 >=
  }      //for jeta
  /////////////////
  c3x5->Update();
  c3x5->Print("RreconoisePositiveDirectionhistD1PhiSymmetryDepth3HE.png");
  c3x5->Clear();
  // clean-up
  if (h2CeffHEpositivedirectionReconoise3)
    delete h2CeffHEpositivedirectionReconoise3;
  //========================================================================================== 7
  //======================================================================
  //======================================================================1D plot: R vs phi , different eta,  depth=4
  //cout<<"      1D plot: R vs phi , different eta,  depth=4 *****" <<endl;
  c3x5->Clear();
  /////////////////
  c3x5->Divide(3, 5);
  c3x5->cd(1);
  int kcountHEpositivedirectionReconoise4 = 1;
  TH1F* h2CeffHEpositivedirectionReconoise4 = new TH1F("h2CeffHEpositivedirectionReconoise4", "", nphi, 0., 72.);

  for (int jeta = 0; jeta < njeta; jeta++) {
    // positivedirectionReconoise:
    if (jeta - 41 >= 15 && jeta - 41 <= 28) {
      //	     for (int i=0;i<ndepth;i++) {
      // depth=4
      for (int i = 3; i < 4; i++) {
        TH1F* HEpositivedirectionReconoise4 = (TH1F*)h2CeffHEpositivedirectionReconoise4->Clone("twod1");

        float ccctest = 0;  // to avoid empty massive elements
        for (int jphi = 0; jphi < nphi; jphi++) {
          double ccc1 = areconoisehe[i][jeta][jphi];
          if (ccc1 != 0.) {
            HEpositivedirectionReconoise4->Fill(jphi, ccc1);
            ccctest = 1.;  //HEpositivedirectionReconoise4->SetBinError(i,0.01);
          }
        }  // for jphi
        if (ccctest > 0.) {
          //cout<<"777        kcountHEpositivedirectionReconoise4   =     "<<kcountHEpositivedirectionReconoise4  <<"   jeta-41=     "<< jeta-41 <<endl;
          c3x5->cd(kcountHEpositivedirectionReconoise4);
          HEpositivedirectionReconoise4->SetMarkerStyle(20);
          HEpositivedirectionReconoise4->SetMarkerSize(0.4);
          HEpositivedirectionReconoise4->GetYaxis()->SetLabelSize(0.04);
          HEpositivedirectionReconoise4->SetXTitle("HEpositivedirectionReconoise4 \b");
          HEpositivedirectionReconoise4->SetMarkerColor(2);
          HEpositivedirectionReconoise4->SetLineColor(0);
          gPad->SetGridy();
          gPad->SetGridx();
          //	   gPad->SetLogy();
          if (kcountHEpositivedirectionReconoise4 == 1)
            HEpositivedirectionReconoise4->SetXTitle("R for HE+ jeta = 15; depth = 4 \b");
          if (kcountHEpositivedirectionReconoise4 == 2)
            HEpositivedirectionReconoise4->SetXTitle("R for HE+ jeta = 17; depth = 4 \b");
          if (kcountHEpositivedirectionReconoise4 == 3)
            HEpositivedirectionReconoise4->SetXTitle("R for HE+ jeta = 18; depth = 4 \b");
          if (kcountHEpositivedirectionReconoise4 == 4)
            HEpositivedirectionReconoise4->SetXTitle("R for HE+ jeta = 19; depth = 4 \b");
          if (kcountHEpositivedirectionReconoise4 == 5)
            HEpositivedirectionReconoise4->SetXTitle("R for HE+ jeta = 20; depth = 4 \b");
          if (kcountHEpositivedirectionReconoise4 == 6)
            HEpositivedirectionReconoise4->SetXTitle("R for HE+ jeta = 21; depth = 4 \b");
          if (kcountHEpositivedirectionReconoise4 == 7)
            HEpositivedirectionReconoise4->SetXTitle("R for HE+ jeta = 22; depth = 4 \b");
          if (kcountHEpositivedirectionReconoise4 == 8)
            HEpositivedirectionReconoise4->SetXTitle("R for HE+ jeta = 23; depth = 4 \b");
          if (kcountHEpositivedirectionReconoise4 == 9)
            HEpositivedirectionReconoise4->SetXTitle("R for HE+ jeta = 24; depth = 4 \b");
          if (kcountHEpositivedirectionReconoise4 == 10)
            HEpositivedirectionReconoise4->SetXTitle("R for HE+ jeta = 25; depth = 4 \b");
          if (kcountHEpositivedirectionReconoise4 == 11)
            HEpositivedirectionReconoise4->SetXTitle("R for HE+ jeta = 26; depth = 4 \b");
          if (kcountHEpositivedirectionReconoise4 == 12)
            HEpositivedirectionReconoise4->SetXTitle("R for HE+ jeta = 27; depth = 4 \b");
          HEpositivedirectionReconoise4->Draw("Error");
          kcountHEpositivedirectionReconoise4++;
          if (kcountHEpositivedirectionReconoise4 > 12)
            break;  // 4x6 = 24
        }           //ccctest>0

      }  // for i
    }    //if(jeta-41 >=  -29 && jeta-41 <= -16)
  }      //for jeta
  /////////////////
  c3x5->Update();
  c3x5->Print("RreconoisePositiveDirectionhistD1PhiSymmetryDepth4HE.png");
  c3x5->Clear();
  // clean-up
  if (h2CeffHEpositivedirectionReconoise4)
    delete h2CeffHEpositivedirectionReconoise4;
  //========================================================================================== 8
  //======================================================================
  //======================================================================1D plot: R vs phi , different eta,  depth=5
  //cout<<"      1D plot: R vs phi , different eta,  depth=5 *****" <<endl;
  c3x5->Clear();
  /////////////////
  c3x5->Divide(3, 5);
  c3x5->cd(1);
  int kcountHEpositivedirectionReconoise5 = 1;
  TH1F* h2CeffHEpositivedirectionReconoise5 = new TH1F("h2CeffHEpositivedirectionReconoise5", "", nphi, 0., 72.);

  for (int jeta = 0; jeta < njeta; jeta++) {
    // positivedirectionReconoise:
    if (jeta - 41 >= 15 && jeta - 41 <= 28) {
      //	     for (int i=0;i<ndepth;i++) {
      // depth=5
      for (int i = 4; i < 5; i++) {
        TH1F* HEpositivedirectionReconoise5 = (TH1F*)h2CeffHEpositivedirectionReconoise5->Clone("twod1");

        float ccctest = 0;  // to avoid empty massive elements
        for (int jphi = 0; jphi < nphi; jphi++) {
          //	       cout<<"888  initial      kcountHEpositivedirectionReconoise5   =     "<<kcountHEpositivedirectionReconoise5  <<"   jeta-41=     "<< jeta-41 <<"   jphi=     "<< jphi <<"   areconoisehe[i][jeta][jphi]=     "<< areconoisehe[i][jeta][jphi] <<"  depth=     "<< i <<endl;

          double ccc1 = areconoisehe[i][jeta][jphi];
          if (ccc1 != 0.) {
            HEpositivedirectionReconoise5->Fill(jphi, ccc1);
            ccctest = 1.;  //HEpositivedirectionReconoise5->SetBinError(i,0.01);
          }
        }  // for jphi
        if (ccctest > 0.) {
          //cout<<"888        kcountHEpositivedirectionReconoise5   =     "<<kcountHEpositivedirectionReconoise5  <<"   jeta-41=     "<< jeta-41 <<endl;
          c3x5->cd(kcountHEpositivedirectionReconoise5);
          HEpositivedirectionReconoise5->SetMarkerStyle(20);
          HEpositivedirectionReconoise5->SetMarkerSize(0.4);
          HEpositivedirectionReconoise5->GetYaxis()->SetLabelSize(0.04);
          HEpositivedirectionReconoise5->SetXTitle("HEpositivedirectionReconoise5 \b");
          HEpositivedirectionReconoise5->SetMarkerColor(2);
          HEpositivedirectionReconoise5->SetLineColor(0);
          gPad->SetGridy();
          gPad->SetGridx();
          //	   gPad->SetLogy();
          if (kcountHEpositivedirectionReconoise5 == 1)
            HEpositivedirectionReconoise5->SetXTitle("R for HE+ jeta = 17; depth = 5 \b");
          if (kcountHEpositivedirectionReconoise5 == 2)
            HEpositivedirectionReconoise5->SetXTitle("R for HE+ jeta = 18; depth = 5 \b");
          if (kcountHEpositivedirectionReconoise5 == 3)
            HEpositivedirectionReconoise5->SetXTitle("R for HE+ jeta = 19; depth = 5 \b");
          if (kcountHEpositivedirectionReconoise5 == 4)
            HEpositivedirectionReconoise5->SetXTitle("R for HE+ jeta = 20; depth = 5 \b");
          if (kcountHEpositivedirectionReconoise5 == 5)
            HEpositivedirectionReconoise5->SetXTitle("R for HE+ jeta = 21; depth = 5 \b");
          if (kcountHEpositivedirectionReconoise5 == 6)
            HEpositivedirectionReconoise5->SetXTitle("R for HE+ jeta = 22; depth = 5 \b");
          if (kcountHEpositivedirectionReconoise5 == 7)
            HEpositivedirectionReconoise5->SetXTitle("R for HE+ jeta = 23; depth = 5 \b");
          if (kcountHEpositivedirectionReconoise5 == 8)
            HEpositivedirectionReconoise5->SetXTitle("R for HE+ jeta = 24; depth = 5 \b");
          if (kcountHEpositivedirectionReconoise5 == 9)
            HEpositivedirectionReconoise5->SetXTitle("R for HE+ jeta = 25; depth = 5 \b");
          if (kcountHEpositivedirectionReconoise5 == 10)
            HEpositivedirectionReconoise5->SetXTitle("R for HE+ jeta = 26; depth = 5 \b");
          if (kcountHEpositivedirectionReconoise5 == 11)
            HEpositivedirectionReconoise5->SetXTitle("R for HE+ jeta = 27; depth = 5 \b");
          HEpositivedirectionReconoise5->Draw("Error");
          kcountHEpositivedirectionReconoise5++;
          if (kcountHEpositivedirectionReconoise5 > 11)
            break;  // 4x6 = 24
        }           //ccctest>0

      }  // for i
    }    //if(jeta-41 >=  -29 && jeta-41 <= -16)
  }      //for jeta
  /////////////////
  c3x5->Update();
  c3x5->Print("RreconoisePositiveDirectionhistD1PhiSymmetryDepth5HE.png");
  c3x5->Clear();
  // clean-up
  if (h2CeffHEpositivedirectionReconoise5)
    delete h2CeffHEpositivedirectionReconoise5;
  //========================================================================================== 9
  //======================================================================
  //======================================================================1D plot: R vs phi , different eta,  depth=6
  //cout<<"      1D plot: R vs phi , different eta,  depth=6 *****" <<endl;
  c3x5->Clear();
  /////////////////
  c3x5->Divide(3, 5);
  c3x5->cd(1);
  int kcountHEpositivedirectionReconoise6 = 1;
  TH1F* h2CeffHEpositivedirectionReconoise6 = new TH1F("h2CeffHEpositivedirectionReconoise6", "", nphi, 0., 72.);

  for (int jeta = 0; jeta < njeta; jeta++) {
    // positivedirectionReconoise:
    if (jeta - 41 >= 15 && jeta - 41 <= 28) {
      //	     for (int i=0;i<ndepth;i++) {
      // depth=6
      for (int i = 5; i < 6; i++) {
        TH1F* HEpositivedirectionReconoise6 = (TH1F*)h2CeffHEpositivedirectionReconoise6->Clone("twod1");

        float ccctest = 0;  // to avoid empty massive elements
        for (int jphi = 0; jphi < nphi; jphi++) {
          double ccc1 = areconoisehe[i][jeta][jphi];
          if (ccc1 != 0.) {
            HEpositivedirectionReconoise6->Fill(jphi, ccc1);
            ccctest = 1.;  //HEpositivedirectionReconoise6->SetBinError(i,0.01);
          }
        }  // for jphi
        if (ccctest > 0.) {
          //cout<<"999        kcountHEpositivedirectionReconoise6   =     "<<kcountHEpositivedirectionReconoise6  <<"   jeta-41=     "<< jeta-41 <<endl;
          c3x5->cd(kcountHEpositivedirectionReconoise6);
          HEpositivedirectionReconoise6->SetMarkerStyle(20);
          HEpositivedirectionReconoise6->SetMarkerSize(0.4);
          HEpositivedirectionReconoise6->GetYaxis()->SetLabelSize(0.04);
          HEpositivedirectionReconoise6->SetXTitle("HEpositivedirectionReconoise6 \b");
          HEpositivedirectionReconoise6->SetMarkerColor(2);
          HEpositivedirectionReconoise6->SetLineColor(0);
          gPad->SetGridy();
          gPad->SetGridx();
          //	   gPad->SetLogy();
          if (kcountHEpositivedirectionReconoise6 == 1)
            HEpositivedirectionReconoise6->SetXTitle("R for HE+ jeta = 18; depth = 6 \b");
          if (kcountHEpositivedirectionReconoise6 == 2)
            HEpositivedirectionReconoise6->SetXTitle("R for HE+ jeta = 19; depth = 6 \b");
          if (kcountHEpositivedirectionReconoise6 == 3)
            HEpositivedirectionReconoise6->SetXTitle("R for HE+ jeta = 20; depth = 6 \b");
          if (kcountHEpositivedirectionReconoise6 == 4)
            HEpositivedirectionReconoise6->SetXTitle("R for HE+ jeta = 21; depth = 6 \b");
          if (kcountHEpositivedirectionReconoise6 == 5)
            HEpositivedirectionReconoise6->SetXTitle("R for HE+ jeta = 22; depth = 6 \b");
          if (kcountHEpositivedirectionReconoise6 == 6)
            HEpositivedirectionReconoise6->SetXTitle("R for HE+ jeta = 23; depth = 6 \b");
          if (kcountHEpositivedirectionReconoise6 == 7)
            HEpositivedirectionReconoise6->SetXTitle("R for HE+ jeta = 24; depth = 6 \b");
          if (kcountHEpositivedirectionReconoise6 == 8)
            HEpositivedirectionReconoise6->SetXTitle("R for HE+ jeta = 25; depth = 6 \b");
          if (kcountHEpositivedirectionReconoise6 == 9)
            HEpositivedirectionReconoise6->SetXTitle("R for HE+ jeta = 26; depth = 6 \b");
          if (kcountHEpositivedirectionReconoise6 == 10)
            HEpositivedirectionReconoise6->SetXTitle("R for HE+ jeta = 27; depth = 6 \b");
          HEpositivedirectionReconoise6->Draw("Error");
          kcountHEpositivedirectionReconoise6++;
          if (kcountHEpositivedirectionReconoise6 > 10)
            break;  // 4x6 = 24
        }           //ccctest>0

      }  // for i
    }    //if(jeta-41 >=  -29 && jeta-41 <= -16)
  }      //for jeta
  /////////////////
  c3x5->Update();
  c3x5->Print("RreconoisePositiveDirectionhistD1PhiSymmetryDepth6HE.png");
  c3x5->Clear();
  // clean-up
  if (h2CeffHEpositivedirectionReconoise6)
    delete h2CeffHEpositivedirectionReconoise6;
  //========================================================================================== 10
  //======================================================================
  //======================================================================1D plot: R vs phi , different eta,  depth=7
  //cout<<"      1D plot: R vs phi , different eta,  depth=7 *****" <<endl;
  c3x5->Clear();
  /////////////////
  c3x5->Divide(3, 5);
  c3x5->cd(1);
  int kcountHEpositivedirectionReconoise7 = 1;
  TH1F* h2CeffHEpositivedirectionReconoise7 = new TH1F("h2CeffHEpositivedirectionReconoise7", "", nphi, 0., 72.);

  for (int jeta = 0; jeta < njeta; jeta++) {
    // positivedirectionReconoise:
    if (jeta - 41 >= 15 && jeta - 41 <= 28) {
      //	     for (int i=0;i<ndepth;i++) {
      // depth=7
      for (int i = 6; i < 7; i++) {
        TH1F* HEpositivedirectionReconoise7 = (TH1F*)h2CeffHEpositivedirectionReconoise7->Clone("twod1");

        float ccctest = 0;  // to avoid empty massive elements
        for (int jphi = 0; jphi < nphi; jphi++) {
          double ccc1 = areconoisehe[i][jeta][jphi];
          if (ccc1 != 0.) {
            HEpositivedirectionReconoise7->Fill(jphi, ccc1);
            ccctest = 1.;  //HEpositivedirectionReconoise7->SetBinError(i,0.01);
          }
        }  // for jphi
        if (ccctest > 0.) {
          //cout<<"1010       kcountHEpositivedirectionReconoise7   =     "<<kcountHEpositivedirectionReconoise7  <<"   jeta-41=     "<< jeta-41 <<endl;
          c3x5->cd(kcountHEpositivedirectionReconoise7);
          HEpositivedirectionReconoise7->SetMarkerStyle(20);
          HEpositivedirectionReconoise7->SetMarkerSize(0.4);
          HEpositivedirectionReconoise7->GetYaxis()->SetLabelSize(0.04);
          HEpositivedirectionReconoise7->SetXTitle("HEpositivedirectionReconoise7 \b");
          HEpositivedirectionReconoise7->SetMarkerColor(2);
          HEpositivedirectionReconoise7->SetLineColor(0);
          gPad->SetGridy();
          gPad->SetGridx();
          //	   gPad->SetLogy();
          if (kcountHEpositivedirectionReconoise7 == 1)
            HEpositivedirectionReconoise7->SetXTitle("R for HE+ jeta = 25; depth = 7 \b");
          if (kcountHEpositivedirectionReconoise7 == 2)
            HEpositivedirectionReconoise7->SetXTitle("R for HE+ jeta = 26; depth = 7 \b");
          if (kcountHEpositivedirectionReconoise7 == 3)
            HEpositivedirectionReconoise7->SetXTitle("R for HE+ jeta = 27; depth = 7 \b");
          HEpositivedirectionReconoise7->Draw("Error");
          kcountHEpositivedirectionReconoise7++;
          if (kcountHEpositivedirectionReconoise7 > 3)
            break;  //
        }           //ccctest>0

      }  // for i
    }    //if(jeta-41 >=  -29 && jeta-41 <= -16)
  }      //for jeta
  /////////////////
  c3x5->Update();
  c3x5->Print("RreconoisePositiveDirectionhistD1PhiSymmetryDepth7HE.png");
  c3x5->Clear();
  // clean-up
  if (h2CeffHEpositivedirectionReconoise7)
    delete h2CeffHEpositivedirectionReconoise7;

  //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  //========================================================================================== 1114
  //======================================================================
  //======================================================================1D plot: R vs phi , different eta,  depth=1
  //cout<<"      1D plot: R vs phi , different eta,  depth=1 *****" <<endl;
  c3x5->Clear();
  /////////////////
  c3x5->Divide(3, 5);
  c3x5->cd(1);
  int kcountHEnegativedirectionReconoise1 = 1;
  TH1F* h2CeffHEnegativedirectionReconoise1 = new TH1F("h2CeffHEnegativedirectionReconoise1", "", nphi, 0., 72.);
  for (int jeta = 0; jeta < njeta; jeta++) {
    // negativedirectionReconoise:
    if (jeta - 41 >= -29 && jeta - 41 <= -16) {
      //	     for (int i=0;i<ndepth;i++) {
      // depth=1
      for (int i = 0; i < 1; i++) {
        TH1F* HEnegativedirectionReconoise1 = (TH1F*)h2CeffHEnegativedirectionReconoise1->Clone("twod1");
        float ccctest = 0;  // to avoid empty massive elements
        for (int jphi = 0; jphi < nphi; jphi++) {
          double ccc1 = areconoisehe[i][jeta][jphi];
          if (ccc1 != 0.) {
            HEnegativedirectionReconoise1->Fill(jphi, ccc1);
            ccctest = 1.;  //HEnegativedirectionReconoise1->SetBinError(i,0.01);
          }
        }  // for jphi
        if (ccctest > 0.) {
          //	  cout<<"444        kcountHEnegativedirectionReconoise1   =     "<<kcountHEnegativedirectionReconoise1  <<"   jeta-41=     "<< jeta-41 <<endl;
          c3x5->cd(kcountHEnegativedirectionReconoise1);
          HEnegativedirectionReconoise1->SetMarkerStyle(20);
          HEnegativedirectionReconoise1->SetMarkerSize(0.4);
          HEnegativedirectionReconoise1->GetYaxis()->SetLabelSize(0.04);
          HEnegativedirectionReconoise1->SetXTitle("HEnegativedirectionReconoise1 \b");
          HEnegativedirectionReconoise1->SetMarkerColor(2);
          HEnegativedirectionReconoise1->SetLineColor(0);
          gPad->SetGridy();
          gPad->SetGridx();
          //	   gPad->SetLogy();
          if (kcountHEnegativedirectionReconoise1 == 1)
            HEnegativedirectionReconoise1->SetXTitle("R for HE- jeta =-29; depth = 1 \b");
          if (kcountHEnegativedirectionReconoise1 == 2)
            HEnegativedirectionReconoise1->SetXTitle("R for HE- jeta =-28; depth = 1 \b");
          if (kcountHEnegativedirectionReconoise1 == 3)
            HEnegativedirectionReconoise1->SetXTitle("R for HE- jeta =-27; depth = 1 \b");
          if (kcountHEnegativedirectionReconoise1 == 4)
            HEnegativedirectionReconoise1->SetXTitle("R for HE- jeta =-26; depth = 1 \b");
          if (kcountHEnegativedirectionReconoise1 == 5)
            HEnegativedirectionReconoise1->SetXTitle("R for HE- jeta =-25; depth = 1 \b");
          if (kcountHEnegativedirectionReconoise1 == 6)
            HEnegativedirectionReconoise1->SetXTitle("R for HE- jeta =-24; depth = 1 \b");
          if (kcountHEnegativedirectionReconoise1 == 7)
            HEnegativedirectionReconoise1->SetXTitle("R for HE- jeta =-23; depth = 1 \b");
          if (kcountHEnegativedirectionReconoise1 == 8)
            HEnegativedirectionReconoise1->SetXTitle("R for HE- jeta =-22; depth = 1 \b");
          if (kcountHEnegativedirectionReconoise1 == 9)
            HEnegativedirectionReconoise1->SetXTitle("R for HE- jeta =-21; depth = 1 \b");
          if (kcountHEnegativedirectionReconoise1 == 10)
            HEnegativedirectionReconoise1->SetXTitle("R for HE- jeta =-20; depth = 1 \b");
          if (kcountHEnegativedirectionReconoise1 == 11)
            HEnegativedirectionReconoise1->SetXTitle("R for HE- jeta =-19; depth = 1 \b");
          if (kcountHEnegativedirectionReconoise1 == 12)
            HEnegativedirectionReconoise1->SetXTitle("R for HE- jeta =-18; depth = 1 \b");
          HEnegativedirectionReconoise1->Draw("Error");
          kcountHEnegativedirectionReconoise1++;
          if (kcountHEnegativedirectionReconoise1 > 12)
            break;  // 4x6 = 24
        }           //ccctest>0

      }  // for i
    }    //if(jeta-41  >= -29 && jeta-41 <= -16)
  }      //for jeta
  /////////////////
  c3x5->Update();
  c3x5->Print("RreconoiseNegativeDirectionhistD1PhiSymmetryDepth1HE.png");
  c3x5->Clear();
  // clean-up
  if (h2CeffHEnegativedirectionReconoise1)
    delete h2CeffHEnegativedirectionReconoise1;

  //========================================================================================== 1115
  //======================================================================
  //======================================================================1D plot: R vs phi , different eta,  depth=2
  //  cout<<"      1D plot: R vs phi , different eta,  depth=2 *****" <<endl;
  c3x5->Clear();
  /////////////////
  c3x5->Divide(3, 5);
  c3x5->cd(1);
  int kcountHEnegativedirectionReconoise2 = 1;
  TH1F* h2CeffHEnegativedirectionReconoise2 = new TH1F("h2CeffHEnegativedirectionReconoise2", "", nphi, 0., 72.);
  for (int jeta = 0; jeta < njeta; jeta++) {
    // negativedirectionReconoise:
    if (jeta - 41 >= -29 && jeta - 41 <= -16) {
      //	     for (int i=0;i<ndepth;i++) {
      // depth=2
      for (int i = 1; i < 2; i++) {
        TH1F* HEnegativedirectionReconoise2 = (TH1F*)h2CeffHEnegativedirectionReconoise2->Clone("twod1");
        float ccctest = 0;  // to avoid empty massive elements
        for (int jphi = 0; jphi < nphi; jphi++) {
          double ccc1 = areconoisehe[i][jeta][jphi];
          if (ccc1 != 0.) {
            HEnegativedirectionReconoise2->Fill(jphi, ccc1);
            ccctest = 1.;  //HEnegativedirectionReconoise2->SetBinError(i,0.01);
          }
        }  // for jphi
        if (ccctest > 0.) {
          //cout<<"555        kcountHEnegativedirectionReconoise2   =     "<<kcountHEnegativedirectionReconoise2  <<"   jeta-41=     "<< jeta-41 <<endl;
          c3x5->cd(kcountHEnegativedirectionReconoise2);
          HEnegativedirectionReconoise2->SetMarkerStyle(20);
          HEnegativedirectionReconoise2->SetMarkerSize(0.4);
          HEnegativedirectionReconoise2->GetYaxis()->SetLabelSize(0.04);
          HEnegativedirectionReconoise2->SetXTitle("HEnegativedirectionReconoise2 \b");
          HEnegativedirectionReconoise2->SetMarkerColor(2);
          HEnegativedirectionReconoise2->SetLineColor(0);
          gPad->SetGridy();
          gPad->SetGridx();
          //	   gPad->SetLogy();
          if (kcountHEnegativedirectionReconoise2 == 1)
            HEnegativedirectionReconoise2->SetXTitle("R for HE- jeta =-29; depth = 2 \b");
          if (kcountHEnegativedirectionReconoise2 == 2)
            HEnegativedirectionReconoise2->SetXTitle("R for HE- jeta =-28; depth = 2 \b");
          if (kcountHEnegativedirectionReconoise2 == 3)
            HEnegativedirectionReconoise2->SetXTitle("R for HE- jeta =-27; depth = 2 \b");
          if (kcountHEnegativedirectionReconoise2 == 4)
            HEnegativedirectionReconoise2->SetXTitle("R for HE- jeta =-26; depth = 2 \b");
          if (kcountHEnegativedirectionReconoise2 == 5)
            HEnegativedirectionReconoise2->SetXTitle("R for HE- jeta =-25; depth = 2 \b");
          if (kcountHEnegativedirectionReconoise2 == 6)
            HEnegativedirectionReconoise2->SetXTitle("R for HE- jeta =-24; depth = 2 \b");
          if (kcountHEnegativedirectionReconoise2 == 7)
            HEnegativedirectionReconoise2->SetXTitle("R for HE- jeta =-23; depth = 2 \b");
          if (kcountHEnegativedirectionReconoise2 == 8)
            HEnegativedirectionReconoise2->SetXTitle("R for HE- jeta =-22; depth = 2 \b");
          if (kcountHEnegativedirectionReconoise2 == 9)
            HEnegativedirectionReconoise2->SetXTitle("R for HE- jeta =-21; depth = 2 \b");
          if (kcountHEnegativedirectionReconoise2 == 10)
            HEnegativedirectionReconoise2->SetXTitle("R for HE- jeta =-20; depth = 2 \b");
          if (kcountHEnegativedirectionReconoise2 == 11)
            HEnegativedirectionReconoise2->SetXTitle("R for HE- jeta =-19; depth = 2 \b");
          if (kcountHEnegativedirectionReconoise2 == 12)
            HEnegativedirectionReconoise2->SetXTitle("R for HE- jeta =-18; depth = 2 \b");
          if (kcountHEnegativedirectionReconoise2 == 13)
            HEnegativedirectionReconoise2->SetXTitle("R for HE- jeta =-17; depth = 2 \b");
          HEnegativedirectionReconoise2->Draw("Error");
          kcountHEnegativedirectionReconoise2++;
          if (kcountHEnegativedirectionReconoise2 > 13)
            break;  // 4x6 = 24
        }           //ccctest>0

      }  // for i
    }    //if(jeta-41  >= -29 && jeta-41 <= -16)
  }      //for jeta
  /////////////////
  c3x5->Update();
  c3x5->Print("RreconoiseNegativeDirectionhistD1PhiSymmetryDepth2HE.png");
  c3x5->Clear();
  // clean-up
  if (h2CeffHEnegativedirectionReconoise2)
    delete h2CeffHEnegativedirectionReconoise2;
  //========================================================================================== 1116
  //======================================================================
  //======================================================================1D plot: R vs phi , different eta,  depth=3
  //cout<<"      1D plot: R vs phi , different eta,  depth=3 *****" <<endl;
  c3x5->Clear();
  /////////////////
  c3x5->Divide(3, 5);
  c3x5->cd(1);
  int kcountHEnegativedirectionReconoise3 = 1;
  TH1F* h2CeffHEnegativedirectionReconoise3 = new TH1F("h2CeffHEnegativedirectionReconoise3", "", nphi, 0., 72.);
  for (int jeta = 0; jeta < njeta; jeta++) {
    // negativedirectionReconoise:
    if (jeta - 41 >= -29 && jeta - 41 <= -16) {
      //	     for (int i=0;i<ndepth;i++) {
      // depth=3
      for (int i = 2; i < 3; i++) {
        TH1F* HEnegativedirectionReconoise3 = (TH1F*)h2CeffHEnegativedirectionReconoise3->Clone("twod1");
        float ccctest = 0;  // to avoid empty massive elements
        for (int jphi = 0; jphi < nphi; jphi++) {
          double ccc1 = areconoisehe[i][jeta][jphi];
          if (ccc1 != 0.) {
            HEnegativedirectionReconoise3->Fill(jphi, ccc1);
            ccctest = 1.;  //HEnegativedirectionReconoise3->SetBinError(i,0.01);
          }
        }  // for jphi
        if (ccctest > 0.) {
          //cout<<"666        kcountHEnegativedirectionReconoise3   =     "<<kcountHEnegativedirectionReconoise3  <<"   jeta-41=     "<< jeta-41 <<endl;
          c3x5->cd(kcountHEnegativedirectionReconoise3);
          HEnegativedirectionReconoise3->SetMarkerStyle(20);
          HEnegativedirectionReconoise3->SetMarkerSize(0.4);
          HEnegativedirectionReconoise3->GetYaxis()->SetLabelSize(0.04);
          HEnegativedirectionReconoise3->SetXTitle("HEnegativedirectionReconoise3 \b");
          HEnegativedirectionReconoise3->SetMarkerColor(2);
          HEnegativedirectionReconoise3->SetLineColor(0);
          gPad->SetGridy();
          gPad->SetGridx();
          //	   gPad->SetLogy();
          if (kcountHEnegativedirectionReconoise3 == 1)
            HEnegativedirectionReconoise3->SetXTitle("R for HE- jeta =-29; depth = 3 \b");
          if (kcountHEnegativedirectionReconoise3 == 2)
            HEnegativedirectionReconoise3->SetXTitle("R for HE- jeta =-28; depth = 3 \b");
          if (kcountHEnegativedirectionReconoise3 == 3)
            HEnegativedirectionReconoise3->SetXTitle("R for HE- jeta =-27; depth = 3 \b");
          if (kcountHEnegativedirectionReconoise3 == 4)
            HEnegativedirectionReconoise3->SetXTitle("R for HE- jeta =-26; depth = 3 \b");
          if (kcountHEnegativedirectionReconoise3 == 5)
            HEnegativedirectionReconoise3->SetXTitle("R for HE- jeta =-25; depth = 3 \b");
          if (kcountHEnegativedirectionReconoise3 == 6)
            HEnegativedirectionReconoise3->SetXTitle("R for HE- jeta =-24; depth = 3 \b");
          if (kcountHEnegativedirectionReconoise3 == 7)
            HEnegativedirectionReconoise3->SetXTitle("R for HE- jeta =-23; depth = 3 \b");
          if (kcountHEnegativedirectionReconoise3 == 8)
            HEnegativedirectionReconoise3->SetXTitle("R for HE- jeta =-22; depth = 3 \b");
          if (kcountHEnegativedirectionReconoise3 == 9)
            HEnegativedirectionReconoise3->SetXTitle("R for HE- jeta =-21; depth = 3 \b");
          if (kcountHEnegativedirectionReconoise3 == 10)
            HEnegativedirectionReconoise3->SetXTitle("R for HE- jeta =-20; depth = 3 \b");
          if (kcountHEnegativedirectionReconoise3 == 11)
            HEnegativedirectionReconoise3->SetXTitle("R for HE- jeta =-19; depth = 3 \b");
          if (kcountHEnegativedirectionReconoise3 == 12)
            HEnegativedirectionReconoise3->SetXTitle("R for HE- jeta =-18; depth = 3 \b");
          if (kcountHEnegativedirectionReconoise3 == 13)
            HEnegativedirectionReconoise3->SetXTitle("R for HE- jeta =-17; depth = 3 \b");
          HEnegativedirectionReconoise3->Draw("Error");
          kcountHEnegativedirectionReconoise3++;
          if (kcountHEnegativedirectionReconoise3 > 13)
            break;  // 4x6 = 24
        }           //ccctest>0

      }  // for i
    }    //if(jeta-41  >= -29 && jeta-41 <= -16)
  }      //for jeta
  /////////////////
  c3x5->Update();
  c3x5->Print("RreconoiseNegativeDirectionhistD1PhiSymmetryDepth3HE.png");
  c3x5->Clear();
  // clean-up
  if (h2CeffHEnegativedirectionReconoise3)
    delete h2CeffHEnegativedirectionReconoise3;
  //========================================================================================== 1117
  //======================================================================
  //======================================================================1D plot: R vs phi , different eta,  depth=4
  //cout<<"      1D plot: R vs phi , different eta,  depth=4 *****" <<endl;
  c3x5->Clear();
  /////////////////
  c3x5->Divide(3, 5);
  c3x5->cd(1);
  int kcountHEnegativedirectionReconoise4 = 1;
  TH1F* h2CeffHEnegativedirectionReconoise4 = new TH1F("h2CeffHEnegativedirectionReconoise4", "", nphi, 0., 72.);

  for (int jeta = 0; jeta < njeta; jeta++) {
    // negativedirectionReconoise:
    if (jeta - 41 >= -29 && jeta - 41 <= -16) {
      //	     for (int i=0;i<ndepth;i++) {
      // depth=4
      for (int i = 3; i < 4; i++) {
        TH1F* HEnegativedirectionReconoise4 = (TH1F*)h2CeffHEnegativedirectionReconoise4->Clone("twod1");

        float ccctest = 0;  // to avoid empty massive elements
        for (int jphi = 0; jphi < nphi; jphi++) {
          double ccc1 = areconoisehe[i][jeta][jphi];
          if (ccc1 != 0.) {
            HEnegativedirectionReconoise4->Fill(jphi, ccc1);
            ccctest = 1.;  //HEnegativedirectionReconoise4->SetBinError(i,0.01);
          }
        }  // for jphi
        if (ccctest > 0.) {
          //cout<<"777        kcountHEnegativedirectionReconoise4   =     "<<kcountHEnegativedirectionReconoise4  <<"   jeta-41=     "<< jeta-41 <<endl;
          c3x5->cd(kcountHEnegativedirectionReconoise4);
          HEnegativedirectionReconoise4->SetMarkerStyle(20);
          HEnegativedirectionReconoise4->SetMarkerSize(0.4);
          HEnegativedirectionReconoise4->GetYaxis()->SetLabelSize(0.04);
          HEnegativedirectionReconoise4->SetXTitle("HEnegativedirectionReconoise4 \b");
          HEnegativedirectionReconoise4->SetMarkerColor(2);
          HEnegativedirectionReconoise4->SetLineColor(0);
          gPad->SetGridy();
          gPad->SetGridx();
          //	   gPad->SetLogy();
          if (kcountHEnegativedirectionReconoise4 == 1)
            HEnegativedirectionReconoise4->SetXTitle("R for HE- jeta =-28; depth = 4 \b");
          if (kcountHEnegativedirectionReconoise4 == 2)
            HEnegativedirectionReconoise4->SetXTitle("R for HE- jeta =-27; depth = 4 \b");
          if (kcountHEnegativedirectionReconoise4 == 3)
            HEnegativedirectionReconoise4->SetXTitle("R for HE- jeta =-26; depth = 4 \b");
          if (kcountHEnegativedirectionReconoise4 == 4)
            HEnegativedirectionReconoise4->SetXTitle("R for HE- jeta =-25; depth = 4 \b");
          if (kcountHEnegativedirectionReconoise4 == 5)
            HEnegativedirectionReconoise4->SetXTitle("R for HE- jeta =-24; depth = 4 \b");
          if (kcountHEnegativedirectionReconoise4 == 6)
            HEnegativedirectionReconoise4->SetXTitle("R for HE- jeta =-23; depth = 4 \b");
          if (kcountHEnegativedirectionReconoise4 == 7)
            HEnegativedirectionReconoise4->SetXTitle("R for HE- jeta =-22; depth = 4 \b");
          if (kcountHEnegativedirectionReconoise4 == 8)
            HEnegativedirectionReconoise4->SetXTitle("R for HE- jeta =-21; depth = 4 \b");
          if (kcountHEnegativedirectionReconoise4 == 9)
            HEnegativedirectionReconoise4->SetXTitle("R for HE- jeta =-20; depth = 4 \b");
          if (kcountHEnegativedirectionReconoise4 == 10)
            HEnegativedirectionReconoise4->SetXTitle("R for HE- jeta =-19; depth = 4 \b");
          if (kcountHEnegativedirectionReconoise4 == 11)
            HEnegativedirectionReconoise4->SetXTitle("R for HE- jeta =-18; depth = 4 \b");
          if (kcountHEnegativedirectionReconoise4 == 12)
            HEnegativedirectionReconoise4->SetXTitle("R for HE- jeta =-16; depth = 4 \b");
          HEnegativedirectionReconoise4->Draw("Error");
          kcountHEnegativedirectionReconoise4++;
          if (kcountHEnegativedirectionReconoise4 > 12)
            break;  // 4x6 = 24
        }           //ccctest>0

      }  // for i
    }    //if(jeta-41  >= -29 && jeta-41 <= -16)
  }      //for jeta
  /////////////////
  c3x5->Update();
  c3x5->Print("RreconoiseNegativeDirectionhistD1PhiSymmetryDepth4HE.png");
  c3x5->Clear();
  // clean-up
  if (h2CeffHEnegativedirectionReconoise4)
    delete h2CeffHEnegativedirectionReconoise4;
  //========================================================================================== 1118
  //======================================================================
  //======================================================================1D plot: R vs phi , different eta,  depth=5
  //cout<<"      1D plot: R vs phi , different eta,  depth=5 *****" <<endl;
  c3x5->Clear();
  /////////////////
  c3x5->Divide(3, 5);
  c3x5->cd(1);
  int kcountHEnegativedirectionReconoise5 = 1;
  TH1F* h2CeffHEnegativedirectionReconoise5 = new TH1F("h2CeffHEnegativedirectionReconoise5", "", nphi, 0., 72.);

  for (int jeta = 0; jeta < njeta; jeta++) {
    // negativedirectionReconoise:
    if (jeta - 41 >= -29 && jeta - 41 <= -16) {
      //	     for (int i=0;i<ndepth;i++) {
      // depth=5
      for (int i = 4; i < 5; i++) {
        TH1F* HEnegativedirectionReconoise5 = (TH1F*)h2CeffHEnegativedirectionReconoise5->Clone("twod1");

        float ccctest = 0;  // to avoid empty massive elements
        for (int jphi = 0; jphi < nphi; jphi++) {
          //	       cout<<"888  initial      kcountHEnegativedirectionReconoise5   =     "<<kcountHEnegativedirectionReconoise5  <<"   jeta-41=     "<< jeta-41 <<"   jphi=     "<< jphi <<"   areconoisehe[i][jeta][jphi]=     "<< areconoisehe[i][jeta][jphi] <<"  depth=     "<< i <<endl;

          double ccc1 = areconoisehe[i][jeta][jphi];
          if (ccc1 != 0.) {
            HEnegativedirectionReconoise5->Fill(jphi, ccc1);
            ccctest = 1.;  //HEnegativedirectionReconoise5->SetBinError(i,0.01);
          }
        }  // for jphi
        if (ccctest > 0.) {
          //cout<<"888        kcountHEnegativedirectionReconoise5   =     "<<kcountHEnegativedirectionReconoise5  <<"   jeta-41=     "<< jeta-41 <<endl;
          c3x5->cd(kcountHEnegativedirectionReconoise5);
          HEnegativedirectionReconoise5->SetMarkerStyle(20);
          HEnegativedirectionReconoise5->SetMarkerSize(0.4);
          HEnegativedirectionReconoise5->GetYaxis()->SetLabelSize(0.04);
          HEnegativedirectionReconoise5->SetXTitle("HEnegativedirectionReconoise5 \b");
          HEnegativedirectionReconoise5->SetMarkerColor(2);
          HEnegativedirectionReconoise5->SetLineColor(0);
          gPad->SetGridy();
          gPad->SetGridx();
          //	   gPad->SetLogy();
          if (kcountHEnegativedirectionReconoise5 == 1)
            HEnegativedirectionReconoise5->SetXTitle("R for HE- jeta =-28; depth = 5 \b");
          if (kcountHEnegativedirectionReconoise5 == 2)
            HEnegativedirectionReconoise5->SetXTitle("R for HE- jeta =-27; depth = 5 \b");
          if (kcountHEnegativedirectionReconoise5 == 3)
            HEnegativedirectionReconoise5->SetXTitle("R for HE- jeta =-26; depth = 5 \b");
          if (kcountHEnegativedirectionReconoise5 == 4)
            HEnegativedirectionReconoise5->SetXTitle("R for HE- jeta =-25; depth = 5 \b");
          if (kcountHEnegativedirectionReconoise5 == 5)
            HEnegativedirectionReconoise5->SetXTitle("R for HE- jeta =-24; depth = 5 \b");
          if (kcountHEnegativedirectionReconoise5 == 6)
            HEnegativedirectionReconoise5->SetXTitle("R for HE- jeta =-23; depth = 5 \b");
          if (kcountHEnegativedirectionReconoise5 == 7)
            HEnegativedirectionReconoise5->SetXTitle("R for HE- jeta =-22; depth = 5 \b");
          if (kcountHEnegativedirectionReconoise5 == 8)
            HEnegativedirectionReconoise5->SetXTitle("R for HE- jeta =-21; depth = 5 \b");
          if (kcountHEnegativedirectionReconoise5 == 9)
            HEnegativedirectionReconoise5->SetXTitle("R for HE- jeta =-20; depth = 5 \b");
          if (kcountHEnegativedirectionReconoise5 == 10)
            HEnegativedirectionReconoise5->SetXTitle("R for HE- jeta =-19; depth = 5 \b");
          if (kcountHEnegativedirectionReconoise5 == 11)
            HEnegativedirectionReconoise5->SetXTitle("R for HE- jeta =-18; depth = 5 \b");
          HEnegativedirectionReconoise5->Draw("Error");
          kcountHEnegativedirectionReconoise5++;
          if (kcountHEnegativedirectionReconoise5 > 11)
            break;  // 4x6 = 24
        }           //ccctest>0

      }  // for i
    }    //if(jeta-41  >= -29 && jeta-41 <= -16)
  }      //for jeta
  /////////////////
  c3x5->Update();
  c3x5->Print("RreconoiseNegativeDirectionhistD1PhiSymmetryDepth5HE.png");
  c3x5->Clear();
  // clean-up
  if (h2CeffHEnegativedirectionReconoise5)
    delete h2CeffHEnegativedirectionReconoise5;
  //========================================================================================== 1119
  //======================================================================
  //======================================================================1D plot: R vs phi , different eta,  depth=6
  //cout<<"      1D plot: R vs phi , different eta,  depth=6 *****" <<endl;
  c3x5->Clear();
  /////////////////
  c3x5->Divide(3, 5);
  c3x5->cd(1);
  int kcountHEnegativedirectionReconoise6 = 1;
  TH1F* h2CeffHEnegativedirectionReconoise6 = new TH1F("h2CeffHEnegativedirectionReconoise6", "", nphi, 0., 72.);

  for (int jeta = 0; jeta < njeta; jeta++) {
    // negativedirectionReconoise:
    if (jeta - 41 >= -29 && jeta - 41 <= -16) {
      //	     for (int i=0;i<ndepth;i++) {
      // depth=6
      for (int i = 5; i < 6; i++) {
        TH1F* HEnegativedirectionReconoise6 = (TH1F*)h2CeffHEnegativedirectionReconoise6->Clone("twod1");

        float ccctest = 0;  // to avoid empty massive elements
        for (int jphi = 0; jphi < nphi; jphi++) {
          double ccc1 = areconoisehe[i][jeta][jphi];
          if (ccc1 != 0.) {
            HEnegativedirectionReconoise6->Fill(jphi, ccc1);
            ccctest = 1.;  //HEnegativedirectionReconoise6->SetBinError(i,0.01);
          }
        }  // for jphi
        if (ccctest > 0.) {
          //cout<<"999        kcountHEnegativedirectionReconoise6   =     "<<kcountHEnegativedirectionReconoise6  <<"   jeta-41=     "<< jeta-41 <<endl;
          c3x5->cd(kcountHEnegativedirectionReconoise6);
          HEnegativedirectionReconoise6->SetMarkerStyle(20);
          HEnegativedirectionReconoise6->SetMarkerSize(0.4);
          HEnegativedirectionReconoise6->GetYaxis()->SetLabelSize(0.04);
          HEnegativedirectionReconoise6->SetXTitle("HEnegativedirectionReconoise6 \b");
          HEnegativedirectionReconoise6->SetMarkerColor(2);
          HEnegativedirectionReconoise6->SetLineColor(0);
          gPad->SetGridy();
          gPad->SetGridx();
          //	   gPad->SetLogy();
          if (kcountHEnegativedirectionReconoise6 == 1)
            HEnegativedirectionReconoise6->SetXTitle("R for HE- jeta =-28; depth = 6 \b");
          if (kcountHEnegativedirectionReconoise6 == 2)
            HEnegativedirectionReconoise6->SetXTitle("R for HE- jeta =-27; depth = 6 \b");
          if (kcountHEnegativedirectionReconoise6 == 3)
            HEnegativedirectionReconoise6->SetXTitle("R for HE- jeta =-26; depth = 6 \b");
          if (kcountHEnegativedirectionReconoise6 == 4)
            HEnegativedirectionReconoise6->SetXTitle("R for HE- jeta =-25; depth = 6 \b");
          if (kcountHEnegativedirectionReconoise6 == 5)
            HEnegativedirectionReconoise6->SetXTitle("R for HE- jeta =-24; depth = 6 \b");
          if (kcountHEnegativedirectionReconoise6 == 6)
            HEnegativedirectionReconoise6->SetXTitle("R for HE- jeta =-23; depth = 6 \b");
          if (kcountHEnegativedirectionReconoise6 == 7)
            HEnegativedirectionReconoise6->SetXTitle("R for HE- jeta =-22; depth = 6 \b");
          if (kcountHEnegativedirectionReconoise6 == 8)
            HEnegativedirectionReconoise6->SetXTitle("R for HE- jeta =-21; depth = 6 \b");
          if (kcountHEnegativedirectionReconoise6 == 9)
            HEnegativedirectionReconoise6->SetXTitle("R for HE- jeta =-20; depth = 6 \b");
          if (kcountHEnegativedirectionReconoise6 == 10)
            HEnegativedirectionReconoise6->SetXTitle("R for HE- jeta =-19; depth = 6 \b");
          HEnegativedirectionReconoise6->Draw("Error");
          kcountHEnegativedirectionReconoise6++;
          if (kcountHEnegativedirectionReconoise6 > 10)
            break;  // 4x6 = 24
        }           //ccctest>0

      }  // for i
    }    //if(jeta-41  >= -29 && jeta-41 <= -16)
  }      //for jeta
  /////////////////
  c3x5->Update();
  c3x5->Print("RreconoiseNegativeDirectionhistD1PhiSymmetryDepth6HE.png");
  c3x5->Clear();
  // clean-up
  if (h2CeffHEnegativedirectionReconoise6)
    delete h2CeffHEnegativedirectionReconoise6;
  //========================================================================================== 11110
  //======================================================================
  //======================================================================1D plot: R vs phi , different eta,  depth=7
  //cout<<"      1D plot: R vs phi , different eta,  depth=7 *****" <<endl;
  c3x5->Clear();
  /////////////////
  c3x5->Divide(3, 5);
  c3x5->cd(1);
  int kcountHEnegativedirectionReconoise7 = 1;
  TH1F* h2CeffHEnegativedirectionReconoise7 = new TH1F("h2CeffHEnegativedirectionReconoise7", "", nphi, 0., 72.);

  for (int jeta = 0; jeta < njeta; jeta++) {
    // negativedirectionReconoise:
    if (jeta - 41 >= -29 && jeta - 41 <= -16) {
      //	     for (int i=0;i<ndepth;i++) {
      // depth=7
      for (int i = 6; i < 7; i++) {
        TH1F* HEnegativedirectionReconoise7 = (TH1F*)h2CeffHEnegativedirectionReconoise7->Clone("twod1");

        float ccctest = 0;  // to avoid empty massive elements
        for (int jphi = 0; jphi < nphi; jphi++) {
          double ccc1 = areconoisehe[i][jeta][jphi];
          if (ccc1 != 0.) {
            HEnegativedirectionReconoise7->Fill(jphi, ccc1);
            ccctest = 1.;  //HEnegativedirectionReconoise7->SetBinError(i,0.01);
          }
        }  // for jphi
        if (ccctest > 0.) {
          //cout<<"1010       kcountHEnegativedirectionReconoise7   =     "<<kcountHEnegativedirectionReconoise7  <<"   jeta-41=     "<< jeta-41 <<endl;
          c3x5->cd(kcountHEnegativedirectionReconoise7);
          HEnegativedirectionReconoise7->SetMarkerStyle(20);
          HEnegativedirectionReconoise7->SetMarkerSize(0.4);
          HEnegativedirectionReconoise7->GetYaxis()->SetLabelSize(0.04);
          HEnegativedirectionReconoise7->SetXTitle("HEnegativedirectionReconoise7 \b");
          HEnegativedirectionReconoise7->SetMarkerColor(2);
          HEnegativedirectionReconoise7->SetLineColor(0);
          gPad->SetGridy();
          gPad->SetGridx();
          //	   gPad->SetLogy();
          if (kcountHEnegativedirectionReconoise7 == 1)
            HEnegativedirectionReconoise7->SetXTitle("R for HE- jeta =-28; depth = 7 \b");
          if (kcountHEnegativedirectionReconoise7 == 2)
            HEnegativedirectionReconoise7->SetXTitle("R for HE- jeta =-27; depth = 7 \b");
          if (kcountHEnegativedirectionReconoise7 == 3)
            HEnegativedirectionReconoise7->SetXTitle("R for HE- jeta =-26; depth = 7 \b");
          HEnegativedirectionReconoise7->Draw("Error");
          kcountHEnegativedirectionReconoise7++;
          if (kcountHEnegativedirectionReconoise7 > 3)
            break;  // 4x6 = 24
        }           //ccctest>0

      }  // for i
    }    //if(jeta-41  >= -29 && jeta-41 <= -16)
  }      //for jeta
  /////////////////
  c3x5->Update();
  c3x5->Print("RreconoiseNegativeDirectionhistD1PhiSymmetryDepth7HE.png");
  c3x5->Clear();
  // clean-up
  if (h2CeffHEnegativedirectionReconoise7)
    delete h2CeffHEnegativedirectionReconoise7;

  //======================================================================================================================
  //======================================================================================================================
  //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  //======================================================================================================================
  //                                   DIFDIFDIFDIFDIFDIFDIFDIFDIFDIFDIFDIFDIFDIFDIFDIFDIFDIFDIFDIF:   Reconoise HE
  //======================================================================================================================
  //======================================================================
  c2x1->Clear();
  /////////////////
  c2x1->Divide(2, 1);
  c2x1->cd(1);
  TH2F* GefzDIFreconoiseHE42D = new TH2F("GefzDIFreconoiseHE42D", "", neta, -41., 41., nphi, 0., 72.);
  TH2F* GefzDIFreconoiseHE42D0 = new TH2F("GefzDIFreconoiseHE42D0", "", neta, -41., 41., nphi, 0., 72.);
  TH2F* GefzDIFreconoiseHE42DF = (TH2F*)GefzDIFreconoiseHE42D0->Clone("GefzDIFreconoiseHE42DF");
  for (int i = 0; i < ndepth; i++) {
    for (int jeta = 0; jeta < neta; jeta++) {
      if ((jeta - 41 >= -29 && jeta - 41 <= -16) || (jeta - 41 >= 15 && jeta - 41 <= 28)) {
        for (int jphi = 0; jphi < nphi; jphi++) {
          double ccc1 = breconoisehe[i][jeta][jphi];
          int k2plot = jeta - 41;
          int kkk = k2plot;  //if(k2plot >0 ) kkk=k2plot+1; //-41 +41 !=0
          if (ccc1 != 0.) {
            GefzDIFreconoiseHE42D->Fill(kkk, jphi, ccc1);
            GefzDIFreconoiseHE42D0->Fill(kkk, jphi, 1.);
          }
        }
      }
    }
  }
  GefzDIFreconoiseHE42DF->Divide(GefzDIFreconoiseHE42D, GefzDIFreconoiseHE42D0, 1, 1, "B");  // average A
  gPad->SetGridy();
  gPad->SetGridx();  //      gPad->SetLogz();
  GefzDIFreconoiseHE42DF->SetXTitle("<DIF>_depth       #eta  \b");
  GefzDIFreconoiseHE42DF->SetYTitle("      #phi \b");
  GefzDIFreconoiseHE42DF->Draw("COLZ");

  //c2x1->cd(2);
  //TH1F *energyhitNoise_HE= (TH1F*)dir->FindObjectAny("h_energyhitNoise_HE");
  //energyhitNoise_HE ->SetMarkerStyle(20);energyhitNoise_HE ->SetMarkerSize(0.4);energyhitNoise_HE ->GetYaxis()->SetLabelSize(0.04);energyhitNoise_HE ->SetXTitle("energyhitNoise_HE \b");energyhitNoise_HE ->SetMarkerColor(2);energyhitNoise_HE ->SetLineColor(0);gPad->SetGridy();gPad->SetGridx();energyhitNoise_HE ->Draw("Error");

  /////////////////
  c2x1->Update();
  c2x1->Print("DIFreconoiseGeneralD2PhiSymmetryHE.png");
  c2x1->Clear();
  // clean-up
  if (GefzDIFreconoiseHE42D)
    delete GefzDIFreconoiseHE42D;
  if (GefzDIFreconoiseHE42D0)
    delete GefzDIFreconoiseHE42D0;
  if (GefzDIFreconoiseHE42DF)
    delete GefzDIFreconoiseHE42DF;
  //====================================================================== 1D plot: DIF vs phi , averaged over depthes & eta
  //======================================================================
  //cout<<"      1D plot: DIF vs phi , averaged over depthes & eta *****" <<endl;
  c1x1->Clear();
  /////////////////
  c1x1->Divide(1, 1);
  c1x1->cd(1);
  TH1F* GefzDIFreconoiseHE41D = new TH1F("GefzDIFreconoiseHE41D", "", nphi, 0., 72.);
  TH1F* GefzDIFreconoiseHE41D0 = new TH1F("GefzDIFreconoiseHE41D0", "", nphi, 0., 72.);
  TH1F* GefzDIFreconoiseHE41DF = (TH1F*)GefzDIFreconoiseHE41D0->Clone("GefzDIFreconoiseHE41DF");
  for (int jphi = 0; jphi < nphi; jphi++) {
    for (int jeta = 0; jeta < neta; jeta++) {
      if ((jeta - 41 >= -29 && jeta - 41 <= -16) || (jeta - 41 >= 15 && jeta - 41 <= 28)) {
        for (int i = 0; i < ndepth; i++) {
          double ccc1 = breconoisehe[i][jeta][jphi];
          if (ccc1 != 0.) {
            GefzDIFreconoiseHE41D->Fill(jphi, ccc1);
            GefzDIFreconoiseHE41D0->Fill(jphi, 1.);
          }
        }
      }
    }
  }
  GefzDIFreconoiseHE41DF->Divide(
      GefzDIFreconoiseHE41D, GefzDIFreconoiseHE41D0, 1, 1, "B");  // DIF averaged over depthes & eta
  GefzDIFreconoiseHE41D0->Sumw2();
  //    for (int jphi=1;jphi<73;jphi++) {GefzDIFreconoiseHE41DF->SetBinError(jphi,0.01);}
  gPad->SetGridy();
  gPad->SetGridx();  //      gPad->SetLogz();
  GefzDIFreconoiseHE41DF->SetMarkerStyle(20);
  GefzDIFreconoiseHE41DF->SetMarkerSize(1.4);
  GefzDIFreconoiseHE41DF->GetZaxis()->SetLabelSize(0.08);
  GefzDIFreconoiseHE41DF->SetXTitle("#phi  \b");
  GefzDIFreconoiseHE41DF->SetYTitle("  <DIF> \b");
  GefzDIFreconoiseHE41DF->SetZTitle("<DIF>_PHI  - AllDepthes \b");
  GefzDIFreconoiseHE41DF->SetMarkerColor(4);
  GefzDIFreconoiseHE41DF->SetLineColor(
      4);  // GefzDIFreconoiseHE41DF->SetMinimum(0.8);     //      GefzDIFreconoiseHE41DF->SetMaximum(1.000);
  GefzDIFreconoiseHE41DF->Draw("Error");
  /////////////////
  c1x1->Update();
  c1x1->Print("DIFreconoiseGeneralD1PhiSymmetryHE.png");
  c1x1->Clear();
  // clean-up
  if (GefzDIFreconoiseHE41D)
    delete GefzDIFreconoiseHE41D;
  if (GefzDIFreconoiseHE41D0)
    delete GefzDIFreconoiseHE41D0;
  if (GefzDIFreconoiseHE41DF)
    delete GefzDIFreconoiseHE41DF;

  //========================================================================================== 4
  //======================================================================
  //======================================================================1D plot: DIF vs phi , different eta,  depth=1
  //cout<<"      1D plot: DIF vs phi , different eta,  depth=1 *****" <<endl;
  c3x5->Clear();
  /////////////////
  c3x5->Divide(3, 5);
  c3x5->cd(1);
  int kcountHEpositivedirectionReconoiseDIF1 = 1;
  TH1F* h2CeffHEpositivedirectionReconoiseDIF1 = new TH1F("h2CeffHEpositivedirectionReconoiseDIF1", "", nphi, 0., 72.);
  for (int jeta = 0; jeta < njeta; jeta++) {
    // positivedirectionReconoiseDIF:
    if (jeta - 41 >= 15 && jeta - 41 <= 28) {
      //	     for (int i=0;i<ndepth;i++) {
      // depth=1
      for (int i = 0; i < 1; i++) {
        TH1F* HEpositivedirectionReconoiseDIF1 = (TH1F*)h2CeffHEpositivedirectionReconoiseDIF1->Clone("twod1");
        float ccctest = 0;  // to avoid empty massive elements
        for (int jphi = 0; jphi < nphi; jphi++) {
          double ccc1 = breconoisehe[i][jeta][jphi];
          if (ccc1 != 0.) {
            HEpositivedirectionReconoiseDIF1->Fill(jphi, ccc1);
            ccctest = 1.;  //HEpositivedirectionReconoiseDIF1->SetBinError(i,0.01);
          }
        }  // for jphi
        if (ccctest > 0.) {
          //	  cout<<"444        kcountHEpositivedirectionReconoiseDIF1   =     "<<kcountHEpositivedirectionReconoiseDIF1  <<"   jeta-41=     "<< jeta-41 <<endl;
          c3x5->cd(kcountHEpositivedirectionReconoiseDIF1);
          HEpositivedirectionReconoiseDIF1->SetMarkerStyle(20);
          HEpositivedirectionReconoiseDIF1->SetMarkerSize(0.4);
          HEpositivedirectionReconoiseDIF1->GetYaxis()->SetLabelSize(0.04);
          HEpositivedirectionReconoiseDIF1->SetXTitle("HEpositivedirectionReconoiseDIF1 \b");
          HEpositivedirectionReconoiseDIF1->SetMarkerColor(2);
          HEpositivedirectionReconoiseDIF1->SetLineColor(0);
          gPad->SetGridy();
          gPad->SetGridx();
          //	   gPad->SetLogy();
          if (kcountHEpositivedirectionReconoiseDIF1 == 1)
            HEpositivedirectionReconoiseDIF1->SetXTitle("DIF for HE+ jeta = 17; depth = 1 \b");
          if (kcountHEpositivedirectionReconoiseDIF1 == 2)
            HEpositivedirectionReconoiseDIF1->SetXTitle("DIF for HE+ jeta = 18; depth = 1 \b");
          if (kcountHEpositivedirectionReconoiseDIF1 == 3)
            HEpositivedirectionReconoiseDIF1->SetXTitle("DIF for HE+ jeta = 19; depth = 1 \b");
          if (kcountHEpositivedirectionReconoiseDIF1 == 4)
            HEpositivedirectionReconoiseDIF1->SetXTitle("DIF for HE+ jeta = 20; depth = 1 \b");
          if (kcountHEpositivedirectionReconoiseDIF1 == 5)
            HEpositivedirectionReconoiseDIF1->SetXTitle("DIF for HE+ jeta = 21; depth = 1 \b");
          if (kcountHEpositivedirectionReconoiseDIF1 == 6)
            HEpositivedirectionReconoiseDIF1->SetXTitle("DIF for HE+ jeta = 22; depth = 1 \b");
          if (kcountHEpositivedirectionReconoiseDIF1 == 7)
            HEpositivedirectionReconoiseDIF1->SetXTitle("DIF for HE+ jeta = 23; depth = 1 \b");
          if (kcountHEpositivedirectionReconoiseDIF1 == 8)
            HEpositivedirectionReconoiseDIF1->SetXTitle("DIF for HE+ jeta = 24; depth = 1 \b");
          if (kcountHEpositivedirectionReconoiseDIF1 == 9)
            HEpositivedirectionReconoiseDIF1->SetXTitle("DIF for HE+ jeta = 25; depth = 1 \b");
          if (kcountHEpositivedirectionReconoiseDIF1 == 10)
            HEpositivedirectionReconoiseDIF1->SetXTitle("DIF for HE+ jeta = 26; depth = 1 \b");
          if (kcountHEpositivedirectionReconoiseDIF1 == 11)
            HEpositivedirectionReconoiseDIF1->SetXTitle("DIF for HE+ jeta = 27; depth = 1 \b");
          if (kcountHEpositivedirectionReconoiseDIF1 == 12)
            HEpositivedirectionReconoiseDIF1->SetXTitle("DIF for HE+ jeta = 28; depth = 1 \b");
          HEpositivedirectionReconoiseDIF1->Draw("Error");
          kcountHEpositivedirectionReconoiseDIF1++;
          if (kcountHEpositivedirectionReconoiseDIF1 > 12)
            break;  // 4x6 = 24
        }           //ccctest>0

      }  // for i
    }    //if(jeta-41 >= 15 && jeta-41 <= 28
  }      //for jeta
  /////////////////
  c3x5->Update();
  c3x5->Print("DIFreconoisePositiveDirectionhistD1PhiSymmetryDepth1HE.png");
  c3x5->Clear();
  // clean-up
  if (h2CeffHEpositivedirectionReconoiseDIF1)
    delete h2CeffHEpositivedirectionReconoiseDIF1;

  //========================================================================================== 5
  //======================================================================
  //======================================================================1D plot: R vs phi , different eta,  depth=2
  //  cout<<"      1D plot: R vs phi , different eta,  depth=2 *****" <<endl;
  c3x5->Clear();
  /////////////////
  c3x5->Divide(3, 5);
  c3x5->cd(1);
  int kcountHEpositivedirectionReconoiseDIF2 = 1;
  TH1F* h2CeffHEpositivedirectionReconoiseDIF2 = new TH1F("h2CeffHEpositivedirectionReconoiseDIF2", "", nphi, 0., 72.);
  for (int jeta = 0; jeta < njeta; jeta++) {
    // positivedirectionReconoiseDIF:
    if (jeta - 41 >= 15 && jeta - 41 <= 28) {
      //	     for (int i=0;i<ndepth;i++) {
      // depth=2
      for (int i = 1; i < 2; i++) {
        TH1F* HEpositivedirectionReconoiseDIF2 = (TH1F*)h2CeffHEpositivedirectionReconoiseDIF2->Clone("twod1");
        float ccctest = 0;  // to avoid empty massive elements
        for (int jphi = 0; jphi < nphi; jphi++) {
          double ccc1 = breconoisehe[i][jeta][jphi];
          if (ccc1 != 0.) {
            HEpositivedirectionReconoiseDIF2->Fill(jphi, ccc1);
            ccctest = 1.;  //HEpositivedirectionReconoiseDIF2->SetBinError(i,0.01);
          }
        }  // for jphi
        if (ccctest > 0.) {
          //cout<<"555        kcountHEpositivedirectionReconoiseDIF2   =     "<<kcountHEpositivedirectionReconoiseDIF2  <<"   jeta-41=     "<< jeta-41 <<endl;
          c3x5->cd(kcountHEpositivedirectionReconoiseDIF2);
          HEpositivedirectionReconoiseDIF2->SetMarkerStyle(20);
          HEpositivedirectionReconoiseDIF2->SetMarkerSize(0.4);
          HEpositivedirectionReconoiseDIF2->GetYaxis()->SetLabelSize(0.04);
          HEpositivedirectionReconoiseDIF2->SetXTitle("HEpositivedirectionReconoiseDIF2 \b");
          HEpositivedirectionReconoiseDIF2->SetMarkerColor(2);
          HEpositivedirectionReconoiseDIF2->SetLineColor(0);
          gPad->SetGridy();
          gPad->SetGridx();
          //	   gPad->SetLogy();
          if (kcountHEpositivedirectionReconoiseDIF2 == 1)
            HEpositivedirectionReconoiseDIF2->SetXTitle("DIF for HE+ jeta = 16; depth = 2 \b");
          if (kcountHEpositivedirectionReconoiseDIF2 == 2)
            HEpositivedirectionReconoiseDIF2->SetXTitle("DIF for HE+ jeta = 17; depth = 2 \b");
          if (kcountHEpositivedirectionReconoiseDIF2 == 3)
            HEpositivedirectionReconoiseDIF2->SetXTitle("DIF for HE+ jeta = 18; depth = 2 \b");
          if (kcountHEpositivedirectionReconoiseDIF2 == 4)
            HEpositivedirectionReconoiseDIF2->SetXTitle("DIF for HE+ jeta = 19; depth = 2 \b");
          if (kcountHEpositivedirectionReconoiseDIF2 == 5)
            HEpositivedirectionReconoiseDIF2->SetXTitle("DIF for HE+ jeta = 20; depth = 2 \b");
          if (kcountHEpositivedirectionReconoiseDIF2 == 6)
            HEpositivedirectionReconoiseDIF2->SetXTitle("DIF for HE+ jeta = 21; depth = 2 \b");
          if (kcountHEpositivedirectionReconoiseDIF2 == 7)
            HEpositivedirectionReconoiseDIF2->SetXTitle("DIF for HE+ jeta = 22; depth = 2 \b");
          if (kcountHEpositivedirectionReconoiseDIF2 == 8)
            HEpositivedirectionReconoiseDIF2->SetXTitle("DIF for HE+ jeta = 23; depth = 2 \b");
          if (kcountHEpositivedirectionReconoiseDIF2 == 9)
            HEpositivedirectionReconoiseDIF2->SetXTitle("DIF for HE+ jeta = 24; depth = 2 \b");
          if (kcountHEpositivedirectionReconoiseDIF2 == 10)
            HEpositivedirectionReconoiseDIF2->SetXTitle("DIF for HE+ jeta = 25; depth = 2 \b");
          if (kcountHEpositivedirectionReconoiseDIF2 == 11)
            HEpositivedirectionReconoiseDIF2->SetXTitle("DIF for HE+ jeta = 26; depth = 2 \b");
          if (kcountHEpositivedirectionReconoiseDIF2 == 12)
            HEpositivedirectionReconoiseDIF2->SetXTitle("DIF for HE+ jeta = 27; depth = 2 \b");
          if (kcountHEpositivedirectionReconoiseDIF2 == 13)
            HEpositivedirectionReconoiseDIF2->SetXTitle("DIF for HE+ jeta = 28; depth = 2 \b");
          HEpositivedirectionReconoiseDIF2->Draw("Error");
          kcountHEpositivedirectionReconoiseDIF2++;
          if (kcountHEpositivedirectionReconoiseDIF2 > 13)
            break;  // 4x6 = 24
        }           //ccctest>0

      }  // for i
    }    //if(jeta-41
  }      //for jeta
  /////////////////
  c3x5->Update();
  c3x5->Print("DIFreconoisePositiveDirectionhistD1PhiSymmetryDepth2HE.png");
  c3x5->Clear();
  // clean-up
  if (h2CeffHEpositivedirectionReconoiseDIF2)
    delete h2CeffHEpositivedirectionReconoiseDIF2;
  //========================================================================================== 6
  //======================================================================
  //======================================================================1D plot: R vs phi , different eta,  depth=3
  //cout<<"      1D plot: R vs phi , different eta,  depth=3 *****" <<endl;
  c3x5->Clear();
  /////////////////
  c3x5->Divide(3, 5);
  c3x5->cd(1);
  int kcountHEpositivedirectionReconoiseDIF3 = 1;
  TH1F* h2CeffHEpositivedirectionReconoiseDIF3 = new TH1F("h2CeffHEpositivedirectionReconoiseDIF3", "", nphi, 0., 72.);
  for (int jeta = 0; jeta < njeta; jeta++) {
    // positivedirectionReconoiseDIF:
    if (jeta - 41 >= 15 && jeta - 41 <= 28) {
      //	     for (int i=0;i<ndepth;i++) {
      // depth=3
      for (int i = 2; i < 3; i++) {
        TH1F* HEpositivedirectionReconoiseDIF3 = (TH1F*)h2CeffHEpositivedirectionReconoiseDIF3->Clone("twod1");
        float ccctest = 0;  // to avoid empty massive elements
        for (int jphi = 0; jphi < nphi; jphi++) {
          double ccc1 = breconoisehe[i][jeta][jphi];
          if (ccc1 != 0.) {
            HEpositivedirectionReconoiseDIF3->Fill(jphi, ccc1);
            ccctest = 1.;  //HEpositivedirectionReconoiseDIF3->SetBinError(i,0.01);
          }
        }  // for jphi
        if (ccctest > 0.) {
          //cout<<"666        kcountHEpositivedirectionReconoiseDIF3   =     "<<kcountHEpositivedirectionReconoiseDIF3  <<"   jeta-41=     "<< jeta-41 <<endl;
          c3x5->cd(kcountHEpositivedirectionReconoiseDIF3);
          HEpositivedirectionReconoiseDIF3->SetMarkerStyle(20);
          HEpositivedirectionReconoiseDIF3->SetMarkerSize(0.4);
          HEpositivedirectionReconoiseDIF3->GetYaxis()->SetLabelSize(0.04);
          HEpositivedirectionReconoiseDIF3->SetXTitle("HEpositivedirectionReconoiseDIF3 \b");
          HEpositivedirectionReconoiseDIF3->SetMarkerColor(2);
          HEpositivedirectionReconoiseDIF3->SetLineColor(0);
          gPad->SetGridy();
          gPad->SetGridx();
          //	   gPad->SetLogy();
          if (kcountHEpositivedirectionReconoiseDIF3 == 1)
            HEpositivedirectionReconoiseDIF3->SetXTitle("DIF for HE+ jeta = 16; depth = 3 \b");
          if (kcountHEpositivedirectionReconoiseDIF3 == 2)
            HEpositivedirectionReconoiseDIF3->SetXTitle("DIF for HE+ jeta = 17; depth = 3 \b");
          if (kcountHEpositivedirectionReconoiseDIF3 == 3)
            HEpositivedirectionReconoiseDIF3->SetXTitle("DIF for HE+ jeta = 18; depth = 3 \b");
          if (kcountHEpositivedirectionReconoiseDIF3 == 4)
            HEpositivedirectionReconoiseDIF3->SetXTitle("DIF for HE+ jeta = 19; depth = 3 \b");
          if (kcountHEpositivedirectionReconoiseDIF3 == 5)
            HEpositivedirectionReconoiseDIF3->SetXTitle("DIF for HE+ jeta = 20; depth = 3 \b");
          if (kcountHEpositivedirectionReconoiseDIF3 == 6)
            HEpositivedirectionReconoiseDIF3->SetXTitle("DIF for HE+ jeta = 21; depth = 3 \b");
          if (kcountHEpositivedirectionReconoiseDIF3 == 7)
            HEpositivedirectionReconoiseDIF3->SetXTitle("DIF for HE+ jeta = 22; depth = 3 \b");
          if (kcountHEpositivedirectionReconoiseDIF3 == 8)
            HEpositivedirectionReconoiseDIF3->SetXTitle("DIF for HE+ jeta = 23; depth = 3 \b");
          if (kcountHEpositivedirectionReconoiseDIF3 == 9)
            HEpositivedirectionReconoiseDIF3->SetXTitle("DIF for HE+ jeta = 24; depth = 3 \b");
          if (kcountHEpositivedirectionReconoiseDIF3 == 10)
            HEpositivedirectionReconoiseDIF3->SetXTitle("DIF for HE+ jeta = 25; depth = 3 \b");
          if (kcountHEpositivedirectionReconoiseDIF3 == 11)
            HEpositivedirectionReconoiseDIF3->SetXTitle("DIF for HE+ jeta = 26; depth = 3 \b");
          if (kcountHEpositivedirectionReconoiseDIF3 == 12)
            HEpositivedirectionReconoiseDIF3->SetXTitle("DIF for HE+ jeta = 27; depth = 3 \b");
          if (kcountHEpositivedirectionReconoiseDIF3 == 13)
            HEpositivedirectionReconoiseDIF3->SetXTitle("DIF for HE+ jeta = 28; depth = 3 \b");
          HEpositivedirectionReconoiseDIF3->Draw("Error");
          kcountHEpositivedirectionReconoiseDIF3++;
          if (kcountHEpositivedirectionReconoiseDIF3 > 13)
            break;  // 4x6 = 24
        }           //ccctest>0

      }  // for i
    }    //if(jeta-41 >=
  }      //for jeta
  /////////////////
  c3x5->Update();
  c3x5->Print("DIFreconoisePositiveDirectionhistD1PhiSymmetryDepth3HE.png");
  c3x5->Clear();
  // clean-up
  if (h2CeffHEpositivedirectionReconoiseDIF3)
    delete h2CeffHEpositivedirectionReconoiseDIF3;
  //========================================================================================== 7
  //======================================================================
  //======================================================================1D plot: R vs phi , different eta,  depth=4
  //cout<<"      1D plot: R vs phi , different eta,  depth=4 *****" <<endl;
  c3x5->Clear();
  /////////////////
  c3x5->Divide(3, 5);
  c3x5->cd(1);
  int kcountHEpositivedirectionReconoiseDIF4 = 1;
  TH1F* h2CeffHEpositivedirectionReconoiseDIF4 = new TH1F("h2CeffHEpositivedirectionReconoiseDIF4", "", nphi, 0., 72.);

  for (int jeta = 0; jeta < njeta; jeta++) {
    // positivedirectionReconoiseDIF:
    if (jeta - 41 >= 15 && jeta - 41 <= 28) {
      //	     for (int i=0;i<ndepth;i++) {
      // depth=4
      for (int i = 3; i < 4; i++) {
        TH1F* HEpositivedirectionReconoiseDIF4 = (TH1F*)h2CeffHEpositivedirectionReconoiseDIF4->Clone("twod1");

        float ccctest = 0;  // to avoid empty massive elements
        for (int jphi = 0; jphi < nphi; jphi++) {
          double ccc1 = breconoisehe[i][jeta][jphi];
          if (ccc1 != 0.) {
            HEpositivedirectionReconoiseDIF4->Fill(jphi, ccc1);
            ccctest = 1.;  //HEpositivedirectionReconoiseDIF4->SetBinError(i,0.01);
          }
        }  // for jphi
        if (ccctest > 0.) {
          //cout<<"777        kcountHEpositivedirectionReconoiseDIF4   =     "<<kcountHEpositivedirectionReconoiseDIF4  <<"   jeta-41=     "<< jeta-41 <<endl;
          c3x5->cd(kcountHEpositivedirectionReconoiseDIF4);
          HEpositivedirectionReconoiseDIF4->SetMarkerStyle(20);
          HEpositivedirectionReconoiseDIF4->SetMarkerSize(0.4);
          HEpositivedirectionReconoiseDIF4->GetYaxis()->SetLabelSize(0.04);
          HEpositivedirectionReconoiseDIF4->SetXTitle("HEpositivedirectionReconoiseDIF4 \b");
          HEpositivedirectionReconoiseDIF4->SetMarkerColor(2);
          HEpositivedirectionReconoiseDIF4->SetLineColor(0);
          gPad->SetGridy();
          gPad->SetGridx();
          //	   gPad->SetLogy();
          if (kcountHEpositivedirectionReconoiseDIF4 == 1)
            HEpositivedirectionReconoiseDIF4->SetXTitle("DIF for HE+ jeta = 15; depth = 4 \b");
          if (kcountHEpositivedirectionReconoiseDIF4 == 2)
            HEpositivedirectionReconoiseDIF4->SetXTitle("DIF for HE+ jeta = 17; depth = 4 \b");
          if (kcountHEpositivedirectionReconoiseDIF4 == 3)
            HEpositivedirectionReconoiseDIF4->SetXTitle("DIF for HE+ jeta = 18; depth = 4 \b");
          if (kcountHEpositivedirectionReconoiseDIF4 == 4)
            HEpositivedirectionReconoiseDIF4->SetXTitle("DIF for HE+ jeta = 19; depth = 4 \b");
          if (kcountHEpositivedirectionReconoiseDIF4 == 5)
            HEpositivedirectionReconoiseDIF4->SetXTitle("DIF for HE+ jeta = 20; depth = 4 \b");
          if (kcountHEpositivedirectionReconoiseDIF4 == 6)
            HEpositivedirectionReconoiseDIF4->SetXTitle("DIF for HE+ jeta = 21; depth = 4 \b");
          if (kcountHEpositivedirectionReconoiseDIF4 == 7)
            HEpositivedirectionReconoiseDIF4->SetXTitle("DIF for HE+ jeta = 22; depth = 4 \b");
          if (kcountHEpositivedirectionReconoiseDIF4 == 8)
            HEpositivedirectionReconoiseDIF4->SetXTitle("DIF for HE+ jeta = 23; depth = 4 \b");
          if (kcountHEpositivedirectionReconoiseDIF4 == 9)
            HEpositivedirectionReconoiseDIF4->SetXTitle("DIF for HE+ jeta = 24; depth = 4 \b");
          if (kcountHEpositivedirectionReconoiseDIF4 == 10)
            HEpositivedirectionReconoiseDIF4->SetXTitle("DIF for HE+ jeta = 25; depth = 4 \b");
          if (kcountHEpositivedirectionReconoiseDIF4 == 11)
            HEpositivedirectionReconoiseDIF4->SetXTitle("DIF for HE+ jeta = 26; depth = 4 \b");
          if (kcountHEpositivedirectionReconoiseDIF4 == 12)
            HEpositivedirectionReconoiseDIF4->SetXTitle("DIF for HE+ jeta = 27; depth = 4 \b");
          HEpositivedirectionReconoiseDIF4->Draw("Error");
          kcountHEpositivedirectionReconoiseDIF4++;
          if (kcountHEpositivedirectionReconoiseDIF4 > 12)
            break;  // 4x6 = 24
        }           //ccctest>0

      }  // for i
    }    //if(jeta-41 >=  -29 && jeta-41 <= -16)
  }      //for jeta
  /////////////////
  c3x5->Update();
  c3x5->Print("DIFreconoisePositiveDirectionhistD1PhiSymmetryDepth4HE.png");
  c3x5->Clear();
  // clean-up
  if (h2CeffHEpositivedirectionReconoiseDIF4)
    delete h2CeffHEpositivedirectionReconoiseDIF4;
  //========================================================================================== 8
  //======================================================================
  //======================================================================1D plot: R vs phi , different eta,  depth=5
  //cout<<"      1D plot: R vs phi , different eta,  depth=5 *****" <<endl;
  c3x5->Clear();
  /////////////////
  c3x5->Divide(3, 5);
  c3x5->cd(1);
  int kcountHEpositivedirectionReconoiseDIF5 = 1;
  TH1F* h2CeffHEpositivedirectionReconoiseDIF5 = new TH1F("h2CeffHEpositivedirectionReconoiseDIF5", "", nphi, 0., 72.);

  for (int jeta = 0; jeta < njeta; jeta++) {
    // positivedirectionReconoiseDIF:
    if (jeta - 41 >= 15 && jeta - 41 <= 28) {
      //	     for (int i=0;i<ndepth;i++) {
      // depth=5
      for (int i = 4; i < 5; i++) {
        TH1F* HEpositivedirectionReconoiseDIF5 = (TH1F*)h2CeffHEpositivedirectionReconoiseDIF5->Clone("twod1");

        float ccctest = 0;  // to avoid empty massive elements
        for (int jphi = 0; jphi < nphi; jphi++) {
          //	       cout<<"888  initial      kcountHEpositivedirectionReconoiseDIF5   =     "<<kcountHEpositivedirectionReconoiseDIF5  <<"   jeta-41=     "<< jeta-41 <<"   jphi=     "<< jphi <<"   breconoisehe[i][jeta][jphi]=     "<< breconoisehe[i][jeta][jphi] <<"  depth=     "<< i <<endl;

          double ccc1 = breconoisehe[i][jeta][jphi];
          if (ccc1 != 0.) {
            HEpositivedirectionReconoiseDIF5->Fill(jphi, ccc1);
            ccctest = 1.;  //HEpositivedirectionReconoiseDIF5->SetBinError(i,0.01);
          }
        }  // for jphi
        if (ccctest > 0.) {
          //cout<<"888        kcountHEpositivedirectionReconoiseDIF5   =     "<<kcountHEpositivedirectionReconoiseDIF5  <<"   jeta-41=     "<< jeta-41 <<endl;
          c3x5->cd(kcountHEpositivedirectionReconoiseDIF5);
          HEpositivedirectionReconoiseDIF5->SetMarkerStyle(20);
          HEpositivedirectionReconoiseDIF5->SetMarkerSize(0.4);
          HEpositivedirectionReconoiseDIF5->GetYaxis()->SetLabelSize(0.04);
          HEpositivedirectionReconoiseDIF5->SetXTitle("HEpositivedirectionReconoiseDIF5 \b");
          HEpositivedirectionReconoiseDIF5->SetMarkerColor(2);
          HEpositivedirectionReconoiseDIF5->SetLineColor(0);
          gPad->SetGridy();
          gPad->SetGridx();
          //	   gPad->SetLogy();
          if (kcountHEpositivedirectionReconoiseDIF5 == 1)
            HEpositivedirectionReconoiseDIF5->SetXTitle("DIF for HE+ jeta = 17; depth = 5 \b");
          if (kcountHEpositivedirectionReconoiseDIF5 == 2)
            HEpositivedirectionReconoiseDIF5->SetXTitle("DIF for HE+ jeta = 18; depth = 5 \b");
          if (kcountHEpositivedirectionReconoiseDIF5 == 3)
            HEpositivedirectionReconoiseDIF5->SetXTitle("DIF for HE+ jeta = 19; depth = 5 \b");
          if (kcountHEpositivedirectionReconoiseDIF5 == 4)
            HEpositivedirectionReconoiseDIF5->SetXTitle("DIF for HE+ jeta = 20; depth = 5 \b");
          if (kcountHEpositivedirectionReconoiseDIF5 == 5)
            HEpositivedirectionReconoiseDIF5->SetXTitle("DIF for HE+ jeta = 21; depth = 5 \b");
          if (kcountHEpositivedirectionReconoiseDIF5 == 6)
            HEpositivedirectionReconoiseDIF5->SetXTitle("DIF for HE+ jeta = 22; depth = 5 \b");
          if (kcountHEpositivedirectionReconoiseDIF5 == 7)
            HEpositivedirectionReconoiseDIF5->SetXTitle("DIF for HE+ jeta = 23; depth = 5 \b");
          if (kcountHEpositivedirectionReconoiseDIF5 == 8)
            HEpositivedirectionReconoiseDIF5->SetXTitle("DIF for HE+ jeta = 24; depth = 5 \b");
          if (kcountHEpositivedirectionReconoiseDIF5 == 9)
            HEpositivedirectionReconoiseDIF5->SetXTitle("DIF for HE+ jeta = 25; depth = 5 \b");
          if (kcountHEpositivedirectionReconoiseDIF5 == 10)
            HEpositivedirectionReconoiseDIF5->SetXTitle("DIF for HE+ jeta = 26; depth = 5 \b");
          if (kcountHEpositivedirectionReconoiseDIF5 == 11)
            HEpositivedirectionReconoiseDIF5->SetXTitle("DIF for HE+ jeta = 27; depth = 5 \b");
          HEpositivedirectionReconoiseDIF5->Draw("Error");
          kcountHEpositivedirectionReconoiseDIF5++;
          if (kcountHEpositivedirectionReconoiseDIF5 > 11)
            break;  // 4x6 = 24
        }           //ccctest>0

      }  // for i
    }    //if(jeta-41 >=  -29 && jeta-41 <= -16)
  }      //for jeta
  /////////////////
  c3x5->Update();
  c3x5->Print("DIFreconoisePositiveDirectionhistD1PhiSymmetryDepth5HE.png");
  c3x5->Clear();
  // clean-up
  if (h2CeffHEpositivedirectionReconoiseDIF5)
    delete h2CeffHEpositivedirectionReconoiseDIF5;
  //========================================================================================== 9
  //======================================================================
  //======================================================================1D plot: R vs phi , different eta,  depth=6
  //cout<<"      1D plot: R vs phi , different eta,  depth=6 *****" <<endl;
  c3x5->Clear();
  /////////////////
  c3x5->Divide(3, 5);
  c3x5->cd(1);
  int kcountHEpositivedirectionReconoiseDIF6 = 1;
  TH1F* h2CeffHEpositivedirectionReconoiseDIF6 = new TH1F("h2CeffHEpositivedirectionReconoiseDIF6", "", nphi, 0., 72.);

  for (int jeta = 0; jeta < njeta; jeta++) {
    // positivedirectionReconoiseDIF:
    if (jeta - 41 >= 15 && jeta - 41 <= 28) {
      //	     for (int i=0;i<ndepth;i++) {
      // depth=6
      for (int i = 5; i < 6; i++) {
        TH1F* HEpositivedirectionReconoiseDIF6 = (TH1F*)h2CeffHEpositivedirectionReconoiseDIF6->Clone("twod1");

        float ccctest = 0;  // to avoid empty massive elements
        for (int jphi = 0; jphi < nphi; jphi++) {
          double ccc1 = breconoisehe[i][jeta][jphi];
          if (ccc1 != 0.) {
            HEpositivedirectionReconoiseDIF6->Fill(jphi, ccc1);
            ccctest = 1.;  //HEpositivedirectionReconoiseDIF6->SetBinError(i,0.01);
          }
        }  // for jphi
        if (ccctest > 0.) {
          //cout<<"999        kcountHEpositivedirectionReconoiseDIF6   =     "<<kcountHEpositivedirectionReconoiseDIF6  <<"   jeta-41=     "<< jeta-41 <<endl;
          c3x5->cd(kcountHEpositivedirectionReconoiseDIF6);
          HEpositivedirectionReconoiseDIF6->SetMarkerStyle(20);
          HEpositivedirectionReconoiseDIF6->SetMarkerSize(0.4);
          HEpositivedirectionReconoiseDIF6->GetYaxis()->SetLabelSize(0.04);
          HEpositivedirectionReconoiseDIF6->SetXTitle("HEpositivedirectionReconoiseDIF6 \b");
          HEpositivedirectionReconoiseDIF6->SetMarkerColor(2);
          HEpositivedirectionReconoiseDIF6->SetLineColor(0);
          gPad->SetGridy();
          gPad->SetGridx();
          //	   gPad->SetLogy();
          if (kcountHEpositivedirectionReconoiseDIF6 == 1)
            HEpositivedirectionReconoiseDIF6->SetXTitle("DIF for HE+ jeta = 18; depth = 6 \b");
          if (kcountHEpositivedirectionReconoiseDIF6 == 2)
            HEpositivedirectionReconoiseDIF6->SetXTitle("DIF for HE+ jeta = 19; depth = 6 \b");
          if (kcountHEpositivedirectionReconoiseDIF6 == 3)
            HEpositivedirectionReconoiseDIF6->SetXTitle("DIF for HE+ jeta = 20; depth = 6 \b");
          if (kcountHEpositivedirectionReconoiseDIF6 == 4)
            HEpositivedirectionReconoiseDIF6->SetXTitle("DIF for HE+ jeta = 21; depth = 6 \b");
          if (kcountHEpositivedirectionReconoiseDIF6 == 5)
            HEpositivedirectionReconoiseDIF6->SetXTitle("DIF for HE+ jeta = 22; depth = 6 \b");
          if (kcountHEpositivedirectionReconoiseDIF6 == 6)
            HEpositivedirectionReconoiseDIF6->SetXTitle("DIF for HE+ jeta = 23; depth = 6 \b");
          if (kcountHEpositivedirectionReconoiseDIF6 == 7)
            HEpositivedirectionReconoiseDIF6->SetXTitle("DIF for HE+ jeta = 24; depth = 6 \b");
          if (kcountHEpositivedirectionReconoiseDIF6 == 8)
            HEpositivedirectionReconoiseDIF6->SetXTitle("DIF for HE+ jeta = 25; depth = 6 \b");
          if (kcountHEpositivedirectionReconoiseDIF6 == 9)
            HEpositivedirectionReconoiseDIF6->SetXTitle("DIF for HE+ jeta = 26; depth = 6 \b");
          if (kcountHEpositivedirectionReconoiseDIF6 == 10)
            HEpositivedirectionReconoiseDIF6->SetXTitle("DIF for HE+ jeta = 27; depth = 6 \b");
          HEpositivedirectionReconoiseDIF6->Draw("Error");
          kcountHEpositivedirectionReconoiseDIF6++;
          if (kcountHEpositivedirectionReconoiseDIF6 > 10)
            break;  // 4x6 = 24
        }           //ccctest>0

      }  // for i
    }    //if(jeta-41 >=  -29 && jeta-41 <= -16)
  }      //for jeta
  /////////////////
  c3x5->Update();
  c3x5->Print("DIFreconoisePositiveDirectionhistD1PhiSymmetryDepth6HE.png");
  c3x5->Clear();
  // clean-up
  if (h2CeffHEpositivedirectionReconoiseDIF6)
    delete h2CeffHEpositivedirectionReconoiseDIF6;
  //========================================================================================== 10
  //======================================================================
  //======================================================================1D plot: R vs phi , different eta,  depth=7
  //cout<<"      1D plot: R vs phi , different eta,  depth=7 *****" <<endl;
  c3x5->Clear();
  /////////////////
  c3x5->Divide(3, 5);
  c3x5->cd(1);
  int kcountHEpositivedirectionReconoiseDIF7 = 1;
  TH1F* h2CeffHEpositivedirectionReconoiseDIF7 = new TH1F("h2CeffHEpositivedirectionReconoiseDIF7", "", nphi, 0., 72.);

  for (int jeta = 0; jeta < njeta; jeta++) {
    // positivedirectionReconoiseDIF:
    if (jeta - 41 >= 15 && jeta - 41 <= 28) {
      //	     for (int i=0;i<ndepth;i++) {
      // depth=7
      for (int i = 6; i < 7; i++) {
        TH1F* HEpositivedirectionReconoiseDIF7 = (TH1F*)h2CeffHEpositivedirectionReconoiseDIF7->Clone("twod1");

        float ccctest = 0;  // to avoid empty massive elements
        for (int jphi = 0; jphi < nphi; jphi++) {
          double ccc1 = breconoisehe[i][jeta][jphi];
          if (ccc1 != 0.) {
            HEpositivedirectionReconoiseDIF7->Fill(jphi, ccc1);
            ccctest = 1.;  //HEpositivedirectionReconoiseDIF7->SetBinError(i,0.01);
          }
        }  // for jphi
        if (ccctest > 0.) {
          //cout<<"1010       kcountHEpositivedirectionReconoiseDIF7   =     "<<kcountHEpositivedirectionReconoiseDIF7  <<"   jeta-41=     "<< jeta-41 <<endl;
          c3x5->cd(kcountHEpositivedirectionReconoiseDIF7);
          HEpositivedirectionReconoiseDIF7->SetMarkerStyle(20);
          HEpositivedirectionReconoiseDIF7->SetMarkerSize(0.4);
          HEpositivedirectionReconoiseDIF7->GetYaxis()->SetLabelSize(0.04);
          HEpositivedirectionReconoiseDIF7->SetXTitle("HEpositivedirectionReconoiseDIF7 \b");
          HEpositivedirectionReconoiseDIF7->SetMarkerColor(2);
          HEpositivedirectionReconoiseDIF7->SetLineColor(0);
          gPad->SetGridy();
          gPad->SetGridx();
          //	   gPad->SetLogy();
          if (kcountHEpositivedirectionReconoiseDIF7 == 1)
            HEpositivedirectionReconoiseDIF7->SetXTitle("DIF for HE+ jeta = 25; depth = 7 \b");
          if (kcountHEpositivedirectionReconoiseDIF7 == 2)
            HEpositivedirectionReconoiseDIF7->SetXTitle("DIF for HE+ jeta = 26; depth = 7 \b");
          if (kcountHEpositivedirectionReconoiseDIF7 == 3)
            HEpositivedirectionReconoiseDIF7->SetXTitle("DIF for HE+ jeta = 27; depth = 7 \b");
          HEpositivedirectionReconoiseDIF7->Draw("Error");
          kcountHEpositivedirectionReconoiseDIF7++;
          if (kcountHEpositivedirectionReconoiseDIF7 > 3)
            break;  //
        }           //ccctest>0

      }  // for i
    }    //if(jeta-41 >=  -29 && jeta-41 <= -16)
  }      //for jeta
  /////////////////
  c3x5->Update();
  c3x5->Print("DIFreconoisePositiveDirectionhistD1PhiSymmetryDepth7HE.png");
  c3x5->Clear();
  // clean-up
  if (h2CeffHEpositivedirectionReconoiseDIF7)
    delete h2CeffHEpositivedirectionReconoiseDIF7;

  //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  //========================================================================================== 1114
  //======================================================================
  //======================================================================1D plot: R vs phi , different eta,  depth=1
  //cout<<"      1D plot: R vs phi , different eta,  depth=1 *****" <<endl;
  c3x5->Clear();
  /////////////////
  c3x5->Divide(3, 5);
  c3x5->cd(1);
  int kcountHEnegativedirectionReconoiseDIF1 = 1;
  TH1F* h2CeffHEnegativedirectionReconoiseDIF1 = new TH1F("h2CeffHEnegativedirectionReconoiseDIF1", "", nphi, 0., 72.);
  for (int jeta = 0; jeta < njeta; jeta++) {
    // negativedirectionReconoiseDIF:
    if (jeta - 41 >= -29 && jeta - 41 <= -16) {
      //	     for (int i=0;i<ndepth;i++) {
      // depth=1
      for (int i = 0; i < 1; i++) {
        TH1F* HEnegativedirectionReconoiseDIF1 = (TH1F*)h2CeffHEnegativedirectionReconoiseDIF1->Clone("twod1");
        float ccctest = 0;  // to avoid empty massive elements
        for (int jphi = 0; jphi < nphi; jphi++) {
          double ccc1 = breconoisehe[i][jeta][jphi];
          if (ccc1 != 0.) {
            HEnegativedirectionReconoiseDIF1->Fill(jphi, ccc1);
            ccctest = 1.;  //HEnegativedirectionReconoiseDIF1->SetBinError(i,0.01);
          }
        }  // for jphi
        if (ccctest > 0.) {
          //	  cout<<"444        kcountHEnegativedirectionReconoiseDIF1   =     "<<kcountHEnegativedirectionReconoiseDIF1  <<"   jeta-41=     "<< jeta-41 <<endl;
          c3x5->cd(kcountHEnegativedirectionReconoiseDIF1);
          HEnegativedirectionReconoiseDIF1->SetMarkerStyle(20);
          HEnegativedirectionReconoiseDIF1->SetMarkerSize(0.4);
          HEnegativedirectionReconoiseDIF1->GetYaxis()->SetLabelSize(0.04);
          HEnegativedirectionReconoiseDIF1->SetXTitle("HEnegativedirectionReconoiseDIF1 \b");
          HEnegativedirectionReconoiseDIF1->SetMarkerColor(2);
          HEnegativedirectionReconoiseDIF1->SetLineColor(0);
          gPad->SetGridy();
          gPad->SetGridx();
          //	   gPad->SetLogy();
          if (kcountHEnegativedirectionReconoiseDIF1 == 1)
            HEnegativedirectionReconoiseDIF1->SetXTitle("DIF for HE- jeta =-29; depth = 1 \b");
          if (kcountHEnegativedirectionReconoiseDIF1 == 2)
            HEnegativedirectionReconoiseDIF1->SetXTitle("DIF for HE- jeta =-28; depth = 1 \b");
          if (kcountHEnegativedirectionReconoiseDIF1 == 3)
            HEnegativedirectionReconoiseDIF1->SetXTitle("DIF for HE- jeta =-27; depth = 1 \b");
          if (kcountHEnegativedirectionReconoiseDIF1 == 4)
            HEnegativedirectionReconoiseDIF1->SetXTitle("DIF for HE- jeta =-26; depth = 1 \b");
          if (kcountHEnegativedirectionReconoiseDIF1 == 5)
            HEnegativedirectionReconoiseDIF1->SetXTitle("DIF for HE- jeta =-25; depth = 1 \b");
          if (kcountHEnegativedirectionReconoiseDIF1 == 6)
            HEnegativedirectionReconoiseDIF1->SetXTitle("DIF for HE- jeta =-24; depth = 1 \b");
          if (kcountHEnegativedirectionReconoiseDIF1 == 7)
            HEnegativedirectionReconoiseDIF1->SetXTitle("DIF for HE- jeta =-23; depth = 1 \b");
          if (kcountHEnegativedirectionReconoiseDIF1 == 8)
            HEnegativedirectionReconoiseDIF1->SetXTitle("DIF for HE- jeta =-22; depth = 1 \b");
          if (kcountHEnegativedirectionReconoiseDIF1 == 9)
            HEnegativedirectionReconoiseDIF1->SetXTitle("DIF for HE- jeta =-21; depth = 1 \b");
          if (kcountHEnegativedirectionReconoiseDIF1 == 10)
            HEnegativedirectionReconoiseDIF1->SetXTitle("DIF for HE- jeta =-20; depth = 1 \b");
          if (kcountHEnegativedirectionReconoiseDIF1 == 11)
            HEnegativedirectionReconoiseDIF1->SetXTitle("DIF for HE- jeta =-19; depth = 1 \b");
          if (kcountHEnegativedirectionReconoiseDIF1 == 12)
            HEnegativedirectionReconoiseDIF1->SetXTitle("DIF for HE- jeta =-18; depth = 1 \b");
          HEnegativedirectionReconoiseDIF1->Draw("Error");
          kcountHEnegativedirectionReconoiseDIF1++;
          if (kcountHEnegativedirectionReconoiseDIF1 > 12)
            break;  // 4x6 = 24
        }           //ccctest>0

      }  // for i
    }    //if(jeta-41  >= -29 && jeta-41 <= -16)
  }      //for jeta
  /////////////////
  c3x5->Update();
  c3x5->Print("DIFreconoiseNegativeDirectionhistD1PhiSymmetryDepth1HE.png");
  c3x5->Clear();
  // clean-up
  if (h2CeffHEnegativedirectionReconoiseDIF1)
    delete h2CeffHEnegativedirectionReconoiseDIF1;

  //========================================================================================== 1115
  //======================================================================
  //======================================================================1D plot: R vs phi , different eta,  depth=2
  //  cout<<"      1D plot: R vs phi , different eta,  depth=2 *****" <<endl;
  c3x5->Clear();
  /////////////////
  c3x5->Divide(3, 5);
  c3x5->cd(1);
  int kcountHEnegativedirectionReconoiseDIF2 = 1;
  TH1F* h2CeffHEnegativedirectionReconoiseDIF2 = new TH1F("h2CeffHEnegativedirectionReconoiseDIF2", "", nphi, 0., 72.);
  for (int jeta = 0; jeta < njeta; jeta++) {
    // negativedirectionReconoiseDIF:
    if (jeta - 41 >= -29 && jeta - 41 <= -16) {
      //	     for (int i=0;i<ndepth;i++) {
      // depth=2
      for (int i = 1; i < 2; i++) {
        TH1F* HEnegativedirectionReconoiseDIF2 = (TH1F*)h2CeffHEnegativedirectionReconoiseDIF2->Clone("twod1");
        float ccctest = 0;  // to avoid empty massive elements
        for (int jphi = 0; jphi < nphi; jphi++) {
          double ccc1 = breconoisehe[i][jeta][jphi];
          if (ccc1 != 0.) {
            HEnegativedirectionReconoiseDIF2->Fill(jphi, ccc1);
            ccctest = 1.;  //HEnegativedirectionReconoiseDIF2->SetBinError(i,0.01);
          }
        }  // for jphi
        if (ccctest > 0.) {
          //cout<<"555        kcountHEnegativedirectionReconoiseDIF2   =     "<<kcountHEnegativedirectionReconoiseDIF2  <<"   jeta-41=     "<< jeta-41 <<endl;
          c3x5->cd(kcountHEnegativedirectionReconoiseDIF2);
          HEnegativedirectionReconoiseDIF2->SetMarkerStyle(20);
          HEnegativedirectionReconoiseDIF2->SetMarkerSize(0.4);
          HEnegativedirectionReconoiseDIF2->GetYaxis()->SetLabelSize(0.04);
          HEnegativedirectionReconoiseDIF2->SetXTitle("HEnegativedirectionReconoiseDIF2 \b");
          HEnegativedirectionReconoiseDIF2->SetMarkerColor(2);
          HEnegativedirectionReconoiseDIF2->SetLineColor(0);
          gPad->SetGridy();
          gPad->SetGridx();
          //	   gPad->SetLogy();
          if (kcountHEnegativedirectionReconoiseDIF2 == 1)
            HEnegativedirectionReconoiseDIF2->SetXTitle("DIF for HE- jeta =-29; depth = 2 \b");
          if (kcountHEnegativedirectionReconoiseDIF2 == 2)
            HEnegativedirectionReconoiseDIF2->SetXTitle("DIF for HE- jeta =-28; depth = 2 \b");
          if (kcountHEnegativedirectionReconoiseDIF2 == 3)
            HEnegativedirectionReconoiseDIF2->SetXTitle("DIF for HE- jeta =-27; depth = 2 \b");
          if (kcountHEnegativedirectionReconoiseDIF2 == 4)
            HEnegativedirectionReconoiseDIF2->SetXTitle("DIF for HE- jeta =-26; depth = 2 \b");
          if (kcountHEnegativedirectionReconoiseDIF2 == 5)
            HEnegativedirectionReconoiseDIF2->SetXTitle("DIF for HE- jeta =-25; depth = 2 \b");
          if (kcountHEnegativedirectionReconoiseDIF2 == 6)
            HEnegativedirectionReconoiseDIF2->SetXTitle("DIF for HE- jeta =-24; depth = 2 \b");
          if (kcountHEnegativedirectionReconoiseDIF2 == 7)
            HEnegativedirectionReconoiseDIF2->SetXTitle("DIF for HE- jeta =-23; depth = 2 \b");
          if (kcountHEnegativedirectionReconoiseDIF2 == 8)
            HEnegativedirectionReconoiseDIF2->SetXTitle("DIF for HE- jeta =-22; depth = 2 \b");
          if (kcountHEnegativedirectionReconoiseDIF2 == 9)
            HEnegativedirectionReconoiseDIF2->SetXTitle("DIF for HE- jeta =-21; depth = 2 \b");
          if (kcountHEnegativedirectionReconoiseDIF2 == 10)
            HEnegativedirectionReconoiseDIF2->SetXTitle("DIF for HE- jeta =-20; depth = 2 \b");
          if (kcountHEnegativedirectionReconoiseDIF2 == 11)
            HEnegativedirectionReconoiseDIF2->SetXTitle("DIF for HE- jeta =-19; depth = 2 \b");
          if (kcountHEnegativedirectionReconoiseDIF2 == 12)
            HEnegativedirectionReconoiseDIF2->SetXTitle("DIF for HE- jeta =-18; depth = 2 \b");
          if (kcountHEnegativedirectionReconoiseDIF2 == 13)
            HEnegativedirectionReconoiseDIF2->SetXTitle("DIF for HE- jeta =-17; depth = 2 \b");
          HEnegativedirectionReconoiseDIF2->Draw("Error");
          kcountHEnegativedirectionReconoiseDIF2++;
          if (kcountHEnegativedirectionReconoiseDIF2 > 13)
            break;  // 4x6 = 24
        }           //ccctest>0

      }  // for i
    }    //if(jeta-41  >= -29 && jeta-41 <= -16)
  }      //for jeta
  /////////////////
  c3x5->Update();
  c3x5->Print("DIFreconoiseNegativeDirectionhistD1PhiSymmetryDepth2HE.png");
  c3x5->Clear();
  // clean-up
  if (h2CeffHEnegativedirectionReconoiseDIF2)
    delete h2CeffHEnegativedirectionReconoiseDIF2;
  //========================================================================================== 1116
  //======================================================================
  //======================================================================1D plot: R vs phi , different eta,  depth=3
  //cout<<"      1D plot: R vs phi , different eta,  depth=3 *****" <<endl;
  c3x5->Clear();
  /////////////////
  c3x5->Divide(3, 5);
  c3x5->cd(1);
  int kcountHEnegativedirectionReconoiseDIF3 = 1;
  TH1F* h2CeffHEnegativedirectionReconoiseDIF3 = new TH1F("h2CeffHEnegativedirectionReconoiseDIF3", "", nphi, 0., 72.);
  for (int jeta = 0; jeta < njeta; jeta++) {
    // negativedirectionReconoiseDIF:
    if (jeta - 41 >= -29 && jeta - 41 <= -16) {
      //	     for (int i=0;i<ndepth;i++) {
      // depth=3
      for (int i = 2; i < 3; i++) {
        TH1F* HEnegativedirectionReconoiseDIF3 = (TH1F*)h2CeffHEnegativedirectionReconoiseDIF3->Clone("twod1");
        float ccctest = 0;  // to avoid empty massive elements
        for (int jphi = 0; jphi < nphi; jphi++) {
          double ccc1 = breconoisehe[i][jeta][jphi];
          if (ccc1 != 0.) {
            HEnegativedirectionReconoiseDIF3->Fill(jphi, ccc1);
            ccctest = 1.;  //HEnegativedirectionReconoiseDIF3->SetBinError(i,0.01);
          }
        }  // for jphi
        if (ccctest > 0.) {
          //cout<<"666        kcountHEnegativedirectionReconoiseDIF3   =     "<<kcountHEnegativedirectionReconoiseDIF3  <<"   jeta-41=     "<< jeta-41 <<endl;
          c3x5->cd(kcountHEnegativedirectionReconoiseDIF3);
          HEnegativedirectionReconoiseDIF3->SetMarkerStyle(20);
          HEnegativedirectionReconoiseDIF3->SetMarkerSize(0.4);
          HEnegativedirectionReconoiseDIF3->GetYaxis()->SetLabelSize(0.04);
          HEnegativedirectionReconoiseDIF3->SetXTitle("HEnegativedirectionReconoiseDIF3 \b");
          HEnegativedirectionReconoiseDIF3->SetMarkerColor(2);
          HEnegativedirectionReconoiseDIF3->SetLineColor(0);
          gPad->SetGridy();
          gPad->SetGridx();
          //	   gPad->SetLogy();
          if (kcountHEnegativedirectionReconoiseDIF3 == 1)
            HEnegativedirectionReconoiseDIF3->SetXTitle("DIF for HE- jeta =-29; depth = 3 \b");
          if (kcountHEnegativedirectionReconoiseDIF3 == 2)
            HEnegativedirectionReconoiseDIF3->SetXTitle("DIF for HE- jeta =-28; depth = 3 \b");
          if (kcountHEnegativedirectionReconoiseDIF3 == 3)
            HEnegativedirectionReconoiseDIF3->SetXTitle("DIF for HE- jeta =-27; depth = 3 \b");
          if (kcountHEnegativedirectionReconoiseDIF3 == 4)
            HEnegativedirectionReconoiseDIF3->SetXTitle("DIF for HE- jeta =-26; depth = 3 \b");
          if (kcountHEnegativedirectionReconoiseDIF3 == 5)
            HEnegativedirectionReconoiseDIF3->SetXTitle("DIF for HE- jeta =-25; depth = 3 \b");
          if (kcountHEnegativedirectionReconoiseDIF3 == 6)
            HEnegativedirectionReconoiseDIF3->SetXTitle("DIF for HE- jeta =-24; depth = 3 \b");
          if (kcountHEnegativedirectionReconoiseDIF3 == 7)
            HEnegativedirectionReconoiseDIF3->SetXTitle("DIF for HE- jeta =-23; depth = 3 \b");
          if (kcountHEnegativedirectionReconoiseDIF3 == 8)
            HEnegativedirectionReconoiseDIF3->SetXTitle("DIF for HE- jeta =-22; depth = 3 \b");
          if (kcountHEnegativedirectionReconoiseDIF3 == 9)
            HEnegativedirectionReconoiseDIF3->SetXTitle("DIF for HE- jeta =-21; depth = 3 \b");
          if (kcountHEnegativedirectionReconoiseDIF3 == 10)
            HEnegativedirectionReconoiseDIF3->SetXTitle("DIF for HE- jeta =-20; depth = 3 \b");
          if (kcountHEnegativedirectionReconoiseDIF3 == 11)
            HEnegativedirectionReconoiseDIF3->SetXTitle("DIF for HE- jeta =-19; depth = 3 \b");
          if (kcountHEnegativedirectionReconoiseDIF3 == 12)
            HEnegativedirectionReconoiseDIF3->SetXTitle("DIF for HE- jeta =-18; depth = 3 \b");
          if (kcountHEnegativedirectionReconoiseDIF3 == 13)
            HEnegativedirectionReconoiseDIF3->SetXTitle("DIF for HE- jeta =-17; depth = 3 \b");
          HEnegativedirectionReconoiseDIF3->Draw("Error");
          kcountHEnegativedirectionReconoiseDIF3++;
          if (kcountHEnegativedirectionReconoiseDIF3 > 13)
            break;  // 4x6 = 24
        }           //ccctest>0

      }  // for i
    }    //if(jeta-41  >= -29 && jeta-41 <= -16)
  }      //for jeta
  /////////////////
  c3x5->Update();
  c3x5->Print("DIFreconoiseNegativeDirectionhistD1PhiSymmetryDepth3HE.png");
  c3x5->Clear();
  // clean-up
  if (h2CeffHEnegativedirectionReconoiseDIF3)
    delete h2CeffHEnegativedirectionReconoiseDIF3;
  //========================================================================================== 1117
  //======================================================================
  //======================================================================1D plot: R vs phi , different eta,  depth=4
  //cout<<"      1D plot: R vs phi , different eta,  depth=4 *****" <<endl;
  c3x5->Clear();
  /////////////////
  c3x5->Divide(3, 5);
  c3x5->cd(1);
  int kcountHEnegativedirectionReconoiseDIF4 = 1;
  TH1F* h2CeffHEnegativedirectionReconoiseDIF4 = new TH1F("h2CeffHEnegativedirectionReconoiseDIF4", "", nphi, 0., 72.);

  for (int jeta = 0; jeta < njeta; jeta++) {
    // negativedirectionReconoiseDIF:
    if (jeta - 41 >= -29 && jeta - 41 <= -16) {
      //	     for (int i=0;i<ndepth;i++) {
      // depth=4
      for (int i = 3; i < 4; i++) {
        TH1F* HEnegativedirectionReconoiseDIF4 = (TH1F*)h2CeffHEnegativedirectionReconoiseDIF4->Clone("twod1");

        float ccctest = 0;  // to avoid empty massive elements
        for (int jphi = 0; jphi < nphi; jphi++) {
          double ccc1 = breconoisehe[i][jeta][jphi];
          if (ccc1 != 0.) {
            HEnegativedirectionReconoiseDIF4->Fill(jphi, ccc1);
            ccctest = 1.;  //HEnegativedirectionReconoiseDIF4->SetBinError(i,0.01);
          }
        }  // for jphi
        if (ccctest > 0.) {
          //cout<<"777        kcountHEnegativedirectionReconoiseDIF4   =     "<<kcountHEnegativedirectionReconoiseDIF4  <<"   jeta-41=     "<< jeta-41 <<endl;
          c3x5->cd(kcountHEnegativedirectionReconoiseDIF4);
          HEnegativedirectionReconoiseDIF4->SetMarkerStyle(20);
          HEnegativedirectionReconoiseDIF4->SetMarkerSize(0.4);
          HEnegativedirectionReconoiseDIF4->GetYaxis()->SetLabelSize(0.04);
          HEnegativedirectionReconoiseDIF4->SetXTitle("HEnegativedirectionReconoiseDIF4 \b");
          HEnegativedirectionReconoiseDIF4->SetMarkerColor(2);
          HEnegativedirectionReconoiseDIF4->SetLineColor(0);
          gPad->SetGridy();
          gPad->SetGridx();
          //	   gPad->SetLogy();
          if (kcountHEnegativedirectionReconoiseDIF4 == 1)
            HEnegativedirectionReconoiseDIF4->SetXTitle("DIF for HE- jeta =-28; depth = 4 \b");
          if (kcountHEnegativedirectionReconoiseDIF4 == 2)
            HEnegativedirectionReconoiseDIF4->SetXTitle("DIF for HE- jeta =-27; depth = 4 \b");
          if (kcountHEnegativedirectionReconoiseDIF4 == 3)
            HEnegativedirectionReconoiseDIF4->SetXTitle("DIF for HE- jeta =-26; depth = 4 \b");
          if (kcountHEnegativedirectionReconoiseDIF4 == 4)
            HEnegativedirectionReconoiseDIF4->SetXTitle("DIF for HE- jeta =-25; depth = 4 \b");
          if (kcountHEnegativedirectionReconoiseDIF4 == 5)
            HEnegativedirectionReconoiseDIF4->SetXTitle("DIF for HE- jeta =-24; depth = 4 \b");
          if (kcountHEnegativedirectionReconoiseDIF4 == 6)
            HEnegativedirectionReconoiseDIF4->SetXTitle("DIF for HE- jeta =-23; depth = 4 \b");
          if (kcountHEnegativedirectionReconoiseDIF4 == 7)
            HEnegativedirectionReconoiseDIF4->SetXTitle("DIF for HE- jeta =-22; depth = 4 \b");
          if (kcountHEnegativedirectionReconoiseDIF4 == 8)
            HEnegativedirectionReconoiseDIF4->SetXTitle("DIF for HE- jeta =-21; depth = 4 \b");
          if (kcountHEnegativedirectionReconoiseDIF4 == 9)
            HEnegativedirectionReconoiseDIF4->SetXTitle("DIF for HE- jeta =-20; depth = 4 \b");
          if (kcountHEnegativedirectionReconoiseDIF4 == 10)
            HEnegativedirectionReconoiseDIF4->SetXTitle("DIF for HE- jeta =-19; depth = 4 \b");
          if (kcountHEnegativedirectionReconoiseDIF4 == 11)
            HEnegativedirectionReconoiseDIF4->SetXTitle("DIF for HE- jeta =-18; depth = 4 \b");
          if (kcountHEnegativedirectionReconoiseDIF4 == 12)
            HEnegativedirectionReconoiseDIF4->SetXTitle("DIF for HE- jeta =-16; depth = 4 \b");
          HEnegativedirectionReconoiseDIF4->Draw("Error");
          kcountHEnegativedirectionReconoiseDIF4++;
          if (kcountHEnegativedirectionReconoiseDIF4 > 12)
            break;  // 4x6 = 24
        }           //ccctest>0

      }  // for i
    }    //if(jeta-41  >= -29 && jeta-41 <= -16)
  }      //for jeta
  /////////////////
  c3x5->Update();
  c3x5->Print("DIFreconoiseNegativeDirectionhistD1PhiSymmetryDepth4HE.png");
  c3x5->Clear();
  // clean-up
  if (h2CeffHEnegativedirectionReconoiseDIF4)
    delete h2CeffHEnegativedirectionReconoiseDIF4;
  //========================================================================================== 1118
  //======================================================================
  //======================================================================1D plot: R vs phi , different eta,  depth=5
  //cout<<"      1D plot: R vs phi , different eta,  depth=5 *****" <<endl;
  c3x5->Clear();
  /////////////////
  c3x5->Divide(3, 5);
  c3x5->cd(1);
  int kcountHEnegativedirectionReconoiseDIF5 = 1;
  TH1F* h2CeffHEnegativedirectionReconoiseDIF5 = new TH1F("h2CeffHEnegativedirectionReconoiseDIF5", "", nphi, 0., 72.);

  for (int jeta = 0; jeta < njeta; jeta++) {
    // negativedirectionReconoiseDIF:
    if (jeta - 41 >= -29 && jeta - 41 <= -16) {
      //	     for (int i=0;i<ndepth;i++) {
      // depth=5
      for (int i = 4; i < 5; i++) {
        TH1F* HEnegativedirectionReconoiseDIF5 = (TH1F*)h2CeffHEnegativedirectionReconoiseDIF5->Clone("twod1");

        float ccctest = 0;  // to avoid empty massive elements
        for (int jphi = 0; jphi < nphi; jphi++) {
          //	       cout<<"888  initial      kcountHEnegativedirectionReconoiseDIF5   =     "<<kcountHEnegativedirectionReconoiseDIF5  <<"   jeta-41=     "<< jeta-41 <<"   jphi=     "<< jphi <<"   breconoisehe[i][jeta][jphi]=     "<< breconoisehe[i][jeta][jphi] <<"  depth=     "<< i <<endl;

          double ccc1 = breconoisehe[i][jeta][jphi];
          if (ccc1 != 0.) {
            HEnegativedirectionReconoiseDIF5->Fill(jphi, ccc1);
            ccctest = 1.;  //HEnegativedirectionReconoiseDIF5->SetBinError(i,0.01);
          }
        }  // for jphi
        if (ccctest > 0.) {
          //cout<<"888        kcountHEnegativedirectionReconoiseDIF5   =     "<<kcountHEnegativedirectionReconoiseDIF5  <<"   jeta-41=     "<< jeta-41 <<endl;
          c3x5->cd(kcountHEnegativedirectionReconoiseDIF5);
          HEnegativedirectionReconoiseDIF5->SetMarkerStyle(20);
          HEnegativedirectionReconoiseDIF5->SetMarkerSize(0.4);
          HEnegativedirectionReconoiseDIF5->GetYaxis()->SetLabelSize(0.04);
          HEnegativedirectionReconoiseDIF5->SetXTitle("HEnegativedirectionReconoiseDIF5 \b");
          HEnegativedirectionReconoiseDIF5->SetMarkerColor(2);
          HEnegativedirectionReconoiseDIF5->SetLineColor(0);
          gPad->SetGridy();
          gPad->SetGridx();
          //	   gPad->SetLogy();
          if (kcountHEnegativedirectionReconoiseDIF5 == 1)
            HEnegativedirectionReconoiseDIF5->SetXTitle("DIF for HE- jeta =-28; depth = 5 \b");
          if (kcountHEnegativedirectionReconoiseDIF5 == 2)
            HEnegativedirectionReconoiseDIF5->SetXTitle("DIF for HE- jeta =-27; depth = 5 \b");
          if (kcountHEnegativedirectionReconoiseDIF5 == 3)
            HEnegativedirectionReconoiseDIF5->SetXTitle("DIF for HE- jeta =-26; depth = 5 \b");
          if (kcountHEnegativedirectionReconoiseDIF5 == 4)
            HEnegativedirectionReconoiseDIF5->SetXTitle("DIF for HE- jeta =-25; depth = 5 \b");
          if (kcountHEnegativedirectionReconoiseDIF5 == 5)
            HEnegativedirectionReconoiseDIF5->SetXTitle("DIF for HE- jeta =-24; depth = 5 \b");
          if (kcountHEnegativedirectionReconoiseDIF5 == 6)
            HEnegativedirectionReconoiseDIF5->SetXTitle("DIF for HE- jeta =-23; depth = 5 \b");
          if (kcountHEnegativedirectionReconoiseDIF5 == 7)
            HEnegativedirectionReconoiseDIF5->SetXTitle("DIF for HE- jeta =-22; depth = 5 \b");
          if (kcountHEnegativedirectionReconoiseDIF5 == 8)
            HEnegativedirectionReconoiseDIF5->SetXTitle("DIF for HE- jeta =-21; depth = 5 \b");
          if (kcountHEnegativedirectionReconoiseDIF5 == 9)
            HEnegativedirectionReconoiseDIF5->SetXTitle("DIF for HE- jeta =-20; depth = 5 \b");
          if (kcountHEnegativedirectionReconoiseDIF5 == 10)
            HEnegativedirectionReconoiseDIF5->SetXTitle("DIF for HE- jeta =-19; depth = 5 \b");
          if (kcountHEnegativedirectionReconoiseDIF5 == 11)
            HEnegativedirectionReconoiseDIF5->SetXTitle("DIF for HE- jeta =-18; depth = 5 \b");
          HEnegativedirectionReconoiseDIF5->Draw("Error");
          kcountHEnegativedirectionReconoiseDIF5++;
          if (kcountHEnegativedirectionReconoiseDIF5 > 11)
            break;  // 4x6 = 24
        }           //ccctest>0

      }  // for i
    }    //if(jeta-41  >= -29 && jeta-41 <= -16)
  }      //for jeta
  /////////////////
  c3x5->Update();
  c3x5->Print("DIFreconoiseNegativeDirectionhistD1PhiSymmetryDepth5HE.png");
  c3x5->Clear();
  // clean-up
  if (h2CeffHEnegativedirectionReconoiseDIF5)
    delete h2CeffHEnegativedirectionReconoiseDIF5;
  //========================================================================================== 1119
  //======================================================================
  //======================================================================1D plot: R vs phi , different eta,  depth=6
  //cout<<"      1D plot: R vs phi , different eta,  depth=6 *****" <<endl;
  c3x5->Clear();
  /////////////////
  c3x5->Divide(3, 5);
  c3x5->cd(1);
  int kcountHEnegativedirectionReconoiseDIF6 = 1;
  TH1F* h2CeffHEnegativedirectionReconoiseDIF6 = new TH1F("h2CeffHEnegativedirectionReconoiseDIF6", "", nphi, 0., 72.);

  for (int jeta = 0; jeta < njeta; jeta++) {
    // negativedirectionReconoiseDIF:
    if (jeta - 41 >= -29 && jeta - 41 <= -16) {
      //	     for (int i=0;i<ndepth;i++) {
      // depth=6
      for (int i = 5; i < 6; i++) {
        TH1F* HEnegativedirectionReconoiseDIF6 = (TH1F*)h2CeffHEnegativedirectionReconoiseDIF6->Clone("twod1");

        float ccctest = 0;  // to avoid empty massive elements
        for (int jphi = 0; jphi < nphi; jphi++) {
          double ccc1 = breconoisehe[i][jeta][jphi];
          if (ccc1 != 0.) {
            HEnegativedirectionReconoiseDIF6->Fill(jphi, ccc1);
            ccctest = 1.;  //HEnegativedirectionReconoiseDIF6->SetBinError(i,0.01);
          }
        }  // for jphi
        if (ccctest > 0.) {
          //cout<<"999        kcountHEnegativedirectionReconoiseDIF6   =     "<<kcountHEnegativedirectionReconoiseDIF6  <<"   jeta-41=     "<< jeta-41 <<endl;
          c3x5->cd(kcountHEnegativedirectionReconoiseDIF6);
          HEnegativedirectionReconoiseDIF6->SetMarkerStyle(20);
          HEnegativedirectionReconoiseDIF6->SetMarkerSize(0.4);
          HEnegativedirectionReconoiseDIF6->GetYaxis()->SetLabelSize(0.04);
          HEnegativedirectionReconoiseDIF6->SetXTitle("HEnegativedirectionReconoiseDIF6 \b");
          HEnegativedirectionReconoiseDIF6->SetMarkerColor(2);
          HEnegativedirectionReconoiseDIF6->SetLineColor(0);
          gPad->SetGridy();
          gPad->SetGridx();
          //	   gPad->SetLogy();
          if (kcountHEnegativedirectionReconoiseDIF6 == 1)
            HEnegativedirectionReconoiseDIF6->SetXTitle("DIF for HE- jeta =-28; depth = 6 \b");
          if (kcountHEnegativedirectionReconoiseDIF6 == 2)
            HEnegativedirectionReconoiseDIF6->SetXTitle("DIF for HE- jeta =-27; depth = 6 \b");
          if (kcountHEnegativedirectionReconoiseDIF6 == 3)
            HEnegativedirectionReconoiseDIF6->SetXTitle("DIF for HE- jeta =-26; depth = 6 \b");
          if (kcountHEnegativedirectionReconoiseDIF6 == 4)
            HEnegativedirectionReconoiseDIF6->SetXTitle("DIF for HE- jeta =-25; depth = 6 \b");
          if (kcountHEnegativedirectionReconoiseDIF6 == 5)
            HEnegativedirectionReconoiseDIF6->SetXTitle("DIF for HE- jeta =-24; depth = 6 \b");
          if (kcountHEnegativedirectionReconoiseDIF6 == 6)
            HEnegativedirectionReconoiseDIF6->SetXTitle("DIF for HE- jeta =-23; depth = 6 \b");
          if (kcountHEnegativedirectionReconoiseDIF6 == 7)
            HEnegativedirectionReconoiseDIF6->SetXTitle("DIF for HE- jeta =-22; depth = 6 \b");
          if (kcountHEnegativedirectionReconoiseDIF6 == 8)
            HEnegativedirectionReconoiseDIF6->SetXTitle("DIF for HE- jeta =-21; depth = 6 \b");
          if (kcountHEnegativedirectionReconoiseDIF6 == 9)
            HEnegativedirectionReconoiseDIF6->SetXTitle("DIF for HE- jeta =-20; depth = 6 \b");
          if (kcountHEnegativedirectionReconoiseDIF6 == 10)
            HEnegativedirectionReconoiseDIF6->SetXTitle("DIF for HE- jeta =-19; depth = 6 \b");
          HEnegativedirectionReconoiseDIF6->Draw("Error");
          kcountHEnegativedirectionReconoiseDIF6++;
          if (kcountHEnegativedirectionReconoiseDIF6 > 10)
            break;  // 4x6 = 24
        }           //ccctest>0

      }  // for i
    }    //if(jeta-41  >= -29 && jeta-41 <= -16)
  }      //for jeta
  /////////////////
  c3x5->Update();
  c3x5->Print("DIFreconoiseNegativeDirectionhistD1PhiSymmetryDepth6HE.png");
  c3x5->Clear();
  // clean-up
  if (h2CeffHEnegativedirectionReconoiseDIF6)
    delete h2CeffHEnegativedirectionReconoiseDIF6;
  //========================================================================================== 11110
  //======================================================================
  //======================================================================1D plot: R vs phi , different eta,  depth=7
  //cout<<"      1D plot: R vs phi , different eta,  depth=7 *****" <<endl;
  c3x5->Clear();
  /////////////////
  c3x5->Divide(3, 5);
  c3x5->cd(1);
  int kcountHEnegativedirectionReconoiseDIF7 = 1;
  TH1F* h2CeffHEnegativedirectionReconoiseDIF7 = new TH1F("h2CeffHEnegativedirectionReconoiseDIF7", "", nphi, 0., 72.);

  for (int jeta = 0; jeta < njeta; jeta++) {
    // negativedirectionReconoiseDIF:
    if (jeta - 41 >= -29 && jeta - 41 <= -16) {
      //	     for (int i=0;i<ndepth;i++) {
      // depth=7
      for (int i = 6; i < 7; i++) {
        TH1F* HEnegativedirectionReconoiseDIF7 = (TH1F*)h2CeffHEnegativedirectionReconoiseDIF7->Clone("twod1");

        float ccctest = 0;  // to avoid empty massive elements
        for (int jphi = 0; jphi < nphi; jphi++) {
          double ccc1 = breconoisehe[i][jeta][jphi];
          if (ccc1 != 0.) {
            HEnegativedirectionReconoiseDIF7->Fill(jphi, ccc1);
            ccctest = 1.;  //HEnegativedirectionReconoiseDIF7->SetBinError(i,0.01);
          }
        }  // for jphi
        if (ccctest > 0.) {
          //cout<<"1010       kcountHEnegativedirectionReconoiseDIF7   =     "<<kcountHEnegativedirectionReconoiseDIF7  <<"   jeta-41=     "<< jeta-41 <<endl;
          c3x5->cd(kcountHEnegativedirectionReconoiseDIF7);
          HEnegativedirectionReconoiseDIF7->SetMarkerStyle(20);
          HEnegativedirectionReconoiseDIF7->SetMarkerSize(0.4);
          HEnegativedirectionReconoiseDIF7->GetYaxis()->SetLabelSize(0.04);
          HEnegativedirectionReconoiseDIF7->SetXTitle("HEnegativedirectionReconoiseDIF7 \b");
          HEnegativedirectionReconoiseDIF7->SetMarkerColor(2);
          HEnegativedirectionReconoiseDIF7->SetLineColor(0);
          gPad->SetGridy();
          gPad->SetGridx();
          //	   gPad->SetLogy();
          if (kcountHEnegativedirectionReconoiseDIF7 == 1)
            HEnegativedirectionReconoiseDIF7->SetXTitle("DIF for HE- jeta =-28; depth = 7 \b");
          if (kcountHEnegativedirectionReconoiseDIF7 == 2)
            HEnegativedirectionReconoiseDIF7->SetXTitle("DIF for HE- jeta =-27; depth = 7 \b");
          if (kcountHEnegativedirectionReconoiseDIF7 == 3)
            HEnegativedirectionReconoiseDIF7->SetXTitle("DIF for HE- jeta =-26; depth = 7 \b");
          HEnegativedirectionReconoiseDIF7->Draw("Error");
          kcountHEnegativedirectionReconoiseDIF7++;
          if (kcountHEnegativedirectionReconoiseDIF7 > 3)
            break;  // 4x6 = 24
        }           //ccctest>0

      }  // for i
    }    //if(jeta-41  >= -29 && jeta-41 <= -16)
  }      //for jeta
  /////////////////
  c3x5->Update();
  c3x5->Print("DIFreconoiseNegativeDirectionhistD1PhiSymmetryDepth7HE.png");
  c3x5->Clear();
  // clean-up
  if (h2CeffHEnegativedirectionReconoiseDIF7)
    delete h2CeffHEnegativedirectionReconoiseDIF7;

  //======================================================================================================================
  //======================================================================================================================
  //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  //======================================================================================================================
  //======================================================================================================================
  //======================================================================================================================

  //                            DDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDD:

  //cout<<"    Start Vaiance: preparation  *****" <<endl;
  TH2F* reconoiseVariance1HE1 = (TH2F*)dir->FindObjectAny("h_recNoiseEnergy2_HE1");
  TH2F* reconoiseVariance0HE1 = (TH2F*)dir->FindObjectAny("h_recNoiseEnergy0_HE1");
  TH2F* reconoiseVarianceHE1 = (TH2F*)reconoiseVariance1HE1->Clone("reconoiseVarianceHE1");
  reconoiseVarianceHE1->Divide(reconoiseVariance1HE1, reconoiseVariance0HE1, 1, 1, "B");
  TH2F* reconoiseVariance1HE2 = (TH2F*)dir->FindObjectAny("h_recNoiseEnergy2_HE2");
  TH2F* reconoiseVariance0HE2 = (TH2F*)dir->FindObjectAny("h_recNoiseEnergy0_HE2");
  TH2F* reconoiseVarianceHE2 = (TH2F*)reconoiseVariance1HE2->Clone("reconoiseVarianceHE2");
  reconoiseVarianceHE2->Divide(reconoiseVariance1HE2, reconoiseVariance0HE2, 1, 1, "B");
  TH2F* reconoiseVariance1HE3 = (TH2F*)dir->FindObjectAny("h_recNoiseEnergy2_HE3");
  TH2F* reconoiseVariance0HE3 = (TH2F*)dir->FindObjectAny("h_recNoiseEnergy0_HE3");
  TH2F* reconoiseVarianceHE3 = (TH2F*)reconoiseVariance1HE3->Clone("reconoiseVarianceHE3");
  reconoiseVarianceHE3->Divide(reconoiseVariance1HE3, reconoiseVariance0HE3, 1, 1, "B");
  TH2F* reconoiseVariance1HE4 = (TH2F*)dir->FindObjectAny("h_recNoiseEnergy2_HE4");
  TH2F* reconoiseVariance0HE4 = (TH2F*)dir->FindObjectAny("h_recNoiseEnergy0_HE4");
  TH2F* reconoiseVarianceHE4 = (TH2F*)reconoiseVariance1HE4->Clone("reconoiseVarianceHE4");
  reconoiseVarianceHE4->Divide(reconoiseVariance1HE4, reconoiseVariance0HE4, 1, 1, "B");
  TH2F* reconoiseVariance1HE5 = (TH2F*)dir->FindObjectAny("h_recNoiseEnergy2_HE5");
  TH2F* reconoiseVariance0HE5 = (TH2F*)dir->FindObjectAny("h_recNoiseEnergy0_HE5");
  TH2F* reconoiseVarianceHE5 = (TH2F*)reconoiseVariance1HE5->Clone("reconoiseVarianceHE5");
  reconoiseVarianceHE5->Divide(reconoiseVariance1HE5, reconoiseVariance0HE5, 1, 1, "B");
  TH2F* reconoiseVariance1HE6 = (TH2F*)dir->FindObjectAny("h_recNoiseEnergy2_HE6");
  TH2F* reconoiseVariance0HE6 = (TH2F*)dir->FindObjectAny("h_recNoiseEnergy0_HE6");
  TH2F* reconoiseVarianceHE6 = (TH2F*)reconoiseVariance1HE6->Clone("reconoiseVarianceHE6");
  reconoiseVarianceHE6->Divide(reconoiseVariance1HE6, reconoiseVariance0HE6, 1, 1, "B");
  TH2F* reconoiseVariance1HE7 = (TH2F*)dir->FindObjectAny("h_recNoiseEnergy2_HE7");
  TH2F* reconoiseVariance0HE7 = (TH2F*)dir->FindObjectAny("h_recNoiseEnergy0_HE7");
  TH2F* reconoiseVarianceHE7 = (TH2F*)reconoiseVariance1HE7->Clone("reconoiseVarianceHE7");
  reconoiseVarianceHE7->Divide(reconoiseVariance1HE7, reconoiseVariance0HE7, 1, 1, "B");
  //cout<<"      Vaiance: preparation DONE *****" <<endl;
  //====================================================================== put Vaiance=Dispersia = Sig**2=<R**2> - (<R>)**2 into massive reconoisevariancehe
  //                                                                                           = sum(R*R)/N - (sum(R)/N)**2
  for (int jeta = 0; jeta < njeta; jeta++) {
    if ((jeta - 41 >= -29 && jeta - 41 <= -16) || (jeta - 41 >= 15 && jeta - 41 <= 28)) {
      //preparation for PHI normalization:
      double sumreconoiseHE0 = 0;
      int nsumreconoiseHE0 = 0;
      double sumreconoiseHE1 = 0;
      int nsumreconoiseHE1 = 0;
      double sumreconoiseHE2 = 0;
      int nsumreconoiseHE2 = 0;
      double sumreconoiseHE3 = 0;
      int nsumreconoiseHE3 = 0;
      double sumreconoiseHE4 = 0;
      int nsumreconoiseHE4 = 0;
      double sumreconoiseHE5 = 0;
      int nsumreconoiseHE5 = 0;
      double sumreconoiseHE6 = 0;
      int nsumreconoiseHE6 = 0;
      for (int jphi = 0; jphi < njphi; jphi++) {
        reconoisevariancehe[0][jeta][jphi] = reconoiseVarianceHE1->GetBinContent(jeta + 1, jphi + 1);
        reconoisevariancehe[1][jeta][jphi] = reconoiseVarianceHE2->GetBinContent(jeta + 1, jphi + 1);
        reconoisevariancehe[2][jeta][jphi] = reconoiseVarianceHE3->GetBinContent(jeta + 1, jphi + 1);
        reconoisevariancehe[3][jeta][jphi] = reconoiseVarianceHE4->GetBinContent(jeta + 1, jphi + 1);
        reconoisevariancehe[4][jeta][jphi] = reconoiseVarianceHE5->GetBinContent(jeta + 1, jphi + 1);
        reconoisevariancehe[5][jeta][jphi] = reconoiseVarianceHE6->GetBinContent(jeta + 1, jphi + 1);
        reconoisevariancehe[6][jeta][jphi] = reconoiseVarianceHE7->GetBinContent(jeta + 1, jphi + 1);
        if (reconoisevariancehe[0][jeta][jphi] != 0.) {
          sumreconoiseHE0 += reconoisevariancehe[0][jeta][jphi];
          ++nsumreconoiseHE0;
        }
        if (reconoisevariancehe[1][jeta][jphi] != 0.) {
          sumreconoiseHE1 += reconoisevariancehe[1][jeta][jphi];
          ++nsumreconoiseHE1;
        }
        if (reconoisevariancehe[2][jeta][jphi] != 0.) {
          sumreconoiseHE2 += reconoisevariancehe[2][jeta][jphi];
          ++nsumreconoiseHE2;
        }
        if (reconoisevariancehe[3][jeta][jphi] != 0.) {
          sumreconoiseHE3 += reconoisevariancehe[3][jeta][jphi];
          ++nsumreconoiseHE3;
        }
        if (reconoisevariancehe[4][jeta][jphi] != 0.) {
          sumreconoiseHE4 += reconoisevariancehe[4][jeta][jphi];
          ++nsumreconoiseHE4;
        }
        if (reconoisevariancehe[5][jeta][jphi] != 0.) {
          sumreconoiseHE5 += reconoisevariancehe[5][jeta][jphi];
          ++nsumreconoiseHE5;
        }
        if (reconoisevariancehe[6][jeta][jphi] != 0.) {
          sumreconoiseHE6 += reconoisevariancehe[6][jeta][jphi];
          ++nsumreconoiseHE6;
        }
      }  // phi
      // PHI normalization :
      for (int jphi = 0; jphi < njphi; jphi++) {
        if (sumreconoiseHE0 != 0.)
          reconoisevariancehe[0][jeta][jphi] /= (sumreconoiseHE0 / nsumreconoiseHE0);
        if (sumreconoiseHE1 != 0.)
          reconoisevariancehe[1][jeta][jphi] /= (sumreconoiseHE1 / nsumreconoiseHE1);
        if (sumreconoiseHE2 != 0.)
          reconoisevariancehe[2][jeta][jphi] /= (sumreconoiseHE2 / nsumreconoiseHE2);
        if (sumreconoiseHE3 != 0.)
          reconoisevariancehe[3][jeta][jphi] /= (sumreconoiseHE3 / nsumreconoiseHE3);
        if (sumreconoiseHE4 != 0.)
          reconoisevariancehe[4][jeta][jphi] /= (sumreconoiseHE4 / nsumreconoiseHE4);
        if (sumreconoiseHE5 != 0.)
          reconoisevariancehe[5][jeta][jphi] /= (sumreconoiseHE5 / nsumreconoiseHE5);
        if (sumreconoiseHE6 != 0.)
          reconoisevariancehe[6][jeta][jphi] /= (sumreconoiseHE6 / nsumreconoiseHE6);
      }  // phi
      //       reconoisevariancehe (D)           = sum(R*R)/N - (sum(R)/N)**2
      for (int jphi = 0; jphi < njphi; jphi++) {
        //	   cout<<"12 12 12   jeta=     "<< jeta <<"   jphi   =     "<<jphi  <<endl;
        reconoisevariancehe[0][jeta][jphi] -= areconoisehe[0][jeta][jphi] * areconoisehe[0][jeta][jphi];
        reconoisevariancehe[0][jeta][jphi] = fabs(reconoisevariancehe[0][jeta][jphi]);
        reconoisevariancehe[1][jeta][jphi] -= areconoisehe[1][jeta][jphi] * areconoisehe[1][jeta][jphi];
        reconoisevariancehe[1][jeta][jphi] = fabs(reconoisevariancehe[1][jeta][jphi]);
        reconoisevariancehe[2][jeta][jphi] -= areconoisehe[2][jeta][jphi] * areconoisehe[2][jeta][jphi];
        reconoisevariancehe[2][jeta][jphi] = fabs(reconoisevariancehe[2][jeta][jphi]);
        reconoisevariancehe[3][jeta][jphi] -= areconoisehe[3][jeta][jphi] * areconoisehe[3][jeta][jphi];
        reconoisevariancehe[3][jeta][jphi] = fabs(reconoisevariancehe[3][jeta][jphi]);
        reconoisevariancehe[4][jeta][jphi] -= areconoisehe[4][jeta][jphi] * areconoisehe[4][jeta][jphi];
        reconoisevariancehe[4][jeta][jphi] = fabs(reconoisevariancehe[4][jeta][jphi]);
        reconoisevariancehe[5][jeta][jphi] -= areconoisehe[5][jeta][jphi] * areconoisehe[5][jeta][jphi];
        reconoisevariancehe[5][jeta][jphi] = fabs(reconoisevariancehe[5][jeta][jphi]);
        reconoisevariancehe[6][jeta][jphi] -= areconoisehe[6][jeta][jphi] * areconoisehe[6][jeta][jphi];
        reconoisevariancehe[6][jeta][jphi] = fabs(reconoisevariancehe[6][jeta][jphi]);
      }
    }
  }
  //cout<<"      Vaiance: DONE*****" <<endl;
  //------------------------  2D-eta/phi-plot: D, averaged over depthes
  //======================================================================
  //======================================================================
  //cout<<"      R2D-eta/phi-plot: D, averaged over depthes *****" <<endl;
  c1x1->Clear();
  /////////////////
  c1x0->Divide(1, 1);
  c1x0->cd(1);
  TH2F* DefzDreconoiseHE42D = new TH2F("DefzDreconoiseHE42D", "", neta, -41., 41., nphi, 0., 72.);
  TH2F* DefzDreconoiseHE42D0 = new TH2F("DefzDreconoiseHE42D0", "", neta, -41., 41., nphi, 0., 72.);
  TH2F* DefzDreconoiseHE42DF = (TH2F*)DefzDreconoiseHE42D0->Clone("DefzDreconoiseHE42DF");
  for (int i = 0; i < ndepth; i++) {
    for (int jeta = 0; jeta < neta; jeta++) {
      if ((jeta - 41 >= -29 && jeta - 41 <= -16) || (jeta - 41 >= 15 && jeta - 41 <= 28)) {
        for (int jphi = 0; jphi < nphi; jphi++) {
          double ccc1 = reconoisevariancehe[i][jeta][jphi];
          int k2plot = jeta - 41;
          int kkk = k2plot;  //if(k2plot >0   kkk=k2plot+1; //-41 +41 !=0
          if (areconoisehe[i][jeta][jphi] > 0.) {
            DefzDreconoiseHE42D->Fill(kkk, jphi, ccc1);
            DefzDreconoiseHE42D0->Fill(kkk, jphi, 1.);
          }
        }
      }
    }
  }
  DefzDreconoiseHE42DF->Divide(DefzDreconoiseHE42D, DefzDreconoiseHE42D0, 1, 1, "B");  // average A
  //    DefzDreconoiseHE1->Sumw2();
  gPad->SetGridy();
  gPad->SetGridx();  //      gPad->SetLogz();
  DefzDreconoiseHE42DF->SetMarkerStyle(20);
  DefzDreconoiseHE42DF->SetMarkerSize(0.4);
  DefzDreconoiseHE42DF->GetZaxis()->SetLabelSize(0.08);
  DefzDreconoiseHE42DF->SetXTitle("<D>_depth       #eta  \b");
  DefzDreconoiseHE42DF->SetYTitle("      #phi \b");
  DefzDreconoiseHE42DF->SetZTitle("<D>_depth \b");
  DefzDreconoiseHE42DF->SetMarkerColor(2);
  DefzDreconoiseHE42DF->SetLineColor(
      0);  //      DefzDreconoiseHE42DF->SetMaximum(1.000);  //      DefzDreconoiseHE42DF->SetMinimum(1.0);
  DefzDreconoiseHE42DF->Draw("COLZ");
  /////////////////
  c1x0->Update();
  c1x0->Print("DreconoiseGeneralD2PhiSymmetryHE.png");
  c1x0->Clear();
  // clean-up
  if (DefzDreconoiseHE42D)
    delete DefzDreconoiseHE42D;
  if (DefzDreconoiseHE42D0)
    delete DefzDreconoiseHE42D0;
  if (DefzDreconoiseHE42DF)
    delete DefzDreconoiseHE42DF;
  //====================================================================== 1D plot: D vs phi , averaged over depthes & eta
  //======================================================================
  //cout<<"      1D plot: D vs phi , averaged over depthes & eta *****" <<endl;
  c1x1->Clear();
  /////////////////
  c1x1->Divide(1, 1);
  c1x1->cd(1);
  TH1F* DefzDreconoiseHE41D = new TH1F("DefzDreconoiseHE41D", "", nphi, 0., 72.);
  TH1F* DefzDreconoiseHE41D0 = new TH1F("DefzDreconoiseHE41D0", "", nphi, 0., 72.);
  TH1F* DefzDreconoiseHE41DF = (TH1F*)DefzDreconoiseHE41D0->Clone("DefzDreconoiseHE41DF");

  for (int jphi = 0; jphi < nphi; jphi++) {
    for (int jeta = 0; jeta < neta; jeta++) {
      if ((jeta - 41 >= -29 && jeta - 41 <= -16) || (jeta - 41 >= 15 && jeta - 41 <= 28)) {
        for (int i = 0; i < ndepth; i++) {
          double ccc1 = reconoisevariancehe[i][jeta][jphi];
          if (areconoisehe[i][jeta][jphi] > 0.) {
            DefzDreconoiseHE41D->Fill(jphi, ccc1);
            DefzDreconoiseHE41D0->Fill(jphi, 1.);
          }
        }
      }
    }
  }
  //     DefzDreconoiseHE41D->Sumw2();DefzDreconoiseHE41D0->Sumw2();

  DefzDreconoiseHE41DF->Divide(DefzDreconoiseHE41D, DefzDreconoiseHE41D0, 1, 1, "B");  // R averaged over depthes & eta
  DefzDreconoiseHE41D0->Sumw2();
  //    for (int jphi=1;jphi<73;jphi++) {DefzDreconoiseHE41DF->SetBinError(jphi,0.01);}
  gPad->SetGridy();
  gPad->SetGridx();  //      gPad->SetLogz();
  DefzDreconoiseHE41DF->SetMarkerStyle(20);
  DefzDreconoiseHE41DF->SetMarkerSize(1.4);
  DefzDreconoiseHE41DF->GetZaxis()->SetLabelSize(0.08);
  DefzDreconoiseHE41DF->SetXTitle("#phi  \b");
  DefzDreconoiseHE41DF->SetYTitle("  <D> \b");
  DefzDreconoiseHE41DF->SetZTitle("<D>_PHI  - AllDepthes \b");
  DefzDreconoiseHE41DF->SetMarkerColor(4);
  DefzDreconoiseHE41DF->SetLineColor(
      4);  // DefzDreconoiseHE41DF->SetMinimum(0.8);     DefzDreconoiseHE41DF->SetMinimum(-0.015);
  DefzDreconoiseHE41DF->Draw("Error");
  /////////////////
  c1x1->Update();
  c1x1->Print("DreconoiseGeneralD1PhiSymmetryHE.png");
  c1x1->Clear();
  // clean-up
  if (DefzDreconoiseHE41D)
    delete DefzDreconoiseHE41D;
  if (DefzDreconoiseHE41D0)
    delete DefzDreconoiseHE41D0;
  if (DefzDreconoiseHE41DF)
    delete DefzDreconoiseHE41DF;
  //========================================================================================== 14
  //======================================================================
  //======================================================================1D plot: D vs phi , different eta,  depth=1
  //cout<<"      1D plot: D vs phi , different eta,  depth=1 *****" <<endl;
  c3x5->Clear();
  /////////////////
  c3x5->Divide(3, 5);
  c3x5->cd(1);
  int kcountHEpositivedirectionReconoiseD1 = 1;
  TH1F* h2CeffHEpositivedirectionReconoiseD1 = new TH1F("h2CeffHEpositivedirectionReconoiseD1", "", nphi, 0., 72.);

  for (int jeta = 0; jeta < njeta; jeta++) {
    // positivedirectionReconoiseD:
    if (jeta - 41 >= 15 && jeta - 41 <= 28) {
      //	     for (int i=0;i<ndepth;i++) {
      // depth=1
      for (int i = 0; i < 1; i++) {
        TH1F* HEpositivedirectionReconoiseD1 = (TH1F*)h2CeffHEpositivedirectionReconoiseD1->Clone("twod1");

        float ccctest = 0;  // to avoid empty massive elements
        for (int jphi = 0; jphi < nphi; jphi++) {
          double ccc1 = reconoisevariancehe[i][jeta][jphi];
          if (areconoisehe[i][jeta][jphi] > 0.) {
            HEpositivedirectionReconoiseD1->Fill(jphi, ccc1);
            ccctest = 1.;  //HEpositivedirectionReconoiseD1->SetBinError(i,0.01);
          }
        }  // for jphi
        if (ccctest > 0.) {
          //cout<<"1414       kcountHEpositivedirectionReconoiseD1   =     "<<kcountHEpositivedirectionReconoiseD1  <<"   jeta-41=     "<< jeta-41 <<endl;
          c3x5->cd(kcountHEpositivedirectionReconoiseD1);
          HEpositivedirectionReconoiseD1->SetMarkerStyle(20);
          HEpositivedirectionReconoiseD1->SetMarkerSize(0.4);
          HEpositivedirectionReconoiseD1->GetYaxis()->SetLabelSize(0.04);
          HEpositivedirectionReconoiseD1->SetXTitle("HEpositivedirectionReconoiseD1 \b");
          HEpositivedirectionReconoiseD1->SetMarkerColor(2);
          HEpositivedirectionReconoiseD1->SetLineColor(0);
          gPad->SetGridy();
          gPad->SetGridx();
          //	   gPad->SetLogy();
          if (kcountHEpositivedirectionReconoiseD1 == 1)
            HEpositivedirectionReconoiseD1->SetXTitle("D for HE+ jeta = 17; depth = 1 \b");
          if (kcountHEpositivedirectionReconoiseD1 == 2)
            HEpositivedirectionReconoiseD1->SetXTitle("D for HE+ jeta = 18; depth = 1 \b");
          if (kcountHEpositivedirectionReconoiseD1 == 3)
            HEpositivedirectionReconoiseD1->SetXTitle("D for HE+ jeta = 19; depth = 1 \b");
          if (kcountHEpositivedirectionReconoiseD1 == 4)
            HEpositivedirectionReconoiseD1->SetXTitle("D for HE+ jeta = 20; depth = 1 \b");
          if (kcountHEpositivedirectionReconoiseD1 == 5)
            HEpositivedirectionReconoiseD1->SetXTitle("D for HE+ jeta = 21; depth = 1 \b");
          if (kcountHEpositivedirectionReconoiseD1 == 6)
            HEpositivedirectionReconoiseD1->SetXTitle("D for HE+ jeta = 22; depth = 1 \b");
          if (kcountHEpositivedirectionReconoiseD1 == 7)
            HEpositivedirectionReconoiseD1->SetXTitle("D for HE+ jeta = 23; depth = 1 \b");
          if (kcountHEpositivedirectionReconoiseD1 == 8)
            HEpositivedirectionReconoiseD1->SetXTitle("D for HE+ jeta = 24; depth = 1 \b");
          if (kcountHEpositivedirectionReconoiseD1 == 9)
            HEpositivedirectionReconoiseD1->SetXTitle("D for HE+ jeta = 25; depth = 1 \b");
          if (kcountHEpositivedirectionReconoiseD1 == 10)
            HEpositivedirectionReconoiseD1->SetXTitle("D for HE+ jeta = 26; depth = 1 \b");
          if (kcountHEpositivedirectionReconoiseD1 == 11)
            HEpositivedirectionReconoiseD1->SetXTitle("D for HE+ jeta = 27; depth = 1 \b");
          if (kcountHEpositivedirectionReconoiseD1 == 12)
            HEpositivedirectionReconoiseD1->SetXTitle("D for HE+ jeta = 28; depth = 1 \b");
          HEpositivedirectionReconoiseD1->Draw("Error");
          kcountHEpositivedirectionReconoiseD1++;
          if (kcountHEpositivedirectionReconoiseD1 > 12)
            break;  // 4x6 = 24
        }           //ccctest>0

      }  // for i
    }    //if(jeta-41 >= 0)
  }      //for jeta
  /////////////////
  c3x5->Update();
  c3x5->Print("DreconoisePositiveDirectionhistD1PhiSymmetryDepth1HE.png");
  c3x5->Clear();
  // clean-up
  if (h2CeffHEpositivedirectionReconoiseD1)
    delete h2CeffHEpositivedirectionReconoiseD1;
  //========================================================================================== 15
  //======================================================================
  //======================================================================1D plot: D vs phi , different eta,  depth=2
  //cout<<"      1D plot: D vs phi , different eta,  depth=2 *****" <<endl;
  c3x5->Clear();
  c3x5->Divide(3, 5);
  c3x5->cd(1);
  int kcountHEpositivedirectionReconoiseD2 = 1;
  TH1F* h2CeffHEpositivedirectionReconoiseD2 = new TH1F("h2CeffHEpositivedirectionReconoiseD2", "", nphi, 0., 72.);

  for (int jeta = 0; jeta < njeta; jeta++) {
    // positivedirectionReconoiseD:
    if (jeta - 41 >= 15 && jeta - 41 <= 28) {
      //	     for (int i=0;i<ndepth;i++) {
      // depth=2
      for (int i = 1; i < 2; i++) {
        TH1F* HEpositivedirectionReconoiseD2 = (TH1F*)h2CeffHEpositivedirectionReconoiseD2->Clone("twod1");

        float ccctest = 0;  // to avoid empty massive elements
        for (int jphi = 0; jphi < nphi; jphi++) {
          double ccc1 = reconoisevariancehe[i][jeta][jphi];
          if (areconoisehe[i][jeta][jphi] > 0.) {
            HEpositivedirectionReconoiseD2->Fill(jphi, ccc1);
            ccctest = 1.;  //HEpositivedirectionReconoiseD2->SetBinError(i,0.01);
          }
        }  // for jphi
        if (ccctest > 0.) {
          //cout<<"1515       kcountHEpositivedirectionReconoiseD2   =     "<<kcountHEpositivedirectionReconoiseD2  <<"   jeta-41=     "<< jeta-41 <<endl;
          c3x5->cd(kcountHEpositivedirectionReconoiseD2);
          HEpositivedirectionReconoiseD2->SetMarkerStyle(20);
          HEpositivedirectionReconoiseD2->SetMarkerSize(0.4);
          HEpositivedirectionReconoiseD2->GetYaxis()->SetLabelSize(0.04);
          HEpositivedirectionReconoiseD2->SetXTitle("HEpositivedirectionReconoiseD2 \b");
          HEpositivedirectionReconoiseD2->SetMarkerColor(2);
          HEpositivedirectionReconoiseD2->SetLineColor(0);
          gPad->SetGridy();
          gPad->SetGridx();
          //	   gPad->SetLogy();
          if (kcountHEpositivedirectionReconoiseD2 == 1)
            HEpositivedirectionReconoiseD2->SetXTitle("D for HE+ jeta = 16; depth = 2 \b");
          if (kcountHEpositivedirectionReconoiseD2 == 2)
            HEpositivedirectionReconoiseD2->SetXTitle("D for HE+ jeta = 17; depth = 2 \b");
          if (kcountHEpositivedirectionReconoiseD2 == 3)
            HEpositivedirectionReconoiseD2->SetXTitle("D for HE+ jeta = 18; depth = 2 \b");
          if (kcountHEpositivedirectionReconoiseD2 == 4)
            HEpositivedirectionReconoiseD2->SetXTitle("D for HE+ jeta = 19; depth = 2 \b");
          if (kcountHEpositivedirectionReconoiseD2 == 5)
            HEpositivedirectionReconoiseD2->SetXTitle("D for HE+ jeta = 20; depth = 2 \b");
          if (kcountHEpositivedirectionReconoiseD2 == 6)
            HEpositivedirectionReconoiseD2->SetXTitle("D for HE+ jeta = 21; depth = 2 \b");
          if (kcountHEpositivedirectionReconoiseD2 == 7)
            HEpositivedirectionReconoiseD2->SetXTitle("D for HE+ jeta = 22; depth = 2 \b");
          if (kcountHEpositivedirectionReconoiseD2 == 8)
            HEpositivedirectionReconoiseD2->SetXTitle("D for HE+ jeta = 23; depth = 2 \b");
          if (kcountHEpositivedirectionReconoiseD2 == 9)
            HEpositivedirectionReconoiseD2->SetXTitle("D for HE+ jeta = 24; depth = 2 \b");
          if (kcountHEpositivedirectionReconoiseD2 == 10)
            HEpositivedirectionReconoiseD2->SetXTitle("D for HE+ jeta = 25; depth = 2 \b");
          if (kcountHEpositivedirectionReconoiseD2 == 11)
            HEpositivedirectionReconoiseD2->SetXTitle("D for HE+ jeta = 26; depth = 2 \b");
          if (kcountHEpositivedirectionReconoiseD2 == 12)
            HEpositivedirectionReconoiseD2->SetXTitle("D for HE+ jeta = 27; depth = 2 \b");
          if (kcountHEpositivedirectionReconoiseD2 == 13)
            HEpositivedirectionReconoiseD2->SetXTitle("D for HE+ jeta = 28; depth = 2 \b");
          HEpositivedirectionReconoiseD2->Draw("Error");
          kcountHEpositivedirectionReconoiseD2++;
          if (kcountHEpositivedirectionReconoiseD2 > 13)
            break;  // 4x6 = 24
        }           //ccctest>0

      }  // for i
    }    //if(jeta-41 >= 0)
  }      //for jeta
  /////////////////
  c3x5->Update();
  c3x5->Print("DreconoisePositiveDirectionhistD1PhiSymmetryDepth2HE.png");
  c3x5->Clear();
  // clean-up
  if (h2CeffHEpositivedirectionReconoiseD2)
    delete h2CeffHEpositivedirectionReconoiseD2;
  //========================================================================================== 16
  //======================================================================
  //======================================================================1D plot: D vs phi , different eta,  depth=3
  //cout<<"      1D plot: D vs phi , different eta,  depth=3 *****" <<endl;
  c3x5->Clear();
  c3x5->Divide(3, 5);
  c3x5->cd(1);
  int kcountHEpositivedirectionReconoiseD3 = 1;
  TH1F* h2CeffHEpositivedirectionReconoiseD3 = new TH1F("h2CeffHEpositivedirectionReconoiseD3", "", nphi, 0., 72.);

  for (int jeta = 0; jeta < njeta; jeta++) {
    // positivedirectionReconoiseD:
    if (jeta - 41 >= 15 && jeta - 41 <= 28) {
      //	     for (int i=0;i<ndepth;i++) {
      // depth=3
      for (int i = 2; i < 3; i++) {
        TH1F* HEpositivedirectionReconoiseD3 = (TH1F*)h2CeffHEpositivedirectionReconoiseD3->Clone("twod1");

        float ccctest = 0;  // to avoid empty massive elements
        for (int jphi = 0; jphi < nphi; jphi++) {
          double ccc1 = reconoisevariancehe[i][jeta][jphi];
          if (areconoisehe[i][jeta][jphi] > 0.) {
            HEpositivedirectionReconoiseD3->Fill(jphi, ccc1);
            ccctest = 1.;  //HEpositivedirectionReconoiseD3->SetBinError(i,0.01);
          }
        }  // for jphi
        if (ccctest > 0.) {
          //cout<<"1616       kcountHEpositivedirectionReconoiseD3   =     "<<kcountHEpositivedirectionReconoiseD3  <<"   jeta-41=     "<< jeta-41 <<endl;
          c3x5->cd(kcountHEpositivedirectionReconoiseD3);
          HEpositivedirectionReconoiseD3->SetMarkerStyle(20);
          HEpositivedirectionReconoiseD3->SetMarkerSize(0.4);
          HEpositivedirectionReconoiseD3->GetYaxis()->SetLabelSize(0.04);
          HEpositivedirectionReconoiseD3->SetXTitle("HEpositivedirectionReconoiseD3 \b");
          HEpositivedirectionReconoiseD3->SetMarkerColor(2);
          HEpositivedirectionReconoiseD3->SetLineColor(0);
          gPad->SetGridy();
          gPad->SetGridx();
          //	   gPad->SetLogy();
          if (kcountHEpositivedirectionReconoiseD3 == 1)
            HEpositivedirectionReconoiseD3->SetXTitle("D for HE+ jeta = 16; depth = 3 \b");
          if (kcountHEpositivedirectionReconoiseD3 == 2)
            HEpositivedirectionReconoiseD3->SetXTitle("D for HE+ jeta = 17; depth = 3 \b");
          if (kcountHEpositivedirectionReconoiseD3 == 3)
            HEpositivedirectionReconoiseD3->SetXTitle("D for HE+ jeta = 18; depth = 3 \b");
          if (kcountHEpositivedirectionReconoiseD3 == 4)
            HEpositivedirectionReconoiseD3->SetXTitle("D for HE+ jeta = 19; depth = 3 \b");
          if (kcountHEpositivedirectionReconoiseD3 == 5)
            HEpositivedirectionReconoiseD3->SetXTitle("D for HE+ jeta = 20; depth = 3 \b");
          if (kcountHEpositivedirectionReconoiseD3 == 6)
            HEpositivedirectionReconoiseD3->SetXTitle("D for HE+ jeta = 21; depth = 3 \b");
          if (kcountHEpositivedirectionReconoiseD3 == 7)
            HEpositivedirectionReconoiseD3->SetXTitle("D for HE+ jeta = 22; depth = 3 \b");
          if (kcountHEpositivedirectionReconoiseD3 == 8)
            HEpositivedirectionReconoiseD3->SetXTitle("D for HE+ jeta = 23; depth = 3 \b");
          if (kcountHEpositivedirectionReconoiseD3 == 9)
            HEpositivedirectionReconoiseD3->SetXTitle("D for HE+ jeta = 24; depth = 3 \b");
          if (kcountHEpositivedirectionReconoiseD3 == 10)
            HEpositivedirectionReconoiseD3->SetXTitle("D for HE+ jeta = 25; depth = 3 \b");
          if (kcountHEpositivedirectionReconoiseD3 == 11)
            HEpositivedirectionReconoiseD3->SetXTitle("D for HE+ jeta = 26; depth = 3 \b");
          if (kcountHEpositivedirectionReconoiseD3 == 12)
            HEpositivedirectionReconoiseD3->SetXTitle("D for HE+ jeta = 27; depth = 3 \b");
          if (kcountHEpositivedirectionReconoiseD3 == 13)
            HEpositivedirectionReconoiseD3->SetXTitle("D for HE+ jeta = 28; depth = 3 \b");
          HEpositivedirectionReconoiseD3->Draw("Error");
          kcountHEpositivedirectionReconoiseD3++;
          if (kcountHEpositivedirectionReconoiseD3 > 13)
            break;  // 4x6 = 24
        }           //ccctest>0

      }  // for i
    }    //if(jeta-41 >= 0)
  }      //for jeta
  /////////////////
  c3x5->Update();
  c3x5->Print("DreconoisePositiveDirectionhistD1PhiSymmetryDepth3HE.png");
  c3x5->Clear();
  // clean-up
  if (h2CeffHEpositivedirectionReconoiseD3)
    delete h2CeffHEpositivedirectionReconoiseD3;
  //========================================================================================== 17
  //======================================================================
  //======================================================================1D plot: D vs phi , different eta,  depth=4
  //cout<<"      1D plot: D vs phi , different eta,  depth=4 *****" <<endl;
  c3x5->Clear();
  c3x5->Divide(3, 5);
  c3x5->cd(1);
  int kcountHEpositivedirectionReconoiseD4 = 1;
  TH1F* h2CeffHEpositivedirectionReconoiseD4 = new TH1F("h2CeffHEpositivedirectionReconoiseD4", "", nphi, 0., 72.);

  for (int jeta = 0; jeta < njeta; jeta++) {
    // positivedirectionReconoiseD:
    if (jeta - 41 >= 15 && jeta - 41 <= 28) {
      //	     for (int i=0;i<ndepth;i++) {
      // depth=4
      for (int i = 3; i < 4; i++) {
        TH1F* HEpositivedirectionReconoiseD4 = (TH1F*)h2CeffHEpositivedirectionReconoiseD4->Clone("twod1");

        float ccctest = 0;  // to avoid empty massive elements
        for (int jphi = 0; jphi < nphi; jphi++) {
          double ccc1 = reconoisevariancehe[i][jeta][jphi];
          if (areconoisehe[i][jeta][jphi] > 0.) {
            HEpositivedirectionReconoiseD4->Fill(jphi, ccc1);
            ccctest = 1.;  //HEpositivedirectionReconoiseD4->SetBinError(i,0.01);
          }
        }  // for jphi
        if (ccctest > 0.) {
          //cout<<"1717       kcountHEpositivedirectionReconoiseD4   =     "<<kcountHEpositivedirectionReconoiseD4  <<"   jeta-41=     "<< jeta-41 <<endl;
          c3x5->cd(kcountHEpositivedirectionReconoiseD4);
          HEpositivedirectionReconoiseD4->SetMarkerStyle(20);
          HEpositivedirectionReconoiseD4->SetMarkerSize(0.4);
          HEpositivedirectionReconoiseD4->GetYaxis()->SetLabelSize(0.04);
          HEpositivedirectionReconoiseD4->SetXTitle("HEpositivedirectionReconoiseD4 \b");
          HEpositivedirectionReconoiseD4->SetMarkerColor(2);
          HEpositivedirectionReconoiseD4->SetLineColor(0);
          gPad->SetGridy();
          gPad->SetGridx();
          //	   gPad->SetLogy();
          if (kcountHEpositivedirectionReconoiseD4 == 1)
            HEpositivedirectionReconoiseD4->SetXTitle("D for HE+ jeta = 15; depth = 4 \b");
          if (kcountHEpositivedirectionReconoiseD4 == 2)
            HEpositivedirectionReconoiseD4->SetXTitle("D for HE+ jeta = 17; depth = 4 \b");
          if (kcountHEpositivedirectionReconoiseD4 == 3)
            HEpositivedirectionReconoiseD4->SetXTitle("D for HE+ jeta = 18; depth = 4 \b");
          if (kcountHEpositivedirectionReconoiseD4 == 4)
            HEpositivedirectionReconoiseD4->SetXTitle("D for HE+ jeta = 19; depth = 4 \b");
          if (kcountHEpositivedirectionReconoiseD4 == 5)
            HEpositivedirectionReconoiseD4->SetXTitle("D for HE+ jeta = 20; depth = 4 \b");
          if (kcountHEpositivedirectionReconoiseD4 == 6)
            HEpositivedirectionReconoiseD4->SetXTitle("D for HE+ jeta = 21; depth = 4 \b");
          if (kcountHEpositivedirectionReconoiseD4 == 7)
            HEpositivedirectionReconoiseD4->SetXTitle("D for HE+ jeta = 22; depth = 4 \b");
          if (kcountHEpositivedirectionReconoiseD4 == 8)
            HEpositivedirectionReconoiseD4->SetXTitle("D for HE+ jeta = 23; depth = 4 \b");
          if (kcountHEpositivedirectionReconoiseD4 == 9)
            HEpositivedirectionReconoiseD4->SetXTitle("D for HE+ jeta = 24; depth = 4 \b");
          if (kcountHEpositivedirectionReconoiseD4 == 10)
            HEpositivedirectionReconoiseD4->SetXTitle("D for HE+ jeta = 25; depth = 4 \b");
          if (kcountHEpositivedirectionReconoiseD4 == 11)
            HEpositivedirectionReconoiseD4->SetXTitle("D for HE+ jeta = 26; depth = 4 \b");
          if (kcountHEpositivedirectionReconoiseD4 == 12)
            HEpositivedirectionReconoiseD4->SetXTitle("D for HE+ jeta = 27; depth = 4 \b");
          HEpositivedirectionReconoiseD4->Draw("Error");
          kcountHEpositivedirectionReconoiseD4++;
          if (kcountHEpositivedirectionReconoiseD4 > 12)
            break;  // 4x6 = 24
        }           //ccctest>0

      }  // for i
    }    //if(jeta-41 >= 0)
  }      //for jeta
  /////////////////
  c3x5->Update();
  c3x5->Print("DreconoisePositiveDirectionhistD1PhiSymmetryDepth4HE.png");
  c3x5->Clear();
  // clean-up
  if (h2CeffHEpositivedirectionReconoiseD4)
    delete h2CeffHEpositivedirectionReconoiseD4;
  //========================================================================================== 18
  //======================================================================
  //======================================================================1D plot: D vs phi , different eta,  depth=5
  //cout<<"      1D plot: D vs phi , different eta,  depth=5 *****" <<endl;
  c3x5->Clear();
  c3x5->Divide(3, 5);
  c3x5->cd(1);
  int kcountHEpositivedirectionReconoiseD5 = 1;
  TH1F* h2CeffHEpositivedirectionReconoiseD5 = new TH1F("h2CeffHEpositivedirectionReconoiseD5", "", nphi, 0., 72.);

  for (int jeta = 0; jeta < njeta; jeta++) {
    // positivedirectionReconoiseD:
    if (jeta - 41 >= 15 && jeta - 41 <= 28) {
      //	     for (int i=0;i<ndepth;i++) {
      // depth=5
      for (int i = 4; i < 5; i++) {
        TH1F* HEpositivedirectionReconoiseD5 = (TH1F*)h2CeffHEpositivedirectionReconoiseD5->Clone("twod1");

        float ccctest = 0;  // to avoid empty massive elements
        for (int jphi = 0; jphi < nphi; jphi++) {
          double ccc1 = reconoisevariancehe[i][jeta][jphi];
          if (areconoisehe[i][jeta][jphi] > 0.) {
            HEpositivedirectionReconoiseD5->Fill(jphi, ccc1);
            ccctest = 1.;  //HEpositivedirectionReconoiseD5->SetBinError(i,0.01);
          }
        }  // for jphi
        if (ccctest > 0.) {
          //cout<<"1818       kcountHEpositivedirectionReconoiseD5   =     "<<kcountHEpositivedirectionReconoiseD5  <<"   jeta-41=     "<< jeta-41 <<endl;
          c3x5->cd(kcountHEpositivedirectionReconoiseD5);
          HEpositivedirectionReconoiseD5->SetMarkerStyle(20);
          HEpositivedirectionReconoiseD5->SetMarkerSize(0.4);
          HEpositivedirectionReconoiseD5->GetYaxis()->SetLabelSize(0.04);
          HEpositivedirectionReconoiseD5->SetXTitle("HEpositivedirectionReconoiseD5 \b");
          HEpositivedirectionReconoiseD5->SetMarkerColor(2);
          HEpositivedirectionReconoiseD5->SetLineColor(0);
          gPad->SetGridy();
          gPad->SetGridx();
          //	   gPad->SetLogy();
          if (kcountHEpositivedirectionReconoiseD5 == 1)
            HEpositivedirectionReconoiseD5->SetXTitle("D for HE+ jeta = 17; depth = 5 \b");
          if (kcountHEpositivedirectionReconoiseD5 == 2)
            HEpositivedirectionReconoiseD5->SetXTitle("D for HE+ jeta = 18; depth = 5 \b");
          if (kcountHEpositivedirectionReconoiseD5 == 3)
            HEpositivedirectionReconoiseD5->SetXTitle("D for HE+ jeta = 19; depth = 5 \b");
          if (kcountHEpositivedirectionReconoiseD5 == 4)
            HEpositivedirectionReconoiseD5->SetXTitle("D for HE+ jeta = 20; depth = 5 \b");
          if (kcountHEpositivedirectionReconoiseD5 == 5)
            HEpositivedirectionReconoiseD5->SetXTitle("D for HE+ jeta = 21; depth = 5 \b");
          if (kcountHEpositivedirectionReconoiseD5 == 6)
            HEpositivedirectionReconoiseD5->SetXTitle("D for HE+ jeta = 22; depth = 5 \b");
          if (kcountHEpositivedirectionReconoiseD5 == 7)
            HEpositivedirectionReconoiseD5->SetXTitle("D for HE+ jeta = 23; depth = 5 \b");
          if (kcountHEpositivedirectionReconoiseD5 == 8)
            HEpositivedirectionReconoiseD5->SetXTitle("D for HE+ jeta = 24; depth = 5 \b");
          if (kcountHEpositivedirectionReconoiseD5 == 9)
            HEpositivedirectionReconoiseD5->SetXTitle("D for HE+ jeta = 25; depth = 5 \b");
          if (kcountHEpositivedirectionReconoiseD5 == 10)
            HEpositivedirectionReconoiseD5->SetXTitle("D for HE+ jeta = 26; depth = 5 \b");
          if (kcountHEpositivedirectionReconoiseD5 == 11)
            HEpositivedirectionReconoiseD5->SetXTitle("D for HE+ jeta = 27; depth = 5 \b");
          HEpositivedirectionReconoiseD5->Draw("Error");
          kcountHEpositivedirectionReconoiseD5++;
          if (kcountHEpositivedirectionReconoiseD5 > 11)
            break;  // 4x6 = 24
        }           //ccctest>0

      }  // for i
    }    //if(jeta-41 >= 0)
  }      //for jeta
  /////////////////
  c3x5->Update();
  c3x5->Print("DreconoisePositiveDirectionhistD1PhiSymmetryDepth5HE.png");
  c3x5->Clear();
  // clean-up
  if (h2CeffHEpositivedirectionReconoiseD5)
    delete h2CeffHEpositivedirectionReconoiseD5;
  //========================================================================================== 19
  //======================================================================
  //======================================================================1D plot: D vs phi , different eta,  depth=6
  //cout<<"      1D plot: D vs phi , different eta,  depth=6 *****" <<endl;
  c3x5->Clear();
  c3x5->Divide(3, 5);
  c3x5->cd(1);
  int kcountHEpositivedirectionReconoiseD6 = 1;
  TH1F* h2CeffHEpositivedirectionReconoiseD6 = new TH1F("h2CeffHEpositivedirectionReconoiseD6", "", nphi, 0., 72.);

  for (int jeta = 0; jeta < njeta; jeta++) {
    // positivedirectionReconoiseD:
    if (jeta - 41 >= 15 && jeta - 41 <= 28) {
      //	     for (int i=0;i<ndepth;i++) {
      // depth=6
      for (int i = 5; i < 6; i++) {
        TH1F* HEpositivedirectionReconoiseD6 = (TH1F*)h2CeffHEpositivedirectionReconoiseD6->Clone("twod1");

        float ccctest = 0;  // to avoid empty massive elements
        for (int jphi = 0; jphi < nphi; jphi++) {
          double ccc1 = reconoisevariancehe[i][jeta][jphi];
          if (areconoisehe[i][jeta][jphi] > 0.) {
            HEpositivedirectionReconoiseD6->Fill(jphi, ccc1);
            ccctest = 1.;  //HEpositivedirectionReconoiseD6->SetBinError(i,0.01);
          }
        }  // for jphi
        if (ccctest > 0.) {
          //cout<<"1919       kcountHEpositivedirectionReconoiseD6   =     "<<kcountHEpositivedirectionReconoiseD6  <<"   jeta-41=     "<< jeta-41 <<endl;
          c3x5->cd(kcountHEpositivedirectionReconoiseD6);
          HEpositivedirectionReconoiseD6->SetMarkerStyle(20);
          HEpositivedirectionReconoiseD6->SetMarkerSize(0.4);
          HEpositivedirectionReconoiseD6->GetYaxis()->SetLabelSize(0.04);
          HEpositivedirectionReconoiseD6->SetXTitle("HEpositivedirectionReconoiseD6 \b");
          HEpositivedirectionReconoiseD6->SetMarkerColor(2);
          HEpositivedirectionReconoiseD6->SetLineColor(0);
          gPad->SetGridy();
          gPad->SetGridx();
          //	   gPad->SetLogy();
          if (kcountHEpositivedirectionReconoiseD6 == 1)
            HEpositivedirectionReconoiseD6->SetXTitle("D for HE+ jeta = 18; depth = 6 \b");
          if (kcountHEpositivedirectionReconoiseD6 == 2)
            HEpositivedirectionReconoiseD6->SetXTitle("D for HE+ jeta = 19; depth = 6 \b");
          if (kcountHEpositivedirectionReconoiseD6 == 3)
            HEpositivedirectionReconoiseD6->SetXTitle("D for HE+ jeta = 20; depth = 6 \b");
          if (kcountHEpositivedirectionReconoiseD6 == 4)
            HEpositivedirectionReconoiseD6->SetXTitle("D for HE+ jeta = 21; depth = 6 \b");
          if (kcountHEpositivedirectionReconoiseD6 == 5)
            HEpositivedirectionReconoiseD6->SetXTitle("D for HE+ jeta = 22; depth = 6 \b");
          if (kcountHEpositivedirectionReconoiseD6 == 6)
            HEpositivedirectionReconoiseD6->SetXTitle("D for HE+ jeta = 23; depth = 6 \b");
          if (kcountHEpositivedirectionReconoiseD6 == 7)
            HEpositivedirectionReconoiseD6->SetXTitle("D for HE+ jeta = 24; depth = 6 \b");
          if (kcountHEpositivedirectionReconoiseD6 == 8)
            HEpositivedirectionReconoiseD6->SetXTitle("D for HE+ jeta = 25; depth = 6 \b");
          if (kcountHEpositivedirectionReconoiseD6 == 9)
            HEpositivedirectionReconoiseD6->SetXTitle("D for HE+ jeta = 26; depth = 6 \b");
          if (kcountHEpositivedirectionReconoiseD6 == 10)
            HEpositivedirectionReconoiseD6->SetXTitle("D for HE+ jeta = 27; depth = 6 \b");
          HEpositivedirectionReconoiseD6->Draw("Error");
          kcountHEpositivedirectionReconoiseD6++;
          if (kcountHEpositivedirectionReconoiseD6 > 10)
            break;  // 4x6 = 24
        }           //ccctest>0

      }  // for i
    }    //if(jeta-41 >= 0)
  }      //for jeta
  /////////////////
  c3x5->Update();
  c3x5->Print("DreconoisePositiveDirectionhistD1PhiSymmetryDepth6HE.png");
  c3x5->Clear();
  // clean-up
  if (h2CeffHEpositivedirectionReconoiseD6)
    delete h2CeffHEpositivedirectionReconoiseD6;
  //========================================================================================== 20
  //======================================================================
  //======================================================================1D plot: D vs phi , different eta,  depth=7
  //cout<<"      1D plot: D vs phi , different eta,  depth=7 *****" <<endl;
  c3x5->Clear();
  c3x5->Divide(3, 5);
  c3x5->cd(1);
  int kcountHEpositivedirectionReconoiseD7 = 1;
  TH1F* h2CeffHEpositivedirectionReconoiseD7 = new TH1F("h2CeffHEpositivedirectionReconoiseD7", "", nphi, 0., 72.);

  for (int jeta = 0; jeta < njeta; jeta++) {
    // positivedirectionReconoiseD:
    if (jeta - 41 >= 15 && jeta - 41 <= 28) {
      //	     for (int i=0;i<ndepth;i++) {
      // depth=7
      for (int i = 6; i < 7; i++) {
        TH1F* HEpositivedirectionReconoiseD7 = (TH1F*)h2CeffHEpositivedirectionReconoiseD7->Clone("twod1");

        float ccctest = 0;  // to avoid empty massive elements
        for (int jphi = 0; jphi < nphi; jphi++) {
          double ccc1 = reconoisevariancehe[i][jeta][jphi];
          if (areconoisehe[i][jeta][jphi] > 0.) {
            HEpositivedirectionReconoiseD7->Fill(jphi, ccc1);
            ccctest = 1.;  //HEpositivedirectionReconoiseD7->SetBinError(i,0.01);
          }
        }  // for jphi
        if (ccctest != 0.) {
          //cout<<"2020       kcountHEpositivedirectionReconoiseD7   =     "<<kcountHEpositivedirectionReconoiseD7  <<"   jeta-41=     "<< jeta-41 <<endl;
          c3x5->cd(kcountHEpositivedirectionReconoiseD7);
          HEpositivedirectionReconoiseD7->SetMarkerStyle(20);
          HEpositivedirectionReconoiseD7->SetMarkerSize(0.4);
          HEpositivedirectionReconoiseD7->GetYaxis()->SetLabelSize(0.04);
          HEpositivedirectionReconoiseD7->SetXTitle("HEpositivedirectionReconoiseD7 \b");
          HEpositivedirectionReconoiseD7->SetMarkerColor(2);
          HEpositivedirectionReconoiseD7->SetLineColor(0);
          gPad->SetGridy();
          gPad->SetGridx();
          //	   gPad->SetLogy();
          if (kcountHEpositivedirectionReconoiseD7 == 1)
            HEpositivedirectionReconoiseD7->SetXTitle("D for HE+ jeta = 25; depth = 7 \b");
          if (kcountHEpositivedirectionReconoiseD7 == 2)
            HEpositivedirectionReconoiseD7->SetXTitle("D for HE+ jeta = 26; depth = 7 \b");
          if (kcountHEpositivedirectionReconoiseD7 == 3)
            HEpositivedirectionReconoiseD7->SetXTitle("D for HE+ jeta = 27; depth = 7 \b");
          HEpositivedirectionReconoiseD7->Draw("Error");
          kcountHEpositivedirectionReconoiseD7++;
          if (kcountHEpositivedirectionReconoiseD7 > 3)
            break;  // 4x6 = 24
        }           //ccctest>0

      }  // for i
    }    //if(jeta-41 >= 0)
  }      //for jeta
  /////////////////
  c3x5->Update();
  c3x5->Print("DreconoisePositiveDirectionhistD1PhiSymmetryDepth7HE.png");
  c3x5->Clear();
  // clean-up
  if (h2CeffHEpositivedirectionReconoiseD7)
    delete h2CeffHEpositivedirectionReconoiseD7;

  //========================================================================================== 22222214
  //======================================================================
  //======================================================================1D plot: D vs phi , different eta,  depth=1
  //cout<<"      1D plot: D vs phi , different eta,  depth=1 *****" <<endl;
  c3x5->Clear();
  /////////////////
  c3x5->Divide(3, 5);
  c3x5->cd(1);
  int kcountHEnegativedirectionReconoiseD1 = 1;
  TH1F* h2CeffHEnegativedirectionReconoiseD1 = new TH1F("h2CeffHEnegativedirectionReconoiseD1", "", nphi, 0., 72.);

  for (int jeta = 0; jeta < njeta; jeta++) {
    // negativedirectionReconoiseD:
    if (jeta - 41 >= -29 && jeta - 41 <= -16) {
      //	     for (int i=0;i<ndepth;i++) {
      // depth=1
      for (int i = 0; i < 1; i++) {
        TH1F* HEnegativedirectionReconoiseD1 = (TH1F*)h2CeffHEnegativedirectionReconoiseD1->Clone("twod1");

        float ccctest = 0;  // to avoid empty massive elements
        for (int jphi = 0; jphi < nphi; jphi++) {
          double ccc1 = reconoisevariancehe[i][jeta][jphi];
          if (areconoisehe[i][jeta][jphi] > 0.) {
            HEnegativedirectionReconoiseD1->Fill(jphi, ccc1);
            ccctest = 1.;  //HEnegativedirectionReconoiseD1->SetBinError(i,0.01);
          }
        }  // for jphi
        if (ccctest > 0.) {
          //cout<<"1414       kcountHEnegativedirectionReconoiseD1   =     "<<kcountHEnegativedirectionReconoiseD1  <<"   jeta-41=     "<< jeta-41 <<endl;
          c3x5->cd(kcountHEnegativedirectionReconoiseD1);
          HEnegativedirectionReconoiseD1->SetMarkerStyle(20);
          HEnegativedirectionReconoiseD1->SetMarkerSize(0.4);
          HEnegativedirectionReconoiseD1->GetYaxis()->SetLabelSize(0.04);
          HEnegativedirectionReconoiseD1->SetXTitle("HEnegativedirectionReconoiseD1 \b");
          HEnegativedirectionReconoiseD1->SetMarkerColor(2);
          HEnegativedirectionReconoiseD1->SetLineColor(0);
          gPad->SetGridy();
          gPad->SetGridx();
          //	   gPad->SetLogy();
          if (kcountHEnegativedirectionReconoiseD1 == 1)
            HEnegativedirectionReconoiseD1->SetXTitle("D for HE- jeta =-29; depth = 1 \b");
          if (kcountHEnegativedirectionReconoiseD1 == 2)
            HEnegativedirectionReconoiseD1->SetXTitle("D for HE- jeta =-28; depth = 1 \b");
          if (kcountHEnegativedirectionReconoiseD1 == 3)
            HEnegativedirectionReconoiseD1->SetXTitle("D for HE- jeta =-27; depth = 1 \b");
          if (kcountHEnegativedirectionReconoiseD1 == 4)
            HEnegativedirectionReconoiseD1->SetXTitle("D for HE- jeta =-26; depth = 1 \b");
          if (kcountHEnegativedirectionReconoiseD1 == 5)
            HEnegativedirectionReconoiseD1->SetXTitle("D for HE- jeta =-25; depth = 1 \b");
          if (kcountHEnegativedirectionReconoiseD1 == 6)
            HEnegativedirectionReconoiseD1->SetXTitle("D for HE- jeta =-24; depth = 1 \b");
          if (kcountHEnegativedirectionReconoiseD1 == 7)
            HEnegativedirectionReconoiseD1->SetXTitle("D for HE- jeta =-23; depth = 1 \b");
          if (kcountHEnegativedirectionReconoiseD1 == 8)
            HEnegativedirectionReconoiseD1->SetXTitle("D for HE- jeta =-22; depth = 1 \b");
          if (kcountHEnegativedirectionReconoiseD1 == 9)
            HEnegativedirectionReconoiseD1->SetXTitle("D for HE- jeta =-21; depth = 1 \b");
          if (kcountHEnegativedirectionReconoiseD1 == 10)
            HEnegativedirectionReconoiseD1->SetXTitle("D for HE- jeta =-20; depth = 1 \b");
          if (kcountHEnegativedirectionReconoiseD1 == 11)
            HEnegativedirectionReconoiseD1->SetXTitle("D for HE- jeta =-19; depth = 1 \b");
          if (kcountHEnegativedirectionReconoiseD1 == 12)
            HEnegativedirectionReconoiseD1->SetXTitle("D for HE- jeta =-18; depth = 1 \b");
          HEnegativedirectionReconoiseD1->Draw("Error");
          kcountHEnegativedirectionReconoiseD1++;
          if (kcountHEnegativedirectionReconoiseD1 > 12)
            break;  // 4x6 = 24
        }           //ccctest>0

      }  // for i
    }    //if(jeta-41 < 0)
  }      //for jeta
  /////////////////
  c3x5->Update();
  c3x5->Print("DreconoiseNegativeDirectionhistD1PhiSymmetryDepth1HE.png");
  c3x5->Clear();
  // clean-up
  if (h2CeffHEnegativedirectionReconoiseD1)
    delete h2CeffHEnegativedirectionReconoiseD1;
  //========================================================================================== 22222215
  //======================================================================
  //======================================================================1D plot: D vs phi , different eta,  depth=2
  //cout<<"      1D plot: D vs phi , different eta,  depth=2 *****" <<endl;
  c3x5->Clear();
  c3x5->Divide(3, 5);
  c3x5->cd(1);
  int kcountHEnegativedirectionReconoiseD2 = 1;
  TH1F* h2CeffHEnegativedirectionReconoiseD2 = new TH1F("h2CeffHEnegativedirectionReconoiseD2", "", nphi, 0., 72.);

  for (int jeta = 0; jeta < njeta; jeta++) {
    // negativedirectionReconoiseD:
    if (jeta - 41 >= -29 && jeta - 41 <= -16) {
      //	     for (int i=0;i<ndepth;i++) {
      // depth=2
      for (int i = 1; i < 2; i++) {
        TH1F* HEnegativedirectionReconoiseD2 = (TH1F*)h2CeffHEnegativedirectionReconoiseD2->Clone("twod1");

        float ccctest = 0;  // to avoid empty massive elements
        for (int jphi = 0; jphi < nphi; jphi++) {
          double ccc1 = reconoisevariancehe[i][jeta][jphi];
          if (areconoisehe[i][jeta][jphi] > 0.) {
            HEnegativedirectionReconoiseD2->Fill(jphi, ccc1);
            ccctest = 1.;  //HEnegativedirectionReconoiseD2->SetBinError(i,0.01);
          }
        }  // for jphi
        if (ccctest > 0.) {
          //cout<<"1515       kcountHEnegativedirectionReconoiseD2   =     "<<kcountHEnegativedirectionReconoiseD2  <<"   jeta-41=     "<< jeta-41 <<endl;
          c3x5->cd(kcountHEnegativedirectionReconoiseD2);
          HEnegativedirectionReconoiseD2->SetMarkerStyle(20);
          HEnegativedirectionReconoiseD2->SetMarkerSize(0.4);
          HEnegativedirectionReconoiseD2->GetYaxis()->SetLabelSize(0.04);
          HEnegativedirectionReconoiseD2->SetXTitle("HEnegativedirectionReconoiseD2 \b");
          HEnegativedirectionReconoiseD2->SetMarkerColor(2);
          HEnegativedirectionReconoiseD2->SetLineColor(0);
          gPad->SetGridy();
          gPad->SetGridx();
          //	   gPad->SetLogy();
          if (kcountHEnegativedirectionReconoiseD2 == 1)
            HEnegativedirectionReconoiseD2->SetXTitle("D for HE- jeta =-29; depth = 2 \b");
          if (kcountHEnegativedirectionReconoiseD2 == 2)
            HEnegativedirectionReconoiseD2->SetXTitle("D for HE- jeta =-28; depth = 2 \b");
          if (kcountHEnegativedirectionReconoiseD2 == 3)
            HEnegativedirectionReconoiseD2->SetXTitle("D for HE- jeta =-27; depth = 2 \b");
          if (kcountHEnegativedirectionReconoiseD2 == 4)
            HEnegativedirectionReconoiseD2->SetXTitle("D for HE- jeta =-26; depth = 2 \b");
          if (kcountHEnegativedirectionReconoiseD2 == 5)
            HEnegativedirectionReconoiseD2->SetXTitle("D for HE- jeta =-25; depth = 2 \b");
          if (kcountHEnegativedirectionReconoiseD2 == 6)
            HEnegativedirectionReconoiseD2->SetXTitle("D for HE- jeta =-24; depth = 2 \b");
          if (kcountHEnegativedirectionReconoiseD2 == 7)
            HEnegativedirectionReconoiseD2->SetXTitle("D for HE- jeta =-23; depth = 2 \b");
          if (kcountHEnegativedirectionReconoiseD2 == 8)
            HEnegativedirectionReconoiseD2->SetXTitle("D for HE- jeta =-22; depth = 2 \b");
          if (kcountHEnegativedirectionReconoiseD2 == 9)
            HEnegativedirectionReconoiseD2->SetXTitle("D for HE- jeta =-21; depth = 2 \b");
          if (kcountHEnegativedirectionReconoiseD2 == 10)
            HEnegativedirectionReconoiseD2->SetXTitle("D for HE- jeta =-20; depth = 2 \b");
          if (kcountHEnegativedirectionReconoiseD2 == 11)
            HEnegativedirectionReconoiseD2->SetXTitle("D for HE- jeta =-19; depth = 2 \b");
          if (kcountHEnegativedirectionReconoiseD2 == 12)
            HEnegativedirectionReconoiseD2->SetXTitle("D for HE- jeta =-18; depth = 2 \b");
          if (kcountHEnegativedirectionReconoiseD2 == 13)
            HEnegativedirectionReconoiseD2->SetXTitle("D for HE- jeta =-17; depth = 2 \b");
          HEnegativedirectionReconoiseD2->Draw("Error");
          kcountHEnegativedirectionReconoiseD2++;
          if (kcountHEnegativedirectionReconoiseD2 > 13)
            break;  // 4x6 = 24
        }           //ccctest>0

      }  // for i
    }    //if(jeta-41 < 0)
  }      //for jeta
  /////////////////
  c3x5->Update();
  c3x5->Print("DreconoiseNegativeDirectionhistD1PhiSymmetryDepth2HE.png");
  c3x5->Clear();
  // clean-up
  if (h2CeffHEnegativedirectionReconoiseD2)
    delete h2CeffHEnegativedirectionReconoiseD2;
  //========================================================================================== 22222216
  //======================================================================
  //======================================================================1D plot: D vs phi , different eta,  depth=3
  //cout<<"      1D plot: D vs phi , different eta,  depth=3 *****" <<endl;
  c3x5->Clear();
  c3x5->Divide(3, 5);
  c3x5->cd(1);
  int kcountHEnegativedirectionReconoiseD3 = 1;
  TH1F* h2CeffHEnegativedirectionReconoiseD3 = new TH1F("h2CeffHEnegativedirectionReconoiseD3", "", nphi, 0., 72.);

  for (int jeta = 0; jeta < njeta; jeta++) {
    // negativedirectionReconoiseD:
    if (jeta - 41 >= -29 && jeta - 41 <= -16) {
      //	     for (int i=0;i<ndepth;i++) {
      // depth=3
      for (int i = 2; i < 3; i++) {
        TH1F* HEnegativedirectionReconoiseD3 = (TH1F*)h2CeffHEnegativedirectionReconoiseD3->Clone("twod1");

        float ccctest = 0;  // to avoid empty massive elements
        for (int jphi = 0; jphi < nphi; jphi++) {
          double ccc1 = reconoisevariancehe[i][jeta][jphi];
          if (areconoisehe[i][jeta][jphi] > 0.) {
            HEnegativedirectionReconoiseD3->Fill(jphi, ccc1);
            ccctest = 1.;  //HEnegativedirectionReconoiseD3->SetBinError(i,0.01);
          }
        }  // for jphi
        if (ccctest > 0.) {
          //cout<<"1616       kcountHEnegativedirectionReconoiseD3   =     "<<kcountHEnegativedirectionReconoiseD3  <<"   jeta-41=     "<< jeta-41 <<endl;
          c3x5->cd(kcountHEnegativedirectionReconoiseD3);
          HEnegativedirectionReconoiseD3->SetMarkerStyle(20);
          HEnegativedirectionReconoiseD3->SetMarkerSize(0.4);
          HEnegativedirectionReconoiseD3->GetYaxis()->SetLabelSize(0.04);
          HEnegativedirectionReconoiseD3->SetXTitle("HEnegativedirectionReconoiseD3 \b");
          HEnegativedirectionReconoiseD3->SetMarkerColor(2);
          HEnegativedirectionReconoiseD3->SetLineColor(0);
          gPad->SetGridy();
          gPad->SetGridx();
          //	   gPad->SetLogy();
          if (kcountHEnegativedirectionReconoiseD3 == 1)
            HEnegativedirectionReconoiseD3->SetXTitle("D for HE- jeta =-29; depth = 3 \b");
          if (kcountHEnegativedirectionReconoiseD3 == 2)
            HEnegativedirectionReconoiseD3->SetXTitle("D for HE- jeta =-28; depth = 3 \b");
          if (kcountHEnegativedirectionReconoiseD3 == 3)
            HEnegativedirectionReconoiseD3->SetXTitle("D for HE- jeta =-27; depth = 3 \b");
          if (kcountHEnegativedirectionReconoiseD3 == 4)
            HEnegativedirectionReconoiseD3->SetXTitle("D for HE- jeta =-26; depth = 3 \b");
          if (kcountHEnegativedirectionReconoiseD3 == 5)
            HEnegativedirectionReconoiseD3->SetXTitle("D for HE- jeta =-25; depth = 3 \b");
          if (kcountHEnegativedirectionReconoiseD3 == 6)
            HEnegativedirectionReconoiseD3->SetXTitle("D for HE- jeta =-24; depth = 3 \b");
          if (kcountHEnegativedirectionReconoiseD3 == 7)
            HEnegativedirectionReconoiseD3->SetXTitle("D for HE- jeta =-23; depth = 3 \b");
          if (kcountHEnegativedirectionReconoiseD3 == 8)
            HEnegativedirectionReconoiseD3->SetXTitle("D for HE- jeta =-22; depth = 3 \b");
          if (kcountHEnegativedirectionReconoiseD3 == 9)
            HEnegativedirectionReconoiseD3->SetXTitle("D for HE- jeta =-21; depth = 3 \b");
          if (kcountHEnegativedirectionReconoiseD3 == 10)
            HEnegativedirectionReconoiseD3->SetXTitle("D for HE- jeta =-20; depth = 3 \b");
          if (kcountHEnegativedirectionReconoiseD3 == 11)
            HEnegativedirectionReconoiseD3->SetXTitle("D for HE- jeta =-19; depth = 3 \b");
          if (kcountHEnegativedirectionReconoiseD3 == 12)
            HEnegativedirectionReconoiseD3->SetXTitle("D for HE- jeta =-18; depth = 3 \b");
          if (kcountHEnegativedirectionReconoiseD3 == 13)
            HEnegativedirectionReconoiseD3->SetXTitle("D for HE- jeta =-17; depth = 3 \b");
          HEnegativedirectionReconoiseD3->Draw("Error");
          kcountHEnegativedirectionReconoiseD3++;
          if (kcountHEnegativedirectionReconoiseD3 > 13)
            break;  // 4x6 = 24
        }           //ccctest>0

      }  // for i
    }    //if(jeta-41 < 0)
  }      //for jeta
  /////////////////
  c3x5->Update();
  c3x5->Print("DreconoiseNegativeDirectionhistD1PhiSymmetryDepth3HE.png");
  c3x5->Clear();
  // clean-up
  if (h2CeffHEnegativedirectionReconoiseD3)
    delete h2CeffHEnegativedirectionReconoiseD3;
  //========================================================================================== 22222217
  //======================================================================
  //======================================================================1D plot: D vs phi , different eta,  depth=4
  //cout<<"      1D plot: D vs phi , different eta,  depth=4 *****" <<endl;
  c3x5->Clear();
  c3x5->Divide(3, 5);
  c3x5->cd(1);
  int kcountHEnegativedirectionReconoiseD4 = 1;
  TH1F* h2CeffHEnegativedirectionReconoiseD4 = new TH1F("h2CeffHEnegativedirectionReconoiseD4", "", nphi, 0., 72.);

  for (int jeta = 0; jeta < njeta; jeta++) {
    // negativedirectionReconoiseD:
    if (jeta - 41 >= -29 && jeta - 41 <= -16) {
      //	     for (int i=0;i<ndepth;i++) {
      // depth=4
      for (int i = 3; i < 4; i++) {
        TH1F* HEnegativedirectionReconoiseD4 = (TH1F*)h2CeffHEnegativedirectionReconoiseD4->Clone("twod1");

        float ccctest = 0;  // to avoid empty massive elements
        for (int jphi = 0; jphi < nphi; jphi++) {
          double ccc1 = reconoisevariancehe[i][jeta][jphi];
          if (areconoisehe[i][jeta][jphi] > 0.) {
            HEnegativedirectionReconoiseD4->Fill(jphi, ccc1);
            ccctest = 1.;  //HEnegativedirectionReconoiseD4->SetBinError(i,0.01);
          }
        }  // for jphi
        if (ccctest > 0.) {
          //cout<<"1717       kcountHEnegativedirectionReconoiseD4   =     "<<kcountHEnegativedirectionReconoiseD4  <<"   jeta-41=     "<< jeta-41 <<endl;
          c3x5->cd(kcountHEnegativedirectionReconoiseD4);
          HEnegativedirectionReconoiseD4->SetMarkerStyle(20);
          HEnegativedirectionReconoiseD4->SetMarkerSize(0.4);
          HEnegativedirectionReconoiseD4->GetYaxis()->SetLabelSize(0.04);
          HEnegativedirectionReconoiseD4->SetXTitle("HEnegativedirectionReconoiseD4 \b");
          HEnegativedirectionReconoiseD4->SetMarkerColor(2);
          HEnegativedirectionReconoiseD4->SetLineColor(0);
          gPad->SetGridy();
          gPad->SetGridx();
          //	   gPad->SetLogy();
          if (kcountHEnegativedirectionReconoiseD4 == 1)
            HEnegativedirectionReconoiseD4->SetXTitle("D for HE- jeta =-28; depth = 4 \b");
          if (kcountHEnegativedirectionReconoiseD4 == 2)
            HEnegativedirectionReconoiseD4->SetXTitle("D for HE- jeta =-27; depth = 4 \b");
          if (kcountHEnegativedirectionReconoiseD4 == 3)
            HEnegativedirectionReconoiseD4->SetXTitle("D for HE- jeta =-26; depth = 4 \b");
          if (kcountHEnegativedirectionReconoiseD4 == 4)
            HEnegativedirectionReconoiseD4->SetXTitle("D for HE- jeta =-25; depth = 4 \b");
          if (kcountHEnegativedirectionReconoiseD4 == 5)
            HEnegativedirectionReconoiseD4->SetXTitle("D for HE- jeta =-24; depth = 4 \b");
          if (kcountHEnegativedirectionReconoiseD4 == 6)
            HEnegativedirectionReconoiseD4->SetXTitle("D for HE- jeta =-23; depth = 4 \b");
          if (kcountHEnegativedirectionReconoiseD4 == 7)
            HEnegativedirectionReconoiseD4->SetXTitle("D for HE- jeta =-22; depth = 4 \b");
          if (kcountHEnegativedirectionReconoiseD4 == 8)
            HEnegativedirectionReconoiseD4->SetXTitle("D for HE- jeta =-21; depth = 4 \b");
          if (kcountHEnegativedirectionReconoiseD4 == 9)
            HEnegativedirectionReconoiseD4->SetXTitle("D for HE- jeta =-20; depth = 4 \b");
          if (kcountHEnegativedirectionReconoiseD4 == 10)
            HEnegativedirectionReconoiseD4->SetXTitle("D for HE- jeta =-19; depth = 4 \b");
          if (kcountHEnegativedirectionReconoiseD4 == 11)
            HEnegativedirectionReconoiseD4->SetXTitle("D for HE- jeta =-18; depth = 4 \b");
          if (kcountHEnegativedirectionReconoiseD4 == 12)
            HEnegativedirectionReconoiseD4->SetXTitle("D for HE- jeta =-16; depth = 4 \b");
          HEnegativedirectionReconoiseD4->Draw("Error");
          kcountHEnegativedirectionReconoiseD4++;
          if (kcountHEnegativedirectionReconoiseD4 > 12)
            break;  // 4x6 = 24
        }           //ccctest>0

      }  // for i
    }    //if(jeta-41 < 0)
  }      //for jeta
  /////////////////
  c3x5->Update();
  c3x5->Print("DreconoiseNegativeDirectionhistD1PhiSymmetryDepth4HE.png");
  c3x5->Clear();
  // clean-up
  if (h2CeffHEnegativedirectionReconoiseD4)
    delete h2CeffHEnegativedirectionReconoiseD4;
  //========================================================================================== 22222218
  //======================================================================
  //======================================================================1D plot: D vs phi , different eta,  depth=5
  //cout<<"      1D plot: D vs phi , different eta,  depth=5 *****" <<endl;
  c3x5->Clear();
  c3x5->Divide(3, 5);
  c3x5->cd(1);
  int kcountHEnegativedirectionReconoiseD5 = 1;
  TH1F* h2CeffHEnegativedirectionReconoiseD5 = new TH1F("h2CeffHEnegativedirectionReconoiseD5", "", nphi, 0., 72.);

  for (int jeta = 0; jeta < njeta; jeta++) {
    // negativedirectionReconoiseD:
    if (jeta - 41 >= -29 && jeta - 41 <= -16) {
      //	     for (int i=0;i<ndepth;i++) {
      // depth=5
      for (int i = 4; i < 5; i++) {
        TH1F* HEnegativedirectionReconoiseD5 = (TH1F*)h2CeffHEnegativedirectionReconoiseD5->Clone("twod1");

        float ccctest = 0;  // to avoid empty massive elements
        for (int jphi = 0; jphi < nphi; jphi++) {
          double ccc1 = reconoisevariancehe[i][jeta][jphi];
          if (areconoisehe[i][jeta][jphi] > 0.) {
            HEnegativedirectionReconoiseD5->Fill(jphi, ccc1);
            ccctest = 1.;  //HEnegativedirectionReconoiseD5->SetBinError(i,0.01);
          }
        }  // for jphi
        if (ccctest > 0.) {
          //cout<<"1818       kcountHEnegativedirectionReconoiseD5   =     "<<kcountHEnegativedirectionReconoiseD5  <<"   jeta-41=     "<< jeta-41 <<endl;
          c3x5->cd(kcountHEnegativedirectionReconoiseD5);
          HEnegativedirectionReconoiseD5->SetMarkerStyle(20);
          HEnegativedirectionReconoiseD5->SetMarkerSize(0.4);
          HEnegativedirectionReconoiseD5->GetYaxis()->SetLabelSize(0.04);
          HEnegativedirectionReconoiseD5->SetXTitle("HEnegativedirectionReconoiseD5 \b");
          HEnegativedirectionReconoiseD5->SetMarkerColor(2);
          HEnegativedirectionReconoiseD5->SetLineColor(0);
          gPad->SetGridy();
          gPad->SetGridx();
          //	   gPad->SetLogy();
          if (kcountHEnegativedirectionReconoiseD5 == 1)
            HEnegativedirectionReconoiseD5->SetXTitle("D for HE- jeta =-28; depth = 5 \b");
          if (kcountHEnegativedirectionReconoiseD5 == 2)
            HEnegativedirectionReconoiseD5->SetXTitle("D for HE- jeta =-27; depth = 5 \b");
          if (kcountHEnegativedirectionReconoiseD5 == 3)
            HEnegativedirectionReconoiseD5->SetXTitle("D for HE- jeta =-26; depth = 5 \b");
          if (kcountHEnegativedirectionReconoiseD5 == 4)
            HEnegativedirectionReconoiseD5->SetXTitle("D for HE- jeta =-25; depth = 5 \b");
          if (kcountHEnegativedirectionReconoiseD5 == 5)
            HEnegativedirectionReconoiseD5->SetXTitle("D for HE- jeta =-24; depth = 5 \b");
          if (kcountHEnegativedirectionReconoiseD5 == 6)
            HEnegativedirectionReconoiseD5->SetXTitle("D for HE- jeta =-23; depth = 5 \b");
          if (kcountHEnegativedirectionReconoiseD5 == 7)
            HEnegativedirectionReconoiseD5->SetXTitle("D for HE- jeta =-22; depth = 5 \b");
          if (kcountHEnegativedirectionReconoiseD5 == 8)
            HEnegativedirectionReconoiseD5->SetXTitle("D for HE- jeta =-21; depth = 5 \b");
          if (kcountHEnegativedirectionReconoiseD5 == 9)
            HEnegativedirectionReconoiseD5->SetXTitle("D for HE- jeta =-20; depth = 5 \b");
          if (kcountHEnegativedirectionReconoiseD5 == 10)
            HEnegativedirectionReconoiseD5->SetXTitle("D for HE- jeta =-19; depth = 5 \b");
          if (kcountHEnegativedirectionReconoiseD5 == 11)
            HEnegativedirectionReconoiseD5->SetXTitle("D for HE- jeta =-18; depth = 5 \b");
          HEnegativedirectionReconoiseD5->Draw("Error");
          kcountHEnegativedirectionReconoiseD5++;
          if (kcountHEnegativedirectionReconoiseD5 > 11)
            break;  // 4x6 = 24
        }           //ccctest>0

      }  // for i
    }    //if(jeta-41 < 0)
  }      //for jeta
  /////////////////
  c3x5->Update();
  c3x5->Print("DreconoiseNegativeDirectionhistD1PhiSymmetryDepth5HE.png");
  c3x5->Clear();
  // clean-up
  if (h2CeffHEnegativedirectionReconoiseD5)
    delete h2CeffHEnegativedirectionReconoiseD5;
  //========================================================================================== 22222219
  //======================================================================
  //======================================================================1D plot: D vs phi , different eta,  depth=6
  //cout<<"      1D plot: D vs phi , different eta,  depth=6 *****" <<endl;
  c3x5->Clear();
  c3x5->Divide(3, 5);
  c3x5->cd(1);
  int kcountHEnegativedirectionReconoiseD6 = 1;
  TH1F* h2CeffHEnegativedirectionReconoiseD6 = new TH1F("h2CeffHEnegativedirectionReconoiseD6", "", nphi, 0., 72.);

  for (int jeta = 0; jeta < njeta; jeta++) {
    // negativedirectionReconoiseD:
    if (jeta - 41 >= -29 && jeta - 41 <= -16) {
      //	     for (int i=0;i<ndepth;i++) {
      // depth=6
      for (int i = 5; i < 6; i++) {
        TH1F* HEnegativedirectionReconoiseD6 = (TH1F*)h2CeffHEnegativedirectionReconoiseD6->Clone("twod1");

        float ccctest = 0;  // to avoid empty massive elements
        for (int jphi = 0; jphi < nphi; jphi++) {
          double ccc1 = reconoisevariancehe[i][jeta][jphi];
          if (areconoisehe[i][jeta][jphi] > 0.) {
            HEnegativedirectionReconoiseD6->Fill(jphi, ccc1);
            ccctest = 1.;  //HEnegativedirectionReconoiseD6->SetBinError(i,0.01);
          }
        }  // for jphi
        if (ccctest > 0.) {
          //cout<<"1919       kcountHEnegativedirectionReconoiseD6   =     "<<kcountHEnegativedirectionReconoiseD6  <<"   jeta-41=     "<< jeta-41 <<endl;
          c3x5->cd(kcountHEnegativedirectionReconoiseD6);
          HEnegativedirectionReconoiseD6->SetMarkerStyle(20);
          HEnegativedirectionReconoiseD6->SetMarkerSize(0.4);
          HEnegativedirectionReconoiseD6->GetYaxis()->SetLabelSize(0.04);
          HEnegativedirectionReconoiseD6->SetXTitle("HEnegativedirectionReconoiseD6 \b");
          HEnegativedirectionReconoiseD6->SetMarkerColor(2);
          HEnegativedirectionReconoiseD6->SetLineColor(0);
          gPad->SetGridy();
          gPad->SetGridx();
          //	   gPad->SetLogy();
          if (kcountHEnegativedirectionReconoiseD6 == 1)
            HEnegativedirectionReconoiseD6->SetXTitle("D for HE- jeta =-28; depth = 6 \b");
          if (kcountHEnegativedirectionReconoiseD6 == 2)
            HEnegativedirectionReconoiseD6->SetXTitle("D for HE- jeta =-27; depth = 6 \b");
          if (kcountHEnegativedirectionReconoiseD6 == 3)
            HEnegativedirectionReconoiseD6->SetXTitle("D for HE- jeta =-26; depth = 6 \b");
          if (kcountHEnegativedirectionReconoiseD6 == 4)
            HEnegativedirectionReconoiseD6->SetXTitle("D for HE- jeta =-25; depth = 6 \b");
          if (kcountHEnegativedirectionReconoiseD6 == 5)
            HEnegativedirectionReconoiseD6->SetXTitle("D for HE- jeta =-24; depth = 6 \b");
          if (kcountHEnegativedirectionReconoiseD6 == 6)
            HEnegativedirectionReconoiseD6->SetXTitle("D for HE- jeta =-23; depth = 6 \b");
          if (kcountHEnegativedirectionReconoiseD6 == 7)
            HEnegativedirectionReconoiseD6->SetXTitle("D for HE- jeta =-22; depth = 6 \b");
          if (kcountHEnegativedirectionReconoiseD6 == 8)
            HEnegativedirectionReconoiseD6->SetXTitle("D for HE- jeta =-21; depth = 6 \b");
          if (kcountHEnegativedirectionReconoiseD6 == 9)
            HEnegativedirectionReconoiseD6->SetXTitle("D for HE- jeta =-20; depth = 6 \b");
          if (kcountHEnegativedirectionReconoiseD6 == 10)
            HEnegativedirectionReconoiseD6->SetXTitle("D for HE- jeta =-19; depth = 6 \b");
          HEnegativedirectionReconoiseD6->Draw("Error");
          kcountHEnegativedirectionReconoiseD6++;
          if (kcountHEnegativedirectionReconoiseD6 > 10)
            break;  // 4x6 = 24
        }           //ccctest>0

      }  // for i
    }    //if(jeta-41 < 0)
  }      //for jeta
  /////////////////
  c3x5->Update();
  c3x5->Print("DreconoiseNegativeDirectionhistD1PhiSymmetryDepth6HE.png");
  c3x5->Clear();
  // clean-up
  if (h2CeffHEnegativedirectionReconoiseD6)
    delete h2CeffHEnegativedirectionReconoiseD6;
  //========================================================================================== 22222220
  //======================================================================
  //======================================================================1D plot: D vs phi , different eta,  depth=7
  //cout<<"      1D plot: D vs phi , different eta,  depth=7 *****" <<endl;
  c3x5->Clear();
  c3x5->Divide(3, 5);
  c3x5->cd(1);
  int kcountHEnegativedirectionReconoiseD7 = 1;
  TH1F* h2CeffHEnegativedirectionReconoiseD7 = new TH1F("h2CeffHEnegativedirectionReconoiseD7", "", nphi, 0., 72.);

  for (int jeta = 0; jeta < njeta; jeta++) {
    // negativedirectionReconoiseD:
    if (jeta - 41 >= -29 && jeta - 41 <= -16) {
      //	     for (int i=0;i<ndepth;i++) {
      // depth=7
      for (int i = 6; i < 7; i++) {
        TH1F* HEnegativedirectionReconoiseD7 = (TH1F*)h2CeffHEnegativedirectionReconoiseD7->Clone("twod1");

        float ccctest = 0;  // to avoid empty massive elements
        for (int jphi = 0; jphi < nphi; jphi++) {
          double ccc1 = reconoisevariancehe[i][jeta][jphi];
          if (areconoisehe[i][jeta][jphi] > 0.) {
            HEnegativedirectionReconoiseD7->Fill(jphi, ccc1);
            ccctest = 1.;  //HEnegativedirectionReconoiseD7->SetBinError(i,0.01);
          }
        }  // for jphi
        if (ccctest != 0.) {
          //cout<<"2020       kcountHEnegativedirectionReconoiseD7   =     "<<kcountHEnegativedirectionReconoiseD7  <<"   jeta-41=     "<< jeta-41 <<endl;
          c3x5->cd(kcountHEnegativedirectionReconoiseD7);
          HEnegativedirectionReconoiseD7->SetMarkerStyle(20);
          HEnegativedirectionReconoiseD7->SetMarkerSize(0.4);
          HEnegativedirectionReconoiseD7->GetYaxis()->SetLabelSize(0.04);
          HEnegativedirectionReconoiseD7->SetXTitle("HEnegativedirectionReconoiseD7 \b");
          HEnegativedirectionReconoiseD7->SetMarkerColor(2);
          HEnegativedirectionReconoiseD7->SetLineColor(0);
          gPad->SetGridy();
          gPad->SetGridx();
          //	   gPad->SetLogy();
          if (kcountHEnegativedirectionReconoiseD7 == 1)
            HEnegativedirectionReconoiseD7->SetXTitle("D for HE- jeta =-28; depth = 7 \b");
          if (kcountHEnegativedirectionReconoiseD7 == 2)
            HEnegativedirectionReconoiseD7->SetXTitle("D for HE- jeta =-27; depth = 7 \b");
          if (kcountHEnegativedirectionReconoiseD7 == 3)
            HEnegativedirectionReconoiseD7->SetXTitle("D for HE- jeta =-26; depth = 7 \b");
          HEnegativedirectionReconoiseD7->Draw("Error");
          kcountHEnegativedirectionReconoiseD7++;
          if (kcountHEnegativedirectionReconoiseD7 > 3)
            break;  // 4x6 = 24
        }           //ccctest>0

      }  // for i
    }    //if(jeta-41 < 0)
  }      //for jeta
  /////////////////
  c3x5->Update();
  c3x5->Print("DreconoiseNegativeDirectionhistD1PhiSymmetryDepth7HE.png");
  c3x5->Clear();
  // clean-up
  if (h2CeffHEnegativedirectionReconoiseD7)
    delete h2CeffHEnegativedirectionReconoiseD7;
  //=====================================================================       END of Reconoise HE for phi-symmetry
  //=====================================================================       END of Reconoise HE for phi-symmetry
  //=====================================================================       END of Reconoise HE for phi-symmetry
  ////////////////////////////////////////////////////////////////////////////////////////////////////         Phi-symmetry for Calibration Group:    Reconoise HF
  ////////////////////////////////////////////////////////////////////////////////////////////////////         Phi-symmetry for Calibration Group:    Reconoise HF
  ////////////////////////////////////////////////////////////////////////////////////////////////////         Phi-symmetry for Calibration Group:    Reconoise HF
  //  int k_max[5]={0,4,7,4,4}; // maximum depth for each subdet
  //ndepth = k_max[5];
  ndepth = 2;
  double areconoiseHF[ndepth][njeta][njphi];
  double breconoiseHF[ndepth][njeta][njphi];
  double reconoisevarianceHF[ndepth][njeta][njphi];

  TH2F* recNoiseEnergy1HF1 = (TH2F*)dir->FindObjectAny("h_recNoiseEnergy1_HF1");
  TH2F* recNoiseEnergy0HF1 = (TH2F*)dir->FindObjectAny("h_recNoiseEnergy0_HF1");
  TH2F* recNoiseEnergyHF1 = (TH2F*)recNoiseEnergy1HF1->Clone("recNoiseEnergyHF1");
  recNoiseEnergyHF1->Divide(recNoiseEnergy1HF1, recNoiseEnergy0HF1, 1, 1, "B");
  TH2F* recNoiseEnergy1HF2 = (TH2F*)dir->FindObjectAny("h_recNoiseEnergy1_HF2");
  TH2F* recNoiseEnergy0HF2 = (TH2F*)dir->FindObjectAny("h_recNoiseEnergy0_HF2");
  TH2F* recNoiseEnergyHF2 = (TH2F*)recNoiseEnergy1HF2->Clone("recNoiseEnergyHF2");
  recNoiseEnergyHF2->Divide(recNoiseEnergy1HF2, recNoiseEnergy0HF2, 1, 1, "B");
  //====================================================================== PHI normalization & put R into massive areconoiseHF
  for (int jeta = 0; jeta < njeta; jeta++) {
    if ((jeta - 41 >= -41 && jeta - 41 <= -29) || (jeta - 41 >= 28 && jeta - 41 <= 40)) {
      //preparation for PHI normalization:
      double sumreconoiseHF0 = 0;
      int nsumreconoiseHF0 = 0;
      double sumreconoiseHF1 = 0;
      int nsumreconoiseHF1 = 0;
      for (int jphi = 0; jphi < njphi; jphi++) {
        areconoiseHF[0][jeta][jphi] = recNoiseEnergyHF1->GetBinContent(jeta + 1, jphi + 1);
        areconoiseHF[1][jeta][jphi] = recNoiseEnergyHF2->GetBinContent(jeta + 1, jphi + 1);
        breconoiseHF[0][jeta][jphi] = recNoiseEnergyHF1->GetBinContent(jeta + 1, jphi + 1);
        breconoiseHF[1][jeta][jphi] = recNoiseEnergyHF2->GetBinContent(jeta + 1, jphi + 1);
        sumreconoiseHF0 += areconoiseHF[0][jeta][jphi];
        ++nsumreconoiseHF0;
        sumreconoiseHF1 += areconoiseHF[1][jeta][jphi];
        ++nsumreconoiseHF1;
      }  // phi

      // PHI normalization for DIF:
      for (int jphi = 0; jphi < njphi; jphi++) {
        if (sumreconoiseHF0 != 0.)
          breconoiseHF[0][jeta][jphi] -= (sumreconoiseHF0 / nsumreconoiseHF0);
        if (sumreconoiseHF1 != 0.)
          breconoiseHF[1][jeta][jphi] -= (sumreconoiseHF1 / nsumreconoiseHF1);
      }  // phi

      // PHI normalization for R:
      for (int jphi = 0; jphi < njphi; jphi++) {
        if (sumreconoiseHF0 != 0.)
          areconoiseHF[0][jeta][jphi] /= (sumreconoiseHF0 / nsumreconoiseHF0);
        if (sumreconoiseHF1 != 0.)
          areconoiseHF[1][jeta][jphi] /= (sumreconoiseHF1 / nsumreconoiseHF1);
      }  // phi

    }  // jeta-41
  }    //eta
  /////////////////////////////////////////

  //                                   RRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRR:   Reconoise HF
  //------------------------  2D-eta/phi-plot: R, averaged over depthfs
  //======================================================================
  //======================================================================
  // cout<<"      R2D-eta/phi-plot: R, averaged over depthfs *****" <<endl;
  c2x1->Clear();
  /////////////////
  c2x1->Divide(2, 1);
  c2x1->cd(1);
  TH2F* GefzRreconoiseHF42D = new TH2F("GefzRreconoiseHF42D", "", neta, -41., 41., nphi, 0., 72.);
  TH2F* GefzRreconoiseHF42D0 = new TH2F("GefzRreconoiseHF42D0", "", neta, -41., 41., nphi, 0., 72.);
  TH2F* GefzRreconoiseHF42DF = (TH2F*)GefzRreconoiseHF42D0->Clone("GefzRreconoiseHF42DF");
  for (int i = 0; i < ndepth; i++) {
    for (int jeta = 0; jeta < neta; jeta++) {
      if ((jeta - 41 >= -41 && jeta - 41 <= -29) || (jeta - 41 >= 28 && jeta - 41 <= 40)) {
        for (int jphi = 0; jphi < nphi; jphi++) {
          double ccc1 = areconoiseHF[i][jeta][jphi];
          int k2plot = jeta - 41;
          int kkk = k2plot;  //if(k2plot >0 ) kkk=k2plot+1; //-41 +41 !=0
          if (ccc1 != 0.) {
            GefzRreconoiseHF42D->Fill(kkk, jphi, ccc1);
            GefzRreconoiseHF42D0->Fill(kkk, jphi, 1.);
          }
        }
      }
    }
  }
  GefzRreconoiseHF42DF->Divide(GefzRreconoiseHF42D, GefzRreconoiseHF42D0, 1, 1, "B");  // average A
  gPad->SetGridy();
  gPad->SetGridx();  //      gPad->SetLogz();
  GefzRreconoiseHF42DF->SetXTitle("<R>_depth       #eta  \b");
  GefzRreconoiseHF42DF->SetYTitle("      #phi \b");
  GefzRreconoiseHF42DF->Draw("COLZ");

  c2x1->cd(2);
  TH1F* energyhitNoise_HF = (TH1F*)dir->FindObjectAny("h_energyhitNoise_HF");
  energyhitNoise_HF->SetMarkerStyle(20);
  energyhitNoise_HF->SetMarkerSize(0.4);
  energyhitNoise_HF->GetYaxis()->SetLabelSize(0.04);
  energyhitNoise_HF->SetXTitle("energyhitNoise_HF \b");
  energyhitNoise_HF->SetMarkerColor(2);
  energyhitNoise_HF->SetLineColor(0);
  gPad->SetGridy();
  gPad->SetGridx();
  energyhitNoise_HF->Draw("Error");

  /////////////////
  c2x1->Update();
  c2x1->Print("RreconoiseGeneralD2PhiSymmetryHF.png");
  c2x1->Clear();
  // clean-up
  if (GefzRreconoiseHF42D)
    delete GefzRreconoiseHF42D;
  if (GefzRreconoiseHF42D0)
    delete GefzRreconoiseHF42D0;
  if (GefzRreconoiseHF42DF)
    delete GefzRreconoiseHF42DF;
  //====================================================================== 1D plot: R vs phi , averaged over depthfs & eta
  //======================================================================
  //cout<<"      1D plot: R vs phi , averaged over depthfs & eta *****" <<endl;
  c1x1->Clear();
  /////////////////
  c1x1->Divide(1, 1);
  c1x1->cd(1);
  TH1F* GefzRreconoiseHF41D = new TH1F("GefzRreconoiseHF41D", "", nphi, 0., 72.);
  TH1F* GefzRreconoiseHF41D0 = new TH1F("GefzRreconoiseHF41D0", "", nphi, 0., 72.);
  TH1F* GefzRreconoiseHF41DF = (TH1F*)GefzRreconoiseHF41D0->Clone("GefzRreconoiseHF41DF");
  for (int jphi = 0; jphi < nphi; jphi++) {
    for (int jeta = 0; jeta < neta; jeta++) {
      if ((jeta - 41 >= -41 && jeta - 41 <= -29) || (jeta - 41 >= 28 && jeta - 41 <= 40)) {
        for (int i = 0; i < ndepth; i++) {
          double ccc1 = areconoiseHF[i][jeta][jphi];
          if (ccc1 != 0.) {
            GefzRreconoiseHF41D->Fill(jphi, ccc1);
            GefzRreconoiseHF41D0->Fill(jphi, 1.);
          }
        }
      }
    }
  }
  GefzRreconoiseHF41DF->Divide(GefzRreconoiseHF41D, GefzRreconoiseHF41D0, 1, 1, "B");  // R averaged over depthfs & eta
  GefzRreconoiseHF41D0->Sumw2();
  //    for (int jphi=1;jphi<73;jphi++) {GefzRreconoiseHF41DF->SetBinError(jphi,0.01);}
  gPad->SetGridy();
  gPad->SetGridx();  //      gPad->SetLogz();
  GefzRreconoiseHF41DF->SetMarkerStyle(20);
  GefzRreconoiseHF41DF->SetMarkerSize(1.4);
  GefzRreconoiseHF41DF->GetZaxis()->SetLabelSize(0.08);
  GefzRreconoiseHF41DF->SetXTitle("#phi  \b");
  GefzRreconoiseHF41DF->SetYTitle("  <R> \b");
  GefzRreconoiseHF41DF->SetZTitle("<R>_PHI  - AllDepthfs \b");
  GefzRreconoiseHF41DF->SetMarkerColor(4);
  GefzRreconoiseHF41DF->SetLineColor(
      4);  //  GefzRreconoiseHF41DF->SetMinimum(0.8);     //      GefzRreconoiseHF41DF->SetMaximum(1.000);
  GefzRreconoiseHF41DF->Draw("Error");
  /////////////////
  c1x1->Update();
  c1x1->Print("RreconoiseGeneralD1PhiSymmetryHF.png");
  c1x1->Clear();
  // clean-up
  if (GefzRreconoiseHF41D)
    delete GefzRreconoiseHF41D;
  if (GefzRreconoiseHF41D0)
    delete GefzRreconoiseHF41D0;
  if (GefzRreconoiseHF41DF)
    delete GefzRreconoiseHF41DF;
  //========================================================================================== 4
  //======================================================================
  //======================================================================1D plot: R vs phi , different eta,  depth=1
  //cout<<"      1D plot: R vs phi , different eta,  depth=1 *****" <<endl;
  c3x5->Clear();
  /////////////////
  c3x5->Divide(3, 5);
  c3x5->cd(1);
  int kcountHFpositivedirectionReconoise1 = 1;
  TH1F* h2CeffHFpositivedirectionReconoise1 = new TH1F("h2CeffHFpositivedirectionReconoise1", "", nphi, 0., 72.);
  for (int jeta = 0; jeta < njeta; jeta++) {
    // positivedirectionReconoise:
    if (jeta - 41 >= 28 && jeta - 41 <= 40) {
      //	     for (int i=0;i<ndepth;i++) {
      // depth=1
      for (int i = 0; i < 1; i++) {
        TH1F* HFpositivedirectionReconoise1 = (TH1F*)h2CeffHFpositivedirectionReconoise1->Clone("twod1");
        float ccctest = 0;  // to avoid empty massive elements
        for (int jphi = 0; jphi < nphi; jphi++) {
          double ccc1 = areconoiseHF[i][jeta][jphi];
          if (ccc1 != 0.) {
            HFpositivedirectionReconoise1->Fill(jphi, ccc1);
            ccctest = 1.;  //HFpositivedirectionReconoise1->SetBinError(i,0.01);
          }
        }  // for jphi
        if (ccctest > 0.) {
          //	  cout<<"444        kcountHFpositivedirectionReconoise1   =     "<<kcountHFpositivedirectionReconoise1  <<"   jeta-41=     "<< jeta-41 <<endl;
          c3x5->cd(kcountHFpositivedirectionReconoise1);
          HFpositivedirectionReconoise1->SetMarkerStyle(20);
          HFpositivedirectionReconoise1->SetMarkerSize(0.4);
          HFpositivedirectionReconoise1->GetYaxis()->SetLabelSize(0.04);
          HFpositivedirectionReconoise1->SetXTitle("HFpositivedirectionReconoise1 \b");
          HFpositivedirectionReconoise1->SetMarkerColor(2);
          HFpositivedirectionReconoise1->SetLineColor(0);
          gPad->SetGridy();
          gPad->SetGridx();
          //	   gPad->SetLogy();
          if (kcountHFpositivedirectionReconoise1 == 1)
            HFpositivedirectionReconoise1->SetXTitle("R for HF+ jeta = 28; depth = 1 \b");
          if (kcountHFpositivedirectionReconoise1 == 2)
            HFpositivedirectionReconoise1->SetXTitle("R for HF+ jeta = 29; depth = 1 \b");
          if (kcountHFpositivedirectionReconoise1 == 3)
            HFpositivedirectionReconoise1->SetXTitle("R for HF+ jeta = 30; depth = 1 \b");
          if (kcountHFpositivedirectionReconoise1 == 4)
            HFpositivedirectionReconoise1->SetXTitle("R for HF+ jeta = 31; depth = 1 \b");
          if (kcountHFpositivedirectionReconoise1 == 5)
            HFpositivedirectionReconoise1->SetXTitle("R for HF+ jeta = 32; depth = 1 \b");
          if (kcountHFpositivedirectionReconoise1 == 6)
            HFpositivedirectionReconoise1->SetXTitle("R for HF+ jeta = 33; depth = 1 \b");
          if (kcountHFpositivedirectionReconoise1 == 7)
            HFpositivedirectionReconoise1->SetXTitle("R for HF+ jeta = 34; depth = 1 \b");
          if (kcountHFpositivedirectionReconoise1 == 8)
            HFpositivedirectionReconoise1->SetXTitle("R for HF+ jeta = 35; depth = 1 \b");
          if (kcountHFpositivedirectionReconoise1 == 9)
            HFpositivedirectionReconoise1->SetXTitle("R for HF+ jeta = 36; depth = 1 \b");
          if (kcountHFpositivedirectionReconoise1 == 10)
            HFpositivedirectionReconoise1->SetXTitle("R for HF+ jeta = 37; depth = 1 \b");
          if (kcountHFpositivedirectionReconoise1 == 11)
            HFpositivedirectionReconoise1->SetXTitle("R for HF+ jeta = 38; depth = 1 \b");
          if (kcountHFpositivedirectionReconoise1 == 12)
            HFpositivedirectionReconoise1->SetXTitle("R for HF+ jeta = 39; depth = 1 \b");
          if (kcountHFpositivedirectionReconoise1 == 13)
            HFpositivedirectionReconoise1->SetXTitle("R for HF+ jeta = 40; depth = 1 \b");
          HFpositivedirectionReconoise1->Draw("Error");
          kcountHFpositivedirectionReconoise1++;
          if (kcountHFpositivedirectionReconoise1 > 13)
            break;  //
        }           //ccctest>0

      }  // for i
    }    //if(jeta-41 >= 28 && jeta-41 <= 40
  }      //for jeta
  /////////////////
  c3x5->Update();
  c3x5->Print("RreconoisePositiveDirectionhistD1PhiSymmetryDepth1HF.png");
  c3x5->Clear();
  // clean-up
  if (h2CeffHFpositivedirectionReconoise1)
    delete h2CeffHFpositivedirectionReconoise1;

  //========================================================================================== 5
  //======================================================================
  //======================================================================1D plot: R vs phi , different eta,  depth=2
  //  cout<<"      1D plot: R vs phi , different eta,  depth=2 *****" <<endl;
  c3x5->Clear();
  /////////////////
  c3x5->Divide(3, 5);
  c3x5->cd(1);
  int kcountHFpositivedirectionReconoise2 = 1;
  TH1F* h2CeffHFpositivedirectionReconoise2 = new TH1F("h2CeffHFpositivedirectionReconoise2", "", nphi, 0., 72.);
  for (int jeta = 0; jeta < njeta; jeta++) {
    // positivedirectionReconoise:
    if (jeta - 41 >= 28 && jeta - 41 <= 40) {
      //	     for (int i=0;i<ndepth;i++) {
      // depth=2
      for (int i = 1; i < 2; i++) {
        TH1F* HFpositivedirectionReconoise2 = (TH1F*)h2CeffHFpositivedirectionReconoise2->Clone("twod1");
        float ccctest = 0;  // to avoid empty massive elements
        for (int jphi = 0; jphi < nphi; jphi++) {
          double ccc1 = areconoiseHF[i][jeta][jphi];
          if (ccc1 != 0.) {
            HFpositivedirectionReconoise2->Fill(jphi, ccc1);
            ccctest = 1.;  //HFpositivedirectionReconoise2->SetBinError(i,0.01);
          }
        }  // for jphi
        if (ccctest > 0.) {
          //cout<<"555        kcountHFpositivedirectionReconoise2   =     "<<kcountHFpositivedirectionReconoise2  <<"   jeta-41=     "<< jeta-41 <<endl;
          c3x5->cd(kcountHFpositivedirectionReconoise2);
          HFpositivedirectionReconoise2->SetMarkerStyle(20);
          HFpositivedirectionReconoise2->SetMarkerSize(0.4);
          HFpositivedirectionReconoise2->GetYaxis()->SetLabelSize(0.04);
          HFpositivedirectionReconoise2->SetXTitle("HFpositivedirectionReconoise2 \b");
          HFpositivedirectionReconoise2->SetMarkerColor(2);
          HFpositivedirectionReconoise2->SetLineColor(0);
          gPad->SetGridy();
          gPad->SetGridx();
          //	   gPad->SetLogy();
          if (kcountHFpositivedirectionReconoise2 == 1)
            HFpositivedirectionReconoise2->SetXTitle("R for HF+ jeta = 28; depth = 2 \b");
          if (kcountHFpositivedirectionReconoise2 == 2)
            HFpositivedirectionReconoise2->SetXTitle("R for HF+ jeta = 29; depth = 2 \b");
          if (kcountHFpositivedirectionReconoise2 == 3)
            HFpositivedirectionReconoise2->SetXTitle("R for HF+ jeta = 30; depth = 2 \b");
          if (kcountHFpositivedirectionReconoise2 == 4)
            HFpositivedirectionReconoise2->SetXTitle("R for HF+ jeta = 31; depth = 2 \b");
          if (kcountHFpositivedirectionReconoise2 == 5)
            HFpositivedirectionReconoise2->SetXTitle("R for HF+ jeta = 32; depth = 2 \b");
          if (kcountHFpositivedirectionReconoise2 == 6)
            HFpositivedirectionReconoise2->SetXTitle("R for HF+ jeta = 33; depth = 2 \b");
          if (kcountHFpositivedirectionReconoise2 == 7)
            HFpositivedirectionReconoise2->SetXTitle("R for HF+ jeta = 34; depth = 2 \b");
          if (kcountHFpositivedirectionReconoise2 == 8)
            HFpositivedirectionReconoise2->SetXTitle("R for HF+ jeta = 35; depth = 2 \b");
          if (kcountHFpositivedirectionReconoise2 == 9)
            HFpositivedirectionReconoise2->SetXTitle("R for HF+ jeta = 36; depth = 2 \b");
          if (kcountHFpositivedirectionReconoise2 == 10)
            HFpositivedirectionReconoise2->SetXTitle("R for HF+ jeta = 37; depth = 2 \b");
          if (kcountHFpositivedirectionReconoise2 == 11)
            HFpositivedirectionReconoise2->SetXTitle("R for HF+ jeta = 38; depth = 2 \b");
          if (kcountHFpositivedirectionReconoise2 == 12)
            HFpositivedirectionReconoise2->SetXTitle("R for HF+ jeta = 39; depth = 2 \b");
          if (kcountHFpositivedirectionReconoise2 == 13)
            HFpositivedirectionReconoise2->SetXTitle("R for HF+ jeta = 40; depth = 2 \b");
          HFpositivedirectionReconoise2->Draw("Error");
          kcountHFpositivedirectionReconoise2++;
          if (kcountHFpositivedirectionReconoise2 > 13)
            break;  // 4x6 = 24
        }           //ccctest>0

      }  // for i
    }    //if(jeta-41 >= 28 && jeta-41 <= 40)
  }      //for jeta
  /////////////////
  c3x5->Update();
  c3x5->Print("RreconoisePositiveDirectionhistD1PhiSymmetryDepth2HF.png");
  c3x5->Clear();
  // clean-up
  if (h2CeffHFpositivedirectionReconoise2)
    delete h2CeffHFpositivedirectionReconoise2;

  //========================================================================================== 1111114
  //======================================================================
  //======================================================================1D plot: R vs phi , different eta,  depth=1
  //cout<<"      1D plot: R vs phi , different eta,  depth=1 *****" <<endl;
  c3x5->Clear();
  /////////////////
  c3x5->Divide(3, 5);
  c3x5->cd(1);
  int kcountHFnegativedirectionReconoise1 = 1;
  TH1F* h2CeffHFnegativedirectionReconoise1 = new TH1F("h2CeffHFnegativedirectionReconoise1", "", nphi, 0., 72.);
  for (int jeta = 0; jeta < njeta; jeta++) {
    // negativedirectionReconoise:
    if (jeta - 41 >= -41 && jeta - 41 <= -29) {
      //	     for (int i=0;i<ndepth;i++) {
      // depth=1
      for (int i = 0; i < 1; i++) {
        TH1F* HFnegativedirectionReconoise1 = (TH1F*)h2CeffHFnegativedirectionReconoise1->Clone("twod1");
        float ccctest = 0;  // to avoid empty massive elements
        for (int jphi = 0; jphi < nphi; jphi++) {
          double ccc1 = areconoiseHF[i][jeta][jphi];
          if (ccc1 != 0.) {
            HFnegativedirectionReconoise1->Fill(jphi, ccc1);
            ccctest = 1.;  //HFnegativedirectionReconoise1->SetBinError(i,0.01);
          }
        }  // for jphi
        if (ccctest > 0.) {
          //	  cout<<"444        kcountHFnegativedirectionReconoise1   =     "<<kcountHFnegativedirectionReconoise1  <<"   jeta-41=     "<< jeta-41 <<endl;
          c3x5->cd(kcountHFnegativedirectionReconoise1);
          HFnegativedirectionReconoise1->SetMarkerStyle(20);
          HFnegativedirectionReconoise1->SetMarkerSize(0.4);
          HFnegativedirectionReconoise1->GetYaxis()->SetLabelSize(0.04);
          HFnegativedirectionReconoise1->SetXTitle("HFnegativedirectionReconoise1 \b");
          HFnegativedirectionReconoise1->SetMarkerColor(2);
          HFnegativedirectionReconoise1->SetLineColor(0);
          gPad->SetGridy();
          gPad->SetGridx();
          //	   gPad->SetLogy();
          if (kcountHFnegativedirectionReconoise1 == 1)
            HFnegativedirectionReconoise1->SetXTitle("R for HF- jeta =-41; depth = 1 \b");
          if (kcountHFnegativedirectionReconoise1 == 2)
            HFnegativedirectionReconoise1->SetXTitle("R for HF- jeta =-40; depth = 1 \b");
          if (kcountHFnegativedirectionReconoise1 == 3)
            HFnegativedirectionReconoise1->SetXTitle("R for HF- jeta =-39; depth = 1 \b");
          if (kcountHFnegativedirectionReconoise1 == 4)
            HFnegativedirectionReconoise1->SetXTitle("R for HF- jeta =-38; depth = 1 \b");
          if (kcountHFnegativedirectionReconoise1 == 5)
            HFnegativedirectionReconoise1->SetXTitle("R for HF- jeta =-37; depth = 1 \b");
          if (kcountHFnegativedirectionReconoise1 == 6)
            HFnegativedirectionReconoise1->SetXTitle("R for HF- jeta =-36; depth = 1 \b");
          if (kcountHFnegativedirectionReconoise1 == 7)
            HFnegativedirectionReconoise1->SetXTitle("R for HF- jeta =-35; depth = 1 \b");
          if (kcountHFnegativedirectionReconoise1 == 8)
            HFnegativedirectionReconoise1->SetXTitle("R for HF- jeta =-34; depth = 1 \b");
          if (kcountHFnegativedirectionReconoise1 == 9)
            HFnegativedirectionReconoise1->SetXTitle("R for HF- jeta =-33; depth = 1 \b");
          if (kcountHFnegativedirectionReconoise1 == 10)
            HFnegativedirectionReconoise1->SetXTitle("R for HF- jeta =-32; depth = 1 \b");
          if (kcountHFnegativedirectionReconoise1 == 11)
            HFnegativedirectionReconoise1->SetXTitle("R for HF- jeta =-31; depth = 1 \b");
          if (kcountHFnegativedirectionReconoise1 == 12)
            HFnegativedirectionReconoise1->SetXTitle("R for HF- jeta =-30; depth = 1 \b");
          if (kcountHFnegativedirectionReconoise1 == 13)
            HFnegativedirectionReconoise1->SetXTitle("R for HF- jeta =-29; depth = 1 \b");
          HFnegativedirectionReconoise1->Draw("Error");
          kcountHFnegativedirectionReconoise1++;
          if (kcountHFnegativedirectionReconoise1 > 13)
            break;  //
        }           //ccctest>0

      }  // for i
    }    //if(jeta-41 >= -41 && jeta-41 <= -29)
  }      //for jeta
  /////////////////
  c3x5->Update();
  c3x5->Print("RreconoiseNegativeDirectionhistD1PhiSymmetryDepth1HF.png");
  c3x5->Clear();
  // clean-up
  if (h2CeffHFnegativedirectionReconoise1)
    delete h2CeffHFnegativedirectionReconoise1;

  //========================================================================================== 1111115
  //======================================================================
  //======================================================================1D plot: R vs phi , different eta,  depth=2
  //  cout<<"      1D plot: R vs phi , different eta,  depth=2 *****" <<endl;
  c3x5->Clear();
  /////////////////
  c3x5->Divide(3, 5);
  c3x5->cd(1);
  int kcountHFnegativedirectionReconoise2 = 1;
  TH1F* h2CeffHFnegativedirectionReconoise2 = new TH1F("h2CeffHFnegativedirectionReconoise2", "", nphi, 0., 72.);
  for (int jeta = 0; jeta < njeta; jeta++) {
    // negativedirectionReconoise:
    if (jeta - 41 >= -41 && jeta - 41 <= -29) {
      //	     for (int i=0;i<ndepth;i++) {
      // depth=2
      for (int i = 1; i < 2; i++) {
        TH1F* HFnegativedirectionReconoise2 = (TH1F*)h2CeffHFnegativedirectionReconoise2->Clone("twod1");
        float ccctest = 0;  // to avoid empty massive elements
        for (int jphi = 0; jphi < nphi; jphi++) {
          double ccc1 = areconoiseHF[i][jeta][jphi];
          if (ccc1 != 0.) {
            HFnegativedirectionReconoise2->Fill(jphi, ccc1);
            ccctest = 1.;  //HFnegativedirectionReconoise2->SetBinError(i,0.01);
          }
        }  // for jphi
        if (ccctest > 0.) {
          //cout<<"555        kcountHFnegativedirectionReconoise2   =     "<<kcountHFnegativedirectionReconoise2  <<"   jeta-41=     "<< jeta-41 <<endl;
          c3x5->cd(kcountHFnegativedirectionReconoise2);
          HFnegativedirectionReconoise2->SetMarkerStyle(20);
          HFnegativedirectionReconoise2->SetMarkerSize(0.4);
          HFnegativedirectionReconoise2->GetYaxis()->SetLabelSize(0.04);
          HFnegativedirectionReconoise2->SetXTitle("HFnegativedirectionReconoise2 \b");
          HFnegativedirectionReconoise2->SetMarkerColor(2);
          HFnegativedirectionReconoise2->SetLineColor(0);
          gPad->SetGridy();
          gPad->SetGridx();
          //	   gPad->SetLogy();
          if (kcountHFnegativedirectionReconoise2 == 1)
            HFnegativedirectionReconoise2->SetXTitle("R for HF- jeta =-41; depth = 2 \b");
          if (kcountHFnegativedirectionReconoise2 == 2)
            HFnegativedirectionReconoise2->SetXTitle("R for HF- jeta =-40; depth = 2 \b");
          if (kcountHFnegativedirectionReconoise2 == 3)
            HFnegativedirectionReconoise2->SetXTitle("R for HF- jeta =-39; depth = 2 \b");
          if (kcountHFnegativedirectionReconoise2 == 4)
            HFnegativedirectionReconoise2->SetXTitle("R for HF- jeta =-38; depth = 2 \b");
          if (kcountHFnegativedirectionReconoise2 == 5)
            HFnegativedirectionReconoise2->SetXTitle("R for HF- jeta =-37; depth = 2 \b");
          if (kcountHFnegativedirectionReconoise2 == 6)
            HFnegativedirectionReconoise2->SetXTitle("R for HF- jeta =-36; depth = 2 \b");
          if (kcountHFnegativedirectionReconoise2 == 7)
            HFnegativedirectionReconoise2->SetXTitle("R for HF- jeta =-35; depth = 2 \b");
          if (kcountHFnegativedirectionReconoise2 == 8)
            HFnegativedirectionReconoise2->SetXTitle("R for HF- jeta =-34; depth = 2 \b");
          if (kcountHFnegativedirectionReconoise2 == 9)
            HFnegativedirectionReconoise2->SetXTitle("R for HF- jeta =-33; depth = 2 \b");
          if (kcountHFnegativedirectionReconoise2 == 10)
            HFnegativedirectionReconoise2->SetXTitle("R for HF- jeta =-32; depth = 2 \b");
          if (kcountHFnegativedirectionReconoise2 == 11)
            HFnegativedirectionReconoise2->SetXTitle("R for HF- jeta =-31; depth = 2 \b");
          if (kcountHFnegativedirectionReconoise2 == 12)
            HFnegativedirectionReconoise2->SetXTitle("R for HF- jeta =-30; depth = 2 \b");
          if (kcountHFnegativedirectionReconoise2 == 13)
            HFnegativedirectionReconoise2->SetXTitle("R for HF- jeta =-20; depth = 2 \b");
          HFnegativedirectionReconoise2->Draw("Error");
          kcountHFnegativedirectionReconoise2++;
          if (kcountHFnegativedirectionReconoise2 > 13)
            break;  // 4x6 = 24
        }           //ccctest>0

      }  // for i
    }    //if(jeta-41 >= -41 && jeta-41 <= -29)
  }      //for jeta
  /////////////////
  c3x5->Update();
  c3x5->Print("RreconoiseNegativeDirectionhistD1PhiSymmetryDepth2HF.png");
  c3x5->Clear();
  // clean-up
  if (h2CeffHFnegativedirectionReconoise2)
    delete h2CeffHFnegativedirectionReconoise2;

  //======================================================================================================================
  //                                   DIFDIFDIFDIFDIFDIFDIFDIFDIFDIFDIFDIFDIFDIFDIFDIFDIFDIFDIFDIF:   Reconoise HF
  //------------------------  2D-eta/phi-plot: DIF, averaged over depthfs
  //======================================================================
  //======================================================================
  // cout<<"      DIF2D-eta/phi-plot: DIF, averaged over depthfs *****" <<endl;
  c2x1->Clear();
  /////////////////
  c2x1->Divide(2, 1);
  c2x1->cd(1);
  TH2F* GefzDIFreconoiseHF42D = new TH2F("GefzDIFreconoiseHF42D", "", neta, -41., 41., nphi, 0., 72.);
  TH2F* GefzDIFreconoiseHF42D0 = new TH2F("GefzDIFreconoiseHF42D0", "", neta, -41., 41., nphi, 0., 72.);
  TH2F* GefzDIFreconoiseHF42DF = (TH2F*)GefzDIFreconoiseHF42D0->Clone("GefzDIFreconoiseHF42DF");
  for (int i = 0; i < ndepth; i++) {
    for (int jeta = 0; jeta < neta; jeta++) {
      if ((jeta - 41 >= -41 && jeta - 41 <= -29) || (jeta - 41 >= 28 && jeta - 41 <= 40)) {
        for (int jphi = 0; jphi < nphi; jphi++) {
          double ccc1 = breconoiseHF[i][jeta][jphi];
          int k2plot = jeta - 41;
          int kkk = k2plot;  //if(k2plot >0 ) kkk=k2plot+1; //-41 +41 !=0
          if (ccc1 != 0.) {
            GefzDIFreconoiseHF42D->Fill(kkk, jphi, ccc1);
            GefzDIFreconoiseHF42D0->Fill(kkk, jphi, 1.);
          }
        }
      }
    }
  }
  GefzDIFreconoiseHF42DF->Divide(GefzDIFreconoiseHF42D, GefzDIFreconoiseHF42D0, 1, 1, "B");  // average A
  gPad->SetGridy();
  gPad->SetGridx();  //      gPad->SetLogz();
  GefzDIFreconoiseHF42DF->SetXTitle("<DIF>_depth       #eta  \b");
  GefzDIFreconoiseHF42DF->SetYTitle("      #phi \b");
  GefzDIFreconoiseHF42DF->Draw("COLZ");

  c2x1->cd(2);
  //  TH1F *energyhitNoiseCut_HF= (TH1F*)dir->FindObjectAny("h_energyhitNoiseCut_HF");
  //  energyhitNoiseCut_HF ->SetMarkerStyle(20);energyhitNoiseCut_HF ->SetMarkerSize(0.4);energyhitNoiseCut_HF ->GetYaxis()->SetLabelSize(0.04);energyhitNoiseCut_HF ->SetXTitle("energyhitNoiseCut_HF \b");energyhitNoiseCut_HF ->SetMarkerColor(2);energyhitNoiseCut_HF ->SetLineColor(0);gPad->SetGridy();gPad->SetGridx();energyhitNoiseCut_HF ->Draw("Error");

  /////////////////
  c2x1->Update();
  c2x1->Print("DIFreconoiseGeneralD2PhiSymmetryHF.png");
  c2x1->Clear();
  // clean-up
  if (GefzDIFreconoiseHF42D)
    delete GefzDIFreconoiseHF42D;
  if (GefzDIFreconoiseHF42D0)
    delete GefzDIFreconoiseHF42D0;
  if (GefzDIFreconoiseHF42DF)
    delete GefzDIFreconoiseHF42DF;
  //====================================================================== 1D plot: DIF vs phi , averaged over depthfs & eta
  //======================================================================
  //cout<<"      1D plot: DIF vs phi , averaged over depthfs & eta *****" <<endl;
  c1x1->Clear();
  /////////////////
  c1x1->Divide(1, 1);
  c1x1->cd(1);
  TH1F* GefzDIFreconoiseHF41D = new TH1F("GefzDIFreconoiseHF41D", "", nphi, 0., 72.);
  TH1F* GefzDIFreconoiseHF41D0 = new TH1F("GefzDIFreconoiseHF41D0", "", nphi, 0., 72.);
  TH1F* GefzDIFreconoiseHF41DF = (TH1F*)GefzDIFreconoiseHF41D0->Clone("GefzDIFreconoiseHF41DF");
  for (int jphi = 0; jphi < nphi; jphi++) {
    for (int jeta = 0; jeta < neta; jeta++) {
      if ((jeta - 41 >= -41 && jeta - 41 <= -29) || (jeta - 41 >= 28 && jeta - 41 <= 40)) {
        for (int i = 0; i < ndepth; i++) {
          double ccc1 = breconoiseHF[i][jeta][jphi];
          if (ccc1 != 0.) {
            GefzDIFreconoiseHF41D->Fill(jphi, ccc1);
            GefzDIFreconoiseHF41D0->Fill(jphi, 1.);
          }
        }
      }
    }
  }
  GefzDIFreconoiseHF41DF->Divide(
      GefzDIFreconoiseHF41D, GefzDIFreconoiseHF41D0, 1, 1, "B");  // DIF averaged over depthfs & eta
  GefzDIFreconoiseHF41D0->Sumw2();
  //    for (int jphi=1;jphi<73;jphi++) {GefzDIFreconoiseHF41DF->SetBinError(jphi,0.01);}
  gPad->SetGridy();
  gPad->SetGridx();  //      gPad->SetLogz();
  GefzDIFreconoiseHF41DF->SetMarkerStyle(20);
  GefzDIFreconoiseHF41DF->SetMarkerSize(1.4);
  GefzDIFreconoiseHF41DF->GetZaxis()->SetLabelSize(0.08);
  GefzDIFreconoiseHF41DF->SetXTitle("#phi  \b");
  GefzDIFreconoiseHF41DF->SetYTitle("  <DIF> \b");
  GefzDIFreconoiseHF41DF->SetZTitle("<DIF>_PHI  - AllDepthfs \b");
  GefzDIFreconoiseHF41DF->SetMarkerColor(4);
  GefzDIFreconoiseHF41DF->SetLineColor(
      4);  //  GefzDIFreconoiseHF41DF->SetMinimum(0.8);     //      GefzDIFreconoiseHF41DF->SetMaximum(1.000);
  GefzDIFreconoiseHF41DF->Draw("Error");
  /////////////////
  c1x1->Update();
  c1x1->Print("DIFreconoiseGeneralD1PhiSymmetryHF.png");
  c1x1->Clear();
  // clean-up
  if (GefzDIFreconoiseHF41D)
    delete GefzDIFreconoiseHF41D;
  if (GefzDIFreconoiseHF41D0)
    delete GefzDIFreconoiseHF41D0;
  if (GefzDIFreconoiseHF41DF)
    delete GefzDIFreconoiseHF41DF;
  //========================================================================================== 4
  //======================================================================
  //======================================================================1D plot: DIF vs phi , different eta,  depth=1
  //cout<<"      1D plot: DIF vs phi , different eta,  depth=1 *****" <<endl;
  c3x5->Clear();
  /////////////////
  c3x5->Divide(3, 5);
  c3x5->cd(1);
  int kcountHFpositivedirectionReconoiseDIF1 = 1;
  TH1F* h2CeffHFpositivedirectionReconoiseDIF1 = new TH1F("h2CeffHFpositivedirectionReconoiseDIF1", "", nphi, 0., 72.);
  for (int jeta = 0; jeta < njeta; jeta++) {
    // positivedirectionReconoiseDIF:
    if (jeta - 41 >= 28 && jeta - 41 <= 40) {
      //	     for (int i=0;i<ndepth;i++) {
      // depth=1
      for (int i = 0; i < 1; i++) {
        TH1F* HFpositivedirectionReconoiseDIF1 = (TH1F*)h2CeffHFpositivedirectionReconoiseDIF1->Clone("twod1");
        float ccctest = 0;  // to avoid empty massive elements
        for (int jphi = 0; jphi < nphi; jphi++) {
          double ccc1 = breconoiseHF[i][jeta][jphi];
          if (ccc1 != 0.) {
            HFpositivedirectionReconoiseDIF1->Fill(jphi, ccc1);
            ccctest = 1.;  //HFpositivedirectionReconoiseDIF1->SetBinError(i,0.01);
          }
        }  // for jphi
        if (ccctest > 0.) {
          //	  cout<<"444        kcountHFpositivedirectionReconoiseDIF1   =     "<<kcountHFpositivedirectionReconoiseDIF1  <<"   jeta-41=     "<< jeta-41 <<endl;
          c3x5->cd(kcountHFpositivedirectionReconoiseDIF1);
          HFpositivedirectionReconoiseDIF1->SetMarkerStyle(20);
          HFpositivedirectionReconoiseDIF1->SetMarkerSize(0.4);
          HFpositivedirectionReconoiseDIF1->GetYaxis()->SetLabelSize(0.04);
          HFpositivedirectionReconoiseDIF1->SetXTitle("HFpositivedirectionReconoiseDIF1 \b");
          HFpositivedirectionReconoiseDIF1->SetMarkerColor(2);
          HFpositivedirectionReconoiseDIF1->SetLineColor(0);
          gPad->SetGridy();
          gPad->SetGridx();
          //	   gPad->SetLogy();
          if (kcountHFpositivedirectionReconoiseDIF1 == 1)
            HFpositivedirectionReconoiseDIF1->SetXTitle("DIF for HF+ jeta = 28; depth = 1 \b");
          if (kcountHFpositivedirectionReconoiseDIF1 == 2)
            HFpositivedirectionReconoiseDIF1->SetXTitle("DIF for HF+ jeta = 29; depth = 1 \b");
          if (kcountHFpositivedirectionReconoiseDIF1 == 3)
            HFpositivedirectionReconoiseDIF1->SetXTitle("DIF for HF+ jeta = 30; depth = 1 \b");
          if (kcountHFpositivedirectionReconoiseDIF1 == 4)
            HFpositivedirectionReconoiseDIF1->SetXTitle("DIF for HF+ jeta = 31; depth = 1 \b");
          if (kcountHFpositivedirectionReconoiseDIF1 == 5)
            HFpositivedirectionReconoiseDIF1->SetXTitle("DIF for HF+ jeta = 32; depth = 1 \b");
          if (kcountHFpositivedirectionReconoiseDIF1 == 6)
            HFpositivedirectionReconoiseDIF1->SetXTitle("DIF for HF+ jeta = 33; depth = 1 \b");
          if (kcountHFpositivedirectionReconoiseDIF1 == 7)
            HFpositivedirectionReconoiseDIF1->SetXTitle("DIF for HF+ jeta = 34; depth = 1 \b");
          if (kcountHFpositivedirectionReconoiseDIF1 == 8)
            HFpositivedirectionReconoiseDIF1->SetXTitle("DIF for HF+ jeta = 35; depth = 1 \b");
          if (kcountHFpositivedirectionReconoiseDIF1 == 9)
            HFpositivedirectionReconoiseDIF1->SetXTitle("DIF for HF+ jeta = 36; depth = 1 \b");
          if (kcountHFpositivedirectionReconoiseDIF1 == 10)
            HFpositivedirectionReconoiseDIF1->SetXTitle("DIF for HF+ jeta = 37; depth = 1 \b");
          if (kcountHFpositivedirectionReconoiseDIF1 == 11)
            HFpositivedirectionReconoiseDIF1->SetXTitle("DIF for HF+ jeta = 38; depth = 1 \b");
          if (kcountHFpositivedirectionReconoiseDIF1 == 12)
            HFpositivedirectionReconoiseDIF1->SetXTitle("DIF for HF+ jeta = 39; depth = 1 \b");
          if (kcountHFpositivedirectionReconoiseDIF1 == 13)
            HFpositivedirectionReconoiseDIF1->SetXTitle("DIF for HF+ jeta = 40; depth = 1 \b");
          HFpositivedirectionReconoiseDIF1->Draw("Error");
          kcountHFpositivedirectionReconoiseDIF1++;
          if (kcountHFpositivedirectionReconoiseDIF1 > 13)
            break;  //
        }           //ccctest>0

      }  // for i
    }    //if(jeta-41 >= 28 && jeta-41 <= 40
  }      //for jeta
  /////////////////
  c3x5->Update();
  c3x5->Print("DIFreconoisePositiveDirectionhistD1PhiSymmetryDepth1HF.png");
  c3x5->Clear();
  // clean-up
  if (h2CeffHFpositivedirectionReconoiseDIF1)
    delete h2CeffHFpositivedirectionReconoiseDIF1;

  //========================================================================================== 5
  //======================================================================
  //======================================================================1D plot: R vs phi , different eta,  depth=2
  //  cout<<"      1D plot: R vs phi , different eta,  depth=2 *****" <<endl;
  c3x5->Clear();
  /////////////////
  c3x5->Divide(3, 5);
  c3x5->cd(1);
  int kcountHFpositivedirectionReconoiseDIF2 = 1;
  TH1F* h2CeffHFpositivedirectionReconoiseDIF2 = new TH1F("h2CeffHFpositivedirectionReconoiseDIF2", "", nphi, 0., 72.);
  for (int jeta = 0; jeta < njeta; jeta++) {
    // positivedirectionReconoiseDIF:
    if (jeta - 41 >= 28 && jeta - 41 <= 40) {
      //	     for (int i=0;i<ndepth;i++) {
      // depth=2
      for (int i = 1; i < 2; i++) {
        TH1F* HFpositivedirectionReconoiseDIF2 = (TH1F*)h2CeffHFpositivedirectionReconoiseDIF2->Clone("twod1");
        float ccctest = 0;  // to avoid empty massive elements
        for (int jphi = 0; jphi < nphi; jphi++) {
          double ccc1 = breconoiseHF[i][jeta][jphi];
          if (ccc1 != 0.) {
            HFpositivedirectionReconoiseDIF2->Fill(jphi, ccc1);
            ccctest = 1.;  //HFpositivedirectionReconoiseDIF2->SetBinError(i,0.01);
          }
        }  // for jphi
        if (ccctest > 0.) {
          //cout<<"555        kcountHFpositivedirectionReconoiseDIF2   =     "<<kcountHFpositivedirectionReconoiseDIF2  <<"   jeta-41=     "<< jeta-41 <<endl;
          c3x5->cd(kcountHFpositivedirectionReconoiseDIF2);
          HFpositivedirectionReconoiseDIF2->SetMarkerStyle(20);
          HFpositivedirectionReconoiseDIF2->SetMarkerSize(0.4);
          HFpositivedirectionReconoiseDIF2->GetYaxis()->SetLabelSize(0.04);
          HFpositivedirectionReconoiseDIF2->SetXTitle("HFpositivedirectionReconoiseDIF2 \b");
          HFpositivedirectionReconoiseDIF2->SetMarkerColor(2);
          HFpositivedirectionReconoiseDIF2->SetLineColor(0);
          gPad->SetGridy();
          gPad->SetGridx();
          //	   gPad->SetLogy();
          if (kcountHFpositivedirectionReconoiseDIF2 == 1)
            HFpositivedirectionReconoiseDIF2->SetXTitle("DIF for HF+ jeta = 28; depth = 2 \b");
          if (kcountHFpositivedirectionReconoiseDIF2 == 2)
            HFpositivedirectionReconoiseDIF2->SetXTitle("DIF for HF+ jeta = 29; depth = 2 \b");
          if (kcountHFpositivedirectionReconoiseDIF2 == 3)
            HFpositivedirectionReconoiseDIF2->SetXTitle("DIF for HF+ jeta = 30; depth = 2 \b");
          if (kcountHFpositivedirectionReconoiseDIF2 == 4)
            HFpositivedirectionReconoiseDIF2->SetXTitle("DIF for HF+ jeta = 31; depth = 2 \b");
          if (kcountHFpositivedirectionReconoiseDIF2 == 5)
            HFpositivedirectionReconoiseDIF2->SetXTitle("DIF for HF+ jeta = 32; depth = 2 \b");
          if (kcountHFpositivedirectionReconoiseDIF2 == 6)
            HFpositivedirectionReconoiseDIF2->SetXTitle("DIF for HF+ jeta = 33; depth = 2 \b");
          if (kcountHFpositivedirectionReconoiseDIF2 == 7)
            HFpositivedirectionReconoiseDIF2->SetXTitle("DIF for HF+ jeta = 34; depth = 2 \b");
          if (kcountHFpositivedirectionReconoiseDIF2 == 8)
            HFpositivedirectionReconoiseDIF2->SetXTitle("DIF for HF+ jeta = 35; depth = 2 \b");
          if (kcountHFpositivedirectionReconoiseDIF2 == 9)
            HFpositivedirectionReconoiseDIF2->SetXTitle("DIF for HF+ jeta = 36; depth = 2 \b");
          if (kcountHFpositivedirectionReconoiseDIF2 == 10)
            HFpositivedirectionReconoiseDIF2->SetXTitle("DIF for HF+ jeta = 37; depth = 2 \b");
          if (kcountHFpositivedirectionReconoiseDIF2 == 11)
            HFpositivedirectionReconoiseDIF2->SetXTitle("DIF for HF+ jeta = 38; depth = 2 \b");
          if (kcountHFpositivedirectionReconoiseDIF2 == 12)
            HFpositivedirectionReconoiseDIF2->SetXTitle("DIF for HF+ jeta = 39; depth = 2 \b");
          if (kcountHFpositivedirectionReconoiseDIF2 == 13)
            HFpositivedirectionReconoiseDIF2->SetXTitle("DIF for HF+ jeta = 40; depth = 2 \b");
          HFpositivedirectionReconoiseDIF2->Draw("Error");
          kcountHFpositivedirectionReconoiseDIF2++;
          if (kcountHFpositivedirectionReconoiseDIF2 > 13)
            break;  // 4x6 = 24
        }           //ccctest>0

      }  // for i
    }    //if(jeta-41 >= 28 && jeta-41 <= 40)
  }      //for jeta
  /////////////////
  c3x5->Update();
  c3x5->Print("DIFreconoisePositiveDirectionhistD1PhiSymmetryDepth2HF.png");
  c3x5->Clear();
  // clean-up
  if (h2CeffHFpositivedirectionReconoiseDIF2)
    delete h2CeffHFpositivedirectionReconoiseDIF2;

  //========================================================================================== 1111114
  //======================================================================
  //======================================================================1D plot: DIF vs phi , different eta,  depth=1
  //cout<<"      1D plot: DIF vs phi , different eta,  depth=1 *****" <<endl;
  c3x5->Clear();
  /////////////////
  c3x5->Divide(3, 5);
  c3x5->cd(1);
  int kcountHFnegativedirectionReconoiseDIF1 = 1;
  TH1F* h2CeffHFnegativedirectionReconoiseDIF1 = new TH1F("h2CeffHFnegativedirectionReconoiseDIF1", "", nphi, 0., 72.);
  for (int jeta = 0; jeta < njeta; jeta++) {
    // negativedirectionReconoiseDIF:
    if (jeta - 41 >= -41 && jeta - 41 <= -29) {
      //	     for (int i=0;i<ndepth;i++) {
      // depth=1
      for (int i = 0; i < 1; i++) {
        TH1F* HFnegativedirectionReconoiseDIF1 = (TH1F*)h2CeffHFnegativedirectionReconoiseDIF1->Clone("twod1");
        float ccctest = 0;  // to avoid empty massive elements
        for (int jphi = 0; jphi < nphi; jphi++) {
          double ccc1 = breconoiseHF[i][jeta][jphi];
          if (ccc1 != 0.) {
            HFnegativedirectionReconoiseDIF1->Fill(jphi, ccc1);
            ccctest = 1.;  //HFnegativedirectionReconoiseDIF1->SetBinError(i,0.01);
          }
        }  // for jphi
        if (ccctest > 0.) {
          //	  cout<<"444        kcountHFnegativedirectionReconoiseDIF1   =     "<<kcountHFnegativedirectionReconoiseDIF1  <<"   jeta-41=     "<< jeta-41 <<endl;
          c3x5->cd(kcountHFnegativedirectionReconoiseDIF1);
          HFnegativedirectionReconoiseDIF1->SetMarkerStyle(20);
          HFnegativedirectionReconoiseDIF1->SetMarkerSize(0.4);
          HFnegativedirectionReconoiseDIF1->GetYaxis()->SetLabelSize(0.04);
          HFnegativedirectionReconoiseDIF1->SetXTitle("HFnegativedirectionReconoiseDIF1 \b");
          HFnegativedirectionReconoiseDIF1->SetMarkerColor(2);
          HFnegativedirectionReconoiseDIF1->SetLineColor(0);
          gPad->SetGridy();
          gPad->SetGridx();
          //	   gPad->SetLogy();
          if (kcountHFnegativedirectionReconoiseDIF1 == 1)
            HFnegativedirectionReconoiseDIF1->SetXTitle("DIF for HF- jeta =-41; depth = 1 \b");
          if (kcountHFnegativedirectionReconoiseDIF1 == 2)
            HFnegativedirectionReconoiseDIF1->SetXTitle("DIF for HF- jeta =-40; depth = 1 \b");
          if (kcountHFnegativedirectionReconoiseDIF1 == 3)
            HFnegativedirectionReconoiseDIF1->SetXTitle("DIF for HF- jeta =-39; depth = 1 \b");
          if (kcountHFnegativedirectionReconoiseDIF1 == 4)
            HFnegativedirectionReconoiseDIF1->SetXTitle("DIF for HF- jeta =-38; depth = 1 \b");
          if (kcountHFnegativedirectionReconoiseDIF1 == 5)
            HFnegativedirectionReconoiseDIF1->SetXTitle("DIF for HF- jeta =-37; depth = 1 \b");
          if (kcountHFnegativedirectionReconoiseDIF1 == 6)
            HFnegativedirectionReconoiseDIF1->SetXTitle("DIF for HF- jeta =-36; depth = 1 \b");
          if (kcountHFnegativedirectionReconoiseDIF1 == 7)
            HFnegativedirectionReconoiseDIF1->SetXTitle("DIF for HF- jeta =-35; depth = 1 \b");
          if (kcountHFnegativedirectionReconoiseDIF1 == 8)
            HFnegativedirectionReconoiseDIF1->SetXTitle("DIF for HF- jeta =-34; depth = 1 \b");
          if (kcountHFnegativedirectionReconoiseDIF1 == 9)
            HFnegativedirectionReconoiseDIF1->SetXTitle("DIF for HF- jeta =-33; depth = 1 \b");
          if (kcountHFnegativedirectionReconoiseDIF1 == 10)
            HFnegativedirectionReconoiseDIF1->SetXTitle("DIF for HF- jeta =-32; depth = 1 \b");
          if (kcountHFnegativedirectionReconoiseDIF1 == 11)
            HFnegativedirectionReconoiseDIF1->SetXTitle("DIF for HF- jeta =-31; depth = 1 \b");
          if (kcountHFnegativedirectionReconoiseDIF1 == 12)
            HFnegativedirectionReconoiseDIF1->SetXTitle("DIF for HF- jeta =-30; depth = 1 \b");
          if (kcountHFnegativedirectionReconoiseDIF1 == 13)
            HFnegativedirectionReconoiseDIF1->SetXTitle("DIF for HF- jeta =-29; depth = 1 \b");
          HFnegativedirectionReconoiseDIF1->Draw("Error");
          kcountHFnegativedirectionReconoiseDIF1++;
          if (kcountHFnegativedirectionReconoiseDIF1 > 13)
            break;  //
        }           //ccctest>0

      }  // for i
    }    //if(jeta-41 >= -41 && jeta-41 <= -29)
  }      //for jeta
  /////////////////
  c3x5->Update();
  c3x5->Print("DIFreconoiseNegativeDirectionhistD1PhiSymmetryDepth1HF.png");
  c3x5->Clear();
  // clean-up
  if (h2CeffHFnegativedirectionReconoiseDIF1)
    delete h2CeffHFnegativedirectionReconoiseDIF1;

  //========================================================================================== 1111115
  //======================================================================
  //======================================================================1D plot: R vs phi , different eta,  depth=2
  //  cout<<"      1D plot: R vs phi , different eta,  depth=2 *****" <<endl;
  c3x5->Clear();
  /////////////////
  c3x5->Divide(3, 5);
  c3x5->cd(1);
  int kcountHFnegativedirectionReconoiseDIF2 = 1;
  TH1F* h2CeffHFnegativedirectionReconoiseDIF2 = new TH1F("h2CeffHFnegativedirectionReconoiseDIF2", "", nphi, 0., 72.);
  for (int jeta = 0; jeta < njeta; jeta++) {
    // negativedirectionReconoiseDIF:
    if (jeta - 41 >= -41 && jeta - 41 <= -29) {
      //	     for (int i=0;i<ndepth;i++) {
      // depth=2
      for (int i = 1; i < 2; i++) {
        TH1F* HFnegativedirectionReconoiseDIF2 = (TH1F*)h2CeffHFnegativedirectionReconoiseDIF2->Clone("twod1");
        float ccctest = 0;  // to avoid empty massive elements
        for (int jphi = 0; jphi < nphi; jphi++) {
          double ccc1 = breconoiseHF[i][jeta][jphi];
          if (ccc1 != 0.) {
            HFnegativedirectionReconoiseDIF2->Fill(jphi, ccc1);
            ccctest = 1.;  //HFnegativedirectionReconoiseDIF2->SetBinError(i,0.01);
          }
        }  // for jphi
        if (ccctest > 0.) {
          //cout<<"555        kcountHFnegativedirectionReconoiseDIF2   =     "<<kcountHFnegativedirectionReconoiseDIF2  <<"   jeta-41=     "<< jeta-41 <<endl;
          c3x5->cd(kcountHFnegativedirectionReconoiseDIF2);
          HFnegativedirectionReconoiseDIF2->SetMarkerStyle(20);
          HFnegativedirectionReconoiseDIF2->SetMarkerSize(0.4);
          HFnegativedirectionReconoiseDIF2->GetYaxis()->SetLabelSize(0.04);
          HFnegativedirectionReconoiseDIF2->SetXTitle("HFnegativedirectionReconoiseDIF2 \b");
          HFnegativedirectionReconoiseDIF2->SetMarkerColor(2);
          HFnegativedirectionReconoiseDIF2->SetLineColor(0);
          gPad->SetGridy();
          gPad->SetGridx();
          //	   gPad->SetLogy();
          if (kcountHFnegativedirectionReconoiseDIF2 == 1)
            HFnegativedirectionReconoiseDIF2->SetXTitle("DIF for HF- jeta =-41; depth = 2 \b");
          if (kcountHFnegativedirectionReconoiseDIF2 == 2)
            HFnegativedirectionReconoiseDIF2->SetXTitle("DIF for HF- jeta =-40; depth = 2 \b");
          if (kcountHFnegativedirectionReconoiseDIF2 == 3)
            HFnegativedirectionReconoiseDIF2->SetXTitle("DIF for HF- jeta =-39; depth = 2 \b");
          if (kcountHFnegativedirectionReconoiseDIF2 == 4)
            HFnegativedirectionReconoiseDIF2->SetXTitle("DIF for HF- jeta =-38; depth = 2 \b");
          if (kcountHFnegativedirectionReconoiseDIF2 == 5)
            HFnegativedirectionReconoiseDIF2->SetXTitle("DIF for HF- jeta =-37; depth = 2 \b");
          if (kcountHFnegativedirectionReconoiseDIF2 == 6)
            HFnegativedirectionReconoiseDIF2->SetXTitle("DIF for HF- jeta =-36; depth = 2 \b");
          if (kcountHFnegativedirectionReconoiseDIF2 == 7)
            HFnegativedirectionReconoiseDIF2->SetXTitle("DIF for HF- jeta =-35; depth = 2 \b");
          if (kcountHFnegativedirectionReconoiseDIF2 == 8)
            HFnegativedirectionReconoiseDIF2->SetXTitle("DIF for HF- jeta =-34; depth = 2 \b");
          if (kcountHFnegativedirectionReconoiseDIF2 == 9)
            HFnegativedirectionReconoiseDIF2->SetXTitle("DIF for HF- jeta =-33; depth = 2 \b");
          if (kcountHFnegativedirectionReconoiseDIF2 == 10)
            HFnegativedirectionReconoiseDIF2->SetXTitle("DIF for HF- jeta =-32; depth = 2 \b");
          if (kcountHFnegativedirectionReconoiseDIF2 == 11)
            HFnegativedirectionReconoiseDIF2->SetXTitle("DIF for HF- jeta =-31; depth = 2 \b");
          if (kcountHFnegativedirectionReconoiseDIF2 == 12)
            HFnegativedirectionReconoiseDIF2->SetXTitle("DIF for HF- jeta =-30; depth = 2 \b");
          if (kcountHFnegativedirectionReconoiseDIF2 == 13)
            HFnegativedirectionReconoiseDIF2->SetXTitle("DIF for HF- jeta =-20; depth = 2 \b");
          HFnegativedirectionReconoiseDIF2->Draw("Error");
          kcountHFnegativedirectionReconoiseDIF2++;
          if (kcountHFnegativedirectionReconoiseDIF2 > 13)
            break;  // 4x6 = 24
        }           //ccctest>0

      }  // for i
    }    //if(jeta-41 >= -41 && jeta-41 <= -29)
  }      //for jeta
  /////////////////
  c3x5->Update();
  c3x5->Print("DIFreconoiseNegativeDirectionhistD1PhiSymmetryDepth2HF.png");
  c3x5->Clear();
  // clean-up
  if (h2CeffHFnegativedirectionReconoiseDIF2)
    delete h2CeffHFnegativedirectionReconoiseDIF2;

  //======================================================================================================================
  //======================================================================================================================
  //======================================================================================================================
  //======================================================================================================================
  //======================================================================================================================
  //======================================================================================================================
  //======================================================================================================================
  //                            DDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDD:

  //cout<<"    Start Vaiance: preparation  *****" <<endl;
  TH2F* reconoiseVariance1HF1 = (TH2F*)dir->FindObjectAny("h_recNoiseEnergy2_HF1");
  TH2F* reconoiseVariance0HF1 = (TH2F*)dir->FindObjectAny("h_recNoiseEnergy0_HF1");
  TH2F* reconoiseVarianceHF1 = (TH2F*)reconoiseVariance1HF1->Clone("reconoiseVarianceHF1");
  reconoiseVarianceHF1->Divide(reconoiseVariance1HF1, reconoiseVariance0HF1, 1, 1, "B");
  TH2F* reconoiseVariance1HF2 = (TH2F*)dir->FindObjectAny("h_recNoiseEnergy2_HF2");
  TH2F* reconoiseVariance0HF2 = (TH2F*)dir->FindObjectAny("h_recNoiseEnergy0_HF2");
  TH2F* reconoiseVarianceHF2 = (TH2F*)reconoiseVariance1HF2->Clone("reconoiseVarianceHF2");
  reconoiseVarianceHF2->Divide(reconoiseVariance1HF2, reconoiseVariance0HF2, 1, 1, "B");
  //cout<<"      Vaiance: preparation DONE *****" <<endl;
  //====================================================================== put Vaiance=Dispersia = Sig**2=<R**2> - (<R>)**2 into massive reconoisevarianceHF
  //                                                                                           = sum(R*R)/N - (sum(R)/N)**2
  for (int jeta = 0; jeta < njeta; jeta++) {
    if ((jeta - 41 >= -41 && jeta - 41 <= -29) || (jeta - 41 >= 28 && jeta - 41 <= 40)) {
      //preparation for PHI normalization:
      double sumreconoiseHF0 = 0;
      int nsumreconoiseHF0 = 0;
      double sumreconoiseHF1 = 0;
      int nsumreconoiseHF1 = 0;
      for (int jphi = 0; jphi < njphi; jphi++) {
        reconoisevarianceHF[0][jeta][jphi] = reconoiseVarianceHF1->GetBinContent(jeta + 1, jphi + 1);
        reconoisevarianceHF[1][jeta][jphi] = reconoiseVarianceHF2->GetBinContent(jeta + 1, jphi + 1);
        sumreconoiseHF0 += reconoisevarianceHF[0][jeta][jphi];
        ++nsumreconoiseHF0;
        sumreconoiseHF1 += reconoisevarianceHF[1][jeta][jphi];
        ++nsumreconoiseHF1;
      }  // phi
      // PHI normalization :
      for (int jphi = 0; jphi < njphi; jphi++) {
        if (reconoisevarianceHF[0][jeta][jphi] != 0.)
          reconoisevarianceHF[0][jeta][jphi] /= (sumreconoiseHF0 / nsumreconoiseHF0);
        if (reconoisevarianceHF[1][jeta][jphi] != 0.)
          reconoisevarianceHF[1][jeta][jphi] /= (sumreconoiseHF1 / nsumreconoiseHF1);
      }  // phi
      //       reconoisevarianceHF (D)           = sum(R*R)/N - (sum(R)/N)**2
      for (int jphi = 0; jphi < njphi; jphi++) {
        //	   cout<<"12 12 12   jeta=     "<< jeta <<"   jphi   =     "<<jphi  <<endl;
        reconoisevarianceHF[0][jeta][jphi] -= areconoiseHF[0][jeta][jphi] * areconoiseHF[0][jeta][jphi];
        reconoisevarianceHF[0][jeta][jphi] = fabs(reconoisevarianceHF[0][jeta][jphi]);
        reconoisevarianceHF[1][jeta][jphi] -= areconoiseHF[1][jeta][jphi] * areconoiseHF[1][jeta][jphi];
        reconoisevarianceHF[1][jeta][jphi] = fabs(reconoisevarianceHF[1][jeta][jphi]);
      }
    }
  }
  //cout<<"      Vaiance: DONE*****" <<endl;
  //------------------------  2D-eta/phi-plot: D, averaged over depthfs
  //======================================================================
  //======================================================================
  //cout<<"      R2D-eta/phi-plot: D, averaged over depthfs *****" <<endl;
  c1x1->Clear();
  /////////////////
  c1x0->Divide(1, 1);
  c1x0->cd(1);
  TH2F* DefzDreconoiseHF42D = new TH2F("DefzDreconoiseHF42D", "", neta, -41., 41., nphi, 0., 72.);
  TH2F* DefzDreconoiseHF42D0 = new TH2F("DefzDreconoiseHF42D0", "", neta, -41., 41., nphi, 0., 72.);
  TH2F* DefzDreconoiseHF42DF = (TH2F*)DefzDreconoiseHF42D0->Clone("DefzDreconoiseHF42DF");
  for (int i = 0; i < ndepth; i++) {
    for (int jeta = 0; jeta < neta; jeta++) {
      if ((jeta - 41 >= -41 && jeta - 41 <= -29) || (jeta - 41 >= 28 && jeta - 41 <= 40)) {
        for (int jphi = 0; jphi < nphi; jphi++) {
          double ccc1 = reconoisevarianceHF[i][jeta][jphi];
          int k2plot = jeta - 41;
          int kkk = k2plot;  //if(k2plot >0   kkk=k2plot+1; //-41 +41 !=0
          if (areconoiseHF[i][jeta][jphi] > 0.) {
            DefzDreconoiseHF42D->Fill(kkk, jphi, ccc1);
            DefzDreconoiseHF42D0->Fill(kkk, jphi, 1.);
          }
        }
      }
    }
  }
  DefzDreconoiseHF42DF->Divide(DefzDreconoiseHF42D, DefzDreconoiseHF42D0, 1, 1, "B");  // average A
  //    DefzDreconoiseHF1->Sumw2();
  gPad->SetGridy();
  gPad->SetGridx();  //      gPad->SetLogz();
  DefzDreconoiseHF42DF->SetMarkerStyle(20);
  DefzDreconoiseHF42DF->SetMarkerSize(0.4);
  DefzDreconoiseHF42DF->GetZaxis()->SetLabelSize(0.08);
  DefzDreconoiseHF42DF->SetXTitle("<D>_depth       #eta  \b");
  DefzDreconoiseHF42DF->SetYTitle("      #phi \b");
  DefzDreconoiseHF42DF->SetZTitle("<D>_depth \b");
  DefzDreconoiseHF42DF->SetMarkerColor(2);
  DefzDreconoiseHF42DF->SetLineColor(
      0);  //      DefzDreconoiseHF42DF->SetMaximum(1.000);  //      DefzDreconoiseHF42DF->SetMinimum(1.0);
  DefzDreconoiseHF42DF->Draw("COLZ");
  /////////////////
  c1x0->Update();
  c1x0->Print("DreconoiseGeneralD2PhiSymmetryHF.png");
  c1x0->Clear();
  // clean-up
  if (DefzDreconoiseHF42D)
    delete DefzDreconoiseHF42D;
  if (DefzDreconoiseHF42D0)
    delete DefzDreconoiseHF42D0;
  if (DefzDreconoiseHF42DF)
    delete DefzDreconoiseHF42DF;
  //====================================================================== 1D plot: D vs phi , averaged over depthfs & eta
  //======================================================================
  //cout<<"      1D plot: D vs phi , averaged over depthfs & eta *****" <<endl;
  c1x1->Clear();
  /////////////////
  c1x1->Divide(1, 1);
  c1x1->cd(1);
  TH1F* DefzDreconoiseHF41D = new TH1F("DefzDreconoiseHF41D", "", nphi, 0., 72.);
  TH1F* DefzDreconoiseHF41D0 = new TH1F("DefzDreconoiseHF41D0", "", nphi, 0., 72.);
  TH1F* DefzDreconoiseHF41DF = (TH1F*)DefzDreconoiseHF41D0->Clone("DefzDreconoiseHF41DF");

  for (int jphi = 0; jphi < nphi; jphi++) {
    for (int jeta = 0; jeta < neta; jeta++) {
      if ((jeta - 41 >= -41 && jeta - 41 <= -29) || (jeta - 41 >= 28 && jeta - 41 <= 40)) {
        for (int i = 0; i < ndepth; i++) {
          double ccc1 = reconoisevarianceHF[i][jeta][jphi];
          if (areconoiseHF[i][jeta][jphi] > 0.) {
            DefzDreconoiseHF41D->Fill(jphi, ccc1);
            DefzDreconoiseHF41D0->Fill(jphi, 1.);
          }
        }
      }
    }
  }
  //     DefzDreconoiseHF41D->Sumw2();DefzDreconoiseHF41D0->Sumw2();

  DefzDreconoiseHF41DF->Divide(DefzDreconoiseHF41D, DefzDreconoiseHF41D0, 1, 1, "B");  // R averaged over depthfs & eta
  DefzDreconoiseHF41D0->Sumw2();
  //    for (int jphi=1;jphi<73;jphi++) {DefzDreconoiseHF41DF->SetBinError(jphi,0.01);}
  gPad->SetGridy();
  gPad->SetGridx();  //      gPad->SetLogz();
  DefzDreconoiseHF41DF->SetMarkerStyle(20);
  DefzDreconoiseHF41DF->SetMarkerSize(1.4);
  DefzDreconoiseHF41DF->GetZaxis()->SetLabelSize(0.08);
  DefzDreconoiseHF41DF->SetXTitle("#phi  \b");
  DefzDreconoiseHF41DF->SetYTitle("  <D> \b");
  DefzDreconoiseHF41DF->SetZTitle("<D>_PHI  - AllDepthfs \b");
  DefzDreconoiseHF41DF->SetMarkerColor(4);
  DefzDreconoiseHF41DF->SetLineColor(
      4);  //  DefzDreconoiseHF41DF->SetMinimum(0.8);     DefzDreconoiseHF41DF->SetMinimum(-0.015);
  DefzDreconoiseHF41DF->Draw("Error");
  /////////////////
  c1x1->Update();
  c1x1->Print("DreconoiseGeneralD1PhiSymmetryHF.png");
  c1x1->Clear();
  // clean-up
  if (DefzDreconoiseHF41D)
    delete DefzDreconoiseHF41D;
  if (DefzDreconoiseHF41D0)
    delete DefzDreconoiseHF41D0;
  if (DefzDreconoiseHF41DF)
    delete DefzDreconoiseHF41DF;
  //========================================================================================== 14
  //======================================================================
  //======================================================================1D plot: D vs phi , different eta,  depth=1
  //cout<<"      1D plot: D vs phi , different eta,  depth=1 *****" <<endl;
  c3x5->Clear();
  /////////////////
  c3x5->Divide(3, 5);
  c3x5->cd(1);
  int kcountHFpositivedirectionReconoiseD1 = 1;
  TH1F* h2CeffHFpositivedirectionReconoiseD1 = new TH1F("h2CeffHFpositivedirectionReconoiseD1", "", nphi, 0., 72.);

  for (int jeta = 0; jeta < njeta; jeta++) {
    // positivedirectionReconoiseD:
    if (jeta - 41 >= 28 && jeta - 41 <= 40) {
      //	     for (int i=0;i<ndepth;i++) {
      // depth=1
      for (int i = 0; i < 1; i++) {
        TH1F* HFpositivedirectionReconoiseD1 = (TH1F*)h2CeffHFpositivedirectionReconoiseD1->Clone("twod1");

        float ccctest = 0;  // to avoid empty massive elements
        for (int jphi = 0; jphi < nphi; jphi++) {
          double ccc1 = reconoisevarianceHF[i][jeta][jphi];
          if (areconoiseHF[i][jeta][jphi] > 0.) {
            HFpositivedirectionReconoiseD1->Fill(jphi, ccc1);
            ccctest = 1.;  //HFpositivedirectionReconoiseD1->SetBinError(i,0.01);
          }
        }  // for jphi
        if (ccctest > 0.) {
          //cout<<"1414       kcountHFpositivedirectionReconoiseD1   =     "<<kcountHFpositivedirectionReconoiseD1  <<"   jeta-41=     "<< jeta-41 <<endl;
          c3x5->cd(kcountHFpositivedirectionReconoiseD1);
          HFpositivedirectionReconoiseD1->SetMarkerStyle(20);
          HFpositivedirectionReconoiseD1->SetMarkerSize(0.4);
          HFpositivedirectionReconoiseD1->GetYaxis()->SetLabelSize(0.04);
          HFpositivedirectionReconoiseD1->SetXTitle("HFpositivedirectionReconoiseD1 \b");
          HFpositivedirectionReconoiseD1->SetMarkerColor(2);
          HFpositivedirectionReconoiseD1->SetLineColor(0);
          gPad->SetGridy();
          gPad->SetGridx();
          //	   gPad->SetLogy();
          if (kcountHFpositivedirectionReconoiseD1 == 1)
            HFpositivedirectionReconoiseD1->SetXTitle("D for HF+ jeta = 28; depth = 1 \b");
          if (kcountHFpositivedirectionReconoiseD1 == 2)
            HFpositivedirectionReconoiseD1->SetXTitle("D for HF+ jeta = 29; depth = 1 \b");
          if (kcountHFpositivedirectionReconoiseD1 == 3)
            HFpositivedirectionReconoiseD1->SetXTitle("D for HF+ jeta = 30; depth = 1 \b");
          if (kcountHFpositivedirectionReconoiseD1 == 4)
            HFpositivedirectionReconoiseD1->SetXTitle("D for HF+ jeta = 31; depth = 1 \b");
          if (kcountHFpositivedirectionReconoiseD1 == 5)
            HFpositivedirectionReconoiseD1->SetXTitle("D for HF+ jeta = 32; depth = 1 \b");
          if (kcountHFpositivedirectionReconoiseD1 == 6)
            HFpositivedirectionReconoiseD1->SetXTitle("D for HF+ jeta = 33; depth = 1 \b");
          if (kcountHFpositivedirectionReconoiseD1 == 7)
            HFpositivedirectionReconoiseD1->SetXTitle("D for HF+ jeta = 34; depth = 1 \b");
          if (kcountHFpositivedirectionReconoiseD1 == 8)
            HFpositivedirectionReconoiseD1->SetXTitle("D for HF+ jeta = 35; depth = 1 \b");
          if (kcountHFpositivedirectionReconoiseD1 == 9)
            HFpositivedirectionReconoiseD1->SetXTitle("D for HF+ jeta = 36; depth = 1 \b");
          if (kcountHFpositivedirectionReconoiseD1 == 10)
            HFpositivedirectionReconoiseD1->SetXTitle("D for HF+ jeta = 37; depth = 1 \b");
          if (kcountHFpositivedirectionReconoiseD1 == 11)
            HFpositivedirectionReconoiseD1->SetXTitle("D for HF+ jeta = 38; depth = 1 \b");
          if (kcountHFpositivedirectionReconoiseD1 == 12)
            HFpositivedirectionReconoiseD1->SetXTitle("D for HF+ jeta = 39; depth = 1 \b");
          if (kcountHFpositivedirectionReconoiseD1 == 13)
            HFpositivedirectionReconoiseD1->SetXTitle("D for HF+ jeta = 40; depth = 1 \b");
          HFpositivedirectionReconoiseD1->Draw("Error");
          kcountHFpositivedirectionReconoiseD1++;
          if (kcountHFpositivedirectionReconoiseD1 > 13)
            break;  // 4x6 = 24
        }           //ccctest>0

      }  // for i
    }    //if(jeta-41 >= 28 && jeta-41 <= 40)
  }      //for jeta
  /////////////////
  c3x5->Update();
  c3x5->Print("DreconoisePositiveDirectionhistD1PhiSymmetryDepth1HF.png");
  c3x5->Clear();
  // clean-up
  if (h2CeffHFpositivedirectionReconoiseD1)
    delete h2CeffHFpositivedirectionReconoiseD1;
  //========================================================================================== 15
  //======================================================================
  //======================================================================1D plot: D vs phi , different eta,  depth=2
  //cout<<"      1D plot: D vs phi , different eta,  depth=2 *****" <<endl;
  c3x5->Clear();
  c3x5->Divide(3, 5);
  c3x5->cd(1);
  int kcountHFpositivedirectionReconoiseD2 = 1;
  TH1F* h2CeffHFpositivedirectionReconoiseD2 = new TH1F("h2CeffHFpositivedirectionReconoiseD2", "", nphi, 0., 72.);

  for (int jeta = 0; jeta < njeta; jeta++) {
    // positivedirectionReconoiseD:
    if (jeta - 41 >= 28 && jeta - 41 <= 40) {
      //	     for (int i=0;i<ndepth;i++) {
      // depth=2
      for (int i = 1; i < 2; i++) {
        TH1F* HFpositivedirectionReconoiseD2 = (TH1F*)h2CeffHFpositivedirectionReconoiseD2->Clone("twod1");

        float ccctest = 0;  // to avoid empty massive elements
        for (int jphi = 0; jphi < nphi; jphi++) {
          double ccc1 = reconoisevarianceHF[i][jeta][jphi];
          if (areconoiseHF[i][jeta][jphi] > 0.) {
            HFpositivedirectionReconoiseD2->Fill(jphi, ccc1);
            ccctest = 1.;  //HFpositivedirectionReconoiseD2->SetBinError(i,0.01);
          }
        }  // for jphi
        if (ccctest > 0.) {
          //cout<<"1515       kcountHFpositivedirectionReconoiseD2   =     "<<kcountHFpositivedirectionReconoiseD2  <<"   jeta-41=     "<< jeta-41 <<endl;
          c3x5->cd(kcountHFpositivedirectionReconoiseD2);
          HFpositivedirectionReconoiseD2->SetMarkerStyle(20);
          HFpositivedirectionReconoiseD2->SetMarkerSize(0.4);
          HFpositivedirectionReconoiseD2->GetYaxis()->SetLabelSize(0.04);
          HFpositivedirectionReconoiseD2->SetXTitle("HFpositivedirectionReconoiseD2 \b");
          HFpositivedirectionReconoiseD2->SetMarkerColor(2);
          HFpositivedirectionReconoiseD2->SetLineColor(0);
          gPad->SetGridy();
          gPad->SetGridx();
          //	   gPad->SetLogy();
          if (kcountHFpositivedirectionReconoiseD2 == 1)
            HFpositivedirectionReconoiseD2->SetXTitle("D for HF+ jeta = 28; depth = 2 \b");
          if (kcountHFpositivedirectionReconoiseD2 == 2)
            HFpositivedirectionReconoiseD2->SetXTitle("D for HF+ jeta = 29; depth = 2 \b");
          if (kcountHFpositivedirectionReconoiseD2 == 3)
            HFpositivedirectionReconoiseD2->SetXTitle("D for HF+ jeta = 30; depth = 2 \b");
          if (kcountHFpositivedirectionReconoiseD2 == 4)
            HFpositivedirectionReconoiseD2->SetXTitle("D for HF+ jeta = 31; depth = 2 \b");
          if (kcountHFpositivedirectionReconoiseD2 == 5)
            HFpositivedirectionReconoiseD2->SetXTitle("D for HF+ jeta = 32; depth = 2 \b");
          if (kcountHFpositivedirectionReconoiseD2 == 6)
            HFpositivedirectionReconoiseD2->SetXTitle("D for HF+ jeta = 33; depth = 2 \b");
          if (kcountHFpositivedirectionReconoiseD2 == 7)
            HFpositivedirectionReconoiseD2->SetXTitle("D for HF+ jeta = 34; depth = 2 \b");
          if (kcountHFpositivedirectionReconoiseD2 == 8)
            HFpositivedirectionReconoiseD2->SetXTitle("D for HF+ jeta = 35; depth = 2 \b");
          if (kcountHFpositivedirectionReconoiseD2 == 9)
            HFpositivedirectionReconoiseD2->SetXTitle("D for HF+ jeta = 36; depth = 2 \b");
          if (kcountHFpositivedirectionReconoiseD2 == 10)
            HFpositivedirectionReconoiseD2->SetXTitle("D for HF+ jeta = 37; depth = 2 \b");
          if (kcountHFpositivedirectionReconoiseD2 == 11)
            HFpositivedirectionReconoiseD2->SetXTitle("D for HF+ jeta = 38; depth = 2 \b");
          if (kcountHFpositivedirectionReconoiseD2 == 12)
            HFpositivedirectionReconoiseD2->SetXTitle("D for HF+ jeta = 39; depth = 2 \b");
          if (kcountHFpositivedirectionReconoiseD2 == 13)
            HFpositivedirectionReconoiseD2->SetXTitle("D for HF+ jeta = 40; depth = 2 \b");
          HFpositivedirectionReconoiseD2->Draw("Error");
          kcountHFpositivedirectionReconoiseD2++;
          if (kcountHFpositivedirectionReconoiseD2 > 13)
            break;  // 4x6 = 24
        }           //ccctest>0

      }  // for i
    }    //if(jeta-41 >= 28 && jeta-41 <= 40)
  }      //for jeta
  /////////////////
  c3x5->Update();
  c3x5->Print("DreconoisePositiveDirectionhistD1PhiSymmetryDepth2HF.png");
  c3x5->Clear();
  // clean-up
  if (h2CeffHFpositivedirectionReconoiseD2)
    delete h2CeffHFpositivedirectionReconoiseD2;
  //========================================================================================== 22222214
  //======================================================================
  //======================================================================1D plot: D vs phi , different eta,  depth=1
  //cout<<"      1D plot: D vs phi , different eta,  depth=1 *****" <<endl;
  c3x5->Clear();
  /////////////////
  c3x5->Divide(3, 5);
  c3x5->cd(1);
  int kcountHFnegativedirectionReconoiseD1 = 1;
  TH1F* h2CeffHFnegativedirectionReconoiseD1 = new TH1F("h2CeffHFnegativedirectionReconoiseD1", "", nphi, 0., 72.);

  for (int jeta = 0; jeta < njeta; jeta++) {
    // negativedirectionReconoiseD:
    if (jeta - 41 >= -41 && jeta - 41 <= -29) {
      //	     for (int i=0;i<ndepth;i++) {
      // depth=1
      for (int i = 0; i < 1; i++) {
        TH1F* HFnegativedirectionReconoiseD1 = (TH1F*)h2CeffHFnegativedirectionReconoiseD1->Clone("twod1");

        float ccctest = 0;  // to avoid empty massive elements
        for (int jphi = 0; jphi < nphi; jphi++) {
          double ccc1 = reconoisevarianceHF[i][jeta][jphi];
          if (areconoiseHF[i][jeta][jphi] > 0.) {
            HFnegativedirectionReconoiseD1->Fill(jphi, ccc1);
            ccctest = 1.;  //HFnegativedirectionReconoiseD1->SetBinError(i,0.01);
          }
        }  // for jphi
        if (ccctest > 0.) {
          //cout<<"1414       kcountHFnegativedirectionReconoiseD1   =     "<<kcountHFnegativedirectionReconoiseD1  <<"   jeta-41=     "<< jeta-41 <<endl;
          c3x5->cd(kcountHFnegativedirectionReconoiseD1);
          HFnegativedirectionReconoiseD1->SetMarkerStyle(20);
          HFnegativedirectionReconoiseD1->SetMarkerSize(0.4);
          HFnegativedirectionReconoiseD1->GetYaxis()->SetLabelSize(0.04);
          HFnegativedirectionReconoiseD1->SetXTitle("HFnegativedirectionReconoiseD1 \b");
          HFnegativedirectionReconoiseD1->SetMarkerColor(2);
          HFnegativedirectionReconoiseD1->SetLineColor(0);
          gPad->SetGridy();
          gPad->SetGridx();
          //	   gPad->SetLogy();
          if (kcountHFnegativedirectionReconoiseD1 == 1)
            HFnegativedirectionReconoiseD1->SetXTitle("D for HF- jeta =-41; depth = 1 \b");
          if (kcountHFnegativedirectionReconoiseD1 == 2)
            HFnegativedirectionReconoiseD1->SetXTitle("D for HF- jeta =-40; depth = 1 \b");
          if (kcountHFnegativedirectionReconoiseD1 == 3)
            HFnegativedirectionReconoiseD1->SetXTitle("D for HF- jeta =-39; depth = 1 \b");
          if (kcountHFnegativedirectionReconoiseD1 == 4)
            HFnegativedirectionReconoiseD1->SetXTitle("D for HF- jeta =-38; depth = 1 \b");
          if (kcountHFnegativedirectionReconoiseD1 == 5)
            HFnegativedirectionReconoiseD1->SetXTitle("D for HF- jeta =-37; depth = 1 \b");
          if (kcountHFnegativedirectionReconoiseD1 == 6)
            HFnegativedirectionReconoiseD1->SetXTitle("D for HF- jeta =-36; depth = 1 \b");
          if (kcountHFnegativedirectionReconoiseD1 == 7)
            HFnegativedirectionReconoiseD1->SetXTitle("D for HF- jeta =-35; depth = 1 \b");
          if (kcountHFnegativedirectionReconoiseD1 == 8)
            HFnegativedirectionReconoiseD1->SetXTitle("D for HF- jeta =-34; depth = 1 \b");
          if (kcountHFnegativedirectionReconoiseD1 == 9)
            HFnegativedirectionReconoiseD1->SetXTitle("D for HF- jeta =-33; depth = 1 \b");
          if (kcountHFnegativedirectionReconoiseD1 == 10)
            HFnegativedirectionReconoiseD1->SetXTitle("D for HF- jeta =-32; depth = 1 \b");
          if (kcountHFnegativedirectionReconoiseD1 == 11)
            HFnegativedirectionReconoiseD1->SetXTitle("D for HF- jeta =-31; depth = 1 \b");
          if (kcountHFnegativedirectionReconoiseD1 == 12)
            HFnegativedirectionReconoiseD1->SetXTitle("D for HF- jeta =-30; depth = 1 \b");
          if (kcountHFnegativedirectionReconoiseD1 == 13)
            HFnegativedirectionReconoiseD1->SetXTitle("D for HF- jeta =-29; depth = 1 \b");
          HFnegativedirectionReconoiseD1->Draw("Error");
          kcountHFnegativedirectionReconoiseD1++;
          if (kcountHFnegativedirectionReconoiseD1 > 13)
            break;  // 4x6 = 24
        }           //ccctest>0

      }  // for i
    }    //if(jeta-41 >= -41 && jeta-41 <= -29)
  }      //for jeta
  /////////////////
  c3x5->Update();
  c3x5->Print("DreconoiseNegativeDirectionhistD1PhiSymmetryDepth1HF.png");
  c3x5->Clear();
  // clean-up
  if (h2CeffHFnegativedirectionReconoiseD1)
    delete h2CeffHFnegativedirectionReconoiseD1;
  //========================================================================================== 22222215
  //======================================================================
  //======================================================================1D plot: D vs phi , different eta,  depth=2
  //cout<<"      1D plot: D vs phi , different eta,  depth=2 *****" <<endl;
  c3x5->Clear();
  c3x5->Divide(3, 5);
  c3x5->cd(1);
  int kcountHFnegativedirectionReconoiseD2 = 1;
  TH1F* h2CeffHFnegativedirectionReconoiseD2 = new TH1F("h2CeffHFnegativedirectionReconoiseD2", "", nphi, 0., 72.);

  for (int jeta = 0; jeta < njeta; jeta++) {
    // negativedirectionReconoiseD:
    if (jeta - 41 >= -41 && jeta - 41 <= -29) {
      //	     for (int i=0;i<ndepth;i++) {
      // depth=2
      for (int i = 1; i < 2; i++) {
        TH1F* HFnegativedirectionReconoiseD2 = (TH1F*)h2CeffHFnegativedirectionReconoiseD2->Clone("twod1");

        float ccctest = 0;  // to avoid empty massive elements
        for (int jphi = 0; jphi < nphi; jphi++) {
          double ccc1 = reconoisevarianceHF[i][jeta][jphi];
          if (areconoiseHF[i][jeta][jphi] > 0.) {
            HFnegativedirectionReconoiseD2->Fill(jphi, ccc1);
            ccctest = 1.;  //HFnegativedirectionReconoiseD2->SetBinError(i,0.01);
          }
        }  // for jphi
        if (ccctest > 0.) {
          //cout<<"1515       kcountHFnegativedirectionReconoiseD2   =     "<<kcountHFnegativedirectionReconoiseD2  <<"   jeta-41=     "<< jeta-41 <<endl;
          c3x5->cd(kcountHFnegativedirectionReconoiseD2);
          HFnegativedirectionReconoiseD2->SetMarkerStyle(20);
          HFnegativedirectionReconoiseD2->SetMarkerSize(0.4);
          HFnegativedirectionReconoiseD2->GetYaxis()->SetLabelSize(0.04);
          HFnegativedirectionReconoiseD2->SetXTitle("HFnegativedirectionReconoiseD2 \b");
          HFnegativedirectionReconoiseD2->SetMarkerColor(2);
          HFnegativedirectionReconoiseD2->SetLineColor(0);
          gPad->SetGridy();
          gPad->SetGridx();
          //	   gPad->SetLogy();
          if (kcountHFnegativedirectionReconoiseD2 == 1)
            HFnegativedirectionReconoiseD2->SetXTitle("D for HF- jeta =-41; depth = 2 \b");
          if (kcountHFnegativedirectionReconoiseD2 == 2)
            HFnegativedirectionReconoiseD2->SetXTitle("D for HF- jeta =-40; depth = 2 \b");
          if (kcountHFnegativedirectionReconoiseD2 == 3)
            HFnegativedirectionReconoiseD2->SetXTitle("D for HF- jeta =-39; depth = 2 \b");
          if (kcountHFnegativedirectionReconoiseD2 == 4)
            HFnegativedirectionReconoiseD2->SetXTitle("D for HF- jeta =-38; depth = 2 \b");
          if (kcountHFnegativedirectionReconoiseD2 == 5)
            HFnegativedirectionReconoiseD2->SetXTitle("D for HF- jeta =-37; depth = 2 \b");
          if (kcountHFnegativedirectionReconoiseD2 == 6)
            HFnegativedirectionReconoiseD2->SetXTitle("D for HF- jeta =-36; depth = 2 \b");
          if (kcountHFnegativedirectionReconoiseD2 == 7)
            HFnegativedirectionReconoiseD2->SetXTitle("D for HF- jeta =-35; depth = 2 \b");
          if (kcountHFnegativedirectionReconoiseD2 == 8)
            HFnegativedirectionReconoiseD2->SetXTitle("D for HF- jeta =-34; depth = 2 \b");
          if (kcountHFnegativedirectionReconoiseD2 == 9)
            HFnegativedirectionReconoiseD2->SetXTitle("D for HF- jeta =-33; depth = 2 \b");
          if (kcountHFnegativedirectionReconoiseD2 == 10)
            HFnegativedirectionReconoiseD2->SetXTitle("D for HF- jeta =-32; depth = 2 \b");
          if (kcountHFnegativedirectionReconoiseD2 == 11)
            HFnegativedirectionReconoiseD2->SetXTitle("D for HF- jeta =-31; depth = 2 \b");
          if (kcountHFnegativedirectionReconoiseD2 == 12)
            HFnegativedirectionReconoiseD2->SetXTitle("D for HF- jeta =-30; depth = 2 \b");
          if (kcountHFnegativedirectionReconoiseD2 == 13)
            HFnegativedirectionReconoiseD2->SetXTitle("D for HF- jeta =-29; depth = 2 \b");
          HFnegativedirectionReconoiseD2->Draw("Error");
          kcountHFnegativedirectionReconoiseD2++;
          if (kcountHFnegativedirectionReconoiseD2 > 13)
            break;  // 4x6 = 24
        }           //ccctest>0

      }  // for i
    }    //if(jeta-41 >= -41 && jeta-41 <= -29)
  }      //for jeta
  /////////////////
  c3x5->Update();
  c3x5->Print("DreconoiseNegativeDirectionhistD1PhiSymmetryDepth2HF.png");
  c3x5->Clear();
  // clean-up
  if (h2CeffHFnegativedirectionReconoiseD2)
    delete h2CeffHFnegativedirectionReconoiseD2;

  //=====================================================================       END of Reconoise HF for phi-symmetry
  //=====================================================================       END of Reconoise HF for phi-symmetry
  //=====================================================================       END of Reconoise HF for phi-symmetry
  //============================================================================================================       END of Reconoise for phi-symmetry
  //============================================================================================================       END of Reconoise for phi-symmetry
  //============================================================================================================       END of Reconoise for phi-symmetry

  //====================================================================================================================================================       END for phi-symmetry
  //====================================================================================================================================================       END for phi-symmetry
  //====================================================================================================================================================       END for phi-symmetry
  //====================================================================================================================================================

  //======================================================================
  // Creating each test kind for each subdet html pages:
  std::string raw_class, raw_class1, raw_class2, raw_class3;
  int ind = 0;

  for (int sub = 1; sub <= 4; sub++) {  //Subdetector: 1-HB, 2-HE, 3-HF, 4-HO
    ofstream htmlFileR, htmlFileN;
    if (sub == 1) {
      htmlFileR.open("HB_PhiSymmetryRecoSignal.html");
      htmlFileN.open("HB_PhiSymmetryRecoNoise.html");
    }
    if (sub == 2) {
      htmlFileR.open("HE_PhiSymmetryRecoSignal.html");
      htmlFileN.open("HE_PhiSymmetryRecoNoise.html");
    }

    if (sub == 4) {
      htmlFileR.open("HF_PhiSymmetryRecoSignal.html");
      htmlFileN.open("HF_PhiSymmetryRecoNoise.html");
    }

    ////////////////////////////////////////////////////////////////////////////////////////////// RecoSignal:
    ////////////////////////////////////////////////////////////////////////////////////////////// RecoSignal:
    ////////////////////////////////////////////////////////////////////////////////////////////// RecoSignal:

    htmlFileR << "</html><html xmlns=\"http://www.w3.org/1999/xhtml\">" << std::endl;
    htmlFileR << "<head>" << std::endl;
    htmlFileR << "<meta http-equiv=\"Content-Type\" content=\"text/html\"/>" << std::endl;
    htmlFileR << "<title> Remote Monitoring Tool </title>" << std::endl;
    htmlFileR << "<style type=\"text/css\">" << std::endl;
    htmlFileR << " body,td{ background-color: #FFFFCC; font-family: arial, arial ce, helvetica; font-size: 12px; }"
              << std::endl;
    htmlFileR << "   td.s0 { font-family: arial, arial ce, helvetica; }" << std::endl;
    htmlFileR << "   td.s1 { font-family: arial, arial ce, helvetica; font-weight: bold; background-color: #FFC169; "
                 "text-align: center;}"
              << std::endl;
    htmlFileR << "   td.s2 { font-family: arial, arial ce, helvetica; background-color: #eeeeee; }" << std::endl;
    htmlFileR << "   td.s3 { font-family: arial, arial ce, helvetica; background-color: #d0d0d0; }" << std::endl;
    htmlFileR << "   td.s4 { font-family: arial, arial ce, helvetica; background-color: #FFC169; }" << std::endl;
    htmlFileR << "</style>" << std::endl;
    htmlFileR << "<body>" << std::endl;

    /////////////////////////////////////////////// RRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRR

    if (sub == 1)
      htmlFileR << "<h1> Phi-symmetry for Calibration Group, HB, RUN = " << runnumber << " </h1>" << std::endl;
    if (sub == 2)
      htmlFileR << "<h1> Phi-symmetry for Calibration Group, HE, RUN = " << runnumber << " </h1>" << std::endl;
    if (sub == 4)
      htmlFileR << "<h1> Phi-symmetry for Calibration Group, HF, RUN = " << runnumber << " </h1>" << std::endl;
    htmlFileR << "<br>" << std::endl;

    htmlFileR << "<h2> 1: R = R_depth_ieta_iphi = E_depth_ieta_iphi/E_depth_ieta </h3>" << std::endl;
    htmlFileR << "<h3> 1A: eta/phi-plot: R, averaged over depthes </h3>" << std::endl;
    //     htmlFileR << "<h4> Legend: Bins less "<<Pedest[0][sub]<<" correpond to bad Pedestals </h4>"<< std::endl;
    if (sub == 1)
      htmlFileR << " <img src=\"RrecosignalGeneralD2PhiSymmetryHB.png\" />" << std::endl;
    if (sub == 2)
      htmlFileR << " <img src=\"RrecosignalGeneralD2PhiSymmetryHE.png\" />" << std::endl;
    if (sub == 4)
      htmlFileR << " <img src=\"RrecosignalGeneralD2PhiSymmetryHF.png\" />" << std::endl;
    htmlFileR << "<br>" << std::endl;

    htmlFileR << "<h3> 1B: R vs phi , averaged over depthes & eta </h3>" << std::endl;
    //     htmlFileR << "<h4> Channel legend: white - good, other colour - bad. </h4>"<< std::endl;
    if (sub == 1)
      htmlFileR << " <img src=\"RrecosignalGeneralD1PhiSymmetryHB.png\" />" << std::endl;
    if (sub == 2)
      htmlFileR << " <img src=\"RrecosignalGeneralD1PhiSymmetryHE.png\" />" << std::endl;
    if (sub == 4)
      htmlFileR << " <img src=\"RrecosignalGeneralD1PhiSymmetryHF.png\" />" << std::endl;
    htmlFileR << "<br>" << std::endl;

    ///////////////////////////////////////////   PositiveDirection:

    /////////////////////////////////////////////// R different Depthes:
    htmlFileR << "<h2>  Positive direction, R = R_depth_ieta_iphi = E_depth_ieta_iphi/E_depth_ieta </h3>" << std::endl;
    htmlFileR << "<h3> 1C: R vs phi , different eta, Depth1 </h3>" << std::endl;
    //     htmlFileR << "<h4> Channel legend: white - good, other colour - bad. </h4>"<< std::endl;
    if (sub == 1)
      htmlFileR << " <img src=\"RrecosignalPositiveDirectionhistD1PhiSymmetryDepth1HB.png\" />" << std::endl;
    if (sub == 2)
      htmlFileR << " <img src=\"RrecosignalPositiveDirectionhistD1PhiSymmetryDepth1HE.png\" />" << std::endl;
    if (sub == 4)
      htmlFileR << " <img src=\"RrecosignalPositiveDirectionhistD1PhiSymmetryDepth1HF.png\" />" << std::endl;
    htmlFileR << "<br>" << std::endl;

    htmlFileR << "<h3> 1D: R vs phi , different eta, Depth2 </h3>" << std::endl;
    //     htmlFileR << "<h4> Channel legend: white - good, other colour - bad. </h4>"<< std::endl;
    if (sub == 1)
      htmlFileR << " <img src=\"RrecosignalPositiveDirectionhistD1PhiSymmetryDepth2HB.png\" />" << std::endl;
    if (sub == 2)
      htmlFileR << " <img src=\"RrecosignalPositiveDirectionhistD1PhiSymmetryDepth2HE.png\" />" << std::endl;
    if (sub == 4)
      htmlFileR << " <img src=\"RrecosignalPositiveDirectionhistD1PhiSymmetryDepth2HF.png\" />" << std::endl;
    htmlFileR << "<br>" << std::endl;

    if (sub == 1 || sub == 2)
      htmlFileR << "<h3> 1E: R vs phi , different eta, Depth3 </h3>" << std::endl;
    //     htmlFileR << "<h4> Channel legend: white - good, other colour - bad. </h4>"<< std::endl;
    if (sub == 1)
      htmlFileR << " <img src=\"RrecosignalPositiveDirectionhistD1PhiSymmetryDepth3HB.png\" />" << std::endl;
    if (sub == 2)
      htmlFileR << " <img src=\"RrecosignalPositiveDirectionhistD1PhiSymmetryDepth3HE.png\" />" << std::endl;
    htmlFileR << "<br>" << std::endl;

    if (sub == 1 || sub == 2)
      htmlFileR << "<h3> 1F: R vs phi , different eta, Depth4 </h3>" << std::endl;
    //     htmlFileR << "<h4> Channel legend: white - good, other colour - bad. </h4>"<< std::endl;
    if (sub == 1)
      htmlFileR << " <img src=\"RrecosignalPositiveDirectionhistD1PhiSymmetryDepth4HB.png\" />" << std::endl;
    if (sub == 2)
      htmlFileR << " <img src=\"RrecosignalPositiveDirectionhistD1PhiSymmetryDepth4HE.png\" />" << std::endl;
    htmlFileR << "<br>" << std::endl;

    if (sub == 2)
      htmlFileR << "<h3> 1G: R vs phi , different eta, Depth5 </h3>" << std::endl;
    //     htmlFileR << "<h4> Channel legend: white - good, other colour - bad. </h4>"<< std::endl;
    if (sub == 2)
      htmlFileR << " <img src=\"RrecosignalPositiveDirectionhistD1PhiSymmetryDepth5HE.png\" />" << std::endl;
    htmlFileR << "<br>" << std::endl;

    if (sub == 2)
      htmlFileR << "<h3> 1H: R vs phi , different eta, Depth6 </h3>" << std::endl;
    //     htmlFileR << "<h4> Channel legend: white - good, other colour - bad. </h4>"<< std::endl;
    if (sub == 2)
      htmlFileR << " <img src=\"RrecosignalPositiveDirectionhistD1PhiSymmetryDepth6HE.png\" />" << std::endl;
    htmlFileR << "<br>" << std::endl;

    if (sub == 2)
      htmlFileR << "<h3> 1I: R vs phi , different eta, Depth7 </h3>" << std::endl;
    //     htmlFileR << "<h4> Channel legend: white - good, other colour - bad. </h4>"<< std::endl;
    if (sub == 2)
      htmlFileR << " <img src=\"RrecosignalPositiveDirectionhistD1PhiSymmetryDepth7HE.png\" />" << std::endl;
    htmlFileR << "<br>" << std::endl;

    /////////////////////////////////////////////// DDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDD
    htmlFileR << "<h2> 2: D(recosignalvarianceSignalhe)   </h3>" << std::endl;
    htmlFileR << "<h3> 2A: eta/phi-plot: D(recosignalvarianceSignalhe), averaged over depthes </h3>" << std::endl;
    //     htmlFileR << "<h4> Legend: Bins less "<<Pedest[0][sub]<<" correpond to bad Pedestals </h4>"<< std::endl;
    if (sub == 1)
      htmlFileR << " <img src=\"DrecosignalGeneralD2PhiSymmetryHB.png\" />" << std::endl;
    if (sub == 2)
      htmlFileR << " <img src=\"DrecosignalGeneralD2PhiSymmetryHE.png\" />" << std::endl;
    if (sub == 4)
      htmlFileR << " <img src=\"DrecosignalGeneralD2PhiSymmetryHF.png\" />" << std::endl;
    htmlFileR << "<br>" << std::endl;

    htmlFileR << "<h3> 2B: D(recosignalvarianceSignalhe) vs phi , averaged over depthes & eta </h3>" << std::endl;
    //     htmlFileR << "<h4> Channel legend: white - good, other colour - bad. </h4>"<< std::endl;
    if (sub == 1)
      htmlFileR << " <img src=\"DrecosignalGeneralD1PhiSymmetryHB.png\" />" << std::endl;
    if (sub == 2)
      htmlFileR << " <img src=\"DrecosignalGeneralD1PhiSymmetryHE.png\" />" << std::endl;
    if (sub == 4)
      htmlFileR << " <img src=\"DrecosignalGeneralD1PhiSymmetryHF.png\" />" << std::endl;
    htmlFileR << "<br>" << std::endl;

    ///////////////////////////////////////////   PositiveDirection:
    ///////////////////////////////////////////////D  different Depthes:
    htmlFileR << "<h2>  Positive direction, D(recosignalvarianceSignalhe) </h3>" << std::endl;
    htmlFileR << "<h3> 2C: D(recosignalvarianceSignalhe) vs phi , different eta, Depth1 </h3>" << std::endl;
    //     htmlFileR << "<h4> Channel legend: white - good, other colour - bad. </h4>"<< std::endl;
    if (sub == 1)
      htmlFileR << " <img src=\"DrecosignalPositiveDirectionhistD1PhiSymmetryDepth1HB.png\" />" << std::endl;
    if (sub == 2)
      htmlFileR << " <img src=\"DrecosignalPositiveDirectionhistD1PhiSymmetryDepth1HE.png\" />" << std::endl;
    if (sub == 4)
      htmlFileR << " <img src=\"DrecosignalPositiveDirectionhistD1PhiSymmetryDepth1HF.png\" />" << std::endl;
    htmlFileR << "<br>" << std::endl;

    htmlFileR << "<h3> 2.D. D(recosignalvarianceSignalhe) vs phi , different eta, Depth2 </h3>" << std::endl;
    //     htmlFileR << "<h4> Channel legend: white - good, other colour - bad. </h4>"<< std::endl;
    if (sub == 1)
      htmlFileR << " <img src=\"DrecosignalPositiveDirectionhistD1PhiSymmetryDepth2HB.png\" />" << std::endl;
    if (sub == 2)
      htmlFileR << " <img src=\"DrecosignalPositiveDirectionhistD1PhiSymmetryDepth2HE.png\" />" << std::endl;
    if (sub == 4)
      htmlFileR << " <img src=\"DrecosignalPositiveDirectionhistD1PhiSymmetryDepth2HF.png\" />" << std::endl;
    htmlFileR << "<br>" << std::endl;

    if (sub == 1 || sub == 2)
      htmlFileR << "<h3> 2E: D(recosignalvarianceSignalhe) vs phi , different eta, Depth3 </h3>" << std::endl;
    //     htmlFileR << "<h4> Channel legend: white - good, other colour - bad. </h4>"<< std::endl;
    if (sub == 1)
      htmlFileR << " <img src=\"DrecosignalPositiveDirectionhistD1PhiSymmetryDepth3HB.png\" />" << std::endl;
    if (sub == 2)
      htmlFileR << " <img src=\"DrecosignalPositiveDirectionhistD1PhiSymmetryDepth3HE.png\" />" << std::endl;
    htmlFileR << "<br>" << std::endl;

    if (sub == 1 || sub == 2)
      htmlFileR << "<h3> 2F: D(recosignalvarianceSignalhe) vs phi , different eta, Depth4 </h3>" << std::endl;
    //     htmlFileR << "<h4> Channel legend: white - good, other colour - bad. </h4>"<< std::endl;
    if (sub == 1)
      htmlFileR << " <img src=\"DrecosignalPositiveDirectionhistD1PhiSymmetryDepth4HB.png\" />" << std::endl;
    if (sub == 2)
      htmlFileR << " <img src=\"DrecosignalPositiveDirectionhistD1PhiSymmetryDepth4HE.png\" />" << std::endl;
    htmlFileR << "<br>" << std::endl;

    if (sub == 2)
      htmlFileR << "<h3> 2G: D(recosignalvarianceSignalhe) vs phi , different eta, Depth5 </h3>" << std::endl;
    //     htmlFileR << "<h4> Channel legend: white - good, other colour - bad. </h4>"<< std::endl;
    if (sub == 2)
      htmlFileR << " <img src=\"DrecosignalPositiveDirectionhistD1PhiSymmetryDepth5HE.png\" />" << std::endl;
    htmlFileR << "<br>" << std::endl;

    if (sub == 2)
      htmlFileR << "<h3> 2H: D(recosignalvarianceSignalhe) vs phi , different eta, Depth6 </h3>" << std::endl;
    //     htmlFileR << "<h4> Channel legend: white - good, other colour - bad. </h4>"<< std::endl;
    if (sub == 2)
      htmlFileR << " <img src=\"DrecosignalPositiveDirectionhistD1PhiSymmetryDepth6HE.png\" />" << std::endl;
    htmlFileR << "<br>" << std::endl;

    if (sub == 2)
      htmlFileR << "<h3> 2I: D(recosignalvarianceSignalhe) vs phi , different eta, Depth7 </h3>" << std::endl;
    //     htmlFileR << "<h4> Channel legend: white - good, other colour - bad. </h4>"<< std::endl;
    if (sub == 2)
      htmlFileR << " <img src=\"DrecosignalPositiveDirectionhistD1PhiSymmetryDepth7HE.png\" />" << std::endl;
    htmlFileR << "<br>" << std::endl;

    ///////////////////////////////////////////   NegativeDirection:
    /////////////////////////////////////////////// RRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRR
    htmlFileR << "<h2> 3:  Negative direction, R = R_depth_ieta_iphi = E_depth_ieta_iphi/E_depth_ieta </h3>"
              << std::endl;

    /////////////////////////////////////////////// different Depthes:
    htmlFileR << "<h3> 3C: R vs phi , different eta, Depth1 </h3>" << std::endl;
    //     htmlFileR << "<h4> Channel legend: white - good, other colour - bad. </h4>"<< std::endl;
    if (sub == 1)
      htmlFileR << " <img src=\"RrecosignalNegativeDirectionhistD1PhiSymmetryDepth1HB.png\" />" << std::endl;
    if (sub == 2)
      htmlFileR << " <img src=\"RrecosignalNegativeDirectionhistD1PhiSymmetryDepth1HE.png\" />" << std::endl;
    if (sub == 4)
      htmlFileR << " <img src=\"RrecosignalNegativeDirectionhistD1PhiSymmetryDepth1HF.png\" />" << std::endl;
    htmlFileR << "<br>" << std::endl;

    htmlFileR << "<h3> 3D: R vs phi , different eta, Depth2 </h3>" << std::endl;
    //     htmlFileR << "<h4> Channel legend: white - good, other colour - bad. </h4>"<< std::endl;
    if (sub == 1)
      htmlFileR << " <img src=\"RrecosignalNegativeDirectionhistD1PhiSymmetryDepth2HB.png\" />" << std::endl;
    if (sub == 2)
      htmlFileR << " <img src=\"RrecosignalNegativeDirectionhistD1PhiSymmetryDepth2HE.png\" />" << std::endl;
    if (sub == 4)
      htmlFileR << " <img src=\"RrecosignalNegativeDirectionhistD1PhiSymmetryDepth2HF.png\" />" << std::endl;
    htmlFileR << "<br>" << std::endl;

    if (sub == 1 || sub == 2)
      htmlFileR << "<h3> 3E: R vs phi , different eta, Depth3 </h3>" << std::endl;
    //     htmlFileR << "<h4> Channel legend: white - good, other colour - bad. </h4>"<< std::endl;
    if (sub == 1)
      htmlFileR << " <img src=\"RrecosignalNegativeDirectionhistD1PhiSymmetryDepth3HB.png\" />" << std::endl;
    if (sub == 2)
      htmlFileR << " <img src=\"RrecosignalNegativeDirectionhistD1PhiSymmetryDepth3HE.png\" />" << std::endl;
    htmlFileR << "<br>" << std::endl;

    if (sub == 1 || sub == 2)
      htmlFileR << "<h3> 3F: R vs phi , different eta, Depth4 </h3>" << std::endl;
    //     htmlFileR << "<h4> Channel legend: white - good, other colour - bad. </h4>"<< std::endl;
    if (sub == 1)
      htmlFileR << " <img src=\"RrecosignalNegativeDirectionhistD1PhiSymmetryDepth4HB.png\" />" << std::endl;
    if (sub == 2)
      htmlFileR << " <img src=\"RrecosignalNegativeDirectionhistD1PhiSymmetryDepth4HE.png\" />" << std::endl;
    htmlFileR << "<br>" << std::endl;

    if (sub == 2)
      htmlFileR << "<h3> 3G: R vs phi , different eta, Depth5 </h3>" << std::endl;
    //     htmlFileR << "<h4> Channel legend: white - good, other colour - bad. </h4>"<< std::endl;
    if (sub == 2)
      htmlFileR << " <img src=\"RrecosignalNegativeDirectionhistD1PhiSymmetryDepth5HE.png\" />" << std::endl;
    htmlFileR << "<br>" << std::endl;

    if (sub == 2)
      htmlFileR << "<h3> 3H: R vs phi , different eta, Depth6 </h3>" << std::endl;
    //     htmlFileR << "<h4> Channel legend: white - good, other colour - bad. </h4>"<< std::endl;
    if (sub == 2)
      htmlFileR << " <img src=\"RrecosignalNegativeDirectionhistD1PhiSymmetryDepth6HE.png\" />" << std::endl;
    htmlFileR << "<br>" << std::endl;

    if (sub == 2)
      htmlFileR << "<h3> 3I: R vs phi , different eta, Depth7 </h3>" << std::endl;
    //     htmlFileR << "<h4> Channel legend: white - good, other colour - bad. </h4>"<< std::endl;
    if (sub == 2)
      htmlFileR << " <img src=\"RrecosignalNegativeDirectionhistD1PhiSymmetryDepth7HE.png\" />" << std::endl;
    htmlFileR << "<br>" << std::endl;

    /////////////////////////////////////////////// DDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDD
    htmlFileR << "<h2> 4: Negative direction,   D(recosignalvarianceSignalhe)   </h3>" << std::endl;
    /////////////////////////////////////////////// different Depthes:
    htmlFileR << "<h3> 4C: D(recosignalvarianceSignalhe) vs phi , different eta, Depth1 </h3>" << std::endl;
    //     htmlFileR << "<h4> Channel legend: white - good, other colour - bad. </h4>"<< std::endl;
    if (sub == 1)
      htmlFileR << " <img src=\"DrecosignalNegativeDirectionhistD1PhiSymmetryDepth1HB.png\" />" << std::endl;
    if (sub == 2)
      htmlFileR << " <img src=\"DrecosignalNegativeDirectionhistD1PhiSymmetryDepth1HE.png\" />" << std::endl;
    if (sub == 4)
      htmlFileR << " <img src=\"DrecosignalNegativeDirectionhistD1PhiSymmetryDepth1HF.png\" />" << std::endl;
    htmlFileR << "<br>" << std::endl;

    htmlFileR << "<h3> 4.D. D(recosignalvarianceSignalhe) vs phi , different eta, Depth2 </h3>" << std::endl;
    //     htmlFileR << "<h4> Channel legend: white - good, other colour - bad. </h4>"<< std::endl;
    if (sub == 1)
      htmlFileR << " <img src=\"DrecosignalNegativeDirectionhistD1PhiSymmetryDepth2HB.png\" />" << std::endl;
    if (sub == 2)
      htmlFileR << " <img src=\"DrecosignalNegativeDirectionhistD1PhiSymmetryDepth2HE.png\" />" << std::endl;
    if (sub == 4)
      htmlFileR << " <img src=\"DrecosignalNegativeDirectionhistD1PhiSymmetryDepth2HF.png\" />" << std::endl;
    htmlFileR << "<br>" << std::endl;

    if (sub == 1 || sub == 2)
      htmlFileR << "<h3> 4E: D(recosignalvarianceSignalhe) vs phi , different eta, Depth3 </h3>" << std::endl;
    //     htmlFileR << "<h4> Channel legend: white - good, other colour - bad. </h4>"<< std::endl;
    if (sub == 1)
      htmlFileR << " <img src=\"DrecosignalNegativeDirectionhistD1PhiSymmetryDepth3HB.png\" />" << std::endl;
    if (sub == 2)
      htmlFileR << " <img src=\"DrecosignalNegativeDirectionhistD1PhiSymmetryDepth3HE.png\" />" << std::endl;
    htmlFileR << "<br>" << std::endl;

    if (sub == 1 || sub == 2)
      htmlFileR << "<h3> 4F: D(recosignalvarianceSignalhe) vs phi , different eta, Depth4 </h3>" << std::endl;
    //     htmlFileR << "<h4> Channel legend: white - good, other colour - bad. </h4>"<< std::endl;
    if (sub == 1)
      htmlFileR << " <img src=\"DrecosignalNegativeDirectionhistD1PhiSymmetryDepth4HB.png\" />" << std::endl;
    if (sub == 2)
      htmlFileR << " <img src=\"DrecosignalNegativeDirectionhistD1PhiSymmetryDepth4HE.png\" />" << std::endl;
    htmlFileR << "<br>" << std::endl;

    if (sub == 2)
      htmlFileR << "<h3> 4G: D(recosignalvarianceSignalhe) vs phi , different eta, Depth5 </h3>" << std::endl;
    //     htmlFileR << "<h4> Channel legend: white - good, other colour - bad. </h4>"<< std::endl;
    if (sub == 2)
      htmlFileR << " <img src=\"DrecosignalNegativeDirectionhistD1PhiSymmetryDepth5HE.png\" />" << std::endl;
    htmlFileR << "<br>" << std::endl;

    if (sub == 2)
      htmlFileR << "<h3> 4H: D(recosignalvarianceSignalhe) vs phi , different eta, Depth6 </h3>" << std::endl;
    //     htmlFileR << "<h4> Channel legend: white - good, other colour - bad. </h4>"<< std::endl;
    if (sub == 2)
      htmlFileR << " <img src=\"DrecosignalNegativeDirectionhistD1PhiSymmetryDepth6HE.png\" />" << std::endl;
    htmlFileR << "<br>" << std::endl;

    if (sub == 2)
      htmlFileR << "<h3> 4I: D(recosignalvarianceSignalhe) vs phi , different eta, Depth7 </h3>" << std::endl;
    //     htmlFileR << "<h4> Channel legend: white - good, other colour - bad. </h4>"<< std::endl;
    if (sub == 2)
      htmlFileR << " <img src=\"DrecosignalNegativeDirectionhistD1PhiSymmetryDepth7HE.png\" />" << std::endl;
    htmlFileR << "<br>" << std::endl;
    ///////////////////////////////////////////
    htmlFileR.close();
    /////////////////////////////////////////// end of Recosignal
    //
    ////////////////////////////////////////////////////////////////////////////////////////////// RecoNoise:
    ////////////////////////////////////////////////////////////////////////////////////////////// RecoNoise:
    ////////////////////////////////////////////////////////////////////////////////////////////// RecoNoise:

    htmlFileN << "</html><html xmlns=\"http://www.w3.org/1999/xhtml\">" << std::endl;
    htmlFileN << "<head>" << std::endl;
    htmlFileN << "<meta http-equiv=\"Content-Type\" content=\"text/html\"/>" << std::endl;
    htmlFileN << "<title> Remote Monitoring Tool </title>" << std::endl;
    htmlFileN << "<style type=\"text/css\">" << std::endl;
    htmlFileN << " body,td{ background-color: #FFFFCC; font-family: arial, arial ce, helvetica; font-size: 12px; }"
              << std::endl;
    htmlFileN << "   td.s0 { font-family: arial, arial ce, helvetica; }" << std::endl;
    htmlFileN << "   td.s1 { font-family: arial, arial ce, helvetica; font-weight: bold; background-color: #FFC169; "
                 "text-align: center;}"
              << std::endl;
    htmlFileN << "   td.s2 { font-family: arial, arial ce, helvetica; background-color: #eeeeee; }" << std::endl;
    htmlFileN << "   td.s3 { font-family: arial, arial ce, helvetica; background-color: #d0d0d0; }" << std::endl;
    htmlFileN << "   td.s4 { font-family: arial, arial ce, helvetica; background-color: #FFC169; }" << std::endl;
    htmlFileN << "</style>" << std::endl;
    htmlFileN << "<body>" << std::endl;

    /////////////////////////////////////////////// RRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRR

    if (sub == 1)
      htmlFileN << "<h1> Phi-symmetry for Calibration Group, HB, RUN = " << runnumber << " </h1>" << std::endl;
    if (sub == 2)
      htmlFileN << "<h1> Phi-symmetry for Calibration Group, HE, RUN = " << runnumber << " </h1>" << std::endl;
    if (sub == 4)
      htmlFileN << "<h1> Phi-symmetry for Calibration Group, HF, RUN = " << runnumber << " </h1>" << std::endl;
    htmlFileN << "<br>" << std::endl;

    htmlFileN << "<h2> 1: R = R_depth_ieta_iphi = E_depth_ieta_iphi/E_depth_ieta </h3>" << std::endl;
    htmlFileN << "<h3> 1A: eta/phi-plot: R, averaged over depthes </h3>" << std::endl;
    //     htmlFileN << "<h4> Legend: Bins less "<<Pedest[0][sub]<<" correpond to bad Pedestals </h4>"<< std::endl;
    if (sub == 1)
      htmlFileN << " <img src=\"RreconoiseGeneralD2PhiSymmetryHB.png\" />" << std::endl;
    if (sub == 2)
      htmlFileN << " <img src=\"RreconoiseGeneralD2PhiSymmetryHE.png\" />" << std::endl;
    if (sub == 4)
      htmlFileN << " <img src=\"RreconoiseGeneralD2PhiSymmetryHF.png\" />" << std::endl;
    htmlFileN << "<br>" << std::endl;

    htmlFileN << "<h3> 1B: R vs phi , averaged over depthes & eta </h3>" << std::endl;
    //     htmlFileN << "<h4> Channel legend: white - good, other colour - bad. </h4>"<< std::endl;
    if (sub == 1)
      htmlFileN << " <img src=\"RreconoiseGeneralD1PhiSymmetryHB.png\" />" << std::endl;
    if (sub == 2)
      htmlFileN << " <img src=\"RreconoiseGeneralD1PhiSymmetryHE.png\" />" << std::endl;
    if (sub == 4)
      htmlFileN << " <img src=\"RreconoiseGeneralD1PhiSymmetryHF.png\" />" << std::endl;
    htmlFileN << "<br>" << std::endl;

    ///////////////////////////////////////////   PositiveDirection:

    /////////////////////////////////////////////// R different Depthes:
    htmlFileN << "<h2>  Positive direction, R = R_depth_ieta_iphi = E_depth_ieta_iphi/E_depth_ieta </h3>" << std::endl;
    htmlFileN << "<h3> 1C: R vs phi , different eta, Depth1 </h3>" << std::endl;
    //     htmlFileN << "<h4> Channel legend: white - good, other colour - bad. </h4>"<< std::endl;
    if (sub == 1)
      htmlFileN << " <img src=\"RreconoisePositiveDirectionhistD1PhiSymmetryDepth1HB.png\" />" << std::endl;
    if (sub == 2)
      htmlFileN << " <img src=\"RreconoisePositiveDirectionhistD1PhiSymmetryDepth1HE.png\" />" << std::endl;
    if (sub == 4)
      htmlFileN << " <img src=\"RreconoisePositiveDirectionhistD1PhiSymmetryDepth1HF.png\" />" << std::endl;
    htmlFileN << "<br>" << std::endl;

    htmlFileN << "<h3> 1D: R vs phi , different eta, Depth2 </h3>" << std::endl;
    //     htmlFileN << "<h4> Channel legend: white - good, other colour - bad. </h4>"<< std::endl;
    if (sub == 1)
      htmlFileN << " <img src=\"RreconoisePositiveDirectionhistD1PhiSymmetryDepth2HB.png\" />" << std::endl;
    if (sub == 2)
      htmlFileN << " <img src=\"RreconoisePositiveDirectionhistD1PhiSymmetryDepth2HE.png\" />" << std::endl;
    if (sub == 4)
      htmlFileN << " <img src=\"RreconoisePositiveDirectionhistD1PhiSymmetryDepth2HF.png\" />" << std::endl;
    htmlFileN << "<br>" << std::endl;

    if (sub == 1 || sub == 2)
      htmlFileN << "<h3> 1E: R vs phi , different eta, Depth3 </h3>" << std::endl;
    //     htmlFileN << "<h4> Channel legend: white - good, other colour - bad. </h4>"<< std::endl;
    if (sub == 1)
      htmlFileN << " <img src=\"RreconoisePositiveDirectionhistD1PhiSymmetryDepth3HB.png\" />" << std::endl;
    if (sub == 2)
      htmlFileN << " <img src=\"RreconoisePositiveDirectionhistD1PhiSymmetryDepth3HE.png\" />" << std::endl;
    htmlFileN << "<br>" << std::endl;

    if (sub == 1 || sub == 2)
      htmlFileN << "<h3> 1F: R vs phi , different eta, Depth4 </h3>" << std::endl;
    //     htmlFileN << "<h4> Channel legend: white - good, other colour - bad. </h4>"<< std::endl;
    if (sub == 1)
      htmlFileN << " <img src=\"RreconoisePositiveDirectionhistD1PhiSymmetryDepth4HB.png\" />" << std::endl;
    if (sub == 2)
      htmlFileN << " <img src=\"RreconoisePositiveDirectionhistD1PhiSymmetryDepth4HE.png\" />" << std::endl;
    htmlFileN << "<br>" << std::endl;

    if (sub == 2)
      htmlFileN << "<h3> 1G: R vs phi , different eta, Depth5 </h3>" << std::endl;
    //     htmlFileN << "<h4> Channel legend: white - good, other colour - bad. </h4>"<< std::endl;
    if (sub == 2)
      htmlFileN << " <img src=\"RreconoisePositiveDirectionhistD1PhiSymmetryDepth5HE.png\" />" << std::endl;
    htmlFileN << "<br>" << std::endl;

    if (sub == 2)
      htmlFileN << "<h3> 1H: R vs phi , different eta, Depth6 </h3>" << std::endl;
    //     htmlFileN << "<h4> Channel legend: white - good, other colour - bad. </h4>"<< std::endl;
    if (sub == 2)
      htmlFileN << " <img src=\"RreconoisePositiveDirectionhistD1PhiSymmetryDepth6HE.png\" />" << std::endl;
    htmlFileN << "<br>" << std::endl;

    if (sub == 2)
      htmlFileN << "<h3> 1I: R vs phi , different eta, Depth7 </h3>" << std::endl;
    //     htmlFileN << "<h4> Channel legend: white - good, other colour - bad. </h4>"<< std::endl;
    if (sub == 2)
      htmlFileN << " <img src=\"RreconoisePositiveDirectionhistD1PhiSymmetryDepth7HE.png\" />" << std::endl;
    htmlFileN << "<br>" << std::endl;

    /////////////////////////////////////////////// DDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDD
    htmlFileN << "<h2> 2: D(reconoisevarianceNoisehe)   </h3>" << std::endl;
    htmlFileN << "<h3> 2A: eta/phi-plot: D(reconoisevarianceNoisehe), averaged over depthes </h3>" << std::endl;
    //     htmlFileN << "<h4> Legend: Bins less "<<Pedest[0][sub]<<" correpond to bad Pedestals </h4>"<< std::endl;
    if (sub == 1)
      htmlFileN << " <img src=\"DreconoiseGeneralD2PhiSymmetryHB.png\" />" << std::endl;
    if (sub == 2)
      htmlFileN << " <img src=\"DreconoiseGeneralD2PhiSymmetryHE.png\" />" << std::endl;
    if (sub == 4)
      htmlFileN << " <img src=\"DreconoiseGeneralD2PhiSymmetryHF.png\" />" << std::endl;
    htmlFileN << "<br>" << std::endl;

    htmlFileN << "<h3> 2B: D(reconoisevarianceNoisehe) vs phi , averaged over depthes & eta </h3>" << std::endl;
    //     htmlFileN << "<h4> Channel legend: white - good, other colour - bad. </h4>"<< std::endl;
    if (sub == 1)
      htmlFileN << " <img src=\"DreconoiseGeneralD1PhiSymmetryHB.png\" />" << std::endl;
    if (sub == 2)
      htmlFileN << " <img src=\"DreconoiseGeneralD1PhiSymmetryHE.png\" />" << std::endl;
    if (sub == 4)
      htmlFileN << " <img src=\"DreconoiseGeneralD1PhiSymmetryHF.png\" />" << std::endl;
    htmlFileN << "<br>" << std::endl;

    ///////////////////////////////////////////   PositiveDirection:
    ///////////////////////////////////////////////D  different Depthes:
    htmlFileN << "<h2>  Positive direction, D(reconoisevarianceNoisehe) </h3>" << std::endl;
    htmlFileN << "<h3> 2C: D(reconoisevarianceNoisehe) vs phi , different eta, Depth1 </h3>" << std::endl;
    //     htmlFileN << "<h4> Channel legend: white - good, other colour - bad. </h4>"<< std::endl;
    if (sub == 1)
      htmlFileN << " <img src=\"DreconoisePositiveDirectionhistD1PhiSymmetryDepth1HB.png\" />" << std::endl;
    if (sub == 2)
      htmlFileN << " <img src=\"DreconoisePositiveDirectionhistD1PhiSymmetryDepth1HE.png\" />" << std::endl;
    if (sub == 4)
      htmlFileN << " <img src=\"DreconoisePositiveDirectionhistD1PhiSymmetryDepth1HF.png\" />" << std::endl;
    htmlFileN << "<br>" << std::endl;

    htmlFileN << "<h3> 2.D. D(reconoisevarianceNoisehe) vs phi , different eta, Depth2 </h3>" << std::endl;
    //     htmlFileN << "<h4> Channel legend: white - good, other colour - bad. </h4>"<< std::endl;
    if (sub == 1)
      htmlFileN << " <img src=\"DreconoisePositiveDirectionhistD1PhiSymmetryDepth2HB.png\" />" << std::endl;
    if (sub == 2)
      htmlFileN << " <img src=\"DreconoisePositiveDirectionhistD1PhiSymmetryDepth2HE.png\" />" << std::endl;
    if (sub == 4)
      htmlFileN << " <img src=\"DreconoisePositiveDirectionhistD1PhiSymmetryDepth2HF.png\" />" << std::endl;
    htmlFileN << "<br>" << std::endl;

    if (sub == 1 || sub == 2)
      htmlFileN << "<h3> 2E: D(reconoisevarianceNoisehe) vs phi , different eta, Depth3 </h3>" << std::endl;
    //     htmlFileN << "<h4> Channel legend: white - good, other colour - bad. </h4>"<< std::endl;
    if (sub == 1)
      htmlFileN << " <img src=\"DreconoisePositiveDirectionhistD1PhiSymmetryDepth3HB.png\" />" << std::endl;
    if (sub == 2)
      htmlFileN << " <img src=\"DreconoisePositiveDirectionhistD1PhiSymmetryDepth3HE.png\" />" << std::endl;
    htmlFileN << "<br>" << std::endl;

    if (sub == 1 || sub == 2)
      htmlFileN << "<h3> 2F: D(reconoisevarianceNoisehe) vs phi , different eta, Depth4 </h3>" << std::endl;
    //     htmlFileN << "<h4> Channel legend: white - good, other colour - bad. </h4>"<< std::endl;
    if (sub == 1)
      htmlFileN << " <img src=\"DreconoisePositiveDirectionhistD1PhiSymmetryDepth4HB.png\" />" << std::endl;
    if (sub == 2)
      htmlFileN << " <img src=\"DreconoisePositiveDirectionhistD1PhiSymmetryDepth4HE.png\" />" << std::endl;
    htmlFileN << "<br>" << std::endl;

    if (sub == 2)
      htmlFileN << "<h3> 2G: D(reconoisevarianceNoisehe) vs phi , different eta, Depth5 </h3>" << std::endl;
    //     htmlFileN << "<h4> Channel legend: white - good, other colour - bad. </h4>"<< std::endl;
    if (sub == 2)
      htmlFileN << " <img src=\"DreconoisePositiveDirectionhistD1PhiSymmetryDepth5HE.png\" />" << std::endl;
    htmlFileN << "<br>" << std::endl;

    if (sub == 2)
      htmlFileN << "<h3> 2H: D(reconoisevarianceNoisehe) vs phi , different eta, Depth6 </h3>" << std::endl;
    //     htmlFileN << "<h4> Channel legend: white - good, other colour - bad. </h4>"<< std::endl;
    if (sub == 2)
      htmlFileN << " <img src=\"DreconoisePositiveDirectionhistD1PhiSymmetryDepth6HE.png\" />" << std::endl;
    htmlFileN << "<br>" << std::endl;

    if (sub == 2)
      htmlFileN << "<h3> 2I: D(reconoisevarianceNoisehe) vs phi , different eta, Depth7 </h3>" << std::endl;
    //     htmlFileN << "<h4> Channel legend: white - good, other colour - bad. </h4>"<< std::endl;
    if (sub == 2)
      htmlFileN << " <img src=\"DreconoisePositiveDirectionhistD1PhiSymmetryDepth7HE.png\" />" << std::endl;
    htmlFileN << "<br>" << std::endl;

    ///////////////////////////////////////////   NegativeDirection:
    /////////////////////////////////////////////// RRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRR
    htmlFileN << "<h2> 3:  Negative direction, R = R_depth_ieta_iphi = E_depth_ieta_iphi/E_depth_ieta </h3>"
              << std::endl;

    /////////////////////////////////////////////// different Depthes:
    htmlFileN << "<h3> 3C: R vs phi , different eta, Depth1 </h3>" << std::endl;
    //     htmlFileN << "<h4> Channel legend: white - good, other colour - bad. </h4>"<< std::endl;
    if (sub == 1)
      htmlFileN << " <img src=\"RreconoiseNegativeDirectionhistD1PhiSymmetryDepth1HB.png\" />" << std::endl;
    if (sub == 2)
      htmlFileN << " <img src=\"RreconoiseNegativeDirectionhistD1PhiSymmetryDepth1HE.png\" />" << std::endl;
    if (sub == 4)
      htmlFileN << " <img src=\"RreconoiseNegativeDirectionhistD1PhiSymmetryDepth1HF.png\" />" << std::endl;
    htmlFileN << "<br>" << std::endl;

    htmlFileN << "<h3> 3D: R vs phi , different eta, Depth2 </h3>" << std::endl;
    //     htmlFileN << "<h4> Channel legend: white - good, other colour - bad. </h4>"<< std::endl;
    if (sub == 1)
      htmlFileN << " <img src=\"RreconoiseNegativeDirectionhistD1PhiSymmetryDepth2HB.png\" />" << std::endl;
    if (sub == 2)
      htmlFileN << " <img src=\"RreconoiseNegativeDirectionhistD1PhiSymmetryDepth2HE.png\" />" << std::endl;
    if (sub == 4)
      htmlFileN << " <img src=\"RreconoiseNegativeDirectionhistD1PhiSymmetryDepth2HF.png\" />" << std::endl;
    htmlFileN << "<br>" << std::endl;

    if (sub == 1 || sub == 2)
      htmlFileN << "<h3> 3E: R vs phi , different eta, Depth3 </h3>" << std::endl;
    //     htmlFileN << "<h4> Channel legend: white - good, other colour - bad. </h4>"<< std::endl;
    if (sub == 1)
      htmlFileN << " <img src=\"RreconoiseNegativeDirectionhistD1PhiSymmetryDepth3HB.png\" />" << std::endl;
    if (sub == 2)
      htmlFileN << " <img src=\"RreconoiseNegativeDirectionhistD1PhiSymmetryDepth3HE.png\" />" << std::endl;
    htmlFileN << "<br>" << std::endl;

    if (sub == 1 || sub == 2)
      htmlFileN << "<h3> 3F: R vs phi , different eta, Depth4 </h3>" << std::endl;
    //     htmlFileN << "<h4> Channel legend: white - good, other colour - bad. </h4>"<< std::endl;
    if (sub == 1)
      htmlFileN << " <img src=\"RreconoiseNegativeDirectionhistD1PhiSymmetryDepth4HB.png\" />" << std::endl;
    if (sub == 2)
      htmlFileN << " <img src=\"RreconoiseNegativeDirectionhistD1PhiSymmetryDepth4HE.png\" />" << std::endl;
    htmlFileN << "<br>" << std::endl;

    if (sub == 2)
      htmlFileN << "<h3> 3G: R vs phi , different eta, Depth5 </h3>" << std::endl;
    //     htmlFileN << "<h4> Channel legend: white - good, other colour - bad. </h4>"<< std::endl;
    if (sub == 2)
      htmlFileN << " <img src=\"RreconoiseNegativeDirectionhistD1PhiSymmetryDepth5HE.png\" />" << std::endl;
    htmlFileN << "<br>" << std::endl;

    if (sub == 2)
      htmlFileN << "<h3> 3H: R vs phi , different eta, Depth6 </h3>" << std::endl;
    //     htmlFileN << "<h4> Channel legend: white - good, other colour - bad. </h4>"<< std::endl;
    if (sub == 2)
      htmlFileN << " <img src=\"RreconoiseNegativeDirectionhistD1PhiSymmetryDepth6HE.png\" />" << std::endl;
    htmlFileN << "<br>" << std::endl;

    if (sub == 2)
      htmlFileN << "<h3> 3I: R vs phi , different eta, Depth7 </h3>" << std::endl;
    //     htmlFileN << "<h4> Channel legend: white - good, other colour - bad. </h4>"<< std::endl;
    if (sub == 2)
      htmlFileN << " <img src=\"RreconoiseNegativeDirectionhistD1PhiSymmetryDepth7HE.png\" />" << std::endl;
    htmlFileN << "<br>" << std::endl;

    /////////////////////////////////////////////// DDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDD
    htmlFileN << "<h2> 4: Negative direction,   D(reconoisevarianceNoisehe)   </h3>" << std::endl;
    /////////////////////////////////////////////// different Depthes:
    htmlFileN << "<h3> 4C: D(reconoisevarianceNoisehe) vs phi , different eta, Depth1 </h3>" << std::endl;
    //     htmlFileN << "<h4> Channel legend: white - good, other colour - bad. </h4>"<< std::endl;
    if (sub == 1)
      htmlFileN << " <img src=\"DreconoiseNegativeDirectionhistD1PhiSymmetryDepth1HB.png\" />" << std::endl;
    if (sub == 2)
      htmlFileN << " <img src=\"DreconoiseNegativeDirectionhistD1PhiSymmetryDepth1HE.png\" />" << std::endl;
    if (sub == 4)
      htmlFileN << " <img src=\"DreconoiseNegativeDirectionhistD1PhiSymmetryDepth1HF.png\" />" << std::endl;
    htmlFileN << "<br>" << std::endl;

    htmlFileN << "<h3> 4.D. D(reconoisevarianceNoisehe) vs phi , different eta, Depth2 </h3>" << std::endl;
    //     htmlFileN << "<h4> Channel legend: white - good, other colour - bad. </h4>"<< std::endl;
    if (sub == 1)
      htmlFileN << " <img src=\"DreconoiseNegativeDirectionhistD1PhiSymmetryDepth2HB.png\" />" << std::endl;
    if (sub == 2)
      htmlFileN << " <img src=\"DreconoiseNegativeDirectionhistD1PhiSymmetryDepth2HE.png\" />" << std::endl;
    if (sub == 4)
      htmlFileN << " <img src=\"DreconoiseNegativeDirectionhistD1PhiSymmetryDepth2HF.png\" />" << std::endl;
    htmlFileN << "<br>" << std::endl;

    if (sub == 1 || sub == 2)
      htmlFileN << "<h3> 4E: D(reconoisevarianceNoisehe) vs phi , different eta, Depth3 </h3>" << std::endl;
    //     htmlFileN << "<h4> Channel legend: white - good, other colour - bad. </h4>"<< std::endl;
    if (sub == 1)
      htmlFileN << " <img src=\"DreconoiseNegativeDirectionhistD1PhiSymmetryDepth3HB.png\" />" << std::endl;
    if (sub == 2)
      htmlFileN << " <img src=\"DreconoiseNegativeDirectionhistD1PhiSymmetryDepth3HE.png\" />" << std::endl;
    htmlFileN << "<br>" << std::endl;

    if (sub == 1 || sub == 2)
      htmlFileN << "<h3> 4F: D(reconoisevarianceNoisehe) vs phi , different eta, Depth4 </h3>" << std::endl;
    //     htmlFileN << "<h4> Channel legend: white - good, other colour - bad. </h4>"<< std::endl;
    if (sub == 1)
      htmlFileN << " <img src=\"DreconoiseNegativeDirectionhistD1PhiSymmetryDepth4HB.png\" />" << std::endl;
    if (sub == 2)
      htmlFileN << " <img src=\"DreconoiseNegativeDirectionhistD1PhiSymmetryDepth4HE.png\" />" << std::endl;
    htmlFileN << "<br>" << std::endl;

    if (sub == 2)
      htmlFileN << "<h3> 4G: D(reconoisevarianceNoisehe) vs phi , different eta, Depth5 </h3>" << std::endl;
    //     htmlFileN << "<h4> Channel legend: white - good, other colour - bad. </h4>"<< std::endl;
    if (sub == 2)
      htmlFileN << " <img src=\"DreconoiseNegativeDirectionhistD1PhiSymmetryDepth5HE.png\" />" << std::endl;
    htmlFileN << "<br>" << std::endl;

    if (sub == 2)
      htmlFileN << "<h3> 4H: D(reconoisevarianceNoisehe) vs phi , different eta, Depth6 </h3>" << std::endl;
    //     htmlFileN << "<h4> Channel legend: white - good, other colour - bad. </h4>"<< std::endl;
    if (sub == 2)
      htmlFileN << " <img src=\"DreconoiseNegativeDirectionhistD1PhiSymmetryDepth6HE.png\" />" << std::endl;
    htmlFileN << "<br>" << std::endl;

    if (sub == 2)
      htmlFileN << "<h3> 4I: D(reconoisevarianceNoisehe) vs phi , different eta, Depth7 </h3>" << std::endl;
    //     htmlFileN << "<h4> Channel legend: white - good, other colour - bad. </h4>"<< std::endl;
    if (sub == 2)
      htmlFileN << " <img src=\"DreconoiseNegativeDirectionhistD1PhiSymmetryDepth7HE.png\" />" << std::endl;
    htmlFileN << "<br>" << std::endl;
    ///////////////////////////////////////////

    /////////////////////////////////////////////// DIFDIFDIFDIFDIFDIFDIFDIFDIFDIFDIFDIF

    if (sub == 1)
      htmlFileN << "<h1> Only for Noise RecHits these lines below, HB, RUN = " << runnumber << " </h1>" << std::endl;
    if (sub == 2)
      htmlFileN << "<h1> Only for Noise RecHits these lines below, HE, RUN = " << runnumber << " </h1>" << std::endl;
    if (sub == 4)
      htmlFileN << "<h1> Only for Noise RecHits these lines below, HF, RUN = " << runnumber << " </h1>" << std::endl;
    htmlFileN << "<br>" << std::endl;

    htmlFileN << "<h2> 5: DIF = DIF_depth_ieta_iphi = E_depth_ieta_iphi - E_depth_ieta </h3>" << std::endl;
    htmlFileN << "<h3> 5A: eta/phi-plot: DIF, averaged over depthes </h3>" << std::endl;
    //     htmlFileN << "<h4> Legend: Bins less "<<Pedest[0][sub]<<" correpond to bad Pedestals </h4>"<< std::endl;
    if (sub == 1)
      htmlFileN << " <img src=\"DIFreconoiseGeneralD2PhiSymmetryHB.png\" />" << std::endl;
    if (sub == 2)
      htmlFileN << " <img src=\"DIFreconoiseGeneralD2PhiSymmetryHE.png\" />" << std::endl;
    if (sub == 4)
      htmlFileN << " <img src=\"DIFreconoiseGeneralD2PhiSymmetryHF.png\" />" << std::endl;
    htmlFileN << "<br>" << std::endl;

    htmlFileN << "<h3> 5B: DIF vs phi , averaged over depthes & eta </h3>" << std::endl;
    //     htmlFileN << "<h4> Channel legend: white - good, other colour - bad. </h4>"<< std::endl;
    if (sub == 1)
      htmlFileN << " <img src=\"DIFreconoiseGeneralD1PhiSymmetryHB.png\" />" << std::endl;
    if (sub == 2)
      htmlFileN << " <img src=\"DIFreconoiseGeneralD1PhiSymmetryHE.png\" />" << std::endl;
    if (sub == 4)
      htmlFileN << " <img src=\"DIFreconoiseGeneralD1PhiSymmetryHF.png\" />" << std::endl;
    htmlFileN << "<br>" << std::endl;

    /////////////////////////////////////////// DIF  PositiveDirection:

    /////////////////////////////////////////////// DIF different Depthes:
    htmlFileN << "<h2>  Positive direction, DIF = DIF_depth_ieta_iphi = E_depth_ieta_iphi - E_depth_ieta </h3>"
              << std::endl;
    htmlFileN << "<h3> 5C: DIF vs phi , different eta, Depth1 </h3>" << std::endl;
    //     htmlFileN << "<h4> Channel legend: white - good, other colour - bad. </h4>"<< std::endl;
    if (sub == 1)
      htmlFileN << " <img src=\"DIFreconoisePositiveDirectionhistD1PhiSymmetryDepth1HB.png\" />" << std::endl;
    if (sub == 2)
      htmlFileN << " <img src=\"DIFreconoisePositiveDirectionhistD1PhiSymmetryDepth1HE.png\" />" << std::endl;
    if (sub == 4)
      htmlFileN << " <img src=\"DIFreconoisePositiveDirectionhistD1PhiSymmetryDepth1HF.png\" />" << std::endl;
    htmlFileN << "<br>" << std::endl;

    htmlFileN << "<h3> 5D: DIF vs phi , different eta, Depth2 </h3>" << std::endl;
    //     htmlFileN << "<h4> Channel legend: white - good, other colour - bad. </h4>"<< std::endl;
    if (sub == 1)
      htmlFileN << " <img src=\"DIFreconoisePositiveDirectionhistD1PhiSymmetryDepth2HB.png\" />" << std::endl;
    if (sub == 2)
      htmlFileN << " <img src=\"DIFreconoisePositiveDirectionhistD1PhiSymmetryDepth2HE.png\" />" << std::endl;
    if (sub == 4)
      htmlFileN << " <img src=\"DIFreconoisePositiveDirectionhistD1PhiSymmetryDepth2HF.png\" />" << std::endl;
    htmlFileN << "<br>" << std::endl;

    if (sub == 1 || sub == 2)
      htmlFileN << "<h3> 1E: DIF vs phi , different eta, Depth3 </h3>" << std::endl;
    //     htmlFileN << "<h4> Channel legend: white - good, other colour - bad. </h4>"<< std::endl;
    if (sub == 1)
      htmlFileN << " <img src=\"DIFreconoisePositiveDirectionhistD1PhiSymmetryDepth3HB.png\" />" << std::endl;
    if (sub == 2)
      htmlFileN << " <img src=\"DIFreconoisePositiveDirectionhistD1PhiSymmetryDepth3HE.png\" />" << std::endl;
    htmlFileN << "<br>" << std::endl;

    if (sub == 1 || sub == 2)
      htmlFileN << "<h3> 5F: DIF vs phi , different eta, Depth4 </h3>" << std::endl;
    //     htmlFileN << "<h4> Channel legend: white - good, other colour - bad. </h4>"<< std::endl;
    if (sub == 1)
      htmlFileN << " <img src=\"DIFreconoisePositiveDirectionhistD1PhiSymmetryDepth4HB.png\" />" << std::endl;
    if (sub == 2)
      htmlFileN << " <img src=\"DIFreconoisePositiveDirectionhistD1PhiSymmetryDepth4HE.png\" />" << std::endl;
    htmlFileN << "<br>" << std::endl;

    if (sub == 2)
      htmlFileN << "<h3> 5G: DIF vs phi , different eta, Depth5 </h3>" << std::endl;
    //     htmlFileN << "<h4> Channel legend: white - good, other colour - bad. </h4>"<< std::endl;
    if (sub == 2)
      htmlFileN << " <img src=\"DIFreconoisePositiveDirectionhistD1PhiSymmetryDepth5HE.png\" />" << std::endl;
    htmlFileN << "<br>" << std::endl;

    if (sub == 2)
      htmlFileN << "<h3> 5H: DIF vs phi , different eta, Depth6 </h3>" << std::endl;
    //     htmlFileN << "<h4> Channel legend: white - good, other colour - bad. </h4>"<< std::endl;
    if (sub == 2)
      htmlFileN << " <img src=\"DIFreconoisePositiveDirectionhistD1PhiSymmetryDepth6HE.png\" />" << std::endl;
    htmlFileN << "<br>" << std::endl;

    if (sub == 2)
      htmlFileN << "<h3> 5I: DIF vs phi , different eta, Depth7 </h3>" << std::endl;
    //     htmlFileN << "<h4> Channel legend: white - good, other colour - bad. </h4>"<< std::endl;
    if (sub == 2)
      htmlFileN << " <img src=\"DIFreconoisePositiveDirectionhistD1PhiSymmetryDepth7HE.png\" />" << std::endl;
    htmlFileN << "<br>" << std::endl;

    /////////////////////////////////////////// DIF  NegativeDirection:

    /////////////////////////////////////////////// DIF different Depthes:
    htmlFileN << "<h2>  Negative direction, DIF = DIF_depth_ieta_iphi = E_depth_ieta_iphi - E_depth_ieta </h3>"
              << std::endl;
    htmlFileN << "<h3> 5C: DIF vs phi , different eta, Depth1 </h3>" << std::endl;
    //     htmlFileN << "<h4> Channel legend: white - good, other colour - bad. </h4>"<< std::endl;
    if (sub == 1)
      htmlFileN << " <img src=\"DIFreconoiseNegativeDirectionhistD1PhiSymmetryDepth1HB.png\" />" << std::endl;
    if (sub == 2)
      htmlFileN << " <img src=\"DIFreconoiseNegativeDirectionhistD1PhiSymmetryDepth1HE.png\" />" << std::endl;
    if (sub == 4)
      htmlFileN << " <img src=\"DIFreconoiseNegativeDirectionhistD1PhiSymmetryDepth1HF.png\" />" << std::endl;
    htmlFileN << "<br>" << std::endl;

    htmlFileN << "<h3> 5D: DIF vs phi , different eta, Depth2 </h3>" << std::endl;
    //     htmlFileN << "<h4> Channel legend: white - good, other colour - bad. </h4>"<< std::endl;
    if (sub == 1)
      htmlFileN << " <img src=\"DIFreconoiseNegativeDirectionhistD1PhiSymmetryDepth2HB.png\" />" << std::endl;
    if (sub == 2)
      htmlFileN << " <img src=\"DIFreconoiseNegativeDirectionhistD1PhiSymmetryDepth2HE.png\" />" << std::endl;
    if (sub == 4)
      htmlFileN << " <img src=\"DIFreconoiseNegativeDirectionhistD1PhiSymmetryDepth2HF.png\" />" << std::endl;
    htmlFileN << "<br>" << std::endl;

    if (sub == 1 || sub == 2)
      htmlFileN << "<h3> 5E: DIF vs phi , different eta, Depth3 </h3>" << std::endl;
    //     htmlFileN << "<h4> Channel legend: white - good, other colour - bad. </h4>"<< std::endl;
    if (sub == 1)
      htmlFileN << " <img src=\"DIFreconoiseNegativeDirectionhistD1PhiSymmetryDepth3HB.png\" />" << std::endl;
    if (sub == 2)
      htmlFileN << " <img src=\"DIFreconoiseNegativeDirectionhistD1PhiSymmetryDepth3HE.png\" />" << std::endl;
    htmlFileN << "<br>" << std::endl;

    if (sub == 1 || sub == 2)
      htmlFileN << "<h3> 5F: DIF vs phi , different eta, Depth4 </h3>" << std::endl;
    //     htmlFileN << "<h4> Channel legend: white - good, other colour - bad. </h4>"<< std::endl;
    if (sub == 1)
      htmlFileN << " <img src=\"DIFreconoiseNegativeDirectionhistD1PhiSymmetryDepth4HB.png\" />" << std::endl;
    if (sub == 2)
      htmlFileN << " <img src=\"DIFreconoiseNegativeDirectionhistD1PhiSymmetryDepth4HE.png\" />" << std::endl;
    htmlFileN << "<br>" << std::endl;

    if (sub == 2)
      htmlFileN << "<h3> 5G: DIF vs phi , different eta, Depth5 </h3>" << std::endl;
    //     htmlFileN << "<h4> Channel legend: white - good, other colour - bad. </h4>"<< std::endl;
    if (sub == 2)
      htmlFileN << " <img src=\"DIFreconoiseNegativeDirectionhistD1PhiSymmetryDepth5HE.png\" />" << std::endl;
    htmlFileN << "<br>" << std::endl;

    if (sub == 2)
      htmlFileN << "<h3> 5H: DIF vs phi , different eta, Depth6 </h3>" << std::endl;
    //     htmlFileN << "<h4> Channel legend: white - good, other colour - bad. </h4>"<< std::endl;
    if (sub == 2)
      htmlFileN << " <img src=\"DIFreconoiseNegativeDirectionhistD1PhiSymmetryDepth6HE.png\" />" << std::endl;
    htmlFileN << "<br>" << std::endl;

    if (sub == 2)
      htmlFileN << "<h3> 5I: DIF vs phi , different eta, Depth7 </h3>" << std::endl;
    //     htmlFileN << "<h4> Channel legend: white - good, other colour - bad. </h4>"<< std::endl;
    if (sub == 2)
      htmlFileN << " <img src=\"DIFreconoiseNegativeDirectionhistD1PhiSymmetryDepth7HE.png\" />" << std::endl;
    htmlFileN << "<br>" << std::endl;

    /////////////////////////////////////////// end of Reconoise

    //
    //
    htmlFileN.close();

    /////////////////////////////////////////// end of Reconoise
    /////////////////////////////////////////// end of Reconoise

    //
    //
  }  // end sub  //for (int sub=1;sub<=4;sub++) {  //Subdetector: 1-HB, 2-HE, 3-HF, 4-HO

  //======================================================================

  std::cout << "********" << std::endl;
  std::cout << "************    Start creating subdet  html pages: - rather long time needed, waiting please"
            << std::endl;
  //======================================================================
  // Creating subdet  html pages:

  for (int sub = 1; sub <= 4; sub++) {  //Subdetector: 1-HB, 2-HE, 3-HF, 4-HO
    ofstream htmlFile;
    if (sub == 1)
      htmlFile.open("HB.html");
    if (sub == 2)
      htmlFile.open("HE.html");
    if (sub == 3)
      htmlFile.open("HO.html");
    if (sub == 4)
      htmlFile.open("HF.html");

    htmlFile << "</html><html xmlns=\"http://www.w3.org/1999/xhtml\">" << std::endl;
    htmlFile << "<head>" << std::endl;
    htmlFile << "<meta http-equiv=\"Content-Type\" content=\"text/html\"/>" << std::endl;
    htmlFile << "<title> Remote Monitoring Tool </title>" << std::endl;
    htmlFile << "<style type=\"text/css\">" << std::endl;
    htmlFile << " body,td{ background-color: #FFFFCC; font-family: arial, arial ce, helvetica; font-size: 12px; }"
             << std::endl;
    htmlFile << "   td.s0 { font-family: arial, arial ce, helvetica; }" << std::endl;
    htmlFile << "   td.s1 { font-family: arial, arial ce, helvetica; font-weight: bold; background-color: #FFC169; "
                "text-align: center;}"
             << std::endl;
    htmlFile << "   td.s2 { font-family: arial, arial ce, helvetica; background-color: #eeeeee; }" << std::endl;
    htmlFile << "   td.s3 { font-family: arial, arial ce, helvetica; background-color: #d0d0d0; }" << std::endl;
    htmlFile << "   td.s4 { font-family: arial, arial ce, helvetica; background-color: #FFC169; }" << std::endl;
    htmlFile << "   td.s5 { font-family: arial, arial ce, helvetica; background-color: #FF00FF; }" << std::endl;
    htmlFile << "   td.s6 { font-family: arial, arial ce, helvetica; background-color: #9ACD32; }" << std::endl;
    htmlFile << "   td.s7 { font-family: arial, arial ce, helvetica; background-color: #32CD32; }" << std::endl;
    htmlFile << "   td.s8 { font-family: arial, arial ce, helvetica; background-color: #00FFFF; }" << std::endl;
    htmlFile << "   td.s9 { font-family: arial, arial ce, helvetica; background-color: #FFE4E1; }" << std::endl;
    htmlFile << "   td.s10 { font-family: arial, arial ce, helvetica; background-color: #A0522D; }" << std::endl;
    htmlFile << "   td.s11 { font-family: arial, arial ce, helvetica; background-color: #1E90FF; }" << std::endl;
    htmlFile << "   td.s12 { font-family: arial, arial ce, helvetica; background-color: #00BFFF; }" << std::endl;
    htmlFile << "   td.s13 { font-family: arial, arial ce, helvetica; background-color: #FFFF00; }" << std::endl;
    htmlFile << "   td.s14 { font-family: arial, arial ce, helvetica; background-color: #B8860B; }" << std::endl;
    htmlFile << "</style>" << std::endl;
    htmlFile << "<body>" << std::endl;
    if (sub == 1)
      htmlFile << "<h1> HCAL BARREL, RUN = " << runnumber << " </h1>" << std::endl;
    if (sub == 2)
      htmlFile << "<h1> HCAL ENDCAP, RUN = " << runnumber << " </h1>" << std::endl;
    if (sub == 3)
      htmlFile << "<h1> HCAL OUTER, RUN = " << runnumber << " </h1>" << std::endl;
    if (sub == 4)
      htmlFile << "<h1> HCAL FORWARD, RUN = " << runnumber << " </h1>" << std::endl;
    htmlFile << "<br>" << std::endl;
    if (sub == 1)
      htmlFile << "<h2> 1. Analysis results for HB</h2>" << std::endl;
    if (sub == 2)
      htmlFile << "<h2> 1. Analysis results for HE</h2>" << std::endl;
    if (sub == 3)
      htmlFile << "<h2> 1. Analysis results for HO</h2>" << std::endl;
    if (sub == 4)
      htmlFile << "<h2> 1. Analysis results for HF</h2>" << std::endl;
    htmlFile << "<table width=\"600\">" << std::endl;
    htmlFile << "<tr>" << std::endl;

    if (sub == 1) {
      htmlFile << "  <td><a href=\"HB_PhiSymmetryRecoSignal.html\">Phi-symmetryRecoSignal</a></td>" << std::endl;
      htmlFile << "  <td><a href=\"HB_PhiSymmetryRecoNoise.html\">Phi-symmetryRecoNoise</a></td>" << std::endl;

      /*
       htmlFile << "  <td><a href=\"https://cms-conddb.cern.ch/eosweb/hcal/HcalRemoteMonitoring/IMPSM/GLOBAL_"<<runnumber<<"/HB_PhiSymmetryRecoSignal.html\">Phi-SymmetryRecoSignal</a></td>"<< std::endl;
       htmlFile << "  <td><a href=\"https://cms-conddb.cern.ch/eosweb/hcal/HcalRemoteMonitoring/IMPSM/GLOBAL_"<<runnumber<<"/HB_PhiSymmetryRecoNoise.html\">Phi-SymmetryRecoNoise</a></td>"<< std::endl;
*/
    }
    if (sub == 2) {
      htmlFile << "  <td><a href=\"HE_PhiSymmetryRecoSignal.html\">Phi-symmetryRecoSignal</a></td>" << std::endl;
      htmlFile << "  <td><a href=\"HE_PhiSymmetryRecoNoise.html\">Phi-symmetryRecoNoise</a></td>" << std::endl;

      /*
       htmlFile << "  <td><a href=\"https://cms-conddb.cern.ch/eosweb/hcal/HcalRemoteMonitoring/IMPSM/GLOBAL_"<<runnumber<<"/HE_PhiSymmetryRecoSignal.html\">Phi-symmetryRecoSignal</a></td>"<< std::endl;
       htmlFile << "  <td><a href=\"https://cms-conddb.cern.ch/eosweb/hcal/HcalRemoteMonitoring/IMPSM/GLOBAL_"<<runnumber<<"/HE_PhiSymmetryRecoNoise.html\">Phi-symmetryRecoNoise</a></td>"<< std::endl;
*/
    }

    if (sub == 4) {
      htmlFile << "  <td><a href=\"HF_PhiSymmetryRecoSignal.html\">Phi-symmetryRecoSignal</a></td>" << std::endl;
      htmlFile << "  <td><a href=\"HF_PhiSymmetryRecoNoise.html\">Phi-symmetryRecoNoise</a></td>" << std::endl;

      /*
       htmlFile << "  <td><a href=\"https://cms-conddb.cern.ch/eosweb/hcal/HcalRemoteMonitoring/IMPSM/GLOBAL_"<<runnumber<<"/HF_PhiSymmetryRecoSignal.html\">Phi-symmetryRecoSignal</a></td>"<< std::endl;
       htmlFile << "  <td><a href=\"https://cms-conddb.cern.ch/eosweb/hcal/HcalRemoteMonitoring/IMPSM/GLOBAL_"<<runnumber<<"/HF_PhiSymmetryRecoNoise.html\">Phi-symmetryRecoNoise</a></td>"<< std::endl;
*/
    }

    htmlFile << "</tr>" << std::endl;
    htmlFile << "</table>" << std::endl;
    htmlFile << "<br>" << std::endl;

    htmlFile << "</body> " << std::endl;
    htmlFile << "</html> " << std::endl;
    htmlFile.close();
  }

  //======================================================================

  std::cout << "********" << std::endl;
  std::cout << "************    Start creating description HELP html file:" << std::endl;
  //======================================================================
  // Creating description html file:
  ofstream htmlFile;
  htmlFile.open("HELP.html");
  htmlFile << "</html><html xmlns=\"http://www.w3.org/1999/xhtml\">" << std::endl;
  htmlFile << "<head>" << std::endl;
  htmlFile << "<meta http-equiv=\"Content-Type\" content=\"text/html\"/>" << std::endl;
  htmlFile << "<title> Remote Monitoring Tool </title>" << std::endl;
  htmlFile << "<style type=\"text/css\">" << std::endl;
  htmlFile << " body,td{ background-color: #FFFFCC; font-family: arial, arial ce, helvetica; font-size: 12px; }"
           << std::endl;
  htmlFile << "   td.s0 { font-family: arial, arial ce, helvetica; }" << std::endl;
  htmlFile << "   td.s1 { font-family: arial, arial ce, helvetica; font-weight: bold; background-color: #FFC169; "
              "text-align: center;}"
           << std::endl;
  htmlFile << "   td.s2 { font-family: arial, arial ce, helvetica; background-color: #eeeeee; }" << std::endl;
  htmlFile << "   td.s3 { font-family: arial, arial ce, helvetica; background-color: #d0d0d0; }" << std::endl;
  htmlFile << "   td.s4 { font-family: arial, arial ce, helvetica; background-color: #FFC169; }" << std::endl;
  htmlFile << "</style>" << std::endl;
  htmlFile << "<body>" << std::endl;
  htmlFile << "<h1>  Description of Remote Monitoring Tool criteria for bad channel selection</h1>" << std::endl;
  htmlFile << "<br>" << std::endl;
  htmlFile << "<h3> - C means CAPID Errors assuming we inspect CAPID non-rotation,error & validation bits, and for "
              "this criterion - no need to apply any cuts to select bcs.</h3> "
           << std::endl;
  htmlFile << "<br>" << std::endl;
  htmlFile << "<h3> - A means full amplitude, collected over all time slices </h3> " << std::endl;
  htmlFile << "<h3> - R means ratio criterion where we define as a bad, the channels, for which the signal portion in "
              "4 middle TSs(plus one, minus two around TS with maximal amplitude) is out of some range of reasonable "
              "values </h3> "
           << std::endl;
  htmlFile << "<br>" << std::endl;
  htmlFile << "<h3> - W means width of shape distribution. Width is defined as square root from dispersion. </h3> "
           << std::endl;
  htmlFile << "<br>" << std::endl;
  htmlFile << "<h3> - TN means mean time position of adc signal. </h3> " << std::endl;
  htmlFile << "<br>" << std::endl;
  htmlFile << "<h3> - TX means TS number of maximum signal </h3> " << std::endl;
  htmlFile << "<br>" << std::endl;
  htmlFile << "<h3> - m means megatile channels. For example Am means Amplitude criteria for megatile channels </h3> "
           << std::endl;
  htmlFile << "<br>" << std::endl;
  htmlFile
      << "<h3> - c means calibration channels. For example Ac means Amplitude criteria for calibration channels </h3> "
      << std::endl;
  htmlFile << "<br>" << std::endl;
  htmlFile << "<h3> - Pm means Pedestals. </h3> " << std::endl;
  htmlFile << "<br>" << std::endl;
  htmlFile << "<h3> - pWm  means pedestal Width. </h3> " << std::endl;
  htmlFile << "<br>" << std::endl;
  htmlFile << "</body> " << std::endl;
  htmlFile << "</html> " << std::endl;
  htmlFile.close();

  //======================================================================

  std::cout << "********" << std::endl;
  std::cout << "************    Start creating MAP html file: - rather long time needed, waiting please" << std::endl;
  //======================================================================
  // Creating main html file:
  htmlFile.open("MAP.html");
  htmlFile << "</html><html xmlns=\"http://www.w3.org/1999/xhtml\">" << std::endl;
  htmlFile << "<head>" << std::endl;
  htmlFile << "<meta http-equiv=\"Content-Type\" content=\"text/html\"/>" << std::endl;
  htmlFile << "<title> Remote Monitoring Tool </title>" << std::endl;
  htmlFile << "<style type=\"text/css\">" << std::endl;
  htmlFile << " body,td{ background-color: #FFFFCC; font-family: arial, arial ce, helvetica; font-size: 12px; }"
           << std::endl;
  htmlFile << "   td.s0 { font-family: arial, arial ce, helvetica; }" << std::endl;
  htmlFile << "   td.s1 { font-family: arial, arial ce, helvetica; font-weight: bold; background-color: #FFC169; "
              "text-align: center;}"
           << std::endl;
  htmlFile << "   td.s2 { font-family: arial, arial ce, helvetica; background-color: #eeeeee; }" << std::endl;
  htmlFile << "   td.s3 { font-family: arial, arial ce, helvetica; background-color: #d0d0d0; }" << std::endl;
  htmlFile << "   td.s4 { font-family: arial, arial ce, helvetica; background-color: #FFC169; }" << std::endl;
  htmlFile << "   td.s5 { font-family: arial, arial ce, helvetica; background-color: #FF00FF; }" << std::endl;
  htmlFile << "   td.s6 { font-family: arial, arial ce, helvetica; background-color: #9ACD32; }" << std::endl;
  htmlFile << "   td.s7 { font-family: arial, arial ce, helvetica; background-color: #32CD32; }" << std::endl;
  htmlFile << "</style>" << std::endl;
  htmlFile << "<body>" << std::endl;

  htmlFile << "<h1> Remote Monitoring Tool, RUN = " << runnumber << ". </h1>" << std::endl;
  htmlFile << "<br>" << std::endl;

  htmlFile << "<h2> 1. Analysis results for subdetectors </h2>" << std::endl;
  htmlFile << "<table width=\"400\">" << std::endl;
  htmlFile << "<tr>" << std::endl;

  /*
     htmlFile << "  <td><a href=\"HB.html\">HB</a></td>"<< std::endl;
     htmlFile << "  <td><a href=\"HE.html\">HE</a></td>"<< std::endl;
     htmlFile << "  <td><a href=\"HO.html\">HO</a></td>"<< std::endl;
     htmlFile << "  <td><a href=\"HF.html\">HF</a></td>"<< std::endl;    
*/
  htmlFile << "  <td><a href=\"https://cms-conddb.cern.ch/eosweb/hcal/HcalRemoteMonitoring/IMPSM/GLOBAL_" << runnumber
           << "/HB.html\">HB</a></td>" << std::endl;
  htmlFile << "  <td><a href=\"https://cms-conddb.cern.ch/eosweb/hcal/HcalRemoteMonitoring/IMPSM/GLOBAL_" << runnumber
           << "/HE.html\">HE</a></td>" << std::endl;
  htmlFile << "  <td><a href=\"https://cms-conddb.cern.ch/eosweb/hcal/HcalRemoteMonitoring/IMPSM/GLOBAL_" << runnumber
           << "/HO.html\">HO</a></td>" << std::endl;
  htmlFile << "  <td><a href=\"https://cms-conddb.cern.ch/eosweb/hcal/HcalRemoteMonitoring/IMPSM/GLOBAL_" << runnumber
           << "/HF.html\">HF</a></td>" << std::endl;

  htmlFile << "</tr>" << std::endl;
  htmlFile << "</table>" << std::endl;
  htmlFile << "<br>" << std::endl;

  htmlFile << "</body> " << std::endl;
  htmlFile << "</html> " << std::endl;
  htmlFile.close();
  //======================================================================

  //======================================================================
  // Close and delete all possible things:
  hfile->Close();
  //  hfile->Delete();
  //  Exit Root
  gSystem->Exit(0);
  //======================================================================
}
