#include <iostream>
#include "TH1D.h"
#include "TH2D.h"
#include "TLegend.h"
#include "TStyle.h"
#include "TCanvas.h"
#include "Riostream.h"
#include <string>
#include "TFile.h"
#include "TTree.h"
#include <math.h>
#include <TFile.h>
#include <sstream>
#include "TPaveStats.h"

void plotHistlog(TH1D* hist1,TH1D* hist2,TH1D* hist3,TH1D* hist4,std::string name);
void plotHist(TH1D* hist1,TH1D* hist2,TH1D* hist3,TH1D* hist4,std::string name);
void plotHistlog(TH1D* hist1,TH1D* hist2,std::string name);
void plotHist(TH1D* hist1,TH1D* hist2,std::string name);

void finalPlots(){
  //gROOT->Reset();
  gStyle->SetOptStat("e");
  gStyle->SetCanvasBorderMode(0);
  gStyle->SetPadBorderMode(0);

  gStyle->SetCanvasColor(10);
  gStyle->SetPadColor(10);
  gStyle->SetFillColor(10);
  gStyle->SetStatColor(10);
  gStyle->SetTitleFillColor(10);
  
//*****  getting real DATA   *****

  TFile *fileDATA = TFile::Open("histosDATA.root");

std::string addMe = "DATA_";

TH1D* eta_All             = (TH1D*)gDirectory->Get( (addMe + std::string("eta_All")).c_str() );
TH1D* eta_withVertex      = (TH1D*)gDirectory->Get( (addMe + std::string("eta_withVertex")).c_str() );
TH1D* eta_AssVertex       = (TH1D*)gDirectory->Get( (addMe + std::string("eta_AssVertex")).c_str() );
  TH1D* phi_All             = (TH1D*)gDirectory->Get( (addMe + std::string("phi_All")).c_str() );
  TH1D* phi_withVertex      = (TH1D*)gDirectory->Get( (addMe + std::string("phi_withVertex")).c_str() );
  TH1D* phi_AssVertex       = (TH1D*)gDirectory->Get( (addMe + std::string("phi_AssVertex")).c_str() );
TH1D* pt_All              = (TH1D*)gDirectory->Get( (addMe + std::string("pt_All")).c_str() );
TH1D* pt_withVertex       = (TH1D*)gDirectory->Get( (addMe + std::string("pt_withVertex")).c_str() );
TH1D* pt_AssVertex        = (TH1D*)gDirectory->Get( (addMe + std::string("pt_AssVertex")).c_str() ); 
  TH1D* track_multip_All       = (TH1D*)gDirectory->Get( (addMe + std::string("track_multip_All")).c_str() );
  TH1D* track_multip_withVertex= (TH1D*)gDirectory->Get( (addMe + std::string("track_multip_withVertex")).c_str() );
  TH1D* track_multip_AssVertex = (TH1D*)gDirectory->Get( (addMe + std::string("track_multip_AssVertex")).c_str() );
TH2D* pt_vs_multip_All              = (TH2D*)gDirectory->Get( (addMe + std::string("pt_vs_multip_All")).c_str() );
TH2D* pt_vs_multip_withVertex       = (TH2D*)gDirectory->Get( (addMe + std::string("pt_vs_multip_withVertex")).c_str() );
TH2D* pt_vs_multip_AssVertex        = (TH2D*)gDirectory->Get( (addMe + std::string("pt_vs_multip_AssVertex")).c_str() );
  TH2D* pt_vs_ndof_All              = (TH2D*)gDirectory->Get( (addMe + std::string("pt_vs_ndof_All")).c_str() );
  TH2D* pt_vs_ndof_withVertex       = (TH2D*)gDirectory->Get( (addMe + std::string("pt_vs_ndof_withVertex")).c_str() );
  TH2D* pt_vs_ndof_AssVertex        = (TH2D*)gDirectory->Get( (addMe + std::string("pt_vs_ndof_AssVertex")).c_str() );
TH1D* ndof_All              = (TH1D*)gDirectory->Get( (addMe + std::string("ndof_All")).c_str() );
TH1D* ndof_withVertex       = (TH1D*)gDirectory->Get( (addMe + std::string("ndof_withVertex")).c_str() );
TH1D* ndof_AssVertex        = (TH1D*)gDirectory->Get( (addMe + std::string("ndof_AssVertex")).c_str() );
  TH1D* normChi2_All        = (TH1D*)gDirectory->Get( (addMe + std::string("normChi2_All")).c_str() );
  TH1D* normChi2_withVertex = (TH1D*)gDirectory->Get( (addMe + std::string("normChi2_withVertex")).c_str() );
  TH1D* normChi2_AssVertex  = (TH1D*)gDirectory->Get( (addMe + std::string("normChi2_AssVertex")).c_str() );
TH2D* chi2_vs_pT_All       = (TH2D*)gDirectory->Get( (addMe + std::string("chi2_vs_pT_All")).c_str() );
TH2D* chi2_vs_pT_withVertex= (TH2D*)gDirectory->Get( (addMe + std::string("chi2_vs_pT_withVertex")).c_str() );
TH2D* chi2_vs_pT_AssVertex = (TH2D*)gDirectory->Get( (addMe + std::string("chi2_vs_pT_AssVertex")).c_str() );
  TH2D* chi2_vs_pT_lowMultip_All       = (TH2D*)gDirectory->Get( (addMe + std::string("chi2_vs_pT_lowMultip_All")).c_str() );
  TH2D* chi2_vs_pT_lowMultip_withVertex= (TH2D*)gDirectory->Get( (addMe + std::string("chi2_vs_pT_lowMultip_withVertex")).c_str() );
  TH2D* chi2_vs_pT_lowMultip_AssVertex = (TH2D*)gDirectory->Get( (addMe + std::string("chi2_vs_pT_lowMultip_AssVertex")).c_str() );
TH2D* chi2_vs_pT_highMultip_All       = (TH2D*)gDirectory->Get( (addMe + std::string("chi2_vs_pT_highMultip_All")).c_str() );
TH2D* chi2_vs_pT_highMultip_withVertex= (TH2D*)gDirectory->Get( (addMe + std::string("chi2_vs_pT_highMultip_withVertex")).c_str() );
TH2D* chi2_vs_pT_highMultip_AssVertex = (TH2D*)gDirectory->Get( (addMe + std::string("chi2_vs_pT_highMultip_AssVertex")).c_str() );
  TH1D* vertexes            = (TH1D*)gDirectory->Get( (addMe + std::string("vertexes")).c_str() );
  TH1D* vertexes_z          = (TH1D*)gDirectory->Get( (addMe + std::string("vertexes_z")).c_str() );
  TH2D* vertexes_xy         = (TH2D*)gDirectory->Get( (addMe + std::string("vertexes_xy")).c_str() );
TH1D* deltaZ_trackPV      = (TH1D*)gDirectory->Get( (addMe + std::string("deltaZ_trackPV")).c_str() );
TH1D* deltaZ_trackPV_ZOOM      = (TH1D*)gDirectory->Get( (addMe + std::string("deltaZ_trackPV_ZOOM")).c_str() );
  TH1D* deltaX_trackPV      = (TH1D*)gDirectory->Get( (addMe + std::string("deltaX_trackPV")).c_str() );
  TH1D* deltaX_trackPV_ZOOM      = (TH1D*)gDirectory->Get( (addMe + std::string("deltaX_trackPV_ZOOM")).c_str() );
  TH1D* deltaY_trackPV      = (TH1D*)gDirectory->Get( (addMe + std::string("deltaY_trackPV")).c_str() );
  TH1D* deltaY_trackPV_ZOOM      = (TH1D*)gDirectory->Get( (addMe + std::string("deltaY_trackPV_ZOOM")).c_str() );
TH1D* vertexesNum_NoutTrk = (TH1D*)gDirectory->Get( (addMe + std::string("vertexesNum_NoutTrk")).c_str() );
TH1D* deltaZ_v1v2         = (TH1D*)gDirectory->Get( (addMe + std::string("deltaZ_v1v2")).c_str() );
  TH1D* dZ_All      = (TH1D*)gDirectory->Get( (addMe + std::string("dZ_All")).c_str() );
  TH1D* d0_All      = (TH1D*)gDirectory->Get( (addMe + std::string("d0_All")).c_str() );
TH1D* dZPoint_Ass      = (TH1D*)gDirectory->Get( (addMe + std::string("dZPoint_Ass")).c_str() );
TH1D* d0Point_Ass      = (TH1D*)gDirectory->Get( (addMe + std::string("d0Point_Ass")).c_str() );
TH1D* dZ_Ass      = (TH1D*)gDirectory->Get( (addMe + std::string("dZ_Ass")).c_str() );
TH1D* d0_Ass      = (TH1D*)gDirectory->Get( (addMe + std::string("d0_Ass")).c_str() );
  TH1D* dZPoint_Vtx      = (TH1D*)gDirectory->Get( (addMe + std::string("dZPoint_Vtx")).c_str() );
  TH1D* d0Point_Vtx      = (TH1D*)gDirectory->Get( (addMe + std::string("d0Point_Vtx")).c_str() );
  TH1D* dZ_Vtx      = (TH1D*)gDirectory->Get( (addMe + std::string("dZ_Vtx")).c_str() );
  TH1D* d0_Vtx      = (TH1D*)gDirectory->Get( (addMe + std::string("d0_Vtx")).c_str() );
TH1D* dZ_Vtx_LARGE      = (TH1D*)gDirectory->Get( (addMe + std::string("dZ_Vtx_LARGE")).c_str() );
  TH1D* hthrust_z_All            = (TH1D*)gDirectory->Get( (addMe + std::string("hthrust_z_All")).c_str() );
  TH1D* hthrust_z_withVertex     = (TH1D*)gDirectory->Get( (addMe + std::string("hthrust_z_withVertex")).c_str() );
  TH1D* hthrust_z_AssVertex      = (TH1D*)gDirectory->Get( (addMe + std::string("hthrust_z_AssVertex")).c_str() );
TH1D* chargeAsymmetry_All =  (TH1D*)gDirectory->Get( (addMe + std::string("chargeAsymmetry_All")).c_str() );
TH1D* chargeAsymmetry_AssVertex =  (TH1D*)gDirectory->Get( (addMe + std::string("chargeAsymmetry_AssVertex")).c_str() );

//*****  getting MC   *****

  TFile *fileMC = TFile::Open("../histoMC_fromChain_addpTcutAndGenLevelObservations.root");

 addMe = "MC_";

TH1D* MC_eta_All             = (TH1D*)gDirectory->Get( (addMe + std::string("eta_All")).c_str() );
TH1D* MC_eta_withVertex      = (TH1D*)gDirectory->Get( (addMe + std::string("eta_withVertex")).c_str() );
TH1D* MC_eta_AssVertex       = (TH1D*)gDirectory->Get( (addMe + std::string("eta_AssVertex")).c_str() );
  TH1D* MC_phi_All             = (TH1D*)gDirectory->Get( (addMe + std::string("phi_All")).c_str() );
  TH1D* MC_phi_withVertex      = (TH1D*)gDirectory->Get( (addMe + std::string("phi_withVertex")).c_str() );
  TH1D* MC_phi_AssVertex       = (TH1D*)gDirectory->Get( (addMe + std::string("phi_AssVertex")).c_str() );
TH1D* MC_pt_All              = (TH1D*)gDirectory->Get( (addMe + std::string("pt_All")).c_str() );
TH1D* MC_pt_withVertex       = (TH1D*)gDirectory->Get( (addMe + std::string("pt_withVertex")).c_str() );
TH1D* MC_pt_AssVertex        = (TH1D*)gDirectory->Get( (addMe + std::string("pt_AssVertex")).c_str() );
  TH1D* MC_track_multip_All       = (TH1D*)gDirectory->Get( (addMe + std::string("track_multip_All")).c_str() );
  TH1D* MC_track_multip_withVertex= (TH1D*)gDirectory->Get( (addMe + std::string("track_multip_withVertex")).c_str() );
  TH1D* MC_track_multip_AssVertex = (TH1D*)gDirectory->Get( (addMe + std::string("track_multip_AssVertex")).c_str() );
TH2D* MC_pt_vs_multip_All              = (TH2D*)gDirectory->Get( (addMe + std::string("pt_vs_multip_All")).c_str() );
TH2D* MC_pt_vs_multip_withVertex       = (TH2D*)gDirectory->Get( (addMe + std::string("pt_vs_multip_withVertex")).c_str() );
TH2D* MC_pt_vs_multip_AssVertex        = (TH2D*)gDirectory->Get( (addMe + std::string("pt_vs_multip_AssVertex")).c_str() );
  TH2D* MC_pt_vs_ndof_All              = (TH2D*)gDirectory->Get( (addMe + std::string("pt_vs_ndof_All")).c_str() );
  TH2D* MC_pt_vs_ndof_withVertex       = (TH2D*)gDirectory->Get( (addMe + std::string("pt_vs_ndof_withVertex")).c_str() );
  TH2D* MC_pt_vs_ndof_AssVertex        = (TH2D*)gDirectory->Get( (addMe + std::string("pt_vs_ndof_AssVertex")).c_str() );
TH1D* MC_ndof_All              = (TH1D*)gDirectory->Get( (addMe + std::string("ndof_All")).c_str() );
TH1D* MC_ndof_withVertex       = (TH1D*)gDirectory->Get( (addMe + std::string("ndof_withVertex")).c_str() );
TH1D* MC_ndof_AssVertex        = (TH1D*)gDirectory->Get( (addMe + std::string("ndof_AssVertex")).c_str() );
  TH1D* MC_normChi2_All        = (TH1D*)gDirectory->Get( (addMe + std::string("normChi2_All")).c_str() );
  TH1D* MC_normChi2_withVertex = (TH1D*)gDirectory->Get( (addMe + std::string("normChi2_withVertex")).c_str() );
  TH1D* MC_normChi2_AssVertex  = (TH1D*)gDirectory->Get( (addMe + std::string("normChi2_AssVertex")).c_str() );
TH2D* MC_chi2_vs_pT_All       = (TH2D*)gDirectory->Get( (addMe + std::string("chi2_vs_pT_All")).c_str() );
TH2D* MC_chi2_vs_pT_withVertex= (TH2D*)gDirectory->Get( (addMe + std::string("chi2_vs_pT_withVertex")).c_str() );
TH2D* MC_chi2_vs_pT_AssVertex = (TH2D*)gDirectory->Get( (addMe + std::string("chi2_vs_pT_AssVertex")).c_str() );
  TH2D* MC_chi2_vs_pT_lowMultip_All       = (TH2D*)gDirectory->Get( (addMe + std::string("chi2_vs_pT_lowMultip_All")).c_str() );
  TH2D* MC_chi2_vs_pT_lowMultip_withVertex= (TH2D*)gDirectory->Get( (addMe + std::string("chi2_vs_pT_lowMultip_withVertex")).c_str() );
  TH2D* MC_chi2_vs_pT_lowMultip_AssVertex = (TH2D*)gDirectory->Get( (addMe + std::string("chi2_vs_pT_lowMultip_AssVertex")).c_str() );
TH2D* MC_chi2_vs_pT_highMultip_All       = (TH2D*)gDirectory->Get( (addMe + std::string("chi2_vs_pT_highMultip_All")).c_str() );
TH2D* MC_chi2_vs_pT_highMultip_withVertex= (TH2D*)gDirectory->Get( (addMe + std::string("chi2_vs_pT_highMultip_withVertex")).c_str() );
TH2D* MC_chi2_vs_pT_highMultip_AssVertex = (TH2D*)gDirectory->Get( (addMe + std::string("chi2_vs_pT_highMultip_AssVertex")).c_str() );
  TH1D* MC_vertexes            = (TH1D*)gDirectory->Get( (addMe + std::string("vertexes")).c_str() );
  TH1D* MC_vertexes_z          = (TH1D*)gDirectory->Get( (addMe + std::string("vertexes_z")).c_str() );
  TH2D* MC_vertexes_xy         = (TH2D*)gDirectory->Get( (addMe + std::string("vertexes_xy")).c_str() );
TH1D* MC_deltaZ_trackPV      = (TH1D*)gDirectory->Get( (addMe + std::string("deltaZ_trackPV")).c_str() );
TH1D* MC_deltaZ_trackPV_ZOOM      = (TH1D*)gDirectory->Get( (addMe + std::string("deltaZ_trackPV_ZOOM")).c_str() );
TH1D* MC_deltaX_trackPV      = (TH1D*)gDirectory->Get( (addMe + std::string("deltaX_trackPV")).c_str() );
TH1D* MC_deltaX_trackPV_ZOOM      = (TH1D*)gDirectory->Get( (addMe + std::string("deltaX_trackPV_ZOOM")).c_str() );
  TH1D* MC_deltaY_trackPV      = (TH1D*)gDirectory->Get( (addMe + std::string("deltaY_trackPV")).c_str() );
  TH1D* MC_deltaY_trackPV_ZOOM      = (TH1D*)gDirectory->Get( (addMe + std::string("deltaY_trackPV_ZOOM")).c_str() );
TH1D* MC_vertexesNum_NoutTrk = (TH1D*)gDirectory->Get( (addMe + std::string("vertexesNum_NoutTrk")).c_str() );
TH1D* MC_deltaZ_v1v2         = (TH1D*)gDirectory->Get( (addMe + std::string("deltaZ_v1v2")).c_str() );
  TH1D* MC_dZ_All      = (TH1D*)gDirectory->Get( (addMe + std::string("dZ_All")).c_str() );
  TH1D* MC_d0_All      = (TH1D*)gDirectory->Get( (addMe + std::string("d0_All")).c_str() );
TH1D* MC_dZPoint_Ass      = (TH1D*)gDirectory->Get( (addMe + std::string("dZPoint_Ass")).c_str() );
TH1D* MC_d0Point_Ass      = (TH1D*)gDirectory->Get( (addMe + std::string("d0Point_Ass")).c_str() );
TH1D* MC_dZ_Ass      = (TH1D*)gDirectory->Get( (addMe + std::string("dZ_Ass")).c_str() );
TH1D* MC_d0_Ass      = (TH1D*)gDirectory->Get( (addMe + std::string("d0_Ass")).c_str() );
  TH1D* MC_dZPoint_Vtx      = (TH1D*)gDirectory->Get( (addMe + std::string("dZPoint_Vtx")).c_str() );
  TH1D* MC_d0Point_Vtx      = (TH1D*)gDirectory->Get( (addMe + std::string("d0Point_Vtx")).c_str() );
  TH1D* MC_dZ_Vtx      = (TH1D*)gDirectory->Get( (addMe + std::string("dZ_Vtx")).c_str() );
  TH1D* MC_d0_Vtx      = (TH1D*)gDirectory->Get( (addMe + std::string("d0_Vtx")).c_str() );
TH1D* MC_dZ_Vtx_LARGE      = (TH1D*)gDirectory->Get( (addMe + std::string("dZ_Vtx_LARGE")).c_str() );
  TH1D* MC_hthrust_z_All            = (TH1D*)gDirectory->Get( (addMe + std::string("hthrust_z_All")).c_str() );
  TH1D* MC_hthrust_z_withVertex     = (TH1D*)gDirectory->Get( (addMe + std::string("hthrust_z_withVertex")).c_str() );
  TH1D* MC_hthrust_z_AssVertex      = (TH1D*)gDirectory->Get( (addMe + std::string("hthrust_z_AssVertex")).c_str() );
TH1D* MC_chargeAsymmetry_All =  (TH1D*)gDirectory->Get( (addMe + std::string("chargeAsymmetry_All")).c_str() );
TH1D* MC_chargeAsymmetry_AssVertex =  (TH1D*)gDirectory->Get( (addMe + std::string("chargeAsymmetry_AssVertex")).c_str() );

//****************************

//**** histos NORMALIZATIONS ******

   eta_All->Scale(1./eta_All->Integral());
   phi_All->Scale(1./phi_All->Integral());
   pt_All->Scale(1./pt_All->Integral());
   track_multip_All->Scale(1./track_multip_All->Integral());
   ndof_All->Scale(1./ndof_All->Integral());
   chargeAsymmetry_All->Scale(1./chargeAsymmetry_All->Integral());
   normChi2_All->Scale(1./normChi2_All->Integral());

 eta_All->SetMarkerColor(kGray);
 phi_All->SetMarkerColor(kGray);
 pt_All->SetMarkerColor(kGray);
 track_multip_All->SetMarkerColor(kGray);
 ndof_All->SetMarkerColor(kGray);
 chargeAsymmetry_All->SetMarkerColor(kGray);
 normChi2_All->SetMarkerColor(kGray);
   eta_AssVertex->Scale(1./eta_AssVertex->Integral());
   phi_AssVertex->Scale(1./phi_AssVertex->Integral());
   pt_AssVertex->Scale(1./pt_AssVertex->Integral());
   track_multip_AssVertex->Scale(1./track_multip_AssVertex->Integral());
   ndof_AssVertex->Scale(1./ndof_AssVertex->Integral());
   chargeAsymmetry_AssVertex->Scale(1./chargeAsymmetry_AssVertex->Integral());
   normChi2_AssVertex->Scale(1./normChi2_AssVertex->Integral());
 vertexes->Scale(1./vertexes->Integral());
 
 MC_eta_All->Scale(1./MC_eta_All->Integral());
 MC_eta_AssVertex->Scale(1./MC_eta_AssVertex->Integral());
 MC_phi_All->Scale(1./MC_phi_All->Integral());
 MC_phi_AssVertex->Scale(1./MC_phi_AssVertex->Integral());
 MC_pt_All->Scale(1./MC_pt_All->Integral());
 MC_pt_AssVertex->Scale(1./MC_pt_AssVertex->Integral());
 MC_track_multip_All->Scale(1./MC_track_multip_All->Integral());
 MC_track_multip_AssVertex->Scale(1./MC_track_multip_AssVertex->Integral());
 MC_ndof_All->Scale(1./MC_ndof_All->Integral());
 MC_ndof_AssVertex->Scale(1./MC_ndof_AssVertex->Integral());
 MC_vertexes->Scale(1./MC_vertexes->Integral());
 MC_normChi2_AssVertex->Scale(1./MC_normChi2_AssVertex->Integral());
 MC_normChi2_All->Scale(1./MC_normChi2_All->Integral());
 MC_chargeAsymmetry_All->Scale(1./MC_chargeAsymmetry_All->Integral());
 MC_chargeAsymmetry_AssVertex->Scale(1./MC_chargeAsymmetry_AssVertex->Integral());
   MC_eta_All->SetLineColor(kMagenta);
   MC_eta_AssVertex->SetLineColor(kRed);
   MC_phi_All->SetLineColor(kMagenta);
   MC_phi_AssVertex->SetLineColor(kRed);
   MC_pt_All->SetLineColor(kMagenta);
   MC_pt_AssVertex->SetLineColor(kRed);
   MC_track_multip_All->SetLineColor(kMagenta);
   MC_track_multip_AssVertex->SetLineColor(kRed);
   MC_ndof_All->SetLineColor(kMagenta);
   MC_ndof_AssVertex->SetLineColor(kRed);
   MC_vertexes->SetLineColor(kRed);
   MC_normChi2_AssVertex->SetLineColor(kRed);
   MC_normChi2_All->SetLineColor(kMagenta);
   MC_chargeAsymmetry_All->SetLineColor(kMagenta);
   MC_chargeAsymmetry_AssVertex->SetLineColor(kRed);
 

eta_AssVertex->GetXaxis()->SetTitle("#eta");
phi_AssVertex->GetXaxis()->SetTitle("#phi");  phi_AssVertex->SetMinimum(0.);
pt_AssVertex->GetXaxis()->SetTitle("p_{T}");  pt_AssVertex->GetXaxis()->SetRangeUser(0,15);
track_multip_AssVertex->GetXaxis()->SetTitle("tk/event");  track_multip_AssVertex->GetXaxis()->SetRangeUser(0,100);
ndof_AssVertex->GetXaxis()->SetTitle("ndof");
normChi2_AssVertex->GetXaxis()->SetTitle("#chi^{2}_{#nu}");
chargeAsymmetry_AssVertex->GetXaxis()->SetTitle("#Deltaq");
  eta_All->GetXaxis()->SetTitle("#eta");
  phi_All->GetXaxis()->SetTitle("#phi");  
  pt_All->GetXaxis()->SetTitle("p_{T}");
  track_multip_All->GetXaxis()->SetTitle("tk/event");
  ndof_All->GetXaxis()->SetTitle("ndof");
  normChi2_All->GetXaxis()->SetTitle("#chi^{2}_{#nu}");

plotHist(eta_AssVertex,MC_eta_AssVertex,"eta");
plotHist(phi_AssVertex,MC_phi_AssVertex,"phi");
plotHist(pt_AssVertex,MC_pt_AssVertex,"pT");
plotHistlog(pt_AssVertex,MC_pt_AssVertex,"pT_log");
plotHistlog(track_multip_AssVertex,MC_track_multip_AssVertex,"trackMultiplicity");
plotHist(ndof_AssVertex,MC_ndof_AssVertex,"ndof");
plotHist(normChi2_AssVertex,MC_normChi2_AssVertex,"chi2Nu");
plotHistlog(normChi2_AssVertex,MC_normChi2_AssVertex,"chi2Nu_log");
plotHistlog(chargeAsymmetry_AssVertex,MC_chargeAsymmetry_AssVertex,"deltaCharge_log");
  plotHist(eta_AssVertex,MC_eta_AssVertex,eta_All,MC_eta_All,"eta");
  plotHist(phi_AssVertex,MC_phi_AssVertex,phi_All,MC_phi_All,"phi");
  plotHist(pt_AssVertex,MC_pt_AssVertex,pt_All,MC_pt_All,"pT");
  plotHistlog(pt_AssVertex,MC_pt_AssVertex,pt_All,MC_pt_All,"pT_log");
  plotHistlog(track_multip_AssVertex,MC_track_multip_AssVertex,track_multip_All,MC_track_multip_All,"trackMultiplicity");
  plotHist(ndof_AssVertex,MC_ndof_AssVertex,ndof_All,MC_ndof_All,"ndof");
  plotHist(normChi2_AssVertex,MC_normChi2_AssVertex,normChi2_All,MC_normChi2_All,"chi2Nu");
  plotHistlog(normChi2_AssVertex,MC_normChi2_AssVertex,normChi2_All,MC_normChi2_All,"chi2Nu_log");
  plotHistlog(chargeAsymmetry_AssVertex,MC_chargeAsymmetry_AssVertex,chargeAsymmetry_All,MC_chargeAsymmetry_All,"deltaCharge_log");


}


void plotHist(TH1D* hist1,TH1D* hist2,TH1D* hist3,TH1D* hist4,std::string name){
  TCanvas c1; c1.cd();

    hist1->Draw("P"); 
  double startingX=0.7;
  double startingY=1;
  double Ystep=0.10;//0.15 if Optstat=1111
  gPad->Update();
    TPaveStats** st = new TPaveStats*[4];
    int ifile=0;
    gPad->Update();
    st[0] =  (TPaveStats*) hist1->GetListOfFunctions()->FindObject("stats");
    //st[0]->SetName(leg[ifile]);
    st[0]->SetX1NDC(startingX);
    st[0]->SetX2NDC(startingX+0.23);
    st[0]->SetY1NDC(startingY-Ystep*(double(ifile)+1.));
    st[0]->SetY2NDC(startingY-Ystep*double(ifile));
    st[0]->SetTextColor(hist1->GetMarkerColor());
  ifile++;
  hist2->Draw("sames");
  gPad->Update();
  st[1] =  (TPaveStats*) hist2->GetListOfFunctions()->FindObject("stats");
  //st[0]->SetName(leg[ifile]);
  st[1]->SetX1NDC(startingX);
  st[1]->SetX2NDC(startingX+0.23);
  st[1]->SetY1NDC(startingY-Ystep*(double(ifile)+1.));
  st[1]->SetY2NDC(startingY-Ystep*double(ifile));
  st[1]->SetTextColor(hist2->GetLineColor());
    ifile++;
    hist3->Draw("samesP"); 
    gPad->Update();
    st[2] =  (TPaveStats*) hist3->GetListOfFunctions()->FindObject("stats");
    //st[0]->SetName(leg[ifile]);
    st[2]->SetX1NDC(startingX);
    st[2]->SetX2NDC(startingX+0.23);
    st[2]->SetY1NDC(startingY-Ystep*(double(ifile)+1.));
    st[2]->SetY2NDC(startingY-Ystep*double(ifile));
    st[2]->SetTextColor(hist3->GetMarkerColor());
  ifile++;
  hist4->Draw("sames"); 
  gPad->Update();
  st[3] =  (TPaveStats*) hist4->GetListOfFunctions()->FindObject("stats");
  //st[0]->SetName(leg[ifile]);
  st[3]->SetX1NDC(startingX);
  st[3]->SetX2NDC(startingX+0.23);
  st[3]->SetY1NDC(startingY-Ystep*(double(ifile)+1.));
  st[3]->SetY2NDC(startingY-Ystep*double(ifile));
  st[3]->SetTextColor(hist4->GetLineColor());


  c1.Print( ( (name).c_str()+std::string("_4histos.gif") ).c_str(),"gif");  
}

void plotHistlog(TH1D* hist1,TH1D* hist2,TH1D* hist3,TH1D* hist4,std::string name){
  TCanvas c1; c1.cd();
  gPad->SetLogy();

    hist1->Draw("P"); 
  double startingX=0.7;
  double startingY=1;
  double Ystep=0.10;//0.15 if Optstat=1111
  gPad->Update();
    TPaveStats** st = new TPaveStats*[4];
    int ifile=0;
    gPad->Update();
    st[0] =  (TPaveStats*) hist1->GetListOfFunctions()->FindObject("stats");
    //st[0]->SetName(leg[ifile]);
    st[0]->SetX1NDC(startingX);
    st[0]->SetX2NDC(startingX+0.23);
    st[0]->SetY1NDC(startingY-Ystep*(double(ifile)+1.));
    st[0]->SetY2NDC(startingY-Ystep*double(ifile));
    st[0]->SetTextColor(hist1->GetMarkerColor());
  ifile++;
  hist2->Draw("sames");
  gPad->Update();
  st[1] =  (TPaveStats*) hist2->GetListOfFunctions()->FindObject("stats");
  //st[0]->SetName(leg[ifile]);
  st[1]->SetX1NDC(startingX);
  st[1]->SetX2NDC(startingX+0.23);
  st[1]->SetY1NDC(startingY-Ystep*(double(ifile)+1.));
  st[1]->SetY2NDC(startingY-Ystep*double(ifile));
  st[1]->SetTextColor(hist2->GetLineColor());
    ifile++;
    hist3->Draw("samesP"); 
    gPad->Update();
    st[2] =  (TPaveStats*) hist3->GetListOfFunctions()->FindObject("stats");
    //st[0]->SetName(leg[ifile]);
    st[2]->SetX1NDC(startingX);
    st[2]->SetX2NDC(startingX+0.23);
    st[2]->SetY1NDC(startingY-Ystep*(double(ifile)+1.));
    st[2]->SetY2NDC(startingY-Ystep*double(ifile));
    st[2]->SetTextColor(hist3->GetMarkerColor());
  ifile++;
  hist4->Draw("sames"); 
  gPad->Update();
  st[3] =  (TPaveStats*) hist4->GetListOfFunctions()->FindObject("stats");
  //st[0]->SetName(leg[ifile]);
  st[3]->SetX1NDC(startingX);
  st[3]->SetX2NDC(startingX+0.23);
  st[3]->SetY1NDC(startingY-Ystep*(double(ifile)+1.));
  st[3]->SetY2NDC(startingY-Ystep*double(ifile));
  st[3]->SetTextColor(hist4->GetLineColor());
  c1.Print( ( (name).c_str()+std::string("_4histos.gif") ).c_str(),"gif");  
}

void plotHist(TH1D* hist1,TH1D* hist2,std::string name){
  TCanvas c1; c1.cd();

    hist1->Draw("P"); 
  double startingX=0.7;
  double startingY=1;
  double Ystep=0.10;//0.15 if Optstat=1111
  gPad->Update();
    TPaveStats** st = new TPaveStats*[4];
    int ifile=0;
    gPad->Update();
    st[0] =  (TPaveStats*) hist1->GetListOfFunctions()->FindObject("stats");
    //st[0]->SetName(leg[ifile]);
    st[0]->SetX1NDC(startingX);
    st[0]->SetX2NDC(startingX+0.23);
    st[0]->SetY1NDC(startingY-Ystep*(double(ifile)+1.));
    st[0]->SetY2NDC(startingY-Ystep*double(ifile));
    st[0]->SetTextColor(hist1->GetMarkerColor());
  ifile++;
  hist2->Draw("sames");
  gPad->Update();
  st[1] =  (TPaveStats*) hist2->GetListOfFunctions()->FindObject("stats");
  //st[0]->SetName(leg[ifile]);
  st[1]->SetX1NDC(startingX);
  st[1]->SetX2NDC(startingX+0.23);
  st[1]->SetY1NDC(startingY-Ystep*(double(ifile)+1.));
  st[1]->SetY2NDC(startingY-Ystep*double(ifile));
  st[1]->SetTextColor(hist2->GetLineColor());

  c1.Print( ( (name).c_str()+std::string(".gif") ).c_str(),"gif");  
}

void plotHistlog(TH1D* hist1,TH1D* hist2,std::string name){
  TCanvas c1; c1.cd();
  gPad->SetLogy();

    hist1->Draw("P"); 
  double startingX=0.7;
  double startingY=1;
  double Ystep=0.10;//0.15 if Optstat=1111
  gPad->Update();
    TPaveStats** st = new TPaveStats*[4];
    int ifile=0;
    gPad->Update();
    st[0] =  (TPaveStats*) hist1->GetListOfFunctions()->FindObject("stats");
    //st[0]->SetName(leg[ifile]);
    st[0]->SetX1NDC(startingX);
    st[0]->SetX2NDC(startingX+0.23);
    st[0]->SetY1NDC(startingY-Ystep*(double(ifile)+1.));
    st[0]->SetY2NDC(startingY-Ystep*double(ifile));
    st[0]->SetTextColor(hist1->GetMarkerColor());
  ifile++;
  hist2->Draw("sames");
  gPad->Update();
  st[1] =  (TPaveStats*) hist2->GetListOfFunctions()->FindObject("stats");
  //st[0]->SetName(leg[ifile]);
  st[1]->SetX1NDC(startingX);
  st[1]->SetX2NDC(startingX+0.23);
  st[1]->SetY1NDC(startingY-Ystep*(double(ifile)+1.));
  st[1]->SetY2NDC(startingY-Ystep*double(ifile));
  st[1]->SetTextColor(hist2->GetLineColor());

  c1.Print( ( (name).c_str()+std::string(".gif") ).c_str(),"gif");  
}

