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
#include "TChain.h"

struct dati 
{
int eventNum;
int runNumber;
int lumiBlock;
int bunchCrossing;
int numVertices;
double beamspot_x;
double beamspot_y;
double beamspot_z;
int vxFake[30];
int vxValid[30];
double vertex_x[30];
double vertex_y[30];
double vertex_z[30];
double vertex_xError[30];
double vertex_yError[30];
double vertex_zError[30];
int vertex_nOutgoingTracks[30]; 
int numTracks;
double track_eta[10000];
double track_phi[10000];
double track_p[10000];
double track_pt[10000];
double track_px[10000];
double track_py[10000];
double track_pz[10000];
double track_d0[10000];
double track_d0Error[10000];
double track_dz[10000];
double track_dzError[10000];
double track_recHitsSize[10000];
double track_chi2[10000];
double track_ndof[10000];
double track_normalizedChi2[10000];
double track_vx[10000];
double track_vy[10000];
double track_vz[10000];
double track_charge[10000];
int track_loose[10000];
int track_tight[10000];
int track_highPurity[10000];
double vtxAssocTrack_eta[10000];
double vtxAssocTrack_phi[10000];
double vtxAssocTrack_p[10000];


double vtxAssocTrack_pt[10000];
double vtxAssocTrack_d0[10000];
double vtxAssocTrack_d0Error[10000];
double vtxAssocTrack_dz[10000];
double vtxAssocTrack_dzError[10000];
double vtxAssocTrack_recHitsSize[10000];
double vtxAssocTrack_chi2[10000];
double vtxAssocTrack_ndof[10000];
double vtxAssocTrack_normalizedChi2[10000];
double vtxAssocTrack_vx[10000];
double vtxAssocTrack_vy[10000];
double vtxAssocTrack_vz[10000];

TBranch * b_eventNum;
TBranch * b_runNumber;
TBranch * b_lumiBlock;
TBranch * b_bunchCrossing;
TBranch * b_numVertices;
TBranch * b_vxFake;
TBranch * b_vxValid;
TBranch * b_vertex_x;
TBranch * b_vertex_y;
TBranch * b_vertex_z;
TBranch * b_beamspot_x;
TBranch * b_beamspot_y;
TBranch * b_beamspot_z;
TBranch * b_vertex_xError;
TBranch * b_vertex_yError;
TBranch * b_vertex_zError;
TBranch * b_vertex_nOutgoingTracks; 
TBranch * b_numTracks;
TBranch * b_track_eta;
TBranch * b_track_ph;
TBranch * b_track_p;
TBranch * b_track_pt;
TBranch * b_track_px;
TBranch * b_track_py;
TBranch * b_track_pz;
TBranch * b_track_d0;
TBranch * b_track_d0Error;
TBranch * b_track_dz;
TBranch * b_track_dzError;
TBranch * b_track_recHitsSize;
TBranch * b_track_chi2;
TBranch * b_track_ndof;
TBranch * b_track_normalizedChi2;
TBranch * b_track_vx;
TBranch * b_track_vy;
TBranch * b_track_vz;
TBranch * b_track_charge;
TBranch * b_track_loose;
TBranch * b_track_tight;
TBranch * b_track_highPurity;
TBranch * b_vtxAssocTrack_eta;
TBranch * b_vtxAssocTrack_phi;
TBranch * b_vtxAssocTrack_p;
TBranch * b_vtxAssocTrack_pt;
TBranch * b_vtxAssocTrack_d0;
TBranch * b_vtxAssocTrack_d0Error;
TBranch * b_vtxAssocTrack_dz;
TBranch * b_vtxAssocTrack_dzError;
TBranch * b_vtxAssocTrack_recHitsSize;
TBranch * b_vtxAssocTrack_chi2;
TBranch * b_vtxAssocTrack_ndof;
TBranch * b_vtxAssocTrack_normalizedChi2;
TBranch * b_vtxAssocTrack_vx;
TBranch * b_vtxAssocTrack_vy;
TBranch * b_vtxAssocTrack_vz;

  dati (TTree * fChain) 
    {
    fChain->SetMakeClass(1);

    fChain->SetBranchAddress("eventNum",&eventNum);
    fChain->SetBranchAddress("runNumber",&runNumber);
    fChain->SetBranchAddress("lumiBlock",&lumiBlock);
    fChain->SetBranchAddress("bunchCrossing",&bunchCrossing);
    fChain->SetBranchAddress("numVertices",&numVertices);
    fChain->SetBranchAddress("beamspot_x",&beamspot_x);
    fChain->SetBranchAddress("beamspot_y",&beamspot_y);
    fChain->SetBranchAddress("beamspot_z",&beamspot_z);
    fChain->SetBranchAddress("vxFake",vxFake);
    fChain->SetBranchAddress("vxValid",vxValid);
    fChain->SetBranchAddress("vertex_x",vertex_x);
    fChain->SetBranchAddress("vertex_y",vertex_y);
    fChain->SetBranchAddress("vertex_z",vertex_z);
    fChain->SetBranchAddress("vertex_xError",vertex_xError);
    fChain->SetBranchAddress("vertex_yError",vertex_yError);
    fChain->SetBranchAddress("vertex_zError",vertex_zError);
    fChain->SetBranchAddress("vertex_nOutgoingTracks",vertex_nOutgoingTracks);
    fChain->SetBranchAddress("numTracks",&numTracks);
    fChain->SetBranchAddress("track_eta",track_eta);
    fChain->SetBranchAddress("track_phi",track_phi);
    fChain->SetBranchAddress("track_p",track_p);
    fChain->SetBranchAddress("track_pt",track_pt);
    fChain->SetBranchAddress("track_px",track_px);
    fChain->SetBranchAddress("track_py",track_py);
    fChain->SetBranchAddress("track_pz",track_pz);
    fChain->SetBranchAddress("track_d0",track_d0);
    fChain->SetBranchAddress("track_d0Error",track_d0Error);
    fChain->SetBranchAddress("track_dz",track_dz);
    fChain->SetBranchAddress("track_dzError",track_dzError);
    fChain->SetBranchAddress("track_recHitsSize",track_recHitsSize);
    fChain->SetBranchAddress("track_chi2",track_chi2);
    fChain->SetBranchAddress("track_ndof",track_ndof);
    fChain->SetBranchAddress("track_normalizedChi2",track_normalizedChi2);
    fChain->SetBranchAddress("track_vx",track_vx);
    fChain->SetBranchAddress("track_vy",track_vy);
    fChain->SetBranchAddress("track_vz",track_vz);
    fChain->SetBranchAddress("track_charge",track_charge);
    fChain->SetBranchAddress("track_loose",track_loose);
    fChain->SetBranchAddress("track_tight",track_tight);
    fChain->SetBranchAddress("track_highPurity",track_highPurity);
    fChain->SetBranchAddress("vtxAssocTrack_eta",vtxAssocTrack_eta);
    fChain->SetBranchAddress("vtxAssocTrack_phi",vtxAssocTrack_phi);
    fChain->SetBranchAddress("vtxAssocTrack_p",vtxAssocTrack_p);
    fChain->SetBranchAddress("vtxAssocTrack_pt",vtxAssocTrack_pt);
    fChain->SetBranchAddress("vtxAssocTrack_d0",vtxAssocTrack_d0);
    fChain->SetBranchAddress("vtxAssocTrack_d0Error",vtxAssocTrack_d0Error);
    fChain->SetBranchAddress("vtxAssocTrack_dz",vtxAssocTrack_dz);
    fChain->SetBranchAddress("vtxAssocTrack_dzError",vtxAssocTrack_dzError);
    fChain->SetBranchAddress("vtxAssocTrack_recHitsSize",vtxAssocTrack_recHitsSize);
    fChain->SetBranchAddress("vtxAssocTrack_chi2",vtxAssocTrack_chi2);
    fChain->SetBranchAddress("vtxAssocTrack_ndof",vtxAssocTrack_ndof);
    fChain->SetBranchAddress("vtxAssocTrack_normalizedChi2",vtxAssocTrack_normalizedChi2);
    fChain->SetBranchAddress("vtxAssocTrack_vx",vtxAssocTrack_vx);
    fChain->SetBranchAddress("vtxAssocTrack_vy",vtxAssocTrack_vy);
    fChain->SetBranchAddress("vtxAssocTrack_vz",vtxAssocTrack_vz);
    
    b_eventNum = fChain->GetBranch("eventNum");
    b_runNumber = fChain->GetBranch("runNumber");
    b_lumiBlock = fChain->GetBranch("lumiBlock");
    b_bunchCrossing = fChain->GetBranch("bunchCrossing");
    b_numVertices = fChain->GetBranch("numVertices");
    b_beamspot_x = fChain->GetBranch("beamspot_x");
    b_beamspot_y = fChain->GetBranch("beamspot_y");
    b_beamspot_z = fChain->GetBranch("beamspot_z");
    b_vxFake = fChain->GetBranch("vxFake");
    b_vxValid = fChain->GetBranch("vxValid");
    b_vertex_x = fChain->GetBranch("vertex_x");
    b_vertex_y = fChain->GetBranch("vertex_y");
    b_vertex_z = fChain->GetBranch("vertex_z");
    b_vertex_xError = fChain->GetBranch("vertex_xError");
    b_vertex_yError = fChain->GetBranch("vertex_yError");
    b_vertex_zError = fChain->GetBranch("vertex_zError");
    b_vertex_nOutgoingTracks = fChain->GetBranch("vertex_nOutgoingTracks"); 
    b_numTracks = fChain->GetBranch("numTracks");
    b_track_eta = fChain->GetBranch("track_eta");
    b_track_ph = fChain->GetBranch("track_ph");
    b_track_p = fChain->GetBranch("track_p");
    b_track_pt = fChain->GetBranch("track_pt");
    b_track_px = fChain->GetBranch("track_px");
    b_track_py = fChain->GetBranch("track_py");
    b_track_pz = fChain->GetBranch("track_pz");
    b_track_d0 = fChain->GetBranch("track_d0");
    b_track_d0Error = fChain->GetBranch("track_d0Error");
    b_track_dz = fChain->GetBranch("track_dz");
    b_track_dzError = fChain->GetBranch("track_dzError");
    b_track_recHitsSize = fChain->GetBranch("track_recHitsSize");
    b_track_chi2 = fChain->GetBranch("track_chi2");
    b_track_ndof = fChain->GetBranch("track_ndof");
    b_track_normalizedChi2 = fChain->GetBranch("track_normalizedChi2");
    b_track_vx = fChain->GetBranch("track_vx");
    b_track_vy = fChain->GetBranch("track_vy");
    b_track_vz = fChain->GetBranch("track_vz");
    b_track_charge = fChain->GetBranch("track_charge");
    b_track_loose = fChain->GetBranch("track_loose");
    b_track_tight = fChain->GetBranch("track_tight");
    b_track_highPurity = fChain->GetBranch("track_highPurity");
    b_vtxAssocTrack_eta = fChain->GetBranch("vtxAssocTrack_eta");
    b_vtxAssocTrack_phi = fChain->GetBranch("vtxAssocTrack_phi");
    b_vtxAssocTrack_p = fChain->GetBranch("vtxAssocTrack_p");
    b_vtxAssocTrack_pt = fChain->GetBranch("vtxAssocTrack_pt");
    b_vtxAssocTrack_d0 = fChain->GetBranch("vtxAssocTrack_d0");
    b_vtxAssocTrack_d0Error = fChain->GetBranch("vtxAssocTrack_d0Error");
    b_vtxAssocTrack_dz = fChain->GetBranch("vtxAssocTrack_dz");
    b_vtxAssocTrack_dzError = fChain->GetBranch("vtxAssocTrack_dzError");
    b_vtxAssocTrack_recHitsSize = fChain->GetBranch("vtxAssocTrack_recHitsSize");
    b_vtxAssocTrack_chi2 = fChain->GetBranch("vtxAssocTrack_chi2");
    b_vtxAssocTrack_ndof = fChain->GetBranch("vtxAssocTrack_ndof");
    b_vtxAssocTrack_normalizedChi2 = fChain->GetBranch("vtxAssocTrack_normalizedChi2");
    b_vtxAssocTrack_vx = fChain->GetBranch("vtxAssocTrack_vx");
    b_vtxAssocTrack_vy = fChain->GetBranch("vtxAssocTrack_vy");
    b_vtxAssocTrack_vz = fChain->GetBranch("vtxAssocTrack_vz");
    
    }

};





bool realdata = true;

void setStatError(TH1D* hist);
bool RunLumi(int run, int lumi);
bool goodBunchCrossing(int,int);
int main(int argc, char* argv[])
{

  int Run=(int)atof(argv[1]);
  char *Runnome=argv[1];

 double beamspot=-2.64157;

//  gROOT->Reset();
  gStyle->SetOptStat(1111);
  gStyle->SetCanvasBorderMode(0);
  gStyle->SetPadBorderMode(0);

  gStyle->SetCanvasColor(10);
  gStyle->SetPadColor(10);
  gStyle->SetFillColor(10);
  gStyle->SetStatColor(10);
  gStyle->SetTitleFillColor(10);

std::string addMe = "DATA_";

TH1D* eta_All             =new TH1D( (addMe + std::string("eta_All")).c_str()        , (addMe + std::string("eta_All")).c_str(),25,-3,3);
TH1D* eta_withVertex      =new TH1D( (addMe + std::string("eta_withVertex")).c_str() , (addMe + std::string("eta_withVertex")).c_str(),25,-3,3);
TH1D* eta_AssVertex       =new TH1D( (addMe + std::string("eta_AssVertex")).c_str()  , (addMe + std::string("eta_AssVertex")).c_str(),25,-3,3);
  TH1D* phi_All             =new TH1D( (addMe + std::string("phi_All")).c_str()        , (addMe + std::string("phi_All")).c_str(),25,-3.1415,3.1415);
  TH1D* phi_withVertex      =new TH1D( (addMe + std::string("phi_withVertex")).c_str() , (addMe + std::string("phi_withVertex")).c_str(),25,-3.1415,3.1415);
  TH1D* phi_AssVertex       =new TH1D( (addMe + std::string("phi_AssVertex")).c_str()  , (addMe + std::string("phi_AssVertex")).c_str(),25,-3.1415,3.1415);
TH1D* pt_All              =new TH1D( (addMe + std::string("pt_All")).c_str()         , (addMe + std::string("pt_All")).c_str(),400,0.,40);
TH1D* pt_withVertex       =new TH1D( (addMe + std::string("pt_withVertex")).c_str()  , (addMe + std::string("pt_withVertex")).c_str(),400,0.,40);
TH1D* pt_AssVertex        =new TH1D( (addMe + std::string("pt_AssVertex")).c_str()   , (addMe + std::string("pt_AssVertex")).c_str(),400,0.,40);
  TH1D* track_multip_All       =new TH1D( (addMe + std::string("track_multip_All")).c_str()       , (addMe + std::string("track_multip_All")).c_str(),201,0,200);
  TH1D* track_multip_withVertex=new TH1D( (addMe + std::string("track_multip_withVertex")).c_str(), (addMe + std::string("track_multip_withVertex")).c_str(),201,0,200);
  TH1D* track_multip_AssVertex =new TH1D( (addMe + std::string("track_multip_AssVertex")).c_str() , (addMe + std::string("track_multip_AssVertex")).c_str(),201,0,200);
TH2D* pt_vs_multip_All              =new TH2D( (addMe + std::string("pt_vs_multip_All")).c_str()         , (addMe + std::string("pt_vs_multip_All")).c_str(),400,0.,40,201,0,200);
TH2D* pt_vs_multip_withVertex       =new TH2D( (addMe + std::string("pt_vs_multip_withVertex")).c_str()  , (addMe + std::string("pt_vs_multip_withVertex")).c_str(),400,0.,40,201,0,200);
TH2D* pt_vs_multip_AssVertex        =new TH2D( (addMe + std::string("pt_vs_multip_AssVertex")).c_str()   , (addMe + std::string("pt_vs_multip_AssVertex")).c_str(),400,0.,40,201,0,200);
  TH2D* pt_vs_ndof_All              =new TH2D( (addMe + std::string("pt_vs_ndof_All")).c_str()         , (addMe + std::string("pt_vs_ndof_All")).c_str(),400,0.,40,61,0,60);
  TH2D* pt_vs_ndof_withVertex       =new TH2D( (addMe + std::string("pt_vs_ndof_withVertex")).c_str()  , (addMe + std::string("pt_vs_ndof_withVertex")).c_str(),400,0.,40,61,0,60);
  TH2D* pt_vs_ndof_AssVertex        =new TH2D( (addMe + std::string("pt_vs_ndof_AssVertex")).c_str()   , (addMe + std::string("pt_vs_ndof_AssVertex")).c_str(),400,0.,40,61,0,60);
TH1D* ndof_All              =new TH1D( (addMe + std::string("ndof_All")).c_str()         , (addMe + std::string("ndof_All")).c_str(),61,0,60);
TH1D* ndof_withVertex       =new TH1D( (addMe + std::string("ndof_withVertex")).c_str()  , (addMe + std::string("ndof_withVertex")).c_str(),61,0,60);
TH1D* ndof_AssVertex        =new TH1D( (addMe + std::string("ndof_AssVertex")).c_str()   , (addMe + std::string("ndof_AssVertex")).c_str(),61,0,60);
  TH1D* normChi2_All        =new TH1D( (addMe + std::string("normChi2_All")).c_str()        , (addMe + std::string("normChi2_All")).c_str(),25,0,20);
  TH1D* normChi2_withVertex =new TH1D( (addMe + std::string("normChi2_withVertex")).c_str() , (addMe + std::string("normChi2_withVertex")).c_str(),25,0,20);
  TH1D* normChi2_AssVertex  =new TH1D( (addMe + std::string("normChi2_AssVertex")).c_str()  , (addMe + std::string("normChi2_AssVertex")).c_str(),25,0,20);
TH2D* chi2_vs_pT_All       =new TH2D( (addMe + std::string("chi2_vs_pT_All")).c_str()       , (addMe + std::string("chi2_vs_pT_All")).c_str(),25,0,20,400,0.,40);
TH2D* chi2_vs_pT_withVertex=new TH2D( (addMe + std::string("chi2_vs_pT_withVertex")).c_str(), (addMe + std::string("chi2_vs_pT_withVertex")).c_str(),25,0,20,400,0.,40);
TH2D* chi2_vs_pT_AssVertex =new TH2D( (addMe + std::string("chi2_vs_pT_AssVertex")).c_str() , (addMe + std::string("chi2_vs_pT_AssVertex")).c_str(),25,0,20,400,0.,40);
  TH2D* chi2_vs_pT_lowMultip_All       =new TH2D( (addMe + std::string("chi2_vs_pT_lowMultip_All")).c_str()       , (addMe + std::string("chi2_vs_pT_lowMultip_All")).c_str(),25,0,20,400,0.,40);
  TH2D* chi2_vs_pT_lowMultip_withVertex=new TH2D( (addMe + std::string("chi2_vs_pT_lowMultip_withVertex")).c_str(), (addMe + std::string("chi2_vs_pT_lowMultip_withVertex")).c_str(),25,0,20,400,0.,40);
  TH2D* chi2_vs_pT_lowMultip_AssVertex =new TH2D( (addMe + std::string("chi2_vs_pT_lowMultip_AssVertex")).c_str() , (addMe + std::string("chi2_vs_pT_lowMultip_AssVertex")).c_str(),25,0,20,400,0.,40);
TH2D* chi2_vs_pT_highMultip_All       =new TH2D( (addMe + std::string("chi2_vs_pT_highMultip_All")).c_str()       , (addMe + std::string("chi2_vs_pT_highMultip_All")).c_str(),25,0,20,400,0.,40);
TH2D* chi2_vs_pT_highMultip_withVertex=new TH2D( (addMe + std::string("chi2_vs_pT_highMultip_withVertex")).c_str(), (addMe + std::string("chi2_vs_pT_highMultip_withVertex")).c_str(),25,0,20,400,0.,40);
TH2D* chi2_vs_pT_highMultip_AssVertex =new TH2D( (addMe + std::string("chi2_vs_pT_highMultip_AssVertex")).c_str() , (addMe + std::string("chi2_vs_pT_highMultip_AssVertex")).c_str(),25,0,20,400,0.,40);
  TH1D* vertexes            =new TH1D( (addMe + std::string("vertexes")).c_str()       , (addMe + std::string("vertexes")).c_str(),5,0,5);
  TH1D* vertexes_z          =new TH1D( (addMe + std::string("vertexes_z")).c_str()     , (addMe + std::string("vertexes_z")).c_str(),39,-20,20);
 TH1D* vertexes_z_beamspot  =new TH1D( (addMe + std::string("vertexes_z_beamspot")).c_str()     , (addMe + std::string("vertexes_z - beamspot_z")).c_str(),39,-20,20);
  TH2D* vertexes_xy         =new TH2D( (addMe + std::string("vertexes_xy")).c_str()    , (addMe + std::string("vertexes_xy")).c_str(),200,-10,10,200,-10,10);
  TH1D* deltaZ_trackPV      =new TH1D( (addMe + std::string("deltaZ_trackPV")).c_str() , (addMe + std::string("deltaZ_trackPV")).c_str(),50,-25,25);
  TH1D* deltaZ_trackPV_ZOOM      =new TH1D( (addMe + std::string("deltaZ_trackPV_ZOOM")).c_str() , (addMe +  std::string("deltaZ_trackPV_ZOOM")).c_str(),100,-3,3);
TH1D* deltaX_trackPV      =new TH1D( (addMe + std::string("deltaX_trackPV")).c_str() , (addMe + std::string("deltaX_trackPV")).c_str(),50,-25,25);
TH1D* deltaX_trackPV_ZOOM      =new TH1D( (addMe + std::string("deltaX_trackPV_ZOOM")).c_str() , (addMe + std::string("deltaX_trackPV_ZOOM")).c_str(),100,-3,3);
  TH1D* deltaY_trackPV      =new TH1D( (addMe + std::string("deltaY_trackPV")).c_str() , (addMe + std::string("deltaY_trackPV")).c_str(),50,-25,25);
  TH1D* deltaY_trackPV_ZOOM      =new TH1D( (addMe + std::string("deltaY_trackPV_ZOOM")).c_str() , (addMe + std::string("deltaY_trackPV_ZOOM")).c_str(),100,-3,3);
TH1D* vertexesNum_NoutTrk =new TH1D( (addMe + std::string("vertexesNum_NoutTrk")).c_str() , (addMe + std::string("vertexesNum_NoutTrk")).c_str(),50,0,50);
TH1D* deltaZ_v1v2         =new TH1D( (addMe + std::string("deltaZ_v1v2")).c_str()    , (addMe + std::string("deltaZ_v1v2")).c_str(),100,0,50);
  TH1D* dZ_All      =new TH1D( (addMe + std::string("dZ_All")).c_str() , (addMe +  std::string("dZ_All")).c_str(),100,-0.3,0.3);
  TH1D* d0_All      =new TH1D( (addMe + std::string("d0_All")).c_str() , (addMe + std::string("d0_All")).c_str(),100,-0.3,0.3);
TH1D* dZPoint_Ass      =new TH1D( (addMe + std::string("dZPoint_Ass")).c_str() , (addMe +  std::string("dZPoint_Ass")).c_str(),100,-0.3,0.3);
TH1D* d0Point_Ass      =new TH1D( (addMe + std::string("d0Point_Ass")).c_str() , (addMe + std::string("d0Point_Ass")).c_str(),100,-0.3,0.3);
TH1D* dZ_Ass      =new TH1D( (addMe + std::string("dZ_Ass")).c_str() , (addMe +  std::string("dZ_Ass")).c_str(),100,-0.3,0.3);
TH1D* d0_Ass      =new TH1D( (addMe + std::string("d0_Ass")).c_str() , (addMe + std::string("d0_Ass")).c_str(),100,-0.3,0.3);
  TH1D* dZPoint_Vtx      =new TH1D( (addMe + std::string("dZPoint_Vtx")).c_str() , (addMe +  std::string("dZPoint_Vtx")).c_str(),100,-0.3,0.3);
  TH1D* d0Point_Vtx      =new TH1D( (addMe + std::string("d0Point_Vtx")).c_str() , (addMe + std::string("d0Point_Vtx")).c_str(),100,-0.3,0.3);
  TH1D* dZ_Vtx      =new TH1D( (addMe + std::string("dZ_Vtx")).c_str() , (addMe +  std::string("dZ_Vtx")).c_str(),100,-0.3,0.3);
  TH1D* d0_Vtx      =new TH1D( (addMe + std::string("d0_Vtx")).c_str() , (addMe + std::string("d0_Vtx")).c_str(),100,-0.3,0.3);
TH1D* dZ_Vtx_LARGE      =new TH1D( (addMe + std::string("dZ_Vtx_LARGE")).c_str() , (addMe + std::string("dZ_Vtx_LARGE")).c_str(),500,-15,15);
  TH1D* hthrust_z_All            =new TH1D( (addMe + std::string("hthrust_z_All")).c_str()       , (addMe + std::string("thrust_z_All")).c_str(),50,-1.1,1.1);
  TH1D* hthrust_z_withVertex     =new TH1D( (addMe + std::string("hthrust_z_withVertex")).c_str(), (addMe + std::string("thrust_y_withVertex")).c_str(),50,-1.1,1.1);
  TH1D* hthrust_z_AssVertex      =new TH1D( (addMe + std::string("hthrust_z_AssVertex")).c_str() , (addMe + std::string("thrust_z_AssVertex")).c_str(),50,-1.1,1.1);
TH1D* chargeAsymmetry_All = new TH1D( (addMe + std::string("chargeAsymmetry_All")).c_str() , (addMe + std::string("chargeAsymmetry_All")).c_str(),20,-10,10);
TH1D* chargeAsymmetry_AssVertex = new TH1D( (addMe + std::string("chargeAsymmetry_AssVertex")).c_str() , (addMe + std::string("chargeAsymmetry_AssVertex")).c_str(),20,-10,10);

//*********************************  

  TChain fChain("MyAnalyzer/EventTree") ;
   fChain.Add("rfio:/castor/cern.ch/cms/store/caf/user/lucaroni/Validazione/MinimumBias_BeamCommissioning09-BSCNOBEAMHALO-Dec19thSkim_336p3_v1_bit0_Trackjet/*.root");
   //  fChain.Add("/afs/cern.ch/cms/CAF/CMSALCA/ALCA_TRACKERALIGN/HIP/lucaroni/QCD_2/Validazione/CMSSW_3_3_6_patch3/src/QCDAnalysis/UEAnalysis/test/prova_06012010_1740/root/*.root");
  dati Input(&fChain);
  int entries = fChain.GetEntries();


//*******  TRACK parameters  ******  

cout<<"corro su "<<entries<<" entries"<<endl;

for(int eventn=0;eventn<entries;eventn++){
  fChain.GetEntry(eventn);

  beamspot = Input.beamspot_z;
  
  if (realdata)
    {
      if (!RunLumi(Input.runNumber,Input.lumiBlock) ) continue; 
      if(!goodBunchCrossing(Input.runNumber,Input.bunchCrossing)) continue;
      if(Run!=Input.runNumber) continue;  
  }


  int chargePlusY=0;
  int chargeMinusY=0;
  double thrust_z=0;
  double ptot=0;
//All
  track_multip_All->Fill(Input.numTracks);
  for(Int_t trackn=0;trackn<Input.numTracks;trackn++){ //tutte le tracce dell'evento	
    
    thrust_z=Input.track_pz[trackn]+thrust_z;
    ptot=ptot+Input.track_p[trackn];
   
    eta_All->Fill(Input.track_eta[trackn]);
    phi_All->Fill(Input.track_phi[trackn]);
    pt_All->Fill(Input.track_pt[trackn]);
    normChi2_All->Fill(Input.track_normalizedChi2[trackn]);
    chi2_vs_pT_All->Fill(Input.track_normalizedChi2[trackn],Input.track_pt[trackn]);
    pt_vs_multip_All->Fill(Input.track_pt[trackn],Input.numTracks);
    pt_vs_ndof_All->Fill(Input.track_pt[trackn],Input.track_ndof[trackn]);
    ndof_All->Fill(Input.track_ndof[trackn]);
    if (Input.numTracks>100)    chi2_vs_pT_highMultip_All->Fill(Input.track_normalizedChi2[trackn],Input.track_pt[trackn]);
    else                  chi2_vs_pT_lowMultip_All->Fill(Input.track_normalizedChi2[trackn],Input.track_pt[trackn]);
    dZ_All->Fill( Input.track_dz[trackn] );
    d0_All->Fill( Input.track_d0[trackn] );
    
    if ( Input.track_py[trackn]>0. )  chargePlusY+=(int)Input.track_charge[trackn];
    else                              chargeMinusY+=(int)Input.track_charge[trackn];
    
  }//tutte le tracce dell'evento
  
  if(ptot!=0 && Input.numTracks>0){
    thrust_z=thrust_z/ptot;
    hthrust_z_All->Fill(thrust_z);}
  thrust_z=0;
  ptot=0;

  chargeAsymmetry_All->Fill(chargePlusY+chargeMinusY);


//vertex presence requirement
  for(Int_t vertexn=0;vertexn<Input.numVertices;vertexn++)
    {
    if(Input.vertex_z[vertexn]!=beamspot){  //vertice non ricostruito valore del beam spot
      if(fabs(Input.vertex_z[vertexn]-beamspot)<10){
        track_multip_withVertex->Fill(Input.numTracks);
	for(Int_t trackn=0;trackn<Input.numTracks;trackn++){
      	  
	  //if (track_pt[trackn]>8.) continue;
	  thrust_z=Input.track_pz[trackn]+thrust_z;
	  ptot=ptot+Input.track_p[trackn];

	  eta_withVertex->Fill(Input.track_eta[trackn]);
	  phi_withVertex->Fill(Input.track_phi[trackn]);
	  pt_withVertex->Fill(Input.track_pt[trackn]);
	  normChi2_withVertex->Fill(Input.track_normalizedChi2[trackn]); 
          chi2_vs_pT_withVertex->Fill(Input.track_normalizedChi2[trackn],Input.track_pt[trackn]);
          pt_vs_multip_withVertex->Fill(Input.track_pt[trackn],Input.numTracks);
          pt_vs_ndof_withVertex->Fill(Input.track_pt[trackn],Input.track_ndof[trackn]);
          ndof_withVertex->Fill(Input.track_ndof[trackn]);
          if (Input.numTracks>100)    chi2_vs_pT_highMultip_withVertex->Fill(Input.track_normalizedChi2[trackn],Input.track_pt[trackn]);
          else                  chi2_vs_pT_lowMultip_withVertex->Fill(Input.track_normalizedChi2[trackn],Input.track_pt[trackn]);

	  dZPoint_Vtx->Fill( (Input.track_vz[trackn]-Input.vertex_z[vertexn])- ( (Input.track_vx[trackn]-Input.vertex_x[vertexn])*Input.track_px[trackn] +  (Input.track_vy[trackn]-Input.vertex_y[vertexn])*Input.track_py[trackn])/(Input.track_pt[trackn]) * Input.track_pz[trackn]/Input.track_pt[trackn] );
	  d0Point_Vtx->Fill( (-(Input.track_vx[trackn]-Input.vertex_x[vertexn])*Input.track_py[trackn]+(Input.track_vy[trackn]-Input.vertex_y[vertexn])*Input.track_px[trackn])/Input.track_pt[trackn] );
	    dZ_Vtx->Fill( Input.track_dz[trackn] );
	    dZ_Vtx_LARGE->Fill( Input.track_dz[trackn] );
	    d0_Vtx->Fill( Input.track_d0[trackn] );
          
	  }
	}
      break; 
      }//vertice non ricostruito valore del beam spot
    }

 if(ptot!=0 && Input.numTracks>0){
    thrust_z=thrust_z/ptot;
    hthrust_z_withVertex->Fill(thrust_z);
    }
  thrust_z=0;
  ptot=0;

//Vertex Association requirement

  chargePlusY=0;
  chargeMinusY=0;

  for(Int_t vertexn=0;vertexn<Input.numVertices;vertexn++)
    {
    //std::cout<<"vertices z: "<<Input.vertex_z[vertexn]<<endl;
    if(Input.vertex_z[vertexn]!=beamspot){  //vertice non ricostruito valore del beam spot
      if(fabs(Input.vertex_z[vertexn]-beamspot)<10){
        int fillCounter=0;
	for(Int_t trackn=0;trackn<Input.numTracks;trackn++){
      	  
	  double dZ_cut= (Input.track_vz[trackn]-Input.vertex_z[vertexn])- ( (Input.track_vx[trackn]-Input.vertex_x[vertexn])*Input.track_px[trackn] +  (Input.track_vy[trackn]-Input.vertex_y[vertexn])*Input.track_py[trackn])/(Input.track_pt[trackn]) * Input.track_pz[trackn]/Input.track_pt[trackn] ;
	  double d0_cut= (-(Input.track_vx[trackn]-Input.vertex_x[vertexn])*Input.track_py[trackn]+(Input.track_vy[trackn]-Input.vertex_y[vertexn])*Input.track_px[trackn])/Input.track_pt[trackn] ;
	  
	  //if ( fabs(track_vz[trackn]-vertex_z[vertexn])<1. )  //old cut in acceptance
	  if ( fabs(dZ_cut)<0.36 && fabs(d0_cut)<0.18 && Input.track_highPurity[trackn]==1 )
	    {
	    fillCounter++;

	    thrust_z=Input.track_pz[trackn]+thrust_z;
	    ptot=ptot+Input.track_p[trackn];

	    eta_AssVertex->Fill(Input.track_eta[trackn]);
	    phi_AssVertex->Fill(Input.track_phi[trackn]);
	    pt_AssVertex->Fill(Input.track_pt[trackn]);
	    normChi2_AssVertex->Fill(Input.track_normalizedChi2[trackn]);
            chi2_vs_pT_AssVertex->Fill(Input.track_normalizedChi2[trackn],Input.track_pt[trackn]);
            pt_vs_multip_AssVertex->Fill(Input.track_pt[trackn],Input.numTracks);
            pt_vs_ndof_AssVertex->Fill(Input.track_pt[trackn],Input.track_ndof[trackn]);
            ndof_AssVertex->Fill(Input.track_ndof[trackn]);
            if (Input.numTracks>100)    chi2_vs_pT_highMultip_AssVertex->Fill(Input.track_normalizedChi2[trackn],Input.track_pt[trackn]);
            else                  chi2_vs_pT_lowMultip_AssVertex->Fill(Input.track_normalizedChi2[trackn],Input.track_pt[trackn]);
	    
	    dZPoint_Ass->Fill( (Input.track_vz[trackn]-Input.vertex_z[vertexn])- ( (Input.track_vx[trackn]-Input.vertex_x[vertexn])*Input.track_px[trackn] +  (Input.track_vy[trackn]-Input.vertex_y[vertexn])*Input.track_py[trackn])/(Input.track_pt[trackn]) * Input.track_pz[trackn]/Input.track_pt[trackn] );
	    d0Point_Ass->Fill( (-(Input.track_vx[trackn]-Input.vertex_x[vertexn])*Input.track_py[trackn]+(Input.track_vy[trackn]-Input.vertex_y[vertexn])*Input.track_px[trackn])/Input.track_pt[trackn] );
	    dZ_Ass->Fill( Input.track_dz[trackn] );
	    d0_Ass->Fill( Input.track_d0[trackn] );

            if ( Input.track_py[trackn]>0. )  chargePlusY+=(int)Input.track_charge[trackn];
            else                        chargeMinusY+=(int)Input.track_charge[trackn];

	    }
          }
	  
        chargeAsymmetry_AssVertex->Fill(chargePlusY+chargeMinusY);
	}
      break; 
      }//vertice non ricostruito valore del beam spot
    }
  if(ptot!=0 && Input.numTracks>0){
    thrust_z=thrust_z/ptot;
    hthrust_z_AssVertex->Fill(thrust_z);
    }
  thrust_z=0;
  ptot=0;
}

  
//***** VERTEX properties *****

int contime=0;
cout<<"corro su "<<entries<<" entries"<<endl;
for(int eventn=0;eventn<entries;eventn++){
  fChain.GetEntry(eventn);
 beamspot = Input.beamspot_z;
 if (realdata){
    if ( !RunLumi(Input.runNumber,Input.lumiBlock) ) continue; 
 if(!goodBunchCrossing(Input.runNumber,Input.bunchCrossing)) continue;
 if(Run!=Input.runNumber) continue;  
 }

  int fill;
  if (Input.numVertices==1){
    if (Input.vertex_z[0]==beamspot) fill = 0;
    else                             fill = 1;
    }
  else fill = Input.numVertices;  
  vertexes->Fill(fill);

  if (Input.numVertices>1)
    deltaZ_v1v2->Fill( fabs(Input.vertex_z[0]-Input.vertex_z[1]) );
  
  for(Int_t vertexn=0;vertexn<Input.numVertices;vertexn++){
    if(Input.vertex_z[vertexn]!=beamspot){  //vertice non ricostruito valore del beam spot
	vertexesNum_NoutTrk->Fill(Input.vertex_nOutgoingTracks[vertexn]);
	vertexes_z->Fill(Input.vertex_z[vertexn]);
	vertexes_z_beamspot->Fill(Input.vertex_z[vertexn]-Input.beamspot_z);
	vertexes_xy->Fill(Input.vertex_x[vertexn],Input.vertex_y[vertexn]);
    }//vertice non ricostruito valore del beam spot
  }

  for(Int_t vertexn=0;vertexn<Input.numVertices;vertexn++)
    {
    if(Input.vertex_z[0]!=beamspot){  //vertice non ricostruito valore del beam spot
      if(fabs(Input.vertex_z[vertexn]-beamspot)<10){
        contime++;
	for(Int_t trackn=0;trackn<Input.numTracks;trackn++){
	  deltaZ_trackPV->Fill(Input.track_vz[trackn]-Input.vertex_z[vertexn]);  
	  deltaZ_trackPV_ZOOM->Fill(Input.track_vz[trackn]-Input.vertex_z[vertexn]);  
	    deltaX_trackPV->Fill(Input.track_vx[trackn]-Input.vertex_x[vertexn]);  
	    deltaX_trackPV_ZOOM->Fill(Input.track_vx[trackn]-Input.vertex_x[vertexn]);  
	  deltaY_trackPV->Fill(Input.track_vy[trackn]-Input.vertex_y[vertexn]);  
	  deltaY_trackPV_ZOOM->Fill(Input.track_vy[trackn]-Input.vertex_y[vertexn]);  
          }
	break;       
      }
      }//vertice non ricostruito valore del beam spot
    }

}
/*
cout<<"li gavemo contai: "<<contime<<endl;
double averageMultip = track_multip_AssVertex->GetMean();
TH1D* KNO_scaling = new TH1D("KNO_scaling","KNO_scaling",track_multip_AssVertex->GetNbinsX(),0.,double(track_multip_AssVertex->GetNbinsX())/averageMultip);
  KNO_scaling->GetXaxis()->SetTitle("n / < n >");
  KNO_scaling->GetYaxis()->SetTitle("< n >   P_{n}");
  for (int k=1; k<=KNO_scaling->GetNbinsX(); k++)
    KNO_scaling->SetBinContent(k,averageMultip*track_multip_AssVertex->GetBinContent(k));
*/

if (realdata){

  //********* SetStatisticalError Real Data *********

  setStatError(eta_withVertex);
  setStatError(eta_All);
  setStatError(eta_AssVertex);
  setStatError(phi_withVertex);
  setStatError(phi_All);
  setStatError(phi_AssVertex);
  setStatError(pt_withVertex);
  setStatError(pt_All);
  setStatError(pt_AssVertex);
  setStatError(normChi2_withVertex);
  setStatError(normChi2_All);
  setStatError(normChi2_AssVertex);
  setStatError(vertexes);
  setStatError(vertexes_z);
  setStatError(vertexes_z_beamspot);
  setStatError(deltaZ_trackPV);
  setStatError(deltaZ_trackPV_ZOOM);
  setStatError(vertexesNum_NoutTrk);
  setStatError(ndof_All);
  setStatError(ndof_withVertex);
  setStatError(ndof_AssVertex);
  setStatError(track_multip_All);
  setStatError(track_multip_withVertex);
  setStatError(track_multip_AssVertex);

  //****   marker styles  *****  

  eta_withVertex->SetMarkerStyle(22);
  eta_AssVertex->SetMarkerStyle(22);
  eta_All->SetMarkerStyle(22);
  phi_withVertex->SetMarkerStyle(22);
  phi_AssVertex->SetMarkerStyle(22);
  phi_All->SetMarkerStyle(22);
  pt_withVertex->SetMarkerStyle(22);
  pt_AssVertex->SetMarkerStyle(22);
  pt_All->SetMarkerStyle(22);
  normChi2_withVertex->SetMarkerStyle(22);
  normChi2_AssVertex->SetMarkerStyle(22);
  normChi2_All->SetMarkerStyle(22);

  vertexes_z->SetMarkerStyle(22);
  vertexes_z_beamspot->SetMarkerStyle(22);
  deltaZ_trackPV->SetMarkerStyle(22);
  deltaZ_trackPV_ZOOM->SetMarkerStyle(22);
  vertexesNum_NoutTrk->SetMarkerStyle(22);

  vertexes->SetMarkerStyle(22);
  }

  vertexes->GetXaxis()->SetBinLabel(1,"BeamSpot");
  vertexes->GetXaxis()->SetBinLabel(2,"1 Vertex");
  vertexes->GetXaxis()->SetBinLabel(3,"2 Vertices");
  vertexes->GetXaxis()->SetBinLabel(4,"3 Vertices");
  vertexes->GetXaxis()->SetBinLabel(5,"4 Vertices");
  vertexes->GetXaxis()->SetTitle("N^{vtx}/evt");

  deltaZ_v1v2->GetXaxis()->SetTitle("|v^{1}_{z}-v^{1}_{z}|");


//***** saving histos *****

  std::string fileToOpen=(Runnome + std::string("validatio_090110.root")).c_str();  
  TFile outFile((fileToOpen).c_str(),"RECREATE");
 
  eta_withVertex->Write();
  eta_All->Write();
  phi_withVertex->Write();
  phi_All->Write();
  pt_withVertex ->Write();
  pt_All->Write();
  normChi2_withVertex->Write();
  normChi2_All->Write();

  vertexes->Write();
  vertexes_z->Write();
  vertexes_z_beamspot->Write();
  vertexes_xy->Write();
  deltaZ_trackPV->Write();
  deltaZ_trackPV_ZOOM->Write();
  vertexesNum_NoutTrk->Write();
  deltaZ_v1v2->Write();

  eta_AssVertex->Write();
  phi_AssVertex->Write();
  pt_AssVertex ->Write();
  normChi2_AssVertex->Write();
  
  hthrust_z_All->Write();
  hthrust_z_withVertex->Write();
  hthrust_z_AssVertex->Write();

  track_multip_All->Write();
  track_multip_withVertex->Write();
  track_multip_AssVertex->Write();
    chi2_vs_pT_All->Write();
    chi2_vs_pT_withVertex->Write();
    chi2_vs_pT_AssVertex->Write();
  chi2_vs_pT_lowMultip_All->Write();
  chi2_vs_pT_lowMultip_withVertex->Write();
  chi2_vs_pT_lowMultip_AssVertex->Write();
    chi2_vs_pT_highMultip_All->Write();
    chi2_vs_pT_highMultip_withVertex->Write();
    chi2_vs_pT_highMultip_AssVertex->Write();

  pt_vs_multip_All->Write();
  pt_vs_multip_withVertex->Write();
  pt_vs_multip_AssVertex->Write();  
    pt_vs_ndof_All->Write();        
    pt_vs_ndof_withVertex->Write(); 
    pt_vs_ndof_AssVertex->Write();  

  deltaX_trackPV->Write();
  deltaX_trackPV_ZOOM->Write(); 
    deltaY_trackPV->Write();  
    deltaY_trackPV_ZOOM->Write();  

    dZ_All->Write();
    d0_All->Write();
  dZPoint_Ass->Write();
  d0Point_Ass->Write();
    dZ_Ass->Write();
    d0_Ass->Write();
  dZPoint_Vtx->Write();
  d0Point_Vtx->Write();
    dZ_Vtx->Write();
    dZ_Vtx_LARGE->Write();
    d0_Vtx->Write();

    // KNO_scaling->Write();

   ndof_All->Write();
   ndof_withVertex->Write();
   ndof_AssVertex->Write();
  
  chargeAsymmetry_All->Write();
  chargeAsymmetry_AssVertex->Write();
  
  outFile.Close();
}


//********  Helpfull functions *******

void setStatError(TH1D* hist){
  for (int bin=1;bin<=hist->GetNbinsX();bin++)
    hist->SetBinError( bin,sqrt(hist->GetBinContent(bin)) );
  }

bool RunLumi(int irun, int ilum){ 

  bool accepted = false;
     if (irun==123592) {
    if(ilum>=1 && ilum<=12) 
      accepted=true;
   }
if (irun==123596) {
    if (/*(ilum>=4 && ilum<=26) ||*/ //pixel timing scan
        (ilum>=69 && ilum<=144) )
      accepted=true;
  } else if (irun==123615) {
    if(ilum>=71) 
      accepted=true;
  } else if (irun==123732) {
    if(ilum>=62 && ilum<=112)        //though phys bit starting 57
      accepted=true;
  } else if (irun==123815) {
    if(ilum>=7 && ilum<=16)
      accepted=true;
  } else if (irun==123818) {
    if(ilum>=2 && ilum<=18)          //RunRegistry says lumi scan starting at lumi 19 until 42
      accepted=true;
  } else if (irun==123906) {
    if(ilum>=17 && ilum<=28)
      accepted=true;
  } else if (irun==123908) {
    if(ilum>=2 && ilum<=13)
      accepted=true;
  }else if (irun==124006) {
    if(ilum>=1 && ilum<=6)           //though Phys bit set from lumi 6
      accepted=true;
  }else if (irun==124008) {        
    if(ilum>=1 && ilum<=2)
      accepted=true;
  }else if (irun==124009) {         //lumi scan through 29-63
    if(ilum>=1 && ilum<=68)
      accepted=true;
  } else if (irun==124020) {
    if(ilum>=11 && ilum<=94)
      accepted=true;
  } else if (irun==124022) {
    if(ilum>=65 && ilum<=160)        //lumi scan through 161-183
      accepted=true;
  } else if (irun==124024) {
    if(ilum>=2 && ilum<=83)
      accepted=true;
  } else if (irun==124025) {
    if(ilum>=3 && ilum<=13)
      accepted=true;
  }
 else if (irun==124027) {
    if(ilum>=24)
      accepted=true;
  }
else if (irun==124030) {
    if(ilum>=1 && ilum<33)
      accepted=true;
  }
else if (irun==124230) {
    if(ilum>=26 && ilum<=68)         //paused at 47, resumed 48
      accepted=true;
  }

  return accepted;

  }


bool goodBunchCrossing(int irun,int bx){

 bool accepted = false;
 int type_=1;  //1 collision 2 non collision 
  if (irun==123592 || irun==123596 || irun==123615) {
    if (type_==1) {
      if ((bx==51) || (bx==2724))
        accepted=true;
    }else if (type_==2) {
      if ((bx==2276) || (bx==3170))
        accepted=true;
    }
  } 

  else if (irun==123732) {
    if (type_==1) {
      if ((bx==3487) || (bx==2596))
        accepted=true;
    } else if (type_==2) {
      if ((bx==2148) || (bx==3042))
        accepted=true;
    }
  } 

else if (irun==123815 || irun==123818) {
    if (type_==1) {
      if ((bx==2724) || (bx==51))
        accepted=true;
    } else if (type_==2) {
      if ((bx==2276) || (bx==3170))
        accepted=true;
    }
  }

else if (irun==123906 || irun==123908) {
    if (type_==1) {
      if ((bx==51))
        accepted=true;
    } else if (type_==2) {
      if ((bx==2276) || (bx==2724) || (bx==1830) || (bx==1833))
        accepted=true;
    }
  }

  else if (irun==124006 || irun==124008 || irun==124009 || irun==124020 || irun==124022 || irun==124024 || irun==124025 ||irun==124027 || irun==124030 ) {
    if (type_==1) {
      if ((bx==2824) || (bx==151) || (bx==51))
        accepted=true;
    } else if (type_==2) {
      if ((bx==1365) || (bx==474) ||(bx==2259) || (bx==1930) )
        accepted=true;
    }
  }
    
 else if (irun==124230) {
    if (type_==1) {
      if (bx==8 || bx==1)
        accepted=true;
    } else if (type_==2) {
      if (bx==13)
        accepted=true;
    }
  }

  return accepted;

  }
