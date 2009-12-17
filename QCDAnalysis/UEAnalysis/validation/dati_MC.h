#include <iostream>
#include <fstream>
#include <vector>

#include "TFile.h"
#include "TBranch.h"
#include "TTree.h"
#include "TChain.h"
#include "TH1.h"
#include "TH2.h"
#include "TROOT.h"
#include "TStyle.h"
#include "TCanvas.h"

struct dati 
{
int eventNum;
int runNumber;
int lumiBlock;
int numVertices;
int genEventScale;
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
TBranch * b_genEventScale; 
TBranch * b_numVertices;
TBranch * b_vertex_x;
TBranch * b_vertex_y;
TBranch * b_vertex_z;
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
    fChain->SetBranchAddress("genEventScale",&genEventScale);
    fChain->SetBranchAddress("numVertices",&numVertices);
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
    b_genEventScale = fChain->GetBranch("genEventScale");
    b_numVertices = fChain->GetBranch("numVertices");
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
