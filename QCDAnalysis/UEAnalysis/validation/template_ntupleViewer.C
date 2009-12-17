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


bool realdata = STRINGrealdata;
bool oneRun=true;
int  fixedRun=123596;

void setStatError(TH1D* hist);
bool rejectRunLumi(int run, int lumi);
bool goodBunchCrossing(int run, int bx);

void ntupleViewer(){

  const double beamspot=STRINGbeamspot;

//  gROOT->Reset();
  gStyle->SetOptStat(1111);
  gStyle->SetCanvasBorderMode(0);
  gStyle->SetPadBorderMode(0);

  gStyle->SetCanvasColor(10);
  gStyle->SetPadColor(10);
  gStyle->SetFillColor(10);
  gStyle->SetStatColor(10);
  gStyle->SetTitleFillColor(10);

std::string addMe = "STRINGhistname";

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
TH1D* vertexes_z          =new TH1D( (addMe + std::string("vertexes_z")).c_str()     , (addMe + std::string("vertexes_z")).c_str(),25,-20,20);
TH2D* vertexes_xy         =new TH2D( (addMe + std::string("vertexes_xy")).c_str()    , (addMe + std::string("vertexes_xy")).c_str(),200,-10,10,200,-10,10);
  TH1D* deltaZ_trackPV      =new TH1D( (addMe + std::string("deltaZ_trackPV")).c_str() , (addMe + std::string("deltaZ_trackPV")).c_str(),50,-25,25);
  TH1D* deltaZ_trackPV_ZOOM      =new TH1D( (addMe + std::string("deltaZ_trackPV_ZOOM")).c_str() , (addMe +  std::string("deltaZ_trackPV_ZOOM")).c_str(),100,-3,3);
TH1D* deltaX_trackPV      =new TH1D( (addMe + std::string("deltaX_trackPV")).c_str() , (addMe + std::string("deltaX_trackPV")).c_str(),50,-25,25);
TH1D* deltaX_trackPV_ZOOM      =new TH1D( (addMe + std::string("deltaX_trackPV_ZOOM")).c_str() , (addMe + std::string("deltaX_trackPV_ZOOM")).c_str(),100,-3,3);
  TH1D* deltaY_trackPV      =new TH1D( (addMe + std::string("deltaY_trackPV")).c_str() , (addMe + std::string("deltaY_trackPV")).c_str(),50,-25,25);
  TH1D* deltaY_trackPV_ZOOM      =new TH1D( (addMe + std::string("deltaY_trackPV_ZOOM")).c_str() , (addMe + std::string("deltaY_trackPV_ZOOM")).c_str(),100,-3,3);
TH1D* vertexesNum_NoutTrk =new TH1D( (addMe + std::string("vertexesNum_NoutTrk")).c_str() , (addMe + std::string("vertexesNum_NoutTrk")).c_str(),50,0,50);
TH1D* deltaZ_v1v2         =new TH1D( (addMe + std::string("deltaZ_v1v2")).c_str()    , (addMe + std::string("deltaZ_v1v2")).c_str(),100,0,50);

  TH1D* dZ_All      =new TH1D( (addMe + std::string("dZ_All")).c_str() , (addMe +  std::string("dZ_All")).c_str(),100,-3,3);
  TH1D* d0_All      =new TH1D( (addMe + std::string("d0_All")).c_str() , (addMe + std::string("d0_All")).c_str(),100,-3,3);
TH1D* dZPoint_Ass      =new TH1D( (addMe + std::string("dZPoint_Ass")).c_str() , (addMe +  std::string("dZPoint_Ass")).c_str(),100,-3,3);
TH1D* d0Point_Ass      =new TH1D( (addMe + std::string("d0Point_Ass")).c_str() , (addMe + std::string("d0Point_Ass")).c_str(),100,-3,3);
TH1D* dZ_Ass      =new TH1D( (addMe + std::string("dZ_Ass")).c_str() , (addMe +  std::string("dZ_Ass")).c_str(),100,-3,3);
TH1D* d0_Ass      =new TH1D( (addMe + std::string("d0_Ass")).c_str() , (addMe + std::string("d0_Ass")).c_str(),100,-3,3);
  TH1D* dZPoint_Vtx      =new TH1D( (addMe + std::string("dZPoint_Vtx")).c_str() , (addMe +  std::string("dZPoint_Vtx")).c_str(),100,-3,3);
  TH1D* d0Point_Vtx      =new TH1D( (addMe + std::string("d0Point_Vtx")).c_str() , (addMe + std::string("d0Point_Vtx")).c_str(),100,-3,3);
  TH1D* dZ_Vtx      =new TH1D( (addMe + std::string("dZ_Vtx")).c_str() , (addMe +  std::string("dZ_Vtx")).c_str(),100,-3,3);
  TH1D* d0_Vtx      =new TH1D( (addMe + std::string("d0_Vtx")).c_str() , (addMe + std::string("d0_Vtx")).c_str(),100,-3,3);
TH1D* dZ_Vtx_LARGE      =new TH1D( (addMe + std::string("dZ_Vtx_LARGE")).c_str() , (addMe + std::string("dZ_Vtx_LARGE")).c_str(),500,-15,15);

TH1D* hthrust_z_All            =new TH1D( (addMe + std::string("hthrust_z_All")).c_str()       , (addMe + std::string("thrust_z_All")).c_str(),50,-1.1,1.1);
TH1D* hthrust_z_withVertex     =new TH1D( (addMe + std::string("hthrust_z_withVertex")).c_str(), (addMe + std::string("thrust_y_withVertex")).c_str(),50,-1.1,1.1);
TH1D* hthrust_z_AssVertex      =new TH1D( (addMe + std::string("hthrust_z_AssVertex")).c_str() , (addMe + std::string("thrust_z_AssVertex")).c_str(),50,-1.1,1.1);

TH1D* chargeAsymmetry_All = new TH1D( (addMe + std::string("chargeAsymmetry_All")).c_str() , (addMe + std::string("chargeAsymmetry_All")).c_str(),20,-10,10);
TH1D* chargeAsymmetry_AssVertex = new TH1D( (addMe + std::string("chargeAsymmetry_AssVertex")).c_str() , (addMe + std::string("chargeAsymmetry_AssVertex")).c_str(),20,-10,10);

TH1D* clusterSizeHist_selectedEvents = new TH1D( (addMe + std::string("clusterSizeHist_selectedEvents")).c_str() , (addMe + std::string("clusterSizeHist_selectedEvents")).c_str(),1000,0,1000);
TH1D* clusterSizeHist_allEvents = new TH1D( (addMe + std::string("clusterSizeHist_allEvents")).c_str() , (addMe + std::string("clusterSizeHist_allEvents")).c_str(),1000,0,1000);
//*********************************  

int eventNum;
int runNumber;
int lumiBlock;
int bunchCrossing;

int numVertices;
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
int track_charge[10000];

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

int cluster_size[50000];
int clusterTotal;

  TFile* myFileDATA = TFile::Open("STRINGinputfile");  
  if(myFileDATA==0) std::cout<<"casin, casin, no xe sto file dei!"<<std::endl;
  if((myFileDATA->GetListOfKeys())->First()==0) cout<< "IS VOID" << endl; 

  TDirectory *DATAdir = (TDirectory*)myFileDATA->Get("MyAnalyzer");
  TTree *treeDATA= (TTree*)DATAdir->Get("EventTree");

treeDATA->SetBranchAddress("eventNum",&eventNum);
treeDATA->SetBranchAddress("runNumber",&runNumber);
treeDATA->SetBranchAddress("lumiBlock",&lumiBlock);
treeDATA->SetBranchAddress("bunchCrossing",&bunchCrossing); 

treeDATA->SetBranchAddress("cluster_size",cluster_size); 
treeDATA->SetBranchAddress("clusterTotal",&clusterTotal); 
 
treeDATA->SetBranchAddress("numVertices",&numVertices);
treeDATA->SetBranchAddress("vertex_x",vertex_x);
treeDATA->SetBranchAddress("vertex_y",vertex_y);
treeDATA->SetBranchAddress("vertex_z",vertex_z);
treeDATA->SetBranchAddress("vertex_xError",vertex_xError);
treeDATA->SetBranchAddress("vertex_yError",vertex_yError);
treeDATA->SetBranchAddress("vertex_zError",vertex_zError);
treeDATA->SetBranchAddress("vertex_nOutgoingTracks",vertex_nOutgoingTracks);
 
treeDATA->SetBranchAddress("numTracks",&numTracks);
treeDATA->SetBranchAddress("track_eta",track_eta);
treeDATA->SetBranchAddress("track_phi",track_phi);
treeDATA->SetBranchAddress("track_p",track_p);
treeDATA->SetBranchAddress("track_pt",track_pt);
treeDATA->SetBranchAddress("track_px",track_px);
treeDATA->SetBranchAddress("track_py",track_py);
treeDATA->SetBranchAddress("track_pz",track_pz);
treeDATA->SetBranchAddress("track_d0",track_d0);
treeDATA->SetBranchAddress("track_d0Error",track_d0Error);
treeDATA->SetBranchAddress("track_dz",track_dz);
treeDATA->SetBranchAddress("track_dzError",track_dzError);
treeDATA->SetBranchAddress("track_recHitsSize",track_recHitsSize);
treeDATA->SetBranchAddress("track_chi2",track_chi2);
treeDATA->SetBranchAddress("track_ndof",track_ndof);
treeDATA->SetBranchAddress("track_normalizedChi2",track_normalizedChi2);
treeDATA->SetBranchAddress("track_vx",track_vx);
treeDATA->SetBranchAddress("track_vy",track_vy);
treeDATA->SetBranchAddress("track_vz",track_vz);
treeDATA->SetBranchAddress("track_charge",track_charge);

treeDATA->SetBranchAddress("track_loose",track_loose);
treeDATA->SetBranchAddress("track_tight",track_tight);
treeDATA->SetBranchAddress("track_highPurity",track_highPurity);

treeDATA->SetBranchAddress("vtxAssocTrack_eta",vtxAssocTrack_eta);
treeDATA->SetBranchAddress("vtxAssocTrack_phi",vtxAssocTrack_phi);
treeDATA->SetBranchAddress("vtxAssocTrack_p",vtxAssocTrack_p);
treeDATA->SetBranchAddress("vtxAssocTrack_pt",vtxAssocTrack_pt);
treeDATA->SetBranchAddress("vtxAssocTrack_d0",vtxAssocTrack_d0);
treeDATA->SetBranchAddress("vtxAssocTrack_d0Error",vtxAssocTrack_d0Error);
treeDATA->SetBranchAddress("vtxAssocTrack_dz",vtxAssocTrack_dz);
treeDATA->SetBranchAddress("vtxAssocTrack_dzError",vtxAssocTrack_dzError);
treeDATA->SetBranchAddress("vtxAssocTrack_recHitsSize",vtxAssocTrack_recHitsSize);
treeDATA->SetBranchAddress("vtxAssocTrack_chi2",vtxAssocTrack_chi2);
treeDATA->SetBranchAddress("vtxAssocTrack_ndof",vtxAssocTrack_ndof);
treeDATA->SetBranchAddress("vtxAssocTrack_normalizedChi2",vtxAssocTrack_normalizedChi2);
treeDATA->SetBranchAddress("vtxAssocTrack_vx",vtxAssocTrack_vx);
treeDATA->SetBranchAddress("vtxAssocTrack_vy",vtxAssocTrack_vy);
treeDATA->SetBranchAddress("vtxAssocTrack_vz",vtxAssocTrack_vz);

Double_t thrust_z=0;
Double_t ptot=0;

//**** TRUST_Z histograms *****

int entries = treeDATA->GetEntries();
/*
cout<<"corro su "<<entries<<" entries"<<endl;
for(int eventn=0;eventn<entries;eventn++){
  treeDATA->GetEntry(eventn);

  if (realdata)
   if ( rejectRunLumi(runNum,lumiBlock) ) continue; 

//All
    for(Int_t trackn=0;trackn<numTracks;trackn++){ //tutte le tracce dell'evento
      thrust_z=track_pz[trackn]+thrust_z;
	ptot=ptot+track_p[trackn];
	}//tutte le tracce dell'evento
    if(ptot!=0 && numTracks>0){
      thrust_z=thrust_z/ptot;
      hthrust_z_All->Fill(thrust_z);
      }
    thrust_z=0;
    ptot=0;

//one real vertex
  for(Int_t vertexn=0;vertexn<numVertices;vertexn++){
    if(vertex_z[vertexn]!=beamspot){  //vertice non ricostruito valore del beam spot
      if(fabs(vertex_z[vertexn]-beamspot)<10){
	for(Int_t trackn=0;trackn<numTracks;trackn++){ //tutte le tracce dell'evento
	  thrust_z=track_pz[trackn]+thrust_z;
	  ptot=ptot+track_p[trackn];
	  }//tutte le tracce dell'evento
	} 
      }//vertice non ricostruito valore del beam spot
      break;
    }
  if(ptot!=0 && numTracks>0){
    thrust_z=thrust_z/ptot;
    hthrust_z_withVertex->Fill(thrust_z);
    }
  thrust_z=0;
  ptot=0;


//tracks Assocaiated to vertex
  for(Int_t vertexn=0;vertexn<numVertices;vertexn++){
    if(vertex_z[vertexn]!=beamspot){  //vertice non ricostruito valore del beam spot
      if(fabs(vertex_z[vertexn]-beamspot)<10){
	for(Int_t trackn=0;trackn<numTracks;trackn++){ //tutte le tracce dell'evento
	  if ( fabs(track_vz[trackn]-vertex_z[vertexn])<1. ){//associazione delle tracce al vertice 
	    thrust_z=track_pz[trackn]+thrust_z;
	    ptot=ptot+track_p[trackn];
	    }
	  }//tutte le tracce dell'evento
	} 
      }//vertice non ricostruito valore del beam spot
      break;
    }
  if(ptot!=0 && numTracks>0){
    thrust_z=thrust_z/ptot;
    hthrust_z_AssVertex->Fill(thrust_z);
    }
  thrust_z=0;
  ptot=0;

}
*/

//*** TRACK parameters ****

cout<<"corro su "<<entries<<" entries"<<endl;

for(int eventn=0;eventn<entries;eventn++){
  treeDATA->GetEntry(eventn);

  if(oneRun)
    if(runNumber!=fixedRun) continue;

  if (realdata){
   if ( rejectRunLumi(runNumber,lumiBlock) ) continue; 
   if ( !goodBunchCrossing(runNumber,bunchCrossing) )  continue;
   }

int chargePlusY=0;
int chargeMinusY=0;
/*
//All
  track_multip_All->Fill(numTracks);

  for(Int_t trackn=0;trackn<numTracks;trackn++){ //tutte le tracce dell'evento	
      	  
    eta_All->Fill(track_eta[trackn]);
    phi_All->Fill(track_phi[trackn]);
    pt_All->Fill(track_pt[trackn]);
    normChi2_All->Fill(track_normalizedChi2[trackn]);
    chi2_vs_pT_All->Fill(track_normalizedChi2[trackn],track_pt[trackn]);
    pt_vs_multip_All->Fill(track_pt[trackn],numTracks);
    pt_vs_ndof_All->Fill(track_pt[trackn],track_ndof[trackn]);
    ndof_All->Fill(track_ndof[trackn]);
    if (numTracks>100)    chi2_vs_pT_highMultip_All->Fill(track_normalizedChi2[trackn],track_pt[trackn]);
    else                  chi2_vs_pT_lowMultip_All->Fill(track_normalizedChi2[trackn],track_pt[trackn]);
    dZ_All->Fill( track_dz[trackn] );
    d0_All->Fill( track_d0[trackn] );

    if ( track_py[trackn]>0. )  chargePlusY+=track_charge[trackn];
    else                        chargeMinusY+=track_charge[trackn];

    }//tutte le tracce dell'evento
  
  chargeAsymmetry_All->Fill(chargePlusY+chargeMinusY);


//vertex presence requirement
  for(Int_t vertexn=0;vertexn<numVertices;vertexn++)
    {
    if(vertex_z[vertexn]!=beamspot){  //vertice non ricostruito valore del beam spot
      if(fabs(vertex_z[vertexn]-beamspot)<10){
        track_multip_withVertex->Fill(numTracks);
	for(Int_t trackn=0;trackn<numTracks;trackn++){
      	  
	  //if (track_pt[trackn]>8.) continue;
	  
	  eta_withVertex->Fill(track_eta[trackn]);
	  phi_withVertex->Fill(track_phi[trackn]);
	  pt_withVertex->Fill(track_pt[trackn]);
	  normChi2_withVertex->Fill(track_normalizedChi2[trackn]); 
          chi2_vs_pT_withVertex->Fill(track_normalizedChi2[trackn],track_pt[trackn]);
          pt_vs_multip_withVertex->Fill(track_pt[trackn],numTracks);
          pt_vs_ndof_withVertex->Fill(track_pt[trackn],track_ndof[trackn]);
          ndof_withVertex->Fill(track_ndof[trackn]);
          if (numTracks>100)    chi2_vs_pT_highMultip_withVertex->Fill(track_normalizedChi2[trackn],track_pt[trackn]);
          else                  chi2_vs_pT_lowMultip_withVertex->Fill(track_normalizedChi2[trackn],track_pt[trackn]);

	  dZPoint_Vtx->Fill( (track_vz[trackn]-vertex_z[vertexn])- ( (track_vx[trackn]-vertex_x[vertexn])*track_px[trackn] +  (track_vy[trackn]-vertex_y[vertexn])*track_py[trackn])/(track_pt[trackn]) * track_pz[trackn]/track_pt[trackn] );
	  d0Point_Vtx->Fill( (-(track_vx[trackn]-vertex_x[vertexn])*track_py[trackn]+(track_vy[trackn]-vertex_y[vertexn])*track_px[trackn])/track_pt[trackn] );
	    dZ_Vtx->Fill( track_dz[trackn] );
	    dZ_Vtx_LARGE->Fill( track_dz[trackn] );
	    d0_Vtx->Fill( track_d0[trackn] );
          
	  }
	}
      break; 
      }//vertice non ricostruito valore del beam spot
    }


  chargePlusY=0;
  chargeMinusY=0;

//Vertex Association requirement
  for(Int_t vertexn=0;vertexn<numVertices;vertexn++)
    {
    //std::cout<<"vertices z: "<<vertex_z[vertexn]<<endl;
    if(vertex_z[vertexn]!=beamspot){  //vertice non ricostruito valore del beam spot
      if(fabs(vertex_z[vertexn]-beamspot)<10){
        int fillCounter=0;
	for(Int_t trackn=0;trackn<numTracks;trackn++){
      	  
	  double dZ_cut= (track_vz[trackn]-vertex_z[vertexn])- ( (track_vx[trackn]-vertex_x[vertexn])*track_px[trackn] +  (track_vy[trackn]-vertex_y[vertexn])*track_py[trackn])/(track_pt[trackn]) * track_pz[trackn]/track_pt[trackn] ;
	  double d0_cut= (-(track_vx[trackn]-vertex_x[vertexn])*track_py[trackn]+(track_vy[trackn]-vertex_y[vertexn])*track_px[trackn])/track_pt[trackn] ;
	  
	  if ( fabs(dZ_cut)<0.36 && fabs(d0_cut)<0.18 && track_highPurity[trackn]==1 && track_pt[trackn]>0.29)
	  //if ( fabs(dZ_cut)<0.36 && fabs(d0_cut)<0.18 && track_highPurity[trackn]==1
	  //if ( fabs(dZ_cut)<0.18 && fabs(d0_cut)<0.18 && track_highPurity[trackn]==1 )
	    {
	    fillCounter++;
	    eta_AssVertex->Fill(track_eta[trackn]);
	    phi_AssVertex->Fill(track_phi[trackn]);
	    pt_AssVertex->Fill(track_pt[trackn]);
	    normChi2_AssVertex->Fill(track_normalizedChi2[trackn]);
            chi2_vs_pT_AssVertex->Fill(track_normalizedChi2[trackn],track_pt[trackn]);
            pt_vs_multip_AssVertex->Fill(track_pt[trackn],numTracks);
            pt_vs_ndof_AssVertex->Fill(track_pt[trackn],track_ndof[trackn]);
            ndof_AssVertex->Fill(track_ndof[trackn]);
            if (numTracks>100)    chi2_vs_pT_highMultip_AssVertex->Fill(track_normalizedChi2[trackn],track_pt[trackn]);
            else                  chi2_vs_pT_lowMultip_AssVertex->Fill(track_normalizedChi2[trackn],track_pt[trackn]);
	    
	    dZPoint_Ass->Fill( (track_vz[trackn]-vertex_z[vertexn])- ( (track_vx[trackn]-vertex_x[vertexn])*track_px[trackn] +  (track_vy[trackn]-vertex_y[vertexn])*track_py[trackn])/(track_pt[trackn]) * track_pz[trackn]/track_pt[trackn] );
	    d0Point_Ass->Fill( (-(track_vx[trackn]-vertex_x[vertexn])*track_py[trackn]+(track_vy[trackn]-vertex_y[vertexn])*track_px[trackn])/track_pt[trackn] );
	    dZ_Ass->Fill( track_dz[trackn] );
	    d0_Ass->Fill( track_d0[trackn] );

            if ( track_py[trackn]>0. )  chargePlusY+=track_charge[trackn];
            else                        chargeMinusY+=track_charge[trackn];

	    }
          }
        track_multip_AssVertex->Fill(fillCounter);
        chargeAsymmetry_AssVertex->Fill(chargePlusY+chargeMinusY);
        }
      break; 
      }//vertice non ricostruito valore del beam spot
    }
*/
//cluster properties
  for(Int_t vertexn=0;vertexn<numVertices;vertexn++)
    {
    if(vertex_z[vertexn]!=beamspot){  //vertice non ricostruito valore del beam spot
      if(fabs(vertex_z[vertexn]-beamspot)<10){
        for (int itClust=0;itClust<clusterTotal;itClust++)
	  clusterSizeHist_selectedEvents->Fill(cluster_size[itClust]);
        break;
        }
      }
    }
    
  for(Int_t vertexn=0;vertexn<numVertices;vertexn++)
    for (int itClust=0;itClust<clusterTotal;itClust++)
      clusterSizeHist_allEvents->Fill(cluster_size[itClust]);  
    
}

/*
//***** VERTEX properties *****

int contime=0;
cout<<"corro su "<<entries<<" entries"<<endl;
for(int eventn=0;eventn<entries;eventn++){
  treeDATA->GetEntry(eventn);

  if(oneRun)
    if(runNumber!=fixedRun) continue;

  if (realdata){
   if ( rejectRunLumi(runNumber,lumiBlock) ) continue; 
   if ( !goodBunchCrossing(runNumber,bunchCrossing) )  continue;
   }

  //cout<<runNumber<<"    ";

  int fill;
  if (numVertices==1){
    if (vertex_z[0]==beamspot) fill = 0;
    else                        fill = 1;
    }
  else fill = numVertices;  
  vertexes->Fill(fill);

  if (numVertices>1)
    deltaZ_v1v2->Fill( fabs(vertex_z[0]-vertex_z[1]) );
  
  for(Int_t vertexn=0;vertexn<numVertices;vertexn++){
    if(vertex_z[vertexn]!=beamspot){  //vertice non ricostruito valore del beam spot
	vertexesNum_NoutTrk->Fill(vertex_nOutgoingTracks[vertexn]);
	vertexes_z->Fill(vertex_z[vertexn]);
	vertexes_xy->Fill(vertex_x[vertexn],vertex_y[vertexn]);
    }//vertice non ricostruito valore del beam spot
  }

  for(Int_t vertexn=0;vertexn<numVertices;vertexn++)
    {
    if(vertex_z[0]!=beamspot){  //vertice non ricostruito valore del beam spot
      if(fabs(vertex_z[vertexn]-beamspot)<10){
        contime++;
	for(Int_t trackn=0;trackn<numTracks;trackn++){
	  deltaZ_trackPV->Fill(track_vz[trackn]-vertex_z[vertexn]);  
	  deltaZ_trackPV_ZOOM->Fill(track_vz[trackn]-vertex_z[vertexn]);  
	    deltaX_trackPV->Fill(track_vx[trackn]-vertex_x[vertexn]);  
	    deltaX_trackPV_ZOOM->Fill(track_vx[trackn]-vertex_x[vertexn]);  
	  deltaY_trackPV->Fill(track_vy[trackn]-vertex_y[vertexn]);  
	  deltaY_trackPV_ZOOM->Fill(track_vy[trackn]-vertex_y[vertexn]);  
          }
	break;       
      }
      }//vertice non ricostruito valore del beam spot
    }

}
*/

//cout<<"li gavemo contai: "<<contime<<endl;
double averageMultip = track_multip_AssVertex->GetMean();
TH1D* KNO_scaling = new TH1D("KNO_scaling","KNO_scaling",track_multip_AssVertex->GetNbinsX(),0.,double(track_multip_AssVertex->GetNbinsX())/averageMultip);
  KNO_scaling->GetXaxis()->SetTitle("n / < n >");
  KNO_scaling->GetYaxis()->SetTitle("< n >   P_{n}");
  for (int k=1; k<=KNO_scaling->GetNbinsX(); k++)
    KNO_scaling->SetBinContent(k,averageMultip*track_multip_AssVertex->GetBinContent(k));


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
  setStatError(deltaZ_trackPV);
  setStatError(vertexesNum_NoutTrk);
  setStatError(ndof_All);
  setStatError(ndof_withVertex);
  setStatError(ndof_AssVertex);
  setStatError(track_multip_All);
  setStatError(track_multip_withVertex);
  setStatError(track_multip_AssVertex);

  //****   marker styles  *****  

  eta_withVertex->SetMarkerStyle(STRINGmarkerstyle);
  eta_AssVertex->SetMarkerStyle(STRINGmarkerstyle);
  eta_All->SetMarkerStyle(STRINGmarkerstyle);
  phi_withVertex->SetMarkerStyle(STRINGmarkerstyle);
  phi_AssVertex->SetMarkerStyle(STRINGmarkerstyle);
  phi_All->SetMarkerStyle(STRINGmarkerstyle);
  pt_withVertex->SetMarkerStyle(STRINGmarkerstyle);
  pt_AssVertex->SetMarkerStyle(STRINGmarkerstyle);
  pt_All->SetMarkerStyle(STRINGmarkerstyle);
  normChi2_withVertex->SetMarkerStyle(STRINGmarkerstyle);
  normChi2_AssVertex->SetMarkerStyle(STRINGmarkerstyle);
  normChi2_All->SetMarkerStyle(STRINGmarkerstyle);
  ndof_All->SetMarkerStyle(STRINGmarkerstyle);
  ndof_withVertex->SetMarkerStyle(STRINGmarkerstyle);
  ndof_AssVertex->SetMarkerStyle(STRINGmarkerstyle);
  track_multip_All->SetMarkerStyle(STRINGmarkerstyle);
  track_multip_withVertex->SetMarkerStyle(STRINGmarkerstyle);
  track_multip_AssVertex->SetMarkerStyle(STRINGmarkerstyle);

  vertexes_z->SetMarkerStyle(STRINGmarkerstyle);
  deltaZ_trackPV->SetMarkerStyle(STRINGmarkerstyle);
  vertexesNum_NoutTrk->SetMarkerStyle(STRINGmarkerstyle);

  vertexes->SetMarkerStyle(STRINGmarkerstyle);

  }

  vertexes->GetXaxis()->SetBinLabel(1,"BeamSpot");
  vertexes->GetXaxis()->SetBinLabel(2,"1 Vertex");
  vertexes->GetXaxis()->SetBinLabel(3,"2 Vertices");
  vertexes->GetXaxis()->SetBinLabel(4,"3 Vertices");
  vertexes->GetXaxis()->SetBinLabel(5,"4 Vertices");
  vertexes->GetXaxis()->SetTitle("N^{vtx}/evt");

  deltaZ_v1v2->GetXaxis()->SetTitle("|v^{1}_{z}-v^{1}_{z}|");


//***** saving histos *****

  std::string fileToOpen="STRINGoutfile";  
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
  vertexes_xy->Write();
  deltaZ_trackPV->Write();
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

   KNO_scaling->Write();

   ndof_All->Write();
   ndof_withVertex->Write();
   ndof_AssVertex->Write();
  
  chargeAsymmetry_All->Write();
  chargeAsymmetry_AssVertex->Write();

  clusterSizeHist_selectedEvents->Write();
  clusterSizeHist_allEvents->Write();
  
  outFile.Close();
}


//********  Helpfull functions *******

void setStatError(TH1D* hist){
  for (int bin=1;bin<=hist->GetNbinsX();bin++)
    hist->SetBinError( bin,sqrt(hist->GetBinContent(bin)) );
  }

//********  Event selections   *******

bool rejectRunLumi(int run, int lumi){  
  if ( (run==123592 && (  lumi<3 || lumi>12)) ||
       //(run==123596 && ( (lumi>=2 && lumi<=9) || lumi==67 || lumi==68 ) ) ||
       (run==123596 && lumi<69 ) ||
       (run==123615 &&    lumi<72 ) ||
       (run==123732 && (  lumi<56 || lumi>=62) ) ||
       (run==123815 && (  lumi<7 || lumi>16) ) ||
      !(run==123592 || run==123596 || run==123615 || run==123732 || run==123815 ) )
              return true;
  else        return false;
  
  }

bool goodBunchCrossing(int run, int bx){
  bool goodbunch=false;
  if (run==123596 || run==123615) {
    if ((bx==51) || (bx==2724))  goodbunch=true;  
    }
  if (run==123732) { 
    if ((bx==3487) || (bx==2596))  goodbunch=true;
    }
      
  return goodbunch;

  }
