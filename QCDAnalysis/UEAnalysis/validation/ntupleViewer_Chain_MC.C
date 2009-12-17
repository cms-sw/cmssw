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
#include "dati_MC.h"

void setStatError(TH1D* hist);
bool rejectRunLumi(int run, int lumi);

void ntupleViewer_Chain_MC(){

  const double beamspot=-0.005;

//  gROOT->Reset();
  gStyle->SetOptStat(1111);
  gStyle->SetCanvasBorderMode(0);
  gStyle->SetPadBorderMode(0);

  gStyle->SetCanvasColor(10);
  gStyle->SetPadColor(10);
  gStyle->SetFillColor(10);
  gStyle->SetStatColor(10);
  gStyle->SetTitleFillColor(10);

std::string addMe = "MC_";

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

//*******  partonic analysis *******  

TH1D* eta_All_HC             =new TH1D( (addMe + std::string("eta_All_HC")).c_str()        , (addMe + std::string("eta_All_HC")).c_str(),25,-3,3);
TH1D* eta_AssVertex_HC       =new TH1D( (addMe + std::string("eta_AssVertex_HC")).c_str()  , (addMe + std::string("eta_AssVertex_HC")).c_str(),25,-3,3);
TH1D* pt_All_HC              =new TH1D( (addMe + std::string("pt_All_HC")).c_str()         , (addMe + std::string("pt_All_HC")).c_str(),400,0.,40);
TH1D* pt_AssVertex_HC        =new TH1D( (addMe + std::string("pt_AssVertex_HC")).c_str()   , (addMe + std::string("pt_AssVertex_HC")).c_str(),400,0.,40);
  TH1D* track_multip_All_HC       =new TH1D( (addMe + std::string("track_multip_All_HC")).c_str()       , (addMe + std::string("track_multip_All_HC")).c_str(),201,0,200);
  TH1D* track_multip_AssVertex_HC =new TH1D( (addMe + std::string("track_multip_AssVertex_HC")).c_str() , (addMe + std::string("track_multip_AssVertex_HC")).c_str(),201,0,200);

TH1D* eta_All_DD             =new TH1D( (addMe + std::string("eta_All_DD")).c_str()        , (addMe + std::string("eta_All_DD")).c_str(),25,-3,3);
TH1D* eta_AssVertex_DD       =new TH1D( (addMe + std::string("eta_AssVertex_DD")).c_str()  , (addMe + std::string("eta_AssVertex_DD")).c_str(),25,-3,3);
TH1D* pt_All_DD              =new TH1D( (addMe + std::string("pt_All_DD")).c_str()         , (addMe + std::string("pt_All_DD")).c_str(),400,0.,40);
TH1D* pt_AssVertex_DD        =new TH1D( (addMe + std::string("pt_AssVertex_DD")).c_str()   , (addMe + std::string("pt_AssVertex_DD")).c_str(),400,0.,40);
  TH1D* track_multip_All_DD       =new TH1D( (addMe + std::string("track_multip_All_DD")).c_str()       , (addMe + std::string("track_multip_All_DD")).c_str(),201,0,200);
  TH1D* track_multip_AssVertex_DD =new TH1D( (addMe + std::string("track_multip_AssVertex_DD")).c_str() , (addMe + std::string("track_multip_AssVertex_DD")).c_str(),201,0,200);

TH1D* eta_All_SD             =new TH1D( (addMe + std::string("eta_All_SD")).c_str()        , (addMe + std::string("eta_All_SD")).c_str(),25,-3,3);
TH1D* eta_AssVertex_SD       =new TH1D( (addMe + std::string("eta_AssVertex_SD")).c_str()  , (addMe + std::string("eta_AssVertex_SD")).c_str(),25,-3,3);
TH1D* pt_All_SD              =new TH1D( (addMe + std::string("pt_All_SD")).c_str()         , (addMe + std::string("pt_All_SD")).c_str(),400,0.,40);
TH1D* pt_AssVertex_SD        =new TH1D( (addMe + std::string("pt_AssVertex_SD")).c_str()   , (addMe + std::string("pt_AssVertex_SD")).c_str(),400,0.,40);
  TH1D* track_multip_All_SD       =new TH1D( (addMe + std::string("track_multip_All_SD")).c_str()       , (addMe + std::string("track_multip_All_SD")).c_str(),201,0,200);
  TH1D* track_multip_AssVertex_SD =new TH1D( (addMe + std::string("track_multip_AssVertex_SD")).c_str() , (addMe + std::string("track_multip_AssVertex_SD")).c_str(),201,0,200);


//*********************************  

  TChain fChain ("MyAnalyzer/EventTree") ;
  fChain.Add("MCntuple_Summer09-D6T_STARTUP3X_V8I_900GeV-v2/res/*root");
  dati Input(&fChain);
  int entries = fChain.GetEntries();

//*******  TRUST_Z histograms *****
//deleted
//

//*******  TRACK parameters  ******

cout<<"corro su "<<entries<<" entries"<<endl;

for(int eventn=0;eventn<entries;eventn++){
  fChain.GetEntry(eventn);

//id_process test

int unClassified=0;
vector<int> unClassifiedId;
if ( !(Input.genEventScale==11 ||  Input.genEventScale==12 ||   Input.genEventScale==13 ||
       Input.genEventScale==28 ||  Input.genEventScale==53 ||   Input.genEventScale==68 ||
       Input.genEventScale==92 ||Input.genEventScale==93 || Input.genEventScale==94 ) )
  {
  unClassified++;
  bool alreadyFound=false;
  for( int i=0;i<unClassifiedId.size();i++)
      if (Input.genEventScale==unClassifiedId[i]) alreadyFound=true;
      
  if(!alreadyFound){
    unClassifiedId.push_back(runNumber);
    } 
  }


  int chargePlusY=0;
  int chargeMinusY=0;
  
  int trackCounter_HC=0;
  int trackCounter_SD=0;
  int trackCounter_DD=0;

//All
  track_multip_All->Fill(Input.numTracks);
  for(Int_t trackn=0;trackn<Input.numTracks;trackn++){ //tutte le tracce dell'evento	
      	  
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

    if ( Input.track_py[trackn]>0. )  chargePlusY+=Input.track_charge[trackn];
    else                              chargeMinusY+=Input.track_charge[trackn];

    //partonic analysis

    if (Input.genEventScale==11 ||  Input.genEventScale==12 ||   Input.genEventScale==13 ||
      Input.genEventScale==28 ||  Input.genEventScale==53 ||   Input.genEventScale==68)
      {//HC
      trackCounter_HC++;
      eta_All_HC->Fill(Input.track_eta[trackn]);
      pt_All_HC->Fill(Input.track_pt[trackn]);
     }
    else if (Input.genEventScale==92 ||Input.genEventScale==93)
      {//SD
      trackCounter_SD++;
      eta_All_SD->Fill(Input.track_eta[trackn]);
      pt_All_SD->Fill(Input.track_pt[trackn]);
      }
    else if (Input.genEventScale==94 )
      {//DD
      trackCounter_DD++;
      eta_All_DD->Fill(Input.track_eta[trackn]);
      pt_All_DD->Fill(Input.track_pt[trackn]);
      }

    }//tutte le tracce dell'evento
 
  track_multip_All_HC->Fill(trackCounter_HC);
  track_multip_All_SD->Fill(trackCounter_SD);
  track_multip_All_DD->Fill(trackCounter_DD);
  
  chargeAsymmetry_All->Fill(chargePlusY+chargeMinusY);


//vertex presence requirement
  for(Int_t vertexn=0;vertexn<Input.numVertices;vertexn++)
    {
    if(Input.vertex_z[vertexn]!=beamspot){  //vertice non ricostruito valore del beam spot
      if(fabs(Input.vertex_z[vertexn]-beamspot)<10){
        track_multip_withVertex->Fill(Input.numTracks);
	for(Int_t trackn=0;trackn<Input.numTracks;trackn++){
      	  
	  //if (track_pt[trackn]>8.) continue;
	  
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


//Vertex Association requirement

  chargePlusY=0;
  chargeMinusY=0;
  trackCounter_HC=0;
  trackCounter_SD=0;
  trackCounter_DD=0;

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
	  if ( fabs(dZ_cut)<0.36 && fabs(d0_cut)<0.18 && 
	       Input.track_highPurity[trackn]==1 && Input.track_pt[trackn]>0.29)
	    {
	    fillCounter++;
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

            if ( Input.track_py[trackn]>0. )  chargePlusY+=Input.track_charge[trackn];
            else                        chargeMinusY+=Input.track_charge[trackn];

            //partonic analysis

            if (Input.genEventScale==11 ||  Input.genEventScale==12 ||   Input.genEventScale==13 ||
                Input.genEventScale==28 ||  Input.genEventScale==53 ||   Input.genEventScale==68)
              {//HC
	      trackCounter_HC++;
              eta_AssVertex_HC->Fill(Input.track_eta[trackn]); 
              pt_AssVertex_HC->Fill(Input.track_pt[trackn]);
              }
            else if (Input.genEventScale==92 ||Input.genEventScale==93)
              {//SD
	      trackCounter_SD++;
              eta_AssVertex_SD->Fill(Input.track_eta[trackn]);
              pt_AssVertex_SD->Fill(Input.track_pt[trackn]);
              }
            else if (Input.genEventScale==94 )
              {//DD
	      trackCounter_DD++;
              eta_AssVertex_DD->Fill(Input.track_eta[trackn]);
              pt_AssVertex_DD->Fill(Input.track_pt[trackn]);
	      }

	    }
          }
	  
        track_multip_AssVertex->Fill(fillCounter);
        track_multip_AssVertex_HC->Fill(trackCounter_HC);
        track_multip_AssVertex_SD->Fill(trackCounter_SD);
        track_multip_AssVertex_DD->Fill(trackCounter_DD);

        chargeAsymmetry_AssVertex->Fill(chargePlusY+chargeMinusY);
	}
      break; 
      }//vertice non ricostruito valore del beam spot
    }

}

  if (unClassifiedId.size()!=0){
    cout<<"cio' dei, e sti muli do che te li meti, ah?!"<<endl;
      for (int id=0;id<unClassifiedId.size();id++)
        cout<<unClassifiedId[id]<<" ";
    cout<<endl;
    }
  
//***** VERTEX properties *****

int contime=0;
int eventsWithRealVertex=0;
int eventsWithRealGoodVertex0;
cout<<"corro su "<<entries<<" entries"<<endl;
for(int eventn=0;eventn<entries;eventn++){
  fChain.GetEntry(eventn);

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

cout<<"eventi con vertice reale e buono, li gavemo contai: "<<contime<<endl;

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
  deltaZ_trackPV->SetMarkerStyle(22);
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

  std::string fileToOpen="histoMC_fromChain_addpTcutAndGenLevelObservations.root";  
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

//   KNO_scaling->Write();

   ndof_All->Write();
   ndof_withVertex->Write();
   ndof_AssVertex->Write();
  
  chargeAsymmetry_All->Write();
  chargeAsymmetry_AssVertex->Write();

  eta_All_HC->Write();
  eta_AssVertex_HC->Write();
  pt_All_HC->Write();
  pt_AssVertex_HC->Write();
  track_multip_All_HC->Write();
  track_multip_AssVertex_HC->Write();
  eta_All_DD->Write();
  eta_AssVertex_DD->Write();
  pt_All_DD->Write();
  pt_AssVertex_DD->Write();
  track_multip_All_DD->Write();
  track_multip_AssVertex_DD->Write();
  eta_All_SD->Write();
  eta_AssVertex_SD->Write();
  pt_All_SD->Write();
  pt_AssVertex_SD->Write();
  track_multip_All_SD->Write();
  track_multip_AssVertex_SD->Write();
  
  outFile.Close();
}


//********  Helpfull functions *******

void setStatError(TH1D* hist){
  for (int bin=1;bin<=hist->GetNbinsX();bin++)
    hist->SetBinError( bin,sqrt(hist->GetBinContent(bin)) );
  }
