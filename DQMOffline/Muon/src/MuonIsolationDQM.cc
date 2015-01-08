// -*- C++ -*-
// MuonIsolationDQM.cc
// Package:    Muon Isolation DQM
// Class:      MuonIsolationDQM
// 
/*

 Description: Muon Isolation DQM class

 Implementation: This code will accept a data set and generate plots of
	various quantities relevent to the Muon Isolation module. We will 
	be using the IsoDeposit class, *not* the MuonIsolation struct.
	 
	The sequence of events is... 
 		* initalize statics (which variables to plot, axis limtis, etc.)
 		* run over events
 			* for each event, run over the muons in that event
 				*record IsoDeposit data
 		* transfer data to histograms, profile plots
 		* save histograms to a root file
*/
//#define DEBUG
 
//Class header file
#include "DQMOffline/Muon/interface/MuonIsolationDQM.h"

//System included files
#include <memory>
#include <string>
#include <typeinfo>
#include <utility>

//Root included files
#include "TH1.h"
#include "TH2.h"
#include "TProfile.h"

//Event framework included files
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

//Other included files

//Using declarations
using std::vector;
using std::pair;
using std::string;

using namespace std;
//
//-----------------Constructors---------------------
//

MuonIsolationDQM::MuonIsolationDQM(const edm::ParameterSet& iConfig){
#ifdef DEBUG
  cout << " Initialise Constructor " << endl;
#endif
  requireSTAMuon = iConfig.getUntrackedParameter<bool>("requireSTAMuon");
  requireTRKMuon = iConfig.getUntrackedParameter<bool>("requireTRKMuon");
  requireGLBMuon = iConfig.getUntrackedParameter<bool>("requireGLBMuon");
  dirName = iConfig.getParameter<std::string>("directory");
  
  //--------Initialize tags-------
  theMuonCollectionLabel_   = consumes<reco::MuonCollection>(iConfig.getUntrackedParameter<edm::InputTag>("Global_Muon_Label"));
  theVertexCollectionLabel_ = consumes<reco::VertexCollection>(iConfig.getUntrackedParameter<edm::InputTag>("vertexLabel"));
  
  //-------Initialize Counterse----------------
  nEvents = 0;
  nSTAMuons = 0;   
  nTRKMuons = 0;
  nGLBMuons = 0;
  
  InitStatics();
  
  

  //------"allocate" space for the data vectors-------
  h_1D.resize(NUM_VARS);
  h_2D.resize(NUM_VARS_2D);
  h_1D_NVTX.resize(NUM_VARS_NVTX);

}

//
//----------Destructor-----------------
//
MuonIsolationDQM::~MuonIsolationDQM(){
#ifdef DEBUG
  cout << "Calling destructor" << endl;
#endif
  //Deallocate memory
  
}

//
//------------Methods-----------------------
//
void MuonIsolationDQM::InitStatics(){
#ifdef DEBUG
  cout<< " InitStatistics() " << endl;
#endif 
  //-----------Initialize primitives-----------
  S_BIN_WIDTH = 1.0;//in GeV
  L_BIN_WIDTH = 2.0;//in GeV
  LOG_BINNING_ENABLED = 1;
  NUM_LOG_BINS = 15;
  LOG_BINNING_RATIO = 1.1;
  //ratio by which each bin is wider than the last for log binning
  //i.e.  bin widths are (x), (r*x), (r^2*x), ..., (r^(nbins)*x)
    
  //-------Initialize Titles---------
  title_sam = "";//"[Sample b-jet events] ";
  title_cone = "";//" [in R=0.3 IsoDeposit Cone]";
  //The above two pieces of info will be printed on the title of the whole page,
  //not for each individual histogram
  //  title_cd = "C.D. of ";
  
  //-------"Allocate" memory for vectors
  main_titles.resize(NUM_VARS);
  axis_titles.resize(NUM_VARS);
  names.resize(NUM_VARS);
  param.resize(NUM_VARS, vector<double>(3) );
  isContinuous.resize(NUM_VARS);
 
  titles_2D.resize(NUM_VARS_2D);
  names_2D.resize(NUM_VARS_2D);

  main_titles_NVtxs.resize(NUM_VARS_NVTX);
  axis_titles_NVtxs.resize(NUM_VARS_NVTX);
  names_NVtxs.resize(NUM_VARS_NVTX);

#ifdef DEBUG
  cout << "InitStatistics(): vectors resized " << endl;
#endif
  //-----Titles of the plots-----------
  main_titles[0 ] = "Total Tracker Momentum, #Delta R = 0.3";
  main_titles[1 ] = "Total EM Cal Energy, #Delta R = 0.3";
  main_titles[2 ] = "Total Had Cal Energy, #Delta R = 0.3";
  main_titles[3 ] = "Total HO Cal Energy, #Delta R = 0.3";
  main_titles[4 ] = "Number of Tracker Tracks, #Delta R = 0.3";
  main_titles[5 ] = "Number of Jets around Muon, #Delta R = 0.3";
  main_titles[6 ] = "Tracker p_{T} within veto cone, #Delta R = 0.3";
  main_titles[7 ] = "EM E_{T} within veto cone, #Delta R = 0.3";
  main_titles[8 ] = "Had E_{T} within veto cone, #Delta R = 0.3";
  main_titles[9 ] = "HO E_{T} within veto cone, #Delta R = 0.3";
  main_titles[10] = "Average Momentum per Track, #Delta R = 0.3";
  main_titles[11] = "Weighted Energy, #Delta R = 0.3";

  main_titles[12] = "Total Tracker Momentum, #Delta R = 0.5";
  main_titles[13] = "Total EM Cal Energy, #Delta R = 0.5";
  main_titles[14] = "Total Had Cal Energy, #Delta R = 0.5";
  main_titles[15] = "Total HO Cal Energy, #Delta R = 0.5";
  main_titles[16] = "Number of Tracker Tracks, #Delta R = 0.5";
  main_titles[17] = "Number of Jets around Muon, #Delta R = 0.5";
  main_titles[18] = "Tracker p_{T} within veto cone, #Delta R = 0.5";
  main_titles[19] = "EM E_{T} within veto cone, #Delta R = 0.5";
  main_titles[20] = "Had E_{T} within veto cone, #Delta R = 0.5";
  main_titles[21] = "HO E_{T} within veto cone, #Delta R = 0.5";
  main_titles[22] = "Average Momentum per Track, #Delta R = 0.5";
  main_titles[23] = "Weighted Energy, #Delta R = 0.5";


  main_titles[24 ] = "Relative Detector-Based Isolation, #Delta R = 0.3";
  main_titles[25 ] = "Relative Detector-Based Isolation, #Delta R = 0.5";

  //-----Titles of the plots-----------
  main_titles[26 ] = "Sum PF Charged Hadron Pt, #Delta R = 0.3";
  main_titles[27 ] = "Sum PF Neutral Hadron Pt, #Delta R = 0.3";
  main_titles[28 ] = "Sum PF Photon Et, #Delta R = 0.3";
  main_titles[29 ] = "Sum PF Neutral Hadron Pt (Higher Pt threshold), #Delta R = 0.3";
  main_titles[30 ] = "Sum PF Photon Et (Higher Pt threshold), #Delta R = 0.3";
  main_titles[31 ] = "Sum PF Charged Particles Pt not from PV  (for Pu corrections), #Delta R = 0.3";

 //-----Titles of the plots-----------
  main_titles[32 ] = "Sum PF Charged Hadron Pt, #Delta R = 0.4";
  main_titles[33 ] = "Sum PF Neutral Hadron Pt, #Delta R = 0.4";
  main_titles[34 ] = "Sum PF Photon Et, #Delta R = 0.4";
  main_titles[35 ] = "Sum PF Neutral Hadron Pt (Higher Pt threshold), #Delta R = 0.4";
  main_titles[36 ] = "Sum PF Photon Et (Higher Pt threshold), #Delta R = 0.4";
  main_titles[37 ] = "Sum PF Charged Particles Pt not from PV  (for Pu corrections), #Delta R = 0.4";
 
  main_titles[38 ] = "Relative PF Isolation, #Delta R = 0.3";
  main_titles[39 ] = "Relative PF Isolation, #Delta R = 0.4";
 
  main_titles[40 ] = "Relative PF Isolation (Higher Pt threshold), #Delta R = 0.3";
  main_titles[41 ] = "Relative PF Isolation (Higher Pt threshold), #Delta R = 0.4";

  main_titles[42 ] = "Sum DR Isolation Profile for Charged Hadron,  #Delta R = 0.4";

  main_titles[43 ] = "Sum DR Isolation Profile for Neutral Hadron,  #Delta R = 0.4";
  
  main_titles[44 ] = "Sum DR Isolation Profile for Photon,  #Delta R = 0.4";
 
  main_titles[45 ] = "Mean DR Isolation Profile for Charged Hadron,  #Delta R = 0.4";

  main_titles[46 ] = "Mean DR Isolation Profile for Neutral Hadron,  #Delta R = 0.4";
  
  main_titles[47 ] = "Mean DR Isolation Profile for Photon,  #Delta R = 0.4";
 



#ifdef DEBUG
  cout << "InitStatistics(): main titles 1D DONE " << endl;
#endif
  titles_2D[0] = "Total Tracker Momentum, #Delta R = 0.3";
  titles_2D[1] = "Total EM Cal Energy, #Delta R = 0.3";
  titles_2D[2] = "Total Had Cal Energy, #Delta R = 0.3";
  titles_2D[3] = "Total HO Cal Energy, #Delta R = 0.3";
  titles_2D[4] = "Sum PF Charged Hadron Pt, #Delta R = 0.4";
  titles_2D[5] = "Sum PF Neutral Hadron Pt, #Delta R = 0.4";
  titles_2D[6] = "Sum PF Photon Et, #Delta R = 0.4";
  titles_2D[7] = "Sum PF Charged Pt Not from PV, #Delta R = 0.4";
  titles_2D[8] = "Relative Detector-Based Isolation, #Delta R = 0.4";
  titles_2D[9] = "Relative PF Isolation, #Delta R = 0.4";


  main_titles_NVtxs[0] = "Sum PF Neutral Hadron Pt, #DeltaR = 0.4 ( 0 < N_{Vtx} < 15)";
  main_titles_NVtxs[1] = "Sum PF Neutral Hadron Pt, #DeltaR = 0.4 (15 < N_{Vtx} < 30)";
  main_titles_NVtxs[2] = "Sum PF Neutral Hadron Pt, #DeltaR = 0.4 (30 < N_{Vtx})";
  main_titles_NVtxs[3] = "Sum PF Photon Et, #DeltaR = 0.4 ( 0 < N_{Vtx} < 15)";
  main_titles_NVtxs[4] = "Sum PF Photon Et, #DeltaR = 0.4 (15 < N_{Vtx} < 30)";
  main_titles_NVtxs[5] = "Sum PF Photon Et, #DeltaR = 0.4 (30 < N_{Vtx})";


#ifdef DEBUG
  cout << "InitStatistics(): main titles 2D DONE " << endl;
#endif
 
  //------Titles on the X or Y axis------------
  axis_titles[0 ] = "#Sigma p_{T}   (GeV)";
  axis_titles[1 ] = "#Sigma E_{T}^{EM}   (GeV)";
  axis_titles[2 ] = "#Sigma E_{T}^{Had}   (GeV)";
  axis_titles[3 ] = "#Sigma E_{T}^{HO}   (GeV)";
  axis_titles[4 ] = "N_{Tracks}";
  axis_titles[5 ] = "N_{Jets}";
  axis_titles[6 ] = "#Sigma p_{T,veto} (GeV)";
  axis_titles[7 ] = "#Sigma E_{T,veto}^{EM}   (GeV)";
  axis_titles[8 ] = "#Sigma E_{T,veto}^{Had}   (GeV)";
  axis_titles[9 ] = "#Sigma E_{T,veto}^{HO}   (GeV)";
  axis_titles[10] = "#Sigma p_{T} / N_{Tracks} (GeV)";
  axis_titles[11] = "(1.5) X #Sigma E_{T}^{EM} + #Sigma E_{T}^{Had}";

  axis_titles[12] = "#Sigma p_{T}   (GeV)";
  axis_titles[13] = "#Sigma E_{T}^{EM}   (GeV)";
  axis_titles[14] = "#Sigma E_{T}^{Had}   (GeV)";
  axis_titles[15] = "#Sigma E_{T}^{HO}   (GeV)";
  axis_titles[16] = "N_{Tracks}";
  axis_titles[17] = "N_{Jets}";
  axis_titles[18] = "#Sigma p_{T,veto} (GeV)";
  axis_titles[19] = "#Sigma E_{T,veto}^{EM}   (GeV)";
  axis_titles[20] = "#Sigma E_{T,veto}^{Had}   (GeV)";
  axis_titles[21] = "#Sigma E_{T,veto}^{HO}   (GeV)";
  axis_titles[22] = "#Sigma p_{T} / N_{Tracks} (GeV)";
  axis_titles[23] = "(1.5) X #Sigma E_{T}^{EM} + #Sigma E_{T}^{Had}";

  axis_titles[24] = "(#Sigma Tk p_{T} + #Sigma ECAL p_{T} + #Sigma HCAL p_{T})/ Mu p_{T}  (GeV)";
  axis_titles[25] = "(#Sigma Tk p_{T} + #Sigma ECAL p_{T} + #Sigma HCAL p_{T})/ Mu p_{T}  (GeV)";

  axis_titles[26] = "#Sigma PFCharged p_{T}";
  axis_titles[27] = "#Sigma PFNeutral p_{T}";
  axis_titles[28] = "#Sigma PFPhoton p_{T}";
  axis_titles[29] = "#Sigma PFNeutral p_{T}";
  axis_titles[30] = "#Sigma PFPhoton p_{T}";
  axis_titles[31] = "#Sigma PFCharged p_{T}";

  axis_titles[32] = "#Sigma PFCharged p_{T}";
  axis_titles[33] = "#Sigma PFNeutral p_{T}";
  axis_titles[34] = "#Sigma PFPhoton p_{T}";
  axis_titles[35] = "#Sigma PFNeutral p_{T}";
  axis_titles[36] = "#Sigma PFPhoton p_{T}";
  axis_titles[37] = "#Sigma PFCharged p_{T}";

  axis_titles[38] = "(#Sigma PFCharged p_{T} + #Sigma PFNeutral p_{T} + #Sigma PFPhoton p_{T}) Mu p_{T}  (GeV)";
  axis_titles[39] = "(#Sigma PFCharged p_{T} + #Sigma PFNeutral p_{T} + #Sigma PFPhoton p_{T}) Mu p_{T}  (GeV)";
  axis_titles[40] = "(#Sigma PFCharged p_{T} + #Sigma PFNeutral p_{T} + #Sigma PFPhoton p_{T}) Mu p_{T}  (GeV)";
  axis_titles[41] = "(#Sigma PFCharged p_{T} + #Sigma PFNeutral p_{T} + #Sigma PFPhoton p_{T}) Mu p_{T}  (GeV)";

  axis_titles[42] = "#Sigma DR PFCharged";
  axis_titles[43] = "#Sigma DR PFNeutral";
  axis_titles[44] = "#Sigma DR PFPhoton";

  axis_titles[45] = "Mean DR PFCharged";
  axis_titles[46] = "Mean DR PFNeutral";
  axis_titles[47] = "Mean DR PFPhoton";




  axis_titles_NVtxs[0] = "#Sigma PFNeutral p_{T}";
  axis_titles_NVtxs[1] = "#Sigma PFNeutral p_{T}";
  axis_titles_NVtxs[2] = "#Sigma PFNeutral p_{T}";
  axis_titles_NVtxs[3] = "#Sigma PFPhoton p_{T}";
  axis_titles_NVtxs[4] = "#Sigma PFPhoton p_{T}";
  axis_titles_NVtxs[5] = "#Sigma PFPhoton p_{T}";
  
#ifdef DEBUG
  cout << "InitStatistics(): main titles 1D DONE " << endl;
#endif
  
  //-----------Names given for the root file----------
  names[0 ] = "sumPt_R03";
  names[1 ] = "emEt_R03";
  names[2 ] = "hadEt_R03";
  names[3 ] = "hoEt_R03";
  names[4 ] = "nTracks_R03";
  names[5 ] = "nJets_R03";
  names[6 ] = "trackerVetoPt_R03";
  names[7 ] = "emVetoEt_R03";
  names[8 ] = "hadVetoEt_R03";
  names[9 ] = "hoVetoEt_R03";
  names[10] = "avgPt_R03";
  names[11] = "weightedEt_R03";

  names[12] = "sumPt_R05";
  names[13] = "emEt_R05";
  names[14] = "hadEt_R05";
  names[15] = "hoEt_R05";
  names[16] = "nTracks_R05";
  names[17] = "nJets_R05";
  names[18] = "trackerVetoPt_R05";
  names[19] = "emVetoEt_R05";
  names[20] = "hadVetoEt_R05";
  names[21] = "hoVetoEt_R05";
  names[22] = "avgPt_R05";
  names[23] = "weightedEt_R05";

  names[24] = "relDetIso_R03";
  names[25] = "relDetIso_R05";

  names[26] = "pfChargedPt_R03";
  names[27] = "pfNeutralPt_R03";
  names[28] = "pfPhotonPt_R03";
  names[29] = "pfNeutralPt_HT_R03";
  names[30] = "pfPhotonPt_HT_R03";
  names[31] = "pfChargedPt_PU_R03";

  names[32] = "pfChargedPt_R04";
  names[33] = "pfNeutralPt_R04";
  names[34] = "pfPhotonPt_R04";
  names[35] = "pfNeutralPt_HT_R04";
  names[36] = "pfPhotonPt_HT_R04";
  names[37] = "pfChargedPt_PU_R04";

  names[38] = "relPFIso_R03";
  names[39] = "relPFIso_R04";

  names[40] = "relPFIso_HT_R03";
  names[41] = "relPFIso_HT_R04";
  
  names[42] = "SumDR_PFCharged_R04";
  names[43] = "SumDR_PFNeutral_R04";
  names[44] = "SumDR_PFPhoton_R04";

  names[45] = "MeanDR_PFCharged_R04";
  names[46] = "MeanDR_PFNeutral_R04";
  names[47] = "MeanDR_PFPhoton_R04";
  


#ifdef DEBUG
  cout << "InitStatistics(): names 1D DONE " << endl;
#endif

  names_2D[0] = "SumPt_R03"         ;
  names_2D[1] = "emEt_R03"          ;
  names_2D[2] = "hadEt_R03"         ;
  names_2D[3] = "hoEt_R03"          ;
  names_2D[4] = "pfChargedPt_R04"   ;
  names_2D[5] = "pfNeutralPt_R04"   ;
  names_2D[6] = "pfPhotonPt_R04"    ;
  names_2D[7] = "pfChargedPUPt_R04" ;
  names_2D[8] = "relDetIso_R03"     ;
  names_2D[9] = "relPFIso_R04"      ;
  
#ifdef DEBUG
  cout << "InitStatistics(): names 2D DONE " << endl;
#endif
  
  names_NVtxs[0] = "pfNeutralPt_R04_PV0to15";
  names_NVtxs[1] = "pfNeutralPt_R04_PV15to30";
  names_NVtxs[2] = "pfNeutralPt_R04_PV30toInf";
  names_NVtxs[3] = "pfPhotonPt_R04_PV0to15";
  names_NVtxs[4] = "pfPhotonPt_R04_PV15to30";
  names_NVtxs[5] = "pfPhotonPt_R04_PV30toInf";
    
  //----------Parameters for binning of histograms---------
  //param[var][0] is the number of bins
  //param[var][1] is the low edge of the low bin
  //param[var][2] is the high edge of the high bin
  //
  // maximum value------,
  //                    |
  //                    V                  
  param[0 ][0]= (int)( 20.0/S_BIN_WIDTH); param[0 ][1]=  0.0; param[0 ][2]= param[0 ][0]*S_BIN_WIDTH;
  param[1 ][0]= (int)( 20.0/S_BIN_WIDTH); param[1 ][1]=  0.0; param[1 ][2]= param[1 ][0]*S_BIN_WIDTH;
  param[2 ][0]= (int)( 20.0/S_BIN_WIDTH); param[2 ][1]=  0.0; param[2 ][2]= param[2 ][0]*S_BIN_WIDTH;
  param[3 ][0]=                       20; param[3 ][1]=  0.0; param[3 ][2]=                      2.0;
  param[4 ][0]= 		      16; param[4 ][1]= -0.5; param[4 ][2]=         param[4 ][0]-0.5;
  param[5 ][0]= 		       4; param[5 ][1]= -0.5; param[5 ][2]=         param[5 ][0]-0.5;
  param[6 ][0]= (int)( 40.0/S_BIN_WIDTH); param[6 ][1]=  0.0; param[6 ][2]= param[6 ][0]*S_BIN_WIDTH;
  param[7 ][0]=                       20; param[7 ][1]=  0.0; param[7 ][2]=                     10.0;
  param[8 ][0]= (int)( 20.0/S_BIN_WIDTH); param[8 ][1]=  0.0; param[8 ][2]= param[8 ][0]*S_BIN_WIDTH;
  param[9 ][0]=                       20; param[9 ][1]=  0.0; param[9 ][2]=                      5.0;
  param[10][0]= (int)( 15.0/S_BIN_WIDTH); param[10][1]=  0.0; param[10][2]= param[10][0]*S_BIN_WIDTH;
  param[11][0]= (int)( 20.0/S_BIN_WIDTH); param[11][1]=  0.0; param[11][2]= param[11][0]*S_BIN_WIDTH;

  param[12][0]= (int)( 20.0/S_BIN_WIDTH); param[12][1]=  0.0; param[12][2]= param[12][0]*S_BIN_WIDTH;
  param[13][0]= (int)( 20.0/S_BIN_WIDTH); param[13][1]=  0.0; param[13][2]= param[13][0]*S_BIN_WIDTH;
  param[14][0]= (int)( 20.0/S_BIN_WIDTH); param[14][1]=  0.0; param[14][2]= param[14][0]*S_BIN_WIDTH;
  param[15][0]=                       20; param[15][1]=  0.0; param[15][2]=                      2.0;
  param[16][0]=                       16; param[16][1]= -0.5; param[16][2]=         param[16][0]-0.5;
  param[17][0]=                        4; param[17][1]= -0.5; param[17][2]=         param[17][0]-0.5;
  param[18][0]= (int)( 40.0/S_BIN_WIDTH); param[18][1]=  0.0; param[18][2]= param[18][0]*S_BIN_WIDTH;
  param[19][0]=                       20; param[19][1]=  0.0; param[19][2]=                     10.0;
  param[20][0]= (int)( 20.0/S_BIN_WIDTH); param[20][1]=  0.0; param[20][2]= param[20][0]*S_BIN_WIDTH;
  param[21][0]=                       20; param[21][1]=  0.0; param[21][2]=                      5.0;
  param[22][0]= (int)( 15.0/S_BIN_WIDTH); param[22][1]=  0.0; param[22][2]= param[22][0]*S_BIN_WIDTH;
  param[23][0]= (int)( 20.0/S_BIN_WIDTH); param[23][1]=  0.0; param[23][2]= param[23][0]*S_BIN_WIDTH;

  param[24][0]= 50; param[24][1]=  0.0; param[24][2]= 1.0;
  param[25][0]= 50; param[25][1]=  0.0; param[25][2]= 1.0;


  param[26 ][0]= (int)( 20.0/S_BIN_WIDTH); param[26 ][1]=  0.0; param[26 ][2]= param[26 ][0]*S_BIN_WIDTH;
  param[27 ][0]= (int)( 20.0/S_BIN_WIDTH); param[27 ][1]=  0.0; param[27 ][2]= param[27 ][0]*S_BIN_WIDTH;
  param[28 ][0]= (int)( 20.0/S_BIN_WIDTH); param[28 ][1]=  0.0; param[28 ][2]= param[28 ][0]*S_BIN_WIDTH;
  param[29 ][0]= (int)( 20.0/S_BIN_WIDTH); param[29 ][1]=  0.0; param[29 ][2]= param[29 ][0]*S_BIN_WIDTH;
  param[30 ][0]= (int)( 20.0/S_BIN_WIDTH); param[30 ][1]=  0.0; param[30 ][2]= param[30 ][0]*S_BIN_WIDTH;
  param[31 ][0]= (int)( 20.0/S_BIN_WIDTH); param[31 ][1]=  0.0; param[31 ][2]= param[31 ][0]*S_BIN_WIDTH;

  param[32 ][0]= (int)( 20.0/S_BIN_WIDTH); param[32 ][1]=  0.0; param[32 ][2]= param[32 ][0]*S_BIN_WIDTH;
  param[33 ][0]= (int)( 20.0/S_BIN_WIDTH); param[33 ][1]=  0.0; param[33 ][2]= param[33 ][0]*S_BIN_WIDTH;
  param[34 ][0]= (int)( 20.0/S_BIN_WIDTH); param[34 ][1]=  0.0; param[34 ][2]= param[34 ][0]*S_BIN_WIDTH;
  param[35 ][0]= (int)( 20.0/S_BIN_WIDTH); param[35 ][1]=  0.0; param[35 ][2]= param[35 ][0]*S_BIN_WIDTH;
  param[36 ][0]= (int)( 20.0/S_BIN_WIDTH); param[36 ][1]=  0.0; param[36 ][2]= param[36 ][0]*S_BIN_WIDTH;
  param[37 ][0]= (int)( 20.0/S_BIN_WIDTH); param[37 ][1]=  0.0; param[37 ][2]= param[37 ][0]*S_BIN_WIDTH;

  param[38][0]= 50; param[38][1]=  0.0; param[38][2]= 1.0;
  param[39][0]= 50; param[39][1]=  0.0; param[39][2]= 1.0;

  param[40][0]= 50; param[40][1]=  0.0; param[40][2]= 1.0;
  param[41][0]= 50; param[41][1]=  0.0; param[41][2]= 1.0;

  param[42][0]= 50; param[42][1]=  0.0; param[42][2]= 5;
  param[43][0]= 50; param[43][1]=  0.0; param[43][2]= 5;
  param[44][0]= 50; param[44][1]=  0.0; param[44][2]= 5;

  param[45][0]= 50; param[45][1]=  0.0; param[45][2]= 0.4;
  param[46][0]= 50; param[46][1]=  0.0; param[46][2]= 0.4;
  param[47][0]= 50; param[47][1]=  0.0; param[47][2]= 0.4;
  

  //--------------Is the variable continuous (i.e. non-integer)?-------------
  //---------(Log binning will only be used for continuous variables)--------
  isContinuous[0 ] = 1;
  isContinuous[1 ] = 1;
  isContinuous[2 ] = 1;
  isContinuous[3 ] = 1;
  isContinuous[4 ] = 0;
  isContinuous[5 ] = 0;
  isContinuous[6 ] = 1;
  isContinuous[7 ] = 1;
  isContinuous[8 ] = 1;
  isContinuous[9 ] = 1;
  isContinuous[10] = 1;
  isContinuous[11] = 1;

  isContinuous[12] = 1;
  isContinuous[13] = 1;
  isContinuous[14] = 1;
  isContinuous[15] = 1;
  isContinuous[16] = 0;
  isContinuous[17] = 0;
  isContinuous[18] = 1;
  isContinuous[19] = 1;
  isContinuous[20] = 1;
  isContinuous[21] = 1;
  isContinuous[22] = 1;
  isContinuous[23] = 1;

  isContinuous[24] = 1;
  isContinuous[25] = 1;
  isContinuous[26] = 1;
  isContinuous[27] = 1;
  isContinuous[28] = 1;
  isContinuous[29] = 1;
  isContinuous[30] = 1;
  isContinuous[31] = 1;
  isContinuous[32] = 1;
  isContinuous[33] = 1;
  isContinuous[34] = 1;
  isContinuous[35] = 1;
  isContinuous[36] = 1;
  isContinuous[37] = 1;
  isContinuous[38] = 1;
  isContinuous[39] = 1;
  isContinuous[40] = 1;
  isContinuous[41] = 1;
  isContinuous[42] = 1;
  isContinuous[43] = 1;
  isContinuous[44] = 1;
  isContinuous[45] = 1;
  isContinuous[46] = 1;
  isContinuous[47] = 1;
 

#ifdef DEBUG
  cout << "InitStatistics(): DONE " << endl;
#endif
}


// ------------ method called for each event  ------------
void MuonIsolationDQM::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup){
  
  ++nEvents;
  edm::LogInfo("Tutorial") << "\nInvestigating event #" << nEvents<<"\n";
#ifdef DEBUG
  cout << "[MuonIsolationDQM]: analyze()"<<endl;
#endif

  // Get Muon Collection 
  edm::Handle<reco::MuonCollection> muons;
  iEvent.getByToken(theMuonCollectionLabel_,muons);

#ifdef DEBUG
  cout << "[MuonIsolationDQM]: Number of muons -> " << muons->size() << endl;
#endif
  
  int theMuonData = muons->size();
  h_nMuons->Fill(theMuonData);
#ifdef DEBUG
  cout << "[MuonIsolationDQM]: Vertex is Valid" << endl;
#endif
  
  //Get Vertex Information
  int _numPV = 0;
  edm::Handle<reco::VertexCollection> vertexHandle;
  iEvent.getByToken(theVertexCollectionLabel_, vertexHandle);

  if (vertexHandle.isValid()){
    reco::VertexCollection vertex = *(vertexHandle.product());
    for (reco::VertexCollection::const_iterator v = vertex.begin(); v!=vertex.end(); ++v){
      if (v->isFake())         continue;
      if (v->ndof() < 4)       continue;
      if (fabs(v->z()) > 24.0) continue;
      ++_numPV;
    }
  }

#ifdef DEBUG
  cout << "[MuonIsolationDQM]: Vertex is Valid" << endl;
#endif
  // Get Muon Collection 

  //Fill historgams concerning muon isolation 
  for (reco::MuonCollection::const_iterator muon = muons->begin(); muon!=muons->end(); ++muon){
    if (requireSTAMuon && muon->isStandAloneMuon()) {
      ++nSTAMuons;
      RecordData(*muon);
      FillHistos(_numPV);
    }
    else if (requireTRKMuon && muon->isTrackerMuon()) {
      ++nTRKMuons;
      RecordData(*muon);
      FillHistos(_numPV);
    }
    else if (requireGLBMuon && muon->isGlobalMuon()) {
      ++nGLBMuons;
      RecordData(*muon);
      FillHistos(_numPV);
      FillNVtxHistos(_numPV);
    }
  }
  
}

//---------------Record data for a signle muon's data---------------------
void MuonIsolationDQM::RecordData(const reco::Muon&  muon){
#ifdef DEBUG
  std::cout << "RecordData()" << endl;
#endif 
  float MuPt = muon.pt();
  
  theData[0] = muon.isolationR03().sumPt;
  theData[1] = muon.isolationR03().emEt;
  theData[2] = muon.isolationR03().hadEt;
  theData[3] = muon.isolationR03().hoEt;

  theData[4] = muon.isolationR03().nTracks;
  theData[5] = muon.isolationR03().nJets;
  theData[6] = muon.isolationR03().trackerVetoPt;
  theData[7] = muon.isolationR03().emVetoEt;
  theData[8] = muon.isolationR03().hadVetoEt;
  theData[9] = muon.isolationR03().hoVetoEt;
  
  // make sure nTracks != 0 before filling this one
  if (theData[4] != 0) theData[10] = (double)theData[0] / (double)theData[4];
  else theData[10] = -99;

  theData[11] = 1.5 * theData[1] + theData[2];

  theData[12] = muon.isolationR05().sumPt;
  theData[13] = muon.isolationR05().emEt;
  theData[14] = muon.isolationR05().hadEt;
  theData[15] = muon.isolationR05().hoEt;

  theData[16] = muon.isolationR05().nTracks;
  theData[17] = muon.isolationR05().nJets;
  theData[18] = muon.isolationR05().trackerVetoPt;
  theData[19] = muon.isolationR05().emVetoEt;
  theData[20] = muon.isolationR05().hadVetoEt;
  theData[21] = muon.isolationR05().hoVetoEt;

  // make sure nTracks != 0 before filling this one
  if (theData[16] != 0) theData[22] = (double)theData[12] / (double)theData[16];
  else theData[22] = -99;

  theData[23] = 1.5 * theData[13] + theData[14];

  theData[24] = (theData[0]+theData[1]+theData[2]) / MuPt; 
  theData[25] = (theData[12]+theData[13]+theData[14]) / MuPt; 

  theData[26] = muon.pfIsolationR03().sumChargedHadronPt;
  theData[27] = muon.pfIsolationR03().sumNeutralHadronEt;
  theData[28] = muon.pfIsolationR03().sumPhotonEt; 
  theData[29] = muon.pfIsolationR03().sumNeutralHadronEtHighThreshold;
  theData[30] = muon.pfIsolationR03().sumPhotonEtHighThreshold; 
  theData[31] = muon.pfIsolationR03().sumPUPt;
  
  theData[32] = muon.pfIsolationR04().sumChargedHadronPt;
  theData[33] = muon.pfIsolationR04().sumNeutralHadronEt;
  theData[34] = muon.pfIsolationR04().sumPhotonEt; 
  theData[35] = muon.pfIsolationR04().sumNeutralHadronEtHighThreshold;
  theData[36] = muon.pfIsolationR04().sumPhotonEtHighThreshold; 
  theData[37] = muon.pfIsolationR04().sumPUPt;

  theData[38] = (theData[26] + theData[27] + theData[28]) / MuPt;
  theData[39] = (theData[32] + theData[33] + theData[34]) / MuPt;

  theData[40] = (theData[26] + theData[29] + theData[30]) / MuPt;
  theData[41] = (theData[32] + theData[35] + theData[36]) / MuPt;
  
  theData[42] = muon.pfSumDRIsoProfileR04().sumChargedHadronPt;
  theData[43] = muon.pfSumDRIsoProfileR04().sumNeutralHadronEt;
  theData[44] = muon.pfSumDRIsoProfileR04().sumPhotonEt;
  theData[45] = muon.pfMeanDRIsoProfileR04().sumChargedHadronPt;
  theData[46] = muon.pfMeanDRIsoProfileR04().sumNeutralHadronEt;
  theData[47] = muon.pfMeanDRIsoProfileR04().sumPhotonEt;
  
  //--------------Filling the 2D Histos Data -------- //
  theData2D[0] = muon.isolationR03().sumPt; 
  theData2D[1] = muon.isolationR03().emEt;
  theData2D[2] = muon.isolationR03().hadEt;
  theData2D[3] = muon.isolationR03().hoEt;
  
  theData2D[4] = muon.pfIsolationR04().sumChargedHadronPt;
  theData2D[5] = muon.pfIsolationR04().sumNeutralHadronEt;
  theData2D[6] = muon.pfIsolationR04().sumPhotonEt;
  theData2D[7] = muon.pfIsolationR04().sumPUPt;
  
  theData2D[8] = theData2D[0] + theData2D[1] + theData2D[2] + theData2D[3] / MuPt; //Det RelIso;
  theData2D[9] = theData2D[4] + theData2D[5] + theData2D[6]                / MuPt; //PF  RelIso;

  //-----------Filling the NVTX 1D HISTOS DATA ------------- // 
  theDataNVtx[0] = muon.pfIsolationR04().sumNeutralHadronEt;
  theDataNVtx[1] = theDataNVtx[0];
  theDataNVtx[2] = theDataNVtx[0];
  
  theDataNVtx[3] = muon.pfIsolationR04().sumPhotonEt;
  theDataNVtx[4] = theDataNVtx[3];
  theDataNVtx[5] = theDataNVtx[3];
}
void MuonIsolationDQM::bookHistograms(DQMStore::IBooker & ibooker,
				      edm::Run const & /*iRun*/,
				      edm::EventSetup const & /* iSetup */){
  
  ibooker.cd();
  ibooker.setCurrentFolder(dirName.c_str());

  ibooker.cd();
  ibooker.setCurrentFolder(dirName.c_str());

  //---initialize number of muons histogram---
  h_nMuons = ibooker.book1D("nMuons", title_sam + "Number of Muons", 20, 0., 20.);
  h_nMuons->setAxisTitle("Number of Muons",XAXIS);
  h_nMuons->setAxisTitle("Fraction of Events",YAXIS);
  
  //---Initialize 1D Histograms---
  for(int var = 0; var < NUM_VARS; var++){
    h_1D[var] = ibooker.book1D(names[var], 
			    title_sam + main_titles[var] + title_cone, 
			    (int)param[var][0], 
			    param[var][1], 
			    param[var][2]
			    );
    h_1D[var]->setAxisTitle(axis_titles[var],XAXIS);
    //    GetTH1FromMonitorElement(h_1D[var])->Sumw2();
  }//Finish 1D
  
  //----Initialize 2D Histograms
  for (int var = 0; var<NUM_VARS_2D; var++){
    h_2D[var] = ibooker.bookProfile(names_2D[var] + "_VsPV", titles_2D[var] + " Vs PV", 50, 0.5, 50.5, 20, 0.0, 20.0);
    
    h_2D[var]->setAxisTitle("Number of PV",            XAXIS);
    h_2D[var]->setAxisTitle(titles_2D[var] + " (GeV)" ,YAXIS);
    //    h_2D[var]->getTH1()->Sumw2();
  }
  
  //-----Initialise PU-Binned histograms
  for (int var=0; var<NUM_VARS_NVTX; var++){
    h_1D_NVTX[var] = ibooker.book1D(names_NVtxs[var], main_titles_NVtxs[var], 50, 0.0, 10.0);
    h_1D_NVTX[var]->setAxisTitle(axis_titles_NVtxs[var],XAXIS);
    ///    GetTH1FromMonitorElement(h_1D_NVTX[var])->Sumw2();
  }
}

void MuonIsolationDQM::NormalizeHistos() {
  for(int var=0; var<NUM_VARS; var++){   
    double entries = GetTH1FromMonitorElement(h_1D[var])->GetEntries();
    GetTH1FromMonitorElement(h_1D[var])->Scale(1./entries);
  }
}

void MuonIsolationDQM::FillHistos(int numPV){
#ifdef DEBUG
  cout << "FillHistos( "<< numPV <<" )"<< endl;
#endif  
  
  //----------Fill 1D histograms---------------
  for(int var=0; var<NUM_VARS; var++){  
    h_1D[var]->Fill(theData[var]);
    //    cd_plots[var]->Fill(theData[var]);//right now, this is a regular PDF (just like h_1D)
//OFBin   if (theData[var] > param[var][2]) {
//OFBin     // fill the overflow bin
//OFBin     overFlowBin = (int) param[var][0] + 1;
//OFBin     overFlow = GetTH1FromMonitorElement(h_1D[var])->GetBinContent(overFlowBin);
//OFBin     GetTH1FromMonitorElement(h_1D[var])->SetBinContent(overFlowBin, overFlow + 1);
//OFBin   }
  }//Finish 1D
  
  for (int var=0; var<NUM_VARS_2D; var++){
    h_2D[var]->Fill(numPV,theData2D[var]);
  }
  
#ifdef DEBUG
  cout << "FillHistos( "<< numPV <<" ): DONE"<< endl;
#endif

}
void MuonIsolationDQM::FillNVtxHistos(int PV){
  if (PV <  15)             {  h_1D_NVTX[0]->Fill(theDataNVtx[0]);    h_1D_NVTX[3]->Fill(theDataNVtx[3]); }
  if (PV >= 15 && PV < 30)  {  h_1D_NVTX[1]->Fill(theDataNVtx[1]);    h_1D_NVTX[4]->Fill(theDataNVtx[4]); }
  if (PV >= 30)             {  h_1D_NVTX[2]->Fill(theDataNVtx[2]);    h_1D_NVTX[5]->Fill(theDataNVtx[5]); }
}

TH1* MuonIsolationDQM::GetTH1FromMonitorElement(MonitorElement* me) {
  return me->getTH1();
}

//define this as a plug-in
DEFINE_FWK_MODULE(MuonIsolationDQM);
