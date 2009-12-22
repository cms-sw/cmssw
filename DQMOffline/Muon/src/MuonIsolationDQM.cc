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
 		
 	Easy-peasy.
	
*/
//
// Original Author:  "C. Jess Riedel", UC Santa Barbara
//         Created:  Tue Jul 17 15:58:24 CDT 2007
//

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
#include "DataFormats/TrackReco/interface/Track.h"

//Using declarations
using std::vector;
using std::pair;
using std::string;



//
//-----------------Constructors---------------------
//
MuonIsolationDQM::MuonIsolationDQM(const edm::ParameterSet& iConfig)
{
  
  //  rootfilename = iConfig.getUntrackedParameter<string>("rootfilename"); // comment out for inclusion
  requireSTAMuon = iConfig.getUntrackedParameter<bool>("requireSTAMuon");
  requireTRKMuon = iConfig.getUntrackedParameter<bool>("requireTRKMuon");
  requireGLBMuon = iConfig.getUntrackedParameter<bool>("requireGLBMuon");
  dirName = iConfig.getParameter<std::string>("directory");
  //  subDirName = iConfig.getParameter<std::string>("@module_label");
  
  //  dirName += subDirName;
  
  //--------Initialize tags-------
  Muon_Tag = iConfig.getUntrackedParameter<edm::InputTag>("Global_Muon_Label");
  
  //-------Initialize counters----------------
  nEvents = 0;
  nSTAMuons = 0;   
  nTRKMuons = 0;
  nGLBMuons = 0;
  
  InitStatics();
  
  //Set up DAQ
  dbe = 0;
  dbe = edm::Service<DQMStore>().operator->();
  
  //------"allocate" space for the data vectors-------
  
  /*
    h_1D        is a 2D vector with indices [var][muon#]
    cd_plots    is a 2D vector with indices [var][muon#]  
  */
  //NOTE:the total number of muons and events is initially unknown, 
  //	   so that dimension is not initialized. Hence, theMuonData
  //     needs no resizing.
  
  h_1D.resize    (NUM_VARS);
  /*  cd_plots.resize(NUM_VARS);  */
  
  dbe->cd();
}

//
//----------Destructor-----------------
//
MuonIsolationDQM::~MuonIsolationDQM(){
  
  //Deallocate memory
  
}

//
//------------Methods-----------------------
//
void MuonIsolationDQM::InitStatics(){
  
  //-----------Initialize primatives-----------
  S_BIN_WIDTH = 1.0;//in GeV
  L_BIN_WIDTH = 2.0;//in GeV
  LOG_BINNING_ENABLED = 1;
  NUM_LOG_BINS = 15;
  LOG_BINNING_RATIO = 1.1;//ratio by which each bin is wider than the last for log binning
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

}


// ------------ method called for each event  ------------
void MuonIsolationDQM::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup){
  
  ++nEvents;
  edm::LogInfo("Tutorial") << "\nInvestigating event #" << nEvents<<"\n";
  
  // Get Muon Collection 
  edm::Handle<edm::View<reco::Muon> > muonsHandle; // 
  iEvent.getByLabel(Muon_Tag, muonsHandle);
  
  //Fill event entry in histogram of number of muons
  edm::LogInfo("Tutorial") << "Number of Muons: " << muonsHandle->size();
  theMuonData = muonsHandle->size();
  h_nMuons->Fill(theMuonData);
  
  //Fill historgams concerning muon isolation 
  uint iMuon=0;
  dbe->setCurrentFolder(dirName.c_str());
  for (MuonIterator muon = muonsHandle->begin(); muon != muonsHandle->end(); ++muon, ++iMuon ) {
    //    ++nMuons;
    if (requireSTAMuon && muon->isStandAloneMuon()) {
      ++nSTAMuons;
      RecordData(muon);
      FillHistos();
    }
    else if (requireTRKMuon && muon->isTrackerMuon()) {
      ++nTRKMuons;
      RecordData(muon);
      FillHistos();
    }
    else if (requireGLBMuon && muon->isGlobalMuon()) {
      ++nGLBMuons;
      RecordData(muon);
      FillHistos();
    }
  }
  dbe->cd();
  
}

//---------------Record data for a signle muon's data---------------------
void MuonIsolationDQM::RecordData(MuonIterator muon){
  
  
  theData[0] = muon->isolationR03().sumPt;
  theData[1] = muon->isolationR03().emEt;
  theData[2] = muon->isolationR03().hadEt;
  theData[3] = muon->isolationR03().hoEt;

  theData[4] = muon->isolationR03().nTracks;
  theData[5] = muon->isolationR03().nJets;
  theData[6] = muon->isolationR03().trackerVetoPt;
  theData[7] = muon->isolationR03().emVetoEt;
  theData[8] = muon->isolationR03().hadVetoEt;
  theData[9] = muon->isolationR03().hoVetoEt;
  
  // make sure nTracks != 0 before filling this one
  if (theData[4] != 0) theData[10] = (double)theData[0] / (double)theData[4];
  else theData[10] = -99;

  theData[11] = 1.5 * theData[1] + theData[2];

  theData[12] = muon->isolationR05().sumPt;
  theData[13] = muon->isolationR05().emEt;
  theData[14] = muon->isolationR05().hadEt;
  theData[15] = muon->isolationR05().hoEt;

  theData[16] = muon->isolationR05().nTracks;
  theData[17] = muon->isolationR05().nJets;
  theData[18] = muon->isolationR05().trackerVetoPt;
  theData[19] = muon->isolationR05().emVetoEt;
  theData[20] = muon->isolationR05().hadVetoEt;
  theData[21] = muon->isolationR05().hoVetoEt;

  // make sure nTracks != 0 before filling this one
  if (theData[16] != 0) theData[22] = (double)theData[12] / (double)theData[16];
  else theData[22] = -99;

  theData[23] = 1.5 * theData[13] + theData[14];

}

// ------------ method called once each job just before starting event loop  ------------
void 
MuonIsolationDQM::beginJob(void)
{
  
  edm::LogInfo("Tutorial") << "\n#########################################\n\n"
			   << "Lets get started! " 
			   << "\n\n#########################################\n";
  dbe->setCurrentFolder(dirName.c_str());
  InitHistos();
  dbe->cd();
  
}

// ------------ method called once each job just after ending the event loop  ------------
void 
MuonIsolationDQM::endJob() {
  
  // check if ME still there (and not killed by MEtoEDM for memory saving)
  if( dbe )
    {
      // check existence of first histo in the list
      if (! dbe->get(dirName+"/nMuons")) return;
    }
  else
    return;

  edm::LogInfo("Tutorial") << "\n#########################################\n\n"
			   << "Total Number of Events: " << nEvents
			   << "\n\n#########################################\n"
			   << "\nInitializing Histograms...\n";
  
  edm::LogInfo("Tutorial") << "\nIntializing Finished.  Filling...\n";
  NormalizeHistos();
  edm::LogInfo("Tutorial") << "\nFilled.  Saving...\n";
  //  dbe->save(rootfilename); // comment out for incorporation
  edm::LogInfo("Tutorial") << "\nSaved.  Peace, homie, I'm out.\n";
  
}

void MuonIsolationDQM::InitHistos(){
  
  //---initialize number of muons histogram---
  h_nMuons = dbe->book1D("nMuons", title_sam + "Number of Muons", 20, 0., 20.);
  h_nMuons->setAxisTitle("Number of Muons",XAXIS);
  h_nMuons->setAxisTitle("Fraction of Events",YAXIS);
  
  
  //---Initialize 1D Histograms---
  for(int var = 0; var < NUM_VARS; var++){
    h_1D[var] = dbe->book1D(
			    names[var], 
			    title_sam + main_titles[var] + title_cone, 
			    (int)param[var][0], 
			    param[var][1], 
			    param[var][2]
			    );
    /*    cd_plots[var] = dbe->book1D(
	  names[var] + "_cd", 
	  title_sam + title_cd + main_titles[var] + title_cone, 
	  (int)param[var][0], 
	  param[var][1], 
	  param[var][2]
	  );
    */    

    h_1D[var]->setAxisTitle(axis_titles[var],XAXIS);
    //    h_1D[var]->setAxisTitle("Fraction of Muons",YAXIS);
    GetTH1FromMonitorElement(h_1D[var])->Sumw2();
    
    /*    cd_plots[var]->setAxisTitle(axis_titles[var],XAXIS);
	  cd_plots[var]->setAxisTitle("Fraction of Muons",YAXIS);
	  GetTH1FromMonitorElement(cd_plots[var])->Sumw2();
    */
    
  }//Finish 1D
  
  //avg pT not defined for zero tracks.
  //MonitorElement is inflxible and won't let me change the
  //number of bins!  I guess all I'm doing here is changing 
  //range of the x axis when it is printed, not the actual
  //bins that are filled
  //  p_2D[4][9]->setAxisRange(0.5,15.5,XAXIS);
  
}

void MuonIsolationDQM::NormalizeHistos() {
  for(int var=0; var<NUM_VARS; var++){   
    //turn cd_plots into CDF's
    //underflow -> bin #0.  overflow -> bin #(nbins+1)
    //0th bin doesn't need changed
    
    double entries = GetTH1FromMonitorElement(h_1D[var])->GetEntries();
    
    /*    int n_max = int(param[var][0])+1;
	  for(int n=1; n<=n_max; ++n){
	  cd_plots[var]->setBinContent(n, cd_plots[var]->getBinContent(n) + cd_plots[var]->getBinContent(n-1)); //Integrate.
	  }
    */
    //----normalize------
    /*    if (requireCombinedMuon) {
	  GetTH1FromMonitorElement(h_1D[var])->Scale(1./entries);
	  GetTH1FromMonitorElement(cd_plots[var])->Scale(1./entries);
	  }
	  else {
    */
    GetTH1FromMonitorElement(h_1D[var])->Scale(1./entries);
    //    GetTH1FromMonitorElement(cd_plots[var])->Scale(1./entries);    
  }
}

void MuonIsolationDQM::FillHistos() {
  
  int overFlowBin;
  double overFlow = 0;
  
  //----------Fill 1D histograms---------------
  for(int var=0; var<NUM_VARS; var++){  
    h_1D[var]->Fill(theData[var]);
    //    cd_plots[var]->Fill(theData[var]);//right now, this is a regular PDF (just like h_1D)
    if (theData[var] > param[var][2]) {
      // fill the overflow bin
      overFlowBin = (int) param[var][0] + 1;
      overFlow = GetTH1FromMonitorElement(h_1D[var])->GetBinContent(overFlowBin);
      GetTH1FromMonitorElement(h_1D[var])->SetBinContent(overFlowBin, overFlow + 1);
    }
  }//Finish 1D
  
}

TH1* MuonIsolationDQM::GetTH1FromMonitorElement(MonitorElement* me) {
  return me->getTH1();
}

//define this as a plug-in
DEFINE_FWK_MODULE(MuonIsolationDQM);
