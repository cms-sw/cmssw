///////////////////////////////////////////////////////////////////////////////
//
// Author M. De Mattia demattia@pd.infn.it
// date 21/5/2007
//
// Compare data with simulation
//
///////////////////////////////////////////////////////////////////////////////

#include "TH1F.h"
#include "TH2F.h"
#include "TProfile.h"
#include "TString.h"
#include "TFile.h"
#include "TStyle.h"
#include "TChain.h"
#include "TCanvas.h"
#include "TPad.h"
#include "TLegend.h"
#include <cmath>
#include <iostream>
#include <fstream>
#include <vector>

// This is needed to avoid errors from CINT
#if !defined(__CINT__) && !defined(__MAKECINT__)
#include "AnalysisExamples/AnalysisObjects/interface/AnalyzedTrack.h"
#include "AnalysisExamples/AnalysisObjects/interface/AnalyzedCluster.h"
#endif

// The number indicates the starting array lenght
TObjArray Hlist(0);

void BookHistos () {

  //  Hlist.Add( new TH1F( "Theta", "Theta", 100, -180, 180 );

}

void bookMulti( const std::string HISTONAME,
           const std::string HISTOTITLE,
           unsigned int BINNUM,
	   float FIRSTBIN,
	   float LASTBIN,
	   int TOTLAYERNUM ) {

  std::string HistoName;
  std::string HistoTitle;
  std::string Layer;
  std::stringstream layernum;

  for (int i=1; i<=TOTLAYERNUM; ++i) {
    layernum << i;
    HistoName = HISTONAME + "_L" + layernum.str();
    HistoTitle = HISTOTITLE + " layer " + layernum.str();

    Hlist.Add( new TH1F(HistoName.c_str(),HistoTitle.c_str(),BINNUM,FIRSTBIN,LASTBIN) );

    // The first two layers of TIB and TOB are stereo

    // Change this for TID and TEC
    // ---------------------------

    if ( i < 3 ) {
      HistoName += "S";
      HistoTitle += " Stereo";
      Hlist.Add( new TH1F(HistoName.c_str(),HistoTitle.c_str(),BINNUM,FIRSTBIN,LASTBIN) );
    }
    // Empty the layernum stringstream
    layernum.str("");
  }
}

int FWLite () {

  using namespace std;
  using namespace edm;
  using namespace anaobj;

  TFile *hfile;
  TChain events("Events");

  TString type_;

  hfile = dynamic_cast<TFile*>(gROOT->FindObject("hdata.root")); if (hfile) hfile->Close();
  hfile = new TFile("hdata.root","RECREATE","data histograms");

  // Input files

  // Room temperature
  //     events.Add("rfio:/castor/cern.ch/user/d/demattia/AnaObj_v7/FNAL/AnaObjMaker_TIF_run0009261_1.root");
  //     events.Add("rfio:/castor/cern.ch/user/d/demattia/AnaObj_v7/FNAL/AnaObjMaker_TIF_run0009264_1.root");
  //     events.Add("rfio:/castor/cern.ch/user/d/demattia/AnaObj_v7/FNAL/AnaObjMaker_TIF_run0009269_1.root");
  //     events.Add("rfio:/castor/cern.ch/user/d/demattia/AnaObj_v7/FNAL/AnaObjMaker_TIF_run0009270_1.root");
  //     events.Add("rfio:/castor/cern.ch/user/d/demattia/AnaObj_v7/FNAL/AnaObjMaker_TIF_run0009273_1.root");

  // 10 degrees
  events.Add("rfio:/castor/cern.ch/user/d/demattia/AnaObj_v7/FNAL/AnaObjMaker_TIF_run00010215_1.root");
  //   events.Add("rfio:/castor/cern.ch/user/d/demattia/AnaObj_v7/FNAL/AnaObjMaker_TIF_run00010238_1.root");
  //   events.Add("rfio:/castor/cern.ch/user/d/demattia/AnaObj_v7/FNAL/AnaObjMaker_TIF_run00010250_1.root");
  //   events.Add("rfio:/castor/cern.ch/user/d/demattia/AnaObj_v7/FNAL/AnaObjMaker_TIF_run00010252_1.root");
  //   events.Add("rfio:/castor/cern.ch/user/d/demattia/AnaObj_v7/FNAL/AnaObjMaker_TIF_run00010254_1.root");
  //   events.Add("rfio:/castor/cern.ch/user/d/demattia/AnaObj_v7/FNAL/AnaObjMaker_TIF_run00010613_1.root");
  //   events.Add("rfio:/castor/cern.ch/user/d/demattia/AnaObj_v7/FNAL/AnaObjMaker_TIF_run00010614_1.root");
  //   events.Add("rfio:/castor/cern.ch/user/d/demattia/AnaObj_v7/FNAL/AnaObjMaker_TIF_run00010616_1.root");
  //   events.Add("rfio:/castor/cern.ch/user/d/demattia/AnaObj_v7/FNAL/AnaObjMaker_TIF_run00010625_1.root");
  //   events.Add("rfio:/castor/cern.ch/user/d/demattia/AnaObj_v7/FNAL/AnaObjMaker_TIF_run00010627_1.root");
  //   events.Add("rfio:/castor/cern.ch/user/d/demattia/AnaObj_v7/FNAL/AnaObjMaker_TIF_run00010628_1.root");
  //   events.Add("rfio:/castor/cern.ch/user/d/demattia/AnaObj_v7/FNAL/AnaObjMaker_TIF_run00010630_1.root");
  //   events.Add("rfio:/castor/cern.ch/user/d/demattia/AnaObj_v7/FNAL/AnaObjMaker_TIF_run00010631_1.root");
  //   events.Add("rfio:/castor/cern.ch/user/d/demattia/AnaObj_v7/FNAL/AnaObjMaker_TIF_run00010635_1.root");
  //   events.Add("rfio:/castor/cern.ch/user/d/demattia/AnaObj_v7/FNAL/AnaObjMaker_TIF_run00010636_1.root");
  //   events.Add("rfio:/castor/cern.ch/user/d/demattia/AnaObj_v7/FNAL/AnaObjMaker_TIF_run00010680_1.root");

  //   events.Add("rfio:/castor/cern.ch/user/d/demattia/AnaObj_v7/FNAL/AnaObjMaker_TIF_run00010634_1.root");

  //set the buffers for the branches
  // AnalyzedClusters
  std::vector<AnalyzedCluster> v_anaclu;
  TBranch* anaclu_B;
  events.SetBranchAddress("anaobjAnalyzedClusters_modAnaObjProducer__TIFAnaObjProducer.obj",&v_anaclu,&anaclu_B);
  // AnalyzedTracks
  std::vector<AnalyzedTrack> v_anatk;
  TBranch* anatk_B;
  events.SetBranchAddress("anaobjAnalyzedTracks_modAnaObjProducer__TIFAnaObjProducer.obj",&v_anatk,&anatk_B);

  // Do not define them, otherwise it will not work
  // ----------------------------------------------
  //  anaobj::AnalyzedClusterCollection v_anaclu;
  //   anaobj::AnalyzedClusterRef anacluRef;
  //   anaobj::AnalyzedClusterRefVector anacluRefVec;

  //  anaobj::AnalyzedTrackCollection v_anatk;
  //   anaobj::AnalyzedTrackRef anatkRef;
  //   anaobj::AnalyzedTrackRefVector anatkRefVec;
  // ----------------------------------------------


  // Setting overflow, underflow and other options
  gStyle->SetOptStat("emrou");

  //histogram per number of tracks 
  TH1F* nhit = new TH1F("nhit"+type_,"Numhit",20, 0, 20);
  TH1F* normchi2 = new TH1F("normchi2"+type_,"normalized chi2",50, 0, 20);
  TProfile* normchi2_vs_hitnum = new TProfile("chi2"+type_,"chi2 vs hit number",50, 0, 20);
  TH1F* nclu_on = new TH1F("nclu_on"+type_,"Numclu on",20, 0, 20);
  TH1F* tk_eta = new TH1F("tk_eta"+type_,"track eta",100, -5, 4);
  TH1F* tk_phi = new TH1F("tk_phi"+type_,"track phi",100, -3.14, 3.14);

  int numberofclusters = 0;
  int numberoftracks = 0;
  
  //  loop over the events
  for( unsigned int index = 0;
       index < events.GetEntries();
       ++index) {


    // Call this once to load the events
//     events.GetEntries();
//     for( unsigned int index = 0;
// 	 index < 10;
// 	 ++index) {


    //need to call SetAddress since TBranch's change for each file read
    anaclu_B->SetAddress(&v_anaclu);
    anaclu_B->GetEntry(index);
    anatk_B->SetAddress(&v_anatk);
    anatk_B->GetEntry(index);
    events.GetEntry(index,0);

    //now can access data

    std::cout << "Event = " << index << std::endl;
    numberofclusters = v_anaclu.size();
    numberoftracks = v_anatk.size();
    std::cout <<"Number of clusters = "<<numberofclusters<<std::endl;
    std::cout <<"Number of tracks = "<<numberoftracks<<std::endl;

    // Consider only events with one track
    //     if(numberofclusters < 80) {
    //       if ( numberoftracks <= 1) {

    // Loop on all the analyzed tracks
    for ( unsigned int anatk_iter = 0; anatk_iter < v_anatk.size(); ++anatk_iter ) { 
      // Take the Track
      AnalyzedTrack Track( v_anatk[anatk_iter] );

      // if (Track.normchi2 < 20) {

      // Take the vector with the indeces to the clusters associated with this track
      std::vector<int> v_clu_id( Track.clu_id );
      int clu_id_size = v_clu_id.size();

      // Fill global track angles
      tk_eta->Fill(Track.eta);
      tk_phi->Fill(Track.phi);

      normchi2->Fill( Track.normchi2 );
      normchi2_vs_hitnum->Fill( Track.hitspertrack, Track.normchi2 );
      nhit->Fill( Track.hitspertrack );
      nclu_on->Fill( clu_id_size );

      // } // end if normchi2 < 20

    } // end loop on tracks

    // Loop on clusters
    for ( unsigned int anaclu_iter = 0; anaclu_iter < v_anaclu.size(); ++anaclu_iter ) { 
      // Take the Cluster
      AnalyzedCluster Cluster( v_anaclu[anaclu_iter] );

    } // end loop on clusters

    //   } // end if numberoftracks <= 1
    // } // end if numberofclusters < 80

  } // end loop on events

  nhit->Write();
  normchi2->Write();
  normchi2_vs_hitnum->Write();
  nclu_on->Write();

  tk_eta->Write();
  tk_phi->Write();

  hfile->Write();

  return 0;
};
