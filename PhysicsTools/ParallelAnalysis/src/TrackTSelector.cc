#include "PhysicsTools/ParallelAnalysis/interface/TrackTSelector.h"
#include <TH2.h>
#include <TStyle.h>
#include <TCanvas.h>
#include <iostream>
using namespace std;

const char * branchName = "recoTracks_CTFAnalytical__CtfAnalytical.obj";

void TrackTSelector::Init( TTree *tree ) {
  cout << "init" << endl;
  if ( tree == 0 ) return;
  chain = tree;
  cout << ">> tree: " << chain->GetName() << endl;
  chain->SetBranchAddress( branchName, & tracks );
}

bool TrackTSelector::Notify() {
  cout << "notify" << endl;
  tracksBranch = chain->GetBranch( branchName );
  assert( tracksBranch != 0 );
  return true;
}

void TrackTSelector::Begin( TTree * ) {
  cout << "begin" << endl;
  TString option = GetOption();
  h_pt  = new TH1F( "pt" , "pt"  , 100,  0, 20 );
  h_eta = new TH1F( "eta", "#eta", 100, -3,    3 );
}

void TrackTSelector::SlaveBegin( TTree * tree ) {
  cout << "slaveBegin" << endl;
  Init( tree );
  TString option = GetOption();
}

bool TrackTSelector::Process( long long entry ) {
  cout << "processing event " << entry << endl;
  //  chain->GetEntry( entry );
  tracksBranch->GetEntry( entry );
  cout << ">> tracks found:" << tracks.size() << endl;
  for ( size_t i = 0; i < tracks.size(); ++i ) {
    const reco::Track & track = tracks[ i ];
    h_pt ->Fill( track.pt() );
    h_eta->Fill( track.eta() );
    cout << ">> pt, eta:  " << track.pt() << ", " << track.eta() << endl;
  }
  return true;
}

void TrackTSelector::SlaveTerminate() {
  cout << "slaveTerminate" << endl;
}

void TrackTSelector::Terminate() {
  cout << "terminate" << endl;
  TCanvas * canvas = new TCanvas( );
  h_pt->Draw();
  canvas->SaveAs( "pt.jpg" );
  h_eta->Draw();
  canvas->SaveAs( "eta.jpg" );
  delete canvas;
}
