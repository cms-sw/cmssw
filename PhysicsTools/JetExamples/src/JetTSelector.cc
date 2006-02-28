#include "PhysicsTools/JetExamples/interface/JetTSelector.h"
#include <TH2.h>
#include <TStyle.h>
#include <TCanvas.h>
#include <iostream>
using namespace std;

void JetTSelector::Init( TTree *tree ) {
  cout << "init" << endl;
  if ( tree == 0 ) return;
  chain = tree;
  cout << ">> tree: " << chain->GetName() << endl;
  chain->SetBranchAddress( "Candidate_jets.obj", & jets );
}

bool JetTSelector::Notify() {
  cout << "notify" << endl;
  jetsBranch = chain->GetBranch( "Candidate_jets.obj" );
  assert( jetsBranch != 0 );
  return true;
}

void JetTSelector::Begin( TTree * ) {
  cout << "begin" << endl;
  TString option = GetOption();
  h_et  = new TH1F( "et" , "Et"  , 100,  0, 1000 );
  h_eta = new TH1F( "eta", "#eta", 100, -3,    3 );
}

void JetTSelector::SlaveBegin( TTree * tree ) {
  cout << "slaveBegin" << endl;
  Init( tree );
  TString option = GetOption();
}

bool JetTSelector::Process( long long entry ) {
  cout << "processing event " << entry << endl;
  //  chain->GetEntry( entry );
  jetsBranch->GetEntry( entry );
  cout << ">> jets found:" << jets.size() << endl;
  for ( size_t i = 0; i < jets.size(); ++i ) {
    const reco::Candidate & jet = jets[ i ];
    h_et ->Fill( jet.et() );
    h_eta->Fill( jet.eta() );
    cout << ">> Et, eta:  " << jet.et() << ", " << jet.eta() << endl;
  }
  return true;
}

void JetTSelector::SlaveTerminate() {
  cout << "slaveTerminate" << endl;
}

void JetTSelector::Terminate() {
  cout << "terminate" << endl;
  TCanvas *ce = new TCanvas( );
  h_et->Draw();
  ce->SaveAs( "et.jpg" );
  h_eta->Draw();
  ce->SaveAs( "eta.jpg" );
}
