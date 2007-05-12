#include "PhysicsTools/ParallelAnalysis/interface/TrackTSelector.h"
#include <iostream>
#include <TCanvas.h>
#include <TH1.h>
using namespace std;
using namespace examples;

TrackTSelector::TrackTSelector() {
  cout << ">> constructing TrackTSelector" << endl;
}

void TrackTSelector::begin( TList * & ) {
  cout << ">> nothing to be done at begin" << endl;
}

void TrackTSelector::terminate( TList & out ) {
  cout << ">> terminating" << endl;
  canvas_ = new TCanvas( );
  draw( out, TrackAnalysisAlgorithm::kPt );
  draw( out, TrackAnalysisAlgorithm::kEta );
  delete canvas_;
}

void TrackTSelector::draw( const TList & out, const char * k ) {
  TObject * hist = out.FindObject( k );
  if( 0 != hist ) {
    hist->Draw();
    canvas_->SaveAs( ( string( k ) + ".jpg" ).c_str() );
  } else {
    cout <<">> no '" << k << "' histogram" << endl;
  }
}
