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
  TrackAnalysisAlgorithm::terminate( out );
}
