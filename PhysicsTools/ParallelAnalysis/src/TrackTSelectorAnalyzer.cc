#include "PhysicsTools/ParallelAnalysis/src/TrackTSelectorAnalyzer.h"
#include <TCanvas.h>
#include <TROOT.h>
#include <TH1.h>
using namespace examples;

TrackTSelectorAnalyzer::TrackTSelectorAnalyzer( const edm::ParameterSet & ) :
  histograms_(), algo_( 0, histograms_ ) {
}

void TrackTSelectorAnalyzer::analyze( const edm::Event & event, const edm::EventSetup & ) {
  algo_.process( event );
}

void TrackTSelectorAnalyzer::endJob() {
  algo_.postProcess( histograms_ );
  gROOT->SetBatch();
  gROOT->SetStyle("Plain");
  TCanvas canvas;
  algo_.h_pt->Draw();
  canvas.SaveAs( "pt-batch.jpg" );
  algo_.h_eta->Draw();
  canvas.SaveAs( "eta-batch.jpg" );
}


