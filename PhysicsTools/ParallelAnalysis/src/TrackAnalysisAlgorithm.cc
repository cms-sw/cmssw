#include "PhysicsTools/ParallelAnalysis/interface/TrackAnalysisAlgorithm.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/Event.h"
#include <TCanvas.h>
#include <TList.h>
#include <TH1.h>
#include <iostream>
using namespace examples;
using namespace std;
using namespace edm;
using namespace reco;

const char * TrackAnalysisAlgorithm::kPt = "pt";
const char * TrackAnalysisAlgorithm::kEta = "eta";

TrackAnalysisAlgorithm::TrackAnalysisAlgorithm( const TList *, TList& out )  {
  cout << ">> booking histograms" << endl;
  out.Add( h_pt  = new TH1F( kPt , "pt"  , 100,  0, 20 ) );
  out.Add( h_eta = new TH1F( kEta, "#eta", 100, -3,    3 ) );
}

void TrackAnalysisAlgorithm::process( const Event & event ) {
  cout << ">> processing event " << endl;
  Handle<TrackCollection> tracks;
  event.getByLabel( "ctfWithMaterialTracks", tracks );

  cout << ">> tracks found:" << tracks->size() << endl;
  for ( size_t i = 0; i < tracks->size(); ++i ) {
    const Track & track = ( * tracks )[ i ];
    h_pt ->Fill( track.pt() );
    h_eta->Fill( track.eta() );
    cout << ">> pt, eta:  " << track.pt() << ", " << track.eta() << endl;
  }
}

void TrackAnalysisAlgorithm::postProcess( TList & ) {
  cout << ">> nothing to be done in post-processing" << endl;
}

void TrackAnalysisAlgorithm::terminate( TList & out ) {
  cout << ">> terminating" << endl;
  TCanvas canvas;
  draw( out, canvas,  kPt );
  draw( out, canvas, kEta );
}

void TrackAnalysisAlgorithm::draw( const TList & out, TCanvas & canvas, const char * k ) {
  TObject * hist = out.FindObject( k );
  if( 0 != hist ) {
    hist->Draw();
    canvas.SaveAs( ( string( k ) + ".jpg" ).c_str() );
  } else {
    cerr <<">> no '" << k << "' histogram" << endl;
  }
}
