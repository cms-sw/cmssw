#include "FWCore/TFWLiteSelectorTest/src/ThingsTSelector2.h"
#include <TH2.h>
#include <TStyle.h>
#include <TCanvas.h>
#include <iostream>
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/TestObjects/interface/ThingCollection.h"
#include "DataFormats/TestObjects/interface/OtherThingCollection.h"

using namespace std;
using namespace tfwliteselectortest;

//Names used in common between the worker and the Selector
static const char* kA = "a";
static const char* kRefA = "refA";



ThingsWorker::ThingsWorker(const TList*, TList& out ) {
    cout << "begin" << endl;
    h_a  = new TH1F( kA , "a"  , 100,  0, 20 );
    out.Add(h_a);
    
    h_refA  = new TH1F( kRefA , "refA"  , 100,  0, 20 );
    out.Add(h_refA);
}
  
  
  
void 
ThingsWorker::process( const edm::Event& iEvent ) {
    cout << "processing event " << endl;
    //  chain->GetEntry( entry );
    using namespace edmtest;
    edm::Handle<OtherThingCollection> hOThings;
    iEvent.getByLabel("OtherThing", "testUserTag", hOThings);
    
    cout << ">> other things found:" << hOThings->size() << endl;
    for ( size_t i = 0; i < hOThings->size(); ++i ) {
      const OtherThing & thing = (*hOThings)[ i ];
      h_refA ->Fill( thing.ref->a );
      cout << ">> ref->a:  " << thing.ref->a <<endl;
    }
    
    edm::Handle<ThingCollection> hThings;
    iEvent.getByLabel("Thing",hThings);
    const ThingCollection& things = *hThings;
    cout << ">> things found:" << things.size() << endl;
    for ( size_t i = 0; i < things.size(); ++i ) {
      const Thing & thing = things[ i ];
      h_a ->Fill( thing.a );
      cout << ">> a:  " << thing.a <<endl;
    }
    
  }
  
void 
ThingsWorker::postProcess(TList&)
{
}



void ThingsTSelector2::begin(TList*&)
{
}

void ThingsTSelector2::terminate(TList& out) {
  cout << "terminate" << endl;
  TCanvas * canvas = new TCanvas( );
  {
     TObject* hist = out.FindObject(kA);
     if(0!=hist) {
	hist->Draw();
	canvas->SaveAs( "a.jpg" );
     } else {
	cout <<"no '"<<kA<<"' histogram"<<endl;
     }
  }
  cout <<"refA"<<endl;
  {
     TObject* hist = out.FindObject(kRefA);
     if( 0 != hist ) {
	hist->Draw();
	canvas->SaveAs( "refA.jpg" );
     } else {
	cout <<"no '"<<kRefA<<"' histogram"<<endl;
     }
  }
  delete canvas;
}


