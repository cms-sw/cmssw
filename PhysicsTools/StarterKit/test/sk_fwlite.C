{

   #include "DataFormats/FWLite/interface/Handle.h"

   
  TFile file("/tmp/PATLayer1Output.root");

  using namespace std;

  TH1D * hist_muPt = new TH1D("hist_muPt", "Muon p_{T}", 20, 0, 100 );

  fwlite::Event ev(&file);
  
  for( ev.toBegin();
          ! ev.atEnd();
          ++ev) {


     fwlite::Handle<std::vector<double> > h_muPt;

     h_muPt   .getByLabel(ev,"StarterKitDemo", "muPt");


     for ( int i = 0; i < h_muPt.ptr()->size();  ++i ) {
       hist_muPt->Fill( h_muPt.ptr()->at(i) );
     }

   }

   hist_muPt->Draw();
}
