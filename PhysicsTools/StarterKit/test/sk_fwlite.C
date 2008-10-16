#include "DataFormats/FWLite/interface/Handle.h"


#if !defined(__CINT__) && !defined(__MAKECINT__)
#include "DataFormats/PatCandidates/interface/Muon.h"
#endif

#include <iostream>

using namespace std;

void sk_fwlite()
{
   
   
  TFile  * file = new TFile("/uscms_data/d1/rappocc/PatAnalyzerSkeletonSkim.root");

  using namespace std;

  TH1D * hist_muPt = new TH1D("hist_muPt", "Muon p_{T}", 20, 0, 100 );

  fwlite::Event ev(file);
  
  for( ev.toBegin();
          ! ev.atEnd();
          ++ev) {


    fwlite::Handle<std::vector<pat::Muon> > h_mu;

    h_mu   .getByLabel(ev,"selectedLayer1Muons");


     for ( int i = 0; i < h_mu.ptr()->size();  ++i ) {

       vector<pat::Muon> const & muons = *h_mu;
       hist_muPt->Fill( muons[i].pt() );
     }

   }

   hist_muPt->Draw();
}
