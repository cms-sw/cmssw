#include "DataFormats/FWLite/interface/Handle.h"
#include "DataFormats/FWLite/interface/Event.h"
#include "TFile.h"
#include "TH1.h"
#include "TH2.h"
#include "TCanvas.h"
#include "TLegend.h"
#include "TMath.h"
#include "TStyle.h"


#if !defined(__CINT__) && !defined(__MAKECINT__)
#include "DataFormats/PatCandidates/interface/Muon.h"
#endif

#include <iostream>

using namespace std;

void sk_fwlite_cosmics()
{
   

  gStyle->SetPalette(1,0);
   
  TFile  * file = new TFile("patcosmics.root");

  using namespace std;

  TH1D * hist_muPt = new TH1D("hist_muPt", "Muon p_{T}", 20, 0, 100 );
  TH2D * hist_muEtaVsPhi = new TH2D("hist_muEtaVsPhi", "Muon #eta vs #phi", 20, -5.0, 5.0, 20, -TMath::Pi(), TMath::Pi() );

  fwlite::Event ev(file);
  
  for( ev.toBegin();
          ! ev.atEnd();
          ++ev) {


    fwlite::Handle<std::vector<pat::Muon> > h_mu;

    h_mu   .getByLabel(ev,"selectedLayer1Muons");


     for ( int i = 0; i < h_mu.ptr()->size();  ++i ) {

       vector<pat::Muon> const & muons = *h_mu;
       hist_muPt->Fill( muons[i].pt() );
       hist_muEtaVsPhi->Fill( muons[i].eta(), muons[i].phi() );
     }

   }

   hist_muEtaVsPhi->Draw("colz");
}
