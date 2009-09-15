/*   A macro for making a histogram of Muon Pt with cuts
This is a basic way to cut out muons of a certain Pt and Eta using an if statement
This example creates a histogram of Muon Pt, using Muons with Pt above 30 and ETA above -2.1 and below 2.1
*/

#include "DataFormats/FWLite/interface/Handle.h"
#include "DataFormats/FWLite/interface/Event.h"
#include "TFile.h"
#include "TH1.h"
#include "TCanvas.h"
#include "TLegend.h"


#if !defined(__CINT__) && !defined(__MAKECINT__)
#include "DataFormats/PatCandidates/interface/Muon.h"
#include "PhysicsTools/PatUtils/interface/MuonVPlusJetsIDSelectionFunctor.h"
#endif

#include <iostream>
#include <cmath>      //necessary for absolute function fabs()

using namespace std;

void sk_fwlitecuts()
{
  MuonVPlusJetsIDSelectionFunctor muId( MuonVPlusJetsIDSelectionFunctor::SUMMER08 );


  TFile  * file = new TFile("PATLayer1_Output.fromAOD_full.root");
  TH1D * hist_muPt = new TH1D("hist_muPt", "Muon p_{T}", 20, 0, 100 );
  fwlite::Event ev(file);

  //loop through each event
  for( ev.toBegin();
         ! ev.atEnd();
         ++ev) {
    fwlite::Handle<std::vector<pat::Muon> > h_mu;
    h_mu.getByLabel(ev,"cleanLayer1Muons");
    if (!h_mu.isValid() ) continue;
    vector<pat::Muon> const & muons = *h_mu;

   //loop through each Muon
   vector<pat::Muon>::const_iterator iter;
   for ( iter = muons.begin(); iter != muons.end() ; ++iter) {
   
     if ( (iter->pt() > 30 ) && ( fabs(iter->eta() ) < 2.1)  ) {
       cout << "Passed kin" << endl;
       if ( muId( *iter ) ) {
	 cout << "Passed ID" << endl;
	 hist_muPt->Fill( iter->pt() );
       }
     }
       

   }   //end Muon loop   
   }   //end event loop

   hist_muPt->Draw();   
}
