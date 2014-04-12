/*   A macro for making a histogram of Electron Pt with cuts
This is a basic way to cut out electrons of a certain Pt and Eta using an if statement
This example creates a histogram of Electron Pt, using Electrons with Pt above 30 and ETA above -2.1 and below 2.1
*/

#include "DataFormats/FWLite/interface/Handle.h"
#include "DataFormats/FWLite/interface/Event.h"
#include "TFile.h"
#include "TH1.h"
#include "TCanvas.h"
#include "TLegend.h"


#if !defined(__CINT__) && !defined(__MAKECINT__)
#include "DataFormats/PatCandidates/interface/Electron.h"
#include "PhysicsTools/PatUtils/interface/ElectronVPlusJetsIDSelectionFunctor.h"
#endif

#include <iostream>
#include <cmath>      //necessary for absolute function fabs()

using namespace std;

void sk_fwlitecuts()
{
  ElectronVPlusJetsIDSelectionFunctor muId( ElectronVPlusJetsIDSelectionFunctor::SUMMER08 );


  TFile  * file = new TFile("PATLayer1_Output.fromAOD_full.root");
  TH1D * hist_ePt = new TH1D("hist_ePt", "Electron p_{T}", 20, 0, 100 );
  fwlite::Event ev(file);

  //loop through each event
  for( ev.toBegin();
         ! ev.atEnd();
         ++ev) {
    fwlite::Handle<std::vector<pat::Electron> > h_mu;
    h_mu.getByLabel(ev,"cleanLayer1Electrons");
    if (!h_mu.isValid() ) continue;
    vector<pat::Electron> const & electrons = *h_mu;

   //loop through each Electron
   vector<pat::Electron>::const_iterator iter;
   for ( iter = electrons.begin(); iter != electrons.end() ; ++iter) {
   
     if ( (iter->pt() > 30 ) && ( fabs(iter->eta() ) < 2.1)  ) {
       cout << "Passed kin" << endl;
       if ( muId( *iter ) ) {
	 cout << "Passed ID" << endl;
	 hist_ePt->Fill( iter->pt() );
       }
     }
       

   }   //end Electron loop   
   }   //end event loop

   hist_ePt->Draw();   
}
