/*   A macro for making a histogram of Jet Pt with cuts
This is a basic way to cut out jets of a certain Pt and Eta using an if statement
This example creates a histogram of Jet Pt, using Jets with Pt above 30 and ETA above -2.1 and below 2.1
*/

#include "DataFormats/FWLite/interface/Handle.h"
#include "DataFormats/FWLite/interface/Event.h"
#include "TFile.h"
#include "TH1.h"
#include "TCanvas.h"
#include "TLegend.h"


#if !defined(__CINT__) && !defined(__MAKECINT__)
#include "DataFormats/PatCandidates/interface/Jet.h"
#include "PhysicsTools/PatUtils/interface/JetIDSelectionFunctor.h"
#endif

#include <iostream>
#include <cmath>      //necessary for absolute function fabs()

using namespace std;

void sk_fwlitecuts()
{
  JetIDSelectionFunctor jetId( JetIDSelectionFunctor::CRAFT08, JetIDSelectionFunctor::TIGHT );


  TFile  * file = new TFile("PATLayer1_Output.fromAOD_full.root");
  TH1D * hist_jetPt = new TH1D("hist_jetPt", "Jet p_{T}", 20, 0, 100 );
  fwlite::Event ev(file);

  //loop through each event
  for( ev.toBegin();
         ! ev.atEnd();
         ++ev) {
    fwlite::Handle<std::vector<pat::Jet> > h_mu;
    h_mu.getByLabel(ev,"cleanLayer1Jets");
    if (!h_mu.isValid() ) continue;
    vector<pat::Jet> const & jets = *h_mu;

   //loop through each Jet
   vector<pat::Jet>::const_iterator iter;
   for ( iter = jets.begin(); iter != jets.end() ; ++iter) {
   
     if ( (iter->pt() > 30 ) && ( fabs(iter->eta() ) < 2.1)  ) {
       cout << "Passed kin" << endl;
       if ( jetId( *iter ) ) {
	 cout << "Passed ID" << endl;
	 hist_jetPt->Fill( iter->pt() );
       }
     }
       

   }   //end Jet loop   
   }   //end event loop

   hist_jetPt->Draw();   
}
