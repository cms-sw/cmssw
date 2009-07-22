// Standard includes
#include <iostream>
#include <string>

// CMS includes
#include "DataFormats/FWLite/interface/Handle.h"
#include "DataFormats/FWLite/interface/Event.h"
#include "DataFormats/PatCandidates/interface/Jet.h"
#include "DataFormats/Math/interface/deltaPhi.h"

#include "PhysicsTools/FWLite/interface/EventContainer.h"
#include "PhysicsTools/FWLite/interface/OptionUtils.h"  // (optutl::)
#include "PhysicsTools/FWLite/interface/dout.h"
#include "PhysicsTools/FWLite/interface/dumpSTL.icc"

// root includes
#include "TFile.h"
#include "TROOT.h"
#include "TSystem.h"
#include "TH1.h"
#include "TH2.h"
#include "TString.h"

using namespace std;


//////////////////////////
// Forward Declarations //
//////////////////////////

// This subroutine, written by you (below), uses the command line
// arguments and creates an output tag (if any).  This subroutine must
// exist.
void outputNameTagFunc (string &tag);

// Book all histograms to be filled this job.  If wanted, you can skip
// this subroutine and book all histograms in the main subroutine.
void bookHistograms (fwlite::EventContainer &event);


///////////////////////////
// ///////////////////// //
// // Main Subroutine // //
// ///////////////////// //
///////////////////////////

int main (int argc, char* argv[]) 
{
   ////////////////////////////////
   // ////////////////////////// //
   // // Command Line Options // //
   // ////////////////////////// //
   ////////////////////////////////

   // Tell people what this analysis code does and setup default options.
   optutl::setUsageAndDefaultOptions ("Playing around with jets",
                                      optutl::kEventContainer);

   //////////////////////////////////////////////////////
   // Add any command line options you would like here //
   //////////////////////////////////////////////////////

   // Parse the command line arguments
   optutl::parseArguments (argc, argv);

   //////////////////////////////////
   // //////////////////////////// //
   // // Create Event Container // //
   // //////////////////////////// //
   //////////////////////////////////

   // This object 'event' is used both to get all information from the
   // event as well as to store histograms, etc.
   fwlite::EventContainer event (&outputNameTagFunc);

   ////////////////////////////////////////
   // ////////////////////////////////// //
   // //         Begin Run            // //
   // // (e.g., book histograms, etc) // //
   // ////////////////////////////////// //
   ////////////////////////////////////////

   // Setup a style
   gROOT->SetStyle ("Plain");

   // Book those histograms!
   bookHistograms (event);

   //////////////////////
   // //////////////// //
   // // Event Loop // //
   // //////////////// //
   //////////////////////

   for (event.toBegin(); ! event.atEnd(); ++event) 
   {
      //////////////////////////////////
      // Take What We Need From Event //
      //////////////////////////////////

      // Get jets
      fwlite::Handle< vector<pat::Jet> > h_jet;
      h_jet.getByLabel(event,"selectedLayer1Jets");
      assert ( h_jet.isValid() );

      const vector< pat::Jet >::const_iterator kJetEnd = h_jet->end();
      for (vector< pat::Jet >::const_iterator jetIter = h_jet->begin(); 
           jetIter != kJetEnd; 
           ++jetIter) 
      {	   
         // A few correctedJet options:
         // ABS - absolute pt calibration (automatic)
         // REL - relative inter eta calibration  
         // EMF - calibration as a function of the jet EMF
         event.hist("reljetpt") ->Fill( jetIter->correctedJet("REL").pt() );
         event.hist("reljeteta")->Fill( jetIter->correctedJet("REL").eta() );
         event.hist("jetpt")    ->Fill( jetIter->correctedJet("ABS").pt() );
         // Automatically ABS
         event.hist("jeteta")   ->Fill( jetIter->eta() );
      } // for jetIter

      // Do we have at least two jets?
      if (h_jet->size() < 2)
      {
         // Nothing to do here
         continue;
      }

      // Store invariant mass and delta phi between two leading jets.
      event.hist("invarMass")->Fill( (h_jet->at(0).p4() + h_jet->at(1).p4()).M() );
      event.hist("phijet1jet2")->Fill( deltaPhi( h_jet->at(0).phi(), h_jet->at(1).phi() ) );
   } // for event

      
   ////////////////////////
   // ////////////////// //
   // // Clean Up Job // //
   // ////////////////// //
   ////////////////////////

   // Histograms will be automatically written to the root file
   // specificed by command line options.

   // All done!  Bye bye.
   return 0;
}


//////////////  //////////////////////////////////  //////////////
//////////////  // //////////////////////////// //  //////////////
//////////////  // // Supporting Subroutines // //  //////////////
//////////////  // //////////////////////////// //  //////////////
//////////////  //////////////////////////////////  //////////////


void outputNameTagFunc (string &tag)
{
   // If you do not want to give you output filename any "tag" based
   // on the command line options, simply do nothing here.  This
   // function is designed to be called by fwlite::EventContainer constructor.

   // if ( optutl::boolValue ("someCondition") )
   // { 
   //    tag += "_someCond";
   // }
}


void bookHistograms (fwlite::EventContainer &event)
{
   event.add( new TH1F( "jetpt",        "Jet p_{T} using standard absolute p_{T} calibration", 100, 0, 60) );
   event.add( new TH1F( "jeteta",       "Jet eta using standard absolute p_{T} calibration",   100, 0, 10) );
   event.add( new TH1F( "reljetpt",     "Jet p_{T} using relative inter eta calibration",      100, 0, 60) );
   event.add( new TH1F( "reljeteta",    "Jet eta using relative inter eta calibration",        100, 0, 10) );
   event.add( new TH1F( "phijet1jet2",  "Phi between Jet 1 and Jet 2",                        100, 0, 3.5) );
   event.add( new TH1F( "invarMass",    "Invariant Mass of the 4-vector sum of Two Jets",     100, 0, 200) );
}

