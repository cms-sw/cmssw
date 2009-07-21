// Standard includes
#include <iostream>
#include <string>

// CMS includes
#include "FWCore/FWLite/interface/AutoLibraryLoader.h"
#include "DataFormats/FWLite/interface/Handle.h"
#include "DataFormats/FWLite/interface/Event.h"
#include "DataFormats/PatCandidates/interface/Jet.h"
#include "DataFormats/PatCandidates/interface/Muon.h"
#include "DataFormats/HepMCCandidate/interface/FlavorHistory.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "Math/GenVector/PxPyPzM4D.h"

#include "FWCore/FWLite/interface/TH1StoreNamespace.h" // (th1store::)
#include "FWCore/FWLite/interface/OptionUtils.h"       // (optutl::)
#include "FWCore/FWLite/interface/dout.h"
#include "FWCore/FWLite/interface/dumpSTL.icc"

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

// Book all histograms to be filled this job.
void bookHistograms();

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

   // Tell people what this analysis code does.
   optutl::setUsageString ("Short description here");

   // Setup default options
   optutl::setupDefaultOptions();

   //////////////////////////////////////////////////////
   // Add any command line options you would like here //
   //////////////////////////////////////////////////////

   // optutl::addOption ("someString",   optutl::kString, 
   //                    "pass in some string");   

   // Parse the command line arguments
   optutl::parseArguments (argc, argv);

   // Here is where you want to want to look at the command line
   // arguments you've received and see if you want to add a "tag" to
   // the output file.
   string tag = "";

   // if (someCondition) tag += "_someCond"
   
   // finish the default options
   optutl::finishDefaultOptions (tag);

   ////////////////////////////////////////
   // ////////////////////////////////// //
   // //         Begin Run            // //
   // // (e.g., book histograms, etc) // //
   // ////////////////////////////////// //
   ////////////////////////////////////////

   // Setup a style
   gROOT->SetStyle ("Plain");

   // Book those histograms!
   bookHistograms();


   //////////////////////
   // //////////////// //
   // // Event Loop // //
   // //////////////// //
   //////////////////////

   AutoLibraryLoader::enable();

   // setup an event counter so we can watch the progress
   const int kOutputEvery = optutl::integerValue ("outputEvery");
   const int kMaxEvent    = optutl::integerValue ("maxEvents");

   int eventNumber (0);
   fwlite::ChainEvent event( optutl::stringVector ("inputFiles") );
   for (event.toBegin(); !event.atEnd(); ++event, ++eventNumber) 
   {
      // Status report
      if (kOutputEvery && eventNumber % kOutputEvery == 0 ) 
      {
         cout << "Processing Event: " << eventNumber << endl;
      }
      // Stop the loop if the maximum number of events has been reached.
      if (kMaxEvent && eventNumber > kMaxEvent)
      {
         break;
      }

      //////////////////////////////////
      // Take What We Need From Event //
      //////////////////////////////////

      // // Get jets
      // fwlite::Handle< vector< pat::Jet > > jetCollection;
      // jetCollection.getByLabel (event, "selectedLayer1Jets");
      // assert ( jetCollection.isValid() );
						
      // const vector< pat::Jet >::const_iterator kJetEnd = jetCollection->end();
      // for (vector< pat::Jet >::const_iterator jetIter = jetCollection->begin(); 
      //     kJetEnd != jetIter; 
      //     ++jetIter) 
      // {
      //    th1store::hist ("jetpt")->Fill (jetIter->pt());
      // } // for jet

   } // for event
      
   ////////////////////////
   // ////////////////// //
   // // Clean Up Job // //
   // ////////////////// //
   ////////////////////////

   ////////////////////////////////////
   // Write histograms to root file. //
   ////////////////////////////////////
   th1store::write( optutl::stringValue ("outputFile") );

   // All done!  Bye bye.
   return 0;
}

//////////////////////////////////
// //////////////////////////// //
// // Supporting Subroutines // //
// //////////////////////////// //
//////////////////////////////////

void bookHistograms()
{
   // th1store::add( new TH1F( "jetpt", "jetpt", 100, 0, 1000) );
}
