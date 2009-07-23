// Standard includes
#include <iostream>
#include <string>

// CMS includes
#include "DataFormats/FWLite/interface/Handle.h"
#include "DataFormats/FWLite/interface/Event.h"
#include "DataFormats/JetReco/interface/CaloJet.h"

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
   optutl::setUsageAndDefaultOptions ("Plots Jet Pt");

   //////////////////////////////////////////////////////
   // Add any command line options you would like here //
   //////////////////////////////////////////////////////
   // optutl::addOption ("sampleName",   optutl::kString, 
   //                    "Sample name (e.g., top, Wqq, etc.)");   

   // Parse the command line arguments
   optutl::parseArguments (argc, argv);

   //////////////////////////////////
   // //////////////////////////// //
   // // Create Event Container // //
   // //////////////////////////// //
   //////////////////////////////////

   // This object 'event' is used both to get all information from the
   // event as well as to store histograms, etc.  It automatically
   // gets input files from command line options.
   fwlite::EventContainer event;

   ////////////////////////////////////////
   // ////////////////////////////////// //
   // //         Begin Run            // //
   // // (e.g., book histograms, etc) // //
   // ////////////////////////////////// //
   ////////////////////////////////////////

   // Setup a style
   gROOT->SetStyle ("Plain");

   // Book those histograms!
   event.add( new TH1F( "jetPt", "jetPt", 1000, 0, 1000) );

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
      fwlite::Handle< vector< reco::CaloJet > > jetCollection;
      jetCollection.getByLabel (event, "sisCone5CaloJets");
      assert ( jetCollection.isValid() );
						
      // Loop over the jets
      const vector< reco::CaloJet >::const_iterator kJetEnd = jetCollection->end();
      for (vector< reco::CaloJet >::const_iterator jetIter = jetCollection->begin();
           kJetEnd != jetIter; 
           ++jetIter) 
      {         
         event.hist("jetPt")->Fill (jetIter->pt());
      } // for jetIter
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
