// -*- C++ -*-

// CMS includes
#include "DataFormats/FWLite/interface/Handle.h"
#include "DataFormats/PatCandidates/interface/Jet.h"

#include "PhysicsTools/FWLite/interface/EventContainer.h"
#include "PhysicsTools/FWLite/interface/CommandLineParser.h" 

// Root includes
#include "TROOT.h"

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
   optutl::CommandLineParser parser ("Plots Jet Pt", 
                                     optutl::CommandLineParser::kEventContOpt);

   ////////////////////////////////////////////////
   // Change any defaults or add any new command //
   //      line options you would like here.     //
   ////////////////////////////////////////////////
   parser.stringValue ("outputFile") = "jetPt"; // .root added automatically

   // Parse the command line arguments
   parser.parseArguments (argc, argv);

   //////////////////////////////////
   // //////////////////////////// //
   // // Create Event Container // //
   // //////////////////////////// //
   //////////////////////////////////

   // This object 'event' is used both to get all information from the
   // event as well as to store histograms, etc.
   fwlite::EventContainer eventCont (parser);

   ////////////////////////////////////////
   // ////////////////////////////////// //
   // //         Begin Run            // //
   // // (e.g., book histograms, etc) // //
   // ////////////////////////////////// //
   ////////////////////////////////////////

   // Setup a style
   gROOT->SetStyle ("Plain");

   // Book those histograms!
   eventCont.add( new TH1F( "jetPt", "jetPt", 1000, 0, 1000) );

   //////////////////////
   // //////////////// //
   // // Event Loop // //
   // //////////////// //
   //////////////////////

   for (eventCont.toBegin(); ! eventCont.atEnd(); ++eventCont) 
   {
      //////////////////////////////////
      // Take What We Need From Event //
      //////////////////////////////////
      fwlite::Handle< vector< pat::Jet > > jetCollection;
      jetCollection.getByLabel (eventCont, "selectedLayer1Jets");
      assert ( jetCollection.isValid() );
						
      // Loop over the jets
      const vector< pat::Jet >::const_iterator kJetEnd = jetCollection->end();
      for (vector< pat::Jet >::const_iterator jetIter = jetCollection->begin();
           kJetEnd != jetIter; 
           ++jetIter) 
      {         
         eventCont.hist("jetPt")->Fill (jetIter->pt());
      } // for jetIter
   } // for eventCont

      
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
