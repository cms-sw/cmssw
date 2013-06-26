// -*- C++ -*-

// CMS includes
#include "FWCore/Utilities/interface/InputTag.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/PatCandidates/interface/Jet.h"
#include "DataFormats/Math/interface/deltaPhi.h"

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
   optutl::CommandLineParser parser ("Playing with jets");

   ////////////////////////////////////////////////
   // Change any defaults or add any new command //
   //      line options you would like here.     //
   ////////////////////////////////////////////////
   // change default output filename
   parser.stringValue ("outputFile") = "jetInfo.root";
   
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
   eventCont.add( new TH1F( "jetpt",        "Jet p_{T} using standard absolute p_{T} calibration", 100, 0, 60) );
   eventCont.add( new TH1F( "jeteta",       "Jet eta using standard absolute p_{T} calibration",   100, 0, 10) );
   eventCont.add( new TH1F( "reljetpt",     "Jet p_{T} using relative inter eta calibration",      100, 0, 60) );
   eventCont.add( new TH1F( "reljeteta",    "Jet eta using relative inter eta calibration",        100, 0, 10) );
   eventCont.add( new TH1F( "phijet1jet2",  "Phi between Jet 1 and Jet 2",                        100, 0, 3.5) );
   eventCont.add( new TH1F( "invarMass",    "Invariant Mass of the 4-vector sum of Two Jets",     100, 0, 200) );

   //////////////////////
   // //////////////// //
   // // Event Loop // //
   // //////////////// //
   //////////////////////

   // create labels
   edm::InputTag jetLabel ("selectedLayer1Jets");

   for (eventCont.toBegin(); ! eventCont.atEnd(); ++eventCont) 
   {
      //////////////////////////////////
      // Take What We Need From Event //
      //////////////////////////////////

      // Get jets
      edm::Handle< vector< pat::Jet > > jetHandle;
      eventCont.getByLabel (jetLabel, jetHandle);
      assert ( jetHandle.isValid() );

      const vector< pat::Jet >::const_iterator kJetEnd = jetHandle->end();
      for (vector< pat::Jet >::const_iterator jetIter = jetHandle->begin(); 
           jetIter != kJetEnd; 
           ++jetIter) 
      {	   
         // A few correctedJet options:
         // ABS - absolute pt calibration (automatic)
         // REL - relative inter eta calibration  
         // EMF - calibration as a function of the jet EMF
         eventCont.hist("reljetpt") ->Fill( jetIter->correctedJet("REL").pt() );
         eventCont.hist("reljeteta")->Fill( jetIter->correctedJet("REL").eta() );
         eventCont.hist("jetpt")    ->Fill( jetIter->correctedJet("ABS").pt() );
         // Automatically ABS
         eventCont.hist("jeteta")   ->Fill( jetIter->eta() );
      } // for jetIter

      // Do we have at least two jets?
      if (jetHandle->size() < 2)
      {
         // Nothing to do here
         continue;
      }

      // Store invariant mass and delta phi between two leading jets.
      eventCont.hist("invarMass")->Fill( (jetHandle->at(0).p4() + jetHandle->at(1).p4()).M() );
      eventCont.hist("phijet1jet2")->Fill( deltaPhi( jetHandle->at(0).phi(), jetHandle->at(1).phi() ) );
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
