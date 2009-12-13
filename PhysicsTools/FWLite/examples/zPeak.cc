// -*- C++ -*-

// CMS includes
#include "FWCore/Utilities/interface/InputTag.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/PatCandidates/interface/Muon.h"

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
   optutl::CommandLineParser parser ("Plot mass of Z candidates");

   ////////////////////////////////////////////////
   // Change any defaults or add any new command //
   //      line options you would like here.     //
   ////////////////////////////////////////////////
   parser.stringValue ("outputFile") = "zpeak1.root";

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
   eventCont.add( new TH1F ("Zmass", "Candidate Z mass", 50, 20, 220) );


   //////////////////////
   // //////////////// //
   // // Event Loop // //
   // //////////////// //
   //////////////////////

   // create labels
   edm::InputTag muonLabel ("selectedLayer1Muons");

   for (eventCont.toBegin(); ! eventCont.atEnd(); ++eventCont) 
   {
      //////////////////////////////////
      // Take What We Need From Event //
      //////////////////////////////////
      edm::Handle<std::vector<pat::Muon> > muonHandle;
      eventCont.getByLabel (muonLabel, muonHandle);
      assert (muonHandle.isValid());
      vector< pat::Muon > const & muonVec = *muonHandle;

      // make sure we have enough muons so that something bad doesn't
      // happen.
      if (muonVec.begin() == muonVec.end())
      {
         // If this statement is true, then we don't have at least one
         // muon, so don't bother.  The reason we need to do this is
         // that the logic below assumes that we have at least one and
         // we'll end up in an infinite loop without it.
         continue;
      }

      // O.k.  Let's loop through our muons and see what we can find.
      const vector< pat::Muon >::const_iterator kEndIter       = muonVec.end();
      const vector< pat::Muon >::const_iterator kAlmostEndIter = kEndIter - 1;
      for (vector< pat::Muon >::const_iterator outerIter = muonVec.begin();
           kAlmostEndIter != outerIter;
           ++outerIter)
      {
         for (vector< pat::Muon >::const_iterator innerIter = outerIter + 1;
              kEndIter != innerIter;
              ++innerIter)
         {
            // make sure that we have muons of opposite charge
            if (outerIter->charge() * innerIter->charge() >= 0) continue;

            // if we're here then we have one positively charged muon
            // and one negatively charged muon.
            eventCont.hist("Zmass")->Fill( (outerIter->p4() + innerIter->p4()).M() );
         } // for innerIter
      } // for outerIter
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
