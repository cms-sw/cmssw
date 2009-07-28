// -*- C++ -*-

// CMS includes
#include "DataFormats/FWLite/interface/Handle.h"
#include "DataFormats/PatCandidates/interface/Jet.h"
#include "DataFormats/PatCandidates/interface/MET.h"
#include "DataFormats/PatCandidates/interface/Photon.h"
#include "DataFormats/PatCandidates/interface/Electron.h"
#include "DataFormats/PatCandidates/interface/Tau.h"
#include "DataFormats/TrackReco/interface/Track.h"

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
   optutl::CommandLineParser parser ("Accessing many different PAT objects");

   ////////////////////////////////////////////////
   // Change any defaults or add any new command //
   //      line options you would like here.     //
   ////////////////////////////////////////////////
   parser.stringValue ("outputFile") = "mostPat"; // .root added automatically

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
   eventCont.add( new TH1F( "jetpt",      "Jet Pt",      100, 0, 200) );
   eventCont.add( new TH1F( "jetnum",     "Jet Size",    100, 0, 50)  );
   eventCont.add( new TH1F( "metpt",      "MET Pt",      100, 0, 200) );
   eventCont.add( new TH1F( "photonpt",   "Photon Pt",   100, 0, 200) );
   eventCont.add( new TH1F( "trackpt",    "Track Pt",    100, 0, 200) );
   eventCont.add( new TH1F( "electronpt", "Electron Pt", 100, 0, 200) );
   eventCont.add( new TH1F( "taupt",      "Tau Pt",      100, 0, 200) );   

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

      // Get jets, METs, photons, tracks, electrons, and taus
      fwlite::Handle< vector<pat::Jet> > h_jet;
      h_jet.getByLabel(eventCont,"selectedLayer1Jets");
      assert( h_jet.isValid() );

      fwlite::Handle< vector<pat::MET> > h_met;
      h_met.getByLabel(eventCont,"selectedLayer1METs");
      assert( h_met.isValid() );

      fwlite::Handle< vector<pat::Photon> > h_photon;
      h_photon.getByLabel(eventCont,"selectedLayer1Photons");
      assert( h_photon.isValid() );

      fwlite::Handle< vector<reco::Track> > h_track;
      h_track.getByLabel(eventCont,"generalTracks");
      assert( h_track.isValid() );

      fwlite::Handle< vector<pat::Electron> > h_electron;
      h_electron.getByLabel(eventCont,"selectedLayer1Electrons");
      assert( h_electron.isValid() );

      fwlite::Handle< vector<pat::Tau> > h_tau;
      h_tau.getByLabel(eventCont,"selectedLayer1Taus");
      assert( h_tau.isValid() );

      // Fill, baby, fill!
 
      eventCont.hist("jetnum")->Fill( h_jet->size() );    
     
      const vector< pat::Jet >::const_iterator kJetEnd = h_jet->end();
      for (vector< pat::Jet >::const_iterator jetIter = h_jet->begin();
           jetIter != kJetEnd;
           ++jetIter) 
      {
         eventCont.hist("jetpt")->Fill( jetIter->pt() );  
      }

      const vector< pat::MET >::const_iterator kMetEnd = h_met->end();
      for (vector< pat::MET >::const_iterator metIter = h_met->begin();
           metIter != kMetEnd;
           ++metIter) 
      {
         eventCont.hist("metpt")->Fill( metIter->pt() );
      }

      const vector< pat::Photon >::const_iterator kPhotonEnd = h_photon->end();
      for (vector< pat::Photon >::const_iterator photonIter = h_photon->begin();
           photonIter != kPhotonEnd;
           ++photonIter) 
      {
         eventCont.hist("photonpt")->Fill( photonIter->pt() );
      }

      const vector< reco::Track >::const_iterator kTrackEnd = h_track->end();
      for (vector< reco::Track >::const_iterator trackIter = h_track->begin();
           trackIter != kTrackEnd;
           ++trackIter) 
      {
         eventCont.hist("trackpt")->Fill( trackIter->pt() );
      }

      const vector< pat::Electron >::const_iterator kElectronEnd = h_electron->end();
      for (vector< pat::Electron >::const_iterator electronIter = h_electron->begin();
           electronIter != kElectronEnd;
           ++electronIter) 
      {
         eventCont.hist("electronpt")->Fill( electronIter->pt() );
      }

      const vector< pat::Tau >::const_iterator kTauEnd = h_tau->end();
      for (vector< pat::Tau >::const_iterator tauIter = h_tau->begin();
           tauIter != kTauEnd;
           ++tauIter) 
      {
         eventCont.hist ("taupt")->Fill (tauIter->pt() );
      }
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
