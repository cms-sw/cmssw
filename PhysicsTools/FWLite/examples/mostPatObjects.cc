// Standard includes
#include <iostream>
#include <string>

// CMS includes
#include "DataFormats/FWLite/interface/Handle.h"
#include "DataFormats/FWLite/interface/Event.h"
#include "DataFormats/PatCandidates/interface/Jet.h"
#include "DataFormats/PatCandidates/interface/MET.h"
#include "DataFormats/PatCandidates/interface/Photon.h"
#include "DataFormats/PatCandidates/interface/Electron.h"
#include "DataFormats/PatCandidates/interface/Tau.h"
#include "DataFormats/TrackReco/interface/Track.h"

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
   optutl::setUsageAndDefaultOptions 
      ("Jet, MET, Photon, Track, Electron, and Tau Pt",
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
   event.add( new TH1F( "jetpt",      "Jet Pt",      100, 0, 200) );
   event.add( new TH1F( "jetnum",     "Jet Size",    100, 0, 50)  );
   event.add( new TH1F( "metpt",      "MET Pt",      100, 0, 200) );
   event.add( new TH1F( "photonpt",   "Photon Pt",   100, 0, 200) );
   event.add( new TH1F( "trackpt",    "Track Pt",    100, 0, 200) );
   event.add( new TH1F( "electronpt", "Electron Pt", 100, 0, 200) );
   event.add( new TH1F( "taupt",      "Tau Pt",      100, 0, 200) );   

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

      // Get jets, METs, photons, tracks, electrons, and taus
      fwlite::Handle< vector<pat::Jet> > h_jet;
      h_jet.getByLabel(event,"selectedLayer1Jets");
      assert( h_jet.isValid() );

      fwlite::Handle< vector<pat::MET> > h_met;
      h_met.getByLabel(event,"selectedLayer1METs");
      assert( h_met.isValid() );

      fwlite::Handle< vector<pat::Photon> > h_photon;
      h_photon.getByLabel(event,"selectedLayer1Photons");
      assert( h_photon.isValid() );

      fwlite::Handle< vector<reco::Track> > h_track;
      h_track.getByLabel(event,"generalTracks");
      assert( h_track.isValid() );

      fwlite::Handle< vector<pat::Electron> > h_electron;
      h_electron.getByLabel(event,"selectedLayer1Electrons");
      assert( h_electron.isValid() );

      fwlite::Handle< vector<pat::Tau> > h_tau;
      h_tau.getByLabel(event,"selectedLayer1Taus");
      assert( h_tau.isValid() );

      // Fill, baby, fill!
 
      event.hist("jetnum")->Fill( h_jet->size() );    
     
      const vector< pat::Jet >::const_iterator kJetEnd = h_jet->end();
      for (vector< pat::Jet >::const_iterator jetIter = h_jet->begin();
           jetIter != kJetEnd;
           ++jetIter) 
      {
         event.hist("jetpt")->Fill( jetIter->pt() );  
      }

      const vector< pat::MET >::const_iterator kMetEnd = h_met->end();
      for (vector< pat::MET >::const_iterator metIter = h_met->begin();
           metIter != kMetEnd;
           ++metIter) 
      {
         event.hist("metpt")->Fill( metIter->pt() );
      }

      const vector< pat::Photon >::const_iterator kPhotonEnd = h_photon->end();
      for (vector< pat::Photon >::const_iterator photonIter = h_photon->begin();
           photonIter != kPhotonEnd;
           ++photonIter) 
      {
         event.hist("photonpt")->Fill( photonIter->pt() );
      }

      const vector< reco::Track >::const_iterator kTrackEnd = h_track->end();
      for (vector< reco::Track >::const_iterator trackIter = h_track->begin();
           trackIter != kTrackEnd;
           ++trackIter) 
      {
         event.hist("trackpt")->Fill( trackIter->pt() );
      }

      const vector< pat::Electron >::const_iterator kElectronEnd = h_electron->end();
      for (vector< pat::Electron >::const_iterator electronIter = h_electron->begin();
           electronIter != kElectronEnd;
           ++electronIter) 
      {
         event.hist("electronpt")->Fill( electronIter->pt() );
      }

      const vector< pat::Tau >::const_iterator kTauEnd = h_tau->end();
      for (vector< pat::Tau >::const_iterator tauIter = h_tau->begin();
           tauIter != kTauEnd;
           ++tauIter) 
      {
         event.hist ("taupt")->Fill (tauIter->pt() );
      }
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
