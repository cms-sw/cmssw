// -*- C++ -*-

// CMS includes
#include "FWCore/Utilities/interface/InputTag.h"
#include "DataFormats/Common/interface/Handle.h"
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
   edm::InputTag jetLabel      ("selectedLayer1Jets");
   edm::InputTag metLabel      ("selectedLayer1METs");
   edm::InputTag photonLabel   ("selectedLayer1Photons");
   edm::InputTag trackLabel    ("generalTracks");
   edm::InputTag electronLabel ("selectedLayer1Electrons");
   edm::InputTag tauLabel      ("selectedLayer1Taus");

   for (eventCont.toBegin(); ! eventCont.atEnd(); ++eventCont) 
   {
      //////////////////////////////////
      // Take What We Need From Event //
      //////////////////////////////////

      // Get jets, METs, photons, tracks, electrons, and taus
      edm::Handle< vector<pat::Jet> > h_jet;
      eventCont.getByLabel (jetLabel, h_jet);
      assert( h_jet.isValid() );

      edm::Handle< vector<pat::MET> > h_met;
      eventCont.getByLabel (metLabel, h_met);
      assert( h_met.isValid() );

      edm::Handle< vector<pat::Photon> > h_photon;
      eventCont.getByLabel (photonLabel, h_photon);
      assert( h_photon.isValid() );

      edm::Handle< vector<reco::Track> > h_track;
      eventCont.getByLabel (trackLabel, h_track);
      assert( h_track.isValid() );

      edm::Handle< vector<pat::Electron> > h_electron;
      eventCont.getByLabel (electronLabel, h_electron);
      assert( h_electron.isValid() );

      edm::Handle< vector<pat::Tau> > h_tau;
      eventCont.getByLabel (tauLabel, h_tau);
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
