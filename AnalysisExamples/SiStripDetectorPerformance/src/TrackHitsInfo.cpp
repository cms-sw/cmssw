// Author : Samvel Khalatian (samvel at fnal dot gov)
// Created: 03/28/07
// Licence: GPL

#include "AnalysisExamples/SiStripDetectorPerformance/interface/TrackOstream.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Handle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/EDMException.h"

#include "AnalysisExamples/SiStripDetectorPerformance/interface/TrackHitsInfo.h"

TrackHitsInfo::TrackHitsInfo( const edm::ParameterSet &roCONFIG) {
  // Extract Tracks Label
  try {
    LogDebug( "SliceTestNtupleMaker::SliceTestNtupleMaker")
      << "\t* Extract Tracks Label";

    oITTrack_ = std::auto_ptr<edm::InputTag>( 
      new edm::InputTag( 
        roCONFIG.getUntrackedParameter<edm::InputTag>( "oTrack")) );
  } catch( edm::Exception &roEX) {
    edm::LogError( "SliceTestNtupleMaker::SliceTestNtupleMaker")
      << "Failed to extract 'oTrack' config value. Check if it is specified "
      << "in config file.";

    // Pass exception on
    throw;
  }
}

void TrackHitsInfo::analyze( const edm::Event      &roEVENT,
                             const edm::EventSetup &roEVENT_SETUP) {

  // Extract Tracks
  edm::Handle<VRecoTracks> oVRecoTracks;
  try {
    LogDebug( "SliceTestNtupleMaker::analyze")
      << "\t* Extract Tracks";

    roEVENT.getByLabel( *oITTrack_, oVRecoTracks);
  } catch( const edm::Exception &roEX) {
    edm::LogError( "SliceTestNtupleMaker::analyze")
      << "Failed to Exctract Tracks. Make sure they are present in event and "
      << "specified InputTag for Tracks is correct";

    // Pass exception on
    throw;
  }

  // Loop over Tracks
  for( VRecoTracks::const_iterator oTRACKS_ITER = oVRecoTracks->begin();
       oTRACKS_ITER != oVRecoTracks->end();
       ++oTRACKS_ITER) {

    edm::LogVerbatim( "TrackHitsInfo::analyze()")
      << TrackOstream( *oTRACKS_ITER);
  } // End loop over Tracks
}
