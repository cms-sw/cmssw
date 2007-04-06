// Author : Samvel Khalatian (samvel at fnal dot gov)
// Created: 03/27/07
// Licence: GPL

#ifndef ANALYSISEXAMPLES_SISTRIPDETECTORPERFORMANCE_TRACK_HITS_INFO_H
#define ANALYSISEXAMPLES_SISTRIPDETECTORPERFORMANCE_TRACK_HITS_INFO_H

#include <memory>
#include <vector>

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/ParameterSet/interface/InputTag.h"

// Save Compile time by forwarding declarations
#include "FWCore/Framework/interface/Frameworkfwd.h"

namespace reco {
  class Track;
}

class TrackHitsInfo: public edm::EDAnalyzer {
  public:
    TrackHitsInfo( const edm::ParameterSet &roCONFIG);
    virtual ~TrackHitsInfo() {}

  protected:
    // Leave possibility of inheritance
    virtual void analyze ( const edm::Event      &roEVENT,
                           const edm::EventSetup &roEVENT_SETUP);

  private:
    typedef std::vector<reco::Track> VRecoTracks;

    std::auto_ptr<edm::InputTag> oITTrack_;
};

#endif // ANALYSISEXAMPLES_SISTRIPDETECTORPERFORMANCE_TRACK_HITS_INFO_H
