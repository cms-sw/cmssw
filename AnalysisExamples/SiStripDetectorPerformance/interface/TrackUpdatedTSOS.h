// Author : Samvel Khalatian ( samvel at fnal dot gov)
// Created: 03/15/07
// Licence: GPL

#ifndef RECO_TRACK_TSOS_H
#define RECO_TRACK_TSOS_H

#include <string>

#include "FWCore/Framework/interface/ESHandle.h"
#include "AnalysisExamples/SiStripDetectorPerformance/interface/MapHITTSOS.h"

// Forward declarations
namespace reco {
  class Track;
}

namespace edm {
  class EventSetup;
}

class TrackerGeometry;
class TrackingRecHit;
class TrajectoryStateOnSurface;
class TransientTrackingRecHitBuilder;
class MagneticField;

namespace reco {
  class GetUpdatedTSOSDebug {
    public:
      /** 
      * @brief 
      *   Construct object that involves extraction of next parameters from 
      *   EventSetup:
      *     - TrackerDigiGeometryRecord
      *     - Magnetic Field 
      *     - Transient RecHit Builder
      *   Make sure to include those Records in job config file.
      * 
      * @param roEVENT_SETUP         EventSetup object
      * @param roTTRH_BUILDER_LABEL  Transient Tracking RecHit Builder Label
      *                              [Note: can be gotten from config file at
      *                                     RecoTracker/
      *                                       RoadSearchTrackCandidateMaker/
      *                                         data/
      *                                           RoadSearchTrackCandidates.cfi]
      */
      GetUpdatedTSOSDebug( const edm::EventSetup &roEVENT_SETUP, 
                           const std::string     &roTTRH_BUILDER_LABEL);
      inline virtual ~GetUpdatedTSOSDebug() {}

      /** 
      * @brief
      *   Create Association of RecHit <-> Updated TSOS for particular reco::Track
      *
      * @param roTRACK  reco::Track for which TSOS should be gotten
      * @param roMHitTSOS  Map of Hit<->UpdatedTSOS association. 
      *                    ARGUMENT IS PASSED BY REFERENCE AND WILL BER OVERWRITTEN!
      * 
      * @return 
      *   true   Success
      *   false  On any error 
      *          [Note: in case of Error Map will be erased]
      */
      virtual bool operator()( const reco::Track &roTRACK,
                                     MHitTSOS    &roMHitTSOS) const;

    private:
      edm::ESHandle<TrackerGeometry>                oHTrackerGeometry_;
      edm::ESHandle<MagneticField>                  oHMagneticField_;
      edm::ESHandle<TransientTrackingRecHitBuilder> oHTTRHBuilder_;
  };
}

#endif // RECO_TRACK_TSOS_H
