// Author : Samvel Khalatian ( samvel at fnal dot gov)
// Created: 03/15/07
// Licence: GPL

#include <sstream>

#include "AnalysisExamples/SiStripDetectorPerformance/interface/TrackOstream.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHitFwd.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "TrackingTools/GeomPropagators/interface/AnalyticalPropagator.h"
#include "TrackingTools/KalmanUpdators/interface/KFUpdator.h"
#include "TrackingTools/Records/interface/TransientRecHitRecord.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"
#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHitBuilder.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"

#include "AnalysisExamples/SiStripDetectorPerformance/interface/TrackUpdatedTSOSDebug.h"

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
reco::GetUpdatedTSOSDebug::GetUpdatedTSOSDebug( 
  const edm::EventSetup &roEVENT_SETUP,
  const std::string     &roTTRH_BUILDER_LABEL) {

  // Extract TrackerGeometry
  try {
    LogDebug( "reco::GetUpdatedTSOSDebug::GetUpdatedTSOSDebug")
      << "\t* Extract Tracker Geometry";

    roEVENT_SETUP.get<TrackerDigiGeometryRecord>().get( oHTrackerGeometry_);
  } catch( const cms::Exception &roEX) {
    edm::LogError( "reco::GetUpdatedTSOSDebug::GetUpdatedTSOSDebug") 
      << "Failed to extract Tracker Geometry. "
      << "Make sure you have included it in your CONFIG FILE";

    // Pass Exception on
    throw;
  }

  // Extract Magnetic Field
  try {
    LogDebug( "reco::GetUpdatedTSOSDebug::GetUpdatedTSOSDebug")
      << "\t* Extract Magnetic Field";

    roEVENT_SETUP.get<IdealMagneticFieldRecord>().get( oHMagneticField_);
  } catch( const cms::Exception &roEX) {
    edm::LogError( "reco::GetUpdatedTSOSDebug::GetUpdatedTSOSDebug") 
      << "Failed to extract Magnetic Field. "
      << "Make sure you have included it in your CONFIG FILE";

    // Pass Exception on
    throw;
  }

  // Extract Transient Builder
  try {
    LogDebug( "reco::GetUpdatedTSOSDebug::GetUpdatedTSOSDebug")
      << "\t* Extract Transient Builder";

    roEVENT_SETUP.get<TransientRecHitRecord>().get( roTTRH_BUILDER_LABEL, 
                                                    oHTTRHBuilder_);
  } catch( const cms::Exception &roEX) {
    edm::LogError( "reco::GetUpdatedTSOSDebug::GetUpdatedTSOSDebug") 
      << "Failed to extract Transient Tracking RecHit Builder. "
      << "Make sure you have included it in your CONFIG FILE " 
      << "and specified correct Label for it.";

    // Pass Exception on
    throw;
  }
}

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
bool reco::GetUpdatedTSOSDebug::operator()( const reco::Track &roTRACK,
                                                  MHitTSOS    &roMHitTSOS) const
{

  bool bResult = true;

  // Clear map of Hit-TSOS
  roMHitTSOS.clear();

  // Create Transformer
  TrajectoryStateTransform oStateTransformer;

  // Get Track inner SOS
  LogDebug( "reco::GetUpdatedTSOSDebug::operator()") 
    << "\t* Extract Track innerTSOS";
  TrajectoryStateOnSurface oTSOSLast =
    oStateTransformer.innerStateOnSurface( roTRACK,
                                           *oHTrackerGeometry_,
                                           oHMagneticField_.product());

  // Cosmic Track finder remedy
  if( roTRACK.innerDetId() == ( *( roTRACK.recHitsBegin()))->geographicalId().rawId()) {
    LogDebug( "reco::GetUpdatedTSOSDebug::operator()") 
      << "\t* Extract Track outerTSOS";
    oTSOSLast =
      oStateTransformer.outerStateOnSurface( roTRACK,
                                             *oHTrackerGeometry_,
                                             oHMagneticField_.product());
  }

  if( oTSOSLast.isValid()) {
    // Create Updator
    KFUpdator oUpdator;

    // Extract Last hit (current oTSOSLast should corerspond to it and 
    // represent at least predicted TSOS)
    trackingRecHit_iterator oHitIter = roTRACK.recHitsEnd() - 1;

    // Update TSOS
    oTSOSLast = oUpdator.update( oTSOSLast, 
                                 *( oHTTRHBuilder_->build( oHitIter->get()) ));

    if( oTSOSLast.isValid()) {
      // Save TSOS: associate Updated TSOS with Hit
      roMHitTSOS[oHitIter->get()] = oTSOSLast;

      // Create Propogator
      AnalyticalPropagator oPropagator( oHMagneticField_.product(), 
                                        alongMomentum);

      // Loop over the rest of the hits and get corresponding UpdatedTSOS
      // [Note: unfortunately CMSSW developers didn't deign to provide reverse
      //        iterator for any of Vectors that are in use]
      do {
        --oHitIter;

        // Propagate TSOS
        LogDebug( "reco::GetUpdatedTSOSDebug::operator()") 
          << "\t* Get Predicted TSOS";
        oTSOSLast = 
          oPropagator.propagate( oTSOSLast, 
                                 oHTrackerGeometry_->idToDet( 
                                   ( *oHitIter)->geographicalId())->surface()); 
        if( !oTSOSLast.isValid()) {
          edm::LogError( "reco::GetUpdatedTSOSDebug::operator()") 
            << "\tINVALID predicted TSOS";
          bResult = false;
          break;
        }

        // Update TSOS: but first check if Hit is Valid
        if( ( *oHitIter)->isValid()) {
          LogDebug( "reco::GetUpdatedTSOSDebug::operator()") 
            << "\t* Get Updated TSOS";
          oTSOSLast = oUpdator.update( oTSOSLast, 
                                       *( oHTTRHBuilder_->build( oHitIter->get()) ));
          if( !oTSOSLast.isValid()) {
            edm::LogError( "reco::GetUpdatedTSOSDebug::operator()") 
              << "\tINVALID updated TSOS";
            bResult = false;
            break;
          }
        } else {
          LogDebug( "reco::GetUpdatedTSOSDebug::operator()")
            << "\t* Dealing with INVALID Hit: UpdatedTSOS=PredictedTSOS";
        }

        // Save TSOS: associate Updated TSOS with Hit
        roMHitTSOS[oHitIter->get()] = oTSOSLast;
      } while( oHitIter != roTRACK.recHitsBegin());

      // Check if any error occured while getting Updated TSOS for hits. Empty
      // map in case any error occured.
      if( !bResult) {
        edm::LogError( "reco::GetUpdatedTSOSDebug::operator()") 
          << "\tError occured while trying to get Updated TSOS for hit #"
          << ( roMHitTSOS.size() + 1) << " out of "
          << roTRACK.recHitsSize() << " hits";

        
        edm::LogVerbatim( "TrackHitsInfo::analyze()")
          << TrackOstream( roTRACK);

        roMHitTSOS.clear();
      }
    } else {
      edm::LogError( "reco::GetUpdatedTSOSDebug::operator()") 
        << "\tINVALID updated TSOS of the Initial TSOS";
      bResult = false;
    }
  } else {
    edm::LogError( "reco::GetUpdatedTSOSDebug::operator()") 
      << "\tINVALID Initial TSOS";
    bResult = false;
  }

  LogDebug( "reco::GetUpdatedTSOSDebug::operator()") 
    << " * Track Hits #/Saved TSOS #        "
    << roTRACK.recHitsSize() << '/' << roMHitTSOS.size();

  return bResult;
}
