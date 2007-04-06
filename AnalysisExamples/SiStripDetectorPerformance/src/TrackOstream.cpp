// Author : Samvel Khalatian (samvel at fnal dot gov)
// Created: 03/29/07
// Licence: GPL

#include <iostream>
#include <ostream>

#include "AnalysisExamples/SiStripDetectorPerformance/interface/DetIdOstream.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHitFwd.h"

#include "AnalysisExamples/SiStripDetectorPerformance/interface/TrackOstream.h"

std::ostream &operator<<( std::ostream &roOut, 
                          const TrackOstream &roTO) {
  roOut << "--[ Track ]-----------------------------------------------"
        << std::endl;
  roOut << "\t* innerMomentum(x, y, z): (" 
        << roTO.roTRACK.innerMomentum().X() << ", "
        << roTO.roTRACK.innerMomentum().Y() << ", "
        << roTO.roTRACK.innerMomentum().Z() << ")"
        << std::endl;
  roOut << "\t* outerMomentum(x, y, z): (" 
        << roTO.roTRACK.outerMomentum().X() << ", "
        << roTO.roTRACK.outerMomentum().Y() << ", "
        << roTO.roTRACK.outerMomentum().Z() << ")"
        << std::endl;
  roOut << "\t* innerDetId: "  << DetIdOstream( DetId( roTO.roTRACK.innerDetId()))
        << std::endl;
  roOut << "\t* outterDetId: " << DetIdOstream( DetId( roTO.roTRACK.outerDetId())) 
        << std::endl;
  roOut << std::endl
        << "\t* Hits [" << roTO.roTRACK.recHitsSize() << " total]:"
        << std::endl;

  // Loop over Tracks Hits
  int nHit = 1;
  for( trackingRecHit_iterator oHitIter = roTO.roTRACK.recHitsBegin();
       oHitIter != roTO.roTRACK.recHitsEnd();
       ++oHitIter, ++nHit) {

      roOut << "\t\t" << nHit << "  " 
            << DetIdOstream( DetId( ( *oHitIter)->geographicalId()))
            << std::endl;
  } // End loop over Tracks hits

  roOut << "----------------------------------------------------------"
        << std::endl;

  return roOut;
}
