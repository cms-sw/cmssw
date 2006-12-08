/** SiStripGaussianSmearingRecHitConverterAlgorithm.cc
 * --------------------------------------------------------------
 * Description:  see SiStripGaussianSmearingRecHitConverterAlgorithm.h
 * Authors:  R. Ranieri (CERN)
 * History: Sep 27, 2006 -  initial version
 * --------------------------------------------------------------
 */


// SiStrip Gaussian Smearing
#include "FastSimulation/TrackingRecHitProducer/interface/SiStripGaussianSmearingRecHitConverterAlgorithm.h"
#include "CommonTools/Statistics/interface/RandomMultiGauss.h"

// Geometry
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/CommonDetUnit/interface/GeomDetUnit.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "Geometry/CommonDetUnit/interface/GeomDetType.h"

// STL
#include <vector>
#include <memory>
#include <string>
#include <iostream>

// MessageLogger
#include "FWCore/MessageLogger/interface/MessageLogger.h"

SiStripGaussianSmearingRecHitConverterAlgorithm::SiStripGaussianSmearingRecHitConverterAlgorithm( 
  edm::ParameterSet& pset) :
  pset_(pset) {

  //--- Algorithm's verbosity
  theVerboseLevel = pset.getUntrackedParameter<int>("VerboseLevel",0);
  //
}  


void SiStripGaussianSmearingRecHitConverterAlgorithm::run(
  const PSimHit& simHit, 
  const HepSymMatrix& localPositionResolution) {
  // SiStrip random positioning
  if (theVerboseLevel > 2) {
    LogDebug("SiTrackerGaussianSmearingRecHits") << " start from position " << thePosition << std::endl;
  }
  // Gaussian Smearing
  HepVector nominalPosition(3);
  nominalPosition[0] = simHit.localPosition().x();
  nominalPosition[1] = simHit.localPosition().y();
  nominalPosition[2] = simHit.localPosition().z();
  //
  RandomMultiGauss randomGauss(nominalPosition, localPositionResolution);
  nominalPosition = randomGauss.fire();
  // change something: y must be 0 for ss detectors (or every sensor rphi or stereo), z must be always 0
  thePosition = Local3DPoint(nominalPosition[0],0.,0.); // y fixed at the centre of the strip, z fixed at the centre of the active area
  // for the double-sided modules it will be a problem for the RecHit matcher (starting from these PSimHits)
  if (theVerboseLevel > 2) {
    LogDebug("SiTrackerGaussianSmearingRecHits") << " gaussian smearing position " << thePosition << std::endl;
  }
  //
  thePositionX = thePosition.x();
  thePositionY = thePosition.y();
  thePositionZ = thePosition.z();
  //
  theErrorX = sqrt( localPositionResolution[0][0] );
  theErrorY = sqrt( localPositionResolution[1][1] );
  theErrorZ = sqrt( localPositionResolution[2][2] );
  theError = LocalError( theErrorX * theErrorX,
			 0.0,
			 theErrorY * theErrorY ); // Local Error is 2D: (xx,xy,yy), square of sigma in first an third position as for resolution matrix
  //
}
