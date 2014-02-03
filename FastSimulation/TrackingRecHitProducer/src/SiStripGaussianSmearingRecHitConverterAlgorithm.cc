/** SiStripGaussianSmearingRecHitConverterAlgorithm.cc
 * --------------------------------------------------------------
 * Description:  see SiStripGaussianSmearingRecHitConverterAlgorithm.h
 * Authors:  R. Ranieri (CERN), M. Galanti
 * History: Sep 27, 2006 -  initial version
 * --------------------------------------------------------------
 */


// SiStrip Gaussian Smearing
#include "SiStripGaussianSmearingRecHitConverterAlgorithm.h"

// Famos
#include "FastSimulation/Utilities/interface/RandomEngine.h"

// STL

// #define FAMOS_DEBUG

SiStripGaussianSmearingRecHitConverterAlgorithm::SiStripGaussianSmearingRecHitConverterAlgorithm
(const RandomEngine* engine) : random(engine) {}


void 
SiStripGaussianSmearingRecHitConverterAlgorithm::smearHit(const PSimHit& simHit, 
							  double localPositionResolutionX, 
							  double localPositionResolutionY,	
							  double localPositionResolutionZ,
                                                          double boundX,
                                                          double boundY) 
{

  // Gaussian Smearing
  // x is smeared,
  // y fixed at the centre of the strip, 
  // z fixed at the centre of the active area
  // For the double-sided modules it will be a problem 
  // for the RecHit matcher (starting from these PSimHits)
  //
  do {
    thePosition = 
      Local3DPoint(random->gaussShoot((double)simHit.localPosition().x(), localPositionResolutionX), 
                  (double)simHit.localPosition().y(),
                  0.);
#ifdef FAMOS_DEBUG
    std::cout << " Detector bounds: "
              << "\t\tx = " << boundX
              << "\ty = " << boundY
              << std::endl;
    std::cout << " Generated local position "
              << "\tx = " << thePosition.x()
              << "\ty = " << thePosition.y()
              << std::endl;       
#endif  
  } while(fabs(thePosition.x()) > boundX);
  
  //
  thePositionX = thePosition.x();
  thePositionY = thePosition.y();
  thePositionZ = thePosition.z();
  //
  theErrorX = localPositionResolutionX;
  theErrorY = localPositionResolutionY;
  theErrorZ = localPositionResolutionZ;

  theError = LocalError( theErrorX * theErrorX,
			 0.0,
			 theErrorY * theErrorY ); 
  // Local Error is 2D: (xx,xy,yy), square of sigma 
  // in first an third position as for resolution matrix
  //
}
