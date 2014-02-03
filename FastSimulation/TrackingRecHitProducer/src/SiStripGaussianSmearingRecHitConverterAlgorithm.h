#ifndef FastSimulation_TrackingRecHitProducer_SiStripGaussianSmearingRecHitConverterAlgorithm_h
#define FastSimulation_TrackingRecHitProducer_SiStripGaussianSmearingRecHitConverterAlgorithm_h

//---------------------------------------------------------------------------
//! \class SiTrackerGaussianSmearingRecHits
//!
//! \brief EDProducer to create RecHits from PSimHits with Gaussian smearing
//!
//---------------------------------------------------------------------------

// PSimHit
#include "SimDataFormats/TrackingHit/interface/PSimHit.h"

// Vectors
#include "DataFormats/GeometryVector/interface/Point3DBase.h"
#include "DataFormats/GeometrySurface/interface/LocalError.h"
//#include "DataFormats/CLHEP/interface/AlgebraicObjects.h"

// STL
#include <string>

class RandomEngine;

class SiStripGaussianSmearingRecHitConverterAlgorithm {

public:
  //--- Constructor, virtual destructor (just in case)
  explicit SiStripGaussianSmearingRecHitConverterAlgorithm(const RandomEngine* engine);
  virtual ~SiStripGaussianSmearingRecHitConverterAlgorithm() {;}
  
  // return results
  const Local3DPoint& getPosition() const {return thePosition;}
  double             getPositionX() const {return thePositionX;}
  double             getPositionY() const {return thePositionY;}
  double             getPositionZ() const {return thePositionZ;}
  const LocalError&  getError()     const {return theError;}
  double             getErrorX()    const {return theErrorX;}
  double             getErrorY()    const {return theErrorY;}
  double             getErrorZ()    const {return theErrorZ;}
  //
  void smearHit( const PSimHit& simHit , 
		 double localPositionResolutionX,
		 double localPositionResolutionY,
		 double localPositionResolutionZ,
                 double boundX,
                 double boundY);
  
private:

  // output
  Local3DPoint thePosition;
  double       thePositionX;
  double       thePositionY;
  double       thePositionZ;
  LocalError   theError;
  double       theErrorX;
  double       theErrorY;
  double       theErrorZ;
  //

  // The random engine
  const RandomEngine* random;
  
};


#endif
