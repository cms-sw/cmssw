#ifndef FastSimulation_TrackingRecHitProducer_SiStripGaussianSmearingRecHitConverterAlgorithm_h
#define FastSimulation_TrackingRecHitProducer_SiStripGaussianSmearingRecHitConverterAlgorithm_h

//---------------------------------------------------------------------------
//! \class SiTrackerGaussianSmearingRecHits
//!
//! \brief EDProducer to create RecHits from PSimHits with Gaussian smearing
//!
//---------------------------------------------------------------------------

// Framework
#include "FWCore/ParameterSet/interface/ParameterSet.h"

// PSimHit
#include "SimDataFormats/TrackingHit/interface/PSimHit.h"

// Vectors
#include "Geometry/Vector/interface/Point3DBase.h"
#include "Geometry/Surface/interface/LocalError.h"

// CLHEP
#include "Geometry/CommonDetAlgo/interface/AlgebraicObjects.h"

// STL
#include <vector>
#include <string>

class SiStripGaussianSmearingRecHitConverterAlgorithm {

public:
  //--- Constructor, virtual destructor (just in case)
  explicit SiStripGaussianSmearingRecHitConverterAlgorithm(edm::ParameterSet pset,
							   const PSimHit& simHit,
							   HepSymMatrix localPositionResolution );
  virtual ~SiStripGaussianSmearingRecHitConverterAlgorithm() {};
  
  // return results
  Local3DPoint getPosition()  {return thePosition;}
  double       getPositionX() {return thePositionX;}
  double       getPositionY() {return thePositionY;}
  double       getPositionZ() {return thePositionZ;}
  LocalError   getError()     {return theError;}
  double       getErrorX()    {return theErrorX;}
  double       getErrorY()    {return theErrorY;}
  double       getErrorZ()    {return theErrorZ;}
  //
  
private:
  //
  void run( const PSimHit& simHit , HepSymMatrix localPositionResolution );
  //
  // parameters
  edm::ParameterSet pset_;
  int theVerboseLevel;
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
  
};


#endif
