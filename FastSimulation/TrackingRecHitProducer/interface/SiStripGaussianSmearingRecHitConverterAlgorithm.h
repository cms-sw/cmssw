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
  explicit SiStripGaussianSmearingRecHitConverterAlgorithm(edm::ParameterSet& pset);
  virtual ~SiStripGaussianSmearingRecHitConverterAlgorithm() {};
  
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
  void run( const PSimHit& simHit , const HepSymMatrix& localPositionResolution );
  
private:
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
