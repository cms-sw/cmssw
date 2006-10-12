#ifndef FastSimulation_TrackingRecHitProducer_SiPixelGaussianSmearingRecHitConverterAlgorithm_h
#define FastSimulation_TrackingRecHitProducer_SiPixelGaussianSmearingRecHitConverterAlgorithm_h

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

// Geometry
#include "Geometry/CommonDetUnit/interface/GeomDetType.h"

// Vectors
#include "Geometry/Vector/interface/Point3DBase.h"
#include "Geometry/Surface/interface/LocalError.h"

// STL
#include <vector>
#include <string>

class TFile;
class TH1F;

class SiPixelGaussianSmearingRecHitConverterAlgorithm {
public:
  //--- Constructor, virtual destructor (just in case)
  explicit SiPixelGaussianSmearingRecHitConverterAlgorithm(edm::ParameterSet pset,
							   const PSimHit& simHit,
							   GeomDetType::SubDetector pixelPart, unsigned int layer,
							   std::vector<TH1F*> theAlphaMultiplicityCumulativeProbabilities,
							   std::vector<TH1F*> theBetaMultiplicityCumulativeProbabilities,
							   TFile* pixelResolutionFile);
  virtual ~SiPixelGaussianSmearingRecHitConverterAlgorithm() {};
  
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
  void run( const PSimHit& simHit,
	    std::vector<TH1F*> theAlphaMultiplicityCumulativeProbabilities,
	    std::vector<TH1F*> theBetaMultiplicityCumulativeProbabilities  );
  //
  // parameters
  edm::ParameterSet pset_;
  int theVerboseLevel;
  // resolution bins
  double resAlpha_binMin , resAlpha_binWidth;
  unsigned int resAlpha_binN;
  double resBeta_binMin  , resBeta_binWidth;
  unsigned int resBeta_binN;
  //
  // Useful private members
  TFile* thePixelResolutionFile;
  GeomDetType::SubDetector thePixelPart;
  unsigned int theLayer;
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
