#ifndef FastSimulation_TrackingRecHitProducer_SiPixelGaussianSmearingRecHitConverterAlgorithm_h
#define FastSimulation_TrackingRecHitProducer_SiPixelGaussianSmearingRecHitConverterAlgorithm_h

//---------------------------------------------------------------------------
//! \class SiTrackerGaussianSmearingRecHits
//!
//! \brief EDProducer to create RecHits from PSimHits with Gaussian smearing
//!
//---------------------------------------------------------------------------


//Framework
#include "FWCore/ParameterSet/interface/ParameterSet.h"

// PSimHit
#include "SimDataFormats/TrackingHit/interface/PSimHit.h"

// Geometry
#include "Geometry/CommonDetUnit/interface/GeomDetType.h"
#include "Geometry/TrackerGeometryBuilder/interface/PixelGeomDetUnit.h"
// template object
#include "RecoLocalTracker/SiPixelRecHits/interface/SiPixelTemplate.h"

// Vectors
#include "DataFormats/GeometryVector/interface/Point3DBase.h"
#include "DataFormats/GeometrySurface/interface/LocalError.h"

// STL
#include <vector>
#include <string>

class TFile;
class RandomEngineAndDistribution;
class SimpleHistogramGenerator;

class SiPixelGaussianSmearingRecHitConverterAlgorithm {
public:
  //--- Constructor, virtual destructor (just in case)
  explicit SiPixelGaussianSmearingRecHitConverterAlgorithm(		   
   const edm::ParameterSet& pset,
   GeomDetType::SubDetector pixelPart);

  // destructor
  virtual ~SiPixelGaussianSmearingRecHitConverterAlgorithm();
  
  // return results
  Local3DPoint getPosition()  {return thePosition;}
  double       getPositionX() {return thePositionX;}
  double       getPositionY() {return thePositionY;}
  double       getPositionZ() {return thePositionZ;}
  LocalError   getError()     {return theError;}
  double       getErrorX()    {return theErrorX;}
  double       getErrorY()    {return theErrorY;}
  double       getErrorZ()    {return theErrorZ;}
  unsigned int getPixelMultiplicityAlpha() {return theClslenx;}
  unsigned int getPixelMultiplicityBeta()  {return theClsleny;}
  //
  
  //
  void smearHit( const PSimHit& simHit, const PixelGeomDetUnit* detUnit, const double boundX, const double boundY,
                 RandomEngineAndDistribution const*);

private:
  // Switch between old (ORCA) and new (CMSSW) pixel parameterization
  bool useCMSSWPixelParameterization;
  // template object
  std::vector< SiPixelTemplateStore > thePixelTemp_;
  int tempId;
  //
  bool isFlipped(const PixelGeomDetUnit* theDet) const;
  void initializeBarrel();
  void initializeForward();
  //isForward, true for forward, false for barrel
  bool isForward;
  //
  //
  // resolution bins
  double rescotAlpha_binMin , rescotAlpha_binWidth;
  unsigned int rescotAlpha_binN;
  double rescotBeta_binMin  , rescotBeta_binWidth;
  unsigned int rescotBeta_binN;
  int resqbin_binMin, resqbin_binWidth;
  unsigned int resqbin_binN;
  //
  edm::ParameterSet pset_;
  // Useful private members
  GeomDetType::SubDetector thePixelPart;

  std::map<unsigned,const SimpleHistogramGenerator*> theXHistos;
  std::map<unsigned,const SimpleHistogramGenerator*> theYHistos;

  TFile* thePixelResolutionFile1;
  std::string thePixelResolutionFileName1;
  //Splite the resolution histograms for cvs uploading
  TFile* thePixelResolutionFile2;
  std::string thePixelResolutionFileName2;
  TFile* thePixelResolutionFile3;
  std::string thePixelResolutionFileName3;

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
  unsigned int theClslenx;
  unsigned int theClsleny;
};
#endif
