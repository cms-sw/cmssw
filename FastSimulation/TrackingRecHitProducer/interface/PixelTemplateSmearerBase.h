#ifndef FastSimulation_TrackingRecHitProducer_PixelTemplateSmearerBase_h
#define FastSimulation_TrackingRecHitProducer_PixelTemplateSmearerBase_h

//---------------------------------------------------------------------------
//! \class SiTrackerGaussianSmearingRecHits
//!
//! \brief EDProducer to create RecHits from PSimHits with Gaussian smearing
//!
//---------------------------------------------------------------------------

// FastSim stuff
#include "FastSimulation/TrackingRecHitProducer/interface/TrackingRecHitAlgorithm.h"

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

class PixelTemplateSmearerBase : public TrackingRecHitAlgorithm {
public:
  //--- Use this type to keep track of groups of hits that need to be merged:
  typedef std::vector<const PSimHit*> MergeGroup;

  //--- Constructor, virtual destructor (just in case)
  explicit PixelTemplateSmearerBase(  const std::string& name,
				      const edm::ParameterSet& config,
				      edm::ConsumesCollector& consumesCollector );
  
  //--- Destructor
  virtual ~PixelTemplateSmearerBase();
  
  //  &&& Why do we need this?
  void setPixelPart(GeomDetType::SubDetector subdet) { thePixelPart = subdet; }
  
  //--- Process all hits on this DetUnit.  Calls the other two processXYZ() methods.
  TrackingRecHitProductPtr process(TrackingRecHitProductPtr product) const ;

  //--- Process all unmerged hits.  Calls smearHit() for each.
  TrackingRecHitProductPtr processUnmergedHits( std::vector< const PSimHit* > & unmergedHits,
						TrackingRecHitProductPtr process ) const;

  //--- Process all groups of merged hits.
  TrackingRecHitProductPtr processMergeGroups( std::vector< MergeGroup* > & mergeGroups,
					       TrackingRecHitProductPtr process ) const;

  //--- Process one umerged hit.  The core of the code :)
  SiTrackerGSRecHit2D smearHit( const PSimHit& simHit, const PixelGeomDetUnit* detUnit, 
				const double boundX, const double boundY,
				RandomEngineAndDistribution const*) const;

  //--- Process one merge group.
  void smearMergeGroup( MergeGroup* mg) const;

  //--- Method to decide if the two hits on the same DetUnit are merged, or not.
  bool hitsMerge(const PSimHit& simHit1,const PSimHit& simHit2) const;


protected:
  // Switch between old (ORCA) and new (CMSSW) pixel parameterization
  bool useCMSSWPixelParameterization;
  // template object
  std::vector< SiPixelTemplateStore > thePixelTemp_;
  int tempId;
  //
  bool isFlipped(const PixelGeomDetUnit* theDet) const;
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

  // &&& Petar: must be mutable, otherwise operator[] on the map won't
  // &&&        work (and I have no idea why)
  mutable std::map<unsigned int, const SimpleHistogramGenerator*> theXHistos;
  mutable std::map<unsigned int, const SimpleHistogramGenerator*> theYHistos;

  TFile* thePixelResolutionFile1;
  std::string thePixelResolutionFileName1;
  //Splite the resolution histograms for cvs uploading
  TFile* thePixelResolutionFile2;
  std::string thePixelResolutionFileName2;
  TFile* thePixelResolutionFile3;
  std::string thePixelResolutionFileName3;
  TFile* probfile;
  std::string probfileName;

  unsigned int theLayer;
};
#endif
