#ifndef FastSimulation_TrackingRecHitProducer_SiTrackerGaussianSmearingRecHitConverter_h
#define FastSimulation_TrackingRecHitProducer_SiTrackerGaussianSmearingRecHitConverter_h

//---------------------------------------------------------------------------
//! \class SiTrackerGaussianSmearingRecHits
//!
//! \brief EDProducer to create RecHits from PSimHits with Gaussian smearing
//!
//---------------------------------------------------------------------------

// Framework
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

// PSimHit
#include "SimDataFormats/TrackingHit/interface/PSimHit.h"
#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"

// Data Formats
#include "SimDataFormats/CrossingFrame/interface/MixCollection.h"
#include "FastSimDataFormats/External/interface/FastTrackerClusterCollection.h"
#include "DataFormats/TrackerRecHit2D/interface/SiTrackerGSRecHit2DCollection.h"
#include "DataFormats/TrackerRecHit2D/interface/SiTrackerGSMatchedRecHit2DCollection.h"
#include "DataFormats/GeometryVector/interface/Point3DBase.h"
#include "DataFormats/GeometrySurface/interface/LocalError.h"

// For Dead Channels 
#include "CondFormats/SiPixelObjects/interface/SiPixelQuality.h"
#include "CondFormats/DataRecord/interface/SiPixelQualityRcd.h"

// STL
#include <vector>
#include <map>
#include <string>

class TFile;
class TH1F;
class TrackerGeometry;
class SiPixelGaussianSmearingRecHitConverterAlgorithm;
class SiStripGaussianSmearingRecHitConverterAlgorithm;

namespace edm { 
  class ParameterSet;
  class Event;
  class EventSetup;
}

class RandomEngineAndDistribution;
class TrackerTopology;

class SiTrackerGaussianSmearingRecHitConverter : public edm::stream::EDProducer <>
{
 public:
  //--- Constructor, virtual destructor (just in case)
  explicit SiTrackerGaussianSmearingRecHitConverter(const edm::ParameterSet& conf);
  virtual ~SiTrackerGaussianSmearingRecHitConverter();
  
  //--- The top-level event method.
  virtual void produce(edm::Event& e, const edm::EventSetup& c) override;
  
  // Begin Run
  virtual void beginRun(edm::Run const& run, const edm::EventSetup & es) override;
  
  //  void smearHits(MixCollection<PSimHit>& input,
  void smearHits(const edm::PSimHitContainer& input,
  //  void smearHits(edm::Handle<std::vector<PSimHit> >& input,
                 std::map<unsigned, edm::OwnVector<SiTrackerGSRecHit2D> >& theRecHits,
                 std::map<unsigned, edm::OwnVector<FastTrackerCluster> >& theClusters,
		 const TrackerTopology *tTopo,
                 RandomEngineAndDistribution const*);

 void  matchHits( std::map<unsigned, edm::OwnVector<SiTrackerGSRecHit2D> >& theRecHits, 
		  std::map<unsigned, edm::OwnVector<SiTrackerGSMatchedRecHit2D> >& matchedMap);//,
		  //		  MixCollection<PSimHit>& simhits);
   //		  const edm::PSimHitContainer& simhits);
		  //		  std::vector<PSimHit>& simhits); 
		  //		  edm::Handle<std::vector<PSimHit> >& simhits);

  void loadRecHits(std::map<unsigned,edm::OwnVector<SiTrackerGSRecHit2D> >& theRecHits, 
		   SiTrackerGSRecHit2DCollection& theRecHitCollection) const;

  void loadMatchedRecHits(std::map<unsigned,edm::OwnVector<SiTrackerGSMatchedRecHit2D> >& theRecHits, 
		   SiTrackerGSMatchedRecHit2DCollection& theRecHitCollection) const;

  void loadClusters(std::map<unsigned,edm::OwnVector<FastTrackerCluster> >& theClusterMap, 
                    FastTrackerClusterCollection& theClusterCollection) const;
  
  private:
  //
  bool gaussianSmearing(const PSimHit& simHit, 
			Local3DPoint& position , 
			LocalError& error, 
			unsigned& alphaMult, 
			unsigned& betaMult,
			const TrackerTopology *tTopo,
                        RandomEngineAndDistribution const*);
  //
  void loadPixelData();
  //
  void loadPixelData(TFile* pixelDataFile, unsigned int nMultiplicity, std::string histName,
		     std::vector<TH1F*>& theMultiplicityCumulativeProbabilities, bool bigPixels = false);
  //
  //
  // parameters
  //  std::vector<edm::InputTag> trackerContainers;
  edm::InputTag simHitLabel;
  edm::EDGetTokenT<edm::PSimHitContainer> simHitToken; 
  edm::ParameterSet pset_;
  double deltaRaysPCut; // GeV/c
  bool trackingPSimHits; // in case it is true make RecHit = replica of PSimHit without errors (1 um)
  //
  bool doMatching;
  bool doDisableChannels;
  

  // Vector with the list of dead modules
  std::vector<SiPixelQuality::disabledModuleType> * disabledModules;
  unsigned int numberOfDisabledModules;


  // Switch between old (ORCA) and new (CMSSW) pixel parameterization
  bool useCMSSWPixelParameterization;
  double ElectronsPerADC;  
  double GevPerElectron;

  const TrackerGeometry* geometry;
  const TrackerGeometry* misAlignedGeometry;
  //
  // Pixel
  std::string thePixelMultiplicityFileName;
  std::string thePixelBarrelResolutionFileName;
  std::string thePixelForwardResolutionFileName;
  TFile* thePixelDataFile;
  TFile* thePixelBarrelResolutionFile;
  TFile* thePixelForwardResolutionFile;
  //
  // multiplicity bins
  unsigned int nAlphaBarrel, nBetaBarrel, nAlphaForward, nBetaForward;
  // internal vector: bins ; external vector: multiplicity
  std::vector<TH1F*> theBarrelMultiplicityAlphaCumulativeProbabilities;
  std::vector<TH1F*> theBarrelMultiplicityBetaCumulativeProbabilities;
  std::vector<TH1F*> theForwardMultiplicityAlphaCumulativeProbabilities;
  std::vector<TH1F*> theForwardMultiplicityBetaCumulativeProbabilities;
   // resolution bins
  double resAlphaBarrel_binMin , resAlphaBarrel_binWidth;
  unsigned int resAlphaBarrel_binN;
  double resBetaBarrel_binMin  , resBetaBarrel_binWidth;
  unsigned int resBetaBarrel_binN;
  double resAlphaForward_binMin , resAlphaForward_binWidth;
  unsigned int resAlphaForward_binN;
  double resBetaForward_binMin  , resBetaForward_binWidth;
  unsigned int resBetaForward_binN;
  //
  double theHitFindingProbability_PXB;
  double theHitFindingProbability_PXF;
  //
  // Strips
  // TIB
  double localPositionResolution_TIB1x; // cm
  double localPositionResolution_TIB1y; // cm
  double localPositionResolution_TIB2x; // cm
  double localPositionResolution_TIB2y; // cm
  double localPositionResolution_TIB3x; // cm
  double localPositionResolution_TIB3y; // cm
  double localPositionResolution_TIB4x; // cm
  double localPositionResolution_TIB4y; // cm
  //
  double theHitFindingProbability_TIB1;
  double theHitFindingProbability_TIB2;
  double theHitFindingProbability_TIB3;
  double theHitFindingProbability_TIB4;
  //
  // TID
  double localPositionResolution_TID1x; // cm
  double localPositionResolution_TID1y; // cm
  double localPositionResolution_TID2x; // cm
  double localPositionResolution_TID2y; // cm
  double localPositionResolution_TID3x; // cm
  double localPositionResolution_TID3y; // cm
  //
  double theHitFindingProbability_TID1;
  double theHitFindingProbability_TID2;
  double theHitFindingProbability_TID3;
  //
  // TOB
  double localPositionResolution_TOB1x; // cm
  double localPositionResolution_TOB1y; // cm
  double localPositionResolution_TOB2x; // cm
  double localPositionResolution_TOB2y; // cm
  double localPositionResolution_TOB3x; // cm
  double localPositionResolution_TOB3y; // cm
  double localPositionResolution_TOB4x; // cm
  double localPositionResolution_TOB4y; // cm
  double localPositionResolution_TOB5x; // cm
  double localPositionResolution_TOB5y; // cm
  double localPositionResolution_TOB6x; // cm
  double localPositionResolution_TOB6y; // cm
  //
  double theHitFindingProbability_TOB1;
  double theHitFindingProbability_TOB2;
  double theHitFindingProbability_TOB3;
  double theHitFindingProbability_TOB4;
  double theHitFindingProbability_TOB5;
  double theHitFindingProbability_TOB6;
  //
  // TEC
  double localPositionResolution_TEC1x; // cm
  double localPositionResolution_TEC1y; // cm
  double localPositionResolution_TEC2x; // cm
  double localPositionResolution_TEC2y; // cm
  double localPositionResolution_TEC3x; // cm
  double localPositionResolution_TEC3y; // cm
  double localPositionResolution_TEC4x; // cm
  double localPositionResolution_TEC4y; // cm
  double localPositionResolution_TEC5x; // cm
  double localPositionResolution_TEC5y; // cm
  double localPositionResolution_TEC6x; // cm
  double localPositionResolution_TEC6y; // cm
  double localPositionResolution_TEC7x; // cm
  double localPositionResolution_TEC7y; // cm
  //
  double theHitFindingProbability_TEC1;
  double theHitFindingProbability_TEC2;
  double theHitFindingProbability_TEC3;
  double theHitFindingProbability_TEC4;
  double theHitFindingProbability_TEC5;
  double theHitFindingProbability_TEC6;
  double theHitFindingProbability_TEC7;
  //
  // valid for all the detectors
  double localPositionResolution_z; // cm
  //
  //  typedef std::map<unsigned int, std::vector<PSimHit>,std::less<unsigned int> > simhit_map;

  // Pixel Error Parametrization (barrel)
  SiPixelGaussianSmearingRecHitConverterAlgorithm* thePixelBarrelParametrization;
  // Pixel Error Parametrization (barrel)
  SiPixelGaussianSmearingRecHitConverterAlgorithm* thePixelEndcapParametrization;
  // Si Strip Error parametrization (generic)
  SiStripGaussianSmearingRecHitConverterAlgorithm* theSiStripErrorParametrization;

  // Temporary RecHit map
  //  std::map< DetId, edm::OwnVector<SiTrackerGSRecHit2D> > temporaryRecHits;

  // Local correspondence between RecHits and SimHits
  //  typedef MixCollection<PSimHit>::iterator SimHiterator;
  typedef edm::PSimHitContainer::const_iterator SimHiterator;
  std::vector<SimHiterator> correspondingSimHit;

  typedef SiTrackerGSRecHit2D::ClusterRef ClusterRef;
  typedef SiTrackerGSRecHit2D::ClusterRefProd ClusterRefProd;
  // Added for cluster reference
  ClusterRefProd FastTrackerClusterRefProd;

  
};


#endif
