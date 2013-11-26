/** SiTrackerGaussianSmearingRecHitConverter.cc
 * --------------------------------------------------------------
 * Description:  see SiTrackerGaussianSmearingRecHitConverter.h
 * Authors:  R. Ranieri (CERN), P. Azzi, A. Schmidt, M. Galanti
 * History: Sep 27, 2006 -  initial version
 * --------------------------------------------------------------
 */
//PAT
//FastTrackerCluster
#include "FastSimDataFormats/External/interface/FastTrackerCluster.h"

// SiTracker Gaussian Smearing
#include "SiTrackerGaussianSmearingRecHitConverter.h"

// SiPixel Gaussian Smearing
#include "SiPixelGaussianSmearingRecHitConverterAlgorithm.h"

// SiStripGaussianSmearing
#include "SiStripGaussianSmearingRecHitConverterAlgorithm.h"

// Geometry and magnetic field
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/CommonDetUnit/interface/GeomDetUnit.h"
#include "Geometry/TrackerGeometryBuilder/interface/GluedGeomDet.h"
#include "Geometry/TrackerGeometryBuilder/interface/StripGeomDetUnit.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "MagneticField/Engine/interface/MagneticField.h"

// Data Formats
#include "DataFormats/Common/interface/Handle.h"
#include "SimDataFormats/CrossingFrame/interface/CrossingFrame.h" 	 

// Framework
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/RandomNumberGenerator.h"
#include "FWCore/Utilities/interface/Exception.h"

// Numbering scheme
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "DataFormats/GeometryCommonDetAlgo/interface/ErrorFrameTransformer.h"
#include "DataFormats/TrackingRecHit/interface/AlignmentPositionError.h"

//For Pileup events
#include "SimDataFormats/EncodedEventId/interface/EncodedEventId.h"

// Random engine
#include "FastSimulation/Utilities/interface/RandomEngine.h"

// topology

// the rec hit matcher
#include "GSRecHitMatcher.h"

// STL
#include <memory>

// ROOT
#include <TFile.h>
#include <TH1F.h>


//#define FAMOS_DEBUG

SiTrackerGaussianSmearingRecHitConverter::SiTrackerGaussianSmearingRecHitConverter(
  edm::ParameterSet const& conf) 
  : pset_(conf)
{
  std::cout << "geometry = " << geometry << std::endl;

  thePixelDataFile = 0;
  thePixelBarrelResolutionFile = 0;
  thePixelForwardResolutionFile = 0;
  thePixelBarrelParametrization = 0;
  thePixelEndcapParametrization = 0;
  theSiStripErrorParametrization = 0;
  numberOfDisabledModules = 0;

  random = 0;

  std::cout << "SiTrackerGaussianSmearingRecHitConverter instantiated" << std::endl;

  // Initialize the random number generator service
  edm::Service<edm::RandomNumberGenerator> rng;
  if ( ! rng.isAvailable() ) {
    throw cms::Exception("Configuration")
      << "SiTrackerGaussianSmearingRecHitConverter requires the RandomGeneratorService\n"
         "which is not present in the configuration file.\n"
         "You must add the service in the configuration file\n"
         "or remove the module that requires it";
  }

  random = new RandomEngine(&(*rng));
  //PAT
  produces<FastTrackerClusterCollection>("TrackerClusters");

  produces<SiTrackerGSRecHit2DCollection>("TrackerGSRecHits");
  produces<SiTrackerGSMatchedRecHit2DCollection>("TrackerGSMatchedRecHits");

  //--- switch to have RecHit == PSimHit
  trackingPSimHits = conf.getParameter<bool>("trackingPSimHits");
  trackingPSimHitsEqualSmearing = conf.getParameter<bool>("trackingPSimHitsEqualSmearing");
  if(trackingPSimHits) std::cout << "### trackingPSimHits chosen " << trackingPSimHits << std::endl;
  if(trackingPSimHitsEqualSmearing) {
    std::cout << "### trackingPSimHitsEqualSmearing chosen " << trackingPSimHitsEqualSmearing << std::endl;
    localPositionResolution_x = conf.getParameter<double>("GeneralResX");
    localPositionResolution_y = conf.getParameter<double>("GeneralResY");
    localPositionResolutionBar_x = conf.getParameter<std::vector<double> >("GeneralBarResX");
    localPositionResolutionBar_y = conf.getParameter<std::vector<double> >("GeneralBarResY");
    localPositionResolutionFwd_x = conf.getParameter<std::vector<double> >("GeneralFwdResX");
    localPositionResolutionFwd_y = conf.getParameter<std::vector<double> >("GeneralFwdResY");
  }

  //--- PSimHit Containers
  trackerContainers.clear();
  trackerContainers = conf.getParameter<std::vector<edm::InputTag> >("ROUList");
  //--- delta rays p cut [GeV/c] to filter PSimHits with p>
  deltaRaysPCut = conf.getParameter<double>("DeltaRaysMomentumCut");


  // switch on/off matching
  doMatching = conf.getParameter<bool>("doRecHitMatching");

  // disable/enable dead channels
  doDisableChannels = conf.getParameter<bool>("killDeadChannels");

  // Switch between old (ORCA) and new (CMSSW) pixel parameterization
  if (!trackingPSimHitsEqualSmearing && !trackingPSimHits) {
    useCMSSWPixelParameterization = conf.getParameter<bool>("UseCMSSWPixelParametrization");
#ifdef FAMOS_DEBUG
    std::cout << (useCMSSWPixelParameterization? "CMSSW" : "ORCA") << " pixel parametrization chosen in config file." << std::endl;
#endif
  }

  //Clusters
  GevPerElectron =  conf.getParameter<double>("GevPerElectron");
  ElectronsPerADC =  conf.getParameter<double>("ElectronsPerADC");
  
  if (!trackingPSimHitsEqualSmearing && !trackingPSimHits) {
    //
    // TIB
    localPositionResolution_TIB1x = conf.getParameter<double>("TIB1x");
    localPositionResolution_TIB1y = conf.getParameter<double>("TIB1y");
    localPositionResolution_TIB2x = conf.getParameter<double>("TIB2x");
    localPositionResolution_TIB2y = conf.getParameter<double>("TIB2y");
    localPositionResolution_TIB3x = conf.getParameter<double>("TIB3x");
    localPositionResolution_TIB3y = conf.getParameter<double>("TIB3y");
    localPositionResolution_TIB4x = conf.getParameter<double>("TIB4x");
    localPositionResolution_TIB4y = conf.getParameter<double>("TIB4y");
    //
    // TID
    localPositionResolution_TID1x = conf.getParameter<double>("TID1x");
    localPositionResolution_TID1y = conf.getParameter<double>("TID1y");
    localPositionResolution_TID2x = conf.getParameter<double>("TID2x");
    localPositionResolution_TID2y = conf.getParameter<double>("TID2y");
    localPositionResolution_TID3x = conf.getParameter<double>("TID3x");
    localPositionResolution_TID3y = conf.getParameter<double>("TID3y");
    //
    // TOB
    localPositionResolution_TOB1x = conf.getParameter<double>("TOB1x");
    localPositionResolution_TOB1y = conf.getParameter<double>("TOB1y");
    localPositionResolution_TOB2x = conf.getParameter<double>("TOB2x");
    localPositionResolution_TOB2y = conf.getParameter<double>("TOB2y");
    localPositionResolution_TOB3x = conf.getParameter<double>("TOB3x");
    localPositionResolution_TOB3y = conf.getParameter<double>("TOB3y");
    localPositionResolution_TOB4x = conf.getParameter<double>("TOB4x");
    localPositionResolution_TOB4y = conf.getParameter<double>("TOB4y");
    localPositionResolution_TOB5x = conf.getParameter<double>("TOB5x");
    localPositionResolution_TOB5y = conf.getParameter<double>("TOB5y");
    localPositionResolution_TOB6x = conf.getParameter<double>("TOB6x");
    localPositionResolution_TOB6y = conf.getParameter<double>("TOB6y");
    //
    // TEC
    localPositionResolution_TEC1x = conf.getParameter<double>("TEC1x");
    localPositionResolution_TEC1y = conf.getParameter<double>("TEC1y");
    localPositionResolution_TEC2x = conf.getParameter<double>("TEC2x");
    localPositionResolution_TEC2y = conf.getParameter<double>("TEC2y");
    localPositionResolution_TEC3x = conf.getParameter<double>("TEC3x");
    localPositionResolution_TEC3y = conf.getParameter<double>("TEC3y");
    localPositionResolution_TEC4x = conf.getParameter<double>("TEC4x");
    localPositionResolution_TEC4y = conf.getParameter<double>("TEC4y");
    localPositionResolution_TEC5x = conf.getParameter<double>("TEC5x");
    localPositionResolution_TEC5y = conf.getParameter<double>("TEC5y");
    localPositionResolution_TEC6x = conf.getParameter<double>("TEC6x");
    localPositionResolution_TEC6y = conf.getParameter<double>("TEC6y");
    localPositionResolution_TEC7x = conf.getParameter<double>("TEC7x");
    localPositionResolution_TEC7y = conf.getParameter<double>("TEC7y");
    //
  }
  localPositionResolution_z = 0.0001; // not to be changed, set to minimum (1 um), Kalman Filter will crash if errors are exactly 0, setting 1 um means 0
  //
  //#ifdef FAMOS_DEBUG
//   std::cout << "RecHit local position error set to" << "\n"
// 	    << "\tTIB1\tx = " << localPositionResolution_TIB1x 
// 	    << " cm\ty = " << localPositionResolution_TIB1y << " cm" << "\n"
// 	    << "\tTIB2\tx = " << localPositionResolution_TIB2x 
// 	    << " cm\ty = " << localPositionResolution_TIB2y << " cm" << "\n"
// 	    << "\tTIB3\tx = " << localPositionResolution_TIB3x 
// 	    << " cm\ty = " << localPositionResolution_TIB3y << " cm" << "\n"
// 	    << "\tTIB4\tx = " << localPositionResolution_TIB4x 
// 	    << " cm\ty = " << localPositionResolution_TIB4y << " cm" << "\n"
// 	    << "\tTID1\tx = " << localPositionResolution_TID1x 
// 	    << " cm\ty = " << localPositionResolution_TID1y << " cm" << "\n"
// 	    << "\tTID2\tx = " << localPositionResolution_TID2x 
// 	    << " cm\ty = " << localPositionResolution_TID2y << " cm" << "\n"
// 	    << "\tTID3\tx = " << localPositionResolution_TID3x 
// 	    << " cm\ty = " << localPositionResolution_TID3y << " cm" << "\n"
// 	    << "\tTOB1\tx = " << localPositionResolution_TOB1x 
// 	    << " cm\ty = " << localPositionResolution_TOB1y << " cm" << "\n"
// 	    << "\tTOB2\tx = " << localPositionResolution_TOB2x 
// 	    << " cm\ty = " << localPositionResolution_TOB2y << " cm" << "\n"
// 	    << "\tTOB3\tx = " << localPositionResolution_TOB3x 
// 	    << " cm\ty = " << localPositionResolution_TOB3y << " cm" << "\n"
// 	    << "\tTOB4\tx = " << localPositionResolution_TOB4x 
// 	    << " cm\ty = " << localPositionResolution_TOB4y << " cm" << "\n"
// 	    << "\tTOB5\tx = " << localPositionResolution_TOB5x 
// 	    << " cm\ty = " << localPositionResolution_TOB5y << " cm" << "\n"
// 	    << "\tTOB6\tx = " << localPositionResolution_TOB6x 
// 	    << " cm\ty = " << localPositionResolution_TOB6y << " cm" << "\n"
// 	    << "\tTEC1\tx = " << localPositionResolution_TEC1x 
// 	    << " cm\ty = " << localPositionResolution_TEC1y << " cm" << "\n"
// 	    << "\tTEC2\tx = " << localPositionResolution_TEC2x 
// 	    << " cm\ty = " << localPositionResolution_TEC2y << " cm" << "\n"
// 	    << "\tTEC3\tx = " << localPositionResolution_TEC3x 
// 	    << " cm\ty = " << localPositionResolution_TEC3y << " cm" << "\n"
// 	    << "\tTEC4\tx = " << localPositionResolution_TEC4x 
// 	    << " cm\ty = " << localPositionResolution_TEC4y << " cm" << "\n"
// 	    << "\tTEC5\tx = " << localPositionResolution_TEC5x 
// 	    << " cm\ty = " << localPositionResolution_TEC5y << " cm" << "\n"
// 	    << "\tTEC6\tx = " << localPositionResolution_TEC6x 
// 	    << " cm\ty = " << localPositionResolution_TEC6y << " cm" << "\n"
// 	    << "\tTEC7\tx = " << localPositionResolution_TEC7x 
// 	    << " cm\ty = " << localPositionResolution_TEC7y << " cm" << "\n"
// 	    << "\tAll:\tz = " << localPositionResolution_z     << " cm" 
// 	    << std::endl;
// #endif

  //--- Number of histograms for alpha/beta barrel/forward multiplicity
  if (!trackingPSimHitsEqualSmearing && !trackingPSimHits) {
    if(useCMSSWPixelParameterization) {
      nAlphaBarrel  = conf.getParameter<int>("AlphaBarrelMultiplicityNew");
      nBetaBarrel   = conf.getParameter<int>("BetaBarrelMultiplicityNew");
      nAlphaForward = conf.getParameter<int>("AlphaForwardMultiplicityNew");
      nBetaForward  = conf.getParameter<int>("BetaForwardMultiplicityNew");
    } else {
      nAlphaBarrel  = conf.getParameter<int>("AlphaBarrelMultiplicity");
      nBetaBarrel   = conf.getParameter<int>("BetaBarrelMultiplicity");
      nAlphaForward = conf.getParameter<int>("AlphaForwardMultiplicity");
      nBetaForward  = conf.getParameter<int>("BetaForwardMultiplicity");
    }
#ifdef FAMOS_DEBUG
    std::cout << "Pixel maximum multiplicity set to " 
	      << "\nBarrel"  << "\talpha " << nAlphaBarrel  
	      << "\tbeta " << nBetaBarrel
	      << "\nForward" << "\talpha " << nAlphaForward 
	      << "\tbeta " << nBetaForward
	      << std::endl;
#endif
    
    // Resolution Barrel    
    if(useCMSSWPixelParameterization) {
      resAlphaBarrel_binMin   = conf.getParameter<double>("AlphaBarrel_BinMinNew"  );
      resAlphaBarrel_binWidth = conf.getParameter<double>("AlphaBarrel_BinWidthNew");
      resAlphaBarrel_binN     = conf.getParameter<int>(   "AlphaBarrel_BinNNew"    );
      resBetaBarrel_binMin    = conf.getParameter<double>("BetaBarrel_BinMinNew"   );
      resBetaBarrel_binWidth  = conf.getParameter<double>("BetaBarrel_BinWidthNew" );
      resBetaBarrel_binN      = conf.getParameter<int>(   "BetaBarrel_BinNNew"     );
    } else {
      resAlphaBarrel_binMin   = conf.getParameter<double>("AlphaBarrel_BinMin"  );
      resAlphaBarrel_binWidth = conf.getParameter<double>("AlphaBarrel_BinWidth");
      resAlphaBarrel_binN     = conf.getParameter<int>(   "AlphaBarrel_BinN"    );
      resBetaBarrel_binMin    = conf.getParameter<double>("BetaBarrel_BinMin"   );
      resBetaBarrel_binWidth  = conf.getParameter<double>("BetaBarrel_BinWidth" );
      resBetaBarrel_binN      = conf.getParameter<int>(   "BetaBarrel_BinN"     );
    }
    
    // Resolution Forward
    if(useCMSSWPixelParameterization) {
      resAlphaForward_binMin   = conf.getParameter<double>("AlphaForward_BinMinNew"   );
      resAlphaForward_binWidth = conf.getParameter<double>("AlphaForward_BinWidthNew" );
      resAlphaForward_binN     = conf.getParameter<int>(   "AlphaForward_BinNNew"     );
      resBetaForward_binMin    = conf.getParameter<double>("BetaForward_BinMinNew"    );
      resBetaForward_binWidth  = conf.getParameter<double>("BetaForward_BinWidthNew"  );
      resBetaForward_binN      = conf.getParameter<int>(   "BetaForward_BinNNew"      );
    } else {
      resAlphaForward_binMin   = conf.getParameter<double>("AlphaForward_BinMin"   );
      resAlphaForward_binWidth = conf.getParameter<double>("AlphaForward_BinWidth" );
      resAlphaForward_binN     = conf.getParameter<int>(   "AlphaForward_BinN"     );
      resBetaForward_binMin    = conf.getParameter<double>("BetaForward_BinMin"    );
      resBetaForward_binWidth  = conf.getParameter<double>("BetaForward_BinWidth"  );
      resBetaForward_binN      = conf.getParameter<int>(   "BetaForward_BinN"      );
    }
    
    // Hit Finding Probability
    theHitFindingProbability_PXB  = conf.getParameter<double>("HitFindingProbability_PXB" );
    theHitFindingProbability_PXF  = conf.getParameter<double>("HitFindingProbability_PXF" );
    theHitFindingProbability_TIB1 = conf.getParameter<double>("HitFindingProbability_TIB1");
    theHitFindingProbability_TIB2 = conf.getParameter<double>("HitFindingProbability_TIB2");
    theHitFindingProbability_TIB3 = conf.getParameter<double>("HitFindingProbability_TIB3");
    theHitFindingProbability_TIB4 = conf.getParameter<double>("HitFindingProbability_TIB4");
    theHitFindingProbability_TID1 = conf.getParameter<double>("HitFindingProbability_TID1");
    theHitFindingProbability_TID2 = conf.getParameter<double>("HitFindingProbability_TID2");
    theHitFindingProbability_TID3 = conf.getParameter<double>("HitFindingProbability_TID3");
    theHitFindingProbability_TOB1 = conf.getParameter<double>("HitFindingProbability_TOB1");
    theHitFindingProbability_TOB2 = conf.getParameter<double>("HitFindingProbability_TOB2");
    theHitFindingProbability_TOB3 = conf.getParameter<double>("HitFindingProbability_TOB3");
    theHitFindingProbability_TOB4 = conf.getParameter<double>("HitFindingProbability_TOB4");
    theHitFindingProbability_TOB5 = conf.getParameter<double>("HitFindingProbability_TOB5");
    theHitFindingProbability_TOB6 = conf.getParameter<double>("HitFindingProbability_TOB6");
    theHitFindingProbability_TEC1 = conf.getParameter<double>("HitFindingProbability_TEC1");
    theHitFindingProbability_TEC2 = conf.getParameter<double>("HitFindingProbability_TEC2");
    theHitFindingProbability_TEC3 = conf.getParameter<double>("HitFindingProbability_TEC3");
    theHitFindingProbability_TEC4 = conf.getParameter<double>("HitFindingProbability_TEC4");
    theHitFindingProbability_TEC5 = conf.getParameter<double>("HitFindingProbability_TEC5");
    theHitFindingProbability_TEC6 = conf.getParameter<double>("HitFindingProbability_TEC6");
    theHitFindingProbability_TEC7 = conf.getParameter<double>("HitFindingProbability_TEC7");
    //
#ifdef FAMOS_DEBUG
    std::cout << "RecHit finding probability set to" << "\n"
	      << "\tPXB  = " << theHitFindingProbability_PXB  << "\n"
	      << "\tPXF  = " << theHitFindingProbability_PXF  << "\n"
	      << "\tTIB1 = " << theHitFindingProbability_TIB1 << "\n"
	      << "\tTIB2 = " << theHitFindingProbability_TIB2 << "\n"
	      << "\tTIB3 = " << theHitFindingProbability_TIB3 << "\n"
	      << "\tTIB4 = " << theHitFindingProbability_TIB4 << "\n"
	      << "\tTID1 = " << theHitFindingProbability_TID1 << "\n"
	      << "\tTID2 = " << theHitFindingProbability_TID2 << "\n"
	      << "\tTID3 = " << theHitFindingProbability_TID3 << "\n"
	      << "\tTOB1 = " << theHitFindingProbability_TOB1 << "\n"
	      << "\tTOB2 = " << theHitFindingProbability_TOB2 << "\n"
	      << "\tTOB3 = " << theHitFindingProbability_TOB3 << "\n"
	      << "\tTOB4 = " << theHitFindingProbability_TOB4 << "\n"
	      << "\tTOB5 = " << theHitFindingProbability_TOB5 << "\n"
	      << "\tTOB6 = " << theHitFindingProbability_TOB6 << "\n"
	      << "\tTEC1 = " << theHitFindingProbability_TEC1 << "\n"
	      << "\tTEC2 = " << theHitFindingProbability_TEC2 << "\n"
	      << "\tTEC3 = " << theHitFindingProbability_TEC3 << "\n"
	      << "\tTEC4 = " << theHitFindingProbability_TEC4 << "\n"
	      << "\tTEC5 = " << theHitFindingProbability_TEC5 << "\n"
	      << "\tTEC6 = " << theHitFindingProbability_TEC6 << "\n"
	      << "\tTEC7 = " << theHitFindingProbability_TEC7 << "\n"
	      << std::endl;
#endif

    // Initialize the si strip error parametrization
  theSiStripErrorParametrization = 
    new SiStripGaussianSmearingRecHitConverterAlgorithm(random);
  
  // Initialization of pixel parameterization posponed to beginRun(), since it depends on the magnetic field
  }
}


void SiTrackerGaussianSmearingRecHitConverter::loadPixelData() {
  // load multiplicity cumulative probabilities
  // root files
  thePixelDataFile              = new TFile ( edm::FileInPath( thePixelMultiplicityFileName      ).fullPath().c_str() , "READ" );
  thePixelBarrelResolutionFile  = new TFile ( edm::FileInPath( thePixelBarrelResolutionFileName  ).fullPath().c_str() , "READ" );
  thePixelForwardResolutionFile = new TFile ( edm::FileInPath( thePixelForwardResolutionFileName ).fullPath().c_str() , "READ" );
  //

  // alpha barrel
  loadPixelData( thePixelDataFile, 
		 nAlphaBarrel  , 
		 std::string("hist_alpha_barrel")  , 
		 theBarrelMultiplicityAlphaCumulativeProbabilities  );
  // 
  // beta barrel
  loadPixelData( thePixelDataFile, 
		 nBetaBarrel   , 
		 std::string("hist_beta_barrel")   , 
		 theBarrelMultiplicityBetaCumulativeProbabilities   );
  // 
  // alpha forward
  loadPixelData( thePixelDataFile, 
		 nAlphaForward , 
		 std::string("hist_alpha_forward") , 
		 theForwardMultiplicityAlphaCumulativeProbabilities );
  // 
  // beta forward
  loadPixelData( thePixelDataFile, 
		 nBetaForward  , 
		 std::string("hist_beta_forward")  , 
		 theForwardMultiplicityBetaCumulativeProbabilities  );

  // Load also big pixel data if CMSSW parametrization is on
  // They are pushed back into the vectors after the normal pixels data:
  // [0, ..., (size/2)-1] -> Normal pixels
  // [size/2, ..., size-1] -> Big pixels
  if(useCMSSWPixelParameterization) {
    // alpha barrel
    loadPixelData( thePixelDataFile, 
                   nAlphaBarrel  , 
                   std::string("hist_alpha_barrel_big")  , 
                   theBarrelMultiplicityAlphaCumulativeProbabilities,
                   true );
    // 
    // beta barrel
    loadPixelData( thePixelDataFile, 
                   nBetaBarrel   , 
                   std::string("hist_beta_barrel_big")   , 
                   theBarrelMultiplicityBetaCumulativeProbabilities,
                   true );
    // 
    // alpha forward
    loadPixelData( thePixelDataFile, 
                   nAlphaForward , 
                   std::string("hist_alpha_forward_big") , 
                   theForwardMultiplicityAlphaCumulativeProbabilities, 
                   true );
    // 
    // beta forward
    loadPixelData( thePixelDataFile, 
                   nBetaForward  , 
                   std::string("hist_beta_forward_big")  , 
                   theForwardMultiplicityBetaCumulativeProbabilities, 
                   true );
  }
  // 
}

void SiTrackerGaussianSmearingRecHitConverter::loadPixelData( 
  TFile* pixelDataFile, 
  unsigned int nMultiplicity, 
  std::string histName,
  std::vector<TH1F*>& theMultiplicityCumulativeProbabilities,
  bool bigPixels) 
{

  std::string histName_i = histName + "_%u"; // needed to open histograms with a for
  if(!bigPixels)
    theMultiplicityCumulativeProbabilities.clear();
  //
  // What's this vector? Not needed - MG
//  std::vector<double> mult; // vector with fixed multiplicity
  for(unsigned int i = 0; i<nMultiplicity; ++i) {
    TH1F addHist = *((TH1F*) pixelDataFile->Get( Form( histName_i.c_str() ,i+1 )));
    if(i==0) {
      theMultiplicityCumulativeProbabilities.push_back( new TH1F(addHist) );
    } else {
      TH1F sumHist;
      if(bigPixels)
        sumHist = *(theMultiplicityCumulativeProbabilities[nMultiplicity+i-1]);
      else
        sumHist = *(theMultiplicityCumulativeProbabilities[i-1]);
      sumHist.Add(&addHist);
      theMultiplicityCumulativeProbabilities.push_back( new TH1F(sumHist) );
    }
  }

  // Logger
#ifdef FAMOS_DEBUG
  const unsigned int maxMult = theMultiplicityCumulativeProbabilities.size();
  unsigned int iMult, multSize;
  if(useCMSSWPixelParameterization) {
    if(bigPixels) {     
      iMult = maxMult / 2;
      multSize = maxMult ;
    } else {                
      iMult = 0;
      multSize = maxMult;
    }
  } else {
    iMult = 0;
    multSize = maxMult ;
  }
  std::cout << " Multiplicity cumulated probability " << histName << std::endl;
  for(/* void */; iMult<multSize; ++iMult) {
    for(int iBin = 1; iBin<=theMultiplicityCumulativeProbabilities[iMult]->GetNbinsX(); ++iBin) {
      std::cout
	<< " Multiplicity " << iMult+1 
	<< " bin " << iBin 
	<< " low edge = " 
	<< theMultiplicityCumulativeProbabilities[iMult]->GetBinLowEdge(iBin)
	<< " prob = " 
	<< (theMultiplicityCumulativeProbabilities[iMult])->GetBinContent(iBin) 
	// remember in ROOT bin starts from 1 (0 underflow, nBin+1 overflow)
	<< std::endl;
    }
  }
#endif

}

// Destructor
SiTrackerGaussianSmearingRecHitConverter::~SiTrackerGaussianSmearingRecHitConverter() {
  theBarrelMultiplicityAlphaCumulativeProbabilities.clear();
  theBarrelMultiplicityBetaCumulativeProbabilities.clear();
  theForwardMultiplicityAlphaCumulativeProbabilities.clear();
  theForwardMultiplicityBetaCumulativeProbabilities.clear();
  
  if(thePixelDataFile) delete thePixelDataFile;
  if(thePixelBarrelResolutionFile) delete thePixelBarrelResolutionFile;
  if(thePixelForwardResolutionFile) delete thePixelForwardResolutionFile;
  if(thePixelBarrelParametrization) delete thePixelBarrelParametrization;
  if(thePixelEndcapParametrization) delete thePixelEndcapParametrization;
  if(theSiStripErrorParametrization) delete theSiStripErrorParametrization;

  if (numberOfDisabledModules>0) delete disabledModules;

  if(random) delete random;

}  

void 
SiTrackerGaussianSmearingRecHitConverter::beginRun(edm::Run const &, const edm::EventSetup & es) 
{

  // Initialize the Tracker Geometry
  edm::ESHandle<TrackerGeometry> theGeometry;
  es.get<TrackerDigiGeometryRecord> ().get (theGeometry);
  geometry = &(*theGeometry);

  edm::ESHandle<TrackerGeometry> theMisAlignedGeometry;
  es.get<TrackerDigiGeometryRecord>().get("MisAligned",theMisAlignedGeometry);
  misAlignedGeometry = &(*theMisAlignedGeometry);

  const MagneticField* magfield;
  edm::ESHandle<MagneticField> magField;
  es.get<IdealMagneticFieldRecord>().get(magField);
  magfield=&(*magField);
  GlobalPoint center(0.0, 0.0, 0.0);
  double magFieldAtCenter = magfield->inTesla(center).mag();

  // For new parameterization: select multiplicity and resolution files according to magnetic field
  if (!trackingPSimHitsEqualSmearing && !trackingPSimHits) {
    if(useCMSSWPixelParameterization) {
      if(magFieldAtCenter > 3.9) {
	thePixelMultiplicityFileName = pset_.getParameter<std::string>( "PixelMultiplicityFile40T");
	thePixelBarrelResolutionFileName = pset_.getParameter<std::string>( "PixelBarrelResolutionFile40T");
	thePixelForwardResolutionFileName = pset_.getParameter<std::string>( "PixelForwardResolutionFile40T");
      } else {
	thePixelMultiplicityFileName = pset_.getParameter<std::string>( "PixelMultiplicityFile38T");
	thePixelBarrelResolutionFileName = pset_.getParameter<std::string>( "PixelBarrelResolutionFile38T");      
	thePixelForwardResolutionFileName = pset_.getParameter<std::string>( "PixelForwardResolutionFile38T");
      }
    } else {
      thePixelMultiplicityFileName = pset_.getParameter<std::string>( "PixelMultiplicityFile" );
      thePixelBarrelResolutionFileName = pset_.getParameter<std::string>( "PixelBarrelResolutionFile");
      thePixelForwardResolutionFileName = pset_.getParameter<std::string>( "PixelForwardResolutionFile");
    }
  }

  // Reading the list of dead pixel modules from DB:
  edm::ESHandle<SiPixelQuality> siPixelBadModule;
  es.get<SiPixelQualityRcd>().get(siPixelBadModule);
  numberOfDisabledModules = 0;
  if (doDisableChannels) {
    disabledModules = new std::vector<SiPixelQuality::disabledModuleType> ( siPixelBadModule->getBadComponentList() );
    numberOfDisabledModules = disabledModules->size();
    size_t numberOfRecoverableModules = 0;
    for (size_t id=0;id<numberOfDisabledModules;id++) {
      //////////////////////////////////////
      //  errortype "whole" = int 0 in DB //
      //  errortype "tbmA" = int 1 in DB  //
      //  errortype "tbmB" = int 2 in DB  //
      //  errortype "none" = int 3 in DB  //
      //////////////////////////////////////
      if ( (*disabledModules)[id-numberOfRecoverableModules].errorType != 0 ){
	// Disable only the modules  totally in error:
	disabledModules->erase(disabledModules->begin()+id-numberOfRecoverableModules);
	numberOfRecoverableModules++;
      }
    }
    numberOfDisabledModules = disabledModules->size();
  }
  


// #ifdef FAMOS_DEBUG
//   std::cout << "Pixel multiplicity data are taken from file " << thePixelMultiplicityFileName << std::endl;

//   std::cout << "Pixel maximum multiplicity set to " 
// 	    << "\nBarrel"  << "\talpha " << nAlphaBarrel  
// 	    << "\tbeta " << nBetaBarrel
// 	    << "\nForward" << "\talpha " << nAlphaForward 
// 	    << "\tbeta " << nBetaForward
// 	    << std::endl;

//   std::cout << "Barrel Pixel resolution data are taken from file " 
// 	    << thePixelBarrelResolutionFileName << "\n"
// 	    << "Alpha bin min = " << resAlphaBarrel_binMin
// 	    << "\twidth = "       << resAlphaBarrel_binWidth
// 	    << "\tbins = "        << resAlphaBarrel_binN
// 	    << "\n"
// 	    << " Beta bin min = " << resBetaBarrel_binMin
// 	    << "\twidth = "       << resBetaBarrel_binWidth
// 	    << "\tbins = "        << resBetaBarrel_binN
// 	    << std::endl;

//   std::cout << "Forward Pixel resolution data are taken from file " 
// 	    << thePixelForwardResolutionFileName << "\n"
// 	    << "Alpha bin min = " << resAlphaForward_binMin
// 	    << "\twidth = "       << resAlphaForward_binWidth
// 	    << "\tbins = "        << resAlphaForward_binN
// 	    << "\n"
// 	    << " Beta bin min = " << resBetaForward_binMin
// 	    << "\twidth = "       << resBetaForward_binWidth
// 	    << "\tbins = "        << resBetaForward_binN
// 	    << std::endl;
// #endif 
  //

  //    
  // load pixel data
  if (!trackingPSimHitsEqualSmearing && !trackingPSimHits ) {
    loadPixelData();
    //
    
    // Initialize and open relevant files for the pixel barrel error parametrization
    thePixelBarrelParametrization = 
      new SiPixelGaussianSmearingRecHitConverterAlgorithm(
							  pset_,
							  GeomDetEnumerators::PixelBarrel,
							  random);
    // Initialize and open relevant files for the pixel forward error parametrization 
    thePixelEndcapParametrization = 
      new SiPixelGaussianSmearingRecHitConverterAlgorithm(
							  pset_,
							  GeomDetEnumerators::PixelEndcap,
							  random);
  }

}

void SiTrackerGaussianSmearingRecHitConverter::produce(edm::Event& e, const edm::EventSetup& es) 
{
  
  //Retrieve tracker topology from geometry
  edm::ESHandle<TrackerTopology> tTopoHandle;
  es.get<IdealGeometryRecord>().get(tTopoHandle);
  const TrackerTopology* const tTopo = tTopoHandle.product();
   
  // Step 0: Declare Ref and RefProd
  FastTrackerClusterRefProd = e.getRefBeforePut<FastTrackerClusterCollection>("TrackerClusters");
  
  // Step A: Get Inputs (PSimHit's)
  edm::Handle<CrossingFrame<PSimHit> > cf_simhit; 
  std::vector<const CrossingFrame<PSimHit> *> cf_simhitvec;
  for(uint32_t i=0; i<trackerContainers.size(); i++){
    e.getByLabel(trackerContainers[i], cf_simhit);
    cf_simhitvec.push_back(cf_simhit.product());
  }
  std::auto_ptr<MixCollection<PSimHit> > allTrackerHits(new MixCollection<PSimHit>(cf_simhitvec));

  // Step B: create temporary RecHit collection and fill it with Gaussian smeared RecHit's
  
  //NEW!!!CREATE CLUSTERS AT THE SAME TIME
  std::map<unsigned, edm::OwnVector<SiTrackerGSRecHit2D> > temporaryRecHits;
  std::map<unsigned, edm::OwnVector<FastTrackerCluster> > theClusters ;
 

  smearHits( *allTrackerHits, temporaryRecHits, theClusters,tTopo);
  
 // Step C: match rechits on stereo layers
  std::map<unsigned, edm::OwnVector<SiTrackerGSMatchedRecHit2D> > temporaryMatchedRecHits ;
  if(doMatching)  matchHits(  temporaryRecHits,  temporaryMatchedRecHits, *allTrackerHits);

  // Step D: from the temporary RecHit collection, create the real one.
  std::auto_ptr<SiTrackerGSRecHit2DCollection> recHitCollection(new SiTrackerGSRecHit2DCollection);
  std::auto_ptr<SiTrackerGSMatchedRecHit2DCollection> recHitCollectionMatched(new SiTrackerGSMatchedRecHit2DCollection);
  if(doMatching){
    loadMatchedRecHits(temporaryMatchedRecHits, *recHitCollectionMatched);
    loadRecHits(temporaryRecHits, *recHitCollection);
  }
  else {
    //might need to have a "matched" hit collection containing the simple hits
    loadRecHits(temporaryRecHits, *recHitCollection);
  }
  
#ifdef FAMOS_DEBUG 

  std::cout << "\n****** TrackerGSRecHits hits are =\t" <<  (*recHitCollection).size()<<std::endl;
  std::cout << "\n****** TrackerGSRecHitsMatched hits are =\t" <<  (*recHitCollectionMatched).size()<< std::endl;
  
#endif 

  // Step E: write output to file
  e.put(recHitCollection,"TrackerGSRecHits");
  e.put(recHitCollectionMatched,"TrackerGSMatchedRecHits");
  
  //STEP F: write clusters
  std::auto_ptr<FastTrackerClusterCollection> clusterCollection(new FastTrackerClusterCollection);
  loadClusters(theClusters, *clusterCollection);
  e.put(clusterCollection,"TrackerClusters");
}



void SiTrackerGaussianSmearingRecHitConverter::smearHits(MixCollection<PSimHit>& input, 
							 std::map<unsigned, edm::OwnVector<SiTrackerGSRecHit2D> >& temporaryRecHits,
							 std::map<unsigned, edm::OwnVector<FastTrackerCluster> >& theClusters, const TrackerTopology *tTopo)
{
  
  int numberOfPSimHits = 0;
  
  //  edm::PSimHitContainer::const_iterator isim;
  MixCollection<PSimHit>::iterator isim = input.begin();
  MixCollection<PSimHit>::iterator lastSimHit = input.end();
  correspondingSimHit.resize(input.size());
  Local3DPoint position;
  LocalError error;
  LocalError inflatedError;
  
  int simHitCounter = -1;
  int recHitCounter = 0;
  
  // loop on PSimHits

  //std::cout << " Begin Number of Sim Hits " <<  input.size() << std::endl;

  for ( ; isim != lastSimHit; ++isim ) {
    ++simHitCounter;
    
    DetId det((*isim).detUnitId());
    unsigned trackID = (*isim).trackId();
    uint32_t eeID = (*isim).eventId().rawId(); //get the rawId of the eeid for pileup treatment


    //// Here comes the Dead Modules rejection, by Suzan 

#ifdef FAMOS_DEBUG

     unsigned int subdetId = det.subdetId();
     //unsigned int  disk  = tTopo->pxfDisk(det);
     //unsigned int  side  = tTopo->pxfSide(det);


     std::cout << " SubDetector = " << subdetId << std::endl;
     //std::cout<< " Pixel Forward Disk Number : "<< disk << " Side : "<<side<<std::endl; 
     if(subdetId==1 || subdetId==2) std::cout<<" Pixel GSRecHits "<<std::endl;
     else if(subdetId==3|| subdetId==4 || subdetId==5 || subdetId == 6) std::cout<<" Strip GSRecHits "<<std::endl;    

     if(subdetId==1){
      
      unsigned int theLayer = tTopo->pxbLayer(det);
      std::cout << "\tPixel Barrel Layer " << theLayer << std::endl;
     }

     if(subdetId==2){
      
      unsigned int theDisk = tTopo->pxfDisk(det);
      std::cout << "\tPixel Forward Disk " << theDisk << std::endl;
     }
#endif


    bool isBad = false;
    unsigned int geoId  = det.rawId();
    for (size_t id=0;id<numberOfDisabledModules;id++) {
      if(geoId==(*disabledModules)[id].DetID){
	//  Already selected in the beginRun() the ones with errorType = 0
	//	if((*disabledModules)[id].errorType == 0) isBad = true;
	isBad = true;
	break;
      }
    }    
    if(isBad)      continue;

#ifdef FAMOS_DEBUG
    const GeomDet* theDet = geometry->idToDet(det);
    std::cout << "SimHit Track/z/r as simulated : "
	      << trackID << " " 
	      << theDet->surface().toGlobal((*isim).localPosition()).z() << " " 
	      << theDet->surface().toGlobal((*isim).localPosition()).perp() << std::endl;
    const GeomDet* theMADet = misAlignedGeometry->idToDet(det);
    std::cout << "SimHit Track/z/r as reconstructed : "
	      << trackID << " " 
	      << theMADet->surface().toGlobal((*isim).localPosition()).z() << " " 
	      << theMADet->surface().toGlobal((*isim).localPosition()).perp() << std::endl;
#endif


    ++numberOfPSimHits;	
    // gaussian smearing
    unsigned int alphaMult = 0;
    unsigned int betaMult  = 0;
    bool isCreated = gaussianSmearing(*isim, position, error, alphaMult, betaMult,tTopo);

#ifdef FAMOS_DEBUG
    std::cout << "Error as simulated     " << error.xx() << " " << error.xy() << " " << error.yy() << std::endl;
#endif
    //
    unsigned int subdet = det.subdetId();
    
    //
    if(isCreated) {


      //      std::cout << "Inside isCreated in subdetector" << subdet << std::endl; 

      //      double dist = theDet->surface().toGlobal((*isim).localPosition()).mag2();
      // create RecHit
      // Fill the temporary RecHit on the current DetId collection
      //      temporaryRecHits[trackID].push_back(
      
      // Fix by P.Azzi, A.Schmidt:
      // if this hit is on a glued detector with radial topology, 
      // the x-coordinate needs to be translated correctly to y=0.
      // this is necessary as intermediate step for the matching of hits

      // do it only if we do rec hit matching
//       if(doMatching){
// 	StripSubdetector specDetId=StripSubdetector(det);
	
// 	if(specDetId.glued()) if(subdet==6 || subdet==4){         // check if on double layer in TEC or TID
	  
// 	  const GeomDetUnit* theDetUnit = geometry->idToDetUnit(det);
// 	  const StripTopology& topol=(const StripTopology&)theDetUnit->type().topology();      
	  
// 	  // check if radial topology
// 	  if(dynamic_cast <const RadialStripTopology*>(&topol)){
	    
// 	    const RadialStripTopology *rtopol = dynamic_cast <const RadialStripTopology*>(&topol);
	    
// 	    // translate to measurement coordinates
// 	    MeasurementPoint mpoint=rtopol->measurementPosition(position);
// 	    // set y=0
// 	    MeasurementPoint mpoint0=MeasurementPoint(mpoint.x(),0.);
// 	    // translate back to local coordinates with y=0
// 	    LocalPoint lpoint0 = rtopol->localPosition(mpoint0);
// 	    position = Local3DPoint(lpoint0.x(),0.,0.);
	    
// 	  }
// 	} // end if( specDetId.glued()...
//       } // end if(doMatching)
      
      // set y=0 in single-sided strip detectors 
      if(doMatching && (subdet>2)){    

	StripSubdetector specDetId=StripSubdetector(det);
	if( !specDetId.glued() ){  // this is a single-sided module
	  position = Local3DPoint(position.x(),0.,0.); // set y to 0 on single-sided modules
	}
      }
      else{  if(subdet>2) position = Local3DPoint(position.x(),0.,0.);    }  // no matching, set y=0 on strips

      // Inflate errors in case of geometry misalignment
      const GeomDet* theMADet = misAlignedGeometry->idToDet(det);
      if ( theMADet->alignmentPositionError() != 0 ) { 
 	LocalError lape = 
 	  ErrorFrameTransformer().transform ( theMADet->alignmentPositionError()->globalError(),
 					      theMADet->surface() );
 	error = LocalError ( error.xx()+lape.xx(),
 			     error.xy()+lape.xy(),
 			     error.yy()+lape.yy() );
      }
      
      float chargeADC = (*isim).energyLoss()/(GevPerElectron * ElectronsPerADC);
      
      //create cluster
      if(subdet > 2) theClusters[trackID].push_back(
						    new FastTrackerCluster(LocalPoint(position.x(), 0.0, 0.0), error, det,
									   simHitCounter, trackID,
									   eeID,
									   //(*isim).energyLoss())
									   chargeADC)
						    );
      else theClusters[trackID].push_back(
					  new FastTrackerCluster(position, error, det,
								 simHitCounter, trackID,
								 eeID,
								 //(*isim).energyLoss())
								 chargeADC)
					  );
    

      //       std::cout << "CLUSTER for simhit " << simHitCounter << "\t energy loss = " <<chargeADC << std::endl;
      
      // std::cout << "Error as reconstructed " << error.xx() << " " << error.xy() << " " << error.yy() << std::endl;
      
      // create rechit
      temporaryRecHits[trackID].push_back(
					  new SiTrackerGSRecHit2D(position, error, det, 
								  simHitCounter, trackID, 
								  eeID, 
								  ClusterRef(FastTrackerClusterRefProd, simHitCounter),
								  alphaMult, betaMult)
					  );
      
       // This a correpondence map between RecHits and SimHits 
      // (for later  use in matchHits)
      correspondingSimHit[recHitCounter++] = isim; 
     
      
    } // end if(isCreated)

  } // end loop on PSimHits

}

bool SiTrackerGaussianSmearingRecHitConverter::gaussianSmearing(const PSimHit& simHit, 
								Local3DPoint& position , 
								LocalError& error,
								unsigned& alphaMult, 
								unsigned& betaMult, const TrackerTopology *tTopo) 
{

  // A few caracteritics of the module the SimHit belongs to.
  unsigned int subdet   = DetId(simHit.detUnitId()).subdetId();

  unsigned int detid    = DetId(simHit.detUnitId()).rawId();

  const GeomDetUnit* theDetUnit = geometry->idToDetUnit((DetId)simHit.detUnitId());

  const BoundPlane& theDetPlane = theDetUnit->surface();

  const Bounds& theBounds = theDetPlane.bounds();
  double boundX = theBounds.width()/2.;
  double boundY = theBounds.length()/2.;


#ifdef FAMOS_DEBUG
  std::cout << "Inside gaussianSmearing \tSubdetector " << subdet 
	    << " rawid " << detid
	    << std::endl;
#endif
  if(trackingPSimHits) {

    // z is fixed for all detectors, in case of errors resolution is fixed also for x and y to 1 um (zero)
    // The Matrix is the Covariance Matrix, sigma^2 on diagonal!!!
    error = LocalError( localPositionResolution_z * localPositionResolution_z , 
			0.0 , 
			localPositionResolution_z * localPositionResolution_z  );
    //
    // starting from PSimHit local position
    position = simHit.localPosition();
#ifdef FAMOS_DEBUG
    std::cout << " Tracking PSimHit position set to  " << position;
#endif
    return true; // RecHit == PSimHit with 100% hit finding efficiency
  } else if (trackingPSimHitsEqualSmearing) {
    
    //split the errors per layer
    if(subdet == 1){ //Pixel Barrel (now includes tracker) 
      
      unsigned int theLayer = tTopo->pxbLayer(detid);
      position = Local3DPoint(random->gaussShoot((double)simHit.localPosition().x(), localPositionResolutionBar_x[theLayer]),(double)simHit.localPosition().y(),localPositionResolutionBar_y[theLayer]);
      //position = Local3DPoint(random->gaussShoot((double)simHit.localPosition().x(), localPositionResolutionBar_x[theLayer]),(double)simHit.localPosition().y(),0.);
      error = LocalError( localPositionResolutionBar_x[theLayer] * localPositionResolutionBar_x[theLayer],
			  0.0,
			  localPositionResolutionBar_y[theLayer] * localPositionResolutionBar_y[theLayer] ); 
    } else if(subdet == 2) {
      
      unsigned int theDisk = tTopo->pxfDisk(detid);     
      position = Local3DPoint(random->gaussShoot((double)simHit.localPosition().x(), localPositionResolutionFwd_x[theDisk]), (double)simHit.localPosition().y(),localPositionResolutionFwd_y[theDisk]);
      // position = Local3DPoint(random->gaussShoot((double)simHit.localPosition().x(), localPositionResolutionFwd_x[theDisk]),(double)simHit.localPosition().y(),0.);
      error = LocalError( localPositionResolutionFwd_x[theDisk] * localPositionResolutionFwd_x[theDisk],
			  0.0,
			  localPositionResolutionFwd_y[theDisk] * localPositionResolutionFwd_y[theDisk] ); 
    } else if (subdet>2) {
      //we are in the Phase 1 tracker same smearing for all channels. Will be improved to use the actual tracker resolutions
      
      position = Local3DPoint(random->gaussShoot((double)simHit.localPosition().x(), localPositionResolution_x), 
			      (double)simHit.localPosition().y(),0.);
      error = LocalError( localPositionResolution_x * localPositionResolution_x,
			  0.0, localPositionResolution_y * localPositionResolution_y ); 
      //      std::cout << "in the Phase 1 tracker subdet" << subdet << std::endl;
    }

    /*
      position = Local3DPoint(random->gaussShoot((double)simHit.localPosition().x(), localPositionResolution_x), 
			      (double)simHit.localPosition().y(),
			      0.);
      error = LocalError( localPositionResolution_x * localPositionResolution_x,
			  0.0,
			  localPositionResolution_y * localPositionResolution_y ); 
      std::cout << "in the Phase 1 old style tracker subdet" << subdet << std::endl;
    */

#ifdef FAMOS_DEBUG
    std::cout << " Tracking PSimHit position set to  " << position;
    std::cout << " Tracking PSimHit error set to  " << error;
#endif
    return true; // RecHit == PSimHit 100% hit finding efficiency
  }
  //
  
  
  //  std::cout << "************************ HERE???" << std::endl;
  
  
  // hit finding probability --> RecHit will be created if and only if hitFindingProbability <= theHitFindingProbability_###
  double hitFindingProbability = random->flatShoot();
#ifdef FAMOS_DEBUG
  std::cout << " Hit finding probability draw: " << hitFindingProbability << std::endl;;
#endif

  switch (subdet) {
    // Pixel Barrel
  case 1:
    {
#ifdef FAMOS_DEBUG
      
      unsigned int theLayer = tTopo->pxfLayer(detid);
      std::cout << "\tPixel Barrel Layer " << theLayer << std::endl;
#endif
      if( hitFindingProbability > theHitFindingProbability_PXB ) return false;
      // Hit smearing
      const PixelGeomDetUnit* pixelDetUnit = dynamic_cast<const PixelGeomDetUnit*>(theDetUnit);
      thePixelBarrelParametrization->smearHit(simHit, pixelDetUnit, boundX, boundY);
      position  = thePixelBarrelParametrization->getPosition();
      error     = thePixelBarrelParametrization->getError();
      alphaMult = thePixelBarrelParametrization->getPixelMultiplicityAlpha();
      betaMult  = thePixelBarrelParametrization->getPixelMultiplicityBeta();
      return true;
      break;
    }
    // Pixel Forward
  case 2:
    {
#ifdef FAMOS_DEBUG
      
      unsigned int theDisk = tTopo->pxfDisk(detid);
      std::cout << "\tPixel Forward Disk " << theDisk << std::endl;
#endif
      if( hitFindingProbability > theHitFindingProbability_PXF ) return false;
      // Hit smearing
      const PixelGeomDetUnit* pixelDetUnit = dynamic_cast<const PixelGeomDetUnit*>(theDetUnit);
      thePixelEndcapParametrization->smearHit(simHit, pixelDetUnit, boundX, boundY);
      position = thePixelEndcapParametrization->getPosition();
      error    = thePixelEndcapParametrization->getError();
      alphaMult = thePixelEndcapParametrization->getPixelMultiplicityAlpha();
      betaMult  = thePixelEndcapParametrization->getPixelMultiplicityBeta();
      return true;
      break;
    }
    // TIB
  case 3:
    {
      
      unsigned int theLayer  = tTopo->tibLayer(detid);
#ifdef FAMOS_DEBUG
      std::cout << "\tTIB Layer " << theLayer << std::endl;
#endif
      //
      double resolutionX, resolutionY, resolutionZ;
      resolutionZ = localPositionResolution_z;
      
      switch (theLayer) {
      case 1:
	{
	  resolutionX = localPositionResolution_TIB1x;
	  resolutionY = localPositionResolution_TIB1y;
	  if( hitFindingProbability > theHitFindingProbability_TIB1 ) return false;
	  break;
	}
      case 2:
	{
	  resolutionX = localPositionResolution_TIB2x;
	  resolutionY = localPositionResolution_TIB2y;
	  if( hitFindingProbability > theHitFindingProbability_TIB2 ) return false;
	  break;
	}
      case 3:
	{
	  resolutionX = localPositionResolution_TIB3x;
	  resolutionY = localPositionResolution_TIB3y;
	  if( hitFindingProbability > theHitFindingProbability_TIB3 ) return false;
	  break;
	}
      case 4:
	{
	  resolutionX = localPositionResolution_TIB4x;
	  resolutionY = localPositionResolution_TIB4y;
	  if( hitFindingProbability > theHitFindingProbability_TIB4 ) return false;
	  break;
	}
      default:
	{
	  edm::LogError ("SiTrackerGaussianSmearingRecHits") 
	    << "\tTIB Layer not valid " << theLayer << std::endl;
	  return false;
	  break;
	}
      }

      // Gaussian smearing
      theSiStripErrorParametrization->smearHit(simHit, resolutionX, resolutionY, resolutionZ, boundX, boundY);
      position = theSiStripErrorParametrization->getPosition();
      error    = theSiStripErrorParametrization->getError();
      alphaMult = 0;
      betaMult  = 0;
      return true;
      break;
    } // TIB
    
    // TID
  case 4:
    {
      
      unsigned int theRing  = tTopo->tidRing(detid);
      double resolutionFactorY = 
	1. - simHit.localPosition().y() / theDetPlane.position().perp(); 

#ifdef FAMOS_DEBUG
      std::cout << "\tTID Ring " << theRing << std::endl;
#endif
      double resolutionX, resolutionY, resolutionZ;
      resolutionZ = localPositionResolution_z;
      
      switch (theRing) {
      case 1:
	{
	  resolutionX = localPositionResolution_TID1x * resolutionFactorY;
	  resolutionY = localPositionResolution_TID1y;
	  if( hitFindingProbability > theHitFindingProbability_TID1 ) return false;
	  break;
	}
      case 2:
	{
	  resolutionX = localPositionResolution_TID2x * resolutionFactorY;
	  resolutionY = localPositionResolution_TID2y;
	  if( hitFindingProbability > theHitFindingProbability_TID2 ) return false;
	  break;
	}
      case 3:
	{
	  resolutionX = localPositionResolution_TID3x * resolutionFactorY;
	  resolutionY = localPositionResolution_TID3y;
	  if( hitFindingProbability > theHitFindingProbability_TID3 ) return false;
	  break;
	}
      default:
	{
	  edm::LogError ("SiTrackerGaussianSmearingRecHits") 
	    << "\tTID Ring not valid " << theRing << std::endl;
	  return false;
	  break;
	}
      }

      boundX *=  resolutionFactorY;

      theSiStripErrorParametrization->smearHit(simHit, resolutionX, resolutionY, resolutionZ, boundX, boundY);
      position = theSiStripErrorParametrization->getPosition();
      error    = theSiStripErrorParametrization->getError();
      alphaMult = 0;
      betaMult  = 0;
      return true;
      break;
    } // TID
    
    // TOB
  case 5:
    {
      
      unsigned int theLayer  = tTopo->tibLayer(detid);
#ifdef FAMOS_DEBUG
      std::cout << "\tTOB Layer " << theLayer << std::endl;
#endif
      double resolutionX, resolutionY, resolutionZ;
      resolutionZ = localPositionResolution_z;
      
      switch (theLayer) {
      case 1:
	{
	  resolutionX = localPositionResolution_TOB1x;
	  resolutionY = localPositionResolution_TOB1y;
	  if( hitFindingProbability > theHitFindingProbability_TOB1 ) return false;
	  break;
	}
      case 2:
	{
	  resolutionX = localPositionResolution_TOB2x;
	  resolutionY = localPositionResolution_TOB2y;
	  if( hitFindingProbability > theHitFindingProbability_TOB2 ) return false;
	  break;
	}
      case 3:
	{
	  resolutionX = localPositionResolution_TOB3x;
	  resolutionY = localPositionResolution_TOB3y;
	  if( hitFindingProbability > theHitFindingProbability_TOB3 ) return false;
	  break;
	}
      case 4:
	{
	  resolutionX = localPositionResolution_TOB4x;
	  resolutionY = localPositionResolution_TOB4y;
	  if( hitFindingProbability > theHitFindingProbability_TOB4 ) return false;
	  break;
	}
      case 5:
	{
	  resolutionX = localPositionResolution_TOB5x;
	  resolutionY = localPositionResolution_TOB5y;
	  if( hitFindingProbability > theHitFindingProbability_TOB5 ) return false;
	  break;
	}
      case 6:
	{
	  resolutionX = localPositionResolution_TOB6x;
	  resolutionY = localPositionResolution_TOB6y;
	  if( hitFindingProbability > theHitFindingProbability_TOB6 ) return false;
	  break;
	}
      default:
	{
	  edm::LogError ("SiTrackerGaussianSmearingRecHits") 
	    << "\tTOB Layer not valid " << theLayer << std::endl;
	  return false;
	  break;
	}
      }
      theSiStripErrorParametrization->smearHit(simHit, resolutionX, resolutionY, resolutionZ, boundX, boundY);
      position = theSiStripErrorParametrization->getPosition();
      error    = theSiStripErrorParametrization->getError();
      alphaMult = 0;
      betaMult  = 0;
      return true;
      break;
    } // TOB
    
    // TEC
  case 6:
    {
      
      unsigned int theRing  = tTopo->tecRing(detid);
      double resolutionFactorY = 
	1. - simHit.localPosition().y() / theDetPlane.position().perp(); 

#ifdef FAMOS_DEBUG
      std::cout << "\tTEC Ring " << theRing << std::endl;
#endif
      double resolutionX, resolutionY, resolutionZ;
      resolutionZ = localPositionResolution_z * localPositionResolution_z;
      
      switch (theRing) {
      case 1:
	{
	  resolutionX = localPositionResolution_TEC1x * resolutionFactorY;
	  resolutionY = localPositionResolution_TEC1y;
	  if( hitFindingProbability > theHitFindingProbability_TEC1 ) return false;
	  break;
	}
      case 2:
	{
	  resolutionX = localPositionResolution_TEC2x * resolutionFactorY;
	  resolutionY = localPositionResolution_TEC2y;
	  if( hitFindingProbability > theHitFindingProbability_TEC2 ) return false;
	  break;
	}
      case 3:
	{
	  resolutionX = localPositionResolution_TEC3x * resolutionFactorY;
	  resolutionY = localPositionResolution_TEC3y;
	  if( hitFindingProbability > theHitFindingProbability_TEC3 ) return false;
	  break;
	}
      case 4:
	{
	  resolutionX = localPositionResolution_TEC4x * resolutionFactorY;
	  resolutionY = localPositionResolution_TEC4y;
	  if( hitFindingProbability > theHitFindingProbability_TEC4 ) return false;
	  break;
	}
      case 5:
	{
	  resolutionX = localPositionResolution_TEC5x * resolutionFactorY;
	  resolutionY = localPositionResolution_TEC5y;
	  if( hitFindingProbability > theHitFindingProbability_TEC5 ) return false;
	  break;
	}
      case 6:
	{
	  resolutionX = localPositionResolution_TEC6x * resolutionFactorY;
	  resolutionY = localPositionResolution_TEC6y;
	  if( hitFindingProbability > theHitFindingProbability_TEC6 ) return false;
	  break;
	}
      case 7:
	{
	  resolutionX = localPositionResolution_TEC7x * resolutionFactorY;
	  resolutionY = localPositionResolution_TEC7y;
	  if( hitFindingProbability > theHitFindingProbability_TEC7 ) return false;
	  break;
	}
      default:
	{
	  edm::LogError ("SiTrackerGaussianSmearingRecHits") 
	    << "\tTEC Ring not valid " << theRing << std::endl;
	  return false;
	  break;
	}
      }

      boundX *= resolutionFactorY;
      theSiStripErrorParametrization->smearHit(simHit, resolutionX, resolutionY, resolutionZ, boundX, boundY);
      position = theSiStripErrorParametrization->getPosition();
      error    = theSiStripErrorParametrization->getError();
      alphaMult = 0;
      betaMult  = 0;
      return true;
      break;
    } // TEC
    
  default:
    {
      edm::LogError ("SiTrackerGaussianSmearingRecHits") << "\tTracker subdetector not valid " << subdet << std::endl;
      return false;
      break;
    }
    
  } // subdetector case
    //
}   



void 
    SiTrackerGaussianSmearingRecHitConverter::loadClusters(
    std::map<unsigned,edm::OwnVector<FastTrackerCluster> >& theClusterMap, 
    FastTrackerClusterCollection& theClusterCollection) const
{
  std::map<unsigned,edm::OwnVector<FastTrackerCluster> >::const_iterator 
      it = theClusterMap.begin();
  std::map<unsigned,edm::OwnVector<FastTrackerCluster> >::const_iterator 
      lastCluster = theClusterMap.end();
  
  for( ; it != lastCluster ; ++it ) { 
    theClusterCollection.put(it->first,(it->second).begin(),(it->second).end());
  }
}



void 
SiTrackerGaussianSmearingRecHitConverter::loadRecHits(
     std::map<unsigned,edm::OwnVector<SiTrackerGSRecHit2D> >& theRecHits, 
     SiTrackerGSRecHit2DCollection& theRecHitCollection) const
{
  std::map<unsigned,edm::OwnVector<SiTrackerGSRecHit2D> >::const_iterator 
    it = theRecHits.begin();
  std::map<unsigned,edm::OwnVector<SiTrackerGSRecHit2D> >::const_iterator 
    lastRecHit = theRecHits.end();

  for( ; it != lastRecHit ; ++it ) { 
    theRecHitCollection.put(it->first,(it->second).begin(),(it->second).end());
  }

}

void 
SiTrackerGaussianSmearingRecHitConverter::loadMatchedRecHits(
     std::map<unsigned,edm::OwnVector<SiTrackerGSMatchedRecHit2D> >& theRecHits, 
     SiTrackerGSMatchedRecHit2DCollection& theRecHitCollection) const
{
  std::map<unsigned,edm::OwnVector<SiTrackerGSMatchedRecHit2D> >::const_iterator 
    it = theRecHits.begin();
  std::map<unsigned,edm::OwnVector<SiTrackerGSMatchedRecHit2D> >::const_iterator 
    lastRecHit = theRecHits.end();

  for( ; it != lastRecHit ; ++it ) { 
    theRecHitCollection.put(it->first,(it->second).begin(),(it->second).end());
  }

}


void
SiTrackerGaussianSmearingRecHitConverter::matchHits(
  std::map<unsigned, edm::OwnVector<SiTrackerGSRecHit2D> >& theRecHits, 
  std::map<unsigned, edm::OwnVector<SiTrackerGSMatchedRecHit2D> >& matchedMap, 
  MixCollection<PSimHit>& simhits ) {

  std::map<unsigned, edm::OwnVector<SiTrackerGSRecHit2D> >::iterator it = theRecHits.begin();
  std::map<unsigned, edm::OwnVector<SiTrackerGSRecHit2D> >::iterator lastTrack = theRecHits.end();


  int recHitCounter = 0;

  //loop over single sided tracker RecHit
  for(  ; it !=  lastTrack; ++it ) {

    edm::OwnVector<SiTrackerGSRecHit2D>::const_iterator rit = it->second.begin();
    edm::OwnVector<SiTrackerGSRecHit2D>::const_iterator firstRecHit = it->second.begin();
    edm::OwnVector<SiTrackerGSRecHit2D>::const_iterator lastRecHit = it->second.end();

    //loop over rechits in track
    for ( ; rit != lastRecHit; ++rit,++recHitCounter){

      DetId detid = rit->geographicalId();
      unsigned int subdet = detid.subdetId();
      // if in the strip (subdet>2)
      if(subdet>2){

	StripSubdetector specDetId=StripSubdetector(detid);

	// if this is on a glued, then place only one hit in vector
	if(specDetId.glued()){

	  // get the track direction from the simhit
	  LocalVector simtrackdir = correspondingSimHit[recHitCounter]->localDirection();	    

	  const GluedGeomDet* gluedDet = (const GluedGeomDet*)geometry->idToDet(DetId(specDetId.glued()));
	  const StripGeomDetUnit* stripdet =(StripGeomDetUnit*) gluedDet->stereoDet();
	  
	  // global direction of track
	  GlobalVector globaldir= stripdet->surface().toGlobal(simtrackdir);
	  LocalVector gluedsimtrackdir=gluedDet->surface().toLocal(globaldir);

	  // get partner layer, it is the next one or previous one in the vector
	  edm::OwnVector<SiTrackerGSRecHit2D>::const_iterator partner = rit;
	  edm::OwnVector<SiTrackerGSRecHit2D>::const_iterator partnerNext = rit;
	  edm::OwnVector<SiTrackerGSRecHit2D>::const_iterator partnerPrev = rit;
	  partnerNext++; partnerPrev--;
	  
	  // check if this hit is on a stereo layer (== the second layer of a double sided module)
	  if(   specDetId.stereo()  ) {
	
	    int partnersFound = 0;
	    // check next one in vector
	    // safety check first
	    if(partnerNext != it->second.end() ) 
	      if( StripSubdetector( partnerNext->geographicalId() ).partnerDetId() == detid.rawId() )	{
		partner= partnerNext;
		partnersFound++;
	      }
	    // check prevoius one in vector     
	    if( rit !=  it->second.begin()) 
	      if(  StripSubdetector( partnerPrev->geographicalId() ).partnerDetId() == detid.rawId() ) {
		partnersFound++;
		partner= partnerPrev;
	      }
	    
	    
	    //std::cout << "CHECKING matched hits: Partners found = " << partnersFound << std::endl;

	    // in case partner has not been found this way, need to loop over all rechits in track to be sure
	    // (rare cases fortunately)
	    if(partnersFound==0){
	      for(edm::OwnVector<SiTrackerGSRecHit2D>::const_iterator iter = firstRecHit; iter != lastRecHit; ++iter  ){
		if( StripSubdetector( iter->geographicalId() ).partnerDetId() == detid.rawId()){
		  partnersFound++;
		  partner = iter;
		}
	      }
      	    }

	    if(partnersFound == 1) {
	      SiTrackerGSMatchedRecHit2D * theMatchedHit =GSRecHitMatcher().match( &(*partner),  &(*rit),  gluedDet  , gluedsimtrackdir );


	      //	      std::cout << "Matched  hit: isMatched =\t" << theMatchedHit->isMatched() << ", "
	      //	<<  theMatchedHit->monoHit() << ", " <<  theMatchedHit->stereoHit() << std::endl;

	      matchedMap[it->first].push_back( theMatchedHit );
	    } 
	    else{
	      // no partner to match, place projected one in map
	      SiTrackerGSMatchedRecHit2D * theProjectedHit = GSRecHitMatcher().projectOnly( &(*rit), geometry->idToDet(detid),gluedDet, gluedsimtrackdir  );
	      matchedMap[it->first].push_back( theProjectedHit );
	      //there is no partner here
	    }
	  } // end if stereo
	  else {   // we are on a mono layer
	    // usually this hit is already matched, but not if stereo hit is missing (rare cases)
	    // check if stereo hit is missing
	    int partnersFound = 0;
	    // check next one in vector
	    // safety check first
	    if(partnerNext != it->second.end() ) 
	      if( StripSubdetector( partnerNext->geographicalId() ).partnerDetId() == detid.rawId() )	{
		partnersFound++;
	      }
	    // check prevoius one in vector     
	    if( rit !=  it->second.begin()) 
	      if(  StripSubdetector( partnerPrev->geographicalId() ).partnerDetId() == detid.rawId() ) {
		partnersFound++;
	      }

	    if(partnersFound==0){ // no partner hit found this way, need to iterate over all hits to be sure (rare cases)
	      for(edm::OwnVector<SiTrackerGSRecHit2D>::const_iterator iter = firstRecHit; iter != lastRecHit; ++iter  ){
		if( StripSubdetector( iter->geographicalId() ).partnerDetId() == detid.rawId()){
		  partnersFound++;
		}
	      }
	    }	    
	    
	    
	    if(partnersFound==0){ // no partner hit found 
	      // no partner to match, place projected one one in map
	      SiTrackerGSMatchedRecHit2D * theProjectedHit = 
		GSRecHitMatcher().projectOnly( &(*rit), geometry->idToDet(detid),gluedDet, gluedsimtrackdir  );
	      
	      //std::cout << "Projected hit: isMatched =\t" << theProjectedHit->isMatched() << ", "
	      //	<<  theProjectedHit->monoHit() << ", " <<  theProjectedHit->stereoHit() << std::endl;
	      
	      matchedMap[it->first].push_back( theProjectedHit );
	    }	    
	  } // end we are on a a mono layer
	} // end if glued 
	// else matchedMap[it->first].push_back( rit->clone() );  // if not glued place the original one in vector 
	else{ //need to copy the original in a "matched" type rechit

	  SiTrackerGSMatchedRecHit2D* rit_copy = new SiTrackerGSMatchedRecHit2D(rit->localPosition(), rit->localPositionError(),
										rit->geographicalId(), 
										rit->simhitId(), rit->simtrackId(), rit->eeId(),
                                                                                rit->cluster(),
                                                                                rit->simMultX(), rit->simMultY()); 
	  //std::cout << "Simple hit  hit: isMatched =\t" << rit_copy->isMatched() << ", "
	  //    <<  rit_copy->monoHit() << ", " <<  rit_copy->stereoHit() << std::endl;
	  
	  matchedMap[it->first].push_back( rit_copy );  // if not strip place the original one in vector (making it into a matched)	
	}
	
      }// end if strip
      //      else  matchedMap[it->first].push_back( rit->clone() );  // if not strip place the original one in vector
      else { //need to copy the original in a "matched" type rechit

	SiTrackerGSMatchedRecHit2D* rit_copy = new SiTrackerGSMatchedRecHit2D(rit->localPosition(), rit->localPositionError(),
									      rit->geographicalId(), 
									      rit->simhitId(), rit->simtrackId(), rit->eeId(), 
                                                                              rit->cluster(),
									      rit->simMultX(), rit->simMultY());	
	
	//std::cout << "Simple hit  hit: isMatched =\t" << rit_copy->isMatched() << ", "
	//	  <<  rit_copy->monoHit() << ", " <<  rit_copy->stereoHit() << std::endl;
	matchedMap[it->first].push_back( rit_copy );  // if not strip place the original one in vector (makining it into a matched)
     }
      
    } // end loop over rechits
    
  }// end loop over tracks
  
  
}
