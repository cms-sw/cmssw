/** SiTrackerGaussianSmearingRecHitConverter.cc
 * --------------------------------------------------------------
 * Description:  see SiTrackerGaussianSmearingRecHitConverter.h
 * Authors:  R. Ranieri (CERN)
 * History: Sep 27, 2006 -  initial version
 * --------------------------------------------------------------
 */


// SiTracker Gaussian Smearing
#include "FastSimulation/TrackingRecHitProducer/interface/SiTrackerGaussianSmearingRecHitConverter.h"
#include "DataFormats/TrackerRecHit2D/interface/SiTrackerGSRecHit2DCollection.h"

// SiPixel Gaussian Smearing
#include "FastSimulation/TrackingRecHitProducer/interface/SiPixelGaussianSmearingRecHitConverterAlgorithm.h"
#include "RecoLocalTracker/SiPixelRecHits/interface/PixelErrorParametrization.h"

// SiStripGaussianSmearing
#include "FastSimulation/TrackingRecHitProducer/interface/SiStripGaussianSmearingRecHitConverterAlgorithm.h"

// Geometry
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/CommonDetUnit/interface/GeomDetUnit.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "Geometry/CommonDetUnit/interface/GeomDetType.h"


// Data Formats
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/Common/interface/Ref.h"
#include "SimDataFormats/CrossingFrame/interface/CrossingFrame.h"
#include "SimDataFormats/CrossingFrame/interface/MixCollection.h"

// Framework
#include "FWCore/Framework/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"

// Numbering scheme
#include "DataFormats/SiPixelDetId/interface/PXBDetId.h"
#include "DataFormats/SiPixelDetId/interface/PXFDetId.h"
#include "DataFormats/SiStripDetId/interface/TIBDetId.h"
#include "DataFormats/SiStripDetId/interface/TIDDetId.h"
#include "DataFormats/SiStripDetId/interface/TOBDetId.h"
#include "DataFormats/SiStripDetId/interface/TECDetId.h"

// CLHEP
#include "CLHEP/Random/RandFlat.h"

// STL
#include <vector>
#include <memory>
#include <string>
#include <iostream>

// ROOT
#include <TFile.h>
#include <TH1F.h>
#include <TAxis.h>

// MessageLogger
#include "FWCore/MessageLogger/interface/MessageLogger.h"

namespace cms
{
  
  SiTrackerGaussianSmearingRecHitConverter::SiTrackerGaussianSmearingRecHitConverter(edm::ParameterSet const& conf) : conf_(conf)
  {
    std::cout << "SiTrackerGaussianSmearingRecHitConverter instantiated" << std::endl;
    //--- Declare to the EDM what kind of collections we will be making.
    theRecHitsTag = conf.getParameter<std::string>( "RecHits" );
    produces<SiTrackerGSRecHit2DCollection>();
    //    LogDebug("SiTrackerGaussianSmearingRecHits") << "RecHit collection to produce: " << theRecHitsTag << std::endl;
    //--- Algorithm's verbosity
    theVerboseLevel = 
      conf.getUntrackedParameter<int>("VerboseLevel",0);
    //--- PSimHit Containers
    trackerContainers.clear();
    trackerContainers = conf.getParameter<std::vector<std::string> >("ROUList");
    //--- delta rays p cut [GeV/c] to filter PSimHits with p>
    deltaRaysPCut = conf.getParameter<double>("DeltaRaysMomentumCut");
    LogDebug("SiTrackerGaussianSmearingRecHits") << "PSimHit filter delta rays cut in momentum p > " << deltaRaysPCut << " GeV/c" << std::endl;
    //--- switch to have RecHit == PSimHit
    trackingPSimHits = conf.getParameter<bool>("trackingPSimHits");
    if(trackingPSimHits) LogDebug("SiTrackerGaussianSmearingRecHits") << "### trackingPSimHits chosen " << trackingPSimHits << std::endl;
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
    localPositionResolution_z = 0.0001; // not to be changed, set to minimum (1 um), Kalman Filter will crash if errors are exactly 0, setting 1 um means 0
    //
    LogDebug("SiTrackerGaussianSmearingRecHits") << "RecHit local position error set to" << "\n"
						 << "\tTIB1\tx = " << localPositionResolution_TIB1x << " cm\ty = " << localPositionResolution_TIB1y << " cm" << "\n"
						 << "\tTIB2\tx = " << localPositionResolution_TIB2x << " cm\ty = " << localPositionResolution_TIB2y << " cm" << "\n"
						 << "\tTIB3\tx = " << localPositionResolution_TIB3x << " cm\ty = " << localPositionResolution_TIB3y << " cm" << "\n"
						 << "\tTIB4\tx = " << localPositionResolution_TIB4x << " cm\ty = " << localPositionResolution_TIB4y << " cm" << "\n"
						 << "\tTID1\tx = " << localPositionResolution_TID1x << " cm\ty = " << localPositionResolution_TID1y << " cm" << "\n"
						 << "\tTID2\tx = " << localPositionResolution_TID2x << " cm\ty = " << localPositionResolution_TID2y << " cm" << "\n"
						 << "\tTID3\tx = " << localPositionResolution_TID3x << " cm\ty = " << localPositionResolution_TID3y << " cm" << "\n"
						 << "\tTOB1\tx = " << localPositionResolution_TOB1x << " cm\ty = " << localPositionResolution_TOB1y << " cm" << "\n"
						 << "\tTOB2\tx = " << localPositionResolution_TOB2x << " cm\ty = " << localPositionResolution_TOB2y << " cm" << "\n"
						 << "\tTOB3\tx = " << localPositionResolution_TOB3x << " cm\ty = " << localPositionResolution_TOB3y << " cm" << "\n"
						 << "\tTOB4\tx = " << localPositionResolution_TOB4x << " cm\ty = " << localPositionResolution_TOB4y << " cm" << "\n"
						 << "\tTOB5\tx = " << localPositionResolution_TOB5x << " cm\ty = " << localPositionResolution_TOB5y << " cm" << "\n"
						 << "\tTOB6\tx = " << localPositionResolution_TOB6x << " cm\ty = " << localPositionResolution_TOB6y << " cm" << "\n"
						 << "\tTEC1\tx = " << localPositionResolution_TEC1x << " cm\ty = " << localPositionResolution_TEC1y << " cm" << "\n"
						 << "\tTEC2\tx = " << localPositionResolution_TEC2x << " cm\ty = " << localPositionResolution_TEC2y << " cm" << "\n"
						 << "\tTEC3\tx = " << localPositionResolution_TEC3x << " cm\ty = " << localPositionResolution_TEC3y << " cm" << "\n"
						 << "\tTEC4\tx = " << localPositionResolution_TEC4x << " cm\ty = " << localPositionResolution_TEC4y << " cm" << "\n"
						 << "\tTEC5\tx = " << localPositionResolution_TEC5x << " cm\ty = " << localPositionResolution_TEC5y << " cm" << "\n"
						 << "\tTEC6\tx = " << localPositionResolution_TEC6x << " cm\ty = " << localPositionResolution_TEC6y << " cm" << "\n"
						 << "\tTEC7\tx = " << localPositionResolution_TEC7x << " cm\ty = " << localPositionResolution_TEC7y << " cm" << "\n"
						 << "\tAll:\tz = " << localPositionResolution_z     << " cm" 
						 << std::endl;
    //    
    // from FAMOS: take into account the angle of the strips in the barrel
    //--- The name of the files with the Pixel information
    thePixelMultiplicityFileName = conf.getParameter<std::string>( "PixelMultiplicityFile" );
    LogDebug("SiTrackerGaussianSmearingRecHits") << "Pixel multiplicity data are taken from file " << thePixelMultiplicityFileName << std::endl;
    //--- Number of histograms for alpha/beta barrel/forward multiplicity
    nAlphaBarrel  = conf.getParameter<int>("AlphaBarrelMultiplicity");
    nBetaBarrel   = conf.getParameter<int>("BetaBarrelMultiplicity");
    nAlphaForward = conf.getParameter<int>("AlphaForwardMultiplicity");
    nBetaForward  = conf.getParameter<int>("BetaForwardMultiplicity");
    LogDebug("SiTrackerGaussianSmearingRecHits") << "Pixel maximum multiplicity set to " 
						 << "\nBarrel"  << "\talpha " << nAlphaBarrel  << "\tbeta " << nBetaBarrel
						 << "\nForward" << "\talpha " << nAlphaForward << "\tbeta " << nBetaForward
						 << std::endl;
    // Resolution Barrel    
    thePixelBarrelResolutionFileName = conf.getParameter<std::string>( "PixelBarrelResolutionFile");
    resAlphaBarrel_binMin   = conf.getParameter<double>("AlphaBarrel_BinMin"  );
    resAlphaBarrel_binWidth = conf.getParameter<double>("AlphaBarrel_BinWidth");
    resAlphaBarrel_binN     = conf.getParameter<int>(   "AlphaBarrel_BinN"    );
    resBetaBarrel_binMin    = conf.getParameter<double>("BetaBarrel_BinMin"   );
    resBetaBarrel_binWidth  = conf.getParameter<double>("BetaBarrel_BinWidth" );
    resBetaBarrel_binN      = conf.getParameter<int>(   "BetaBarrel_BinN"     );
    LogDebug("SiTrackerGaussianSmearingRecHits") << "Barrel Pixel resolution data are taken from file " << thePixelBarrelResolutionFileName 
						 << "\n"
						 << "Alpha bin min = " << resAlphaBarrel_binMin
						 << "\twidth = "       << resAlphaBarrel_binWidth
						 << "\tbins = "        << resAlphaBarrel_binN
						 << "\n"
						 << " Beta bin min = " << resBetaBarrel_binMin
						 << "\twidth = "       << resBetaBarrel_binWidth
						 << "\tbins = "        << resBetaBarrel_binN
						 << std::endl;
    //
    
    // Resolution Forward
    thePixelForwardResolutionFileName = conf.getParameter<std::string>( "PixelForwardResolutionFile");
    resAlphaForward_binMin   = conf.getParameter<double>("AlphaForward_BinMin"   );
    resAlphaForward_binWidth = conf.getParameter<double>("AlphaForward_BinWidth" );
    resAlphaForward_binN     = conf.getParameter<int>(   "AlphaForward_BinN"     );
    resBetaForward_binMin    = conf.getParameter<double>("BetaForward_BinMin"    );
    resBetaForward_binWidth  = conf.getParameter<double>("BetaForward_BinWidth"  );
    resBetaForward_binN      = conf.getParameter<int>(   "BetaForward_BinN"      );
    LogDebug("SiTrackerGaussianSmearingRecHits") << "Forward Pixel resolution data are taken from file " << thePixelForwardResolutionFileName 
						 << "\n"
						 << "Alpha bin min = " << resAlphaForward_binMin
						 << "\twidth = "       << resAlphaForward_binWidth
						 << "\tbins = "        << resAlphaForward_binN
						 << "\n"
						 << " Beta bin min = " << resBetaForward_binMin
						 << "\twidth = "       << resBetaForward_binWidth
						 << "\tbins = "        << resBetaForward_binN
						 << std::endl;
    //
    
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
    LogDebug("SiTrackerGaussianSmearingRecHits") << "RecHit finding probability set to" << "\n"
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
    //    
    // load pixel data
    loadPixelData();
    //
  }
  
  void SiTrackerGaussianSmearingRecHitConverter::loadPixelData() {
    // load multiplicity cumulative probabilities
    // root files
    thePixelDataFile              = new TFile ( edm::FileInPath( thePixelMultiplicityFileName      ).fullPath().c_str() , "READ" );
    thePixelBarrelResolutionFile  = new TFile ( edm::FileInPath( thePixelBarrelResolutionFileName  ).fullPath().c_str() , "READ" );
    thePixelForwardResolutionFile = new TFile ( edm::FileInPath( thePixelForwardResolutionFileName ).fullPath().c_str() , "READ" );
    //
    // alpha barrel
    loadPixelData( thePixelDataFile, nAlphaBarrel  , std::string("hist_alpha_barrel")  , theBarrelMultiplicityAlphaCumulativeProbabilities  );
    // 
    // beta barrel
    loadPixelData( thePixelDataFile, nBetaBarrel   , std::string("hist_beta_barrel")   , theBarrelMultiplicityBetaCumulativeProbabilities   );
    // 
    // alpha forward
    loadPixelData( thePixelDataFile, nAlphaForward , std::string("hist_alpha_forward") , theForwardMultiplicityAlphaCumulativeProbabilities );
    // 
    // beta forward
    loadPixelData( thePixelDataFile, nBetaForward  , std::string("hist_beta_forward")  , theForwardMultiplicityBetaCumulativeProbabilities  );
    // 
  }
  
  void SiTrackerGaussianSmearingRecHitConverter::loadPixelData( TFile* pixelDataFile, unsigned int nMultiplicity, std::string histName,
								std::vector<TH1F*>& theMultiplicityCumulativeProbabilities ) {
    std::string histName_i = histName + "_%u"; // needed to open histograms with a for
    theMultiplicityCumulativeProbabilities.clear();
    //
    std::vector<double> mult; // vector with fixed multiplicity
    for(unsigned int i = 0; i<nMultiplicity; i++) {
      TH1F addHist = *((TH1F*) pixelDataFile->Get( Form( histName_i.c_str() ,i+1 )));
      if(i==0) {
	theMultiplicityCumulativeProbabilities.push_back( new TH1F(addHist) );
      } else {
	TH1F sumHist = *(theMultiplicityCumulativeProbabilities[i-1]);
	sumHist.Add(&addHist);
	theMultiplicityCumulativeProbabilities.push_back( new TH1F(sumHist) );
      }
    }
    // Logger
    if (theVerboseLevel > 3) {
      std::cout << " Multiplicity cumulated probability " << histName << std::endl;
      for(unsigned int iMult = 0; iMult<theMultiplicityCumulativeProbabilities.size(); iMult++) {
	for(int iBin = 1; iBin<=theMultiplicityCumulativeProbabilities[iMult]->GetNbinsX(); iBin++) {
	  std::cout << " Multiplicity " << iMult+1 << " bin " << iBin << " low edge = " << theMultiplicityCumulativeProbabilities[iMult]->GetBinLowEdge(iBin)
		    << " prob = " << (theMultiplicityCumulativeProbabilities[iMult])->GetBinContent(iBin) // remember in ROOT bin starts from 1 (0 underflow, nBin+1 overflow)
		    << std::endl;
	}
      }
    }
    //
  }
  
  // Destructor
  SiTrackerGaussianSmearingRecHitConverter::~SiTrackerGaussianSmearingRecHitConverter() {
    theBarrelMultiplicityAlphaCumulativeProbabilities.clear();
    theBarrelMultiplicityBetaCumulativeProbabilities.clear();
    theForwardMultiplicityAlphaCumulativeProbabilities.clear();
    theForwardMultiplicityBetaCumulativeProbabilities.clear();
    //
    delete thePixelDataFile;
    delete thePixelBarrelResolutionFile;
    delete thePixelForwardResolutionFile;
  }  
  
  void SiTrackerGaussianSmearingRecHitConverter::beginJob(const edm::EventSetup& c) {}
  
  void SiTrackerGaussianSmearingRecHitConverter::produce(edm::Event& e, const edm::EventSetup& es) {
    // Step A: Get Inputs (PSimHit's)
    edm::Handle<CrossingFrame> cf;
    e.getByType(cf);
    
    std::auto_ptr<MixCollection<PSimHit> > allTrackerHits(new MixCollection<PSimHit>(cf.product(),trackerContainers));
    
    edm::ESHandle<TrackerGeometry> geometry;
    es.get<TrackerDigiGeometryRecord> ().get (geometry);
    
    // Step B: create empty output collection
    std::auto_ptr<SiTrackerGSRecHit2DCollection> output(new SiTrackerGSRecHit2DCollection);
    
    // Step C: Iterate over DetIds and invoke the Gaussian Smearing algorithm
    // on each DetUnit
    run( *allTrackerHits , *output , geometry );
    
    // Step D: write output to file
    e.put(output);
    
  }
  
  
  void SiTrackerGaussianSmearingRecHitConverter::run(MixCollection<PSimHit>& input,
						     SiTrackerGSRecHit2DCollection& output,
						     edm::ESHandle<TrackerGeometry> & geometry) {
    int numberOfPSimHits = 0;
    
    MixCollection<PSimHit>::iterator isim;
    
    int simHitCounter = -1;
    
    // loop on PSimHits
    for (isim=input.begin(); isim!= input.end();isim++) {
      simHitCounter++;
      DetId det((*isim).detUnitId());
      unsigned int detid = det.rawId();
      // filter PSimHit (delta rays momentum cut)
      if( (*isim).pabs() > deltaRaysPCut ) {
	//
	edm::OwnVector<SiTrackerGSRecHit2D> recHits;
	recHits.clear();
	numberOfPSimHits++;	
	// gaussian smearing
	Local3DPoint position;
	LocalError error;
	unsigned int alphaMult = 0;
	unsigned int betaMult  = 0;
	bool isCreated = gaussianSmearing(*isim, position, error, alphaMult, betaMult);
	//
	if(isCreated) {
	  // create RecHit
	  if (theVerboseLevel > 2) {
	    LogDebug("SiTrackerGaussianSmearingRecHits") << " *** " << std::endl 
							 << "Created a RecHit with local position " << position << " and local error " << error << "\n"
							 << "   from PSimHit number " << simHitCounter << " with local position " << (*isim).localPosition()
							 << " from track " << (*isim).trackId()
							 << " with pixel multiplicity alpha(x) = " << alphaMult << " beta(y) = " << betaMult
							 << " in detector " << detid
							 << std::endl;
	  }
	  recHits.push_back( new SiTrackerGSRecHit2D( position , error , det, simHitCounter, (*isim).trackId(), alphaMult , betaMult ) );
	  output.put(det, recHits.begin(), recHits.end());
	  if (recHits.size() > 0) {
	    if (theVerboseLevel > 2) 
	      LogDebug("SiTrackerGaussianSmearingRecHits") << " Found " 
							   << recHits.size() << " RecHits on " << detid;	
	  }
	} else {
	  LogDebug("SiTrackerGaussianSmearingRecHits") << " *** " << " RecHit not created due to hit finding in-efficiency " << "\n"
						       << "   from a PSimHit with local position " << (*isim).localPosition() << " from track " << (*isim).trackId()
						       << " in detector " << detid
						       << std::endl;
	}
      } else {
	if (theVerboseLevel > 2) 
	  LogDebug("SiTrackerGaussianSmearingRecHits") << " PSimHit skipped p = " 
						       << (*isim).pabs() << " GeV/c on " << detid
						       << "(momentum cut set to " << deltaRaysPCut << " GeV/c)";		
      }
    }
    
    if (theVerboseLevel > 2) LogDebug ("SiTrackerGaussianSmearingRecHits") 
      << "SiTrackerGaussianSmearingRecHits converted " << numberOfPSimHits
      << " PSimHit's into SiTrackerGSRecHit2D"; 
    
  }
  
  bool SiTrackerGaussianSmearingRecHitConverter::gaussianSmearing(const PSimHit& simHit, Local3DPoint& position , LocalError& error,
								  unsigned int& alphaMult, unsigned int& betaMult) {
    unsigned int subdet   = DetId(simHit.detUnitId()).subdetId();
    unsigned int detid    = DetId(simHit.detUnitId()).rawId();
    
    LogDebug ("SiTrackerGaussianSmearingRecHits") << "\tSubdetector " << subdet << " rawid " << detid << std::endl;
    
    if(trackingPSimHits) {
      // z is fixed for all detectors, in case of errors resolution is fixed also for x and y to 1 um (zero)
      // The Matrix is the Covariance Matrix, sigma^2 on diagonal!!!
      error = LocalError( localPositionResolution_z * localPositionResolution_z , 
			  0.0 , 
			  localPositionResolution_z * localPositionResolution_z  );
      //
      // starting from PSimHit local position
      position = simHit.localPosition();
      if (theVerboseLevel > 2) {
	LogDebug("SiTrackerGaussianSmearingRecHits") << " Tracking PSimHit position set to  " << position;
      }
      return true; // RecHit == PSimHit with 100% hit finding efficiency
    }
    //
    
    // hit finding probability --> RecHit will be created if and only if hitFindingProbability <= theHitFindingProbability_###
    double hitFindingProbability = RandFlat::shoot();
    if (theVerboseLevel > 2) {
      LogDebug("SiTrackerGaussianSmearingRecHits") << " Hit finding probability draw: " << hitFindingProbability << std::endl;;
    }
    
    switch (subdet) {
      // Pixel Barrel
    case 1:
      {
	PXBDetId module(detid);
	unsigned int theLayer = module.layer();
	LogDebug ("SiTrackerGaussianSmearingRecHits") << "\tPixel Barrel Layer " << theLayer << std::endl;
	if( hitFindingProbability > theHitFindingProbability_PXB ) return false;
	//
	SiPixelGaussianSmearingRecHitConverterAlgorithm siPixelAlgorithm(conf_, simHit,
									 GeomDetEnumerators::PixelBarrel, theLayer,
									 theBarrelMultiplicityAlphaCumulativeProbabilities,
									 theBarrelMultiplicityBetaCumulativeProbabilities,
									 thePixelBarrelResolutionFile);
	position  = siPixelAlgorithm.getPosition();
	error     = siPixelAlgorithm.getError();
	alphaMult = siPixelAlgorithm.getPixelMultiplicityAlpha();
	betaMult  = siPixelAlgorithm.getPixelMultiplicityBeta();
	return true;
	break;
      }
      // Pixel Forward
    case 2:
      {
	PXFDetId module(detid);
	unsigned int theDisk = module.disk();
	LogDebug ("SiTrackerGaussianSmearingRecHits") << "\tPixel Forward Disk " << theDisk << std::endl;
	if( hitFindingProbability > theHitFindingProbability_PXF ) return false;
	//
	SiPixelGaussianSmearingRecHitConverterAlgorithm siPixelAlgorithm(conf_, simHit,
									 GeomDetEnumerators::PixelEndcap, theDisk,
									 theForwardMultiplicityAlphaCumulativeProbabilities,
									 theForwardMultiplicityBetaCumulativeProbabilities,
									 thePixelForwardResolutionFile);
	position = siPixelAlgorithm.getPosition();
	error    = siPixelAlgorithm.getError();
	alphaMult = siPixelAlgorithm.getPixelMultiplicityAlpha();
	betaMult  = siPixelAlgorithm.getPixelMultiplicityBeta();
	return true;
	break;
      }
      // TIB
    case 3:
      {
	TIBDetId module(detid);
	unsigned int theLayer  = module.layer();
	LogDebug ("SiTrackerGaussianSmearingRecHits") << "\tTIB Layer " << theLayer << std::endl;
	//
	HepSymMatrix resolution(3,1); // 3x3 symmetric matrix, second element: set to Identity
	resolution[2][2] = localPositionResolution_z * localPositionResolution_z;
	
	switch (theLayer) {
	case 1:
	  {
	    resolution[0][0] = localPositionResolution_TIB1x * localPositionResolution_TIB1x;
	    resolution[1][1] = localPositionResolution_TIB1y * localPositionResolution_TIB1y;
	    if( hitFindingProbability > theHitFindingProbability_TIB1 ) return false;
	    break;
	  }
	case 2:
	  {
	    resolution[0][0] = localPositionResolution_TIB2x * localPositionResolution_TIB2x;
	    resolution[1][1] = localPositionResolution_TIB2y * localPositionResolution_TIB2y;
	    if( hitFindingProbability > theHitFindingProbability_TIB2 ) return false;
	    break;
	  }
	case 3:
	  {
	    resolution[0][0] = localPositionResolution_TIB3x * localPositionResolution_TIB3x;
	    resolution[1][1] = localPositionResolution_TIB3y * localPositionResolution_TIB3y;
	    if( hitFindingProbability > theHitFindingProbability_TIB3 ) return false;
	    break;
	  }
	case 4:
	  {
	    resolution[0][0] = localPositionResolution_TIB4x * localPositionResolution_TIB4x;
	    resolution[1][1] = localPositionResolution_TIB4y * localPositionResolution_TIB4y;
	    if( hitFindingProbability > theHitFindingProbability_TIB4 ) return false;
	    break;
	  }
	default:
	  {
	    LogError ("SiTrackerGaussianSmearingRecHits") << "\tTIB Layer not valid " << theLayer << std::endl;
	    return false;
	    break;
	  }
	}
	SiStripGaussianSmearingRecHitConverterAlgorithm siStripAlgorithm(conf_, simHit, resolution);
	position = siStripAlgorithm.getPosition();
	error    = siStripAlgorithm.getError();
	alphaMult = 0;
	betaMult  = 0;
	return true;
	break;
      } // TIB
      
	// TID
    case 4:
      {
	TIDDetId module(detid);
	unsigned int theRing  = module.ring();
	LogDebug ("SiTrackerGaussianSmearingRecHits") << "\tTID Ring " << theRing << std::endl;
	HepSymMatrix resolution(3,1); // 3x3 symmetric matrix, second element: set to Identity
	resolution[0][0] = localPositionResolution_z * localPositionResolution_z;
	
	switch (theRing) {
	case 1:
	  {
	    resolution[0][0] = localPositionResolution_TID1x * localPositionResolution_TID1x;
	    resolution[1][1] = localPositionResolution_TID1y * localPositionResolution_TID1y;
	    if( hitFindingProbability > theHitFindingProbability_TID1 ) return false;
	    break;
	  }
	case 2:
	  {
	    resolution[0][0] = localPositionResolution_TID2x * localPositionResolution_TID2x;
	    resolution[1][1] = localPositionResolution_TID2y * localPositionResolution_TID2y;
	    if( hitFindingProbability > theHitFindingProbability_TID2 ) return false;
	    break;
	  }
	case 3:
	  {
	    resolution[0][0] = localPositionResolution_TID3x * localPositionResolution_TID3x;
	    resolution[1][1] = localPositionResolution_TID3y * localPositionResolution_TID3y;
	    if( hitFindingProbability > theHitFindingProbability_TID3 ) return false;
	    break;
	  }
	default:
	  {
	    LogError ("SiTrackerGaussianSmearingRecHits") << "\tTID Ring not valid " << theRing << std::endl;
	    return false;
	    break;
	  }
	}
	SiStripGaussianSmearingRecHitConverterAlgorithm siStripAlgorithm(conf_, simHit, resolution);
	position = siStripAlgorithm.getPosition();
	error    = siStripAlgorithm.getError();
	alphaMult = 0;
	betaMult  = 0;
	return true;
	break;
      } // TID
      
	// TOB
    case 5:
      {
	TOBDetId module(detid);
	unsigned int theLayer  = module.layer();
	LogDebug ("SiTrackerGaussianSmearingRecHits") << "\tTOB Layer " << theLayer << std::endl;
	HepSymMatrix resolution(3,1); // 3x3 symmetric matrix, second element: set to Identity
	resolution[0][0] = localPositionResolution_z * localPositionResolution_z;
	
	switch (theLayer) {
	case 1:
	  {
	    resolution[0][0] = localPositionResolution_TOB1x * localPositionResolution_TOB1x;
	    resolution[1][1] = localPositionResolution_TOB1y * localPositionResolution_TOB1y;
	    if( hitFindingProbability > theHitFindingProbability_TOB1 ) return false;
	    break;
	  }
	case 2:
	  {
	    resolution[0][0] = localPositionResolution_TOB2x * localPositionResolution_TOB2x;
	    resolution[1][1] = localPositionResolution_TOB2y * localPositionResolution_TOB2y;
	    if( hitFindingProbability > theHitFindingProbability_TOB2 ) return false;
	    break;
	  }
	case 3:
	  {
	    resolution[0][0] = localPositionResolution_TOB3x * localPositionResolution_TOB3x;
	    resolution[1][1] = localPositionResolution_TOB3y * localPositionResolution_TOB3y;
	    if( hitFindingProbability > theHitFindingProbability_TOB3 ) return false;
	    break;
	  }
	case 4:
	  {
	    resolution[0][0] = localPositionResolution_TOB4x * localPositionResolution_TOB4x;
	    resolution[1][1] = localPositionResolution_TOB4y * localPositionResolution_TOB4y;
	    if( hitFindingProbability > theHitFindingProbability_TOB4 ) return false;
	    break;
	  }
	case 5:
	  {
	    resolution[0][0] = localPositionResolution_TOB5x * localPositionResolution_TOB5x;
	    resolution[1][1] = localPositionResolution_TOB5y * localPositionResolution_TOB5y;
	    if( hitFindingProbability > theHitFindingProbability_TOB5 ) return false;
	    break;
	  }
	case 6:
	  {
	    resolution[0][0] = localPositionResolution_TOB6x * localPositionResolution_TOB6x;
	    resolution[1][1] = localPositionResolution_TOB6y * localPositionResolution_TOB6y;
	    if( hitFindingProbability > theHitFindingProbability_TOB6 ) return false;
	    break;
	  }
	default:
	  {
	    LogError ("SiTrackerGaussianSmearingRecHits") << "\tTOB Layer not valid " << theLayer << std::endl;
	    return false;
	    break;
	  }
	}
	SiStripGaussianSmearingRecHitConverterAlgorithm siStripAlgorithm(conf_, simHit, resolution);
	position = siStripAlgorithm.getPosition();
	error    = siStripAlgorithm.getError();
	alphaMult = 0;
	betaMult  = 0;
	return true;
	break;
      } // TOB
      
	// TEC
    case 6:
      {
	TECDetId module(detid);
	unsigned int theRing  = module.ring();
	LogDebug ("SiTrackerGaussianSmearingRecHits") << "\tTEC Ring " << theRing << std::endl;
	HepSymMatrix resolution(3,1); // 3x3 symmetric matrix, second element: set to Identity
	resolution[0][0] = localPositionResolution_z * localPositionResolution_z;
	
	switch (theRing) {
	case 1:
	  {
	    resolution[0][0] = localPositionResolution_TEC1x * localPositionResolution_TEC1x;
	    resolution[1][1] = localPositionResolution_TEC1y * localPositionResolution_TEC1y;
	    if( hitFindingProbability > theHitFindingProbability_TEC1 ) return false;
	    break;
	  }
	case 2:
	  {
	    resolution[0][0] = localPositionResolution_TEC2x * localPositionResolution_TEC2x;
	    resolution[1][1] = localPositionResolution_TEC2y * localPositionResolution_TEC2y;
	    if( hitFindingProbability > theHitFindingProbability_TEC2 ) return false;
	    break;
	  }
	case 3:
	  {
	    resolution[0][0] = localPositionResolution_TEC3x * localPositionResolution_TEC3x;
	    resolution[1][1] = localPositionResolution_TEC3y * localPositionResolution_TEC3y;
	    if( hitFindingProbability > theHitFindingProbability_TEC3 ) return false;
	    break;
	  }
	case 4:
	  {
	    resolution[0][0] = localPositionResolution_TEC4x * localPositionResolution_TEC4x;
	    resolution[1][1] = localPositionResolution_TEC4y * localPositionResolution_TEC4y;
	    if( hitFindingProbability > theHitFindingProbability_TEC4 ) return false;
	    break;
	  }
	case 5:
	  {
	    resolution[0][0] = localPositionResolution_TEC5x * localPositionResolution_TEC5x;
	    resolution[1][1] = localPositionResolution_TEC5y * localPositionResolution_TEC5y;
	    if( hitFindingProbability > theHitFindingProbability_TEC5 ) return false;
	    break;
	  }
	case 6:
	  {
	    resolution[0][0] = localPositionResolution_TEC6x * localPositionResolution_TEC6x;
	    resolution[1][1] = localPositionResolution_TEC6y * localPositionResolution_TEC6y;
	    if( hitFindingProbability > theHitFindingProbability_TEC6 ) return false;
	    break;
	  }
	case 7:
	  {
	    resolution[0][0] = localPositionResolution_TEC7x * localPositionResolution_TEC7x;
	    resolution[1][1] = localPositionResolution_TEC7y * localPositionResolution_TEC7y;
	    if( hitFindingProbability > theHitFindingProbability_TEC7 ) return false;
	    break;
	  }
	default:
	  {
	    LogError ("SiTrackerGaussianSmearingRecHits") << "\tTEC Ring not valid " << theRing << std::endl;
	    return false;
	    break;
	  }
	}
	SiStripGaussianSmearingRecHitConverterAlgorithm siStripAlgorithm(conf_, simHit, resolution);
	position = siStripAlgorithm.getPosition();
	error    = siStripAlgorithm.getError();
	alphaMult = 0;
	betaMult  = 0;
	return true;
	break;
      } // TEC
      
    default:
      {
	LogError ("SiTrackerGaussianSmearingRecHits") << "\tTracker subdetector not valid " << subdet << std::endl;
	return false;
	break;
      }
      
    } // subdetector case
    //
  }   
  
}  // end of namespace cms

