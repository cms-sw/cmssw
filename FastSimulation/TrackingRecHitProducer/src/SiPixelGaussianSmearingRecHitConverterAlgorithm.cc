/** SiPixelGaussianSmearingRecHitConverterAlgorithm.cc
 * ---------------------------------------------------------------------
 * Description:  see SiPixelGaussianSmearingRecHitConverterAlgorithm.h
 * Authors:  R. Ranieri (CERN)
 * History: Oct 11, 2006 -  initial version
 * ---------------------------------------------------------------------
 */

// SiPixel Gaussian Smearing
#include "FastSimulation/TrackingRecHitProducer/interface/SiPixelGaussianSmearingRecHitConverterAlgorithm.h"
#include "DataFormats/TrackerRecHit2D/interface/SiTrackerGSRecHit2DCollection.h"
#include "RecoLocalTracker/SiPixelRecHits/interface/PixelErrorParametrization.h"

// Geometry
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/CommonDetUnit/interface/GeomDetUnit.h"
#include "Geometry/CommonDetUnit/interface/GeomDetType.h"

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

SiPixelGaussianSmearingRecHitConverterAlgorithm::SiPixelGaussianSmearingRecHitConverterAlgorithm(edm::ParameterSet pset,
												 const PSimHit& simHit,
												 GeomDetType::SubDetector pixelPart, unsigned int layer,
												 std::vector<TH1F*> theAlphaMultiplicityCumulativeProbabilities,
												 std::vector<TH1F*> theBetaMultiplicityCumulativeProbabilities,
												 TFile* pixelResolutionFile) :
  pset_(pset) {
  
  // private members
  thePixelPart           = pixelPart;
  theLayer               = layer;
  thePixelResolutionFile = pixelResolutionFile;
  //
  
  //--- Algorithm's verbosity
  theVerboseLevel = pset_.getUntrackedParameter<int>("VerboseLevel",0);
  //
  if( thePixelPart == GeomDetEnumerators::PixelBarrel ) {
    // Resolution Barrel    
    resAlpha_binMin   = pset.getUntrackedParameter<double>("AlphaBarrel_BinMin"   ,  -0.2);
    resAlpha_binWidth = pset.getUntrackedParameter<double>("AlphaBarrel_BinWidth" ,   0.1);
    resAlpha_binN     = pset.getUntrackedParameter<int>("AlphaBarrel_BinN"        ,   4  );
    resBeta_binMin    = pset.getUntrackedParameter<double>("BetaBarrel_BinMin"    ,   0.0);
    resBeta_binWidth  = pset.getUntrackedParameter<double>("BetaBarrel_BinWidth"  ,   0.2);
    resBeta_binN      = pset.getUntrackedParameter<int>(   "BetaBarrel_BinN"      ,   7  );
    //
  } else if( thePixelPart == GeomDetEnumerators::PixelEndcap ) {
    // Resolution Forward
    resAlpha_binMin   = pset.getUntrackedParameter<double>("AlphaForward_BinMin"   ,  0.0);
    resAlpha_binWidth = pset.getUntrackedParameter<double>("AlphaForward_BinWidth" ,  0.0);
    resAlpha_binN     = pset.getUntrackedParameter<int>("AlphaBarrel_BinN"        ,   4  );
    resBeta_binMin    = pset.getUntrackedParameter<double>("BetaForward_BinMin"    ,  0.0);
    resBeta_binWidth  = pset.getUntrackedParameter<double>("BetaForward_BinWidth"  ,  0.0);
    resBeta_binN      = pset.getUntrackedParameter<int>(   "BetaBarrel_BinN"      ,   7  );
  }
  //
  
  // Run Pixel Gaussian Smearing Algorithm
  run( simHit,
       theAlphaMultiplicityCumulativeProbabilities,
       theBetaMultiplicityCumulativeProbabilities );
  //
}

void SiPixelGaussianSmearingRecHitConverterAlgorithm::run(const PSimHit& simHit,
							  std::vector<TH1F*> theAlphaMultiplicityCumulativeProbabilities,
							  std::vector<TH1F*> theBetaMultiplicityCumulativeProbabilities) {
  //
  if (theVerboseLevel > 3) {
    LogDebug("SiPixelGaussianSmearingRecHits") << " Pixel smearing in " << thePixelPart << ", Layer is " << theLayer << std::endl;
  }
  //
  // at the beginning the position is the Local Point in the local pixel module reference frame
  // alpha: angle with respect to local x axis in local (x,z) plane
  float alpha = 3.141592654 / 2.
    - acos( simHit.localDirection().x() / sqrt( simHit.localDirection().x()*simHit.localDirection().x() + simHit.localDirection().z()*simHit.localDirection().z() ) );
  // beta: angle with respect to local y axis in local (y,z) plane
  float beta = fabs( 3.141592654 / 2.
		     - acos( simHit.localDirection().y() / sqrt( simHit.localDirection().y()*simHit.localDirection().y() + simHit.localDirection().z()*simHit.localDirection().z() ) )
		     );
  //
  if (theVerboseLevel > 3) {
    LogDebug("SiPixelGaussianSmearingRecHits") << " Local Direction " << simHit.localDirection()
					       << " alpha(x) = " << alpha
					       << " beta(y) = "  << beta
					       << std::endl;
  }
  /*
    if( thePixelPart == GeomDetEnumerators::PixelBarrel ) {
    // it means that we are in the barrel
    // the barrel layers are assumed to be cylinders, to mimic the staggering of the pixel modules, alpha is smeared
    double alphaMin = theAlphaMultiplicityCumulativeProbabilities.front()->GetXaxis()->GetXmin();
    double alphaMax = theAlphaMultiplicityCumulativeProbabilities.front()->GetXaxis()->GetXmax();
    alpha -= ( alphaMin + RandFlat::shoot() * ( alphaMax - alphaMin ) ) / (float)theLayer;
    if (theVerboseLevel > 3) {
    LogDebug("SiTrackerGaussianSmearingRecHits") << " Smearing of alpha in range [" << alphaMin << "," << alphaMax << "]"
    << "\t new alpha(x) = " << alpha
    << std::endl;
    }
    }
  */
  // Generate alpha and beta multiplicity
  unsigned int alphaMultiplicity = 0;
  unsigned int betaMultiplicity  = 0;
  // random multiplicity for alpha and beta
  double alphaProbability = RandFlat::shoot();
  double betaProbability  = RandFlat::shoot();
  // search which multiplicity correspond
  int alphaBin = theAlphaMultiplicityCumulativeProbabilities.front()->GetXaxis()->FindFixBin(alpha);
  int betaBin  = theBetaMultiplicityCumulativeProbabilities.front()->GetXaxis()->FindFixBin(beta);
  for(unsigned int iMult = 0; iMult < theAlphaMultiplicityCumulativeProbabilities.size(); iMult++) {
    if(alphaProbability < theAlphaMultiplicityCumulativeProbabilities[iMult]->GetBinContent(alphaBin) ) {
      alphaMultiplicity = iMult+1;
      break;
    }
  }
  for(unsigned int iMult = 0; iMult < theBetaMultiplicityCumulativeProbabilities.size(); iMult++) {
    if(betaProbability < theBetaMultiplicityCumulativeProbabilities[iMult]->GetBinContent(betaBin) ) {
      betaMultiplicity = iMult+1;
      break;
    }
  }
  // protection against 0 or max multiplicity
  if( alphaMultiplicity == 0 || alphaMultiplicity > theAlphaMultiplicityCumulativeProbabilities.size() ) alphaMultiplicity = theAlphaMultiplicityCumulativeProbabilities.size();
  if( betaMultiplicity == 0  || alphaMultiplicity > theBetaMultiplicityCumulativeProbabilities.size()  )  betaMultiplicity  = theBetaMultiplicityCumulativeProbabilities.size();
  // protection against out-of-range (undeflows and overflows)
  if( alphaBin == 0 ) alphaBin = 1;
  if( alphaBin > theAlphaMultiplicityCumulativeProbabilities.front()->GetNbinsX() ) alphaBin = theAlphaMultiplicityCumulativeProbabilities.front()->GetNbinsX();
  if( betaBin == 0 ) betaBin = 1;
  if( betaBin > theBetaMultiplicityCumulativeProbabilities.front()->GetNbinsX() )   betaBin = theBetaMultiplicityCumulativeProbabilities.front()->GetNbinsX();
  //
  //
  if (theVerboseLevel > 3) {
    LogDebug("SiPixelGaussianSmearingRecHits") << " Multiplicity set to"
					       << "\talpha = " << alphaMultiplicity
					       << "\tbeta = "  << betaMultiplicity
					       << "\n"
					       << "  from random probability"
					       << "\talpha = " << alphaProbability
					       << "\tbeta = "  << betaProbability
					       << "\n"
					       << "  taken from bin         "
					       << "\talpha = " << alphaBin
					       << "\tbeta = "  << betaBin
					       << std::endl;	
  }
  //    
  // Compute pixel errors
  PixelErrorParametrization pixelError(pset_);
  std::pair<float,float> theErrors = pixelError.getError( thePixelPart ,
							  (int)alphaMultiplicity , (int)betaMultiplicity ,
							  alpha                  , beta                    );
  // define private mebers --> Errors
  theErrorX = (double)theErrors.first;
  theErrorY = (double)theErrors.second;
  theErrorZ = 0.0001; // 1 um means zero
  theError = LocalError( theErrorX * theErrorX,
			 0.,
			 theErrorY * theErrorY
			 ); // Local Error is 2D: (xx,xy,yy), square of sigma in first an third position as for resolution matrix
  //
  if (theVerboseLevel > 3) {
    LogDebug("SiPixelGaussianSmearingRecHits") << " Pixel Errors "
					       << "\talpha(x) = " << theErrorX
					       << "\tbeta(y) = "  << theErrorY
					       << std::endl;	
  }
  // 
  // Generate position
  // get resolution histograms
  int alphaHistBin = (int)( ( alpha - resAlpha_binMin ) / resAlpha_binWidth + 1 );
  int betaHistBin  = (int)( ( beta - resBeta_binMin ) / resBeta_binWidth + 1 );
  // protection against out-of-range (undeflows and overflows)
  if( alphaHistBin < 1 ) alphaHistBin = 1; 
  if( betaHistBin  < 1 ) betaHistBin  = 1; 
  if( alphaHistBin > (int)resAlpha_binN ) alphaHistBin = (int)resAlpha_binN; 
  if( betaHistBin  > (int)resBeta_binN  ) betaHistBin  = (int)resBeta_binN; 
  //  
  unsigned int alphaHistN = (resAlpha_binWidth != 0 ?
			     100 * alphaHistBin
			     + 10
			     + alphaMultiplicity
			     :
			     1110
			     + alphaMultiplicity);    //
  //
  unsigned int betaHistN = (resBeta_binWidth != 0 ?
			    100 * betaHistBin
			    + betaMultiplicity
			    :
			    1100 + betaMultiplicity);    //
  //
  if (theVerboseLevel > 3) {
    LogDebug("SiPixelGaussianSmearingRecHits") << " Resolution histograms chosen "
					       << "\talpha = " << alphaHistN
					       << "\tbeta = "  << betaHistN
					       << std::endl;	
  }
  //
  TH1F alphaHist = *( (TH1F*) thePixelResolutionFile->Get(  Form( "h%u" , alphaHistN ) ) );
  TH1F betaHist  = *( (TH1F*) thePixelResolutionFile->Get(  Form( "h%u" , betaHistN  ) ) );
  //
  // define private mebers --> Positions
  thePositionX = alphaHist.GetRandom();
  thePositionY = betaHist.GetRandom();
  thePositionZ = 0.0; // set at the centre of the active area
  thePosition = Local3DPoint( thePositionX , thePositionY , thePositionZ );
  //
  
}
  
