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

const float PI = 3.141593;

SiPixelGaussianSmearingRecHitConverterAlgorithm::SiPixelGaussianSmearingRecHitConverterAlgorithm(
  edm::ParameterSet& pset,
  GeomDetType::SubDetector pixelPart,
  std::vector<TH1F*>& alphaMultiplicityCumulativeProbabilities,
  std::vector<TH1F*>& betaMultiplicityCumulativeProbabilities, 
  TFile* pixelResolutionFile) :
  pset_(pset),
  thePixelPart(pixelPart),
  theAlphaMultiplicityCumulativeProbabilities(alphaMultiplicityCumulativeProbabilities),
  theBetaMultiplicityCumulativeProbabilities(betaMultiplicityCumulativeProbabilities),
  thePixelResolutionFile(pixelResolutionFile)
{
  
  //--- Algorithm's verbosity
  theVerboseLevel = pset_.getUntrackedParameter<int>("VerboseLevel",0);

  //
  if( thePixelPart == GeomDetEnumerators::PixelBarrel ) {
    // Resolution Barrel    
    resAlpha_binMin   = pset.getParameter<double>("AlphaBarrel_BinMin"  );
    resAlpha_binWidth = pset.getParameter<double>("AlphaBarrel_BinWidth");
    resAlpha_binN     = pset.getParameter<int>("AlphaBarrel_BinN"       );
    resBeta_binMin    = pset.getParameter<double>("BetaBarrel_BinMin"   );
    resBeta_binWidth  = pset.getParameter<double>("BetaBarrel_BinWidth" );
    resBeta_binN      = pset.getParameter<int>(   "BetaBarrel_BinN"     );
    //
  } else if( thePixelPart == GeomDetEnumerators::PixelEndcap ) {
    // Resolution Forward
    resAlpha_binMin   = pset.getParameter<double>("AlphaForward_BinMin"  );
    resAlpha_binWidth = pset.getParameter<double>("AlphaForward_BinWidth");
    resAlpha_binN     = pset.getParameter<int>("AlphaBarrel_BinN"        );
    resBeta_binMin    = pset.getParameter<double>("BetaForward_BinMin"   );
    resBeta_binWidth  = pset.getParameter<double>("BetaForward_BinWidth" );
    resBeta_binN      = pset.getParameter<int>(   "BetaBarrel_BinN"      );
  }
  // Initialize PixelErrorParametrization (time consuming!)
  pixelError = new PixelErrorParametrization(pset_);
  
  // Run Pixel Gaussian Smearing Algorithm
  //  run( simHit, detUnit,
  //       theAlphaMultiplicityCumulativeProbabilities,
  //       theBetaMultiplicityCumulativeProbabilities );
  //
}

SiPixelGaussianSmearingRecHitConverterAlgorithm::~SiPixelGaussianSmearingRecHitConverterAlgorithm()
{

  delete pixelError;

}

void SiPixelGaussianSmearingRecHitConverterAlgorithm::run(
  const PSimHit& simHit, 
  const PixelGeomDetUnit* detUnit)
{

  if (theVerboseLevel > 3) {
    LogDebug("SiPixelGaussianSmearingRecHits") 
      << " Pixel smearing in " << thePixelPart 
      << std::endl;
  }
  //
  // at the beginning the position is the Local Point in the local pixel module reference frame
  // same code as in PixelCPEBase
  LocalVector localDir = simHit.momentumAtEntry().unit();
  float locx = localDir.x();
  float locy = localDir.y();
  float locz = localDir.z();
  // alpha: angle with respect to local x axis in local (x,z) plane
  float alpha = acos(locx/sqrt(locx*locx+locz*locz));
  if ( isFlipped( detUnit ) ) { // &&& check for FPIX !!!
    LogDebug("SiPixelGaussianSmearingRecHits") << " isFlipped " << std::endl;
    alpha = PI - alpha ;
  }
  // beta: angle with respect to local y axis in local (y,z) plane
  float beta = acos(locy/sqrt(locy*locy+locz*locz));
  
  float alphaToBeUsedForRootFiles = alpha - PI/2.;
  float betaToBeUsedForRootFiles  = PI/2. - beta;
  
  //
  if (theVerboseLevel > 3) {
    LogDebug("SiPixelGaussianSmearingRecHits") 
      << " Local Direction " << simHit.localDirection()
      << " alpha(x) = " << alpha
      << " beta(y) = "  << beta
      << " alpha for root files = " << alphaToBeUsedForRootFiles
      << " beta  for root files = " << betaToBeUsedForRootFiles
      << std::endl;
  }
  // Generate alpha and beta multiplicity
  unsigned int alphaMultiplicity = 0;
  unsigned int betaMultiplicity  = 0;
  // random multiplicity for alpha and beta
  double alphaProbability = RandFlat::shoot();
  double betaProbability  = RandFlat::shoot();
  // search which multiplicity correspond
  int alphaBin = theAlphaMultiplicityCumulativeProbabilities.front()->GetXaxis()->FindFixBin(alphaToBeUsedForRootFiles);
  int betaBin  = theBetaMultiplicityCumulativeProbabilities.front()->GetXaxis()->FindFixBin(betaToBeUsedForRootFiles);
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
  if( betaMultiplicity == 0  || alphaMultiplicity > theBetaMultiplicityCumulativeProbabilities.size()  ) betaMultiplicity  = theBetaMultiplicityCumulativeProbabilities.size();
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
  std::pair<float,float> theErrors = pixelError->getError( thePixelPart ,
							  (int)alphaMultiplicity , (int)betaMultiplicity ,
							  alpha                  , beta                    );
  // define private mebers --> Errors
  theErrorX = sqrt((double)theErrors.first);  // PixelErrorParametrization returns sigma^2
  theErrorY = sqrt((double)theErrors.second); // PixelErrorParametrization returns sigma^2
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
  int alphaHistBin = (int)( ( alphaToBeUsedForRootFiles - resAlpha_binMin ) / resAlpha_binWidth + 1 );
  int betaHistBin  = (int)( ( betaToBeUsedForRootFiles  - resBeta_binMin )  / resBeta_binWidth + 1 );
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
  TH1F* alphaHist = (TH1F*) thePixelResolutionFile->Get(  Form( "h%u" , alphaHistN ) );
  TH1F* betaHist  = (TH1F*) thePixelResolutionFile->Get(  Form( "h%u" , betaHistN  ) );
  //
  // define private mebers --> Positions
  thePositionX = alphaHist->GetRandom();
  thePositionY = betaHist->GetRandom();
  thePositionZ = 0.0; // set at the centre of the active area
  thePosition = Local3DPoint( thePositionX , thePositionY , thePositionZ );
  // define private mebers --> Multiplicities
  thePixelMultiplicityAlpha = alphaMultiplicity;
  thePixelMultiplicityBeta  = betaMultiplicity;
  //
  
}
 
//-----------------------------------------------------------------------------
// I COPIED FROM THE PixelCPEBase BECAUSE IT'S BETTER THAN REINVENT IT
// The isFlipped() is a silly way to determine which detectors are inverted.
// In the barrel for every 2nd ladder the E field direction is in the
// global r direction (points outside from the z axis), every other
// ladder has the E field inside. Something similar is in the 
// forward disks (2 sides of the blade). This has to be recognised
// because the charge sharing effect is different.
//
// The isFliped does it by looking and the relation of the local (z always
// in the E direction) to global coordinates. There is probably a much 
// better way.
//-----------------------------------------------------------------------------
bool SiPixelGaussianSmearingRecHitConverterAlgorithm::isFlipped(const PixelGeomDetUnit* theDet) const {
  // Check the relative position of the local +/- z in global coordinates.
  float tmp1 = theDet->surface().toGlobal(Local3DPoint(0.,0.,0.)).perp();
  float tmp2 = theDet->surface().toGlobal(Local3DPoint(0.,0.,1.)).perp();
  //  std::cout << " 1: " << tmp1 << " 2: " << tmp2 << std::endl;
  if ( tmp2<tmp1 ) return true;
  else return false;    
}
 
