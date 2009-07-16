/** SiPixelGaussianSmearingRecHitConverterAlgorithm.cc
 * ---------------------------------------------------------------------
 * Description:  see SiPixelGaussianSmearingRecHitConverterAlgorithm.h
 * Authors:  R. Ranieri (CERN), M. Galanti
 * History: Oct 11, 2006 -  initial version
 * ---------------------------------------------------------------------
 */

// SiPixel Gaussian Smearing
#include "FastSimulation/TrackingRecHitProducer/interface/SiPixelGaussianSmearingRecHitConverterAlgorithm.h"
//#include "DataFormats/TrackerRecHit2D/interface/SiTrackerGSRecHit2DCollection.h"
#include "RecoLocalTracker/SiPixelRecHits/interface/PixelErrorParametrization.h"

// Geometry
//#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/CommonDetUnit/interface/GeomDetUnit.h"
//#include "Geometry/CommonDetUnit/interface/GeomDetType.h"
#include "Geometry/TrackerTopology/interface/RectangularPixelTopology.h"

// Famos
#include "FastSimulation/Utilities/interface/RandomEngine.h"
#include "FastSimulation/Utilities/interface/SimpleHistogramGenerator.h"

// STL

// ROOT
#include <TFile.h>
//#include <TH1F.h>
//#include <TAxis.h>

//#define FAMOS_DEBUG

const double PI = 3.14159265358979323;

SiPixelGaussianSmearingRecHitConverterAlgorithm::SiPixelGaussianSmearingRecHitConverterAlgorithm(
  const edm::ParameterSet& pset,
  GeomDetType::SubDetector pixelPart,
  std::vector<TH1F*>& alphaMultiplicityCumulativeProbabilities,
  std::vector<TH1F*>& betaMultiplicityCumulativeProbabilities, 
  TFile* pixelResolutionFile,
  const RandomEngine* engine)
:
  pset_(pset),
  thePixelPart(pixelPart),
  theAlphaMultiplicityCumulativeProbabilities(alphaMultiplicityCumulativeProbabilities),
  theBetaMultiplicityCumulativeProbabilities(betaMultiplicityCumulativeProbabilities),
  thePixelResolutionFile(pixelResolutionFile),
  random(engine)
{
  // Switch between old (ORCA) and new (CMSSW) pixel parameterization
  useCMSSWPixelParameterization = pset.getParameter<bool>("UseCMSSWPixelParametrization");

  if(useCMSSWPixelParameterization) {
    if( thePixelPart == GeomDetEnumerators::PixelBarrel ) {
      // Resolution Barrel    
      resAlpha_binMin   = 
        pset.getParameter<double>("AlphaBarrel_BinMinNew"  );
      resAlpha_binWidth = 
        pset.getParameter<double>("AlphaBarrel_BinWidthNew");
      resAlpha_binN     = 
        pset.getParameter<int>("AlphaBarrel_BinNNew"       );
      resBeta_binMin    = 
        pset.getParameter<double>("BetaBarrel_BinMinNew"   );
      resBeta_binWidth  = 
        pset.getParameter<double>("BetaBarrel_BinWidthNew" );
      resBeta_binN      = 
        pset.getParameter<int>(   "BetaBarrel_BinNNew"     );
      //
    } else if( thePixelPart == GeomDetEnumerators::PixelEndcap ) {
      // Resolution Forward
      resAlpha_binMin   = 
        pset.getParameter<double>("AlphaForward_BinMinNew"  );
      resAlpha_binWidth = 
        pset.getParameter<double>("AlphaForward_BinWidthNew");
      resAlpha_binN     = 
        pset.getParameter<int>("AlphaBarrel_BinNNew"        );
      resBeta_binMin    = 
        pset.getParameter<double>("BetaForward_BinMinNew"   );
      resBeta_binWidth  = 
        pset.getParameter<double>("BetaForward_BinWidthNew" );
      resBeta_binN      = 
        pset.getParameter<int>(   "BetaBarrel_BinNNew"      );
    }
  } else {
        if( thePixelPart == GeomDetEnumerators::PixelBarrel ) {
      // Resolution Barrel    
      resAlpha_binMin   = 
        pset.getParameter<double>("AlphaBarrel_BinMin"  );
      resAlpha_binWidth = 
        pset.getParameter<double>("AlphaBarrel_BinWidth");
      resAlpha_binN     = 
        pset.getParameter<int>("AlphaBarrel_BinN"       );
      resBeta_binMin    = 
        pset.getParameter<double>("BetaBarrel_BinMin"   );
      resBeta_binWidth  = 
        pset.getParameter<double>("BetaBarrel_BinWidth" );
      resBeta_binN      = 
        pset.getParameter<int>(   "BetaBarrel_BinN"     );
      //
    } else if( thePixelPart == GeomDetEnumerators::PixelEndcap ) {
      // Resolution Forward
      resAlpha_binMin   = 
        pset.getParameter<double>("AlphaForward_BinMin"  );
      resAlpha_binWidth = 
        pset.getParameter<double>("AlphaForward_BinWidth");
      resAlpha_binN     = 
        pset.getParameter<int>("AlphaBarrel_BinN"        );
      resBeta_binMin    = 
        pset.getParameter<double>("BetaForward_BinMin"   );
      resBeta_binWidth  = 
        pset.getParameter<double>("BetaForward_BinWidth" );
      resBeta_binN      = 
        pset.getParameter<int>(   "BetaBarrel_BinN"      );
    }
  }
  // Initialize PixelErrorParametrization (time consuming!)
  pixelError = new PixelErrorParametrization(pset_);

  // Initialize the histos once and for all, and prepare the random generation
  for ( unsigned alphaHistBin=1; alphaHistBin<=resAlpha_binN; ++alphaHistBin ) {
    unsigned int maxSize;
    if(useCMSSWPixelParameterization)
      maxSize = theAlphaMultiplicityCumulativeProbabilities.size() / 2;
    else
      maxSize = theAlphaMultiplicityCumulativeProbabilities.size();
    for ( unsigned alphaMultiplicity=1; 
	  alphaMultiplicity<=maxSize;
	  ++alphaMultiplicity ) {
      unsigned int alphaHistN = (resAlpha_binWidth != 0. ?
				 100 * alphaHistBin
				 + 10
				 + alphaMultiplicity
				 :
				 1110
				 + alphaMultiplicity);    //
      theAlphaHistos[alphaHistN] = new SimpleHistogramGenerator(
        (TH1F*) thePixelResolutionFile->Get(  Form( "h%u" , alphaHistN ) ),
        random);
      // Fill also big pixels if new parametrization is used. Their code is 10000 + histogram number
      if(useCMSSWPixelParameterization) {
          theAlphaHistos[alphaHistN+10000] = new SimpleHistogramGenerator(
	    (TH1F*) thePixelResolutionFile->Get(  Form( "h%ub" , alphaHistN ) ),
	    random);
      }
    }
  }


  //
  for ( unsigned betaHistBin=1; betaHistBin<=resBeta_binN; ++betaHistBin ) {
    unsigned int maxSize;
    if(useCMSSWPixelParameterization)
      maxSize = theBetaMultiplicityCumulativeProbabilities.size() / 2;
    else
      maxSize = theBetaMultiplicityCumulativeProbabilities.size();
    for ( unsigned betaMultiplicity=1; 
	  betaMultiplicity<=maxSize;
	  ++betaMultiplicity ) {
      unsigned int betaHistN = (resBeta_binWidth != 0. ?
				100 * betaHistBin
				+ betaMultiplicity
				:
				1100 + betaMultiplicity);    //
      theBetaHistos[betaHistN] = new SimpleHistogramGenerator(
        (TH1F*) thePixelResolutionFile->Get(  Form( "h%u" , betaHistN  ) ),
	random);
      // Fill also big pixels if new parametrization is used. Their code is 10000 + histogram number
      if(useCMSSWPixelParameterization) {
	theBetaHistos[betaHistN+10000] = new SimpleHistogramGenerator(
          (TH1F*) thePixelResolutionFile->Get(  Form( "h%ub" , betaHistN  ) ),
	  random);
      }
    }
  }
}

SiPixelGaussianSmearingRecHitConverterAlgorithm::~SiPixelGaussianSmearingRecHitConverterAlgorithm()
{
  
  // Some cleaning
  delete pixelError;

  std::map<unsigned,const SimpleHistogramGenerator*>::const_iterator it;
  for ( it=theAlphaHistos.begin(); it!=theAlphaHistos.end(); ++it ) 
    delete it->second;

  for ( it=theBetaHistos.begin(); it!=theBetaHistos.end(); ++it ) 
    delete it->second;

  theAlphaHistos.clear();
  theBetaHistos.clear();

}

void SiPixelGaussianSmearingRecHitConverterAlgorithm::smearHit(
  const PSimHit& simHit, 
  const PixelGeomDetUnit* detUnit,
  const double boundX,
  const double boundY)
{

#ifdef FAMOS_DEBUG
  std::cout << " Pixel smearing in " << thePixelPart 
	    << std::endl;
#endif
  //
  // at the beginning the position is the Local Point in the local pixel module reference frame
  // same code as in PixelCPEBase
  LocalVector localDir = simHit.momentumAtEntry().unit();
  float locx = localDir.x();
  float locy = localDir.y();
  float locz = localDir.z();

  //
  bool hasBigPixelInX = false;
  bool hasBigPixelInY = false;

  if(useCMSSWPixelParameterization) {
    // If the sim track crosses a region in which there are big pixels,
    // then we set to true the variables above

    // Get the topology of the pixel module
    const PixelTopology* theSpecificTopology = &(detUnit->specificTopology());
    RectangularPixelTopology rectPixelTopology(theSpecificTopology->nrows(), 
                                               theSpecificTopology->ncolumns(), 
                                               theSpecificTopology->pitch().first, 
                                               theSpecificTopology->pitch().second);
    
    // Get the rows and columns of entry and exit points
    // FIXME - these are not guaranteed to be the same as the cluster limits (as they should be)
    const int firstPixelInX = int(rectPixelTopology.pixel(simHit.entryPoint()).first);
    const int firstPixelInY = int(rectPixelTopology.pixel(simHit.entryPoint()).second);
    const int lastPixelInX = int(rectPixelTopology.pixel(simHit.exitPoint()).first);
    const int lastPixelInY = int(rectPixelTopology.pixel(simHit.exitPoint()).second);
    
    // Check if there is a big pixel inside and set hasBigPixelInX and hasBigPixelInY accordingly
    // This function only works if first <= last
    if(rectPixelTopology.containsBigPixelInX(firstPixelInX < lastPixelInX ? firstPixelInX : lastPixelInX,
                                             firstPixelInX > lastPixelInX ? firstPixelInX : lastPixelInX))
      hasBigPixelInX = true;
    if(rectPixelTopology.containsBigPixelInY(firstPixelInY < lastPixelInY ? firstPixelInY : lastPixelInY,
                                             firstPixelInY > lastPixelInY ? firstPixelInY : lastPixelInY))
      hasBigPixelInY = true;
#ifdef FAMOS_DEBUG
    std::cout << " Simhit first pixel in X = " << (firstPixelInX < lastPixelInX ? firstPixelInX : lastPixelInX)
              << " last pixel in X = " << (firstPixelInX > lastPixelInX ? firstPixelInX : lastPixelInX)
              << " has big pixel in X is " << (hasBigPixelInX ? " true" : " false")
              << std::endl;
    std::cout << " Simhit first pixel in Y = " << (firstPixelInY < lastPixelInY ? firstPixelInY : lastPixelInY)
              << " last pixel in Y = " << (firstPixelInY > lastPixelInY ? firstPixelInY : lastPixelInY)
              << " has big pixel in Y is " << (hasBigPixelInY ? " true" : " false")
              << std::endl;
#endif
  }

  // alpha: angle with respect to local x axis in local (x,z) plane
  float alpha = std::acos(locx/std::sqrt(locx*locx+locz*locz));
  if ( isFlipped( detUnit ) ) { // &&& check for FPIX !!!
#ifdef FAMOS_DEBUG
    std::cout << " isFlipped " << std::endl;
#endif
    alpha = PI - alpha ;
  }
  // beta: angle with respect to local y axis in local (y,z) plane
  float beta = std::acos(locy/std::sqrt(locy*locy+locz*locz));
  
  // look old FAMOS: FamosGeneric/FamosTracker/src/FamosPixelErrorParametrization
  float alphaToBeUsedForRootFiles = alpha;
  float betaToBeUsedForRootFiles  = beta;
  if( thePixelPart == GeomDetEnumerators::PixelBarrel ) { // BARREL
    alphaToBeUsedForRootFiles = PI/2. - alpha;
    betaToBeUsedForRootFiles  = fabs( PI/2. - beta );
  } else { // FORWARD
    betaToBeUsedForRootFiles = fabs( PI/2. - beta );
    alphaToBeUsedForRootFiles  = fabs( PI/2. - alpha );    
  }
  //
#ifdef FAMOS_DEBUG
  std::cout << " Local Direction " << simHit.localDirection()
	    << " alpha(x) = " << alpha
	    << " beta(y) = "  << beta
	    << " alpha for root files = " << alphaToBeUsedForRootFiles
	    << " beta for root files = "  << betaToBeUsedForRootFiles
	    << std::endl;
#endif

  // Generate alpha and beta multiplicity
  unsigned int alphaMultiplicity = 0;
  unsigned int betaMultiplicity  = 0;
  // random multiplicity for alpha and beta
  double alphaProbability = random->flatShoot();
  double betaProbability  = random->flatShoot();

  // search which multiplicity correspond
  int alphaBin = 
    theAlphaMultiplicityCumulativeProbabilities.front()->GetXaxis()->FindFixBin(alphaToBeUsedForRootFiles);
  int betaBin = 
    theBetaMultiplicityCumulativeProbabilities.front()->GetXaxis()->FindFixBin(betaToBeUsedForRootFiles);

  // protection against out-of-range (undeflows and overflows)
  if( alphaBin == 0 ) alphaBin = 1;
  if( alphaBin > theAlphaMultiplicityCumulativeProbabilities.front()->GetNbinsX() ) 
    alphaBin = theAlphaMultiplicityCumulativeProbabilities.front()->GetNbinsX();
  if( betaBin == 0 ) betaBin = 1;
  if( betaBin > theBetaMultiplicityCumulativeProbabilities.front()->GetNbinsX() )   
    betaBin = theBetaMultiplicityCumulativeProbabilities.front()->GetNbinsX();
  
  unsigned int iMult, multSize;
  const unsigned int maxMultX = theAlphaMultiplicityCumulativeProbabilities.size();
  const unsigned int maxMultY = theBetaMultiplicityCumulativeProbabilities.size();
  if(useCMSSWPixelParameterization) {
    if(hasBigPixelInX) {     // Big pixels: second half of histograms vector
      iMult = maxMultX / 2;
      multSize = maxMultX;
    } else {                 // Normal pixels: first half of histograms vector
      iMult = 0;
      multSize = maxMultX / 2;
    }
  } else {
    iMult = 0;
    multSize = maxMultX;
  }
  for(/* void */; iMult < multSize; iMult++) {
    if(alphaProbability < theAlphaMultiplicityCumulativeProbabilities[iMult]->GetBinContent(alphaBin) ) {
      alphaMultiplicity = iMult+1;
      break;
    }
  }
  
  if(useCMSSWPixelParameterization) {
    if(hasBigPixelInY) {     // Big pixels: second half of histograms vector
      iMult = maxMultY / 2;
      multSize = maxMultY;
    } else {                 // Normal pixels: first half of histograms vector
      iMult = 0;
      multSize = maxMultY / 2;
    }
  } else {
    iMult = 0;
    multSize = maxMultY;
  }
  for(/* void */; iMult < multSize; iMult++) {
    if(betaProbability < theBetaMultiplicityCumulativeProbabilities[iMult]->GetBinContent(betaBin) ) {
      betaMultiplicity = iMult+1;
      break;
    }
  }

  // Correct multiplicity for big pixels
  if(hasBigPixelInX)
    alphaMultiplicity -= maxMultX / 2;
  if(hasBigPixelInY)
    betaMultiplicity -= maxMultY / 2;

  // protection against 0 or max multiplicity
  if(useCMSSWPixelParameterization) {
    if( alphaMultiplicity == 0 || alphaMultiplicity > maxMultX / 2 ) 
      alphaMultiplicity = maxMultX / 2;
    if( betaMultiplicity  == 0 || betaMultiplicity  > maxMultY / 2  ) 
      betaMultiplicity  = maxMultY / 2;
  } else {
    if( alphaMultiplicity == 0 || alphaMultiplicity > maxMultX ) 
      alphaMultiplicity = maxMultX;
    if( betaMultiplicity  == 0 || betaMultiplicity  > maxMultY ) 
      betaMultiplicity  = maxMultY;
  }
  //
  
//
#ifdef FAMOS_DEBUG
  std::cout << " Multiplicity set to"
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
#endif
  
  // Compute pixel errors
  std::pair<float,float> theErrors = pixelError->getError( thePixelPart ,
							  (int)alphaMultiplicity , (int)betaMultiplicity ,
                                                           alpha                 , beta                  ,
                                                           hasBigPixelInX        , hasBigPixelInY         );
  // define private mebers --> Errors
  theErrorX = theErrors.first;  // PixelErrorParametrization returns sigma, not sigma^2
  theErrorY = theErrors.second; // PixelErrorParametrization returns sigma, not sigma^2
  theErrorZ = 1e-8; // 1 um means zero
  theError = LocalError( theErrorX*theErrorX, 0., theErrorY*theErrorY);
  // Local Error is 2D: (xx,xy,yy), square of sigma in first an third position 
  // as for resolution matrix
  //
#ifdef FAMOS_DEBUG
  std::cout << " Pixel Errors "
	    << "\talpha(x) = " << theErrorX
	    << "\tbeta(y) = "  << theErrorY
	    << std::endl;	
#endif
  
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
  unsigned int alphaHistN = (resAlpha_binWidth != 0. ?
			     100 * alphaHistBin
			     + 10
			     + alphaMultiplicity
			     :
			     1110
			     + alphaMultiplicity);    //
  //
  unsigned int betaHistN = (resBeta_binWidth != 0. ?
			    100 * betaHistBin
			    + betaMultiplicity
			    :
			    1100 + betaMultiplicity);    //
  //
  if(hasBigPixelInX)
    alphaHistN += 10000;
  if(hasBigPixelInY)
    betaHistN += 10000;
#ifdef FAMOS_DEBUG
  std::cout << " Resolution histograms chosen "
	    << "\talpha = " << alphaHistN
	    << "\tbeta = "  << betaHistN
	    << std::endl;	
#endif

  unsigned int counter = 0;
  do {
    //
    // Smear the hit Position
    thePositionX = theAlphaHistos[alphaHistN]->generate();
    thePositionY = theBetaHistos[betaHistN]->generate();
    thePositionZ = 0.0; // set at the centre of the active area
    thePosition = 
      Local3DPoint(simHit.localPosition().x() + thePositionX , 
                   simHit.localPosition().y() + thePositionY , 
                   simHit.localPosition().z() + thePositionZ );
#ifdef FAMOS_DEBUG
    std::cout << " Detector bounds: "
              << "\t\tx = " << boundX
              << "\ty = " << boundY
              << std::endl;
    std::cout << " Generated local position "
              << "\tx = " << thePosition.x()
              << "\ty = " << thePosition.y()
              << std::endl;       
#endif  
    counter++;
    if(counter > 20) {
      thePosition = Local3DPoint(simHit.localPosition().x(), 
                                 simHit.localPosition().y(), 
                                 simHit.localPosition().z());
      break;
    }
  } while(fabs(thePosition.x()) > boundX  || fabs(thePosition.y()) > boundY);
  

  
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
// better way.(PJ: And faster!)
//-----------------------------------------------------------------------------
bool SiPixelGaussianSmearingRecHitConverterAlgorithm::isFlipped(const PixelGeomDetUnit* theDet) const {
  // Check the relative position of the local +/- z in global coordinates.
  float tmp1 = theDet->surface().toGlobal(Local3DPoint(0.,0.,0.)).perp();
  float tmp2 = theDet->surface().toGlobal(Local3DPoint(0.,0.,1.)).perp();
  //  std::cout << " 1: " << tmp1 << " 2: " << tmp2 << std::endl;
  if ( tmp2<tmp1 ) return true;
  else return false;    
}
 
