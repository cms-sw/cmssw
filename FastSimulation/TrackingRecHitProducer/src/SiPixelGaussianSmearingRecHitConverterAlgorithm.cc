/** SiPixelGaussianSmearingRecHitConverterAlgorithm.cc
 * ---------------------------------------------------------------------
 * Description:  see SiPixelGaussianSmearingRecHitConverterAlgorithm.h
 * Authors:  R. Ranieri (CERN), M. Galanti
 * History: Oct 11, 2006 -  initial version
 * 
 * New Pixel Resolution Parameterization 
 * Introduce SiPixelTemplate Object to Assign Pixel Errors
 * by G. Hu
 * ---------------------------------------------------------------------
 */

// SiPixel Gaussian Smearing
#include "SiPixelGaussianSmearingRecHitConverterAlgorithm.h"

// Geometry
//#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/CommonDetUnit/interface/GeomDetUnit.h"
//#include "Geometry/CommonDetUnit/interface/GeomDetType.h"
#include "Geometry/TrackerGeometryBuilder/interface/RectangularPixelTopology.h"
#include "DataFormats/GeometryVector/interface/LocalPoint.h"
#include "DataFormats/GeometryCommonDetAlgo/interface/MeasurementPoint.h"

// Famos
#include "FastSimulation/Utilities/interface/RandomEngineAndDistribution.h"
#include "FastSimulation/Utilities/interface/SimpleHistogramGenerator.h"

// STL

// ROOT
#include <TFile.h>
#include <TH1F.h>
//#include <TAxis.h>

//#define FAMOS_DEBUG

const double PI = 3.14159265358979323;
const double microntocm = 0.0001;
using namespace std;

SiPixelGaussianSmearingRecHitConverterAlgorithm::SiPixelGaussianSmearingRecHitConverterAlgorithm(
  const edm::ParameterSet& pset,
  GeomDetType::SubDetector pixelPart)
:
  pset_(pset),
  thePixelPart(pixelPart)
{
  // Switch between old (ORCA) and new (CMSSW) pixel parameterization
  useCMSSWPixelParameterization = pset.getParameter<bool>("UseCMSSWPixelParametrization");

  if( thePixelPart == GeomDetEnumerators::PixelBarrel ) {
     isForward = false;
     thePixelResolutionFileName1 = pset_.getParameter<string>( "NewPixelBarrelResolutionFile1" );
     thePixelResolutionFile1 = new 
       TFile( edm::FileInPath( thePixelResolutionFileName1 ).fullPath().c_str()  ,"READ");
     thePixelResolutionFileName2 =  pset_.getParameter<string>( "NewPixelBarrelResolutionFile2" );
     thePixelResolutionFile2 = new
       TFile( edm::FileInPath( thePixelResolutionFileName2 ).fullPath().c_str()  ,"READ");
     initializeBarrel();
     tempId = pset_.getParameter<int> ( "templateIdBarrel" );
     if( ! SiPixelTemplate::pushfile(tempId, thePixelTemp_) )
         throw cms::Exception("SiPixelGaussianSmearingRecHitConverterAlgorithm:")
	 	<<"SiPixel Barrel Template Not Loaded Correctly!"<<endl;
#ifdef  FAMOS_DEBUG
     cout<<"The Barrel map size is "<<theXHistos.size()<<" and "<<theYHistos.size()<<endl;
#endif
  }
  else
  if ( thePixelPart == GeomDetEnumerators::PixelEndcap ) {
     isForward = true;
     thePixelResolutionFileName1 = pset_.getParameter<string>( "NewPixelForwardResolutionFile" );
     thePixelResolutionFile1 = new
       TFile( edm::FileInPath( thePixelResolutionFileName1 ).fullPath().c_str()  ,"READ");
     initializeForward();
     tempId = pset_.getParameter<int> ( "templateIdForward" );
     if( ! SiPixelTemplate::pushfile(tempId, thePixelTemp_) )
         throw cms::Exception("SiPixelGaussianSmearingRecHitConverterAlgorithm:")
	        <<"SiPixel Forward Template Not Loaded Correctly!"<<endl;
#ifdef  FAMOS_DEBUG
     cout<<"The Forward map size is "<<theXHistos.size()<<" and "<<theYHistos.size()<<endl;
#endif
  }
  else
     throw cms::Exception("SiPixelGaussianSmearingRecHitConverterAlgorithm :")
       <<"Not a pixel detector"<<endl;
}

SiPixelGaussianSmearingRecHitConverterAlgorithm::~SiPixelGaussianSmearingRecHitConverterAlgorithm()
{
  
  std::map<unsigned,const SimpleHistogramGenerator*>::const_iterator it;
  for ( it=theXHistos.begin(); it!=theXHistos.end(); ++it )
    delete it->second;
  for ( it=theYHistos.begin(); it!=theYHistos.end(); ++it )
    delete it->second;

  theXHistos.clear();
  theYHistos.clear();

}

void SiPixelGaussianSmearingRecHitConverterAlgorithm::smearHit(
  const PSimHit& simHit,
  const PixelGeomDetUnit* detUnit,
  const double boundX,
  const double boundY,
  RandomEngineAndDistribution const* random)
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

  // alpha: angle with respect to local x axis in local (x,z) plane
  float cotalpha = locx/locz;
  if ( isFlipped( detUnit ) ) { // &&& check for FPIX !!!
#ifdef FAMOS_DEBUG
    std::cout << " isFlipped " << std::endl;
#endif
  }
  // beta: angle with respect to local y axis in local (y,z) plane
  float cotbeta = locy/locz;
  float sign=1.;
  if( isForward ) {
    if( cotbeta < 0 ) sign=-1.;
    cotbeta = sign*cotbeta;
  }
  
  
  //
#ifdef FAMOS_DEBUG
  std::cout << " Local Direction " << simHit.localDirection()
	    << " cotalpha(x) = " << cotalpha
	    << " cotbeta(y) = "  << cotbeta
	    << std::endl;
#endif
  
  const PixelTopology* theSpecificTopology = &(detUnit->specificType().specificTopology());
  const RectangularPixelTopology *rectPixelTopology = static_cast<const RectangularPixelTopology*>(theSpecificTopology);

  const int nrows = theSpecificTopology->nrows();
  const int ncolumns = theSpecificTopology->ncolumns();

  const Local3DPoint lp = simHit.localPosition();
  //Transform local position to measurement position
  const MeasurementPoint mp = rectPixelTopology->measurementPosition( lp );
  float mpy = mp.y();
  float mpx = mp.x();
  //Get the center of the struck pixel in measurement position
  float pixelCenterY = 0.5 + (int)mpy;
  float pixelCenterX = 0.5 + (int)mpx;
#ifdef FAMOS_DEBUG
  cout<<"Struck pixel center at pitch units x: "<<pixelCenterX<<" y: "<<pixelCenterY<<endl;
#endif

  const MeasurementPoint mpCenter(pixelCenterX, pixelCenterY);
  //Transform the center of the struck pixel back into local position
  const Local3DPoint lpCenter = rectPixelTopology->localPosition( mpCenter );
#ifdef FAMOS_DEBUG
  cout<<"Struck point at cm x: "<<lp.x()<<" y: "<<lp.y()<<endl;
  cout<<"Struck pixel center at cm x: "<<lpCenter.x()<<" y: "<<lpCenter.y()<<endl;
  cout<<"The boundX is "<<boundX<<" boundY is "<<boundY<<endl;
#endif

  //Get the relative position of struck point to the center of the struck pixel
  float xtrk = lp.x() - lpCenter.x();
  float ytrk = lp.y() - lpCenter.y();
  //Pixel Y, X pitch
  const float ysize={0.015}, xsize={0.01};
  //Variables for SiPixelTemplate input, see SiPixelTemplate reco
  float yhit = 20. + 8.*(ytrk/ysize);
  float xhit = 20. + 8.*(xtrk/xsize);
  int   ybin = (int)yhit;
  int	xbin = (int)xhit;
  float yfrac= yhit - (float)ybin;
  float xfrac= xhit - (float)xbin;
  //Protect againt ybin, xbin being outside of range [0-39]
  if( ybin < 0 )    ybin = 0;
  if( ybin > 39 )   ybin = 39;
  if( xbin < 0 )    xbin = 0;
  if( xbin > 39 )   xbin = 39; 

  //Variables for SiPixelTemplate output
  //qBin -- normalized pixel charge deposition
  float qbin_frac[4];
  //Single pixel cluster projection possibility
  float ny1_frac, ny2_frac, nx1_frac, nx2_frac;
  bool singlex = false, singley = false;
  SiPixelTemplate templ(thePixelTemp_);
  templ.interpolate(tempId, cotalpha, cotbeta);
  templ.qbin_dist(tempId, cotalpha, cotbeta, qbin_frac, ny1_frac, ny2_frac, nx1_frac, nx2_frac );
  int  nqbin;

  double xsizeProbability = random->flatShoot();
  double ysizeProbability = random->flatShoot();
  bool hitbigx = rectPixelTopology->isItBigPixelInX( (int)mpx );
  bool hitbigy = rectPixelTopology->isItBigPixelInY( (int)mpy );
  
  if( hitbigx ) 
    if( xsizeProbability < nx2_frac )  singlex = true;
    else singlex = false;
  else
    if( xsizeProbability < nx1_frac )  singlex = true;
    else singlex = false;

  if( hitbigy )
    if( ysizeProbability < ny2_frac )  singley = true;
    else singley = false;
  else
    if( ysizeProbability < ny1_frac )  singley = true;
    else singley = false;
  


  // random multiplicity for alpha and beta
  double qbinProbability = random->flatShoot();
  for(int i = 0; i<4; ++i) {
     nqbin = i;
     if(qbinProbability < qbin_frac[i]) break;
  }

  //Store interpolated pixel cluster profile
  //BYSIZE, BXSIZE, const definition from SiPixelTemplate
  float ytempl[41][BYSIZE] = {{0}}, xtempl[41][BXSIZE] = {{0}} ;
  templ.ytemp(0, 40, ytempl);
  templ.xtemp(0, 40, xtempl);

  std::vector<double> ytemp(BYSIZE);
  for( int i=0; i<BYSIZE; ++i) {
     ytemp[i]=(1.-yfrac)*ytempl[ybin][i]+yfrac*ytempl[ybin+1][i];
  }

  std::vector<double> xtemp(BXSIZE);
  for(int i=0; i<BXSIZE; ++i) {
     xtemp[i]=(1.-xfrac)*xtempl[xbin][i]+xfrac*xtempl[xbin+1][i];
  }

  //Pixel readout threshold
  const float qThreshold = templ.s50()*2.0;

  //Cut away pixels below readout threshold
  //For cluster lengths calculation
  int offsetX1=0, offsetX2=0, offsetY1=0, offsetY2=0;
  int firstY, lastY, firstX, lastX;
  for( firstY = 0; firstY < BYSIZE; ++firstY ) {
    bool yCluster = ytemp[firstY] > qThreshold ;
    if( yCluster )
    {
      offsetY1 = BHY -firstY;
      break;
    }
  }
  for( lastY = firstY; lastY < BYSIZE; ++lastY )
  {
    bool yCluster = ytemp[lastY] > qThreshold ;
    if( !yCluster )
    {
      lastY = lastY - 1;
      offsetY2 = lastY - BHY;
      break;
    }
  }

  for( firstX = 0; firstX < BXSIZE; ++firstX )  {
    bool xCluster = xtemp[firstX] > qThreshold ;
    if( xCluster ) {
      offsetX1 = BHX - firstX;
      break;
    }
  }
  for( lastX = firstX; lastX < BXSIZE; ++ lastX ) {
    bool xCluster = xtemp[lastX] > qThreshold ;
    if( !xCluster ) {
      lastX = lastX - 1;
      offsetX2 = lastX - BHX;
      break;
    }
  }

  bool edge, edgex, edgey;
  //  bool bigx, bigy;
  unsigned int clslenx = offsetX1 + offsetX2 + 1;
  unsigned int clsleny = offsetY1 + offsetY2 + 1;

  theClslenx = clslenx;
  theClsleny = clsleny;

  int firstPixelInX = (int)mpx - offsetX1 ;
  int firstPixelInY = (int)mpy - offsetY1 ;
  int lastPixelInX  = (int)mpx + offsetX2 ;
  int lastPixelInY  = (int)mpy + offsetY2 ;
  firstPixelInX = (firstPixelInX >= 0) ? firstPixelInX : 0 ;
  firstPixelInY = (firstPixelInY >= 0) ? firstPixelInY : 0 ;
  lastPixelInX  = (lastPixelInX < nrows ) ? lastPixelInX : nrows-1 ;
  lastPixelInY  = (lastPixelInY < ncolumns ) ? lastPixelInY : ncolumns-1;

  edgex = rectPixelTopology->isItEdgePixelInX( firstPixelInX ) || rectPixelTopology->isItEdgePixelInX( lastPixelInX );
  edgey = rectPixelTopology->isItEdgePixelInY( firstPixelInY ) || rectPixelTopology->isItEdgePixelInY( lastPixelInY );
  edge = edgex || edgey;

  //  bigx = rectPixelTopology->isItBigPixelInX( firstPixelInX ) || rectPixelTopology->isItBigPixelInX( lastPixelInX );
  //  bigy = rectPixelTopology->isItBigPixelInY( firstPixelInY ) || rectPixelTopology->isItBigPixelInY( lastPixelInY );
  bool hasBigPixelInX = rectPixelTopology->containsBigPixelInX( firstPixelInX, lastPixelInX );
  bool hasBigPixelInY = rectPixelTopology->containsBigPixelInY( firstPixelInY, lastPixelInY );

  //Variables for SiPixelTemplate pixel hit error output
  float sigmay, sigmax, sy1, sy2, sx1, sx2;
  templ.temperrors(tempId, cotalpha, cotbeta, nqbin, sigmay, sigmax, sy1, sy2, sx1, sx2 );

  // define private mebers --> Errors
  if( edge ) {
    if( edgex && !edgey ) {
      theErrorX = 23.0*microntocm;
      theErrorY = 39.0*microntocm;
    }
    else if( !edgex && edgey ) {
      theErrorX = 24.0*microntocm;
      theErrorY = 96.0*microntocm;
    }
    else
    {
      theErrorX = 31.0*microntocm;
      theErrorY = 90.0*microntocm;
    }
    
  }  
  else {
    if( singlex )
      if ( hitbigx )
        theErrorX = sx2*microntocm;
      else
        theErrorX = sx1*microntocm;
    else  theErrorX = sigmax*microntocm;
    if( singley )
      if( hitbigy )
        theErrorY = sy2*microntocm;
      else
        theErrorY = sy1*microntocm;
    else  theErrorY = sigmay*microntocm;
  }
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
  // Generate position
  // get resolution histograms
  int cotalphaHistBin = (int)( ( cotalpha - rescotAlpha_binMin ) / rescotAlpha_binWidth + 1 );
  int cotbetaHistBin  = (int)( ( cotbeta  - rescotBeta_binMin )  / rescotBeta_binWidth + 1 );
  // protection against out-of-range (undeflows and overflows)
  if( cotalphaHistBin < 1 ) cotalphaHistBin = 1; 
  if( cotbetaHistBin  < 1 ) cotbetaHistBin  = 1; 
  if( cotalphaHistBin > (int)rescotAlpha_binN ) cotalphaHistBin = (int)rescotAlpha_binN; 
  if( cotbetaHistBin  > (int)rescotBeta_binN  ) cotbetaHistBin  = (int)rescotBeta_binN; 
  //
  unsigned int theXHistN;
  unsigned int theYHistN;
  if( !isForward ) {
     if(edge)
     {
       theXHistN = cotalphaHistBin * 1000 + cotbetaHistBin * 10	 +  (nqbin+1);
       theYHistN = theXHistN;	      
     }
     else
     {
       if(singlex)
       {
	 if(hitbigx)  theXHistN = 1 * 100000 + cotalphaHistBin * 100 + cotbetaHistBin ;
	 else 	theXHistN = 1 * 100000 + 1 * 1000 + cotalphaHistBin * 100 + cotbetaHistBin ;
       }
       else
       {
	 if(hasBigPixelInX)  theXHistN = 1 * 1000000 + 1 * 100000 + cotalphaHistBin * 1000 + cotbetaHistBin * 10 + (nqbin+1);
	 else 	theXHistN = 1 * 1000000 + 1 * 100000 + 1 * 10000 + cotalphaHistBin * 1000 + cotbetaHistBin * 10 + (nqbin+1);
       }
       if(singley)
       {
	 if(hitbigy)  theYHistN = 1 * 100000 + cotalphaHistBin * 100 + cotbetaHistBin ;
	 else 	theYHistN = 1 * 100000 + 1 * 1000 + cotalphaHistBin * 100 + cotbetaHistBin ;
       }
       else
       {
	 if(hasBigPixelInY)  theYHistN = 1 * 1000000 + 1 * 100000 + cotalphaHistBin * 1000 + cotbetaHistBin * 10 + (nqbin+1);
	 else 	theYHistN = 1 * 1000000 + 1 * 100000 + 1 * 10000 + cotalphaHistBin * 1000 + cotbetaHistBin * 10 + (nqbin+1);
       }
     }
  }
  else
  {
     if(edge)
     {
       theXHistN = cotalphaHistBin * 1000 +  cotbetaHistBin * 10 +  (nqbin+1);
       theYHistN = theXHistN;
     }
     else
     {
       if( singlex )
         if( hitbigx )
	   theXHistN = 100000 + cotalphaHistBin * 100 + cotbetaHistBin;
	 else
	   theXHistN = 100000 + 1000 + cotalphaHistBin * 100 + cotbetaHistBin;
       else
         theXHistN = 100000 + 10000 + cotalphaHistBin * 1000 +  cotbetaHistBin * 10 +  (nqbin+1);
       if( singley )
         if( hitbigy )
	   theYHistN = 100000 + cotalphaHistBin * 100 + cotbetaHistBin;
	 else
	   theYHistN = 100000 + 1000 + cotalphaHistBin * 100 + cotbetaHistBin;
       else
         theYHistN = 100000 + 10000 + cotalphaHistBin * 1000 +  cotbetaHistBin * 10 +  (nqbin+1);
     }
  }
  unsigned int counter = 0;
  do {
    //
    // Smear the hit Position
    thePositionX = theXHistos[theXHistN]->generate(random);
    thePositionY = theYHistos[theYHistN]->generate(random);
    if( isForward ) thePositionY *= sign;
    thePositionZ = 0.0; // set at the centre of the active area
    //protect from empty resolution histograms
    //if( thePositionX > 0.0799 )  thePositionX = 0;
    //if( thePositionY > 0.0799 )  thePositionY = 0;
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

void SiPixelGaussianSmearingRecHitConverterAlgorithm::initializeBarrel()
{
  //Hard coded at the moment, can easily be changed to be configurable
  rescotAlpha_binMin = -0.2;
  rescotAlpha_binWidth = 0.08 ;
  rescotAlpha_binN = 5;
  rescotBeta_binMin = -5.5;
  rescotBeta_binWidth = 1.0;
  rescotBeta_binN = 11;
  resqbin_binMin = 0;
  resqbin_binWidth = 1;
  resqbin_binN = 4;

  // Initialize the barrel histos once and for all, and prepare the random generation
  for ( unsigned cotalphaHistBin=1; cotalphaHistBin<=rescotAlpha_binN; ++cotalphaHistBin )
     for ( unsigned cotbetaHistBin=1; cotbetaHistBin<=rescotBeta_binN; ++cotbetaHistBin )  {
         unsigned int singleBigPixelHistN = 1 * 100000
                                        + cotalphaHistBin * 100
                                        + cotbetaHistBin ;
         theXHistos[singleBigPixelHistN] = new SimpleHistogramGenerator(
              (TH1F*) thePixelResolutionFile1->Get(  Form( "DQMData/clustBPIX/hx%u" , singleBigPixelHistN ) ) );
         theYHistos[singleBigPixelHistN] = new SimpleHistogramGenerator(
              (TH1F*) thePixelResolutionFile1->Get(  Form( "DQMData/clustBPIX/hy%u" , singleBigPixelHistN ) ) );
         unsigned int singlePixelHistN = 1 * 100000 + 1 * 1000
                                        + cotalphaHistBin * 100
                                        + cotbetaHistBin ;
         theXHistos[singlePixelHistN] = new SimpleHistogramGenerator(
              (TH1F*) thePixelResolutionFile1->Get(  Form( "DQMData/clustBPIX/hx%u" , singlePixelHistN ) ) );
         theYHistos[singlePixelHistN] = new SimpleHistogramGenerator(
              (TH1F*) thePixelResolutionFile1->Get(  Form( "DQMData/clustBPIX/hy%u" , singlePixelHistN ) ) );
         for( unsigned qbinBin=1;  qbinBin<=resqbin_binN; ++qbinBin )  {
             unsigned int edgePixelHistN = cotalphaHistBin * 1000
                                        +  cotbetaHistBin * 10
                                        +  qbinBin;
             theXHistos[edgePixelHistN] = new SimpleHistogramGenerator(
              (TH1F*) thePixelResolutionFile2->Get(  Form( "DQMData/clustBPIX/hx0%u" ,edgePixelHistN ) ) );
             theYHistos[edgePixelHistN] = new SimpleHistogramGenerator(
              (TH1F*) thePixelResolutionFile2->Get(  Form( "DQMData/clustBPIX/hy0%u" ,edgePixelHistN ) ) );
             unsigned int multiPixelBigHistN = 1 * 1000000 + 1 * 100000
                                           + cotalphaHistBin * 1000
                                           + cotbetaHistBin * 10
                                           + qbinBin;
             theXHistos[multiPixelBigHistN] = new SimpleHistogramGenerator(
              (TH1F*) thePixelResolutionFile1->Get(  Form( "DQMData/clustBPIX/hx%u" ,multiPixelBigHistN ) ) );
             theYHistos[multiPixelBigHistN] = new SimpleHistogramGenerator(
              (TH1F*) thePixelResolutionFile1->Get(  Form( "DQMData/clustBPIX/hy%u" ,multiPixelBigHistN ) ) );
             unsigned int multiPixelHistN = 1 * 1000000 + 1 * 100000 + 1 * 10000
                                           + cotalphaHistBin * 1000
                                           + cotbetaHistBin * 10
                                           + qbinBin;
             theXHistos[multiPixelHistN] = new SimpleHistogramGenerator(
             (TH1F*) thePixelResolutionFile1->Get(  Form( "DQMData/clustBPIX/hx%u" , multiPixelHistN ) ) );
             theYHistos[multiPixelHistN] = new SimpleHistogramGenerator(
             (TH1F*) thePixelResolutionFile1->Get(  Form( "DQMData/clustBPIX/hy%u" , multiPixelHistN ) ) );
          } //end for qbinBin
     }//end for cotalphaHistBin, cotbetaHistBin
}

void SiPixelGaussianSmearingRecHitConverterAlgorithm::initializeForward()
{
  //Hard coded at the moment, can easily be changed to be configurable
  rescotAlpha_binMin = 0.1;
  rescotAlpha_binWidth = 0.1 ;
  rescotAlpha_binN = 4;
  rescotBeta_binMin = 0.;
  rescotBeta_binWidth = 0.15;
  rescotBeta_binN = 4;
  resqbin_binMin = 0;
  resqbin_binWidth = 1;
  resqbin_binN = 4;

  // Initialize the forward histos once and for all, and prepare the random generation
  for ( unsigned cotalphaHistBin=1; cotalphaHistBin<=rescotAlpha_binN; ++cotalphaHistBin )
     for ( unsigned cotbetaHistBin=1; cotbetaHistBin<=rescotBeta_binN; ++cotbetaHistBin )  
         for( unsigned qbinBin=1;  qbinBin<=resqbin_binN; ++qbinBin )  {
	    unsigned int edgePixelHistN = cotalphaHistBin * 1000 +  cotbetaHistBin * 10 +  qbinBin;
	    theXHistos[edgePixelHistN] = new SimpleHistogramGenerator(
	    (TH1F*) thePixelResolutionFile1->Get(  Form( "DQMData/clustFPIX/fhx0%u" ,edgePixelHistN ) ) );
	    theYHistos[edgePixelHistN] = new SimpleHistogramGenerator(
	    (TH1F*) thePixelResolutionFile1->Get(  Form( "DQMData/clustFPIX/fhy0%u" ,edgePixelHistN ) ) );
	    unsigned int PixelHistN = 100000 + 10000 + cotalphaHistBin * 1000 +  cotbetaHistBin * 10 +  qbinBin;
	    theXHistos[PixelHistN] = new SimpleHistogramGenerator(
	    (TH1F*) thePixelResolutionFile1->Get(  Form( "DQMData/clustFPIX/fhx%u" ,PixelHistN ) ) );
	    theYHistos[PixelHistN] = new SimpleHistogramGenerator(
	    (TH1F*) thePixelResolutionFile1->Get(  Form( "DQMData/clustFPIX/fhy%u" ,PixelHistN ) ) );
	 }//end cotalphaHistBin, cotbetaHistBin, qbinBin

  for ( unsigned cotalphaHistBin=1; cotalphaHistBin<=rescotAlpha_binN; ++cotalphaHistBin )
    for ( unsigned cotbetaHistBin=1; cotbetaHistBin<=rescotBeta_binN; ++cotbetaHistBin )
    {
      unsigned int SingleBigPixelHistN = 100000 + cotalphaHistBin * 100 + cotbetaHistBin;
      theXHistos[SingleBigPixelHistN] = new SimpleHistogramGenerator(
      (TH1F*) thePixelResolutionFile1->Get(  Form( "DQMData/clustFPIX/fhx%u" ,SingleBigPixelHistN ) ) );
      theYHistos[SingleBigPixelHistN] = new SimpleHistogramGenerator(
      (TH1F*) thePixelResolutionFile1->Get(  Form( "DQMData/clustFPIX/fhy%u" ,SingleBigPixelHistN ) ) );
      unsigned int SinglePixelHistN = 100000 + 1000 + cotalphaHistBin * 100 + cotbetaHistBin;
      theXHistos[SinglePixelHistN]  = new SimpleHistogramGenerator(
      (TH1F*) thePixelResolutionFile1->Get(  Form( "DQMData/clustFPIX/fhx%u" ,SinglePixelHistN ) ) );
      theYHistos[SinglePixelHistN]  = new SimpleHistogramGenerator(
      (TH1F*) thePixelResolutionFile1->Get(  Form( "DQMData/clustFPIX/fhy%u" ,SinglePixelHistN ) ) );

    }
}
