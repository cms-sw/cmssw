
/** PixelTemplateSmearerBase.cc
 * ---------------------------------------------------------------------
 * Base class for FastSim plugins to simulate all simHits on one DetUnit.
 * 
 * Petar Maksimovic (JHU), based the code by 
 * Guofan Hu (JHU) from SiPixelGaussianSmearingRecHitConverterAlgorithm.cc
 * Alice Sady (JHU): new pixel resolutions (2015) and hit merging code.
 * ---------------------------------------------------------------------
 */

// SiPixel Gaussian Smearing
#include "FastSimulation/TrackingRecHitProducer/interface/PixelTemplateSmearerBase.h"
#include "FastSimulation/TrackingRecHitProducer/interface/TrackingRecHitAlgorithmFactory.h"
#include "FastSimulation/TrackingRecHitProducer/interface/TrackingRecHitProduct.h"

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
#include <TH2F.h>
//#include <TAxis.h>

//#define FAMOS_DEBUG

const double microntocm = 0.0001;

//------------------------------------------------------------------------------
//  Constructor.  Only one is possible, since its signature needs to
//  match the base class in order to be made by the factory.
//  ------------------------------------------------------------------------------
PixelTemplateSmearerBase::PixelTemplateSmearerBase(
  const std::string& name,
  const edm::ParameterSet& config,
  edm::ConsumesCollector& consumesCollector 
  ) :
  TrackingRecHitAlgorithm(name,config,consumesCollector),
  //const edm::ParameterSet& pset,
  //GeomDetType::SubDetector pixelPart) :
  pset_(config),   // &&& obviously now a duplicate of a base class variable
  thePixelPart(GeomDetEnumerators::PixelBarrel)  // default.  Derived classes will fix it.
{
  std::cout << "PixelTemplateSmearerBase"<< std::endl;
  // Switch between old (ORCA) and new (CMSSW) pixel parameterization
  useCMSSWPixelParameterization = pset_.getParameter<bool>("UseCMSSWPixelParametrization");
}



//------------------------------------------------------------------------------
//  Destructor.
//  &&& Check what happens with this when we destruct the derived classes.
//------------------------------------------------------------------------------
PixelTemplateSmearerBase::~PixelTemplateSmearerBase()
{
  std::cout << "P ~PixelTemplateSmearerBase"<< std::endl;
  std::map<unsigned,const SimpleHistogramGenerator*>::const_iterator it;
  for ( it=theXHistos.begin(); it!=theXHistos.end(); ++it )
    delete it->second;
  for ( it=theYHistos.begin(); it!=theYHistos.end(); ++it )
    delete it->second;

  theXHistos.clear();
  theYHistos.clear();
  std::cout << "end P ~PixelTemplateSmearerBase"<< std::endl;
}



//------------------------------------------------------------------------------
//  Simulate all simHits on one DetUnit (i.e. module/plaquette).
//
//  NB: the loop logic uses [] to access the elements of the vector.  I (Petar)
//  tried to make it happen with iterators, but it turned out that using [] was 
//  both easier to write and *much* easier to understand (and thus to support
//  long term).
//------------------------------------------------------------------------------
TrackingRecHitProductPtr 
PixelTemplateSmearerBase::process(TrackingRecHitProductPtr product) const
{
  //--- getSimHits() returns a reference to a vector of pointers to
  //    PSimHits.  Since we need to keep track of various locations in
  //    this array, we need to use either indices or iterators.
  std::vector<const PSimHit*> & simHits = product->getSimHits();  


  //--- Random engine
  //
  RandomEngineAndDistribution const & randomEngine = getRandomEngine();


  //--- Geometry of our pixel DetUnit
  //
  const GeomDet* geomDet = getTrackerGeometry().idToDetUnit(product->getDetId());
  const PixelGeomDetUnit * pixelGeomDet = dynamic_cast< const PixelGeomDetUnit* >( geomDet );
  if (pixelGeomDet == 0) {
    throw cms::Exception("FastSimulation/TrackingRecHitProducer") << "The GeomDetUnit is not a PixelGeomDetUnit.  This should never happen!";
  }
  const BoundPlane& theDetPlane = pixelGeomDet->surface();
  const Bounds& theBounds = theDetPlane.bounds();
  const double boundX = theBounds.width()/2.;
  const double boundY = theBounds.length()/2.;


  //--- Make some collections
  //
  std::vector< const PSimHit* > listOfUnmergedHits; // this that were not merged
  std::vector< MergeGroup* > listOfMergeGroups;     // groups of hits that should be merged
  int nHits = simHits.size();                       //  Number of hits on this DetUnit
  MergeGroup* mergeGroupByHit[ nHits ];             // fixed size array, 0 if hit is unmerged

  
  //--- Check special cases (nHits = 0 or 1), and finish them off first.
  //
  if ( nHits == 0 ) {
    return product;
  }
  else if ( nHits == 1 ) {
    //--- One unmerged hit, no merge groups...
    listOfUnmergedHits.push_back( simHits[0] );
  }
  else {
    //--- The usual case.  More than one hit on this DetUnit.
    //    Iterate over hits.
    //
    for ( int i = 0; i < nHits-1; ++i ) {
      //
      mergeGroupByHit[i] = 0;     // initialize this cell to a NULL pointer.
      
      //--- Inner loop over remaining hits.  j starts from i+1 and goes to nHits
      for ( int j = i+1 ; j < nHits; ++j ) {
	
	//--- Calculate the distance between hits i and j:
	bool merged = hitsMerge( *simHits[i], *simHits[j] );
	
	//--- Decide what to do about these two hits.
	if ( merged ) {
	  // First, check if the other guy (j) is in some merge group already
	  if ( mergeGroupByHit[j] != 0 ) {
	    // It is... use it.
	    mergeGroupByHit[i] = mergeGroupByHit[j];          // use the same MG
	    mergeGroupByHit[i]->push_back( simHits[i] );      // save i in there
	  }
	  else { 
	    // j is not merged.  Check if i is merged with another hit yet.
	    //
	    if ( mergeGroupByHit[i] == 0 ) {
	      // This is the first time we realized i is merged with any
	      // other hit.  Create a new merge group for i and j.
	      mergeGroupByHit[i] = new MergeGroup();
	      listOfMergeGroups.push_back( mergeGroupByHit[i] );   // keep track of it
	      //
	      // Add hit i as the first to its own merge group
	      // (simHits[i] is a const pointer to PSimHit).
	      mergeGroupByHit[i]->push_back( simHits[i] );
	    }
	    //--- Add hit j as well
	    mergeGroupByHit[i]->push_back( simHits[j] );
	    //
	    //--- Mark that hit j is a part of the same merge group.  This
	    //    way, we can find the same merge group starting from
	    //    index j, too.  Note that if hit i was a part of a merge
	    //    goup found earlier (for some hit prior to i), then
	    //    mergeGroupByHit[i] would not be zero even if it wasn't
	    //    made for hit i :)
	    mergeGroupByHit[j] = mergeGroupByHit[i];
	    
	  } // --- end of else if ( j has merge group )
	  
	} //--- end of if (merged)
	
      } //--- end of loop over j

      //--- At this point, there are two possibilities.  Either hit i
      //    was already chosen to be merged with some hit prior to it,
      //    or the loop over j found another merged hit.  In either
      //    case, if mergeGroupByHit[i] is empty, then the hit is
      //    unmerged.
      //
      if ( mergeGroupByHit[i] == 0 ) {
	//--- Keep track of it.
	listOfUnmergedHits.push_back( simHits[i] );
      }
    } //--- end of loop over i
    
  } // --- end of if (nHits == 1) else {...}


  //--- We now have two lists: a list of hits that are unmerged, and
  //    the list of merge groups.  Process each separately.
  //
  product = processUnmergedHits( listOfUnmergedHits, product,
				 pixelGeomDet, boundX, boundY,
				 &randomEngine );


  product = processMergeGroups(  listOfMergeGroups,  product,
				 pixelGeomDet, boundX, boundY,
				 &randomEngine );

  //--- We're done with this det unit, and ought to clean up used
  //    memory.  We don't own the PSimHits, and the vector of
  //    listOfUnmergedHits simply goes out of scope.  However, we
  //    created the MergeGroups and thus we need to get rid of them.
  //
  for ( std::vector<MergeGroup*>::iterator 
	  mg_it = listOfMergeGroups.begin(),
	  mg_end = listOfMergeGroups.end();
	mg_it != mg_end;
	++mg_it ) {
    delete *mg_it;    // each MergeGroup is deleted; its ptrs to PSimHits we do not own...
  }

  return product;
}


//------------------------------------------------------------------------------
//   Smear one hit.  The main action is in here.
// &&& @ALICE: this function may also need to be "const" :(
//------------------------------------------------------------------------------
SiTrackerGSRecHit2D PixelTemplateSmearerBase::smearHit(
  const PSimHit& simHit,
  const PixelGeomDetUnit* detUnit,
  const double boundX,
  const double boundY,
  RandomEngineAndDistribution const* random)  const
{
  std::cout << "P smearHit"<< std::endl;
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
  //ALICE TESTING AREA
  /*  float loceta = fabs(-log((double)(-cotbeta+sqrt((double)(1.+cotbeta*cotbeta)))));
  float locdis = sqrt(pow(locx, 2) + pow(locy, 2));
  std::cout << "cotbeta local x local y local eta local distance" << std::endl;
  std::cout << cotbeta << " " << locx << " " << locy << " " << loceta << " " << locdis << std::endl;
  probfile = new TFile( "FastSimulation/TrackingRecHitProducer/data/mergeprob.root"  ,"READ");
  TH2F * probhisto = (TH2F*)probfile->Get("h2bc");
  float prob = probhisto->GetBinContent(probhisto->GetXaxis()->FindFixBin(locdis),probhisto->GetYaxis()->FindFixBin(loceta));
  std::cout << "probability of merging: " << prob << std::endl;
  float probability = ((double) rand() / (RAND_MAX));
  std::cout << probability << std::endl;*/
  if( isForward ) {
    if( cotbeta < 0 ) sign=-1.;
    cotbeta = sign*cotbeta;
  }

  
 
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
  // &&& Petar: I've no idea what this code does.  Hopefully Morris will remember :)
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


  //--- Prepare to return results
  Local3DPoint thePosition;  
  double       thePositionX; 
  double       thePositionY; 
  double       thePositionZ; 
  LocalError   theError;     
  double       theErrorX;    
  double       theErrorY;    
  //double       theErrorZ;    
  unsigned int theClslenx;   
  unsigned int theClsleny;   



  //------------------------------
  //  Check if the cluster is near an edge.  If it protrudes
  //  outside the edge of the sensor, the truncate it and it will
  //  get significantly messed up.
  //------------------------------
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
  templ.temperrors(tempId, cotalpha, cotbeta, nqbin,          // inputs
		   sigmay, sigmax, sy1, sy2, sx1, sx2 );      // outputs

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
  // theErrorZ = 1e-8; // 1 um means zero  (not needed)
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
  //ALICE: redid numbers associated with regular barrel and forward hits to match new histogram labeling
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
	 else 	theXHistN = 1 * 10000 + cotbetaHistBin * 10 + cotalphaHistBin ; 
	 //else   theXHistN = 1 * 100000 + 1 * 1000 + cotalphaHistBin * 100 + cotbetaHistBin ;
       }
       else
       {
	 if(hasBigPixelInX)  theXHistN = 1 * 1000000 + 1 * 100000 + cotalphaHistBin * 1000 + cotbetaHistBin * 10 + (nqbin+1);
	 else 	theXHistN = 1 * 100000 + 1 * 10000 + cotbetaHistBin * 100 + cotalphaHistBin * 10 + (nqbin+1);
	 //else   theXHistN = 1 * 1000000 + 1 * 100000 + 1 * 10000 + cotalphaHistBin * 1000 + cotbetaHistBin * 10 + (nqbin+1);
       }
       if(singley)
       {
	 if(hitbigy)  theYHistN = 1 * 100000 + cotalphaHistBin * 100 + cotbetaHistBin ;
	 else 	theYHistN = 1 * 10000 + cotbetaHistBin * 10 + cotalphaHistBin ;
	 //else   theYHistN = 1 * 100000 + 1 * 1000 + cotalphaHistBin * 100 + cotbetaHistBin ;
       }
       else
       {
	 if(hasBigPixelInY)  theYHistN = 1 * 1000000 + 1 * 100000 + cotalphaHistBin * 1000 + cotbetaHistBin * 10 + (nqbin+1);
	 else 	theYHistN = 1 * 100000 + 1 * 10000 + cotbetaHistBin * 100 + cotalphaHistBin * 10 + (nqbin+1);
	 //else   theYHistN = 1 * 1000000 + 1 * 100000 + 1 * 10000 + cotalphaHistBin * 1000 + cotbetaHistBin * 10 + (nqbin+1);
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
	   theXHistN = cotbetaHistBin * 10 + cotalphaHistBin;
       //theXHistN = 100000 + 1000 + cotalphaHistBin * 100 + cotbetaHistBin;
       else
         theXHistN = 10000 + cotbetaHistBin * 100 +  cotalphaHistBin * 10 +  (nqbin+1);
       //theXHistN = 100000 + 10000 + cotalphaHistBin * 1000 +  cotbetaHistBin * 10 +  (nqbin+1);
       if( singley )
         if( hitbigy )
	   theYHistN = 100000 + cotalphaHistBin * 100 + cotbetaHistBin;
	 else
	   theYHistN = cotbetaHistBin * 10 + cotalphaHistBin;
       //theYHistN = 100000 + 1000 + cotalphaHistBin * 100 + cotbetaHistBin;
       else
         theYHistN = 10000 + cotbetaHistBin * 100 +  cotalphaHistBin * 10 + (nqbin+1);
       //theYHistN = 100000 + 10000 + cotalphaHistBin * 1000 +  cotbetaHistBin * 10 +  (nqbin+1);
     }
  }
  unsigned int counter = 0;
  do {
    //
    // Smear the hit Position
    
    std::map<unsigned int, const SimpleHistogramGenerator*>::const_iterator xgenIt = theXHistos.find(theXHistN);
    std::map<unsigned int, const SimpleHistogramGenerator*>::const_iterator ygenIt = theYHistos.find(theYHistN);
    if (xgenIt==theXHistos.cend() || ygenIt==theYHistos.cend())
    {
        throw cms::Exception("FastSimulation/TrackingRecHitProducer") << "Histogram ("<<theXHistN<<","<<theYHistN<<") was not found for PixelTemplateSmearer. Check if the smearing template exists.";
    }
    
    
    const SimpleHistogramGenerator* xgen = xgenIt->second;
    const SimpleHistogramGenerator* ygen = ygenIt->second;
    
    thePositionX = xgen->generate(random);
    thePositionY = ygen->generate(random);

    // The old version:
    //    thePositionX = theXHistos[theXHistN]->generate(random);
    //    thePositionY = theYHistos[theYHistN]->generate(random);


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
      // If we tried to generate thePosition, and it's out of the bounds
      // for 20 times, then punt and return the simHit's location.
      // &&& Petar: how often does this happen?
      thePosition = Local3DPoint(simHit.localPosition().x(), 
                                 simHit.localPosition().y(), 
                                 simHit.localPosition().z());
      break;
    }
  } while(fabs(thePosition.x()) > boundX  || fabs(thePosition.y()) > boundY);
  std::cout << "end P smearHit"<< std::endl;

  //--- We now have everything to make a SiTrackerGSRecHit2D
  //
  SiTrackerGSRecHit2D recHit( thePosition, //const LocalPoint &   (LocalPoint is a typedef  Local3DPoint)
			      theError,    //const LocalError &
			      *detUnit,    //GeomDet const &idet    (PixelGeomDetUnit : ... : GeomDet)
			      0,           //const int simhitId
			      0,           //const int simtrackId
			      0,           //const uint32_t eeId
			      SiTrackerGSRecHit2D::ClusterRef(),   //ClusterRef const &cluster
			      theClslenx,  //const int pixelMultiplicityX
			      theClsleny   //const int pixelMultiplicityY
			      );
  return recHit;
}


//------------------------------------------------------------------------------
//   Process all unmerged hits.  Calls smearHit() for each PSimHit, which returns
//   a RecHit, which we then append to the product.
//------------------------------------------------------------------------------
TrackingRecHitProductPtr PixelTemplateSmearerBase::
processUnmergedHits( std::vector< const PSimHit* > & unmergedHits, 
                     TrackingRecHitProductPtr product,
		     const PixelGeomDetUnit * detUnit,
		     const double boundX, const double boundY,
		     RandomEngineAndDistribution const * random
		     )  const
{
  //--- Iterate over unmergedHits vector, and call smearHit() for each
  //    dereferenced iterator. 
  //
  for (const PSimHit* simHit : unmergedHits) {
    SiTrackerGSRecHit2D recHit = smearHit( *simHit, detUnit, boundX, boundY, random );
    product->getRecHits().push_back( recHit );
  }
  return product;
}



//------------------------------------------------------------------------------
//   Process all groups of merged hits.
//------------------------------------------------------------------------------
TrackingRecHitProductPtr PixelTemplateSmearerBase::
processMergeGroups( std::vector< MergeGroup* > & mergeGroups,
		    TrackingRecHitProductPtr product,
		    const PixelGeomDetUnit * detUnit,
		    const double boundX, const double boundY,
		    RandomEngineAndDistribution const * random
		    )  const
{
  for ( std::vector<MergeGroup*>::iterator 
	  mg_it = mergeGroups.begin(),
	  mg_end = mergeGroups.end();
	mg_it != mg_end;
	++mg_it ) {
    //
    SiTrackerGSRecHit2D recHit = smearMergeGroup( *mg_it, detUnit, boundX, boundY, random );
    product->getRecHits().push_back( recHit );
  }
  return product;
}


//------------------------------------------------------------------------------
//   Process all groups of merged hits.
//------------------------------------------------------------------------------
SiTrackerGSRecHit2D PixelTemplateSmearerBase::
smearMergeGroup( MergeGroup* mg,
		 const PixelGeomDetUnit * detUnit,
		 const double boundX, const double boundY,
		 RandomEngineAndDistribution const * random
		 ) const 
{
  // &&& TOY EXAMPLE -- just to make the code compile, with the correct return value.  
  // &&& @ALICE: please implement.
  // &&& For each merge group, iterate over hits, find those with max y and min y
  // &&& and hits with max x and min x, and then take the average of the two,
  // &&& with appropriate errors.
  LocalPoint position(-1,-1,-1);
  LocalError error(-1,-1,-1);

  SiTrackerGSRecHit2D recHit( position, //const LocalPoint &     (LocalPoint is a typedef for Local3DPoint)
			      error,    //const LocalError &
			      *detUnit,    //GeomDet const &idet    (PixelGeomDetUnit : ... : GeomDet)
			      0,           //const int simhitId
			      0,           //const int simtrackId
			      0,           //const uint32_t eeId
			      SiTrackerGSRecHit2D::ClusterRef(),   //ClusterRef const &cluster
			      -1,  //const int pixelMultiplicityX
			      -1   //const int pixelMultiplicityY
			      );
  return recHit;
}


//-----------------------------------------------------------------------------
//   Method to determine if hits merge.
//
// &&& To speed the things up, maybe we should have min and max distances, and
// &&& throw random numbers for only those in between ?  --> @ALICE please decide.
// 	if ( d < d_min ) {
// 	  // for sure merged
// 	  merged = True;
// 	}
// 	else if ( d > d_max ) {
// 	  // for sure unmerged
// 	  merged = False;
// 	}
// 	else {
//        // run the code below ... 
// 	}
//-----------------------------------------------------------------------------
bool PixelTemplateSmearerBase::hitsMerge(const PSimHit& simHit1,const PSimHit& simHit2) const
{
  //getting cotbeta and the local eta
  LocalVector localDir = simHit1.momentumAtEntry().unit();
  float locy1 = localDir.y();
  float locz1 = localDir.z();
  float cotbeta = locy1/locz1;
  float loceta = fabs(-log((double)(-cotbeta+sqrt((double)(1.+cotbeta*cotbeta)))));

  //getting local x and y position and distance between two points
  const Local3DPoint lp1 = simHit1.localPosition();
  const Local3DPoint lp2 = simHit2.localPosition();
  float lpy1 = lp1.y();
  float lpx1 = lp1.x();
  float lpy2 = lp2.y();
  float lpx2 = lp2.x();
  float locdis = 10000.* sqrt(pow(lpx1 - lpx2, 2) + pow(lpy1 - lpy2, 2));

  std::cout << "cotbeta local x1 local y1 local x2 local y2 local eta local distance" << std::endl;
  std::cout << cotbeta << " " << lpx1 << " " << lpy1 << " " << lpx2 << " " << lpy2 << " " << loceta << " " << locdis << std::endl;
 
  //assigning probability file based on location (barrel or forward)
  //  if( isForward ) {
  //  probfile = new TFile( "FastSimulation/TrackingRecHitProducer/data/fmergeprob.root"  ,"READ");}
  //else {
  //  probfile = new TFile( "FastSimulation/TrackingRecHitProducer/data/bmergeprob.root"  ,"READ");}

  //getting probability from histogram generated with Morris's code
  TH2F * probhisto = (TH2F*)probfile->Get("h2bc");
  float prob = probhisto->GetBinContent(probhisto->GetXaxis()->FindFixBin(locdis),probhisto->GetYaxis()->FindFixBin(loceta));
  //assinging random probability between 0 and 1
  float randprob = ((double) rand() / (RAND_MAX));
  
  std::cout << "probability of merging: " << prob << " and assigned probability: " << randprob << std::endl;
  
  //returning true if random probability is less than calculated, returning false if its more than calculated
  if (randprob <= prob) return true;
  else return false;
}


//-----------------------------------------------------------------------------
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
bool PixelTemplateSmearerBase::isFlipped(const PixelGeomDetUnit* theDet) const {
  std::cout << "P isFlipped"<< std::endl;
 // Check the relative position of the local +/- z in global coordinates.
  float tmp1 = theDet->surface().toGlobal(Local3DPoint(0.,0.,0.)).perp();
  float tmp2 = theDet->surface().toGlobal(Local3DPoint(0.,0.,1.)).perp();
  //  std::cout << " 1: " << tmp1 << " 2: " << tmp2 << std::endl;
  std::cout << "end P isFlipped"<< std::endl;
  if ( tmp2<tmp1 ) return true;
  else return false;    
}


