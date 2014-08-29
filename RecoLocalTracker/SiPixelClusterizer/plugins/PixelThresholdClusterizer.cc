//----------------------------------------------------------------------------
//! \class PixelThresholdClusterizer
//! \brief A specific threshold-based pixel clustering algorithm
//!
//! General logic of PixelThresholdClusterizer:
//!
//! The clusterization is performed on a matrix with size
//! equal to the size of the pixel detector, each cell containing 
//! the ADC count of the corresponding pixel.
//! The matrix is reset after each clusterization.
//!
//! The search starts from seed pixels, i.e. pixels with sufficiently
//! large amplitudes, found at the time of filling of the matrix
//! and stored in a SiPixelArrayBuffer.
//! 
//! Translate the pixel charge to electrons, we are suppose to
//! do the calibrations ADC->electrons here.
//! Modify the thresholds to be in electrons, convert adc to electrons. d.k. 20/3/06
//! Get rid of the noiseVector. d.k. 28/3/06
//----------------------------------------------------------------------------

// Our own includes
#include "PixelThresholdClusterizer.h"
#include "SiPixelArrayBuffer.h"
#include "CondFormats/SiPixelObjects/interface/SiPixelGainCalibrationOffline.h"
// Geometry
#include "Geometry/TrackerGeometryBuilder/interface/PixelGeomDetUnit.h"
#include "Geometry/CommonTopologies/interface/PixelTopology.h"
//#include "Geometry/CommonTopologies/RectangularPixelTopology.h"
#include "DataFormats/SiPixelDetId/interface/PXBDetId.h"

// STL
#include <stack>
#include <vector>
#include <iostream>
#include <atomic>
using namespace std;

//----------------------------------------------------------------------------
//! Constructor: 
//!  Initilize the buffer to hold pixels from a detector module.
//!  This is a vector of 44k ints, stays valid all the time.  
//----------------------------------------------------------------------------
PixelThresholdClusterizer::PixelThresholdClusterizer
  (edm::ParameterSet const& conf) :
    conf_(conf), bufferAlreadySet(false), theNumOfRows(0), theNumOfCols(0), detid_(0) 
{
  // Get thresholds in electrons
  thePixelThreshold   = 
    conf_.getParameter<int>("ChannelThreshold");
  theSeedThreshold    = 
    conf_.getParameter<int>("SeedThreshold");
  theClusterThreshold = 
    conf_.getParameter<double>("ClusterThreshold");
  theConversionFactor = 
    conf_.getParameter<int>("VCaltoElectronGain");
  theOffset = 
    conf_.getParameter<int>("VCaltoElectronOffset");
  if ( conf_.exists("AdcFullScaleStack") ) theStackADC_=conf_.getParameter<int>("AdcFullScaleStack");
  else 
    theStackADC_=255;
  if ( conf_.exists("FirstStackLayer") ) theFirstStack_=conf_.getParameter<int>("FirstStackLayer");
  else
    theFirstStack_=5;
  
  // Get the constants for the miss-calibration studies
  doMissCalibrate=conf_.getUntrackedParameter<bool>("MissCalibrate",true); 
  doSplitClusters = conf.getParameter<bool>("SplitClusters");
  theBuffer.setSize( theNumOfRows, theNumOfCols );
}
/////////////////////////////////////////////////////////////////////////////
PixelThresholdClusterizer::~PixelThresholdClusterizer() {}

//----------------------------------------------------------------------------
//!  Prepare the Clusterizer to work on a particular DetUnit.  Re-init the
//!  size of the panel/plaquette (so update nrows and ncols), 
//----------------------------------------------------------------------------
bool PixelThresholdClusterizer::setup(const PixelGeomDetUnit * pixDet) 
{
  // Cache the topology.
  const PixelTopology & topol = pixDet->specificTopology();
  
  // Get the new sizes.
  int nrows = topol.nrows();      // rows in x
  int ncols = topol.ncolumns();   // cols in y
  
  theNumOfRows = nrows;  // Set new sizes
  theNumOfCols = ncols;
  
  if ( nrows > theBuffer.rows() || 
       ncols > theBuffer.columns() ) 
    { // change only when a larger is needed
      //if( nrows != theNumOfRows || ncols != theNumOfCols ) {
      //cout << " PixelThresholdClusterizer: pixel buffer redefined to " 
      // << nrows << " * " << ncols << endl;      
      //theNumOfRows = nrows;  // Set new sizes
      //theNumOfCols = ncols;
      // Resize the buffer
      theBuffer.setSize(nrows,ncols);  // Modify
      bufferAlreadySet = true;
    }
  
  return true;   
}
//----------------------------------------------------------------------------
//!  \brief Cluster pixels.
//!  This method operates on a matrix of pixels
//!  and finds the largest contiguous cluster around
//!  each seed pixel.
//!  Input and output data stored in DetSet
//----------------------------------------------------------------------------
void PixelThresholdClusterizer::clusterizeDetUnit( const edm::DetSet<PixelDigi> & input,
						   const PixelGeomDetUnit * pixDet,
						   const std::vector<short>& badChannels,
                                                   edmNew::DetSetVector<SiPixelCluster>::FastFiller& output) {
  
  DigiIterator begin = input.begin();
  DigiIterator end   = input.end();
  
  // Do not bother for empty detectors
  //if (begin == end) cout << " PixelThresholdClusterizer::clusterizeDetUnit - No digis to clusterize";
  
  //  Set up the clusterization on this DetId.
  if ( !setup(pixDet) ) 
    return;
  
  detid_ = input.detId();
  
  //  Copy PixelDigis to the buffer array; select the seed pixels
  //  on the way, and store them in theSeeds.
  copy_to_buffer(begin, end);
  
  //  Loop over all seeds.  TO DO: wouldn't using iterators be faster?
  //  edm::LogError("PixelThresholdClusterizer") <<  "Starting clusterizing" << endl;
  for (unsigned int i = 0; i < theSeeds.size(); i++) 
    {
      
      // Gavril : The charge of seeds that were already inlcuded in clusters is set to 1 electron
      // so we don't want to call "make_cluster" for these cases 
      if ( theBuffer(theSeeds[i]) >= theSeedThreshold ) 
	{  // Is this seed still valid?
	  //  Make a cluster around this seed
	  SiPixelCluster && cluster = make_cluster( theSeeds[i] , output);
	  
	  //  Check if the cluster is above threshold  
	  // (TO DO: one is signed, other unsigned, gcc warns...)
	  if ( cluster.charge() >= theClusterThreshold) 
	    {
	      // std::cout << "putting in this cluster " << i << " " << cluster.charge() << " " << cluster.pixelADC().size() << endl;
	      output.push_back( std::move(cluster) );
	    }
	}
    }
  
  // Erase the seeds.
  theSeeds.clear();
  
  //  Need to clean unused pixels from the buffer array.
  clear_buffer(begin, end);
  
}

//----------------------------------------------------------------------------
//!  \brief Clear the internal buffer array.
//!
//!  Pixels which are not part of recognized clusters are NOT ERASED 
//!  during the cluster finding.  Erase them now.
//!
//!  TO DO: ask Danek... wouldn't it be faster to simply memcopy() zeros into
//!  the whole buffer array?
//----------------------------------------------------------------------------
void PixelThresholdClusterizer::clear_buffer( DigiIterator begin, DigiIterator end ) 
{
  for(DigiIterator di = begin; di != end; ++di ) 
    {
      theBuffer.set_adc( di->row(), di->column(), 0 );   // reset pixel adc to 0
    }
}

//----------------------------------------------------------------------------
//! \brief Copy adc counts from PixelDigis into the buffer, identify seeds.
//----------------------------------------------------------------------------
void PixelThresholdClusterizer::copy_to_buffer( DigiIterator begin, DigiIterator end ) 
{
#ifdef PIXELREGRESSION
  static std::atomic<int> s_ic=0;
  in ic = ++s_ic;
  if (ic==1) {
    // std::cout << (doMissCalibrate ? "VI from db" : "VI linear") << std::endl;
  }
#endif
  int electron[end-begin];
  if ( doMissCalibrate ) {
    (*theSiPixelGainCalibrationService_).calibrate(detid_,begin,end,theConversionFactor, theOffset,electron);
  } else {
    int layer = (DetId(detid_).subdetId()==1) ? PXBDetId(detid_).layer() : 0;
    int i=0;
    for(DigiIterator di = begin; di != end; ++di) {
      auto adc = di->adc();
      const float gain = 135.; // 1 ADC = 135 electrons
      const float pedestal = 0.; //
      electron[i] = int(adc * gain + pedestal);
      if (layer>=theFirstStack_) {
	if (theStackADC_==1&&adc==1) {
	  electron[i] = int(255*135); // Arbitrarily use overflow value.
	}
	if (theStackADC_>1&&theStackADC_!=255&&adc>=1){
	  const float gain = 135.; // 1 ADC = 135 electrons
	  electron[i] = int((adc-1) * gain * 255/float(theStackADC_-1));
	}
      }
      ++i;
    }
    assert(i==(end-begin));
  }

  int i=0;
#ifdef PIXELREGRESSION
  static std::atomic<int> eqD=0;
#endif
  for(DigiIterator di = begin; di != end; ++di) {
    int row = di->row();
    int col = di->column();
    int adc = electron[i++];
#ifdef PIXELREGRESSION
    int adcOld = calibrate(di->adc(),col,row);
    //assert(adc==adcOld);
    if (adc!=adcOld) std::cout << "VI " << eqD  <<' '<< ic  <<' '<< end-begin <<' '<< i <<' '<< di->adc() <<' ' << adc <<' '<< adcOld << std::endl; else ++eqD;
#endif
    if ( adc >= thePixelThreshold) {
      theBuffer.set_adc( row, col, adc);
      if ( adc >= theSeedThreshold) theSeeds.push_back( SiPixelCluster::PixelPos(row,col) );
    }
  }
  assert(i==(end-begin));

}

//----------------------------------------------------------------------------
// Calibrate adc counts to electrons
//-----------------------------------------------------------------
int PixelThresholdClusterizer::calibrate(int adc, int col, int row) 
{
  int electrons = 0;
  int layer= 0;
  if (DetId(detid_).subdetId()==1){ layer = PXBDetId(detid_).layer();}

  if ( doMissCalibrate ) 
    {
      // do not perform calibration if pixel is dead!
      
      if ( !theSiPixelGainCalibrationService_->isDead(detid_,col,row) && 
	   !theSiPixelGainCalibrationService_->isNoisy(detid_,col,row) )
	{
	  
	  // Linear approximation of the TANH response
	  // Pixel(0,0,0)
	  //const float gain = 2.95; // 1 ADC = 2.95 VCALs (1/0.339)
	  //const float pedestal = -83.; // -28/0.339
	  // Roc-0 average
	  //const float gain = 1./0.357; // 1 ADC = 2.80 VCALs 
	  //const float pedestal = -28.2 * gain; // -79.
	  
	  float DBgain     = theSiPixelGainCalibrationService_->getGain(detid_, col, row);
	  float DBpedestal = theSiPixelGainCalibrationService_->getPedestal(detid_, col, row) * DBgain;
	  
	  
	  // Roc-6 average
	  //const float gain = 1./0.313; // 1 ADC = 3.19 VCALs 
	  //const float pedestal = -6.2 * gain; // -19.8
	  // 
	  float vcal = adc * DBgain - DBpedestal;
	  
	  // atanh calibration 
	  // Roc-6 average
	  //const float p0 = 0.00492;
	  //const float p1 = 1.998;
	  //const float p2 = 90.6;
	  //const float p3 = 134.1; 
	  // Roc-6 average
	  //const float p0 = 0.00382;
	  //const float p1 = 0.886;
	  //const float p2 = 112.7;
	  //const float p3 = 113.0; 
	  //float vcal = ( atanh( (adc-p3)/p2) + p1)/p0;
	  
	  electrons = int( vcal * theConversionFactor + theOffset); 
	}
    }
  else 
    { // No misscalibration in the digitizer
      // Simple (default) linear gain 
      const float gain = 135.; // 1 ADC = 135 electrons
      const float pedestal = 0.; //
      electrons = int(adc * gain + pedestal);
      if (layer>=theFirstStack_) {
	if (theStackADC_==1&&adc==1)
	  {
	    electrons = int(255*135); // Arbitrarily use overflow value.
	  }
	if (theStackADC_>1&&theStackADC_!=255&&adc>=1)
	  {
	    const float gain = 135.; // 1 ADC = 135 electrons
	    electrons = int((adc-1) * gain * 255/float(theStackADC_-1));
	  }
      }
    }
  
  return electrons;
}


namespace {

  struct AccretionCluster {
    typedef unsigned short UShort;
    static constexpr UShort MAXSIZE = 256;
    UShort adc[256];
    UShort x[256];
    UShort y[256];
    UShort xmin=16000;
    UShort ymin=16000;
    unsigned int isize=0;
    unsigned int curr=0;

    // stack interface (unsafe ok for use below)
    UShort top() const { return curr;}
    void pop() { ++curr;}   
    bool empty() { return curr==isize;}

    bool add(SiPixelCluster::PixelPos const & p, UShort const iadc) {
      if (isize==MAXSIZE) return false;
      xmin=std::min(xmin,(unsigned short)(p.row()));
      ymin=std::min(ymin,(unsigned short)(p.col()));
      adc[isize]=iadc;
      x[isize]=p.row();
      y[isize++]=p.col();
      return true;
    }
  };

}

//----------------------------------------------------------------------------
//!  \brief The actual clustering algorithm: group the neighboring pixels around the seed.
//----------------------------------------------------------------------------
SiPixelCluster 
PixelThresholdClusterizer::make_cluster( const SiPixelCluster::PixelPos& pix, 
					 edmNew::DetSetVector<SiPixelCluster>::FastFiller& output) 
{
  
  //First we acquire the seeds for the clusters
  int seed_adc;
  stack<SiPixelCluster::PixelPos, vector<SiPixelCluster::PixelPos> > dead_pixel_stack;
  
  //The individual modules have been loaded into a buffer.
  //After each pixel has been considered by the clusterizer, we set the adc count to 1
  //to mark that we have already considered it.
  //The only difference between dead/noisy pixels and standard ones is that for dead/noisy pixels,
  //We consider the charge of the pixel to always be zero.

  /*  this is not possible as dead and noisy pixel cannot make it into a seed...
  if ( doMissCalibrate &&
       (theSiPixelGainCalibrationService_->isDead(detid_,pix.col(),pix.row()) || 
	theSiPixelGainCalibrationService_->isNoisy(detid_,pix.col(),pix.row())) )
    {
      std::cout << "IMPOSSIBLE" << std::endl;
      seed_adc = 0;
      theBuffer.set_adc(pix, 1);
    }
    else {
  */
  seed_adc = theBuffer(pix.row(), pix.col());
  theBuffer.set_adc( pix, 1);
      //  }
  
  AccretionCluster acluster;
  acluster.add(pix, seed_adc);
  
  //Here we search all pixels adjacent to all pixels in the cluster.
  bool dead_flag = false;
  while ( ! acluster.empty()) 
    {
      //This is the standard algorithm to find and add a pixel
      auto curInd = acluster.top(); acluster.pop();
      for ( auto c = std::max(0,int(acluster.y[curInd])-1); c < std::min(int(acluster.y[curInd])+2,theBuffer.columns()) ; ++c) {
	for ( auto r = std::max(0,int(acluster.x[curInd])-1); r < std::min(int(acluster.x[curInd])+2,theBuffer.rows()); ++r)  {
	  if ( theBuffer(r,c) >= thePixelThreshold) {
	    SiPixelCluster::PixelPos newpix(r,c);
	    if (!acluster.add( newpix, theBuffer(r,c))) goto endClus;
	    theBuffer.set_adc( newpix, 1);
	  }
	     

	      /* //Commenting out the addition of dead pixels to the cluster until further testing -- dfehling 06/09
	      //Check on the bounds of the module; this is to keep the isDead and isNoisy modules from returning errors 
	      else if(r>= 0 && c >= 0 && (r <= (theNumOfRows-1.)) && (c <= (theNumOfCols-1.))){ 
	      //Check for dead/noisy pixels check that the buffer is not -1 (already considered).  Check whether we want to split clusters separated by dead pixels or not.
	      if((theSiPixelGainCalibrationService_->isDead(detid_,c,r) || theSiPixelGainCalibrationService_->isNoisy(detid_,c,r)) && theBuffer(r,c) != 1){
	      
	      //If a pixel is dead or noisy, check to see if we want to split the clusters or not.  
	      //Push it into a dead pixel stack in case we want to split the clusters.  Otherwise add it to the cluster.
   	      //If we are splitting the clusters, we will iterate over the dead pixel stack later.
	      
	      SiPixelCluster::PixelPos newpix(r,c);
	      if(!doSplitClusters){
	      
	      cluster.add(newpix, theBuffer(r,c));}
	      else if(doSplitClusters){
	      dead_pixel_stack.push(newpix);
	      dead_flag = true;}
	      
	      theBuffer.set_adc(newpix, 1);
	      } 
	      
	      }
	      */
	    


	    }
	}
      
    }  // while accretion
 endClus:
  SiPixelCluster cluster(acluster.isize,acluster.adc, acluster.x,acluster.y, acluster.xmin,acluster.ymin);
  //Here we split the cluster, if the flag to do so is set and we have found a dead or noisy pixel.
  
  if (dead_flag && doSplitClusters) 
    {
      //Set the first cluster equal to the existing cluster.
      SiPixelCluster first_cluster = cluster;
      bool have_second_cluster = false;
      while ( !dead_pixel_stack.empty() )
	{
	  //consider each found dead pixel
	  SiPixelCluster::PixelPos deadpix = dead_pixel_stack.top(); dead_pixel_stack.pop();
	  theBuffer.set_adc(deadpix, 1);
	 
	  //Clusterize the split cluster using the dead pixel as a seed
	  SiPixelCluster second_cluster = make_cluster(deadpix, output);
	  
	  //If both clusters would normally have been found by the clusterizer, put them into output
	  if ( second_cluster.charge() >= theClusterThreshold && 
	       first_cluster.charge() >= theClusterThreshold )
	    {
	      output.push_back( second_cluster );
	      have_second_cluster = true;	
	    }
	  
	  //We also want to keep the merged cluster in data and let the RecHit algorithm decide which set to keep
	  //This loop adds the second cluster to the first.
	  const std::vector<SiPixelCluster::Pixel>& branch_pixels = second_cluster.pixels();
	  for ( unsigned int i = 0; i<branch_pixels.size(); i++)
	    {
	      int temp_x = branch_pixels[i].x;
	      int temp_y = branch_pixels[i].y;
	      int temp_adc = branch_pixels[i].adc;
	      SiPixelCluster::PixelPos newpix(temp_x, temp_y);
	      cluster.add(newpix, temp_adc);}
	}
      
      //Remember to also add the first cluster if we added the second one.
      if ( first_cluster.charge() >= theClusterThreshold && have_second_cluster) 
	{
	  output.push_back( first_cluster );
	}
    }
  
  return cluster;
}

