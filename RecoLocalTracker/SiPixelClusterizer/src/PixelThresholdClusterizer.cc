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
#include "RecoLocalTracker/SiPixelClusterizer/interface/PixelThresholdClusterizer.h"
#include "RecoLocalTracker/SiPixelClusterizer/interface/SiPixelArrayBuffer.h"

// Geometry
#include "Geometry/TrackerGeometryBuilder/interface/PixelGeomDetUnit.h"
#include "Geometry/CommonTopologies/interface/PixelTopology.h"
//#include "Geometry/CommonTopologies/RectangularPixelTopology.h"

// STL
#include <stack>
#include <vector>
#include <iostream>
using namespace std;

//----------------------------------------------------------------------------
//! Constructor: 
//!  Initilize the buffer to hold pixels from a detector module.
//!  This is a vector of 44k ints, stays valid all the time.  
//----------------------------------------------------------------------------
PixelThresholdClusterizer::PixelThresholdClusterizer
  (edm::ParameterSet const& conf) :
    conf_(conf), bufferAlreadySet(false), theNumOfRows(0), theNumOfCols(0), detid_(0) {

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
	
   
   // Get the constants for the miss-calibration studies
   doMissCalibrate=conf_.getUntrackedParameter<bool>("MissCalibrate",true); 

   theBuffer.setSize( theNumOfRows, theNumOfCols );
   //initTiming();
}
/////////////////////////////////////////////////////////////////////////////
PixelThresholdClusterizer::~PixelThresholdClusterizer() {}

//----------------------------------------------------------------------------
//!  Prepare the Clusterizer to work on a particular DetUnit.  Re-init the
//!  size of the panel/plaquette (so update nrows and ncols), 
//----------------------------------------------------------------------------
bool PixelThresholdClusterizer::setup(const PixelGeomDetUnit * pixDet) {

  // Cache the topology.
  const PixelTopology & topol = pixDet->specificTopology();

  // Get the new sizes.
  int nrows = topol.nrows();      // rows in x
  int ncols = topol.ncolumns();   // cols in y

  if( nrows > theNumOfRows || ncols > theNumOfCols ) { // change only when a larger is needed
    //if( nrows != theNumOfRows || ncols != theNumOfCols ) {
    //cout << " PixelThresholdClusterizer: pixel buffer redefined to " 
    // << nrows << " * " << ncols << endl;      
    theNumOfRows = nrows;  // Set new sizes
    theNumOfCols = ncols;
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

  //TimeMe tm1( *theClustersTimer, false);

   DigiIterator begin = input.begin();
   DigiIterator end   = input.end();

   // Do not bother for empty detectors
   //if (begin == end) cout << " PixelThresholdClusterizer::clusterizeDetUnit - No digis to clusterize";

   //  Set up the clusterization on this DetId.
   if (!setup(pixDet)) return;
   detid_ = input.detId();
   //  Copy PixelDigis to the buffer array; select the seed pixels
   //  on the way, and store them in theSeeds.
   copy_to_buffer(begin, end);
  //  At this point we know the number of seeds on this DetUnit, and thus
  //  also the maximal number of possible clusters, so resize theClusters
  //  in order to make vector<>::push_back() efficient.
  // output.reserve ( theSeeds.size() ); //GPetruc: It is better *not* to reserve, with the new DetSetVector!
    
  //  Loop over all seeds.  TO DO: wouldn't using iterators be faster?
  for (unsigned int i = 0; i < theSeeds.size(); i++) {
      
    if ( theBuffer(theSeeds[i]) != 0) {  // Is this seed still valid?
	
      //  Make a cluster around this seed
      SiPixelCluster cluster = make_cluster( theSeeds[i] );
	
      //  Check if the cluster is above threshold  
      // (TO DO: one is signed, other unsigned, gcc warns...)
      if ( cluster.charge() >= theClusterThreshold ) {
	output.push_back( cluster );
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
void PixelThresholdClusterizer::clear_buffer( DigiIterator begin, DigiIterator end ) {
  // TimeMe tm1( *theClearTimer, false);
  for(DigiIterator di = begin; di != end; ++di ) {
    theBuffer.set_adc( di->row(), di->column(), 0 );   // reset pixel adc to 0
  }
}

//----------------------------------------------------------------------------
//! \brief Copy adc counts from PixelDigis into the buffer, identify seeds.
//----------------------------------------------------------------------------
void PixelThresholdClusterizer::copy_to_buffer( DigiIterator begin, DigiIterator end ) {
  //TimeMe tm1( *theCopyTimer, false);
  for(DigiIterator di = begin; di != end; ++di) {
    int row = di->row();
    int col = di->column();
    int adc = calibrate(di->adc(),col,row); // convert ADC -> electrons
    if ( adc >= thePixelThreshold) {
      theBuffer.set_adc( row, col, adc);
      if ( adc >= theSeedThreshold) { 
	theSeeds.push_back( SiPixelCluster::PixelPos(row,col) );
      }
    }
  }
}

//----------------------------------------------------------------------------
// Calibrate adc counts to electrons
//-----------------------------------------------------------------
int PixelThresholdClusterizer::calibrate(int adc, int col, int row) {
  int electrons = 0;

  if(doMissCalibrate) {
    // do not perform calibration if pixel is dead!

    if(!theSiPixelGainCalibrationService_->isDead(detid_,col,row)){

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
  else { // No misscalibration in the digitizer
    // Simple (default) linear gain 
    const float gain = 135.; // 1 ADC = 135 electrons
    const float pedestal = 0.; //
    electrons = int(adc * gain + pedestal);
  }
  
  return electrons;
}
//----------------------------------------------------------------------------
//!  \brief The actual clustering algorithm: group the neighboring pixels around the seed.
//----------------------------------------------------------------------------
SiPixelCluster 
PixelThresholdClusterizer::make_cluster( const SiPixelCluster::PixelPos& pix) 
{
  //TimeMe tm1( *theMakeClustTimer, false);

  // Make the cluster
  SiPixelCluster cluster( pix, theBuffer( pix.row(), pix.col()) );
  stack<SiPixelCluster::PixelPos, vector<SiPixelCluster::PixelPos> > pixel_stack;

  theBuffer.set_adc( pix, 0);
  pixel_stack.push( pix);

  while ( ! pixel_stack.empty()) {
    SiPixelCluster::PixelPos curpix = pixel_stack.top(); pixel_stack.pop();
    for ( int r = curpix.row()-1; r <= curpix.row()+1; ++r) {
      for ( int c = curpix.col()-1; c <= curpix.col()+1; ++c) {
	  if ( theBuffer(r,c) >= thePixelThreshold) {
	    SiPixelCluster::PixelPos newpix(r,c);
	    cluster.add( newpix, theBuffer(r,c));
	    theBuffer.set_adc( newpix, 0);
	    pixel_stack.push( newpix);
	  
	}
      }
    }
  }
  return cluster;
}

//----------------------------------------------------------------------------
//! \brief Initialize the timers.
//----------------------------------------------------------------------------
// void PixelThresholdClusterizer::initTiming() {
//   TimingReport& tr(*TimingReport::current());
//   theSetupTimer =      &tr["PixelClusterizer setup+digi"];
//   theClustersTimer =   &tr["PixelClusterizer clusters"];
//   theClusterizeTimer = &tr["PixelClusterizer clusterize"];
//   theRecHitTimer =     &tr["PixelClusterizer create RecHits"];
//   theCopyTimer =       &tr["PixelClusterizer copy to buffer"];
//   theClearTimer =      &tr["PixelClusterizer clear buffer"];
//   theMakeClustTimer =  &tr["PixelClusterizer make one cluster"];
//   theCacheGetTimer =   &tr["PixelClusterizer cache access"];
//   theCachePutTimer =   &tr["PixelClusterizer cache fill"];
//   static bool detailedTiming 
//     = conf_.getUntrackedParameter<bool>("DetailedTiming",false);
//   if (!detailedTiming) {
//     tr.switchOn( "PixelClusterizer setup+digi",false);
//     tr.switchOn( "PixelClusterizer clusters",false);
//     tr.switchOn( "PixelClusterizer create RecHits",false);
//     tr.switchOn( "PixelClusterizer copy to buffer",false);
//     tr.switchOn( "PixelClusterizer clear buffer",false);
//     tr.switchOn( "PixelClusterizer make one cluster",false);
//     tr.switchOn( "PixelClusterizer cache access",false);
//     tr.switchOn( "PixelClusterizer cache fill",false);
//   }
//}
