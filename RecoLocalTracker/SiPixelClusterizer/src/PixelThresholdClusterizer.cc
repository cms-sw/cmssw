
#include "RecoLocalTracker/SiPixelClusterizer/interface/PixelThresholdClusterizer.h"
#include "RecoLocalTracker/SiPixelClusterizer/interface/SiPixelArrayBuffer.h"

// #include "TrackerReco/TkPixelClusterizer/interface/PixelThresholdClusterizer.h"
// #include "Tracker/SiPixelDet/interface/PixelDigi.h"
// #include "Tracker/SiPixelDet/interface/PixelReadout.h"
// #include "Tracker/SiPixelDet/interface/PixelTopology.h"
// #include "Tracker/SiPixelDet/interface/PixelDetType.h"
// #include "Tracker/SiPixelDet/interface/PixelDet.h"

// #include "TrackerReco/TkPixelClusterizer/interface/ClusterParameterEstimator.h"
// #include "TrackerReco/TkPixelClusterizer/src/ClusterParameterEstimatorFactory.h"
// #include "TrackerReco/TkPixelClusterizer/interface/PixelRecHit.h"

#include <stack>
#include <vector>
#include <iostream>
using namespace std;

//----------------------------------------------------------------------------
//! General logic of PixelThresholdClusterizer:
//!
//! The clusterization is performed on a matrix with size
//! equal to the size of the pixel detector, each cell containing 
//! the ADC count of the corresponding pixel.
//! The matrix is reset after each clusterization.
//!
//! The search starts from seed pixels, i.e. pixels with sufficiently
//! large amplitudes, found at the time of filling of the matrix
//! and stored in a
//----------------------------------------------------------------------------




//----------------------------------------------------------------------------
//! Constructor: 
//!  Initilize the buffer to hold pixels from a detector module.
//!  This is a vector of 44k ints, stays valid all the time.  
//----------------------------------------------------------------------------
PixelThresholdClusterizer::PixelThresholdClusterizer(edm::ParameterSet const& conf)
  :
  conf_(conf),
  bufferAlreadySet(false), 
  theNumOfRows(0), theNumOfCols(0)
{
  // Set the thresholds -- NOTE: in units of noise!
  thePixelThresholdInNoiseUnits   = conf_.getParameter<double>("ChannelThreshold");
  theSeedThresholdInNoiseUnits    = conf_.getParameter<double>("SeedThreshold");
  theClusterThresholdInNoiseUnits = conf_.getParameter<double>("ClusterThreshold");

  initTiming();
}



PixelThresholdClusterizer::~PixelThresholdClusterizer() 
{
}


//----------------------------------------------------------------------------
//!  Prepare the Clusterizer to work on a particular DetUnit.  Re-init the
//!  size of the panel/plaquette (so update nrows and ncols), 
//----------------------------------------------------------------------------
bool PixelThresholdClusterizer::setup( unsigned int detid )
{
#if 0

  //  Set the bounds for this 
  if(bufferAlreadySet == false){
    theBuffer.setSize(rd.specificDet().specificType().specificTopology().nrows(), 
		   rd.specificDet().specificType().specificTopology().ncolumns());
    bufferAlreadySet = true;
  }

  // Check if topology has changed
  int nrows = readout.specificTopology().nrows();    // rows in x
  int ncols = readout.specificTopology().ncolumns(); // cols in y


  // TO DO: need to convert to the new geometry
  if( nrows>theNumOfRows || ncols>theNumOfCols ) {
    if (infoV) {
      cout << "PixelThresholdClusterizer::setup" << endl;
      cout << readout.specificTopology().pitch().first << " "; // pitch size in x
      cout << readout.specificTopology().pitch().second << " "; // pitch size in y
      cout << readout.specificTopology().nrows() << " " ; // rows in x
      cout << readout.specificTopology().ncolumns() << endl;
      cout << " Thresholds " << thePixelThresholdInNoiseUnits << " " 
	   << theSeedThresholdInNoiseUnits << " " 
	   << theClusterThresholdInNoiseUnits << endl;
    }
    cout << " PixelThresholdClusterizer: pixel buffer redefined " 
	 << nrows << " " << ncols << endl;      
    theNumOfRows = nrows;  // Set new sizes
    theNumOfCols = ncols;
    // Resize the buffer
    theBuffer.setSize(nrows,ncols);  // Modify
  }

#endif
  
  // Get all noise/threshold parameters 
  float noise = 2;  // Get noise in adc units. TO DO: add DB access.

  // TO DO: need to convert to the new geometry
      
  // Convert thresholds to adc units.
  // Single pixel thr. 
  thePixelThreshold = int( noise * thePixelThresholdInNoiseUnits);  
  // To start the cluster search
  theSeedThreshold  = int( noise * theSeedThresholdInNoiseUnits);    
  // Full cluster thr.
  theClusterThreshold = noise * theClusterThresholdInNoiseUnits;    

  return true;   
  // TO DO: is there really a scenario where we could fail? Why not return void?
}



//----------------------------------------------------------------------------
//!  \brief Cluster pixels.
//!  This method operates on a matrix of pixels
//!  and finds the largest contiguous cluster around
//!  each seed pixel.
//----------------------------------------------------------------------------
std::vector<SiPixelCluster>
PixelThresholdClusterizer::clusterizeDetUnit( DigiIterator begin, DigiIterator end,
					      unsigned int detid,
					      const std::vector<float>& noiseVec,
					      const std::vector<short>& badChannels)
{
  TimeMe tm1( *theClustersTimer, false);
  theClusters.clear();

  // Do not bother for empty detectors
  if (begin == end)
    return theClusters;

  //  Set up the clusterization on this DetId.
  if (!setup(detid))
    return theClusters;

  //  Copy PixelDigis to the buffer array; select the seed pixels
  //  on the way, and store them in theSeeds.
  copy_to_buffer(begin, end);

  //  At this point we know the number of seeds on this DetUnit, and thus
  //  also the maximal number of possible clusters, so resize theClusters
  //  in order to make vector<>::push_back() efficient.
  theClusters.reserve( theSeeds.size() );
    
  //  Loop over all seeds.  TO DO: wouldn't using iterators be faster?
  for ( int i = 0; i < theSeeds.size(); i++) {
      
    if ( theBuffer(theSeeds[i]) != 0) {  // Is this seed still valid?
	
      //  Make a cluster around this seed
      SiPixelCluster cluster = make_cluster( theSeeds[i] );
	
      //  Check if the cluster is above threshold  
      // (TO DO: one is signed, other unsigned, gcc warns...)
      if ( cluster.charge() >= theClusterThreshold ) {
	theClusters.push_back( cluster );
      }
    }
  }
  // Erase the seeds.
  theSeeds.clear();

  //  Need to clean unused pixels from the buffer array.
  clear_buffer(begin, end);

  return theClusters;
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
  TimeMe tm1( *theClearTimer, false);
  DigiIterator di = begin;
  for( ; di != end; ++di ) {
    theBuffer.set_adc( di->row(), di->column(), 0 );   // reset pixel adc to 0
  }
}



//----------------------------------------------------------------------------
//! \brief Copy adc counts from PixelDigis into the buffer, identify seeds.
//----------------------------------------------------------------------------
void PixelThresholdClusterizer::copy_to_buffer( DigiIterator begin, DigiIterator end )
{
  TimeMe tm1( *theCopyTimer, false);
  DigiIterator di = begin;
  for( ; di != end; ++di ) {
    int adc = di->adc();
    if ( adc >= thePixelThreshold) {
      int row = di->row();
      int col = di->column();
      theBuffer.set_adc( row, col, adc);
      if ( adc >= theSeedThreshold) { 
	theSeeds.push_back( SiPixelCluster::PixelPos(row,col) );
      }
    }
  }
}



//----------------------------------------------------------------------------
//!  \brief The actual clustering algorithm: group the neighboring pixels around the seed.
//----------------------------------------------------------------------------
SiPixelCluster 
PixelThresholdClusterizer::make_cluster( const SiPixelCluster::PixelPos& pix) 
{
  TimeMe tm1( *theMakeClustTimer, false);

  // Make the cluster
  SiPixelCluster cluster( pix, theBuffer( pix.row(), pix.col()) );

  // TO DO: this C++ issue has probably been sorted out by now (2005)!
#ifndef CMS_STACK_2_ARG
  stack<SiPixelCluster::PixelPos> pixel_stack;
#else
  stack<SiPixelCluster::PixelPos, vector<SiPixelCluster::PixelPos> > pixel_stack;
#endif

  theBuffer.set_adc( pix, 0);
  pixel_stack.push( pix);

  while ( ! pixel_stack.empty()) {
    SiPixelCluster::PixelPos curpix = pixel_stack.top(); pixel_stack.pop();
    for ( int r = curpix.row()-1; r <= curpix.row()+1; r++) {
      for ( int c = curpix.col()-1; c <= curpix.col()+1; c++) {
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
void 
PixelThresholdClusterizer::initTiming() 
{
  TimingReport& tr(*TimingReport::current());

  theSetupTimer =      &tr["PixelClusterizer setup+digi"];
  theClustersTimer =   &tr["PixelClusterizer clusters"];
  theClusterizeTimer = &tr["PixelClusterizer clusterize"];
  theRecHitTimer =     &tr["PixelClusterizer create RecHits"];
  theCopyTimer =       &tr["PixelClusterizer copy to buffer"];
  theClearTimer =      &tr["PixelClusterizer clear buffer"];
  theMakeClustTimer =  &tr["PixelClusterizer make one cluster"];
  theCacheGetTimer =   &tr["PixelClusterizer cache access"];
  theCachePutTimer =   &tr["PixelClusterizer cache fill"];

  static bool detailedTiming 
    = conf_.getParameter<bool>("DetailedTiming");

  if (!detailedTiming) {
    tr.switchOn( "PixelClusterizer setup+digi",false);
    tr.switchOn( "PixelClusterizer clusters",false);
    tr.switchOn( "PixelClusterizer create RecHits",false);
    tr.switchOn( "PixelClusterizer copy to buffer",false);
    tr.switchOn( "PixelClusterizer clear buffer",false);
    tr.switchOn( "PixelClusterizer make one cluster",false);
    tr.switchOn( "PixelClusterizer cache access",false);
    tr.switchOn( "PixelClusterizer cache fill",false);
  }
}
