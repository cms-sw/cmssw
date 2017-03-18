/****************************************************************************
 *
 * This is a part of CTPPS offline software.
 * Authors:
 *   Laurent Forthomme (laurent.forthomme@cern.ch)
 *   Nicola Minafra (nicola.minafra@cern.ch)
 *
 ****************************************************************************/

#include "RecoCTPPS/TotemRPLocal/interface/CTPPSDiamondTrackRecognition.h"

#include <cmath>
#include <cstdio>
#include <algorithm>

//----------------------------------------------------------------------------------------------------

const std::string CTPPSDiamondTrackRecognition::pixelEfficiencyDefaultFunction_ = "(x>[0]-0.5*[1])*(x<[0]+0.5*[1])";

CTPPSDiamondTrackRecognition::CTPPSDiamondTrackRecognition( const edm::ParameterSet& iConfig ) :
  threshold_( iConfig.getParameter<double>( "threshold" ) ),
  thresholdFromMaximum_( iConfig.getParameter<double>( "thresholdFromMaximum" ) ),
  resolution_( iConfig.getParameter<double>( "resolution" ) ),
  sigma_( iConfig.getParameter<double>( "sigma" ) ),
  startFromX_( iConfig.getParameter<double>( "startFromX" ) ),
  stopAtX_( iConfig.getParameter<double>( "stopAtX" ) ),
  pixelEfficiencyFunction_( iConfig.getParameter<std::string>( "pixelEfficiencyFunction" ) ),
  yPosition(0.0), yWidth(0.0), nameCounter(0)
{
  if (sigma_==.0) {
    pixelEfficiencyFunction_ = pixelEfficiencyDefaultFunction_; // simple step function
  }
  hit_f_ = TF1( "hit_TF1_CTPPS", pixelEfficiencyFunction_.c_str(), startFromX_, stopAtX_ );
}

//----------------------------------------------------------------------------------------------------

CTPPSDiamondTrackRecognition::~CTPPSDiamondTrackRecognition()
{}

//----------------------------------------------------------------------------------------------------

void
CTPPSDiamondTrackRecognition::clear()
{
  hitParametersVectorMap_.clear();
  mhMap_.clear();
  nameCounter = 0;
}

//----------------------------------------------------------------------------------------------------

void
CTPPSDiamondTrackRecognition::addHit( const CTPPSDiamondRecHit recHit )
{
  // store hit parameters
  hitParametersVectorMap_[recHit.getOOTIndex()].emplace_back( recHit.getX(), recHit.getXWidth() );
  
  // Check vertical coordinates
  if (yPosition == .0 and yWidth == .0) {
    yPosition = recHit.getY();
    yWidth = recHit.getYWidth();
  }
  
  //Multiple hits in the RP
  if ( recHit.getMultipleHits() ) {
    if (mhMap_.find( recHit.getOOTIndex() ) == mhMap_.end())
      mhMap_[recHit.getOOTIndex()] = 1;
    else
      ++(mhMap_[recHit.getOOTIndex()]);
  }
}

//----------------------------------------------------------------------------------------------------

int
CTPPSDiamondTrackRecognition::produceTracks( edm::DetSet<CTPPSDiamondLocalTrack> &tracks )
{
  int number_of_tracks = 0;
  for ( HitParametersVectorMap::const_iterator ootIt=hitParametersVectorMap_.begin(); ootIt!=hitParametersVectorMap_.end(); ++ootIt ) {
    std::vector<float> hit_profile( ( stopAtX_-startFromX_ )/resolution_, 0. );
    for ( HitParametersVector::const_iterator param_it=ootIt->second.begin(); param_it!=ootIt->second.end(); ++param_it ) {
      hit_f_.SetParameters( param_it->center, param_it->width, sigma_ );
      for ( unsigned int i=0; i<hit_profile.size(); ++i ) {
	hit_profile[i] += hit_f_.Eval( startFromX_ + i*resolution_ );
      }
    }
   
    float maximum = 0.;
    bool below = true; // start below the threshold
    int track_start_n = 0;

    for ( unsigned int i=0; i<hit_profile.size(); ++i ) {
      if ( below && hit_profile[i] >= threshold_ ) { // going above the threshold
	track_start_n = i;
	maximum=0;
	below = false;
      }
      if ( !below ) {
	if ( hit_profile[i] > maximum ) {
	  maximum = hit_profile[i];
	}
	if ( hit_profile[i] < threshold_ ) { // going back below the threshold
	  below = true;

	  // go back and use new threshold
	  const float threshold = maximum - thresholdFromMaximum_;
	  for ( unsigned int j=track_start_n; j<=i; ++j ) {
	    if ( below && hit_profile[j] >= threshold ) { // going above the threshold
	      track_start_n = j;
	      below = false;
	    }
	    if ( !below && hit_profile[j] < threshold ) { // going back below the threshold
	      below = true;

	      //store track
              math::XYZPoint pos0_sigma( ( j-track_start_n )*resolution_*0.5, yWidth * 0.5, 0. );
	      math::XYZPoint pos0( startFromX_ + track_start_n*resolution_ + pos0_sigma.X(), yPosition, 0. );
              int mult_hits = 0;
	      if ( mhMap_.find( ootIt->first ) != mhMap_.end() ) mult_hits = mhMap_[ootIt->first];

              tracks.push_back( CTPPSDiamondLocalTrack( pos0, pos0_sigma, 0., 0., 0., ootIt->first, mult_hits ) );
	      ++number_of_tracks;
	    }
	  }
	}
      }
    }
  }
  
  return number_of_tracks;
}

