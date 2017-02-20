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

#define PAD_DEFAULT_FUNCTION "(x>[0]-0.5*[1])*(x<[0]+0.5*[1])"

//----------------------------------------------------------------------------------------------------

CTPPSDiamondTrackRecognition::CTPPSDiamondTrackRecognition() :
  threshold_( 1.5 ), thresholdFromMaximum_( 0.5 ),
  resolution_( 0.025 ), sigma_( 0.0 ),
  startFromX_( -2 ), stopAtX_( 38 ),
  pixelEfficiencyFunction_( PAD_DEFAULT_FUNCTION ),
  yPosition(0.0), yWidth(0.0), nameCounter(0)
{}

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
    pixelEfficiencyFunction_ = PAD_DEFAULT_FUNCTION; // simple step function
  }
}

//----------------------------------------------------------------------------------------------------

CTPPSDiamondTrackRecognition::~CTPPSDiamondTrackRecognition()
{}

//----------------------------------------------------------------------------------------------------

void
CTPPSDiamondTrackRecognition::clear()
{
  hitFunctionVectorMap_.clear();
}

//----------------------------------------------------------------------------------------------------

void
CTPPSDiamondTrackRecognition::addHit( const CTPPSDiamondRecHit recHit )
{
  std::string f_name( "hit_f_" );
  f_name.append( std::to_string( ++nameCounter ) );

  // define the hit function
  TF1 hit_f( f_name.c_str(), pixelEfficiencyFunction_.c_str(), startFromX_, stopAtX_ );
  hit_f.SetParameters( recHit.getX(), recHit.getXWidth(), sigma_ );
  hitFunctionVectorMap_[recHit.getOOTIndex()].push_back( hit_f );
  
  // Check vertical coordinates
  if (yPosition == .0 and yWidth == .0) {
    yPosition = recHit.getY();
    yWidth = recHit.getYWidth();
  }
  
}

//----------------------------------------------------------------------------------------------------

int
CTPPSDiamondTrackRecognition::produceTracks( edm::DetSet<CTPPSDiamondLocalTrack> &tracks )
{
  int number_of_tracks = 0;
  for ( hitFunctionVectorMap_t::const_iterator ootIt=hitFunctionVectorMap_.begin(); ootIt!=hitFunctionVectorMap_.end(); ++ootIt ) {
    std::vector<float> hit_profile( ( stopAtX_-startFromX_ )/resolution_, 0. );
    for ( unsigned int i=0; i<hit_profile.size(); ++i ) {
      for ( hitFunctionVector_t::const_iterator fun_it=ootIt->second.begin(); fun_it!=ootIt->second.end(); ++fun_it ) { 
	hit_profile[i] += fun_it->Eval( startFromX_ + i*resolution_ );
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
	      CTPPSDiamondLocalTrack track;
	      track.setX0Sigma( ( j-track_start_n )*resolution_*0.5 );
	      track.setX0( startFromX_ + track_start_n*resolution_ + track.getX0Sigma() );
	      track.setY0( yPosition );
	      track.setY0Sigma( yWidth * 0.5 );
	      track.setOOTIndex( ootIt->first );
	      tracks.push_back( track );
	      ++number_of_tracks;
	    }
	  }
	}
      }
    }
  }
  
  return number_of_tracks;
}

