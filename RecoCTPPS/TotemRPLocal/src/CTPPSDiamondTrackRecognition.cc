/****************************************************************************
 *
 * This is a part of CTPPS offline software.
 * Authors:
 *   Laurent Forthomme (laurent.forthomme@cern.ch)
 *   Nicola Minafra (nicola.minafra@cern.ch)
 *
 ****************************************************************************/

#include "RecoCTPPS/TotemRPLocal/interface/CTPPSDiamondTrackRecognition.h"

#include <map>
#include <cmath>
#include <cstdio>
#include <algorithm>

#define PAD_DEFAULT_FUNCTION "(x>[0]-0.5*[1])*(x<[0]+0.5*[1])"

using namespace std;
using namespace edm;

//----------------------------------------------------------------------------------------------------

CTPPSDiamondTrackRecognition::CTPPSDiamondTrackRecognition() :
  threshold_( 1.5 ), thresholdFromMaximum_(0.5), resolution_( 0.025), sigma_( 0.0 ), startFromX_(-2), stopAtX_(38), pixelEfficiencyFunction_(PAD_DEFAULT_FUNCTION)
{}

CTPPSDiamondTrackRecognition::CTPPSDiamondTrackRecognition( const edm::ParameterSet& iConfig ) :
  threshold_( iConfig.getParameter<double>( "threshold" ) ),
  thresholdFromMaximum_( iConfig.getParameter<double>( "thresholdFromMaximum" ) ),
  resolution_( iConfig.getParameter<double>( "resolution" ) ),
  sigma_( iConfig.getParameter<double>( "sigma" ) ),
  startFromX_( iConfig.getParameter<double>( "startFromX" ) ),
  stopAtX_( iConfig.getParameter<double>( "stopAtX" ) ),
  pixelEfficiencyFunction_( iConfig.getParameter<std::string>( "pixelEfficiencyFunction" ) )
{
    if (sigma_==.0) pixelEfficiencyFunction_="(x>[0]-0.5*[1])*(x<[0]+0.5*[1])";	// Simple step function
}

//----------------------------------------------------------------------------------------------------

CTPPSDiamondTrackRecognition::~CTPPSDiamondTrackRecognition() {
}

//----------------------------------------------------------------------------------------------------

void CTPPSDiamondTrackRecognition::clear() {
  hit_function_v.clear();
}

//----------------------------------------------------------------------------------------------------

void CTPPSDiamondTrackRecognition::addHit(const CTPPSDiamondRecHit recHit) {
  string f_name("hit_f_");
  f_name.append(to_string(hit_function_v.size() + 1));
  TF1 hit_f(f_name.c_str(), pixelEfficiencyFunction_.c_str(), startFromX_, stopAtX_);
  
  hit_f.SetParameters(recHit.getX(), recHit.getXWidth(), sigma_);
  
  hit_function_v.push_back(hit_f);
}

//----------------------------------------------------------------------------------------------------

int CTPPSDiamondTrackRecognition::produceTracks(DetSet<CTPPSDiamondLocalTrack> &tracks) {
  vector<double> hit_profile((stopAtX_ - startFromX_) / resolution_, .0);
  for (unsigned int i=0; i<hit_profile.size(); ++i) {
    for (vector<TF1>::const_iterator fun_it=hit_function_v.begin(); fun_it!=hit_function_v.end(); ++fun_it) { 
      hit_profile[i] += fun_it->Eval(startFromX_ + i*resolution_);
    }
  }
  
  int number_of_tracks=0;
  double maximum=0;
  bool below = true;	// below the threshold
  int track_start_n = 0;
  for (unsigned int i=0; i<hit_profile.size(); ++i) {
    if (below && hit_profile[i] >= threshold_) {	// going above the threshold
      track_start_n = i;
      maximum=0;
      below = false;
    }
    if (!below) {
      if (hit_profile[i] > maximum) maximum = hit_profile[i];
      if (hit_profile[i] < threshold_) {		// going back below the threshold
	below = true;
	
	//go back and use new threshold
	double threshold = maximum - thresholdFromMaximum_;
	for (unsigned int j=track_start_n; j<=i; ++j) {
	  if (below && hit_profile[j] >= threshold) {	// going above the threshold
	    track_start_n = j;
	    below = false;
	  }
	  if (!below && hit_profile[j] < threshold) {	// going back below the threshold
	    below = true;
	    //store track
	    CTPPSDiamondLocalTrack track;
	    track.setX0Sigma( (j-track_start_n)*resolution_ );
	    track.setX0( startFromX_ + track_start_n*resolution_ + track.getX0Sigma()/2);
	    tracks.push_back(track);
	    ++number_of_tracks;
	  }
	}
      }
    }
  }
  
  return number_of_tracks;
}

