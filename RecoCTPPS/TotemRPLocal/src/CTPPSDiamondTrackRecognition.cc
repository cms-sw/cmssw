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

#define LOWER_HIT_LIMIT_MM -1e3
#define HIGHER_HIT_LIMIT_MM 1e3
#define PAD_FUNCTION "(1/(1+exp(-(x-[0])/[2])))*(1/(1+exp((x-[0]-[1])/[2])))"

using namespace std;
using namespace edm;

//----------------------------------------------------------------------------------------------------

CTPPSDiamondTrackRecognition::CTPPSDiamondTrackRecognition() :
  threshold_( 2.0 ), resolution_mm_( 0.01 ), sigma_( 0.0 )
{}

CTPPSDiamondTrackRecognition::CTPPSDiamondTrackRecognition( const edm::ParameterSet& iConfig ) :
  threshold_( iConfig.getParameter<double>( "threshold" ) ),
  resolution_mm_( iConfig.getParameter<double>( "resolution" ) ),
  sigma_( iConfig.getParameter<double>( "sigma" ) )
{
    if (sigma_==.0) sigma_=1.0e-10;
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
  TF1 hit_f(f_name.c_str(), PAD_FUNCTION, LOWER_HIT_LIMIT_MM, HIGHER_HIT_LIMIT_MM);
  
  double center = recHit.getX();
  double width = recHit.getXWidth();
  hit_f.SetParameters(center, width, sigma_);
  
  hit_function_v.push_back(hit_f);
}

//----------------------------------------------------------------------------------------------------

void CTPPSDiamondTrackRecognition::produceTracks(DetSet<CTPPSDiamondLocalTrack> &tracks) {
  vector<double> hit_profile((HIGHER_HIT_LIMIT_MM - LOWER_HIT_LIMIT_MM) / resolution_mm_, .0);
  for (unsigned int i=0; i<hit_profile.size(); ++i) {
    for (vector<TF1>::const_iterator fun_it=hit_function_v.begin(); fun_it!=hit_function_v.end(); ++fun_it) { 
      hit_profile[i] += fun_it->Eval(LOWER_HIT_LIMIT_MM + i*resolution_mm_);
    }
  }
  
  bool below = true;	// below the threhold
  int track_start_n = 0;
  for (unsigned int i=0; i<hit_profile.size(); ++i) {
    if (below && hit_profile[i] >= threshold_) {	// going above the threhold
      track_start_n = i;
      below = false;
    }
    if (!below && hit_profile[i] < threshold_) {	// going back below the threhold
      below = true;
      //store track
      CTPPSDiamondLocalTrack track;
      track.setX0Sigma( (i-track_start_n)*resolution_mm_ );
      track.setX0( LOWER_HIT_LIMIT_MM + track_start_n*resolution_mm_ + track.getX0Sigma()/2);
      tracks.push_back(track);
    }
  }
  
}

