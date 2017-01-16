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

//#define CTPPS_DEBUG 1

using namespace std;
using namespace edm;


//----------------------------------------------------------------------------------------------------

CTPPSDiamondTrackRecognition::CTPPSDiamondTrackRecognition(const double threshold, const double sigma, const double resolution_mm) :
  resolution_mm(resolution_mm), geometry(NULL), hit_function("hit_function","PAD_FUNCTION", LOWER_HIT_LIMIT_MM, HIGHER_HIT_LIMIT_MM) {
    if (sigma==.0) sigma=1e-10;
}

//----------------------------------------------------------------------------------------------------

CTPPSDiamondTrackRecognition::~CTPPSDiamondTrackRecognition() {
}


//----------------------------------------------------------------------------------------------------

CTPPSDiamondTrackRecognition::clear() {
  hit_function_v.clear();
}



//----------------------------------------------------------------------------------------------------

void CTPPSDiamondTrackRecognition::addHit(const CTPPSDiamondRecHit recHit) {
  string f_name("hit_f_");
  f_name.append(to_string(hit_function_v.size() + 1));
  TF1 hit_f(f_name.c_str(), PAD_FUNCTION, LOWER_HIT_LIMIT_MM, HIGHER_HIT_LIMIT_MM);
  
  double center = recHit.getX();	//TODO retrive using geometry
  double width = recHit.getXWidth();	//TODO retrive using geometry  
  hit_f.SetParameters(center, width, sigma);
  
  hit_function_v.push_back(hit_f);
  
}


//----------------------------------------------------------------------------------------------------

void CTPPSDiamondTrackRecognition::produceTracks(DetSet<CTPPSDiamondLocalTrack> &tracks) {
  vector<double> hit_profile((HIGHER_HIT_LIMIT_MM - LOWER_HIT_LIMIT_MM) / resolution_mm, .0);
  for (int i=0; i<hit_profile.size(); ++i) {
    for (vector<TF1>::const_iterator fun=hit_function_v.begin(); fun!=hit_function_v.end(); ++fun) { 
      hit_profile[i] += fun_it->Eval(LOWER_HIT_LIMIT_MM + i*resolution_mm);
    }
  }
  
  bool below = true;	// below the threhold
  int track_start_n = 0;
  int track_width_n = 0;
  for (int i=0; i<hit_profile.size(); ++i) {
    if (below && hit_profile[i] >= threhold) {	// going above the threhold
      track_start_n = i;
      below = false;
    }
    if (!below && hit_profile[i] < threhold) {	// going back below the threhold
      below = true;
      //store track
      CTPPSDiamondLocalTrack track;
      track.setX0Sigma( (i-track_start_n)*resolution_mm );
      track.setX0( LOWER_HIT_LIMIT_MM + track_start_n*resolution_mm + track.getX0Sigma()/2);
      tracks.push_back(track);
    }
  }
  
}




