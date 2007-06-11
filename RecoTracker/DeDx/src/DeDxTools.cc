#include "RecoTracker/DeDx/interface/DeDxTools.h"
#include <vector>
#include "TrackingTools/PatternTools/interface/Trajectory.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripMatchedRecHit2D.h"
#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHit.h"
#include "DataFormats/TrackReco/interface/DeDxHit.h"

#include <numeric>
namespace DeDxTools {
using namespace std;
using namespace reco;


vector<RawHits> trajectoryRawHits(const Trajectory & trajectory)
  {

    vector<RawHits> hits;

    vector<TrajectoryMeasurement> measurements = trajectory.measurements();
    for(vector<TrajectoryMeasurement>::const_iterator it = measurements.begin(); it!=measurements.end(); it++){

      //FIXME: check that "updated" State is the best one (wrt state in the middle) 
      TrajectoryStateOnSurface trajState=it->updatedState();
      if( !trajState.isValid()) continue;     
      const TrackingRecHit * recHit=(*it->recHit()).hit();

       LocalVector trackDirection = trajState.localDirection();
       double cosine = trackDirection.z()/trackDirection.mag();
      
      if(const SiStripMatchedRecHit2D* matchedHit=dynamic_cast<const SiStripMatchedRecHit2D*>(recHit))
      {
         RawHits mono,stereo;
	 mono.angleCosine = cosine; 
	 stereo.angleCosine = cosine;
         const std::vector<uint16_t> &  amplitudes = matchedHit->monoHit()->cluster()->amplitudes(); 
         mono.charge = accumulate(amplitudes.begin(), amplitudes.end(), 0);
       
         const std::vector<uint16_t> & amplitudesSt = matchedHit->stereoHit()->cluster()->amplitudes();
         stereo.charge = accumulate(amplitudesSt.begin(), amplitudesSt.end(), 0);
	 
         mono.detId= matchedHit->monoHit()->geographicalId();
         stereo.detId= matchedHit->stereoHit()->geographicalId();

         hits.push_back(mono);
         hits.push_back(stereo);

      } 
      else {
        if(const SiStripRecHit2D* singleHit=dynamic_cast<const SiStripRecHit2D*>(recHit))
         {
         RawHits mono;
	 mono.angleCosine = cosine; 
         const std::vector<uint16_t> & amplitudes = singleHit->cluster()->amplitudes();
         mono.charge = accumulate(amplitudes.begin(), amplitudes.end(), 0);
         mono.detId= singleHit->geographicalId();
         hits.push_back(mono);
      
        }
      else 
        if(const SiPixelRecHit* pixelHit=dynamic_cast<const SiPixelRecHit*>(recHit))
      {
         RawHits pixel;
	 pixel.angleCosine = cosine; 
         pixel.charge = pixelHit->cluster()->charge();;
         pixel.detId= pixelHit->geographicalId();
         hits.push_back(pixel);
      }
    }
   }
   return hits;
}



double genericAverage(const reco::DeDxHitCollection &hits, float expo )
{
 double result=0;
 for(size_t i = 0; i< hits.size(); i ++)
 {
    result+=pow(hits[i].charge(),expo); 
 }
 return pow(result,1./expo);
}


}
