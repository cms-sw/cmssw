#include "RecoTracker/DeDx/interface/DeDxTools.h"
#include <vector>
#include "TrackingTools/PatternTools/interface/Trajectory.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripMatchedRecHit2D.h"
#include "DataFormats/TrackerRecHit2D/interface/ProjectedSiStripRecHit2D.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit1D.h"
#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHit.h"
#include "DataFormats/TrackReco/interface/DeDxHit.h"

#include <numeric>
namespace DeDxTools {
using namespace std;
using namespace reco;

                   
  void trajectoryRawHits(const edm::Ref<std::vector<Trajectory> >& trajectory, vector<RawHits>& hits, bool usePixel, bool useStrip)
  {

    //    vector<RawHits> hits;

    const vector<TrajectoryMeasurement> & measurements = trajectory->measurements();
    for(vector<TrajectoryMeasurement>::const_iterator it = measurements.begin(); it!=measurements.end(); it++){

      //FIXME: check that "updated" State is the best one (wrt state in the middle) 
      TrajectoryStateOnSurface trajState=it->updatedState();
      if( !trajState.isValid()) continue;
     
      const TrackingRecHit * recHit=(*it->recHit()).hit();

       LocalVector trackDirection = trajState.localDirection();
       double cosine = trackDirection.z()/trackDirection.mag();
              
       if(const SiStripMatchedRecHit2D* matchedHit=dynamic_cast<const SiStripMatchedRecHit2D*>(recHit)){
	   if(!useStrip) continue;

	   RawHits mono,stereo; 
	   mono.trajectoryMeasurement = &(*it);
	   stereo.trajectoryMeasurement = &(*it);
	   mono.angleCosine = cosine; 
	   stereo.angleCosine = cosine;
	   const std::vector<uint8_t> &  amplitudes = matchedHit->monoHit()->cluster()->amplitudes(); 
	   mono.charge = accumulate(amplitudes.begin(), amplitudes.end(), 0);
           mono.NSaturating =0;
           for(unsigned int i=0;i<amplitudes.size();i++){if(amplitudes[i]>=254)mono.NSaturating++;}
       
	   const std::vector<uint8_t> & amplitudesSt = matchedHit->stereoHit()->cluster()->amplitudes();
	   stereo.charge = accumulate(amplitudesSt.begin(), amplitudesSt.end(), 0);
           stereo.NSaturating =0;
           for(unsigned int i=0;i<amplitudes.size();i++){if(amplitudes[i]>=254)stereo.NSaturating++;}
   
	   mono.detId= matchedHit->monoHit()->geographicalId();
	   stereo.detId= matchedHit->stereoHit()->geographicalId();

	   hits.push_back(mono);
	   hits.push_back(stereo);

        }else if(const ProjectedSiStripRecHit2D* projectedHit=dynamic_cast<const ProjectedSiStripRecHit2D*>(recHit)) {
           if(!useStrip) continue;

           const SiStripRecHit2D* singleHit=&(projectedHit->originalHit());
           RawHits mono;

           mono.trajectoryMeasurement = &(*it);

           mono.angleCosine = cosine; 
           const std::vector<uint8_t> & amplitudes = singleHit->cluster()->amplitudes();
           mono.charge = accumulate(amplitudes.begin(), amplitudes.end(), 0);
           mono.NSaturating =0;
           for(unsigned int i=0;i<amplitudes.size();i++){if(amplitudes[i]>=254)mono.NSaturating++;}

           mono.detId= singleHit->geographicalId();
           hits.push_back(mono);
      
        }else if(const SiStripRecHit2D* singleHit=dynamic_cast<const SiStripRecHit2D*>(recHit)){
           if(!useStrip) continue;

           RawHits mono;
	       
           mono.trajectoryMeasurement = &(*it);

           mono.angleCosine = cosine; 
           const std::vector<uint8_t> & amplitudes = singleHit->cluster()->amplitudes();
           mono.charge = accumulate(amplitudes.begin(), amplitudes.end(), 0);
           mono.NSaturating =0;
           for(unsigned int i=0;i<amplitudes.size();i++){if(amplitudes[i]>=254)mono.NSaturating++;}

           mono.detId= singleHit->geographicalId();
           hits.push_back(mono);

        }else if(const SiStripRecHit1D* single1DHit=dynamic_cast<const SiStripRecHit1D*>(recHit)){
           if(!useStrip) continue;

           RawHits mono;

           mono.trajectoryMeasurement = &(*it);

           mono.angleCosine = cosine;
           const std::vector<uint8_t> & amplitudes = single1DHit->cluster()->amplitudes();
           mono.charge = accumulate(amplitudes.begin(), amplitudes.end(), 0);
           mono.NSaturating =0;
           for(unsigned int i=0;i<amplitudes.size();i++){if(amplitudes[i]>=254)mono.NSaturating++;}

           mono.detId= single1DHit->geographicalId();
           hits.push_back(mono);

      
        }else if(const SiPixelRecHit* pixelHit=dynamic_cast<const SiPixelRecHit*>(recHit)){
           if(!usePixel) continue;

           RawHits pixel;

           pixel.trajectoryMeasurement = &(*it);

           pixel.angleCosine = cosine; 
           pixel.charge = pixelHit->cluster()->charge();
           pixel.NSaturating=-1;
           pixel.detId= pixelHit->geographicalId();
           hits.push_back(pixel);
       }
       
    }
    // return hits;
  }




double genericAverage(const reco::DeDxHitCollection &hits, float expo )
{
 double result=0;
 size_t n = hits.size();
 for(size_t i = 0; i< n; i ++)
 {
    result+=pow(hits[i].charge(),expo); 
 }
 return (n>0)?pow(result/n,1./expo):0.;
}









}
