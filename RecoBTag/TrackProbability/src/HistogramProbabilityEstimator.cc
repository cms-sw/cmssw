
#include "RecoBTag/TrackProbability/interface/HistogramProbabilityEstimator.h"
#include "RecoBTag/TrackProbability/interface/TrackClassFilter.h"
#include "CondFormats/BTauObjects/interface/CalibratedHistogram.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <algorithm>
using namespace reco;
using namespace std;

pair<bool,double> HistogramProbabilityEstimator::probability(bool quality, int ipType,float significance, const Track& track, const Jet & jet, const Vertex & vertex) const 
{
 
  TrackClassFilter::Input input(quality, track, jet, vertex);
 
  double trackProbability=0;
 
     double absSignificance= fabs(significance);
  
       
      const CalibratedHistogram * probabilityHistogram  = nullptr;
      vector<TrackProbabilityCalibration::Entry>::const_iterator found;
      vector<TrackProbabilityCalibration::Entry>::const_iterator it;
      vector<TrackProbabilityCalibration::Entry>::const_iterator it_end;
      if(ipType==0) {it=m_calibration3D->data.begin(); it_end=m_calibration3D->data.end(); }
      else if(ipType==1) {it=m_calibration2D->data.begin(); it_end=m_calibration2D->data.end(); }
      else return pair<bool,double>(probabilityHistogram ,trackProbability);

      found = std::find_if(it,it_end,[&input](auto const& c) {return TrackClassFilter()(input,c);});
      if(found!=it_end) probabilityHistogram = &found->histogram;
     if(!probabilityHistogram)
       {
//	  edm::LogWarning ("TrackProbability|HistogramMissing") << " PDF Histogram not found for this track" ;
       } else {
          trackProbability = 1. - probabilityHistogram->normalizedIntegral(absSignificance);
       } 	     
      if(absSignificance!=0)  
        trackProbability*=significance/absSignificance;   //Return a "signed" probability
    
    return pair<bool,double>(probabilityHistogram ,trackProbability);

}











