
#include "RecoBTag/TrackProbability/interface/HistogramProbabilityEstimator.h"
#include "CondFormats/BTagObjects/interface/CalibratedHistogram.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

using namespace reco;
using namespace std;

pair<bool,double> HistogramProbabilityEstimator::probability(int ipType,float significance, const Track& track, const Jet & jet, const Vertex & vertex) const 
{
 
  TrackClassFilterInput input(track,jet,vertex);
 
  double trackProbability=0;
  const CalibratedHistogram * probabilityHistogram  = 0;
 
     double absSignificance= fabs(significance);
  
      if(ipType == 1) probabilityHistogram = m_calibrationTransverse->getCalibData(input);
      if(ipType == 0) probabilityHistogram = m_calibration3D->getCalibData(input);
        
     if(!probabilityHistogram)
       {
	  edm::LogWarning ("TrackProbability|HistogramMissing") << " PDF Histogram not found for this track" ;
       } else {
          trackProbability = 1. - probabilityHistogram->normalizedIntegral(absSignificance);
       } 	     
      if(absSignificance!=0)  
        trackProbability*=significance/absSignificance;   //Return a "signed" probability
    
    return pair<bool,double>(probabilityHistogram ,trackProbability);

}











