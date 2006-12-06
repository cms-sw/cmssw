
#include "RecoBTag/TrackProbability/interface/HistogramProbabilityEstimator.h"
#include "RecoBTag/XMLCalibration/interface/CalibratedHistogram.h"

using namespace reco;
pair<bool,double> HistogramProbabilityEstimator::probability(int ipType,float significance, const Track& track, const Jet & jet, const Vertex & vertex) const 
{
 
  TrackClassFilterInput input(track,jet,vertex);
 
  double trackProbability=0;
  const CalibratedHistogram * probabilityHistogram  = 0;
 
//     cout << "Significance: " <<  significance << " tr " << aRecTrack.momentumAtVertex().mag() <<  endl;   
     double absSignificance= fabs(significance);
//     double sign=(absSignficince!=0) ? significance/absSignificance : 1 ;
  
      if(ipType == 1) probabilityHistogram = m_calibrationTransverse->fetch(input);
      if(ipType == 0) probabilityHistogram = m_calibration3D->fetch(input);
        
     if(!probabilityHistogram)
       {
          cout << "Histogram not found !! "<< endl;
       } else {
//          cout << "H:"<< probabilityHistogram << " " << probabilityHistogram->binContent(2) << endl;
          trackProbability = 1. - probabilityHistogram->normalizedIntegral(absSignificance);
       } 	     
      if(absSignificance!=0)  
        trackProbability*=significance/absSignificance;   //Return a "signed" probability
    
    return pair<bool,double>(probabilityHistogram ,trackProbability);

}











