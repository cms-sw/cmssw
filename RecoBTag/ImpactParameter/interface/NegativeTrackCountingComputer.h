#include "RecoBTag/ImpactParameter/interface/TrackCountingComputer.h"


class NegativeTrackCountingComputer : public TrackCountingComputer
{
 public:
  NegativeTrackCountingComputer(const edm::ParameterSet & parameters ) : TrackCountingComputer(parameters)
  {
  }
 
  float discriminator(const TagInfoHelper & ti) const 
   {
     const reco::TrackIPTagInfo & tkip = ti.get<reco::TrackIPTagInfo>();
     std::multiset<float> significances = orderedSignificances(tkip);
     std::multiset<float>::iterator nth=significances.begin();
     for(int i=0;i<m_nthTrack-1 && nth!=significances.end();i++) nth++;  
     if(nth!=significances.end()) return -(*nth); else return -100.;
   }

};
