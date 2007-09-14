#include "RecoBTag/TrackCounting/interface/TrackCountingComputer.h"


class NegativeTrackCountingComputer : public TrackCountingComputer
{
 public:
  NegativeTrackCountingComputer(const edm::ParameterSet  & parameters ): TrackCountingComputer(parameters)
  {
  }
 
  float discriminator(const reco::BaseTagInfo & ti) const 
   {
     const reco::TrackIPTagInfo * tkip = dynamic_cast<const reco::TrackIPTagInfo *>(&ti);
      if(tkip!=0)  {
          std::multiset<float> significances = orderedSignificances(*tkip);
          std::multiset<float>::iterator nth=significances.begin();
          for(int i=0;i<m_nthTrack-1 && nth!=significances.end();i++) nth++;  
          if(nth!=significances.end()) return -(*nth); else return -100.;
        }
        else 
          {
            //FIXME: report some error? 
            return -100. ;   
          }
   }

};
