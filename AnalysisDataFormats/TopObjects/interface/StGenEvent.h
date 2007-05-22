#ifndef TopObjects_StGenEvent_h
#define TopObjects_StGenEvent_h
#include "DataFormats/Candidate/interface/Candidate.h"
#include <vector>


class StGenEvent
{

   public:
      StGenEvent();
      StGenEvent(int,std::vector<const reco::Candidate *>);
      virtual ~StGenEvent();
      
      int decay() const {return decay_;};
      std::vector<reco::Candidate *> particles() const {return particles_;};

   private:
      int decay_;
      std::vector<reco::Candidate *> particles_;


};

#endif
