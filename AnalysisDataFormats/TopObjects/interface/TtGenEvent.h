#ifndef TopObjects_TtGenEvent_h
#define TopObjects_TtGenEvent_h
#include "DataFormats/Candidate/interface/Candidate.h"
#include <vector>

class TtGenEvent
{

   public:
      TtGenEvent();
      TtGenEvent(int,std::vector<const reco::Candidate *>);
      virtual ~TtGenEvent();
      
      int decay() const {return decay_;};
      std::vector<reco::Candidate *> particles() const {return particles_;};

   private:
      int decay_;
      std::vector<reco::Candidate *> particles_;


};

#endif
