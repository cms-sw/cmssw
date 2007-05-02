#ifndef TopObjects_TtGenEvent_h
#define TopObjects_TtGenEvent_h
#include "DataFormats/Candidate/interface/Candidate.h"
#include <vector>

using namespace reco;
using namespace std;

class TtGenEvent
{

   public:
      TtGenEvent();
      TtGenEvent(int,vector<const Candidate *>);
      virtual ~TtGenEvent();
      
      int decay() const {return decay_;};
      vector<Candidate *> particles() const {return particles_;};

   private:
      int decay_;
      vector<Candidate *> particles_;


};

#endif
