#ifndef TopObjects_StGenEvent_h
#define TopObjects_StGenEvent_h
#include "DataFormats/Candidate/interface/Candidate.h"
#include <vector>

using namespace reco;
using namespace std;

class StGenEvent
{

   public:
      StGenEvent();
      StGenEvent(int,vector<const Candidate *>);
      virtual ~StGenEvent();
      
      int decay() const {return decay_;};
      vector<Candidate *> particles() const {return particles_;};

   private:
      int decay_;
      vector<Candidate *> particles_;


};

#endif
