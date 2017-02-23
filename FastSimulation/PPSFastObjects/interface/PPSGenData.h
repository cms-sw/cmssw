#ifndef PPSGenData_h
#define PPSGenData_h
#include <vector>
#include "FastSimulation/PPSFastObjects/interface/PPSGenParticles.h"
#include "TObject.h"
#include "TLorentzVector.h"

class PPSGenData: public TObject {
  public:
      PPSGenData();
      virtual ~PPSGenData() {};
      int addParticle(const TLorentzVector& p,double _t,double _xi)
                      {genParticles.push_back(PPSGenParticle(p,_t,_xi));return genParticles.size()-1;};
      void clear() {genParticles.clear();};
  public:
  //private:
      PPSGenParticles genParticles;
ClassDef(PPSGenData,1);
};
#endif
