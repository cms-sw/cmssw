#include "DataFormats/HeavyIonEvent/interface/EvtPlane.h"

using namespace reco;
using namespace std;
EvtPlane::EvtPlane(int epindx, int level, double planeA,double sumSin, double sumCos, double sumw, double sumw2, double sumPtOrEt, double sumPtOrEt2, uint mult)
  : 
   indx_(epindx),
   sumw_(sumw),
   sumw2_(sumw2),
   sumPtOrEt_(sumPtOrEt),
   sumPtOrEt2_(sumPtOrEt2),
   mult_(mult)
{
  for ( int i = 0; i < 4; ++i ) {
    angle_[i] = sumSin_[i] = sumCos_[i] = -10;
  }
  angle_[level] = planeA;
  sumSin_[level] = sumSin;
  sumCos_[level] = sumCos;
  // default constructor
}
void EvtPlane::addLevel(int level, double ang, double sumsin, double sumcos) {
  angle_[level] = ang;
  sumSin_[level] = sumsin;
  sumCos_[level] = sumcos;
}

EvtPlane::~EvtPlane()
{
}


