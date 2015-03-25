// //
// // $Id: EvtPlane.cc,v 1.4 2009/09/08 12:33:12 edwenger Exp $
// //
// 
// #include "DataFormats/HeavyIonEvent/interface/EvtPlane.h"
// 
// using namespace reco;
// 
// EvtPlane::EvtPlane(double planeA, std::string label)
//   : 
//    angle_(planeA),
//    label_(label)
// {
//   // default constructor
// }
// 
// 
// EvtPlane::~EvtPlane()
// {
// }
// 
// 


//
// $Id: EvtPlane.cc,v 1.4 2009/09/08 12:33:12 edwenger Exp $
//

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
  angle_[level] = planeA;
  sumSin_[level] = sumSin;
  sumCos_[level] = sumCos;
  if(level<2) {
    angle_[2] = planeA;
    sumSin_[2] = sumSin;
    sumCos_[2] = sumCos;
  }
  // default constructor
}
void EvtPlane::AddLevel(int level, double ang, double sumsin, double sumcos) {
  angle_[level] = ang;
  sumSin_[level] = sumsin;
  sumCos_[level] = sumcos;
}

EvtPlane::~EvtPlane()
{
}


