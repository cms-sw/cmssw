// //
// // $Id: EvtPlane.cc,v 1.3 2009/09/08 10:50:57 edwenger Exp $
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
// $Id: EvtPlane.cc,v 1.3 2009/09/08 10:50:57 edwenger Exp $
//

#include "DataFormats/HeavyIonEvent/interface/EvtPlane.h"

using namespace reco;

EvtPlane::EvtPlane(double planeA,double sumSin, double sumCos, std::string label)
  : 
   angle_(planeA),
   sumSin_(sumSin),
   sumCos_(sumCos),
   label_(label)
{
  // default constructor
}


EvtPlane::~EvtPlane()
{
}


