// //
// // $Id: EvtPlane.cc,v 1.2 2009/08/17 18:08:14 yilmaz Exp $
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
// $Id: EvtPlane.cc,v 1.2 2009/08/17 18:08:14 yilmaz Exp $
//

#include "DataFormats/HeavyIonEvent/interface/EvtPlane.h"

using namespace reco;

EvtPlane::EvtPlane(double planeA,double sumSin, double sumCos, std::string label)
  : 
   angle_(planeA),
   label_(label),
   sumSin_(sumSin),
   sumCos_(sumCos)
{
  // default constructor
}


EvtPlane::~EvtPlane()
{
}


