// //
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


