//
// $Id: EvtPlane.cc,v 1.1 2008/07/20 19:18:24 yilmaz Exp $
//

#include "DataFormats/HeavyIonEvent/interface/EvtPlane.h"

using namespace reco;

EvtPlane::EvtPlane(double planeA, std::string label)
  : 
   angle_(planeA),
   label_(label)
{
  // default constructor
}


EvtPlane::~EvtPlane()
{
}


