#ifndef Fireworks_Calo_scaledMarker_h
#define Fireworks_Calo_scaledMarker_h
// -*- C++ -*-
//
// Package:     Calo
// Class  :     scaledMarker
// 
/**\class scaledMarker scaledMarker.h Fireworks/Calo/interface/scaledMarker.h

 Description: [one line class summary]

 Usage:
    <usage>

*/
//
// Original Author:  Alja Mrak-Tadel
//         Created:  Fri Oct 22 15:53:19 CEST 2010
// $Id: scaleMarker.h,v 1.1 2010/10/22 14:34:44 amraktad Exp $
//

class TEveScalableStraightLineSet;
class FWViewContext;

namespace fireworks
{
struct scaleMarker {
   scaleMarker(TEveScalableStraightLineSet* ls, float et, float e, const FWViewContext* vc):
      m_ls(ls),
      m_et(et),
      m_energy(e),
      m_vc(vc) 
   {
   };

   virtual ~scaleMarker() {}

   TEveScalableStraightLineSet* m_ls;
   float m_et;
   float m_energy;
   const FWViewContext* m_vc;
};
}

#endif
