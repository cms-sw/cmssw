#ifndef _FWPFLEGOCANDIDATE_H_
#define _FWPFLEGOCANDIDATE_H_

// -*- C++ -*-
//
// Package:     ParticleFlow
// Class  :     FWPFLegoCandidate
// 
// Implementation:
//     <Notes on implementation>
//
// Original Author:  Simon Harris
//

// System include files
#include "TEveStraightLineSet.h"

// Forward declarations
class FWViewContext;

namespace fireworks
{
   class Context;
}

//-----------------------------------------------------------------------------
// FWPFLegoCandidate
//-----------------------------------------------------------------------------
class FWPFLegoCandidate : public TEveStraightLineSet
{
   public:
   // ---------------- Constructor(s)/Destructor ----------------------
      FWPFLegoCandidate( const FWViewContext *vc, const fireworks::Context &context, float et, float energy, float pt, float eta, float phi );
      FWPFLegoCandidate(){}
      virtual ~FWPFLegoCandidate(){}

   // --------------------- Member Functions --------------------------
      void updateScale( const FWViewContext *vc, const fireworks::Context& );

   private:
      FWPFLegoCandidate( const FWPFLegoCandidate& );                    // Disable default copy constructor
      const FWPFLegoCandidate& operator=( const FWPFLegoCandidate& );   // Disable default assignment operator

   // ----------------------- Data Members ----------------------------
      float m_energy;
      float m_et;
      float m_pt;
      float m_eta;
      float m_phi;
};
#endif
//=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_
