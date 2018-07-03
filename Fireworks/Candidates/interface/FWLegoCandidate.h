#ifndef _FWLEGOCANDIDATE_H_
#define _FWLEGOCANDIDATE_H_

// -*- C++ -*-
//
// Package:     Candidates
// Class  :     FWLegoCandidate
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
// FWLegoCandidate
//-----------------------------------------------------------------------------
class FWLegoCandidate : public TEveStraightLineSet
{
   public:
   // ---------------- Constructor(s)/Destructor ----------------------
      FWLegoCandidate( const FWViewContext *vc, const fireworks::Context &context, 
            float et, float energy, float pt, float eta, float phi );
      FWLegoCandidate(){}
      ~FWLegoCandidate() override{}

   // --------------------- Member Functions --------------------------
      void updateScale( const FWViewContext *vc, const fireworks::Context& );

   private:
      FWLegoCandidate( const FWLegoCandidate& ) = delete;                    // Disable default copy constructor
      const FWLegoCandidate& operator=( const FWLegoCandidate& ) = delete;   // Disable default assignment operator

   // ----------------------- Data Members ----------------------------
      float m_energy;
      float m_et;
      float m_pt;
      float m_eta;
      float m_phi;
};
#endif
//=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_
