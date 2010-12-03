#ifndef _FWPFLEGOCANDIDATE_H_
#define _FWPFLEGOCANDIDATE_H_

#include "TEveStraightLineSet.h"

class FWViewContext;

namespace fireworks
{
   class Context;
}

class FWPFLegoCandidate : public TEveStraightLineSet
{
   public:
   // -------------------- Constructor(s)/Destructors --------------------------
      FWPFLegoCandidate( const FWViewContext *vc, const fireworks::Context &context, float et, float energy, float pt, float eta, float phi );
      //virtual ~FWPFLegoCandidate(){}

   // ------------------------- Member Functions -------------------------------
      void updateScale( const FWViewContext *vc, const fireworks::Context& );

   private:
      FWPFLegoCandidate( const FWPFLegoCandidate& );                    // Disable default copy constructor
      const FWPFLegoCandidate& operator=( const FWPFLegoCandidate& );   // Disable default assignment operator

   // --------------------------- Data Members ---------------------------------
      float m_energy;
      float m_et;
      float m_pt;
      float m_eta;
      float m_phi;
};
#endif
