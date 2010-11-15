#ifndef _FWPFLegoCandidate_H_
#define _FWPFLegoCandidate_H_

#include "TEveLine.h"
#include "TEveStraightLineSet.h"
#include "Rtypes.h"


class TEveTrack;
class FWViewContext;

namespace reco {
  class PFCluster;
}

namespace fireworks
{
   class Context;
}

struct LegoCandidateData
{
    float et;
    float energy;
    float pt;
    float eta;
    float phi;
};

class FWPFLegoCandidate : public TEveStraightLineSet
{
private:
   FWPFLegoCandidate( const FWPFLegoCandidate& );              // Disable default copy constructor
   FWPFLegoCandidate& operator=( const FWPFLegoCandidate& );   // Disable default assignment operator

   // --------------------------- Data Members ---------------------------------
   float m_et, m_energy;


   // ------------------------- Member Functions -------------------------------
   //        float getScale( const FWViewContext *vc, const fireworks::Context &context ) const;

public:
   // -------------------- Constructor(s)/Destructors --------------------------
   FWPFLegoCandidate(){}
   FWPFLegoCandidate( const LegoCandidateData &lc, const FWViewContext *vc, const fireworks::Context &context );
   virtual ~FWPFLegoCandidate(){}

   // ------------------------- Member Functions -------------------------------
   void updateScale( const FWViewContext*, const fireworks::Context& );
};
#endif
