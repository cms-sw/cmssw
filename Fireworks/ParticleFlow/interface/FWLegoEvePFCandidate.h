#ifndef Fireworks_ParticleFlow_FWLegoEvePFCandidate_h
#define Fireworks_ParticleFlow_FWLegoEvePFCandidate_h


#include "TEveLine.h"
#include "TEveStraightLineSet.h"
#include "Rtypes.h"

class TEveTrack;
class FWViewContext;

namespace reco {
  class PFCandidate;
}

namespace fireworks
{
   class Context;
}

class FWLegoEvePFCandidate : public TEveStraightLineSet 
{
public:
   FWLegoEvePFCandidate(const reco::PFCandidate& pfc, const FWViewContext*, const fireworks::Context&);
   void updateScale( const FWViewContext*,  const fireworks::Context&);

private:
   float  m_energy;
   float  m_et;
};

#endif
