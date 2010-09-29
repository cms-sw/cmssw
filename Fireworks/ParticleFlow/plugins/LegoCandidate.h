#ifndef _LegoCandidate_H_
#define _LegoCandidate_H_

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

class LegoCandidate : public TEveStraightLineSet
{
    private:
        LegoCandidate( const LegoCandidate& );              // Disable default copy constructor
        LegoCandidate& operator=( const LegoCandidate& );   // Disable default assignment operator

        // --------------------------- Data Members ---------------------------------
        float m_et, m_energy;

        // ------------------------- Member Functions -------------------------------
        float getScale( const FWViewContext *vc, const fireworks::Context &context ) const;

    public:
        // -------------------- Constructor(s)/Destructors --------------------------
        LegoCandidate(){}
        LegoCandidate( float eta, float phi, float energy, float et, float pt, const FWViewContext *vc, const fireworks::Context &context );
        virtual ~LegoCandidate(){}

        // ------------------------- Member Functions -------------------------------
        void updateScale( const FWViewContext*, const fireworks::Context& );
};
#endif
