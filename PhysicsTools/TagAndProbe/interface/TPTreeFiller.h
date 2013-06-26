#ifndef PhysicsTools_TagAndProbe_TreeFiller_h
#define PhysicsTools_TagAndProbe_TreeFiller_h

#include "PhysicsTools/TagAndProbe/interface/BaseTreeFiller.h"

namespace tnp {
class TPTreeFiller : public BaseTreeFiller {
    public:
        TPTreeFiller(const edm::ParameterSet config);
        ~TPTreeFiller();

        // We declare 'const' the methods which don't change the configuration
        void init(const edm::Event &iEvent) const ;
        void fill(const reco::CandidateBaseRef &probe, double mass, bool mcTrue=false) const ;

    protected:
        /// extra branch for the mass
        mutable float  mass_;
        /// extra branch for the mc truth
        mutable int32_t mcTrue_;
};
}

#endif
