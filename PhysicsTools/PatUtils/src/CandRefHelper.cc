#include "PhysicsTools/PatUtils/interface/RefHelper.h"
#include "DataFormats/Candidate/interface/Candidate.h"
/* Just to test that it compiles */
namespace { namespace { 
    ::reco::helper::RefHelper< ::reco::Candidate > crh( edm::ValueMap< ::edm::RefToBase< ::reco::Candidate > >() );
} }
