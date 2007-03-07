#include "PhysicsTools/JetMCUtils/interface/CandMatchMapMany.h"

namespace {
  namespace {
    reco::CandMatchMapMany cmm2;
    edm::Wrapper<reco::CandMatchMapMany> wcmm2;
    edm::helpers::KeyVal< edm::RefProd<edm::OwnVector<reco::Candidate,edm::ClonePolicy<reco::Candidate> > > ,
                          edm::RefProd<edm::OwnVector<reco::Candidate,edm::ClonePolicy<reco::Candidate> > > 
                        > kv2;
 }
}
