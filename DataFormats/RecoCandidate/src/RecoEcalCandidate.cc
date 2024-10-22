#include "DataFormats/RecoCandidate/interface/RecoEcalCandidate.h"

using namespace reco;

RecoEcalCandidate::~RecoEcalCandidate() {}

RecoEcalCandidate *RecoEcalCandidate::clone() const { return new RecoEcalCandidate(*this); }

SuperClusterRef RecoEcalCandidate::superCluster() const { return superCluster_; }

bool RecoEcalCandidate::overlap(const Candidate &c) const {
  const RecoCandidate *o = dynamic_cast<const RecoCandidate *>(&c);
  return (o != nullptr && checkOverlap(superCluster(), o->superCluster()));
}
