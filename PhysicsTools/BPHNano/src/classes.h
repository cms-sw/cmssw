#ifndef BPHNANO_CLASSES_H
#define BPHNANO_CLASSES_H

#include <vector>

#include "DataFormats/Common/interface/Association.h"
#include "DataFormats/Common/interface/Wrapper.h"
#include "DataFormats/PatCandidates/interface/CompositeCandidate.h"
#include "PhysicsTools/BPHNano/plugins/KinVtxFitter.h"
#include "TrackingTools/TransientTrack/interface/TransientTrack.h"

namespace {
struct dictionary {
  std::vector<reco::TransientTrack> ttv;
  edm::Wrapper<std::vector<reco::TransientTrack> > wttv;
  edm::Wrapper<std::vector<KinVtxFitter> > wkv;
};
}  // namespace

#endif
