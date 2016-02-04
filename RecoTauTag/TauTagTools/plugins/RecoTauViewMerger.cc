/*
 * Produces a PFTauCollection that merges together the PFTau views given
 * by the VInputTag src
 *
 * Author: Evan K. Friis (UC Davis)
 */

#include "CommonTools/UtilAlgos/interface/Merger.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/TauReco/interface/PFTau.h"
#include "DataFormats/TauReco/interface/PFTauFwd.h"
#include "DataFormats/Common/interface/View.h"

namespace {
struct ClonePolicy {
  static reco::PFTau clone(const reco::PFTau &tau) { return tau; }
};
}

typedef Merger<edm::View<reco::PFTau>, reco::PFTauCollection, ClonePolicy>
  RecoTauViewMerger;

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(RecoTauViewMerger);
