#include "DataFormats/Common/interface/Wrapper.h"
#include "DataFormats/HcalIsolatedTrack/interface/IsolatedPixelTrackCandidate.h"
namespace {
  namespace {

    reco::IsolatedPixelTrackCandidate                                          ptc;
    reco::IsolatedPixelTrackCandidateCollection                                ptc_c;
    reco::IsolatedPixelTrackCandidateRef                                       ptc_r;
    reco::IsolatedPixelTrackCandidateRefProd                                   ptc_rp;
    reco::IsolatedPixelTrackCandidateRefVector                                 ptc_rv;
    edm::Wrapper<reco::IsolatedPixelTrackCandidateCollection>                  ptc_wc;
    edm::reftobase::Holder<reco::Candidate, reco::IsolatedPixelTrackCandidateRef> ptc_h;
  }
}
