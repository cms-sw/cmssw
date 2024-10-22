#include "DataFormats/HepMCCandidate/interface/FlavorHistory.h"

using namespace reco;
using namespace edm;
using namespace std;

FlavorHistory::FlavorHistory() { flavorSource_ = FLAVOR_NULL; }

FlavorHistory::FlavorHistory(FLAVOR_T flavorSource,
                             CandidatePtr const& parton,
                             CandidatePtr const& progenitor,
                             CandidatePtr const& sister,
                             reco::ShallowClonePtrCandidate const& matchedJet,
                             reco::ShallowClonePtrCandidate const& sisterJet)
    : flavorSource_(flavorSource),
      parton_(parton),
      progenitor_(progenitor),
      sister_(sister),
      matchedJet_(matchedJet),
      sisterJet_(sisterJet) {}

FlavorHistory::FlavorHistory(FLAVOR_T flavorSource,
                             Handle<View<Candidate> > h_partons,
                             int parton,
                             int progenitor,
                             int sister,
                             reco::ShallowClonePtrCandidate const& matchedJet,
                             reco::ShallowClonePtrCandidate const& sisterJet)
    : flavorSource_(flavorSource),
      parton_(parton >= 0 && static_cast<unsigned int>(parton) < h_partons->size() ? CandidatePtr(h_partons, parton)
                                                                                   : CandidatePtr()),
      progenitor_(progenitor >= 0 && static_cast<unsigned int>(progenitor) < h_partons->size()
                      ? CandidatePtr(h_partons, progenitor)
                      : CandidatePtr()),
      sister_(sister >= 0 && static_cast<unsigned int>(sister) < h_partons->size() ? CandidatePtr(h_partons, sister)
                                                                                   : CandidatePtr()),
      matchedJet_(matchedJet),
      sisterJet_(sisterJet) {}

FlavorHistory::FlavorHistory(FLAVOR_T flavorSource,
                             Handle<CandidateCollection> h_partons,
                             int parton,
                             int progenitor,
                             int sister,
                             reco::ShallowClonePtrCandidate const& matchedJet,
                             reco::ShallowClonePtrCandidate const& sisterJet)
    : flavorSource_(flavorSource),
      parton_(parton >= 0 && static_cast<unsigned int>(parton) < h_partons->size() ? CandidatePtr(h_partons, parton)
                                                                                   : CandidatePtr()),
      progenitor_(progenitor >= 0 && static_cast<unsigned int>(progenitor) < h_partons->size()
                      ? CandidatePtr(h_partons, progenitor)
                      : CandidatePtr()),
      sister_(sister >= 0 && static_cast<unsigned int>(sister) < h_partons->size() ? CandidatePtr(h_partons, sister)
                                                                                   : CandidatePtr()),
      matchedJet_(matchedJet),
      sisterJet_(sisterJet) {}

ostream& operator<<(ostream& out, Candidate const& cand) {
  char buff[1000];
  sprintf(buff,
          "%5d, status = %5d, nmo = %5d, nda = %5d, pt = %6.2f, eta = %6.2f, phi = %6.2f, m = %6.2f",
          cand.pdgId(),
          cand.status(),
          static_cast<int>(cand.numberOfMothers()),
          static_cast<int>(cand.numberOfDaughters()),
          cand.pt(),
          cand.eta(),
          cand.phi(),
          cand.mass());
  out << buff;
  return out;
}

ostream& operator<<(ostream& out, FlavorHistory const& cand) {
  out << "Source     = " << cand.flavorSource() << endl;
  if (cand.hasParton())
    out << "Parton     = " << cand.parton().key() << " : " << *(cand.parton()) << endl;
  if (cand.hasProgenitor())
    out << "Progenitor = " << cand.progenitor().key() << " : " << *(cand.progenitor()) << endl;
  if (cand.hasSister())
    out << "Sister     = " << cand.sister().key() << " : " << *(cand.sister()) << endl;
  if (cand.hasMatchedJet())
    out << "jet        = " << cand.matchedJet() << endl;
  if (cand.hasSisterJet())
    out << "sister jet = " << cand.sisterJet() << endl;
  if (cand.hasParton()) {
    out << "Ancestry: " << endl;
    Candidate const* ipar = cand.parton()->mother();
    while (ipar->numberOfMothers() > 0) {
      out << *ipar << endl;
      ipar = ipar->mother();
    }
  }
  return out;
}
