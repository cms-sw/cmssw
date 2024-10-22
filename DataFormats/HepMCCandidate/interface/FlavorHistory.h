#ifndef HepMCCandidate_FlavorHistory_h
#define HepMCCandidate_FlavorHistory_h

/** \class reco::FlavorHistory
 *
 * Stores information about the flavor history of a parton
 *
 * \author: Stephen Mrenna (FNAL), Salvatore Rappoccio (JHU)
 *
 */

// -------------------------------------------------------------
// Identify the ancestry of the Quark
//
//
// Matrix Element:
//    Status 3 parent with precisely 2 "grandparents" that
//    is outside of the "initial" section (0-5) that has the
//    same ID as the status 2 parton in question.
//
// Flavor excitation:
//    If we find only one outgoing parton.
//
// Gluon splitting:
//    Parent is a quark of a different flavor than the parton
//    in question, or a gluon.
//    Can come from either ISR or FSR.
//
// True decay:
//    Decays from a resonance like top, Higgs, etc.
// -------------------------------------------------------------

#include "DataFormats/Common/interface/Ptr.h"
#include "DataFormats/Common/interface/OwnVector.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/View.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/Candidate/interface/CandidateFwd.h"
#include "DataFormats/Candidate/interface/ShallowClonePtrCandidate.h"

#include <fstream>

namespace reco {

  class FlavorHistory {
  public:
    enum FLAVOR_T {
      FLAVOR_NULL = 0,  // No flavor, unset
      FLAVOR_GS,        // gluon splitting
      FLAVOR_EXC,       // flavor excitation
      FLAVOR_ME,        // matrix element
      FLAVOR_DECAY,     // flavor decay
      N_FLAVOR_TYPES
    };  // total number

    static const int gluonId = 21;
    static const int tQuarkId = 6;
    static const int bQuarkId = 5;
    static const int cQuarkId = 4;

    FlavorHistory();
    FlavorHistory(FLAVOR_T flavorSource,
                  reco::CandidatePtr const& parton,
                  reco::CandidatePtr const& progenitor,
                  reco::CandidatePtr const& sister,
                  reco::ShallowClonePtrCandidate const& matchedJet,
                  reco::ShallowClonePtrCandidate const& sisterJet);
    FlavorHistory(FLAVOR_T flavorSource,
                  edm::Handle<edm::View<reco::Candidate> > h_partons,
                  int iparton,
                  int iprogenitor,
                  int isister,
                  reco::ShallowClonePtrCandidate const& matchedJet,
                  reco::ShallowClonePtrCandidate const& sisterJet);
    FlavorHistory(FLAVOR_T flavorSource,
                  edm::Handle<reco::CandidateCollection> h_partons,
                  int iparton,
                  int iprogenitor,
                  int isister,
                  reco::ShallowClonePtrCandidate const& matchedJet,
                  reco::ShallowClonePtrCandidate const& sisterJet);
    ~FlavorHistory() {}

    // Accessors
    FLAVOR_T flavorSource() const { return flavorSource_; }
    bool hasParton() const { return parton_.isNonnull(); }
    bool hasSister() const { return sister_.isNonnull(); }
    bool hasProgenitor() const { return progenitor_.isNonnull(); }
    bool hasMatchedJet() const { return matchedJet_.masterClonePtr().isNonnull(); }
    bool hasSisterJet() const { return sisterJet_.masterClonePtr().isNonnull(); }
    const reco::CandidatePtr& parton() const { return parton_; }
    const reco::CandidatePtr& sister() const { return sister_; }
    const reco::CandidatePtr& progenitor() const { return progenitor_; }
    const reco::ShallowClonePtrCandidate& matchedJet() const { return matchedJet_; }
    const reco::ShallowClonePtrCandidate& sisterJet() const { return sisterJet_; }

    // Operators for sorting and keys
    bool operator<(FlavorHistory const& right) const { return parton_.key() < right.parton_.key(); }
    bool operator>(FlavorHistory const& right) const { return parton_.key() > right.parton_.key(); }
    bool operator==(FlavorHistory const& right) const { return parton_.key() == right.parton_.key(); }

  protected:
    FLAVOR_T flavorSource_;
    reco::CandidatePtr parton_;
    reco::CandidatePtr progenitor_;
    reco::CandidatePtr sister_;
    reco::ShallowClonePtrCandidate matchedJet_;
    reco::ShallowClonePtrCandidate sisterJet_;
  };

}  // namespace reco

std::ostream& operator<<(std::ostream& out, reco::FlavorHistory const& cand);

#endif
