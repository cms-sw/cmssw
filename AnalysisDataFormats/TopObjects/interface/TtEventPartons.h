#ifndef TtEventPartons_h
#define TtEventPartons_h

#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "DataFormats/Candidate/interface/CandidateOnlyFwd.h"

#include <vector>

/**
   \class   TtEventPartons TtEventPartons.h "AnalysisDataFormats/TopObjects/interface/TtEventPartons.h"

   \brief   Common base class for TtFullLepEvtPartons, TtFullHadEvtPartons and TtSemiLepEvtPartons
*/

class TtGenEvent;

class TtEventPartons {
public:
  /// default constructor
  TtEventPartons() = default;
  /// default destructor
  virtual ~TtEventPartons() = default;

  /// return vector of partons in the order defined in the corresponding enum
  /// (method implemented in the derived classes)
  virtual std::vector<const reco::Candidate*> vec(const TtGenEvent& genEvt) const = 0;

  /// insert dummy index -3 for all partons that were chosen to be ignored
  void expand(std::vector<int>& vec) const;

protected:
  /// return pointer to an empty reco::Candidate
  reco::Candidate* dummyCandidatePtr() const {
    return new reco::GenParticle(0, reco::Particle::LorentzVector(), reco::Particle::Point(), 0, 0, false);
  };

  /// erase partons from vector if they where chosen to be ignored
  void prune(std::vector<const reco::Candidate*>& vec) const;

  /// flag partons that were chosen not to be used
  std::vector<bool> ignorePartons_;
};

#endif
