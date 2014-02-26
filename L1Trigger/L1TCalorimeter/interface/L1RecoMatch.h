/*
 * L1RecoMatch
 *
 * Intermediate light-weight dataformat that puts all of the
 * matched objects in one object.
 *
 * This is just a trick so we can operate on the objects using the string cut
 * parser to build the TTrees.
 *
 */

#ifndef L1RECOMATCH_L5Q3TJND
#define L1RECOMATCH_L5Q3TJND

#include "L1Trigger/L1TCalorimeter/interface/L1GObject.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/Provenance/interface/EventID.h"

class L1RecoMatch {
  public:
    // Default needed for persistency
    L1RecoMatch() {}
    L1RecoMatch(const reco::Candidate* reco, const reco::Candidate* l1,
        const reco::Candidate* l1g, edm::EventID id,
        unsigned int index, unsigned int nTotalObjects, unsigned int nPVs);

    const reco::Candidate* reco() const;
    const reco::Candidate* l1() const;
    const reco::Candidate* l1g() const;

    // Returns true if l1() is not NULL (i.e. there is a match)
    bool l1Match() const;

    // Returns true if l1g() is not NULL (i.e. there is a match)
    bool l1gMatch() const;

    /// Get the run-lumi-event numbers
    const edm::EventID& id() const;
    /// Get the index of this match in the event.
    unsigned int index() const;
    /// Get the total number of reco objects in this event.
    unsigned int nTotalObjects() const;
    /// Get number of PVs
    unsigned int nPVs() const;

  private:
    const reco::Candidate* reco_;
    const reco::Candidate* l1extra_;
    const reco::Candidate* l1g_;
    edm::EventID id_;
    unsigned int index_;
    unsigned int nTotalObjects_;
    unsigned int nPVs_;
};

#endif /* end of include guard: L1RECOMATCH_L5Q3TJND */
