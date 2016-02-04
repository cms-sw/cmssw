#ifndef DataFormats_TauReco_RecoTauPiZero_h
#define DataFormats_TauReco_RecoTauPiZero_h

#include "DataFormats/Candidate/interface/CompositePtrCandidate.h"

namespace reco {
class RecoTauPiZero : public CompositePtrCandidate {
  public:
    enum PiZeroAlgorithm {
      // Algorithm where each photon becomes a pi zero
      kUndefined = 0,
      kTrivial = 1,
      kCombinatoric = 2,
      kStrips = 3
    };

    RecoTauPiZero():CompositePtrCandidate(),algoName_(kUndefined){
      this->setPdgId(111); }

    RecoTauPiZero(PiZeroAlgorithm algoName):
        CompositePtrCandidate(), algoName_(algoName) { this->setPdgId(111); }

    /// constructor from values
    RecoTauPiZero(Charge q, const LorentzVector& p4,
                  const Point& vtx = Point( 0, 0, 0 ),
                  int pdgId = 111, int status = 0, bool integerCharge = true,
                  PiZeroAlgorithm algoName=kUndefined):
        CompositePtrCandidate(
            q, p4, vtx, pdgId, status, integerCharge ),algoName_(algoName) {}

    /// constructor from values
    RecoTauPiZero(Charge q, const PolarLorentzVector& p4,
                  const Point& vtx = Point( 0, 0, 0 ),
                  int pdgId = 111, int status = 0, bool integerCharge = true,
                  PiZeroAlgorithm algoName=kUndefined):
        CompositePtrCandidate(
            q, p4, vtx, pdgId, status, integerCharge ),algoName_(algoName) {}

    /// constructor from a Candidate
    explicit RecoTauPiZero(
        const Candidate & p, PiZeroAlgorithm algoName=kUndefined):
        CompositePtrCandidate(p),algoName_(algoName) { this->setPdgId(111); }

    /// destructor
    ~RecoTauPiZero(){};

    /// Number of PFGamma constituents
    size_t numberOfGammas() const;

    /// Number of electron constituents
    size_t numberOfElectrons() const;

    /// Maximum DeltaPhi between a constituent and the four vector
    double maxDeltaPhi() const;

    /// Maxmum DeltaEta between a constituent and the four vector
    double maxDeltaEta() const;

    /// Algorithm that built this piZero
    PiZeroAlgorithm algo() const;

    /// Check whether a given algo produced this pi zero
    bool algoIs(PiZeroAlgorithm algo) const;

    void print(std::ostream& out=std::cout) const;

  private:
    PiZeroAlgorithm algoName_;

};

std::ostream & operator<<(std::ostream& out, const RecoTauPiZero& c);

}

#endif
