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

    RecoTauPiZero()
      : CompositePtrCandidate(),
        algoName_(kUndefined), bendCorrEta_ (0.), bendCorrPhi_ (0.)
    {
      this->setPdgId(111); 
    }

    RecoTauPiZero(PiZeroAlgorithm algoName)
      : CompositePtrCandidate(), 
        algoName_(algoName), bendCorrEta_ (0.), bendCorrPhi_ (0.)
    { 
      this->setPdgId(111); 
    }

    /// constructor from values
    RecoTauPiZero(Charge q, const LorentzVector& p4,
                  const Point& vtx = Point( 0, 0, 0 ),
                  int pdgId = 111, int status = 0, bool integerCharge = true,
                  PiZeroAlgorithm algoName = kUndefined)
      : CompositePtrCandidate(q, p4, vtx, pdgId, status, integerCharge ),
        algoName_(algoName), bendCorrEta_ (0.), bendCorrPhi_ (0.)
    {
    }

    /// constructor from values
    RecoTauPiZero(Charge q, const PolarLorentzVector& p4,
                  const Point& vtx = Point( 0, 0, 0 ),
                  int pdgId = 111, int status = 0, bool integerCharge = true,
                  PiZeroAlgorithm algoName=kUndefined)
      : CompositePtrCandidate(q, p4, vtx, pdgId, status, integerCharge ),
        algoName_(algoName), bendCorrEta_ (0.), bendCorrPhi_ (0.)
    {
    }

    /// constructor from a Candidate
    explicit RecoTauPiZero(const Candidate& p, PiZeroAlgorithm algoName = kUndefined)
      : CompositePtrCandidate(p),
        algoName_(algoName), bendCorrEta_ (0.), bendCorrPhi_ (0.)
    { 
      this->setPdgId(111); 
    }

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

    /// Size of correction to account for spread of photon energy in eta and phi
    /// in case charged pions make nuclear interactions or photons convert within the tracking detector
    float bendCorrEta() const { return bendCorrEta_; }
    float bendCorrPhi() const { return bendCorrPhi_; }
    void setBendCorrEta(float bendCorrEta) { bendCorrEta_ = bendCorrEta; }
    void setBendCorrPhi(float bendCorrPhi) { bendCorrPhi_ = bendCorrPhi; }

    void print(std::ostream& out = std::cout) const;

  private:
    PiZeroAlgorithm algoName_;

    float bendCorrEta_;
    float bendCorrPhi_;
};

std::ostream & operator<<(std::ostream& out, const RecoTauPiZero& c);

}

#endif
