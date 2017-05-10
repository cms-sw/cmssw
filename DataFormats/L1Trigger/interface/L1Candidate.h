#ifndef L1Candidate_h
#define L1Candidate_h


#include "DataFormats/Candidate/interface/LeafCandidate.h"

namespace l1t {

	// All L1 data formats which encode physically meaningful quantities inherit from Candidate
  class L1Candidate : public reco::LeafCandidate {

  public:
    L1Candidate();

    // construct from *both* physical and integer values
    L1Candidate( const LorentzVector& p4,
		 int pt=0,
		 int eta=0,
		 int phi=0,
		 int qual=0);

    ~L1Candidate();

    // methods to set integer values
    // in general, these should not be needed
    void setHwPt(int pt);
    void setHwEta(int eta);
    void setHwPhi(int phi);
    void setHwQual(int qual);

    // methods to retrieve integer values
    int hwPt() const;
    int hwEta() const;
    int hwPhi() const;
    int hwQual() const;

  private:

    // integer "hardware" values
    int hwPt_;
    int hwEta_;
    int hwPhi_;
    int hwQual_;

  };

};

#endif
