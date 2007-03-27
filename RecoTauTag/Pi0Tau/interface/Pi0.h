#ifndef RecoTauTag_Pi0Tau_Pi0_h
#define RecoTauTag_Pi0Tau_Pi0_h

#include "DataFormats/Math/interface/Point3D.h"
#include "DataFormats/Math/interface/LorentzVector.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateFwd.h"

namespace reco {

  /**\class Pi0
     \brief reconstructed Pi0 candidates from either PFCandidate or BasicCluster
     
     \author Dongwook Jang
     \date   December 2006
  */

  const double PI0MASS = 0.135;

  class Pi0 {

  public:

    enum Pi0Type { UnDefined = 0,
		   UnResolved = 1,
		   Resolved = 2,
		   Converted1 = 3,
		   Converted2 = 4 };


    Pi0();
  
    Pi0(int type, double e, math::XYZPoint pos, math::XYZTLorentzVector mon,
	reco::PFCandidateRefVector &source_candidates);

    Pi0(const Pi0& other);

    // \return pi0 type
    int type() const { return type_; }

    // \return pi0 energy
    double energy() const { return energy_; }

    // \return pi0 position in ECAL
    const math::XYZPoint &position() const { return position_; }

    // \return reference to vector of source candidates
    const reco::PFCandidateRefVector &sourceCandidates() const { return sourceCandidates_; }

    // \return momentum given by a certain vertex
    math::XYZTLorentzVector momentum(const math::XYZPoint &vtx) const;

    // \return momentum given by vertex (0,0,0)
    const math::XYZTLorentzVector &momentum() const { return momentum_; };

    // overwriting ostream <<
    friend  std::ostream& operator<<(std::ostream& out, 
				     const Pi0& pi0);

  private:

    // pi0 type
    int type_;

    // pi0 candidate energy
    double energy_;

    // pi0 candidate position
    math::XYZPoint position_;

    // pi0 candidate momentum calculated with respect to vertex(0,0,0)
    math::XYZTLorentzVector momentum_;

    // reference to vector of source candidates
    reco::PFCandidateRefVector sourceCandidates_;

  };

}
#endif
