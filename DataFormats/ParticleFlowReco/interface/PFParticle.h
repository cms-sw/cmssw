#ifndef DataFormats_ParticleFlowReco_PFParticle_h
#define DataFormats_ParticleFlowReco_PFParticle_h

#include "DataFormats/ParticleFlowReco/interface/PFTrack.h"

#include <iostream>

namespace reco {

  /**\class PFParticle
     \brief true particle for particle flow
     
     \author Renaud Bruneliere
     \date   July 2006
  */
  class PFParticle : public PFTrack {

  public:

    PFParticle();
  
    PFParticle(double charge, int pdgCode, 
	       unsigned id, int motherId,
	       const std::vector<int>& daughterIds);

    PFParticle(const PFParticle& other);

    /// \return pdg code
    int      pdgCode() const {return pdgCode_; }

    /// \return id
    unsigned id() const { return id_; }

    /// \return mother id
    int motherId() const { return motherId_; }

    /// \return vector of daughter ids
    const std::vector<int>& daughterIds() const {return daughterIds_;}


    friend  std::ostream& operator<<(std::ostream& out, 
				     const PFParticle& track);

  private:
    
    /// pdg code 
    int       pdgCode_;

    /// position in particle vector
    unsigned  id_;

    /// id of mother particle. -1 if no mother
    int  motherId_;

    /// id of daughter particles (can be > 2 in hadron showers)
    std::vector<int> daughterIds_;
  };

}

#endif
