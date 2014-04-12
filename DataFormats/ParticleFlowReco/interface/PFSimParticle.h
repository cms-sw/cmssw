#ifndef DataFormats_ParticleFlowReco_PFSimParticle_h
#define DataFormats_ParticleFlowReco_PFSimParticle_h

#include "DataFormats/ParticleFlowReco/interface/PFTrack.h"

#include <iostream>

namespace reco {

  /**\class PFSimParticle
     \brief true particle for particle flow
     
     Additional information w/r to PFTrack: 
     - pdg code 
     - information about mother and daughters
     \author Renaud Bruneliere
     \date   July 2006
  */
  class PFSimParticle : public PFTrack {

  public:

    PFSimParticle();
  
    PFSimParticle(double charge, int pdgCode, 
                  unsigned id, int motherId,
                  const std::vector<int>& daughterIds,
		  unsigned rectrackId,
		  const std::vector<unsigned>& recHitContrib,
		  const std::vector<double>&   recHitContribFrac );

    PFSimParticle(const PFSimParticle& other);

    /// \return pdg code
    int      pdgCode() const {return pdgCode_; }

    /// \return id
    unsigned id() const { return id_; }

    /// \return mother id
    int motherId() const { return motherId_; }

    /// \return vector of daughter ids
    const std::vector<int>& daughterIds() const {return daughterIds_;}

   //accessing MCTruth Matching Info
    unsigned rectrackId() 
      const {return rectrackId_;} 
    std::vector<unsigned> recHitContrib() 
      const {return recHitContrib_;} 
    std::vector<double> recHitContribFrac() 
      const {return recHitContribFrac_;} 

    friend  std::ostream& operator<<(std::ostream& out, 
                                     const PFSimParticle& track);

  private:
    
    /// pdg code 
    int       pdgCode_;

    /// position in particle vector
    unsigned  id_;

    /// id of mother particle. -1 if no mother
    int  motherId_;

    /// id of daughter particles (can be > 2 in hadron showers)
    std::vector<int> daughterIds_;

    unsigned rectrackId_; 
    std::vector<unsigned> recHitContrib_; 
    std::vector<double>   recHitContribFrac_;

  };

}

#endif
