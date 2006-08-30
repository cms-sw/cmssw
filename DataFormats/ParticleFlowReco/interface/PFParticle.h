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
	       unsigned id, unsigned daughter1Id, unsigned daughter2Id);

    PFParticle(const PFParticle& other);

    /// \return pdg code
    int      pdgCode() const {return pdgCode_; }

    /// \return id
    unsigned id() const { return id_; }

    /// \return id of first daughter
    unsigned daughter1Id() const { return daughter1Id_; }

    /// \return id of second daughter
    unsigned daughter2Id() const { return daughter2Id_; }


    friend  std::ostream& operator<<(std::ostream& out, 
				     const PFParticle& track);

  private:

    int       pdgCode_;
    unsigned  id_;
    unsigned  daughter1Id_;
    unsigned  daughter2Id_;
  };

}

#endif
