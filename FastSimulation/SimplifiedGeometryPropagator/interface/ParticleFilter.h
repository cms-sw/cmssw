#ifndef FASTSIM_PARTICLEFILTER
#define FASTSIM_PARTICLEFILTER

#include "DataFormats/Math/interface/LorentzVector.h"
#include <vector>

///////////////////////////////////////////////
// Author: Patrick Janot
// Date: 09 Dez 2003
//
// Revision: Class structure modified to match SimplifiedGeometryPropagator
//           S. Kurz, 29 May 2017
//////////////////////////////////////////////////////////

namespace edm {
  class ParameterSet;
}

namespace fastsim {
  class Particle;

  //! (Kinematic) cuts on the particles that are propagated.
  /*!
        All other particles are skipped.
    */
  class ParticleFilter {
  public:
    //! Default Constructor.
    ParticleFilter(const edm::ParameterSet& cfg);

    //! Check all if all criteria are fullfilled.
    /*!
            - Particle is invisible (neutrinos by default, list can be extended)
            - Kinematic cuts (calls acceptsEN(...))
            - Vertex within tracker volume (calls acceptsVtx(...))
            \sa acceptsEn(const Particle & particle)
            \sa acceptsVtx(const math::XYZTLorentzVector & originVertexPosition)
        */
    bool accepts(const Particle& particle) const;

    //! Kinematic cuts on the particle
    bool acceptsEn(const Particle& particle) const;

    //! Vertex within tracker volume
    /*!
            \param originVertexPosition Position of origin vertex.
        */
    bool acceptsVtx(const math::XYZTLorentzVector& originVertexPosition) const;

  private:
    double chargedPtMin2_;            //!< Minimum pT^2 of a charged particle
    double EMin_;                     //!< Minimum energy of a particle
    double protonEMin_;               //!< Allow *ALL* protons with energy > protonEMin
    double cos2ThetaMax_;             //!< Particles must have abs(eta) < etaMax if close to beampipe
    double vertexRMax2_;              //!< Radius^2 of tracker volume
    double vertexZMax_;               //!< Z of tracker volume
    std::vector<int> skipParticles_;  //!< List of invisible particles (neutrinos are excluded by default)
  };
}  // namespace fastsim

#endif
