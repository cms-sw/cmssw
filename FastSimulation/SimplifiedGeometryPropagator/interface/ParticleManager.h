#ifndef FASTSIM_PARTICLEMANAGER_H
#define FASTSIM_PARTICLEMANAGER_H

#include "DataFormats/Math/interface/LorentzVector.h"
#include "HepMC/GenEvent.h"
#include <vector>
#include <memory>

#include "SimDataFormats/Track/interface/SimTrack.h"
#include "SimDataFormats/Vertex/interface/SimVertex.h"

///////////////////////////////////////////////
// Author: L. Vanelderen, S. Kurz
// Date: 29 May 2017
//////////////////////////////////////////////////////////

namespace HepPDT {
  class ParticleDataTable;
}

class RandomEngineAndDistribution;

namespace fastsim {
  class Particle;
  class ParticleFilter;
  class SimplifiedGeometry;

  //! Manages GenParticles and Secondaries from interactions.
  /*!
        Manages which particle has to be propagated next, this includes GenParticles and secondaries from the interactions.
        Furthermore, checks if all necessary information is included with the GenParticles (charge, lifetime), otherwise 
        reads information from HepPDT::ParticleDataTable.
        Also handles secondaries, including closestChargedDaughter algorithm which is used for FastSim (cheat) tracking: a charged daughter can
        continue the track of a charged mother, see addSecondaries(...).
    */
  class ParticleManager {
  public:
    //! Constructor.
    /*!
            \param genEvent Get the GenEvent.
            \param particleDataTable Get information about particles, e.g. charge, lifetime.
            \param beamPipeRadius Radius of the beampipe.
            \param deltaRchargedMother For FastSim (cheat) tracking: cut on the angle between a charged mother and charged daughter.
            \param particleFilter Selects which particles have to be propagated.
            \param simTracks The SimTracks.
            \param simVertices The SimVertices.
        */
    ParticleManager(const HepMC::GenEvent& genEvent,
                    const HepPDT::ParticleDataTable& particleDataTable,
                    double beamPipeRadius,
                    double deltaRchargedMother,
                    const ParticleFilter& particleFilter,
                    std::vector<SimTrack>& simTracks,
                    std::vector<SimVertex>& simVertices,
                    bool useFastSimsDecayer);

    //! Default destructor.
    ~ParticleManager();

    //! Returns the next particle that has to be propagated (secondary or genParticle).
    /*!
            Main method of this class. At first consideres particles from the buffer (secondaries) if there are none, then the
            next GenParticle is considered. Only returns particles that pass the (kinetic) cuts of the ParticleFilter.
            Furthermore, uses the ParticleDataTable to ensure all necessary information about the particle is stored (lifetime, charge).
            \return The next particle that has to be propagated.
            \sa ParticleFilter
        */
    std::unique_ptr<Particle> nextParticle(const RandomEngineAndDistribution& random);

    //! Adds secondaries that are produced by any of the interactions (or particle decay) to the buffer.
    /*!
            Also checks which charged daughter is closest to a charged mother (in deltaR) and assigns the same SimTrack ID.
            \param vertexPosition The origin vertex (interaction or particle decay took place here).
            \param motherSimTrackId SimTrack ID of the mother particle, necessary for FastSim (cheat) tracking.
            \param secondaries All secondaries that where produced in a single particle decay or interaction.
        */
    void addSecondaries(const math::XYZTLorentzVector& vertexPosition,
                        int motherSimTrackId,
                        std::vector<std::unique_ptr<Particle> >& secondaries,
                        const SimplifiedGeometry* layer = nullptr);

    //! Returns the position of a given SimVertex. Needed for interfacing the code with the old calorimetry.
    const SimVertex getSimVertex(unsigned i) { return simVertices_->at(i); }

    //! Returns a given SimTrack. Needed for interfacing the code with the old calorimetry.
    const SimTrack getSimTrack(unsigned i) { return simTracks_->at(i); }

    //! Necessary to add an end vertex to a particle.
    /*!
            Needed if particle is no longer propagated for some reason (e.g. remaining energy below threshold) and no
            secondaries where produced at that point.
            \return Index of that vertex.
        */
    unsigned addEndVertex(const Particle* particle);

  private:
    //! Add a simVertex (simVertex contains information about the track it was produced).
    /*!
            Add a origin vertex for any particle.
            \param position Position of the vertex.
            \param motherIndex Index of the parent's simTrack.
            \return Index of that simVertex.
        */
    unsigned addSimVertex(const math::XYZTLorentzVector& position, int motherIndex);

    //! Add a simTrack (simTrack contains some basic info about the particle, e.g. pdgId).
    /*!
            Add a simTrack for a given particle and assign an index for that track. This might also be the index of the track
            of the mother particle (FastSim cheat tracking).
            \param particle Particle that produces that simTrack.
            \return Index of that simTrack.
        */
    unsigned addSimTrack(const Particle* particle);
    void exoticRelativesChecker(const HepMC::GenVertex* originVertex, int& hasExoticAssociation, int ngendepth);

    //! Returns next particle from the GenEvent that has to be propagated.
    /*!
            Tries to get some basic information about the status of the particle from the GenEvent and does some first rejection cuts based on them.
        */
    std::unique_ptr<Particle> nextGenParticle();

    // data members
    const HepMC::GenEvent* const genEvent_;  //!< The GenEvent
    HepMC::GenEvent::particle_const_iterator
        genParticleIterator_;  //!< Iterator to keep track on which GenParticles where already considered.
    const HepMC::GenEvent::particle_const_iterator genParticleEnd_;  //!< The last particle of the GenEvent.
    int genParticleIndex_;  //!< Index of particle in the GenEvent (if it is a GenParticle)
    const HepPDT::ParticleDataTable* const
        particleDataTable_;         //!< Necessary to get information like lifetime and charge of a particle if unknown.
    const double beamPipeRadius2_;  //!< (Radius of the beampipe)^2
    const double
        deltaRchargedMother_;  //!< For FastSim (cheat) tracking: cut on the angle between a charged mother and charged daughter.
    const ParticleFilter* const particleFilter_;  //!< (Kinematic) cuts on the particles that have to be propagated.
    std::vector<SimTrack>* simTracks_;            //!< The generated SimTrack of this event.
    std::vector<SimVertex>* simVertices_;         //!< The generated SimVertices of this event.
    bool useFastSimsDecayer_;
    double momentumUnitConversionFactor_;         //!< Convert pythia units to GeV (FastSim standard)
    double lengthUnitConversionFactor_;           //!< Convert pythia unis to cm (FastSim standard)
    double lengthUnitConversionFactor2_;          //!< Convert pythia unis to cm^2 (FastSim standard)
    double timeUnitConversionFactor_;             //!< Convert pythia unis to ns (FastSim standard)
    std::vector<std::unique_ptr<Particle> >
        particleBuffer_;  //!< The vector of all secondaries that are not yet propagated in the event.
  };
}  // namespace fastsim

inline bool isExotic(int pdgid_) {
  unsigned int pdgid = std::abs(pdgid_);
  return ((pdgid >= 1000000 && pdgid < 4000000 && pdgid != 3000022) ||  // SUSY, R-hadron, and technicolor particles
          pdgid == 17 ||                                                // 4th generation lepton
          pdgid == 34 ||                                                // W-prime
          pdgid == 37 ||                                                // charged Higgs
          pdgid == 39);                                                 // bulk graviton
}

#endif
