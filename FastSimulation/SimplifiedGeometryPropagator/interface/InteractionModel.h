#ifndef FASTSIM_INTERACTIONMODEL
#define FASTSIM_INTERACTIONMODEL

#include "FWCore/Framework/interface/ProducesCollector.h"

#include <string>
#include <vector>
#include <memory>

///////////////////////////////////////////////
// Author: L. Vanelderen, S. Kurz
// Date: 29 May 2017
//////////////////////////////////////////////////////////

namespace edm {
  class Event;
}  // namespace edm

class RandomEngineAndDistribution;

namespace fastsim {
  class SimplifiedGeometry;
  class Particle;

  //! Base class for any interaction model between a particle and a tracker layer.
  /*!
        Every instance should have a distinct std::string name.
    */
  class InteractionModel {
  public:
    //! Constructor.
    /*!
            \param name Enique name for every instance.
        */
    InteractionModel(std::string name) : name_(name) {}

    //! Default destructor.
    virtual ~InteractionModel() { ; }

    //! Perform the interaction.
    /*!
            \param particle The particle that interacts with the matter.
            \param layer The detector layer that interacts with the particle.
            \param secondaries Particles that are produced in the interaction (if any).
            \param random The Random Engine.
        */
    virtual void interact(Particle& particle,
                          const SimplifiedGeometry& layer,
                          std::vector<std::unique_ptr<Particle> >& secondaries,
                          const RandomEngineAndDistribution& random) = 0;

    //! In case interaction produces and stores content in the event (e.g. TrackerSimHits).
    virtual void registerProducts(edm::ProducesCollector) const {}

    //! In case interaction produces and stores content in the event (e.g. TrackerSimHits).
    virtual void storeProducts(edm::Event& iEvent) { ; }

    //! Return (unique) name of this interaction.
    const std::string getName() { return name_; }

    //! Basic information output.
    friend std::ostream& operator<<(std::ostream& o, const InteractionModel& model);

  private:
    const std::string name_;  //!< A unique name for every instance of any interaction.
  };
  std::ostream& operator<<(std::ostream& os, const InteractionModel& interactionModel);

}  // namespace fastsim

#endif
