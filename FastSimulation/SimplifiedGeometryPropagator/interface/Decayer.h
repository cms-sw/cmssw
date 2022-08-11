#ifndef FASTSIM_DECAYER_H
#define FASTSIM_DECAYER_H

#include <memory>
#include <vector>


///////////////////////////////////////////////
// Author: L. Vanelderen
// Date: 13 May 2014
//
// Revision: Class structure modified to match SimplifiedGeometryPropagator
//           S. Kurz, 29 May 2017
//////////////////////////////////////////////////////////


namespace gen {
  class P8RndmEngine;
}

namespace CLHEP {
  class HepRandomEngine;
}

namespace Pythia8 {
  class Pythia;
}

namespace fastsim
{
    class Particle;

    //! Implementation of non-stable particle decays.
    /*!
        Inspired by method Pythia8Hadronizer::residualDecay() in GeneratorInterface/Pythia8Interface/src/Py8GunBase.cc
    */
    class Decayer 
    {
        public:
        //! Default Constructor.
        Decayer();

        //! Default destructor.
        ~Decayer();

        //! Decay particle using pythia.
        /*!
            \param particle The particle that should be decayed.
            \param secondaries The decay products.
            \param engine The Random Engine.
        */
        void decay(const Particle & particle, std::vector<std::unique_ptr<Particle> > & secondaries, CLHEP::HepRandomEngine & engine) const;
        private:    
        std::unique_ptr<Pythia8::Pythia> pythia_;  //!< Instance of pythia
        std::unique_ptr<gen::P8RndmEngine> pythiaRandomEngine_;  //!< Instance of pythia Random Engine
    };
}
#endif
