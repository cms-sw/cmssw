#include "FastSimulation/Utilities/interface/LandauFluctuationGenerator.h"
#include "FastSimulation/SimplifiedGeometryPropagator/interface/Particle.h"
#include "FastSimulation/SimplifiedGeometryPropagator/interface/SimplifiedGeometry.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <cmath>
#include <memory>

#include "FastSimulation/SimplifiedGeometryPropagator/interface/InteractionModelFactory.h"
#include "FastSimulation/SimplifiedGeometryPropagator/interface/InteractionModel.h"
#include "FastSimulation/SimplifiedGeometryPropagator/interface/Constants.h"
#include "DataFormats/Math/interface/LorentzVector.h"


///////////////////////////////////////////////
// Author: Patrick Janot
// Date: 8-Jan-2004
//
// Revision: Class structure modified to match SimplifiedGeometryPropagator
//           Fixed a bug in which particles could deposit more energy than they have
//           S. Kurz, 29 May 2017
//////////////////////////////////////////////////////////


namespace fastsim
{
    //! Implementation of most probable energy loss by ionization in the tracker layers.
    /*!
        Computes the most probable energy loss by ionization from a charged particle in the tracker layer,
        smears it with Landau fluctuations and returns the particle with modified energy.
        The deposited energy is assigned with a produced SimHit (if active material hit).
        \sa TrackerSimHitProducer
    */
    class EnergyLoss : public InteractionModel
    {
        public:
        //! Constructor.
        EnergyLoss(const std::string & name,const edm::ParameterSet & cfg);

        //! Default destructor.
        ~EnergyLoss() override{;};

        //! Perform the interaction.
        /*!
            \param particle The particle that interacts with the matter.
            \param layer The detector layer that interacts with the particle.
            \param secondaries Particles that are produced in the interaction (if any).
            \param random The Random Engine.
        */
        void interact(fastsim::Particle & particle, const SimplifiedGeometry & layer, std::vector<std::unique_ptr<fastsim::Particle> > & secondaries, const RandomEngineAndDistribution & random) override;
        
        private:
        LandauFluctuationGenerator theGenerator;  //!< Generator to do Landau fluctuation
        double minMomentum_;  //!< Minimum momentum of incoming (charged) particle
        double density_;  //!< Density of material (usually silicon rho=2.329)
        double radLenInCm_;  //!< Radiation length of material (usually silicon X0=9.360)
        double A_;  //!< Atomic weight of material (usually silicon A=28.0855)
        double Z_;  //!< Atomic number of material (usually silicon Z=14)
    };
}

fastsim::EnergyLoss::EnergyLoss(const std::string & name,const edm::ParameterSet & cfg)
    : fastsim::InteractionModel(name), theGenerator(LandauFluctuationGenerator())
{
    // Set the minimal momentum
    minMomentum_ = cfg.getParameter<double>("minMomentumCut");
    // Material properties
    A_ = cfg.getParameter<double>("A");
    Z_ = cfg.getParameter<double>("Z");
    density_ = cfg.getParameter<double>("density");
    radLenInCm_ = cfg.getParameter<double>("radLen");
}

void fastsim::EnergyLoss::interact(fastsim::Particle & particle, const SimplifiedGeometry & layer,std::vector<std::unique_ptr<fastsim::Particle> > & secondaries,const RandomEngineAndDistribution & random)
{
    // Reset the energy deposit in the layer
    particle.setEnergyDeposit(0);

    //
    // no material
    //
    double radLengths = layer.getThickness(particle.position(),particle.momentum());
    if(radLengths < 1E-10)
    {
    return;
    }

    //
    // only charged particles
    //
    if(particle.charge()==0)
    {
    return;
    }

    //
    // minimum momentum
    //
    double p2  = particle.momentum().Vect().Mag2();
    if(p2 < minMomentum_ * minMomentum_){
        return;
    }

    // Mean excitation energy (in GeV)
    double excitE = 12.5E-9 * Z_;
    
    // The thickness in cm
    double thick = radLengths * radLenInCm_;

    // This is a simple version (a la PDG) of a dE/dx generator.
    // It replaces the buggy GEANT3 -> C++ former version.
    // Author : Patrick Janot - 8-Jan-2004

    double m2  = particle.momentum().mass() * particle.momentum().mass();
    double e2  = p2 + m2;

    double beta2 = p2 / e2;
    double gama2 = e2 / m2;

    double charge2 = particle.charge() * particle.charge();

    // Energy loss spread in GeV
    double eSpread  = 0.1536E-3 * charge2 * (Z_ / A_) * density_ * thick / beta2;

    // Most probable energy loss (from the integrated Bethe-Bloch equation)
    double mostProbableLoss = eSpread * (
                            log(2. * fastsim::Constants::eMass * beta2 * gama2 * eSpread / (excitE*excitE))
                            - beta2 + 0.200);

    // Generate the energy loss with Landau fluctuations
    double dedx = mostProbableLoss + eSpread * theGenerator.landau(&random);

    // Compute the new energy and momentum
    double newE = particle.momentum().e() - dedx;

    // Particle is stopped
    double eDiff2 = newE * newE - m2;
    if(eDiff2 < 0){
        particle.momentum().SetXYZT(0.,0.,0.,0.);
        // The energy is deposited in the detector
        // Assigned with SimHit (if active layer) -> see TrackerSimHitProducer
        particle.setEnergyDeposit(particle.momentum().e() - particle.momentum().mass());
        return;
    }

    // Relative change in momentum
    double fac  = std::sqrt(eDiff2 / p2);

    // The energy is deposited in the detector
    // Assigned with SimHit (if active layer) -> see TrackerSimHitProducer
    particle.setEnergyDeposit(dedx);

    // Update the momentum
    particle.momentum().SetXYZT(particle.momentum().Px() * fac,
        particle.momentum().Py() * fac,
        particle.momentum().Pz() * fac,
        newE);
}   

DEFINE_EDM_PLUGIN(
    fastsim::InteractionModelFactory,
    fastsim::EnergyLoss,
    "fastsim::EnergyLoss"
    );
