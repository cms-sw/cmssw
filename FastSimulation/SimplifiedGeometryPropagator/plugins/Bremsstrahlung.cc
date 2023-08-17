#include "FastSimulation/Utilities/interface/RandomEngineAndDistribution.h"
#include "FastSimulation/SimplifiedGeometryPropagator/interface/Particle.h"
#include "FastSimulation/SimplifiedGeometryPropagator/interface/SimplifiedGeometry.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <cmath>
#include <memory>

#include <Math/RotationY.h>
#include <Math/RotationZ.h>

#include "FastSimulation/SimplifiedGeometryPropagator/interface/InteractionModelFactory.h"
#include "FastSimulation/SimplifiedGeometryPropagator/interface/InteractionModel.h"
#include "FastSimulation/SimplifiedGeometryPropagator/interface/Constants.h"
#include "DataFormats/Math/interface/LorentzVector.h"


///////////////////////////////////////////////
// Author: Patrick Janot
// Date: 25-Dec-2003
//
// Revision: Class structure modified to match SimplifiedGeometryPropagator
//           Fixed a bug in which particles could radiate more energy than they have
//           S. Kurz, 29 May 2017
//////////////////////////////////////////////////////////


namespace fastsim
{
    //! Implementation of Bremsstrahlung from e+/e- in the tracker layers.
    /*!
        Computes the number, energy and angles of Bremsstrahlung photons emitted by electrons and positrons
        and modifies e+/e- particle accordingly.
    */
    class Bremsstrahlung : public InteractionModel
    {
        public:
        //! Constructor.
        Bremsstrahlung(const std::string & name, const edm::ParameterSet & cfg);

        //! Default destructor.
        ~Bremsstrahlung() override{;};

        //! Perform the interaction.
        /*!
            \param particle The particle that interacts with the matter.
            \param layer The detector layer that interacts with the particle.
            \param secondaries Particles that are produced in the interaction (if any).
            \param random The Random Engine.
        */
        void interact(Particle & particle, const SimplifiedGeometry & layer, std::vector<std::unique_ptr<Particle> > & secondaries, const RandomEngineAndDistribution & random) override;
        
        private:
        //! Compute Brem photon energy and angles, if any.
        /*!
            \param particle The particle that interacts with the matter.
            \param xmin Minimum fraction of the particle's energy that has to be converted to a photon.
            \param random The Random Engine.
            \return Momentum 4-vector of a bremsstrahlung photon.
        */
        math::XYZTLorentzVector brem(Particle & particle, double xmin, const RandomEngineAndDistribution & random) const;
        
        //! A universal angular distribution.
        /*!
            \param ener 
            \param partm 
            \param efrac 
            \param random The Random Engine.
            \return Theta from universal distribution
        */
        double gbteth(const double ener,
                  const double partm,
                  const double efrac,
                  const RandomEngineAndDistribution & random) const ;

        double minPhotonEnergy_;  //!< Cut on minimum energy of bremsstrahlung photons
        double minPhotonEnergyFraction_;  //!< Cut on minimum fraction of particle's energy which has to be carried by photon
        double Z_;  //!< Atomic number of material (usually silicon Z=14)
    };
}

fastsim::Bremsstrahlung::Bremsstrahlung(const std::string & name,const edm::ParameterSet & cfg)
    : fastsim::InteractionModel(name)
{
    // Set the minimal photon energy for a Brem from e+/-
    minPhotonEnergy_ = cfg.getParameter<double>("minPhotonEnergy");
    minPhotonEnergyFraction_ = cfg.getParameter<double>("minPhotonEnergyFraction");
    // Material properties
    Z_ = cfg.getParameter<double>("Z");
}


void fastsim::Bremsstrahlung::interact(fastsim::Particle & particle, const SimplifiedGeometry & layer,std::vector<std::unique_ptr<fastsim::Particle> > & secondaries,const RandomEngineAndDistribution & random)
{
    // only consider electrons and positrons
    if(std::abs(particle.pdgId())!=11)
    {
    return;
    }
    
    double radLengths = layer.getThickness(particle.position(),particle.momentum());
    //
    // no material
    //
    if(radLengths < 1E-10)
    {
    return;
    }

    // Protection : Just stop the electron if more than 1 radiation lengths.
    // This case corresponds to an electron entering the layer parallel to 
    // the layer axis - no reliable simulation can be done in that case...
    if(radLengths > 4.) 
    {
    particle.momentum().SetXYZT(0.,0.,0.,0.);
    return;
    }

    // electron must have more energy than minimum photon energy
    if(particle.momentum().E() - particle.momentum().mass() < minPhotonEnergy_)
    {
    return;
    }

    // Hard brem probability with a photon Energy above threshold.
    double xmin = std::max(minPhotonEnergy_/particle.momentum().E(), minPhotonEnergyFraction_);
    if(xmin >=1. || xmin <=0.) 
    {
    return;
    }

    // probability to radiate a photon
    double bremProba = radLengths * (4./3. * std::log(1./xmin)
                      - 4./3. * (1.-xmin)
                      + 1./2. * (1.-xmin*xmin));
    
  
    // Number of photons to be radiated.
    unsigned int nPhotons = random.poissonShoot(bremProba);
    if(nPhotons == 0) 
    {
    return;
    }

    // Needed to rotate photons to the lab frame
    double theta = particle.momentum().Theta();
    double phi = particle.momentum().Phi();
    double m2dontchange = particle.momentum().mass()*particle.momentum().mass();
    
    // Calculate energy of these photons and add them to the event
    for(unsigned int i=0; i<nPhotons; ++i) 
    {
        // Throw momentum of the photon
        math::XYZTLorentzVector photonMom = brem(particle, xmin, random);

        // Check that there is enough energy left.
        if(particle.momentum().E() - particle.momentum().mass() < photonMom.E()) break;

        // Rotate to the lab frame
        photonMom = ROOT::Math::RotationZ(phi) * (ROOT::Math::RotationY(theta) * photonMom);

        // Add a photon
        secondaries.emplace_back(new fastsim::Particle(22, particle.position(), photonMom));        
        
        // Update the original e+/-
        particle.momentum() -= photonMom;
        
        // Reset mass to original, since the above codes for decay e->e+gamma, ignoring proton
        particle.momentum().SetXYZT(particle.momentum().px(),particle.momentum().py(), particle.momentum().pz(),sqrt(pow(particle.momentum().P(),2)+m2dontchange));
    }
}   


math::XYZTLorentzVector
fastsim::Bremsstrahlung::brem(fastsim::Particle & particle, double xmin, const RandomEngineAndDistribution & random) const 
{
    double xp=0;
    double weight = 0.;
  
    do{
        xp = xmin * std::exp ( -std::log(xmin) * random.flatShoot() );
        weight = 1. - xp + 3./4.*xp*xp;
    }while(weight < random.flatShoot());
  
  
    // Have photon energy. Now generate angles with respect to the z axis 
    // defined by the incoming particle's momentum.

    // Isotropic in phi
    const double phi = random.flatShoot() * 2. * M_PI;
    // theta from universal distribution
    const double theta = gbteth(particle.momentum().E(), fastsim::Constants::eMass, xp, random)
                            * fastsim::Constants::eMass / particle.momentum().E();
  
    // Make momentum components
    double stheta = std::sin(theta);
    double ctheta = std::cos(theta);
    double sphi   = std::sin(phi);
    double cphi   = std::cos(phi);
  
    return xp * particle.momentum().E() * math::XYZTLorentzVector(stheta*cphi, stheta*sphi, ctheta, 1.);  
}

double
fastsim::Bremsstrahlung::gbteth(const double ener,
                const double partm,
                const double efrac,
                const RandomEngineAndDistribution & random) const 
{
    // Details on implementation here
    // http://www.dnp.fmph.uniba.sk/cernlib/asdoc/geant_html3/node299.html#GBTETH
    // http://svn.cern.ch/guest/AliRoot/tags/v3-07-03/GEANT321/gphys/gbteth.F

    const double alfa = 0.625;
    
    const double d = 0.13*(0.8+1.3/Z_)*(100.0+(1.0/ener))*(1.0+efrac);
    const double w1 = 9.0/(9.0+d);
    const double umax = ener*M_PI/partm;
    double u;
    
    do 
    {
        double beta = (random.flatShoot()<=w1) ? alfa : 3.0*alfa;
        u = -std::log(random.flatShoot()*random.flatShoot())/beta;
    }while (u >= umax);

    return u;
}

DEFINE_EDM_PLUGIN(
    fastsim::InteractionModelFactory,
    fastsim::Bremsstrahlung,
    "fastsim::Bremsstrahlung"
    );
