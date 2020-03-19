#include "FastSimulation/Utilities/interface/RandomEngineAndDistribution.h"
#include "FastSimulation/SimplifiedGeometryPropagator/interface/Particle.h"
#include "FastSimulation/SimplifiedGeometryPropagator/interface/SimplifiedGeometry.h"
#include "FastSimulation/SimplifiedGeometryPropagator/interface/Constants.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <cmath>
#include <memory>

#include <Math/AxisAngle.h>

#include "FastSimulation/SimplifiedGeometryPropagator/interface/InteractionModelFactory.h"
#include "FastSimulation/SimplifiedGeometryPropagator/interface/InteractionModel.h"
#include "DataFormats/Math/interface/LorentzVector.h"
#include "DataFormats/Math/interface/Vector3D.h"

///////////////////////////////////////////////
// Author: Patrick Janot
// Date: 8-Jan-2004
//
// Revision: Class structure modified to match SimplifiedGeometryPropagator
//           Fixed a bug in which particles could no longer be on the layer after scattering
//           S. Kurz, 29 May 2017
//////////////////////////////////////////////////////////

typedef math::XYZVector XYZVector;

namespace fastsim {
  //! Implementation of multiple scattering in the tracker layers.
  /*!
        Computes the direction change by multiple scattering of a charged particle (assumes constant properties of material).
    */
  class MultipleScattering : public InteractionModel {
  public:
    //! Constructor.
    MultipleScattering(const std::string& name, const edm::ParameterSet& cfg);

    //! Default destructor.
    ~MultipleScattering() override { ; };

    //! Perform the interaction.
    /*!
            \param particle The particle that interacts with the matter.
            \param layer The detector layer that interacts with the particle.
            \param secondaries Particles that are produced in the interaction (if any).
            \param random The Random Engine.
        */
    void interact(Particle& particle,
                  const SimplifiedGeometry& layer,
                  std::vector<std::unique_ptr<Particle> >& secondaries,
                  const RandomEngineAndDistribution& random) override;

  private:
    //! Return an orthogonal vector.
    XYZVector orthogonal(const XYZVector& aVector) const;

    double minPt_;       //!< Cut on minimum pT of particle
    double radLenInCm_;  //!< Radiation length of material (usually silicon X0=9.360)
  };
}  // namespace fastsim

fastsim::MultipleScattering::MultipleScattering(const std::string& name, const edm::ParameterSet& cfg)
    : fastsim::InteractionModel(name) {
  // Set the minimal pT for interaction
  minPt_ = cfg.getParameter<double>("minPt");
  // Material properties
  radLenInCm_ = cfg.getParameter<double>("radLen");
}

void fastsim::MultipleScattering::interact(fastsim::Particle& particle,
                                           const SimplifiedGeometry& layer,
                                           std::vector<std::unique_ptr<fastsim::Particle> >& secondaries,
                                           const RandomEngineAndDistribution& random) {
  //
  // only charged particles
  //
  if (particle.charge() == 0) {
    return;
  }

  double radLengths = layer.getThickness(particle.position(), particle.momentum());
  //
  // no material
  //
  if (radLengths < 1E-10) {
    return;
  }

  // particle must have minimum pT
  if (particle.momentum().Pt() < minPt_) {
    return;
  }

  double p2 = particle.momentum().Vect().Mag2();
  double m2 = particle.momentum().mass2();
  double e = std::sqrt(p2 + m2);

  // This is p*beta
  double pbeta = p2 / e;

  // Average multiple scattering angle from Moliere radius
  // The sqrt(2) factor is because of the *space* angle
  double theta0 = 0.0136 / pbeta * particle.charge() * std::sqrt(2. * radLengths) * (1. + 0.038 * std::log(radLengths));

  // Generate multiple scattering space angle perpendicular to the particle motion
  double theta = random.gaussShoot(0., theta0);
  // Plus a random rotation angle around the particle motion
  double phi = 2. * M_PI * random.flatShoot();

  // The two rotations
  ROOT::Math::AxisAngle rotation1(orthogonal(particle.momentum().Vect()), theta);
  ROOT::Math::AxisAngle rotation2(particle.momentum().Vect(), phi);
  // Rotate!
  XYZVector rotated = rotation2((rotation1(particle.momentum().Vect())));
  particle.momentum().SetXYZT(rotated.X(), rotated.Y(), rotated.Z(), particle.momentum().E());

  // Generate mutiple scattering displacements in cm (assuming the detectors
  // are silicon only to determine the thickness) in the directions orthogonal
  // to the vector normal to the surface
  double xp = (cos(phi) * theta / 2. + random.gaussShoot(0., theta0) / sqrt(12.)) * radLengths * radLenInCm_;
  double yp = (sin(phi) * theta / 2. + random.gaussShoot(0., theta0) / sqrt(12.)) * radLengths * radLenInCm_;

  // Determine a unitary vector tangent to the surface
  XYZVector normal;
  if (layer.isForward()) {
    normal = XYZVector(0., 0., particle.momentum().Pz() > 0. ? 1. : -1.);
  } else {
    double norm = particle.position().Rho();
    normal = XYZVector(particle.position().X() / norm, particle.position().Y() / norm, 0.);
  }
  XYZVector tangent = orthogonal(normal);
  // The total displacement
  XYZVector delta = xp * tangent + yp * normal.Cross(tangent);
  // Translate!
  particle.position().SetXYZT(particle.position().X() + delta.X(),
                              particle.position().Y() + delta.Y(),
                              particle.position().Z() + delta.Z(),
                              particle.position().T());

  // Make sure particle is still physically on the layer (particle moved on a tangent but a barrel layer is bent)
  // Happens only in very very few cases
  if (!layer.isForward() && std::abs(layer.getGeomProperty() - particle.position().Rho()) > 1e-2) {
    // Scale positions so the radii agree
    double scalePos = layer.getGeomProperty() / particle.position().Rho();
    particle.position().SetXYZT(particle.position().X() * scalePos,
                                particle.position().Y() * scalePos,
                                particle.position().Z(),
                                particle.position().T());

    // Add a protection in case something goes wrong
    if (std::abs(layer.getGeomProperty() - particle.position().Rho()) > 1e-2) {
      throw cms::Exception("fastsim::MultipleScattering") << "particle no longer on layer's surface";
    }
  }
}

XYZVector fastsim::MultipleScattering::orthogonal(const XYZVector& aVector) const {
  double x = fabs(aVector.X());
  double y = fabs(aVector.Y());
  double z = fabs(aVector.Z());

  if (x < y)
    return x < z ? XYZVector(0., aVector.Z(), -aVector.Y()) : XYZVector(aVector.Y(), -aVector.X(), 0.);
  else
    return y < z ? XYZVector(-aVector.Z(), 0., aVector.X()) : XYZVector(aVector.Y(), -aVector.X(), 0.);
}

DEFINE_EDM_PLUGIN(fastsim::InteractionModelFactory, fastsim::MultipleScattering, "fastsim::MultipleScattering");
