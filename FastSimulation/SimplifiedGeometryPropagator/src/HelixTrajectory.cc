#include "FastSimulation/SimplifiedGeometryPropagator/interface/HelixTrajectory.h"
#include "FastSimulation/SimplifiedGeometryPropagator/interface/StraightTrajectory.h"
#include "FastSimulation/SimplifiedGeometryPropagator/interface/BarrelSimplifiedGeometry.h"
#include "FastSimulation/SimplifiedGeometryPropagator/interface/ForwardSimplifiedGeometry.h"
#include "FastSimulation/SimplifiedGeometryPropagator/interface/Particle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FastSimulation/SimplifiedGeometryPropagator/interface/Constants.h"
#include <cmath>

// helix phi definition
// ranges from 0 to 2PI
// 0 corresponds to the positive x direction
// phi increases counterclockwise

fastsim::HelixTrajectory::HelixTrajectory(const fastsim::Particle & particle,double magneticFieldZ)
    : Trajectory(particle)
    // exact: r = gamma*beta*m_0*c / (q*e*B) = p_T / (q * e * B)
    // momentum in units of GeV/c: r = p_T * 10^9 / (c * q * B)
    // in cmssw units: r = p_T / (c * 10^-4 * q * B)
    , radius_(std::abs(momentum_.Pt() / (fastsim::Constants::speedOfLight * 1e-4 * particle.charge() * magneticFieldZ)))
    , phi_(std::atan(momentum_.Py()/momentum_.Px()) + (momentum_.Px()*particle.charge() < 0 ? 3.*M_PI/2. : M_PI/2. ))
    // maybe consider (for -pi/2<x<pi/2)
    // cos(atan(x)) = 1 / sqrt(x^2+1)
    // -> cos(atan(x) + pi/2)  = - x / sqrt(x^2+1)
    // -> cos(atan(x) +3*pi/2) = + x / sqrt(x^2+1)
    // sin(atan(x)) = x / sqrt(x^2+1)
    // -> sin(atan(x) + pi/2)  = + 1 / sqrt(x^2+1)
    // -> sin(atan(x) +3*pi/2) = - 1 / sqrt(x^2+1)
    , centerX_(position_.X() - radius_ * (momentum_.Py()/momentum_.Px()) / std::sqrt((momentum_.Py()/momentum_.Px())*(momentum_.Py()/momentum_.Px())+1) * (momentum_.Px()*particle.charge() < 0 ? 1. : -1.))
    , centerY_(position_.Y() - radius_ * 1                               / std::sqrt((momentum_.Py()/momentum_.Px())*(momentum_.Py()/momentum_.Px())+1) * (momentum_.Px()*particle.charge() < 0 ? -1. : 1.))
    //, centerX_(position_.X() - radius_*std::cos(phi_))
    //, centerY_(position_.Y() - radius_*std::sin(phi_))
    , centerR_(std::sqrt(centerX_*centerX_ + centerY_*centerY_))
    , minR_(std::abs(centerR_ - radius_))
    , maxR_(centerR_ + radius_)
    // omega = q * e * B / (gamma * m) = q * e *B / (E / c^2) = q * e * B * c^2 / E
    // omega: negative for negative q -> seems to be what we want.
    // energy in units of GeV: omega = q * B * c^2 / (E * 10^9)
    // in cmssw units: omega[1/ns] = q * B * c^2 * 10^-4 / E
    , phiSpeed_(-particle.charge() * magneticFieldZ * fastsim::Constants::speedOfLight * fastsim::Constants::speedOfLight * 1e-4 / momentum_.E())
{;}

bool fastsim::HelixTrajectory::crosses(const BarrelSimplifiedGeometry & layer) const
{
    return (minR_ < layer.getRadius() && maxR_ > layer.getRadius());
}

double fastsim::HelixTrajectory::nextCrossingTimeC(const BarrelSimplifiedGeometry & layer, bool onLayer) const
{
    if(!crosses(layer)) return -1;

    // solve the following equation for sin(phi)
    // (x^2 + y^2 = R_L^2)     (1)      the layer 
    // x = x_c + R_H*cos(phi)  (2)      the helix in the xy plane
    // y = y_c + R_H*sin(phi)  (3)      the helix in the xy plane
    // with
    // R_L: the radius of the layer
    // x_c,y_c the center of the helix in xy plane
    // R_H, the radius of the helix
    // phi, the phase of the helix
    //
    // substitute (2) and (3) in (1)
    // =>
    //   x_c^2 + 2*x_c*R_H*cos(phi) + R_H^2*cos^2(phi)
    // + y_c^2 + 2*y_c*R_H*sin(phi) + R_H^2*sin^2(phi)
    // = R_L^2
    // =>
    // (x_c^2 + y_c^2 + R_H^2 - R_L^2) + (2*y_c*R_H)*sin(phi) = -(2*x_c*R_H)*cos(phi)
    //
    // rewrite
    //               E                 +       F    *sin(phi) =      G     *cos(phi)
    // =>
    // E^2 + 2*E*F*sin(phi) + F^2*sin^2(phi) = G^2*(1-sin^2(phi))
    // rearrange
    // sin^2(phi)*(F^2 + G^2) + sin(phi)*(2*E*F) + (E^2 - G^2) = 0
    //
    // rewrite
    // sin^2(phi)*     a      + sin(phi)*   b    +      c      = 0
    // => sin(phi) = (-b +/- sqrt(b^2 - 4*ac)) / (2*a)
    // with
    // a = F^2 + G^2
    // b = 2*E*F
    // c = E^2 - G^2

    double E = centerX_*centerX_ + centerY_*centerY_ + radius_*radius_ - layer.getRadius()*layer.getRadius();
    double F = 2*centerY_*radius_;
    double G = 2*centerX_*radius_;

    double a = F*F + G*G;
    double b = 2*E*F;
    double c = E*E - G*G;

    double delta = b*b - 4*a*c;

    // case of no solution
    if(delta < 0)
    {   
        // Should not be reached: Full Propagation does always have a solution "if(crosses(layer)) == -1"
        // Even if particle is outside all layers -> can turn around in magnetic field
        return -1;
    }

    // Uses a numerically more stable procedure:
    // https://people.csail.mit.edu/bkph/articles/Quadratics.pdf
    double sqrtDelta = sqrt(delta);
    double phi1 = 0, phi2 = 0;
    if(b < 0){
        phi1 = std::asin((2.*c) / (-b + sqrtDelta));
        phi2 = std::asin((-b + sqrtDelta) / (2.*a));
    }else{
        phi1 = std::asin((-b - sqrtDelta) / (2.*a));
        phi2 = std::asin((2.*c) / (-b - sqrtDelta));
    }

    // asin is ambiguous, make sure to have the right solution
    if(std::abs(layer.getRadius() - getRadParticle(phi1)) > 1.0e-2){
        phi1 = - phi1 + M_PI;
    }
    if(std::abs(layer.getRadius() - getRadParticle(phi2)) > 1.0e-2){
        phi2 = - phi2 + M_PI;
    }

    // another ambiguity
    if(phi1 < 0){
        phi1 += 2. * M_PI;
    }
    if(phi2 < 0){
        phi2 += 2. * M_PI;
    }

    // find the corresponding times when the intersection occurs
    // make sure they are positive
    double t1 = (phi1 - phi_)/phiSpeed_;
    while(t1 < 0)
    {
       t1 += 2*M_PI/std::abs(phiSpeed_);
    }
    double t2 = (phi2 - phi_)/phiSpeed_;
    while(t2 < 0)
    {
       t2 += 2*M_PI/std::abs(phiSpeed_);
    }

    // Check if propagation successful (numerical reasons): both solutions (phi1, phi2) have to be on the layer (same radius)
    // Can happen due to numerical instabilities of geometrical function (if momentum is almost parallel to x/y axis)
    // Get crossingTimeC from StraightTrajectory as good approximation
    if(std::abs(layer.getRadius() - getRadParticle(phi1)) > 1.0e-2 
        || std::abs(layer.getRadius() - getRadParticle(phi2)) > 1.0e-2)
    {
        StraightTrajectory traj(*this);
        return traj.nextCrossingTimeC(layer, onLayer);
    }

    // if the particle is already on the layer, we need to make sure the 2nd solution is picked.
    // happens if particle turns around in the magnetic field instead of hitting the next layer
    if(onLayer)
    {
        bool particleMovesInwards = momentum_.X()*position_.X() + momentum_.Y()*position_.Y() < 0;

        double posX1 = centerX_ + radius_*std::cos(phi1);
        double posY1 = centerY_ + radius_*std::sin(phi1);
        double momX1 = momentum_.X()*std::cos(phi1 - phi_) - momentum_.Y()*std::sin(phi1 - phi_);
        double momY1 = momentum_.X()*std::sin(phi1 - phi_) + momentum_.Y()*std::cos(phi1 - phi_);
        bool particleMovesInwards1 = momX1*posX1 + momY1*posY1 < 0;

        double posX2 = centerX_ + radius_*std::cos(phi2);
        double posY2 = centerY_ + radius_*std::sin(phi2);
        double momX2 = momentum_.X()*std::cos(phi2 - phi_) - momentum_.Y()*std::sin(phi2 - phi_);
        double momY2 = momentum_.X()*std::sin(phi2 - phi_) + momentum_.Y()*std::cos(phi2 - phi_);
        bool particleMovesInwards2 = momX2*posX2 + momY2*posY2 < 0;

        if(particleMovesInwards1 != particleMovesInwards)
        {
            return t1*fastsim::Constants::speedOfLight;
        }
        else if(particleMovesInwards2 != particleMovesInwards)
        {
            return t2*fastsim::Constants::speedOfLight;
        }
        // try to catch numerical issues again..
        else
        {
            return -1;
        }
    }

    return std::min(t1,t2)*fastsim::Constants::speedOfLight;
}


void fastsim::HelixTrajectory::move(double deltaTimeC)
{
    double deltaT = deltaTimeC/fastsim::Constants::speedOfLight;
    double deltaPhi = phiSpeed_*deltaT;
    position_.SetXYZT(
       centerX_ + radius_*std::cos(phi_ + deltaPhi),
       centerY_ + radius_*std::sin(phi_ + deltaPhi),
       position_.Z() + momentum_.Z()/momentum_.E()*deltaTimeC,
       position_.T() + deltaT);
    // Rotation defined by
    // x' = x cos θ - y sin θ
    // y' = x sin θ + y cos θ
    momentum_.SetXYZT(
       momentum_.X()*std::cos(deltaPhi) - momentum_.Y()*std::sin(deltaPhi),
       momentum_.X()*std::sin(deltaPhi) + momentum_.Y()*std::cos(deltaPhi),
       momentum_.Z(),
       momentum_.E());
}

double fastsim::HelixTrajectory::getRadParticle(double phi) const
{
    return sqrt((centerX_ + radius_*std::cos(phi))*(centerX_ + radius_*std::cos(phi)) + (centerY_ + radius_*std::sin(phi))*(centerY_ + radius_*std::sin(phi)));
}
