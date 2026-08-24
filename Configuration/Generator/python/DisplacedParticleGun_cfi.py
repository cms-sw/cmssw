import math

import FWCore.ParameterSet.Config as cms


# Displaced particle gun
# ----------------------
#
# The generator constructs a straight line from a point sampled on the z=0
# Origin plane.  The HepMC production vertex is placed where that line crosses
# the Production plane.  When Target is present, the line must cross that plane
# as well.  Geometry lengths are in cm and angles are in radians.
#
# Sampling measure
# ----------------
#
# * Origin radius is sampled according to Geometry.RadialDistribution:
#     "uniformArea"   samples R^2 uniformly (constant density per unit area),
#     "uniformRadius" samples R uniformly.
#   Origin phi is sampled uniformly.
#
# * Direction phi is drawn uniformly from Momentum.Direction.PhiMin/PhiMax.
#   For that phi, the producer determines the theta intervals that satisfy the
#   radial bounds of Production and Target, intersects them with the configured
#   theta interval, and samples theta uniformly over the surviving intervals.
#   Spatial-phi bounds on the destination planes are then checked explicitly.
#
# * Momentum magnitude is sampled only after the geometry accepts a direction:
#     "pt"     samples transverse momentum uniformly in [Min, Max] in GeV;
#     "energy" samples total energy uniformly in [Min, Max] in GeV.
#   For "energy", Min must exceed the particle mass.  For "pt", ThetaMin
#   must be greater than zero.
#
# Production and Target are constraints, not independently sampled spatial
# distributions.  Consequently, accepted points on those planes are generally
# not uniform over their configured regions.  Their spatial-phi bounds are
# checked after theta sampling and may cause a direction candidate to be
# rejected.  Origin phi is sampled directly and does not cause rejection.
#
# Parameter constraints and behavior
# ----------------------------------
#
# * PartID is the signed PDG ID.  NParticles must be positive and gives the
#   number of independently sampled particles generated per event.
# * Magnitude.Min must be positive.  Plane radii must satisfy
#   0 <= RMin < RMax.
# * Direction bounds must satisfy 0 <= ThetaMin < ThetaMax < pi/2.  Phi bounds
#   must satisfy -pi <= PhiMin < PhiMax <= pi.
# * Origin is fixed at z=0.  Production.Z must be positive.  If Target is
#   present, Target.Z must be greater than Production.Z.
# * Omitting the entire Target PSet disables the target constraint.  A Target
#   assumes straight propagation beyond the production vertex and is therefore
#   supported only for neutral particles.
# * Momentum phi is a direction angle; plane phi is the azimuth of a position.
#
# The origin point remains fixed while at most MaxSamplingAttempts direction
# candidates are tried.  Exhaustion raises an exception rather than resampling
# the origin and biasing its configured distribution.
#
# This example generates one photon.  Origin points are uniform in area at
# z=0, production vertices are placed at z=319 cm before HGCAL, and accepted
# trajectories cross the HGCAL CE-E back surface at z=362.18 cm.

generator = cms.EDProducer(
    "DisplacedParticleGunProducer",
    PGunParameters=cms.PSet(
        PartID=cms.int32(22),
        NParticles=cms.int32(1),

        Momentum=cms.PSet(
            Magnitude=cms.PSet(
                # "pt" or "energy"; bounds are in GeV.
                Variable=cms.string("energy"),
                Min=cms.double(5.0),
                Max=cms.double(100.0),
            ),
            Direction=cms.PSet(
                # Polar and azimuthal momentum-angle bounds in radians.
                ThetaMin=cms.double(0.0),
                ThetaMax=cms.double(math.pi / 4.0),
                PhiMin=cms.double(-math.pi),
                PhiMax=cms.double(math.pi),
            ),
        ),

        Geometry=cms.PSet(
            # Controls only the radial measure used to sample Origin.
            RadialDistribution=cms.string("uniformArea"),

            # Sampled annular sector at z=0.  Radii are in cm, phi in radians.
            Origin=cms.PSet(
                RMin=cms.double(0.0),
                RMax=cms.double(100.0),
                PhiMin=cms.double(-math.pi),
                PhiMax=cms.double(math.pi),
            ),

            # Production region before HGCAL.
            Production=cms.PSet(
                Z=cms.double(319.0),
                RMin=cms.double(60.0),
                RMax=cms.double(90.0),
                PhiMin=cms.double(-math.pi),
                PhiMax=cms.double(math.pi),
            ),

            # Optional HGCAL CE-E back-surface sector that the post-production
            # trajectory must cross.  Remove the PSet to disable this constraint.
            Target=cms.PSet(
                Z=cms.double(362.18),
                RMin=cms.double(75.80),
                RMax=cms.double(120.23),
                PhiMin=cms.double(-math.pi),
                PhiMax=cms.double(math.pi),
            ),
        ),

        # Maximum direction candidates for each fixed sampled Origin point.
        MaxSamplingAttempts=cms.uint32(10000),
    ),

    Verbosity=cms.untracked.int32(0),
)
