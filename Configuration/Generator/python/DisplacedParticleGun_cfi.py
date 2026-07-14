import math

import FWCore.ParameterSet.Config as cms


# Displaced particle gun
# ----------------------
#
# The generator constructs a straight line from a point sampled on the z=0
# Origin plane.  It places the HepMC production vertex where that line crosses
# the Production plane and, when Target is present, requires the continuation
# of the line to cross the Target plane as well.  Each plane region is an
# annular sector described by [RMin, RMax] and [PhiMin, PhiMax].  Geometry
# lengths are in cm and angles are in radians.
#
# Sampling measure
# ----------------
#
# * Origin radius is sampled according to Geometry.RadialDistribution:
#     "uniformArea"   samples R^2 uniformly (constant density per unit area),
#     "uniformRadius" samples R uniformly.
#   Origin spatial phi is uniform in its configured interval.
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
#   For "energy", Min must exceed the generated particle's rest mass.
#
# Production and Target are constraints, not independently sampled spatial
# distributions.  Consequently, accepted points on those planes are generally
# not uniform over their annular sectors.
#
# Parameter constraints and behavior
# ----------------------------------
#
# * PartID is the signed PDG ID.  NParticles must be positive and gives the
#   number of independently sampled particles generated per event.
# * Magnitude.Min must be positive.  Plane radii must satisfy
#   0 <= RMin < RMax.  Every other Min/Max pair must also have Max > Min.
# * Direction theta must lie strictly inside (0, pi/2), so particles point in
#   the +z direction.
# * Origin is fixed at z=0.  Production.Z must be positive.  If Target is
#   present, Target.Z must be greater than Production.Z.
# * Omitting the entire Target PSet disables the target constraint.  A Target
#   assumes straight propagation beyond the production vertex and is therefore
#   supported only for neutral particles.
# * Momentum phi is a direction angle; plane phi is the azimuth of a position.
#   Phi intervals may be written outside [-pi, pi] to express a sector crossing
#   that boundary.  Production and Target intervals spanning at least 2*pi are
#   treated as covering all azimuths.
# * The RandomNumberGeneratorService must be configured by the process.
#
# The origin point remains fixed while at most MaxSamplingAttempts direction
# candidates are tried.  Exhaustion raises an exception rather than resampling
# the origin and biasing its configured distribution.  Narrow spatial-phi
# sectors can require a larger attempt limit even when the geometry is
# reachable.
#
# Origin is only a sampling reference and is not written to HepMC.  The event
# contains the Production vertex, with its time coordinate calculated from the
# origin-to-production path length and the generated particle's speed.  The
# producer writes a HepMCProduct under the "unsmeared" instance and a
# GenEventInfoProduct.  Verbosity values greater than zero enable event and
# particle printing.
#
# This example generates one photon.  Origin points are uniform in area at
# z=0; accepted lines place production vertices in the configured HGCAL CE-E
# front-surface sector at z=321 cm and cross the back-surface sector at
# z=362.18 cm.

generator = cms.EDProducer(
    "DisplacedParticleGunProducer",
    PGunParameters=cms.PSet(
        PartID=cms.int32(22),
        NParticles=cms.int32(1),

        Momentum=cms.PSet(
            Magnitude=cms.PSet(
                # "pt" or "energy"; bounds are in GeV.
                Variable=cms.string("pt"),
                Min=cms.double(5.0),
                Max=cms.double(100.0),
            ),
            Direction=cms.PSet(
                # Polar and azimuthal momentum-angle bounds in radians.
                ThetaMin=cms.double(1.0e-6),
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

            # HGCAL CE-E front-surface sector containing the production vertex.
            Production=cms.PSet(
                Z=cms.double(321.0),
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
