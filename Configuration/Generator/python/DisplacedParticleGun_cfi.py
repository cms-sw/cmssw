import math

import FWCore.ParameterSet.Config as cms


# Interface contract
# ------------------
#
# A generated particle is described by
#
#   1. a momentum distribution;
#   2. one transverse plane on which its spatial distribution is defined;
#   3. the z plane on which the HepMC production vertex is placed; and
#   4. optional regions through which the straight-line trajectory must pass.
#
# `SampleAt` is a physics choice, not merely an implementation strategy.  The
# sampled position is held fixed while direction candidates are retried.  This
# preserves the requested spatial distribution on that plane.  Momentum
# magnitude does not affect straight-line plane intersections, so it is sampled
# only after a direction has been accepted.
#
# The selected plane must provide RMin/RMax and PhiMin/PhiMax.  Bounds on the
# other planes are acceptance requirements.  An optional plane is disabled by
# omitting the entire PSet, not by pairing its parameters with an enable bool.
#
# Allowed values used below:
#
#   Momentum.Magnitude.Variable:       "energy", "pt"
#   Geometry.SampleAt:                 "origin", "production"
#   Geometry.RadialDistribution:       "uniformArea", "uniformRadius"
#
# Momentum PhiMin/PhiMax are global momentum azimuths.  PhiMin/PhiMax inside a
# geometry plane describe spatial positions on that plane.  Any correlation
# between them is produced by the configured intersection requirements.
#
# In this example the primary population is uniform in area at z=0.  Its HepMC
# vertices are created at z=321 cm and must lie in the configured production
# annulus.  The outgoing straight-line trajectory must subsequently cross the
# configured HGCAL back-surface annulus.

generator = cms.EDProducer(
    "DisplacedParticleGunProducer",
    PGunParameters=cms.PSet(
        PartID=cms.int32(22),
        NParticles=cms.int32(1),

        Momentum=cms.PSet(
            Magnitude=cms.PSet(
                Variable=cms.string("pt"),
                Min=cms.double(5.0),
                Max=cms.double(100.0),
            ),
            Direction=cms.PSet(
                ThetaMin=cms.double(1.0e-6),
                ThetaMax=cms.double(math.pi / 4.0),
                PhiMin=cms.double(-math.pi),
                PhiMax=cms.double(math.pi),
            ),
        ),

        Geometry=cms.PSet(
            # The region named here defines the generated spatial measure.
            # Valid choices are "origin" and "production".
            SampleAt=cms.string("origin"),
            RadialDistribution=cms.string("uniformArea"),

            # CMS coordinate-origin plane.  Its z coordinate is intrinsically
            # zero and therefore is not configurable.
            Origin=cms.PSet(
                RMin=cms.double(0.0),
                RMax=cms.double(100.0),
                PhiMin=cms.double(-math.pi),
                PhiMax=cms.double(math.pi),
            ),

            # The HepMC particle is always created on this plane.  When
            # SampleAt == "production", these bounds define the sampled
            # population.  Otherwise they are acceptance requirements.
            Production=cms.PSet(
                Z=cms.double(321.0),
                RMin=cms.double(60.0),
                RMax=cms.double(90.0),
                PhiMin=cms.double(-math.pi),
                PhiMax=cms.double(math.pi),
            ),

            # Optional as a whole.  Its presence requires the outgoing
            # straight-line trajectory to cross this region.  The descriptive
            # label is used in diagnostics only.
            Target=cms.PSet(
                Z=cms.double(362.18),
                RMin=cms.double(75.80),
                RMax=cms.double(120.23),
                PhiMin=cms.double(-math.pi),
                PhiMax=cms.double(math.pi),
            ),
        ),

        # Number of direction candidates tried for one fixed sampled position.
        # Exhaustion is a configuration/geometry failure; the primary point is
        # not silently resampled because that would change the requested
        # spatial distribution.
        MaxDirectionTries=cms.uint32(10000),
    ),

    Verbosity=cms.untracked.int32(0),
)
