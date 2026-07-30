import FWCore.ParameterSet.Config as cms

# This modifier enables the portable (alpaka) HLT electron pixel-seed matching,
# replacing ElectronNHitSeedProducer with the ElectronNHitSeedAlpakaProducer +
# ElectronSeedConverter pair.
# Needs to be used on top of the alpaka modifier.
egammaPixelSeedingPortable = cms.Modifier()
