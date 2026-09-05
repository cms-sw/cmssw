import FWCore.ParameterSet.Config as cms

# This modifier enables the portable (alpaka) HLT electron pixel-seed matching,
# replacing ElectronNHitSeedProducer with the ElectronNHitSeedAlpakaProducer +
# ElectronSeedConverter pair.
egammaPixelSeedingPortable = cms.Modifier()
