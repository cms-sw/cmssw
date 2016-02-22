import FWCore.ParameterSet.Config as cms
bunchSpacingProducer = cms.EDProducer("BunchSpacingProducer")

from Configuration.StandardSequences.Eras import eras
eras.run2_50ns_specific.toModify( bunchSpacingProducer, bunchSpacingOverride = cms.uint32(50))
eras.run2_50ns_specific.toModify( bunchSpacingProducer, overrideBunchSpacing = cms.bool(True))
