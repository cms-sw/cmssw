import FWCore.ParameterSet.Config as cms
bunchSpacingProducer = cms.EDProducer("BunchSpacingProducer")

from Configuration.Eras.Modifier_run2_50ns_specific_cff import run2_50ns_specific
run2_50ns_specific.toModify( bunchSpacingProducer, bunchSpacingOverride = cms.uint32(50))
run2_50ns_specific.toModify( bunchSpacingProducer, overrideBunchSpacing = cms.bool(True))
