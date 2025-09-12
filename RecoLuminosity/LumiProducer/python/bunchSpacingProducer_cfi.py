import FWCore.ParameterSet.Config as cms

from RecoLuminosity.LumiProducer.default_bunchSpacingProducer_cfi import default_bunchSpacingProducer as _bunchSpacingProducer

bunchSpacingProducer = _bunchSpacingProducer.clone()
from Configuration.Eras.Modifier_run2_50ns_specific_cff import run2_50ns_specific
run2_50ns_specific.toModify( bunchSpacingProducer, bunchSpacingOverride = 50)
run2_50ns_specific.toModify( bunchSpacingProducer, overrideBunchSpacing = True)

##
## Turn on bunch spacing Producer for tau embedding cleaning step
##
from Configuration.ProcessModifiers.tau_embedding_cleaning_cff import tau_embedding_cleaning
tau_embedding_cleaning.toModify(
    bunchSpacingProducer,
    bunchSpacingOverride=cms.uint32(25),
    overrideBunchSpacing=cms.bool(True),
)
