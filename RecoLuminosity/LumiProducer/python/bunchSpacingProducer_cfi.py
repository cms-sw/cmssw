import FWCore.ParameterSet.Config as cms

import RecoLuminosity.LumiProducer.BunchSpacingProducer_cfi as _mod

bunchSpacingProducer = _mod.BunchSpacingProducer.clone()

from Configuration.Eras.Modifier_run2_50ns_specific_cff import run2_50ns_specific
run2_50ns_specific.toModify( bunchSpacingProducer, bunchSpacingOverride = 50)
run2_50ns_specific.toModify( bunchSpacingProducer, overrideBunchSpacing = True)
