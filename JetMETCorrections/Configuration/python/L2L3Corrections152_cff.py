import FWCore.ParameterSet.Config as cms

from JetMETCorrections.Configuration.L2RelativeCorrection152_cff import *
from JetMETCorrections.Configuration.L3AbsoluteCorrection152_cff import *
L2L3JetCorrectorIcone5 = cms.ESSource("JetCorrectionServiceChain",
    correctors = cms.vstring('L2RelativeJetCorrectorIcone5', 
        'L3AbsoluteJetCorrectorMcone5'),
    label = cms.string('L2L3JetCorrectorIcone5')
)


