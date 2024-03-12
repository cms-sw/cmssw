import FWCore.ParameterSet.Config as cms

from RecoTauTag.Configuration.updateHPSPFTaus_cff import *

patFixedConePFTauDiscrimination = cms.Sequence()

patHPSPFTauDiscriminationTask = cms.Task(updateHPSPFTausTask)
patHPSPFTauDiscrimination = cms.Sequence(patHPSPFTauDiscriminationTask)

patShrinkingConePFTauDiscrimination = cms.Sequence()
# foo bar baz
# zQS9wu2mZg2Su
# 6zbX1k7aB8Uoi
