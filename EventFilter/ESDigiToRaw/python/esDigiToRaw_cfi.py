import FWCore.ParameterSet.Config as cms

esDigiToRaw = cms.EDProducer("ESDigiToRaw",
    debugMode = cms.untracked.bool(False),
    InstanceES = cms.string(''),
    Label = cms.string('simEcalPreshowerDigis'),
    LookupTable = cms.untracked.FileInPath('EventFilter/ESDigiToRaw/data/ES_lookup_table.dat')
)

# bypass zero suppression
from Configuration.ProcessModifiers.premix_stage1_cff import premix_stage1
premix_stage1.toModify(esDigiToRaw, Label = 'mix')

