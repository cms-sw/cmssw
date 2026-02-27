from PhysicsTools.PatAlgos.producersLayer1.tauProducer_cfi import *
patTaus.addGenMatch      = cms.bool(False)
patTaus.embedGenMatch    = cms.bool(False)
patTaus.addGenJetMatch   = cms.bool(False)
patTaus.embedGenJetMatch = cms.bool(False)

from Configuration.StandardSequences.MagneticField_cff import *

patAlgosToolsTask = cms.Task()
patAlgosToolsTask.add(patTaus)

HLTTauDQMOfflineTauProducer = cms.Sequence(patAlgosToolsTask)
