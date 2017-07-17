import FWCore.ParameterSet.Config as cms

#from PhysicsTools.PatAlgos.recoLayer0.jetCorrFactors_cfi import *
from JetMETCorrections.Type1MET.correctionTermsPfMetType1Type2_cff import *
from JetMETCorrections.Type1MET.correctionTermsCaloMet_cff import *
from JetMETCorrections.Type1MET.correctedMet_cff import caloMetT1, caloMetT1T2, pfMetT1, pfMetT1T2
#from JetMETCorrections.Configuration.JetCorrectionServicesAllAlgos_cff import *

#from JetMETCorrections.Type1MET.pfMETCorrections_cff import *

## for scheduled mode
patMETCorrectionsTask = cms.Task(correctionTermsCaloMetTask, caloMetT1, caloMetT1T2, correctionTermsPfMetType1Type2Task, pfMetT1, pfMetT1T2)

patMETCorrections = cms.Sequence(patMETCorrectionsTask)
