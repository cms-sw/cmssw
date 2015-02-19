import FWCore.ParameterSet.Config as cms

#from PhysicsTools.PatAlgos.recoLayer0.jetCorrFactors_cfi import *
from JetMETCorrections.Type1MET.correctionTermsPfMetType1Type2_cff import *
from JetMETCorrections.Type1MET.correctionTermsCaloMet_cff import *
from JetMETCorrections.Type1MET.correctedMet_cff import *
#from JetMETCorrections.Configuration.JetCorrectionServicesAllAlgos_cff import *

#from JetMETCorrections.Type1MET.pfMETCorrections_cff import *

## for scheduled mode
patMETCorrections = cms.Sequence(correctionTermsCaloMet+caloMetT1+caloMetT1T2+correctionTermsPfMetType1Type2+pfMetT1+pfMetT1T2)
