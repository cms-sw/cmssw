import FWCore.ParameterSet.Config as cms

#from PhysicsTools.PatAlgos.recoLayer0.jetCorrFactors_cfi import *
from JetMETCorrections.Type1MET.pfMETCorrections_cff import *
from JetMETCorrections.Type1MET.caloMETCorrections_cff import *
#from JetMETCorrections.Configuration.JetCorrectionServicesAllAlgos_cff import *

## for scheduled mode
patMETCorrections = cms.Sequence(produceCaloMETCorrections+producePFMETCorrections)
