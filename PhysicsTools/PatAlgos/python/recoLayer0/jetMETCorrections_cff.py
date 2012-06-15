import FWCore.ParameterSet.Config as cms

# produce associated jet correction factors in a valuemap
from PhysicsTools.PatAlgos.recoLayer0.jetCorrFactors_cfi import *
patJetCorrections = cms.Sequence(patJetCorrFactors)

# MET correction for JES
from JetMETCorrections.Type1MET.caloMETCorrections_cff import *
from JetMETCorrections.Type1MET.pfMETCorrections_cff import *
#from JetMETCorrections.Configuration.JetCorrectionCondDB_cff import *
from JetMETCorrections.Configuration.JetCorrectionServicesAllAlgos_cff import *

# MET corrections
patMETCorrections = cms.Sequence(produceCaloMETCorrections+producePFMETCorrections)

# default PAT sequence for JetMET corrections before cleaners
patJetMETCorrections = cms.Sequence(patJetCorrections)


