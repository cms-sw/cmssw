import FWCore.ParameterSet.Config as cms

# default JetMET calibration on IC, KT and MC Jets
from JetMETCorrections.Configuration.MCJetCorrections152_cff import *

# define jet flavour correction services
from JetMETCorrections.Configuration.L5FlavorCorrections_cff import *

# produce associated jet correction factors in a valuemap
from PhysicsTools.PatAlgos.recoLayer0.jetCorrFactors_cfi import *

# re-key jet energy corrections to layer 0 output
layer0JetCorrFactors = cms.EDFilter("JetCorrFactorsValueMapSkimmer",
    collection  = cms.InputTag("allLayer0Jets"),
    backrefs    = cms.InputTag("allLayer0Jets"),
    association = cms.InputTag("jetCorrFactors"),
)

# MET corrections from JES
from JetMETCorrections.Type1MET.MetType1Corrections_cff import *

# default PAT sequence for JetMET corrections before cleaners
patAODJetMETCorrections = cms.Sequence(jetCorrFactors * corMetType1Icone5)

# default PAT sequence for JetMET corrections after cleaners
patLayer0JetMETCorrections = cms.Sequence(layer0JetCorrFactors)

