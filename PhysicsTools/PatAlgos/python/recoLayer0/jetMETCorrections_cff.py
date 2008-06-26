import FWCore.ParameterSet.Config as cms

# default JetMET calibration on IC, KT and MC Jets
from JetMETCorrections.Configuration.MCJetCorrections152_cff import *

# define jet flavour correction services
from JetMETCorrections.Configuration.L5FlavorCorrections_cff import *

# produce associated jet correction factors in a valuemap
from PhysicsTools.PatAlgos.recoLayer0.jetCorrFactors_cfi import *

import JetMETCorrections.Type1MET.corMetMuons_cfi
# muon MET correction maker 
corMetType1Icone5Muons = JetMETCorrections.Type1MET.corMetMuons_cfi.corMetMuons.clone()
# JetMET corrections for muons: input jet-corrected MET
corMetType1Icone5Muons.inputUncorMetLabel = cms.InputTag('corMetType1Icone5')

# It would be better to get this config to JetMETCorrections/Type1MET/data/ at some point
corMetType1Icone5Muons.TrackAssociatorParameters.useEcal = False ## RecoHits
corMetType1Icone5Muons.TrackAssociatorParameters.useHcal = False ## RecoHits
corMetType1Icone5Muons.TrackAssociatorParameters.useHO = False ## RecoHits
corMetType1Icone5Muons.TrackAssociatorParameters.useCalo = True ## CaloTowers
corMetType1Icone5Muons.TrackAssociatorParameters.useMuon = False ## RecoHits
corMetType1Icone5Muons.TrackAssociatorParameters.truthMatch = False


# re-key jet energy corrections to layer 0 output
layer0JetCorrFactors = cms.EDFilter("JetCorrFactorsValueMapSkimmer",
    collection  = cms.InputTag("allLayer0Jets"),
    backrefs    = cms.InputTag("allLayer0Jets"),
    association = cms.InputTag("jetCorrFactors"),
)

# MET corrections from JES
from JetMETCorrections.Type1MET.MetType1Corrections_cff import *

# default PAT sequence for JetMET corrections before cleaners
patAODJetMETCorrections = cms.Sequence(jetCorrFactors * corMetType1Icone5 * corMetType1Icone5Muons)

# default PAT sequence for JetMET corrections after cleaners
patLayer0JetMETCorrections = cms.Sequence(layer0JetCorrFactors)

