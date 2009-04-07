import FWCore.ParameterSet.Config as cms

# produce associated jet correction factors in a valuemap
from PhysicsTools.PatAlgos.recoLayer0.jetCorrFactors_cfi import *

# MET correction for JES
from JetMETCorrections.Type1MET.MetType1Corrections_cff import *
from JetMETCorrections.Configuration.L2L3Corrections_Summer08_cff import *
corMetType1Icone5.corrector = cms.string('L2L3JetCorrectorIC5Calo')
patJetCorrections = cms.Sequence(jetCorrFactors)

# MET correction for Muons
from JetMETCorrections.Type1MET.MuonMETValueMapProducer_cff import *
from JetMETCorrections.Type1MET.MetMuonCorrections_cff import corMetGlobalMuons
corMetType1Icone5Muons = corMetGlobalMuons.clone(uncorMETInputTag = cms.InputTag('corMetType1Icone5'))
patMETCorrections = cms.Sequence(corMetType1Icone5 * muonMETValueMapProducer * corMetType1Icone5Muons)

# default PAT sequence for JetMET corrections before cleaners
patJetMETCorrections = cms.Sequence(patJetCorrections + patMETCorrections)

