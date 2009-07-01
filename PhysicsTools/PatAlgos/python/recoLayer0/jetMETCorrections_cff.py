import FWCore.ParameterSet.Config as cms

# produce associated jet correction factors in a valuemap
from PhysicsTools.PatAlgos.recoLayer0.jetCorrFactors_cfi import *

# MET correction for JES
from JetMETCorrections.Type1MET.MetType1Corrections_cff import *
from JetMETCorrections.Configuration.L2L3Corrections_Summer08_cff import *
metJESCorIC5CaloJet.corrector = cms.string('L2L3JetCorrectorIC5Calo')
patJetCorrections = cms.Sequence(jetCorrFactors)

# MET correction for Muons
from JetMETCorrections.Type1MET.MuonMETValueMapProducer_cff import *
from JetMETCorrections.Type1MET.MetMuonCorrections_cff import corMetGlobalMuons
metJESCorIC5CaloJetMuons = corMetGlobalMuons.clone(uncorMETInputTag = cms.InputTag('metJESCorIC5CaloJet'))
patMETCorrections = cms.Sequence(metJESCorIC5CaloJet * metJESCorIC5CaloJetMuons)

# default PAT sequence for JetMET corrections before cleaners
patJetMETCorrections = cms.Sequence(patJetCorrections + patMETCorrections)

