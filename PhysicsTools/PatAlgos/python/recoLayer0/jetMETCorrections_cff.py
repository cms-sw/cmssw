import FWCore.ParameterSet.Config as cms

# produce associated jet correction factors in a valuemap
from PhysicsTools.PatAlgos.recoLayer0.jetCorrFactors_cfi import *

# MET correction for JES
from JetMETCorrections.Type1MET.MetType1Corrections_cff import *
from JetMETCorrections.Configuration.L2L3Corrections_Summer09_cff import *
metJESCorAK5CaloJet.corrector = cms.string('L2L3JetCorrectorAK5Calo')
patJetCorrections = cms.Sequence(jetCorrFactors)

# MET correction for Muons
from JetMETCorrections.Type1MET.MuonMETValueMapProducer_cff import *
from JetMETCorrections.Type1MET.MetMuonCorrections_cff import corMetGlobalMuons
metJESCorAK5CaloJetMuons = corMetGlobalMuons.clone(uncorMETInputTag = cms.InputTag('metJESCorAK5CaloJet'))
patMETCorrections = cms.Sequence(metJESCorAK5CaloJet * metJESCorAK5CaloJetMuons)

# default PAT sequence for JetMET corrections before cleaners
patJetMETCorrections = cms.Sequence(patJetCorrections + patMETCorrections)

