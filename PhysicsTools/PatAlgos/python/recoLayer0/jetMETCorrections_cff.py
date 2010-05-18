import FWCore.ParameterSet.Config as cms

# produce associated jet correction factors in a valuemap
from PhysicsTools.PatAlgos.recoLayer0.jetCorrFactors_cfi import *
patJetCorrections = cms.Sequence(patJetCorrFactors)

# MET correction for JES
from JetMETCorrections.Type1MET.MetType1Corrections_cff import *
#from JetMETCorrections.Configuration.JetCorrectionCondDB_cff import *
from JetMETCorrections.Configuration.JetCorrectionServicesAllAlgos_cff import *

# MET correction for Muons
from RecoMET.METProducers.MuonMETValueMapProducer_cff import *
from RecoMET.METProducers.MetMuonCorrections_cff import corMetGlobalMuons
metJESCorAK5CaloJetMuons = corMetGlobalMuons.clone(uncorMETInputTag = cms.InputTag('metJESCorAK5CaloJet'))
patMETCorrections = cms.Sequence(metJESCorAK5CaloJet * metJESCorAK5CaloJetMuons)

# default PAT sequence for JetMET corrections before cleaners
patJetMETCorrections = cms.Sequence(patJetCorrections)


