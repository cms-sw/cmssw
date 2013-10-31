import FWCore.ParameterSet.Config as cms

#from JetMETCorrections.Configuration.L2L3Corrections_Summer08Redigi_cff import *
from JetMETCorrections.Type1MET.MetType1Corrections_cff import metJESCorSC5CaloJet

metMuonJESCorSC5 = metJESCorSC5CaloJet.clone()
metMuonJESCorSC5.inputUncorJetsLabel = "sisCone5CaloJets"
metMuonJESCorSC5.corrector = "L2L3JetCorrectorSC5Calo"
metMuonJESCorSC5.inputUncorMetLabel = "corMetGlobalMuons"

metCorSequence = cms.Sequence(metMuonJESCorSC5)
