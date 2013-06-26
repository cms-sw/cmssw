import FWCore.ParameterSet.Config as cms
#MET
mc_metMin = cms.double(0.0)
#HT
mc_ptJetForHtMin = cms.double(0.0)
mc_htMin = cms.double(0.0)
#Jets
mc_nJet = cms.int32(0)
mc_ptJetMin = cms.double(0.0)
#Photons
mc_nPhot = cms.int32(0)
mc_ptPhotMin = cms.double(0.0)
#Electrons
mc_nElec = cms.int32(3)
mc_nElecRule = cms.string("strictEqual")
mc_ptElecMin = cms.double(5.0)
#Muons
mc_nMuon = cms.int32(0)
mc_nMuonRule = cms.string("strictEqual")
mc_ptMuonMin = cms.double(5.0)
#Taus
mc_nTau = cms.int32(0)
mc_ptTauMin = cms.double(0.0)

