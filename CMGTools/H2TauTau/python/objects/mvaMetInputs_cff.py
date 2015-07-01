import FWCore.ParameterSet.Config as cms

from RecoMET.METPUSubtraction.mvaPFMET_cff import puJetIdForPFMVAMEt, calibratedAK4PFJetsForPFMVAMEt


puJetIdForPFMVAMEt.jec =  cms.string('AK4PF')
#process.puJetIdForPFMVAMEt.jets = cms.InputTag("ak4PFJets")
puJetIdForPFMVAMEt.vertexes = cms.InputTag("offlineSlimmedPrimaryVertices")
puJetIdForPFMVAMEt.rho = cms.InputTag("fixedGridRhoFastjetAll")

mvaMetInputSequence = cms.Sequence(
    calibratedAK4PFJetsForPFMVAMEt*
    puJetIdForPFMVAMEt
  )
