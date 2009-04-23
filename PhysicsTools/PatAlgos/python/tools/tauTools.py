import FWCore.ParameterSet.Config as cms

from PhysicsTools.PatAlgos.tools.helpers import *
   
def switchToCaloTau(process,layers=[0,1]):
    process.patLayer0.remove(process.pfRecoTauDiscriminationByIsolation)
    process.load("PhysicsTools.PatAlgos.cleaningLayer0.caloTauCleaner_cfi")
    process.patLayer0.remove(process.pfRecoTauDiscriminationByIsolation)
    process.patLayer0.replace(process.allLayer0Taus, process.allLayer0CaloTaus)
    process.patLayer0.replace(process.patPFTauDiscrimination, process.patCaloTauDiscrimination)
    # reconfigure MC, Trigger match and Layer 1 to use CaloTaus
    process.tauMatch.src                 = cms.InputTag('allLayer0CaloTaus')
    process.tauGenJetMatch.src           = cms.InputTag('allLayer0CaloTaus')
    massSearchReplaceParam(process.patTrigMatch, 'src', cms.InputTag("allLayer0Taus"), cms.InputTag('allLayer0CaloTaus'))
    massSearchReplaceParam(process.patTrigMatch_patTuple, 'src', cms.InputTag("allLayer0Taus"), cms.InputTag('allLayer0CaloTaus'))
    if layers.count(1) != 0:
        process.allLayer1Taus.tauSource      = cms.InputTag('allLayer0CaloTaus')
        process.allLayer1Taus.tauIDSources = cms.PSet(
                leadingTrackFinding = cms.InputTag("patCaloRecoTauDiscriminationByLeadingTrackFinding"),
                leadingTrackPtCut   = cms.InputTag("patCaloRecoTauDiscriminationByLeadingTrackPtCut"),
                byIsolation         = cms.InputTag("patCaloRecoTauDiscriminationByIsolation"),
                #againstElectron = cms.InputTag("patCaloRecoTauDiscriminationAgainstElectron"),  # Not on AOD
        )

