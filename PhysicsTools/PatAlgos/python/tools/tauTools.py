import FWCore.ParameterSet.Config as cms

from PhysicsTools.PatAlgos.tools.coreTools import *

def redoPFTauDiscriminators(process, oldPFTauLabel=cms.InputTag('pfRecoTauProducer'), newPFTauLabel=cms.InputTag('pfRecoTauProducer')):
    process.patAODExtraReco += process.patPFTauDiscrimination
    massSearchReplaceParam(process.patPFTauDiscrimination, 'PFTauProducer', oldPFTauLabel, newPFTauLabel)
       
def switchToCaloTau(process,pfTauLabel=cms.InputTag('pfRecoTauProducer'), caloTauLabel=cms.InputTag('caloRecoTauProducer')):
    # swap the discriminators we compute # FIXME: remove if they're already on AOD
    switchMCAndTriggerMatch(process,pfTauLabel,caloTauLabel)
    process.allLayer1Taus.tauSource      = caloTauLabel
    process.allLayer1Taus.tauIDSources = cms.PSet(  # All these are already present in 2.2.X AODSIMs
            leadingTrackFinding = cms.InputTag("caloRecoTauDiscriminationByLeadingTrackFinding"),
            leadingTrackPtCut   = cms.InputTag("caloRecoTauDiscriminationByLeadingTrackPtCut"),
            byIsolation         = cms.InputTag("caloRecoTauDiscriminationByIsolation"),
            againstElectron     = cms.InputTag("caloRecoTauDiscriminationAgainstElectron"),  
    )
    if pfTauLabel in process.aodSummary.candidates:
        process.aodSummary.candidates[process.aodSummary.candidates.index(pfTauLabel)] = caloTauLabel
    else:
        process.aodSummary.candidates += [caloTauLabel]
