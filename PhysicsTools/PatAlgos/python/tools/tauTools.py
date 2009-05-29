import FWCore.ParameterSet.Config as cms

from PhysicsTools.PatAlgos.tools.coreTools import *

def redoPFTauDiscriminators(process, oldPFTauLabel=cms.InputTag('pfRecoTauProducer'), newPFTauLabel=cms.InputTag('pfRecoTauProducer')):
    process.patAODExtraReco += process.patPFTauDiscrimination
    massSearchReplaceParam(process.patPFTauDiscrimination, 'PFTauProducer', oldPFTauLabel, newPFTauLabel)
       
def switchToCaloTau(process,
                    pfTauLabel=cms.InputTag('pfRecoTauProducer'),
                    caloTauLabel=cms.InputTag('caloRecoTauProducer')):
    switchMCAndTriggerMatch(process,pfTauLabel,caloTauLabel)
    process.allLayer1Taus.tauSource    = caloTauLabel
    process.allLayer1Taus.isolation    = cms.PSet() # there is no path for calo tau isolation available at the moment
    process.allLayer1Taus.isoDeposits  = cms.PSet() # there is no path for calo tau isolation available at the moment
    process.allLayer1Taus.tauIDSources = cms.PSet(  # all these are already present in 2.2.X AODSIM
            leadingTrackFinding = cms.InputTag("caloRecoTauDiscriminationByLeadingTrackFinding"),
            leadingTrackPtCut   = cms.InputTag("caloRecoTauDiscriminationByLeadingTrackPtCut"),
            byIsolation         = cms.InputTag("caloRecoTauDiscriminationByIsolation"),
            againstElectron     = cms.InputTag("caloRecoTauDiscriminationAgainstElectron"),  
    )
    if pfTauLabel in process.aodSummary.candidates:
        process.aodSummary.candidates[process.aodSummary.candidates.index(pfTauLabel)] = caloTauLabel
    else:
        process.aodSummary.candidates += [caloTauLabel]
