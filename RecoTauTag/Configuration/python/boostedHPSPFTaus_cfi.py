import FWCore.ParameterSet.Config as cms
import PhysicsTools.PatAlgos.tools.helpers as configtools

def addBoostedTaus(process):
    from PhysicsTools.PatAlgos.tools.helpers import cloneProcessingSnippet
    from PhysicsTools.PatAlgos.tools.helpers import massSearchReplaceAnyInputTag

    process.load("RecoTauTag.Configuration.boostedHPSPFTaus_cff")
    patAlgosToolsTask = configtools.getPatAlgosToolsTask(process)
    patAlgosToolsTask.add(process.boostedHPSPFTausTask)

    process.load("RecoTauTag.Configuration.RecoPFTauTag_cff")
    process.ptau = cms.Path( process.PFTau )
    process.PATTauSequence = cms.Sequence(process.PFTau+process.makePatTaus+process.selectedPatTaus)
    process.PATTauSequenceBoosted = cloneProcessingSnippet(process,process.PATTauSequence, "Boosted", addToTask = True)
    process.recoTauAK4PFJets08RegionBoosted.src = cms.InputTag('boostedTauSeeds')
    process.recoTauAK4PFJets08RegionBoosted.pfCandSrc = cms.InputTag('particleFlow')
    process.recoTauAK4PFJets08RegionBoosted.pfCandAssocMapSrc = cms.InputTag('boostedTauSeeds', 'pfCandAssocMapForIsolation')
    process.ak4PFJetsLegacyHPSPiZerosBoosted.jetSrc = cms.InputTag('boostedTauSeeds')
    process.ak4PFJetsRecoTauChargedHadronsBoosted.jetSrc = cms.InputTag('boostedTauSeeds')
    process.ak4PFJetsRecoTauChargedHadronsBoosted.builders[1].dRcone = cms.double(0.3)
    process.ak4PFJetsRecoTauChargedHadronsBoosted.builders[1].dRconeLimitedToJetArea = cms.bool(True)
    process.combinatoricRecoTausBoosted.jetSrc = cms.InputTag('boostedTauSeeds')
    process.combinatoricRecoTausBoosted.builders[0].pfCandSrc = cms.InputTag('particleFlow')
    ## Note JetArea is not defined for subjets (-> do not switch to True in hpsPFTauDiscriminationByLooseMuonRejection3Boosted, False is default)
    ## The restiction to jetArea is turned to dRMatch=0.1 (-> use explicitly this modified value)
    process.hpsPFTauDiscriminationByLooseMuonRejection3Boosted.dRmuonMatch = 0.1
    process.hpsPFTauDiscriminationByTightMuonRejection3Boosted.dRmuonMatch = 0.1
    massSearchReplaceAnyInputTag(process.PATTauSequenceBoosted,cms.InputTag("ak4PFJets"),cms.InputTag("boostedTauSeeds"))  
    process.slimmedTausBoosted = process.slimmedTaus.clone(src = cms.InputTag("selectedPatTausBoosted"))
    patAlgosToolsTask.add(process.slimmedTausBoosted)

    return process
