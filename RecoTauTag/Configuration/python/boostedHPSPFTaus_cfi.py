import FWCore.ParameterSet.Config as cms

def addBoostedTaus(process):
    from PhysicsTools.PatAlgos.tools.helpers import cloneProcessingSnippet
    from PhysicsTools.PatAlgos.tools.helpers import massSearchReplaceAnyInputTag
    process.load("RecoTauTag.Configuration.boostedHPSPFTaus_cff")
    process.load("RecoTauTag.Configuration.RecoPFTauTag_cff")
    process.ptau = cms.Path( process.PFTau )
    process.PATTauSequence = cms.Sequence(process.PFTau+process.makePatTaus+process.selectedPatTaus)
    process.PATTauSequenceBoosted = cloneProcessingSnippet(process,process.PATTauSequence, "Boosted")
    process.recoTauAK4PFJets08RegionBoosted.src = cms.InputTag('boostedTauSeeds')
    process.recoTauAK4PFJets08RegionBoosted.pfCandSrc = cms.InputTag('particleFlow')
    process.recoTauAK4PFJets08RegionBoosted.pfCandAssocMapSrc = cms.InputTag('boostedTauSeeds', 'pfCandAssocMapForIsolation')
    process.ak4PFJetsLegacyHPSPiZerosBoosted.jetSrc = cms.InputTag('boostedTauSeeds')
    process.ak4PFJetsRecoTauChargedHadronsBoosted.jetSrc = cms.InputTag('boostedTauSeeds')
    process.ak4PFJetsRecoTauChargedHadronsBoosted.builders[1].dRcone = cms.double(0.3)
    process.ak4PFJetsRecoTauChargedHadronsBoosted.builders[1].dRconeLimitedToJetArea = cms.bool(True)
    process.combinatoricRecoTausBoosted.jetSrc = cms.InputTag('boostedTauSeeds')
    process.combinatoricRecoTausBoosted.modifiers.remove(process.combinatoricRecoTausBoosted.modifiers[3])
    #process.combinatoricRecoTausBoosted.builders[0].pfCandSrc = cms.InputTag('pfNoPileUpForBoostedTaus')
    process.combinatoricRecoTausBoosted.builders[0].pfCandSrc = cms.InputTag('particleFlow')
    process.hpsPFTauDiscriminationByLooseMuonRejection3Boosted.dRmuonMatchLimitedToJetArea = cms.bool(True)
    process.hpsPFTauDiscriminationByTightMuonRejection3Boosted.dRmuonMatchLimitedToJetArea = cms.bool(True)
    massSearchReplaceAnyInputTag(process.PATTauSequenceBoosted,cms.InputTag("ak4PFJets"),cms.InputTag("boostedTauSeeds"))  
    process.slimmedTausBoosted = process.slimmedTaus.clone(src = cms.InputTag("selectedPatTausBoosted"))
    
    return process
