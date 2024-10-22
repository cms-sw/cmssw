import FWCore.ParameterSet.Config as cms
import PhysicsTools.PatAlgos.tools.helpers as configtools

def addBoostedTaus(process):
    from PhysicsTools.PatAlgos.tools.helpers import cloneProcessingSnippet
    from PhysicsTools.PatAlgos.tools.helpers import massSearchReplaceAnyInputTag

    process.load("RecoTauTag.Configuration.boostedHPSPFTaus_cff")
    patAlgosToolsTask = configtools.getPatAlgosToolsTask(process)
    patAlgosToolsTask.add(process.boostedHPSPFTausTask)

    process.load("RecoTauTag.Configuration.RecoPFTauTag_cff")
    # BDT-based tauIDs removed from standard tau sequence, but still used by boosed taus
    process.PFTauMVAIdSequence = cms.Sequence(
        process.hpsPFTauDiscriminationByMVA6rawElectronRejection+
        process.hpsPFTauDiscriminationByMVA6ElectronRejection+
        process.hpsPFTauDiscriminationByIsolationMVArun2v1DBoldDMwLTraw+
        process.hpsPFTauDiscriminationByIsolationMVArun2v1DBoldDMwLT+
        process.hpsPFTauDiscriminationByIsolationMVArun2v1DBnewDMwLTraw+
        process.hpsPFTauDiscriminationByIsolationMVArun2v1DBnewDMwLT
    )
    process.PATTauSequence = cms.Sequence(
        process.PFTau+process.PFTauMVAIdSequence+
        process.makePatTaus+process.selectedPatTaus)
    process.PATTauSequenceBoosted = cloneProcessingSnippet(process,process.PATTauSequence, "Boosted", addToTask = True)
    process.recoTauAK4PFJets08RegionBoosted.src = 'boostedTauSeeds'
    process.recoTauAK4PFJets08RegionBoosted.pfCandSrc = 'particleFlow'
    process.recoTauAK4PFJets08RegionBoosted.pfCandAssocMapSrc = ('boostedTauSeeds', 'pfCandAssocMapForIsolation')
    process.ak4PFJetsLegacyHPSPiZerosBoosted.jetSrc = 'boostedTauSeeds'
    process.ak4PFJetsRecoTauChargedHadronsBoosted.jetSrc = 'boostedTauSeeds'
    process.ak4PFJetsRecoTauChargedHadronsBoosted.builders[1].dRcone = 0.3
    process.ak4PFJetsRecoTauChargedHadronsBoosted.builders[1].dRconeLimitedToJetArea = True
    process.combinatoricRecoTausBoosted.jetSrc = 'boostedTauSeeds'
    process.combinatoricRecoTausBoosted.builders[0].pfCandSrc = cms.InputTag('particleFlow')
    ## Note JetArea is not defined for subjets (-> do not switch to True in hpsPFTauDiscriminationByLooseMuonRejection3Boosted, False is default)
    ## The restiction to jetArea is turned to dRMatch=0.1 (-> use explicitly this modified value)
    process.hpsPFTauDiscriminationByMuonRejection3Boosted.dRmuonMatch = 0.1
    massSearchReplaceAnyInputTag(process.PATTauSequenceBoosted,cms.InputTag("ak4PFJets"),cms.InputTag("boostedTauSeeds"))  
    #Add BDT-based tauIDs still used by boosed taus
    from PhysicsTools.PatAlgos.producersLayer1.tauProducer_cfi import containerID
    containerID(process.patTausBoosted.tauIDSources, "hpsPFTauDiscriminationByIsolationMVArun2v1DBoldDMwLTBoosted", "rawValues", [
        ["byIsolationMVArun2DBoldDMwLTraw", "discriminator"]
    ])
    containerID(process.patTausBoosted.tauIDSources, "hpsPFTauDiscriminationByIsolationMVArun2v1DBoldDMwLTBoosted", "workingPoints", [
        ["byVVLooseIsolationMVArun2DBoldDMwLT", "_VVLoose"],
        ["byVLooseIsolationMVArun2DBoldDMwLT", "_VLoose"],
        ["byLooseIsolationMVArun2DBoldDMwLT", "_Loose"],
        ["byMediumIsolationMVArun2DBoldDMwLT", "_Medium"],
        ["byTightIsolationMVArun2DBoldDMwLT", "_Tight"],
        ["byVTightIsolationMVArun2DBoldDMwLT", "_VTight"],
        ["byVVTightIsolationMVArun2DBoldDMwLT", "_VVTight"]
    ])
    containerID(process.patTausBoosted.tauIDSources, "hpsPFTauDiscriminationByIsolationMVArun2v1DBnewDMwLTBoosted", "rawValues", [
        ["byIsolationMVArun2DBnewDMwLTraw", "discriminator"]
    ])
    containerID(process.patTausBoosted.tauIDSources, "hpsPFTauDiscriminationByIsolationMVArun2v1DBnewDMwLTBoosted", "workingPoints", [
        ["byVVLooseIsolationMVArun2DBnewDMwLT", "_VVLoose"],
        ["byVLooseIsolationMVArun2DBnewDMwLT", "_VLoose"],
        ["byLooseIsolationMVArun2DBnewDMwLT", "_Loose"],
        ["byMediumIsolationMVArun2DBnewDMwLT", "_Medium"],
        ["byTightIsolationMVArun2DBnewDMwLT", "_Tight"],
        ["byVTightIsolationMVArun2DBnewDMwLT", "_VTight"],
        ["byVVTightIsolationMVArun2DBnewDMwLT", "_VVTight"]
    ])
    containerID(process.patTausBoosted.tauIDSources, "hpsPFTauDiscriminationByMVA6ElectronRejectionBoosted", "rawValues", [
        ["againstElectronMVA6Raw", "discriminator"],
        ["againstElectronMVA6category", "category"]
    ])
    containerID(process.patTausBoosted.tauIDSources, "hpsPFTauDiscriminationByMVA6ElectronRejectionBoosted", "workingPoints", [
        ["againstElectronVLooseMVA6", "_VLoose"],
        ["againstElectronLooseMVA6", "_Loose"],
        ["againstElectronMediumMVA6", "_Medium"],
        ["againstElectronTightMVA6", "_Tight"],
        ["againstElectronVTightMVA6", "_VTight"]
    ])
    process.slimmedTausBoosted = process.slimmedTaus.clone(src = "selectedPatTausBoosted")
    patAlgosToolsTask.add(process.slimmedTausBoosted)

    return process
