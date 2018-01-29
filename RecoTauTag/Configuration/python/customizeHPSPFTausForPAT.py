import FWCore.ParameterSet.Config as cms

'''
Tools to update tau configuration for special runs like re-MiniAOD production
'''

# Update MVAIso tau-Id for MiniAODv2 production with CMSSW_9_4_X (X>3) 
def cloneAndModifyMVAIsolationFor94XMiniAODv2(process):
    process.load("RecoTauTag.Configuration.RecoPFTauTag_cff")
    ##Clone and modify modules of interest
    process.hpsPFTauDiscriminationByIsolationMVArun2v1DBoldDMwLTrawMVAIsoFor94XMiniv2 = process.hpsPFTauDiscriminationByIsolationMVArun2v1DBoldDMwLTraw.clone(
        mvaName = cms.string("RecoTauTag_tauIdMVAIsoDBoldDMwLT2017v1"),
        mvaOpt = cms.string("DBoldDMwLTwGJ")
        )
    process.hpsPFTauDiscriminationByVLooseIsolationMVArun2v1DBoldDMwLTMVAIsoFor94XMiniv2 = process.hpsPFTauDiscriminationByVLooseIsolationMVArun2v1DBoldDMwLT.clone(
        mvaOutput_normalization = cms.string("RecoTauTag_tauIdMVAIsoDBoldDMwLT2017v1_mvaOutput_normalization"),
        key = cms.InputTag("hpsPFTauDiscriminationByIsolationMVArun2v1DBoldDMwLTrawMVAIsoFor94XMiniv2","category"),
        toMultiplex = cms.InputTag("hpsPFTauDiscriminationByIsolationMVArun2v1DBoldDMwLTrawMVAIsoFor94XMiniv2"),
        )
    process.hpsPFTauDiscriminationByVLooseIsolationMVArun2v1DBoldDMwLTMVAIsoFor94XMiniv2.mapping[0].cut = cms.string("RecoTauTag_tauIdMVAIsoDBoldDMwLT2017v1_WPEff90")
    process.hpsPFTauDiscriminationByVVLooseIsolationMVArun2v1DBoldDMwLTMVAIsoFor94XMiniv2 = process.hpsPFTauDiscriminationByVLooseIsolationMVArun2v1DBoldDMwLTMVAIsoFor94XMiniv2.clone()
    process.hpsPFTauDiscriminationByVVLooseIsolationMVArun2v1DBoldDMwLTMVAIsoFor94XMiniv2.mapping[0].cut = cms.string("RecoTauTag_tauIdMVAIsoDBoldDMwLT2017v1_WPEff95")
    process.hpsPFTauDiscriminationByLooseIsolationMVArun2v1DBoldDMwLTMVAIsoFor94XMiniv2 = process.hpsPFTauDiscriminationByVLooseIsolationMVArun2v1DBoldDMwLTMVAIsoFor94XMiniv2.clone()
    process.hpsPFTauDiscriminationByLooseIsolationMVArun2v1DBoldDMwLTMVAIsoFor94XMiniv2.mapping[0].cut = cms.string("RecoTauTag_tauIdMVAIsoDBoldDMwLT2017v1_WPEff80")
    process.hpsPFTauDiscriminationByMediumIsolationMVArun2v1DBoldDMwLTMVAIsoFor94XMiniv2 = process.hpsPFTauDiscriminationByVLooseIsolationMVArun2v1DBoldDMwLTMVAIsoFor94XMiniv2.clone()
    process.hpsPFTauDiscriminationByMediumIsolationMVArun2v1DBoldDMwLTMVAIsoFor94XMiniv2.mapping[0].cut = cms.string("RecoTauTag_tauIdMVAIsoDBoldDMwLT2017v1_WPEff70")
    process.hpsPFTauDiscriminationByTightIsolationMVArun2v1DBoldDMwLTMVAIsoFor94XMiniv2 = process.hpsPFTauDiscriminationByVLooseIsolationMVArun2v1DBoldDMwLTMVAIsoFor94XMiniv2.clone()
    process.hpsPFTauDiscriminationByTightIsolationMVArun2v1DBoldDMwLTMVAIsoFor94XMiniv2.mapping[0].cut = cms.string("RecoTauTag_tauIdMVAIsoDBoldDMwLT2017v1_WPEff60")
    process.hpsPFTauDiscriminationByVTightIsolationMVArun2v1DBoldDMwLTMVAIsoFor94XMiniv2 = process.hpsPFTauDiscriminationByVLooseIsolationMVArun2v1DBoldDMwLTMVAIsoFor94XMiniv2.clone()
    process.hpsPFTauDiscriminationByVTightIsolationMVArun2v1DBoldDMwLTMVAIsoFor94XMiniv2.mapping[0].cut = cms.string("RecoTauTag_tauIdMVAIsoDBoldDMwLT2017v1_WPEff50")
    process.hpsPFTauDiscriminationByVVTightIsolationMVArun2v1DBoldDMwLTMVAIsoFor94XMiniv2 = process.hpsPFTauDiscriminationByVLooseIsolationMVArun2v1DBoldDMwLTMVAIsoFor94XMiniv2.clone()
    process.hpsPFTauDiscriminationByVVTightIsolationMVArun2v1DBoldDMwLTMVAIsoFor94XMiniv2.mapping[0].cut = cms.string("RecoTauTag_tauIdMVAIsoDBoldDMwLT2017v1_WPEff40")   
    #Create a new task and put the moduled therin
    process.hpsPFTauMVAIsoFor94XMiniv2Task = cms.Task(
        process.hpsPFTauDiscriminationByIsolationMVArun2v1DBoldDMwLTrawMVAIsoFor94XMiniv2,    
        process.hpsPFTauDiscriminationByVVLooseIsolationMVArun2v1DBoldDMwLTMVAIsoFor94XMiniv2,
        process.hpsPFTauDiscriminationByVLooseIsolationMVArun2v1DBoldDMwLTMVAIsoFor94XMiniv2, 
        process.hpsPFTauDiscriminationByLooseIsolationMVArun2v1DBoldDMwLTMVAIsoFor94XMiniv2,  
        process.hpsPFTauDiscriminationByMediumIsolationMVArun2v1DBoldDMwLTMVAIsoFor94XMiniv2, 
        process.hpsPFTauDiscriminationByTightIsolationMVArun2v1DBoldDMwLTMVAIsoFor94XMiniv2,  
        process.hpsPFTauDiscriminationByVTightIsolationMVArun2v1DBoldDMwLTMVAIsoFor94XMiniv2, 
        process.hpsPFTauDiscriminationByVVTightIsolationMVArun2v1DBoldDMwLTMVAIsoFor94XMiniv2,
        )
    #Add all this to makePatTausTask
    process.makePatTausTask.add(process.hpsPFTauMVAIsoFor94XMiniv2Task)
    #Now modify patTau by replacing old by new discriminats
    process.patTaus.tauIDSources.byIsolationMVArun2v1DBoldDMwLTraw = cms.InputTag("hpsPFTauDiscriminationByIsolationMVArun2v1DBoldDMwLTrawMVAIsoFor94XMiniv2")
    process.patTaus.tauIDSources.byVVLooseIsolationMVArun2v1DBoldDMwLT = cms.InputTag("hpsPFTauDiscriminationByVVLooseIsolationMVArun2v1DBoldDMwLTMVAIsoFor94XMiniv2")
    process.patTaus.tauIDSources.byVLooseIsolationMVArun2v1DBoldDMwLT = cms.InputTag("hpsPFTauDiscriminationByVLooseIsolationMVArun2v1DBoldDMwLTMVAIsoFor94XMiniv2")
    process.patTaus.tauIDSources.byLooseIsolationMVArun2v1DBoldDMwLT = cms.InputTag("hpsPFTauDiscriminationByLooseIsolationMVArun2v1DBoldDMwLTMVAIsoFor94XMiniv2")
    process.patTaus.tauIDSources.byMediumIsolationMVArun2v1DBoldDMwLT = cms.InputTag("hpsPFTauDiscriminationByMediumIsolationMVArun2v1DBoldDMwLTMVAIsoFor94XMiniv2")
    process.patTaus.tauIDSources.byTightIsolationMVArun2v1DBoldDMwLT = cms.InputTag("hpsPFTauDiscriminationByTightIsolationMVArun2v1DBoldDMwLTMVAIsoFor94XMiniv2")
    process.patTaus.tauIDSources.byVTightIsolationMVArun2v1DBoldDMwLT = cms.InputTag("hpsPFTauDiscriminationByVTightIsolationMVArun2v1DBoldDMwLTMVAIsoFor94XMiniv2")
    process.patTaus.tauIDSources.byVVTightIsolationMVArun2v1DBoldDMwLT = cms.InputTag("hpsPFTauDiscriminationByVVTightIsolationMVArun2v1DBoldDMwLTMVAIsoFor94XMiniv2")
