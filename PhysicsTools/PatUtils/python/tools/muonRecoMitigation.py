import FWCore.ParameterSet.Config as cms

from PhysicsTools.PatAlgos.tools.ConfigToolBase import *

from PhysicsTools.PatAlgos.tools.helpers import getPatAlgosToolsTask, addToProcessAndTask

def muonRecoMitigation(process,
                       pfCandCollection,
                       runOnMiniAOD,
                       selection="",
                       muonCollection="",
                       cleanCollName="cleanMuonsPFCandidates",
                       cleaningScheme="all",
                       postfix=""):

    task = getPatAlgosToolsTask(process)

    sequence=cms.Sequence()    

    if selection=="":
        typeFix=""

        if runOnMiniAOD:
            from RecoMET.METFilters.badGlobalMuonTaggersMiniAOD_cff import badGlobalMuonTaggerMAOD, cloneGlobalMuonTaggerMAOD
            typeFix="MAOD"
            badMuModule=badGlobalMuonTaggerMAOD
            cloneMuModule=cloneGlobalMuonTaggerMAOD
        else:
            from RecoMET.METFilters.badGlobalMuonTaggersAOD_cff import badGlobalMuonTagger, cloneGlobalMuonTagger
            badMuModule=badGlobalMuonTagger
            cloneMuModule=cloneGlobalMuonTagger

        vtags=cms.VInputTag()
        if cleaningScheme in ["bad","all","computeAllApplyBad","computeAllApplyClone"]:
            addToProcessAndTask('badGlobalMuonTagger'+typeFix+postfix, badMuModule.clone(), process, task )
            sequence +=getattr(process,"badGlobalMuonTagger"+typeFix+postfix)
            if cleaningScheme in ["bad","computeAllApplyBad"]:
                badMuonCollection = 'badGlobalMuonTagger'+typeFix+postfix+':bad'
        if cleaningScheme in ["clone","duplicated","all","computeAllApplyBad","computeAllApplyClone"]:
            addToProcessAndTask('cloneGlobalMuonTagger'+typeFix+postfix, cloneMuModule.clone(), process, task )
            sequence +=getattr(process,"cloneGlobalMuonTagger"+typeFix+postfix)
            if cleaningScheme in ["clone","duplicated","computeAllApplyClone"]:
                badMuonCollection = 'cloneGlobalMuonTagger'+typeFix+postfix+':bad'
        
        if cleaningScheme=="all":
            badMuonCollection="badMuons"+postfix
            badMuonProducer = cms.EDProducer(
                "CandViewMerger",
                src = cms.VInputTag(
                    cms.InputTag('badGlobalMuonTagger'+typeFix+postfix,'bad'),
                    cms.InputTag('cloneGlobalMuonTagger'+typeFix+postfix,'bad'),
                    )
                )
            addToProcessAndTask(badMuonCollection, badMuonProducer, process, task)
            sequence +=getattr(process, badMuonCollection )
    else:
        badMuonCollection="badMuons"+postfix
        badMuonModule = cms.EDFilter("CandViewSelector", 
                                     src = cms.InputTag(muonCollection), 
                                     cut = cms.string(selection)
                                     )
    
    # now cleaning ================================
    cleanedPFCandCollection=cleanCollName+postfix
    if runOnMiniAOD:
        cleanedPFCandProducer = cms.EDProducer("CandPtrProjector", 
                                               src = cms.InputTag(pfCandCollection),
                                               veto = cms.InputTag(badMuonCollection)
                                               )
    else:
        cleanedPFCandProducer = cms.EDProducer("PFCandPtrProjector", 
                                               src = cms.InputTag(pfCandCollection),
                                               veto = cms.InputTag(badMuonCollection)
                                         )

        addToProcessAndTask(cleanedPFCandCollection, cleanedPFCandProducer, process, task) 
    sequence +=getattr(process, cleanedPFCandCollection )

    return sequence
