import FWCore.ParameterSet.Config as cms

from FWCore.GuiBrowsers.ConfigToolBase import *

def muonRecoMitigation(process,
                       pfCandCollection,
                       runOnMiniAOD,
                       selection="",
                       muonCollection="",
                       cleanCollName="cleanMuonsPFCandidates",
                       cleaningScheme="all",
                       postfix=""):

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
            setattr(process, 'badGlobalMuonTagger'+typeFix+postfix, badMuModule.clone() )
            sequence +=getattr(process,"badGlobalMuonTagger"+typeFix+postfix)
            if cleaningScheme in ["bad","computeAllApplyBad"]:
                badMuonCollection = 'badGlobalMuonTagger'+typeFix+postfix+':bad'
        if cleaningScheme in ["clone","duplicated","all","computeAllApplyBad","computeAllApplyClone"]:
            setattr(process, 'cloneGlobalMuonTagger'+typeFix+postfix, cloneMuModule.clone() )
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
            setattr(process,badMuonCollection,badMuonProducer)
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
        #pfCandTmpPtrs = cms.EDProducer("PFCandidateFwdPtrProducer",
        #                                      src = cms.InputTag(pfCandCollection)
        #                                      )
        #muonTmpPtrs = cms.EDProducer("PFCandidateFwdPtrProducer",
        #                                      src = cms.InputTag(badMuonCollection)
        #                                      )
        #setattr(process,"candPtrTmp"+postfix,pfCandTmpPtrs) 
        #setattr(process,"muonPtrTmp"+postfix,muonTmpPtrs) 
        #cleanedPFCandProducer = cms.EDProducer(
        #    "TPPFCandidatesOnPFCandidates",
        #    enable =  cms.bool( True ),
        #    verbose = cms.untracked.bool( False ),
        #    name = cms.untracked.string(""),
        #    topCollection = cms.InputTag(pfCandCollection),
        #    bottomCollection = cms.InputTag(badMuonCollection),
        #    )

    setattr(process,cleanedPFCandCollection,cleanedPFCandProducer) 
    sequence +=getattr(process, cleanedPFCandCollection )

    return sequence
