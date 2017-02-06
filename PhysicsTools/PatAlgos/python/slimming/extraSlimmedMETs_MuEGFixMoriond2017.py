import FWCore.ParameterSet.Config as cms
from PhysicsTools.PatAlgos.tools.helpers import massSearchReplaceAnyInputTag,cloneProcessingSnippet,addKeepStatement

def addExtraMETCollections(process, unCleanPFCandidateCollection,
                           cleanElectronCollection,
                           cleanPhotonCollection,
                           unCleanElectronCollection,
                           unCleanPhotonCollection ):

    # Muon/EGamma un/corrected pfMET ============
    from PhysicsTools.PatUtils.tools.corMETFromMuonAndEG import corMETFromMuonAndEG
    from PhysicsTools.PatUtils.tools.runMETCorrectionsAndUncertainties import runMetCorAndUncForMiniAODProduction
    
    # uncorrected MET
    cloneProcessingSnippet(process, getattr(process,"makePatJets"),"BackupAllEvents")
    massSearchReplaceAnyInputTag(getattr(process,"makePatJetsBackupAllEvents"), "ak4PFJetsCHS", "ak4PFJetsCHSBackupAllEvents")
    massSearchReplaceAnyInputTag(getattr(process,"makePatJetsBackupAllEvents"), "pfCandidatesBadMuonsCleaned", "particleFlow")
    del process.patJetsBackupAllEvents.userData
    process.patJetsBackupAllEvents.addAssociatedTracks = cms.bool(False)
    process.patJetsBackupAllEvents.addBTagInfo = cms.bool(False)
    process.patJetsBackupAllEvents.addDiscriminators = cms.bool(False)
    process.patJetsBackupAllEvents.addGenJetMatch = cms.bool(False)
    process.patJetsBackupAllEvents.addGenPartonMatch = cms.bool(False)
    process.patJetsBackupAllEvents.addJetCharge = cms.bool(False)
    process.patJetsBackupAllEvents.addJetCorrFactors = cms.bool(True)
    process.patJetsBackupAllEvents.addJetFlavourInfo = cms.bool(False)
    process.patJetsBackupAllEvents.addJetID = cms.bool(False)
    process.patJetsBackupAllEvents.addPartonJetMatch = cms.bool(False)
    process.patJetsBackupAllEvents.addResolutions = cms.bool(False)
    process.patJetsBackupAllEvents.addTagInfos = cms.bool(False)
    process.patJetsBackupAllEvents.discriminatorSources = cms.VInputTag()
    process.patJetsBackupAllEvents.embedGenJetMatch = cms.bool(False)
  
    runMetCorAndUncForMiniAODProduction(process,
                                        metType="PF",
                                        pfCandColl=cms.InputTag(unCleanPFCandidateCollection),
                                        recoMetFromPFCs=True,
                                        jetCollUnskimmed="patJetsBackupAllEvents",
                                        postfix="Uncorrected"
                                        )

    if not hasattr(process, "slimmedMETs"):
        process.load('PhysicsTools.PatAlgos.slimming.slimmedMETs_cfi')
        
    process.slimmedMETsUncorrected = process.slimmedMETs.clone()
    process.slimmedMETsUncorrected.src = cms.InputTag("patPFMetT1Uncorrected")
    process.slimmedMETsUncorrected.rawVariation =  cms.InputTag("patPFMetUncorrected")
    process.slimmedMETsUncorrected.t1Uncertainties = cms.InputTag("patPFMetT1%sUncorrected") 
    process.slimmedMETsUncorrected.t01Variation = cms.InputTag("patPFMetT0pcT1Uncorrected")
    process.slimmedMETsUncorrected.t1SmearedVarsAndUncs = cms.InputTag("patPFMetT1Smear%sUncorrected")
    process.slimmedMETsUncorrected.tXYUncForRaw = cms.InputTag("patPFMetTxyUncorrected")
    process.slimmedMETsUncorrected.tXYUncForT1 = cms.InputTag("patPFMetT1TxyUncorrected")
    process.slimmedMETsUncorrected.tXYUncForT01 = cms.InputTag("patPFMetT0pcT1TxyUncorrected")
    process.slimmedMETsUncorrected.tXYUncForT1Smear = cms.InputTag("patPFMetT1SmearTxyUncorrected")
    process.slimmedMETsUncorrected.tXYUncForT01Smear = cms.InputTag("patPFMetT0pcT1SmearTxyUncorrected")
    del process.slimmedMETsUncorrected.caloMET
    
    # EG corrected MET
    corMETFromMuonAndEG(process,
                        pfCandCollection="", #not needed
                        electronCollection=unCleanElectronCollection,
                        photonCollection=unCleanPhotonCollection,
                        corElectronCollection=cleanElectronCollection,
                        corPhotonCollection=cleanPhotonCollection,
                        allMETEGCorrected=True,
                        muCorrection=False,
                        eGCorrection=True,
                        runOnMiniAOD=False,
                        eGPFix="Uncorrected",
                        postfix="EGOnly"
                        )
    process.slimmedMETsEGClean = process.slimmedMETs.clone()
    process.slimmedMETsEGClean.src = cms.InputTag("patPFMetT1UncorrectedEGOnly")
    process.slimmedMETsEGClean.rawVariation =  cms.InputTag("patPFMetRawUncorrectedEGOnly")
    process.slimmedMETsEGClean.t1Uncertainties = cms.InputTag("patPFMetT1%sUncorrectedEGOnly") 
    process.slimmedMETsEGClean.t01Variation = cms.InputTag("patPFMetT0pcT1UncorrectedEGOnly")
    process.slimmedMETsEGClean.t1SmearedVarsAndUncs = cms.InputTag("patPFMetT1Smear%sUncorrectedEGOnly")
    process.slimmedMETsEGClean.tXYUncForRaw = cms.InputTag("patPFMetTxyUncorrectedEGOnly")
    process.slimmedMETsEGClean.tXYUncForT1 = cms.InputTag("patPFMetT1TxyUncorrectedEGOnly")
    process.slimmedMETsEGClean.tXYUncForT01 = cms.InputTag("patPFMetT0pcT1TxyUncorrectedEGOnly")
    process.slimmedMETsEGClean.tXYUncForT1Smear = cms.InputTag("patPFMetT1SmearTxyUncorrectedEGOnly")
    process.slimmedMETsEGClean.tXYUncForT01Smear = cms.InputTag("patPFMetT0pcT1SmearTxyUncorrectedEGOnly")
    del process.slimmedMETsEGClean.caloMET
 
    # fully corrected MET
    corMETFromMuonAndEG(process,
                        pfCandCollection="", #not needed
                        electronCollection=unCleanElectronCollection,
                        photonCollection=unCleanPhotonCollection,
                        corElectronCollection=cleanElectronCollection,
                        corPhotonCollection=cleanPhotonCollection,
                        allMETEGCorrected=True,
                        muCorrection=False,
                        eGCorrection=True,
                        runOnMiniAOD=False,
                        postfix="MuEGClean"
                        )
    process.slimmedMETsMuEGClean = process.slimmedMETs.clone()
    process.slimmedMETsMuEGClean.src = cms.InputTag("patPFMetT1MuEGClean")
    process.slimmedMETsMuEGClean.rawVariation =  cms.InputTag("patPFMetRawMuEGClean")
    process.slimmedMETsMuEGClean.t1Uncertainties = cms.InputTag("patPFMetT1%sMuEGClean") 
    process.slimmedMETsMuEGClean.t01Variation = cms.InputTag("patPFMetT0pcT1MuEGClean")
    process.slimmedMETsMuEGClean.t1SmearedVarsAndUncs = cms.InputTag("patPFMetT1Smear%sMuEGClean")
    process.slimmedMETsMuEGClean.tXYUncForRaw = cms.InputTag("patPFMetTxyMuEGClean")
    process.slimmedMETsMuEGClean.tXYUncForT1 = cms.InputTag("patPFMetT1TxyMuEGClean")
    process.slimmedMETsMuEGClean.tXYUncForT01 = cms.InputTag("patPFMetT0pcT1TxyMuEGClean")
    process.slimmedMETsMuEGClean.tXYUncForT1Smear = cms.InputTag("patPFMetT1SmearTxyMuEGClean")
    process.slimmedMETsMuEGClean.tXYUncForT01Smear = cms.InputTag("patPFMetT0pcT1SmearTxyMuEGClean")
    del process.slimmedMETsMuEGClean.caloMET

    addKeepStatement(process, "keep *_slimmedMETs_*_*",
                    ["keep *_slimmedMETsUncorrected_*_*", "keep *_slimmedMETsEGClean_*_*", "keep *_slimmedMETsMuEGClean_*_*"])

    

def addExtraPuppiMETCorrections(process,
                                cleanPFCandidateCollection,
                                unCleanPFCandidateCollection,
                                cleanElectronCollection,
                                cleanPhotonCollection,
                                unCleanElectronCollection,
                                unCleanPhotonCollection
                                ):

    from PhysicsTools.PatUtils.tools.corMETFromMuonAndEG import corMETFromMuonAndEG

    #EG correction for puppi, muon correction done right above
    corMETFromMuonAndEG(process,
                        pfCandCollection="puppiForMET",
                        electronCollection=unCleanElectronCollection,
                        photonCollection=unCleanPhotonCollection,
                        corElectronCollection=cleanElectronCollection,
                        corPhotonCollection=cleanPhotonCollection,
                        allMETEGCorrected=True,
                        muCorrection=False,
                        eGCorrection=True,
                        runOnMiniAOD=False,
                        eGPfCandMatching=True,
                        eGPFix="Puppi",
                        postfix="PuppiClean"
                        )

    if not hasattr(process, "slimmedMETs"):
        process.load('PhysicsTools.PatAlgos.slimming.slimmedMETs_cfi')

    process.slimmedMETsPuppi.src = cms.InputTag("patPFMetT1PuppiPuppiClean")
    process.slimmedMETsPuppi.rawVariation =  cms.InputTag("patPFMetRawPuppiPuppiClean")
    process.slimmedMETsPuppi.t1Uncertainties = cms.InputTag("patPFMetT1%sPuppiPuppiClean")
    process.slimmedMETsPuppi.t01Variation = cms.InputTag("patPFMetT0pcT1PuppiPuppiClean")
    process.slimmedMETsPuppi.t1SmearedVarsAndUncs = cms.InputTag("patPFMetT1Smear%sPuppiPuppiClean")
    process.slimmedMETsPuppi.tXYUncForRaw = cms.InputTag("patPFMetTxyPuppiPuppiClean")
    process.slimmedMETsPuppi.tXYUncForT1 = cms.InputTag("patPFMetT1TxyPuppiPuppiClean")
    process.slimmedMETsPuppi.tXYUncForT01 = cms.InputTag("patPFMetT0pcT1TxyPuppiPuppiClean")
    process.slimmedMETsPuppi.tXYUncForT1Smear = cms.InputTag("patPFMetT1SmearTxyPuppiPuppiClean")
    process.slimmedMETsPuppi.tXYUncForT01Smear = cms.InputTag("patPFMetT0pcT1SmearTxyPuppiPuppiClean")
    #del process.slimmedMETsPuppi.caloMET    

    #EGamma correction
    process.puppiMETEGCor = cms.EDProducer("CorrMETDataExtractor",
                    corrections = cms.VInputTag(
                        cms.InputTag("corMETPhotonPuppiClean"),
                        cms.InputTag("corMETElectronPuppiClean") )
                                           )

    #Muon correction, restarting from PF candidates to take the weights
    process.puppiMuonCorrection = cms.EDProducer("ShiftedParticleMETcorrInputProducer",
                        srcOriginal = cms.InputTag(unCleanPFCandidateCollection),
                        srcShifted = cms.InputTag(cleanPFCandidateCollection),
                                  )
    
    process.puppiMETMuCor = cms.EDProducer("CorrMETDataExtractor",
                    corrections = cms.VInputTag(
                        cms.InputTag("puppiMuonCorrection") )
                                           )
    addKeepStatement(process, "keep *_slimmedMETsPuppi_*_*",
                    ["keep *_puppiMETEGCor_*_*", "keep *_puppiMETMuCor_*_*"])
