import FWCore.ParameterSet.Config as cms
from PhysicsTools.PatAlgos.tools.helpers import listDependencyChain, massSearchReplaceAnyInputTag, cloneProcessingSnippet, addKeepStatement,listModules

from PhysicsTools.PatAlgos.tools.helpers import getPatAlgosToolsTask, addToProcessAndTask

def makeRecoJetCollection(process, 
                          pfCandCollection,
                          coneSize,
                          useCHSAlgo,
                          postfix):

    task = getPatAlgosToolsTask(process)

    jetColName="ak"+str(int(coneSize*10))+"PFJets"
    internalPfCandColl=pfCandCollection
    if useCHSAlgo:
        addToProcessAndTask("tmpPFCandCollPtr"+postfix, 
                            cms.EDProducer("PFCandidateFwdPtrProducer",
                                src = cms.InputTag(pfCandCollection) ),
                            process, task )
        process.load("CommonTools.ParticleFlow.pfNoPileUpJME_cff")
        task.add(process.pfNoPileUpJMETask)
        cloneProcessingSnippet(process, getattr(process,"pfNoPileUpJMESequence"), postfix, addToTask = True )
        getattr(process, "pfPileUpJME"+postfix).PFCandidates = cms.InputTag("tmpPFCandCollPtr"+postfix)
        getattr(process, "pfNoPileUpJME"+postfix).bottomCollection = cms.InputTag("tmpPFCandCollPtr"+postfix) 
        
        internalPfCandColl = "pfNoPileUpJME"+postfix
        jetColName+="CHS"
 
    addToProcessAndTask(jetColName+postfix,
                        getattr(process,jetColName).clone(
                            src = cms.InputTag(internalPfCandColl),
                            rParam=cms.double(coneSize),
                            doAreaFastjet = True),
                        process, task )

def reduceInputJetCollection(process, jetCollection, badMuons):

    task = getPatAlgosToolsTask(process)

    label = jetCollection.label()
    addToProcessAndTask(label+"AllEvents",
                        jetCollection.clone(),
                        process, task )
    process.globalReplace(label,
            cms.EDProducer("PFJetCollectionReducer",
                           writeEmptyCollection = cms.bool(True),
                           jetCollection = cms.InputTag(label+"AllEvents"),
                           triggeringCollections=badMuons,
                           )
            )

def reduceFinalJetCollection(process, jetCollection, badMuons):

    task = getPatAlgosToolsTask(process)

    label = jetCollection.label()
    addToProcessAndTask(label+"AllEvents",
                        jetCollection.clone(),
                        process, task )
    process.globalReplace(label,
            cms.EDProducer("PATJetCollectionReducer",
                           writeEmptyCollection = cms.bool(False),
                           jetCollection = cms.InputTag(label+"AllEvents"),
                           triggeringCollections=badMuons,
                           ) 
            )
    

def backupJetsFirstStep(process):

    task = getPatAlgosToolsTask(process)

    """Take snapshots of the sequences before we change the PFCandidates"""
    process.originalAK4JetTask, process.originalAK4JetSequence = listDependencyChain(process, getattr(process,"slimmedJets"), ('particleFlow', 'muons'))
    backupAK4JetSequence = cloneProcessingSnippet(process, getattr(process,"originalAK4JetSequence"), "Backup", addToTask = True )
    process.originalAK4PuppiJetTask, process.originalAK4PuppiJetSequence = listDependencyChain(process,getattr(process,"slimmedJetsPuppi"), ('particleFlow', 'muons'))
    backupAK4PuppiJetSequence = cloneProcessingSnippet(process, getattr(process,"originalAK4PuppiJetSequence"), "Backup", addToTask = True )
    process.originalAK8JetTask,process.originalAK8JetSequence = listDependencyChain(process, getattr(process,"slimmedJetsAK8"), ('particleFlow', 'muons'))
    backupAK8JetSequence = cloneProcessingSnippet(process, getattr(process,"originalAK8JetSequence"), "Backup", addToTask = True )

    task.add(process.originalAK4JetTask)
    task.add(process.originalAK4PuppiJetTask)
    task.add(process.originalAK8JetTask)

    return { 'AK4':backupAK4JetSequence, 'AK4Puppi':backupAK4PuppiJetSequence, 'AK8':backupAK8JetSequence }

    
def backupJetsSecondStep(process, sequences, badMuons, verbose=False):
    """Deploy the snapshots after the change of PFCandidates"""

    task = getPatAlgosToolsTask(process)

    # put back the old input tags and copy in task
    for sequence in sequences.itervalues():
        massSearchReplaceAnyInputTag(sequence, "pfCandidatesBadMuonsCleaned", "particleFlow")
        massSearchReplaceAnyInputTag(sequence, "muonsCleaned", "muons")
        for mod in listModules(sequence):
            task.add(mod)
    # gate the input collections to avoid re-running most of PAT on good events
    reduceInputJetCollection(process, process.ak4PFJetsCHSBackup, badMuons)
    reduceInputJetCollection(process, process.ak4PFJetsPuppiBackup, badMuons)
    # fix names in the valuemaps
    process.patJetsBackup.userData.userInts.labelPostfixesToStrip = cms.vstring("Backup",)
    process.patJetsBackup.userData.userFloats.labelPostfixesToStrip = cms.vstring("Backup",)
    process.patJetsAK8Backup.userData.userFloats.labelPostfixesToStrip = cms.vstring("Backup",)
    process.patJetsAK8PuppiBackup.userData.userFloats.labelPostfixesToStrip = cms.vstring("Backup",)    
    #
    # now deal with daughter links
    # for these we can keep the daughters
    if hasattr(process,"slimmedJetsBackup"):
        process.slimmedJetsBackup.mixedDaughters = True
        process.slimmedJetsBackup.packedPFCandidates = cms.InputTag("oldPFCandToPackedOrDiscarded")
        process.packedPatJetsAK8Backup.fixDaughters = False
       #for this one the link is broken using oldPFCandToPackedOrDiscarded...
        #not sure why, but result is the same
        process.slimmedJetsAK8Backup.rekeyDaughters = '1'
        process.slimmedJetsAK8Backup.mixedDaughters = False
        process.slimmedJetsAK8Backup.packedPFCandidates = cms.InputTag("packedPFCandidatesBackup") #oldPFCandToPackedOrDiscarded
        reduceFinalJetCollection(process, process.slimmedJetsBackup, badMuons)

    if hasattr(process,"slimmedJetsAK8PFCHSSoftDropSubjetsBackup"):
        process.slimmedJetsAK8PFCHSSoftDropSubjetsBackup.mixedDaughters = True
        process.slimmedJetsAK8PFCHSSoftDropSubjetsBackup.packedPFCandidates = cms.InputTag("oldPFCandToPackedOrDiscarded")
        reduceFinalJetCollection(process, process.slimmedJetsAK8Backup, badMuons)
        #for this one the link is broken using oldPFCandToPackedOrDiscarded...
        #not sure why, but result is the same
        process.slimmedJetsAK8BackupAllEvents.packedPFCandidates = cms.InpuTag("packedPFCandidatesBackup")

    # for these we can't
    if hasattr(process,"slimmedJetsPuppiBackup"):
        process.slimmedJetsPuppiBackup.dropDaughters = '1'
        process.slimmedJetsAK8PFPuppiSoftDropSubjetsBackup.dropDaughters = '1'
        reduceFinalJetCollection(process, process.slimmedJetsPuppiBackup, badMuons)
  
    #
    addKeepStatement(process,
                     "keep *_slimmedJets_*_*",
                     ["keep *_slimmedJetsBackup_*_*"],
                     verbose=verbose)
    addKeepStatement(process, "keep *_slimmedJetsPuppi_*_*",
                     ["keep *_slimmedJetsPuppiBackup_*_*"],
                     verbose=verbose)
    addKeepStatement(process,
                     "keep *_slimmedJetsAK8_*_*",
                     ["keep *_slimmedJetsAK8Backup_*_*"],
                     verbose=verbose)
    addKeepStatement(process,"keep *_slimmedJetsAK8PFCHSSoftDropPacked_SubJets_*",
                     ["keep *_slimmedJetsAK8PFCHSSoftDropPackedBackup_SubJets_*"],
                     verbose=verbose)
    addKeepStatement(process,"keep *_slimmedJetsAK8PFPuppiSoftDropPacked_SubJets_*",
                     ["keep *_slimmedJetsAK8PFPuppiSoftDropPackedBackup_SubJets_*"],
                     verbose=verbose)

    
