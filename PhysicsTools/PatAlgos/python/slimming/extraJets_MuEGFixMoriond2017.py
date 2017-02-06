import FWCore.ParameterSet.Config as cms
from PhysicsTools.PatAlgos.tools.helpers import listDependencyChain, massSearchReplaceAnyInputTag, cloneProcessingSnippet, addKeepStatement

def makeRecoJetCollection(process, 
                          pfCandCollection,
                          coneSize,
                          useCHSAlgo,
                          postfix):

    jetColName="ak"+str(int(coneSize*10))+"PFJets"
    internalPfCandColl=pfCandCollection
    if useCHSAlgo:
        setattr( process, "tmpPFCandCollPtr"+postfix, 
                 cms.EDProducer("PFCandidateFwdPtrProducer",
                                src = cms.InputTag(pfCandCollection) ) 
                 )
        process.load("CommonTools.ParticleFlow.pfNoPileUpJME_cff")
        cloneProcessingSnippet(process, getattr(process,"pfNoPileUpJMESequence"), postfix )
        getattr(process, "pfPileUpJME"+postfix).PFCandidates = cms.InputTag("tmpPFCandCollPtr"+postfix)
        getattr(process, "pfNoPileUpJME"+postfix).bottomCollection = cms.InputTag("tmpPFCandCollPtr"+postfix) 
        
        internalPfCandColl = "pfNoPileUpJME"+postfix
        jetColName+="CHS"
 
    setattr(process, jetColName+postfix, getattr(process,"ak4PFJets").clone(
            src = cms.InputTag(internalPfCandColl),
            rParam=cms.double(coneSize),
            doAreaFastjet = True)
            )
    


def reduceInputJetCollection(process, jetCollection, badMuons):
    label = jetCollection.label()
    setattr(process, label+"AllEvents", jetCollection.clone())
    process.globalReplace(label,
            cms.EDProducer("PFJetCollectionReducer",
                           writeEmptyCollection = cms.bool(True),
                           jetCollection = cms.InputTag(label+"AllEvents"),
                           triggeringCollections=badMuons,
                           )
            )

def reduceFinalJetCollection(process, jetCollection, badMuons):
    label = jetCollection.label()
    setattr(process, label+"AllEvents", jetCollection.clone())
    process.globalReplace(label,
            cms.EDProducer("PATJetCollectionReducer",
                           writeEmptyCollection = cms.bool(False),
                           jetCollection = cms.InputTag(label+"AllEvents"),
                           triggeringCollections=badMuons,
                           ) 
            )
    

def backupJetsFirstStep(process):
    """Take snapshots of the sequences before we change the PFCandidates"""
    process.originalAK4JetSequence = listDependencyChain(process, process.slimmedJets, ('particleFlow', 'muons'))
    backupAK4JetSequence = cloneProcessingSnippet(process, process.originalAK4JetSequence, "Backup")
    process.originalAK4PuppiJetSequence = listDependencyChain(process, process.slimmedJetsPuppi, ('particleFlow', 'muons'))
    backupAK4PuppiJetSequence = cloneProcessingSnippet(process, process.originalAK4PuppiJetSequence, "Backup")
    process.originalAK8JetSequence = listDependencyChain(process, process.slimmedJetsAK8, ('particleFlow', 'muons'))
    backupAK8JetSequence = cloneProcessingSnippet(process, process.originalAK8JetSequence, "Backup")
    return { 'AK4':backupAK4JetSequence, 'AK4Puppi':backupAK4PuppiJetSequence, 'AK8':backupAK8JetSequence }

    
def backupJetsSecondStep(process, sequences, badMuons, verbose=False):
    """Deploy the snapshots after the change of PFCandidates"""
    # put back the old input tags
    for sequence in sequences.itervalues():
        massSearchReplaceAnyInputTag(sequence, "pfCandidatesBadMuonsCleaned", "particleFlow")
        massSearchReplaceAnyInputTag(sequence, "muonsCleaned", "muons")
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
    process.slimmedJetsBackup.mixedDaughters = True
    process.slimmedJetsBackup.packedPFCandidates = cms.InputTag("oldPFCandToPackedOrDiscarded")
    process.slimmedJetsAK8PFCHSSoftDropSubjetsBackup.mixedDaughters = True
    process.slimmedJetsAK8PFCHSSoftDropSubjetsBackup.packedPFCandidates = cms.InputTag("oldPFCandToPackedOrDiscarded")
    # for these we can't
    process.slimmedJetsPuppiBackup.dropDaughters = '1'
    process.slimmedJetsAK8PFPuppiSoftDropSubjetsBackup.dropDaughters = '1'
    # for these we do even if we wouldn't have done in the standard case, since we couldn't for the subjets
    process.packedPatJetsAK8Backup.fixDaughters = False
    process.slimmedJetsAK8Backup.rekeyDaughters = '1'
    process.slimmedJetsAK8Backup.mixedDaughters = True
    process.slimmedJetsAK8Backup.packedPFCandidates = cms.InputTag("oldPFCandToPackedOrDiscarded")
    #
    reduceFinalJetCollection(process, process.slimmedJetsBackup, badMuons)
    reduceFinalJetCollection(process, process.slimmedJetsPuppiBackup, badMuons)
    reduceFinalJetCollection(process, process.slimmedJetsAK8Backup, badMuons)
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

    
