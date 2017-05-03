import FWCore.ParameterSet.Config as cms

from PhysicsTools.PatAlgos.tools.helpers import MassSearchReplaceAnyInputTagVisitor, cloneProcessingSnippet, addKeepStatement
from PhysicsTools.PatAlgos.slimming.extraJets_MuEGFixMoriond2017 import backupJetsFirstStep, backupJetsSecondStep
#from RecoEgamma.EgammaTools.egammaGainSwitchFixToolsForPAT_cff import customizeGSFixForPAT
from PhysicsTools.PatAlgos.tools.helpers import getPatAlgosToolsTask, addToProcessAndTask
import sys

def addBadMuonFilters(process):

    task = getPatAlgosToolsTask(process)

    process.load("RecoMET.METFilters.badGlobalMuonTaggersAOD_cff")
    task.add(process.badGlobalMuonTagger)
    task.add(process.cloneGlobalMuonTagger)
    process.Flag_noBadMuons = cms.Path(process.noBadGlobalMuons)
    process.Flag_badMuons = cms.Path(process.badGlobalMuonTagger)
    process.Flag_duplicateMuons = cms.Path(process.cloneGlobalMuonTagger)
    for P in process.Flag_noBadMuons, process.Flag_badMuons, process.Flag_duplicateMuons:
        process.schedule.insert(0, P)

def cleanPFCandidates(process, badMuons, verbose=False):

    task = getPatAlgosToolsTask(process)

    process.load("CommonTools.ParticleFlow.muonsCleaned_cfi")
    task.add(process.muonsCleaned)
    process.patMuons.userData.userInts.src = [ cms.InputTag("muonsCleaned:oldPF") ]

    process.load("CommonTools.ParticleFlow.pfCandidatesBadMuonsCleaned_cfi")
    task.add(process.pfCandidatesBadMuonsCleaned)
    process.muonsCleaned.badmuons = badMuons
    replaceMuons = MassSearchReplaceAnyInputTagVisitor("muons", "muonsCleaned", verbose=verbose)
    needOriginalMuons = [ process.muonsCleaned ] + [ getattr(process,l.moduleLabel) for l in badMuons ]
    replacePFCandidates = MassSearchReplaceAnyInputTagVisitor("particleFlow", "pfCandidatesBadMuonsCleaned", verbose=verbose)
    for everywhere in [ process.producers, process.filters, process.analyzers, process.psets, process.vpsets ]:
        for name,obj in everywhere.iteritems():
            if obj not in needOriginalMuons:
                replaceMuons.doIt(obj, name)
            if obj != process.pfCandidatesBadMuonsCleaned: 
                replacePFCandidates.doIt(obj, name)
            
    process.load("CommonTools.ParticleFlow.pfEGammaToCandidateRemapper_cfi")
    task.add(process.pfEGammaToCandidateRemapper)
    process.pfEGammaToCandidateRemapper.pf2pf = cms.InputTag("pfCandidatesBadMuonsCleaned")
    process.reducedEgamma.gsfElectronsPFValMap = cms.InputTag("pfEGammaToCandidateRemapper","electrons")
    process.reducedEgamma.photonsPFValMap      = cms.InputTag("pfEGammaToCandidateRemapper","photons")
    if hasattr(process,"gedGsfElectronsFixed"):
        # also reconfigure pfEGammaToCandidateRemapper because of GS Fix
        # first the old one
        process.pfEGammaToCandidateRemapperBeforeGSFix = process.pfEGammaToCandidateRemapper.clone()
        task.add(process.pfEGammaToCandidateRemapperBeforeGSFix)
        process.reducedEgammaBeforeGSFix.gsfElectronsPFValMap = cms.InputTag("pfEGammaToCandidateRemapperBeforeGSFix","electrons")
        process.reducedEgammaBeforeGSFix.photonsPFValMap      = cms.InputTag("pfEGammaToCandidateRemapperBeforeGSFix","photons")
        # then the new one
        process.pfEGammaToCandidateRemapper.electrons = cms.InputTag("gedGsfElectronsFixed")
        process.pfEGammaToCandidateRemapper.photons   = cms.InputTag("gedPhotonsFixed")
        process.pfEGammaToCandidateRemapper.electron2pf = cms.InputTag("particleBasedIsolationGSFixed","gedGsfElectrons")
        process.pfEGammaToCandidateRemapper.photon2pf   = cms.InputTag("particleBasedIsolationGSFixed","gedPhotons")
    else:
        sys.stderr.write("WARNING : attempt to use gain switch corrected electron/photon collection gedGsfElectronsFixed, but the current process does not contain such collection")

def addDiscardedPFCandidates(process, inputCollection, verbose=False):

    task = getPatAlgosToolsTask(process)

    process.primaryVertexAssociationDiscardedCandidates = process.primaryVertexAssociation.clone(
        particles = inputCollection,
        )
    task.add(process.primaryVertexAssociationDiscardedCandidates)
    process.packedPFCandidatesDiscarded = process.packedPFCandidates.clone(
        inputCollection = inputCollection,
        PuppiNoLepSrc = cms.InputTag(""),
        PuppiSrc = cms.InputTag(""),
        secondaryVerticesForWhiteList = cms.VInputTag(),
        vertexAssociator = cms.InputTag("primaryVertexAssociationDiscardedCandidates","original")
        )
    task.add(process.packedPFCandidatesDiscarded)
    addKeepStatement(process, "keep patPackedCandidates_packedPFCandidates_*_*",
                             ["keep patPackedCandidates_packedPFCandidatesDiscarded_*_*"],
                              verbose=verbose)
    # Now make the mixed map for rekeying
    from PhysicsTools.PatAlgos.slimming.packedPFCandidateRefMixer_cfi import packedPFCandidateRefMixer
    process.oldPFCandToPackedOrDiscarded = packedPFCandidateRefMixer.clone(
        pf2pf = cms.InputTag(inputCollection.moduleLabel),
        pf2packed = cms.VInputTag(cms.InputTag("packedPFCandidates"), cms.InputTag("packedPFCandidatesDiscarded"))
    )
    task.add(process.oldPFCandToPackedOrDiscarded)
    # Fix slimmed muon keying
    process.slimmedMuons.pfCandidates = cms.VInputTag(cms.InputTag(inputCollection.moduleLabel), inputCollection)
    process.slimmedMuons.packedPFCandidates = cms.VInputTag(cms.InputTag("packedPFCandidates"), cms.InputTag("packedPFCandidatesDiscarded"))
  
    #MM point to uncleaned collection for hadronic taus, to avoid remaking them
    #no impact expected, as no muons are included here
    process.slimmedTaus.packedPFCandidates=cms.InputTag("packedPFCandidatesBackup")

def loadJetMETBTag(process):

    task = getPatAlgosToolsTask(process)

    import RecoJets.Configuration.RecoPFJets_cff
    process.ak4PFJetsCHS = RecoJets.Configuration.RecoPFJets_cff.ak4PFJetsCHS.clone()
    task.add(process.ak4PFJetsCHS)
    process.ak8PFJetsCHS = RecoJets.Configuration.RecoPFJets_cff.ak8PFJetsCHS.clone()
    task.add(process.ak8PFJetsCHS)
    process.load("RecoMET.METProducers.PFMET_cfi")
    task.add(process.pfMet)
    process.load("RecoBTag.ImpactParameter.impactParameter_cff")
    task.add(process.impactParameterTask)
    process.load("RecoBTag.SecondaryVertex.secondaryVertex_cff")
    task.add(process.secondaryVertexTask)
    process.load("RecoBTag.SoftLepton.softLepton_cff")
    task.add(process.softLeptonTask)
    process.load("RecoBTag.Combined.combinedMVA_cff")
    task.add(process.combinedMVATask)
    process.load("RecoBTag.CTagging.cTagging_cff")
    task.add(process.cTaggingTask)
    process.load("RecoVertex.AdaptiveVertexFinder.inclusiveVertexing_cff")
    task.add(process.inclusiveVertexingTask)
    task.add(process.inclusiveCandidateVertexingTask)
    task.add(process.inclusiveCandidateVertexingCvsLTask)

def customizeAll(process, verbose=False):
    
    #disabled for now, backup in case 90X needs similar fix
    #process = customizeGSFixForPAT(process)

    loadJetMETBTag(process)
    backupJetSequences = backupJetsFirstStep(process)

    addBadMuonFilters(process)    
    badMuons = cms.VInputTag( cms.InputTag("badGlobalMuonTagger","bad"), cms.InputTag("cloneGlobalMuonTagger","bad") )
    # clean the muons and PF candidates, and make *everything* point to the new candidates
    cleanPFCandidates(process, badMuons, verbose=verbose) 

    addDiscardedPFCandidates(process, cms.InputTag("pfCandidatesBadMuonsCleaned","discarded"), verbose=verbose)

    # now make the backup sequences point to the right place
    backupJetsSecondStep(process, backupJetSequences, badMuons, verbose=verbose)

    
    process.patMuons.embedCaloMETMuonCorrs = False # FIXME
    ##extra METs and MET corrections ===============================================================
    from PhysicsTools.PatAlgos.slimming.extraSlimmedMETs_MuEGFixMoriond2017 import addExtraMETCollections,addExtraPuppiMETCorrections
    
    ### Gain switch collections not existing in 90X+ 
    ### -> corrections are set up to give no change on the MET computation
    addExtraMETCollections(process,
                           unCleanPFCandidateCollection="particleFlow",
                           cleanElectronCollection="slimmedElectrons",
                           cleanPhotonCollection="slimmedPhotons",
                           unCleanElectronCollection="slimmedElectrons",
                           unCleanPhotonCollection="slimmedPhotons")
    
    addExtraPuppiMETCorrections(process,
                                cleanPFCandidateCollection="particleFlow",
                                unCleanPFCandidateCollection="pfCandidatesBadMuonsCleaned",
                                cleanElectronCollection="slimmedElectrons",
                                cleanPhotonCollection="slimmedPhotons",
                                unCleanElectronCollection="slimmedElectrons",
                                unCleanPhotonCollection="slimmedPhotons")

    addKeepStatement(process,
                     "keep *_slimmedMETs_*_*",
                     ["keep *_slimmedMETsUncorrected_*_*",
                      "keep *_slimmedMETsEGClean_*_*",
                      "keep *_slimmedMETsMuEGClean_*_*"],
                     verbose=verbose)
    addKeepStatement(process,
                     "keep *_slimmedMETsPuppi_*_*",
                     ["keep *_puppiMETEGCor_*_*",
                      "keep *_puppiMETMuCor_*_*"],
                     verbose=verbose)



    #redo the miniAOD data customization for new JEC modules created during the backup process
    from PhysicsTools.PatAlgos.slimming.miniAOD_tools import miniAOD_customizeData
    miniAOD_customizeData(process)

    return process
