import FWCore.ParameterSet.Config as cms

from PhysicsTools.PatAlgos.tools.helpers import MassSearchReplaceAnyInputTagVisitor, addKeepStatement
from PhysicsTools.PatAlgos.tools.helpers import getPatAlgosToolsTask

def loadJetMETBTag(process):

    task = getPatAlgosToolsTask(process)

    import RecoJets.Configuration.RecoPFJets_cff
    process.ak4PFJetsCHS = RecoJets.Configuration.RecoPFJets_cff.ak4PFJetsCHS.clone()
    task.add(process.ak4PFJetsCHS)
    # need also the non-CHS ones as they are used to seed taus
    process.ak4PFJets = RecoJets.Configuration.RecoPFJets_cff.ak4PFJets.clone()
    task.add(process.ak4PFJets)
    process.ak8PFJetsCHS = RecoJets.Configuration.RecoPFJets_cff.ak8PFJetsCHS.clone()
    task.add(process.ak8PFJetsCHS)
    process.load("RecoJets.JetAssociationProducers.ak4JTA_cff")
    task.add(process.ak4JetTracksAssociatorAtVertexPF)

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

    process.load("RecoMET.METProducers.PFMET_cfi")
    task.add(process.pfMet)


def cleanPfCandidates(process, verbose=False):
    task = getPatAlgosToolsTask(process)

    #add producer at the beginning of the schedule
    process.load("CommonTools.ParticleFlow.pfCandidatesBadHadRecalibrated_cfi")
    task.add(process.pfCandidatesBadHadRecalibrated)

    replacePFCandidates = MassSearchReplaceAnyInputTagVisitor("particleFlow", "pfCandidatesBadHadRecalibrated", verbose=verbose)
    for everywhere in [ process.producers, process.filters, process.analyzers, process.psets, process.vpsets ]:
        for name,obj in everywhere.iteritems():
            if obj != process.pfCandidatesBadHadRecalibrated:
                replacePFCandidates.doIt(obj, name)


    process.load("CommonTools.ParticleFlow.pfEGammaToCandidateRemapper_cfi")
    task.add(process.pfEGammaToCandidateRemapper)
    process.pfEGammaToCandidateRemapper.pf2pf = cms.InputTag("pfCandidatesBadHadRecalibrated")
    process.reducedEgamma.gsfElectronsPFValMap = cms.InputTag("pfEGammaToCandidateRemapper","electrons")
    process.reducedEgamma.photonsPFValMap      = cms.InputTag("pfEGammaToCandidateRemapper","photons")


    #add bugged conditions to GT for comparison
    process.GlobalTag.toGet.append(cms.PSet(
        record = cms.string("HcalRespCorrsRcd"),
        label = cms.untracked.string("bugged"),
        tag = cms.string("HcalRespCorrs_v6.3_offline")
        )
    )


    #=== TMP FOR TESTING ONLY
    #process.load("CondCore.DBCommon.CondDBSetup_cfi")
    #process.es_pool = cms.ESSource("PoolDBESSource",
    #                               process.CondDBSetup,
    #                               timetype = cms.string('runnumber'),
    #                               toGet = cms.VPSet(
    #                                            cms.PSet(record = cms.string("HcalRespCorrsRcd"),
    #                                            tag = cms.string("HcalRespCorrs_2016legacy_fixBadCalib")
    #                                                     )
    #                                            ),
    #                               connect = cms.string('frontier://FrontierProd/CMS_CONDITIONS'),
    #                               authenticationMethod = cms.untracked.uint32(0)
    #                               )
    #process.es_prefer_es_pool = cms.ESPrefer( "PoolDBESSource", "es_pool" )
    #=== END - TMP FOR TESTING ONLY


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


def customizeAll(process, verbose=False):

    if verbose:
        print "===>>> customizing the process for legacy rereco 2016"

    loadJetMETBTag(process)

    cleanPfCandidates(process, verbose)
    addDiscardedPFCandidates(process, cms.InputTag("pfCandidatesBadHadRecalibrated","discarded"), verbose=verbose)

    #=== TMP FOR TESTING ONLY
    #addKeepStatement(process, "keep patPackedCandidates_packedPFCandidates_*_*",
    #                         ["keep patPackedCandidates_packedPFCandidatesDiscarded_*_*",
    #                          "keep *_particleFlow__*",
    #                          "keep *_pfCandidatesBadHadRecalibrated__*"],
    #                          verbose=verbose)
    #=== END - TMP FOR TESTING ONLY


    return process
