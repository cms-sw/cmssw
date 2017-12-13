import FWCore.ParameterSet.Config as cms

from PhysicsTools.PatAlgos.tools.helpers import MassSearchReplaceAnyInputTagVisitor, addKeepStatement
from PhysicsTools.PatAlgos.tools.helpers import getPatAlgosToolsTask

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
    process.load("RecoBTag.Combined.deepFlavour_cff")
    task.add(process.pfDeepFlavourTask)
    process.load("RecoBTag.CTagging.cTagging_cff")
    task.add(process.cTaggingTask)
    process.load("RecoVertex.AdaptiveVertexFinder.inclusiveVertexing_cff")
    task.add(process.inclusiveVertexingTask)
    task.add(process.inclusiveCandidateVertexingTask)
    task.add(process.inclusiveCandidateVertexingCvsLTask)


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
        tag = cms.string("HcalRespCorrs_v1.02_express") #to be replaced with proper tag name once available
        )
    )

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


def customizeAll(process, verbose=True): #change to false by default

    if verbose:
        print "===>>> customizing the process for legacy rereco 2016"

    loadJetMETBTag(process)

    cleanPfCandidates(process, verbose)
    addDiscardedPFCandidates(process, cms.InputTag("pfCandidatesBadHadRecalibrated","discarded"), verbose=verbose)


    return process
