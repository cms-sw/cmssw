import FWCore.ParameterSet.Config as cms

from PhysicsTools.PatAlgos.tools.helpers import MassSearchReplaceAnyInputTagVisitor, addKeepStatement
from PhysicsTools.PatAlgos.tools.helpers import getPatAlgosToolsTask

def loadJetMETBTag(process):

    task = getPatAlgosToolsTask(process)

    import RecoParticleFlow.PFProducer.pfLinker_cff
    process.particleFlowPtrs = RecoParticleFlow.PFProducer.pfLinker_cff.particleFlowPtrs.clone()
    task.add(process.particleFlowPtrs)

    process.load("CommonTools.ParticleFlow.pfNoPileUpIso_cff")
    task.add(process.pfNoPileUpIsoTask)
    process.load("CommonTools.ParticleFlow.ParticleSelectors.pfSortByType_cff")
    task.add(process.pfSortByTypeTask)

    import RecoJets.Configuration.RecoPFJets_cff
    process.ak4PFJetsCHS = RecoJets.Configuration.RecoPFJets_cff.ak4PFJetsCHS.clone()
    task.add(process.ak4PFJetsCHS)
    # need also the non-CHS ones as they are used to seed taus
    process.ak4PFJets = RecoJets.Configuration.RecoPFJets_cff.ak4PFJets.clone()
    task.add(process.ak4PFJets)
    process.ak8PFJetsCHS = RecoJets.Configuration.RecoPFJets_cff.ak8PFJetsCHS.clone()
    task.add(process.ak8PFJetsCHS)

    process.fixedGridRhoAll = RecoJets.Configuration.RecoPFJets_cff.fixedGridRhoAll.clone()
    process.fixedGridRhoFastjetAll = RecoJets.Configuration.RecoPFJets_cff.fixedGridRhoFastjetAll.clone()
    process.fixedGridRhoFastjetCentral = RecoJets.Configuration.RecoPFJets_cff.fixedGridRhoFastjetCentral.clone()
    process.fixedGridRhoFastjetCentralChargedPileUp = RecoJets.Configuration.RecoPFJets_cff.fixedGridRhoFastjetCentralChargedPileUp.clone()
    process.fixedGridRhoFastjetCentralNeutral = RecoJets.Configuration.RecoPFJets_cff.fixedGridRhoFastjetCentralNeutral.clone()
    task.add( process.fixedGridRhoAll,
              process.fixedGridRhoFastjetAll,
              process.fixedGridRhoFastjetCentral,
              process.fixedGridRhoFastjetCentralChargedPileUp,
              process.fixedGridRhoFastjetCentralNeutral )

    process.load("RecoJets.JetAssociationProducers.ak4JTA_cff")
    task.add(process.ak4JetTracksAssociatorAtVertexPF)

    process.load('RecoBTag.Configuration.RecoBTag_cff')
    task.add(process.btaggingTask)

    process.load("RecoMET.METProducers.PFMET_cfi")
    task.add(process.pfMet)


def cleanPfCandidates(process, verbose=False):
    task = getPatAlgosToolsTask(process)

    #add producer at the beginning of the schedule
    process.load("CommonTools.ParticleFlow.pfCandidateRecalibrator_cfi")
    task.add(process.pfCandidateRecalibrator)

    replacePFCandidates = MassSearchReplaceAnyInputTagVisitor("particleFlow", "pfCandidateRecalibrator", verbose=verbose)
    replacePFTmpPtrs = MassSearchReplaceAnyInputTagVisitor("particleFlowTmpPtrs", "particleFlowPtrs", verbose=verbose)
    for everywhere in [ process.producers, process.filters, process.analyzers, process.psets, process.vpsets ]:
        for name,obj in everywhere.iteritems():
            if obj != process.pfCandidateRecalibrator:
                replacePFCandidates.doIt(obj, name)
                replacePFTmpPtrs.doIt(obj, name)


    process.load("CommonTools.ParticleFlow.pfEGammaToCandidateRemapper_cfi")
    task.add(process.pfEGammaToCandidateRemapper)
    process.pfEGammaToCandidateRemapper.pf2pf = cms.InputTag("pfCandidateRecalibrator")
    process.reducedEgamma.gsfElectronsPFValMap = cms.InputTag("pfEGammaToCandidateRemapper","electrons")
    process.reducedEgamma.photonsPFValMap      = cms.InputTag("pfEGammaToCandidateRemapper","photons")


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


def customizeAll(process, verbose=False):

    if verbose:
        print "===>>> customizing the process for legacy rereco 2016"

    loadJetMETBTag(process)

    cleanPfCandidates(process, verbose)
    addDiscardedPFCandidates(process, cms.InputTag("pfCandidateRecalibrator","discarded"), verbose=verbose)

    return process
