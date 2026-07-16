# Original author: Felice Pantaleo (CERN) <felice.pantaleo@cern.ch>
#
# cmsDriver --customise hook for the truth-graph history-guard unit test. Appended
# to a short GEN,SIM job run with --procModifiers enableTruth, it builds the truth
# graph straight from the freshly simulated SimTracks/SimVertices and runs the
# TruthGraphTopologyChecker in failOnViolations mode, so the job throws if the
# SimTrack/SimVertex history is not one tree fully connected to the generator -
# exactly the regression a simulation change that drops the per-track parentage
# (e.g. a port that no longer records parentID for every track) would cause.

import FWCore.ParameterSet.Config as cms


def addTruthHistoryGuard(process):
    process.truthGraphProducer = cms.EDProducer(
        "TruthGraphProducer",
        genEventHepMC3=cms.InputTag("generatorSmeared"),
        genEventHepMC=cms.InputTag("generatorSmeared"),
        simTracks=cms.InputTag("g4SimHits"),
        simVertices=cms.InputTag("g4SimHits"),
        addGenToSimEdges=cms.bool(True),
    )

    process.truthLogicalGraphProducer = cms.EDProducer(
        "TruthLogicalGraphProducer",
        src=cms.InputTag("truthGraphProducer"),
        simTracks=cms.InputTag("g4SimHits"),
        simVertices=cms.InputTag("g4SimHits"),
        genEventHepMC3=cms.InputTag("generatorSmeared"),
        genEventHepMC=cms.InputTag("generatorSmeared"),
        mergeGenSimVertices=cms.bool(True),
        postProcessing=cms.PSet(
            collapseIntermediateGenParticles=cms.bool(True),
            seedPdgIds=cms.vint32(),  # full graph, no selection
            seedHadronFlavors=cms.vint32(),
            seedParentDepth=cms.uint32(0),
            keepStableSpectators=cms.bool(True),
            decayPdgIdGroups=cms.VPSet(),
            ignoredPdgIds=cms.vint32(),
            ignoredParticleIds=cms.vuint32(),
        ),
    )

    process.truthHistoryGuard = cms.EDAnalyzer(
        "TruthGraphTopologyChecker",
        rawSrc=cms.InputTag("truthGraphProducer"),
        src=cms.InputTag("truthLogicalGraphProducer"),
        # Throw at endJob if the history is fragmented (orphan components / cycles).
        failOnViolations=cms.untracked.bool(True),
    )

    process.MessageLogger.cerr.TruthGraphTopologyChecker = cms.untracked.PSet(limit=cms.untracked.int32(-1))

    process.truthHistoryGuardPath = cms.EndPath(
        process.truthGraphProducer + process.truthLogicalGraphProducer + process.truthHistoryGuard
    )
    process.schedule.append(process.truthHistoryGuardPath)

    return process
