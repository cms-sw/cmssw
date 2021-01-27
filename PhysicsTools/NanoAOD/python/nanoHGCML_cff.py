from __future__ import print_function
import FWCore.ParameterSet.Config as cms
from PhysicsTools.NanoAOD.common_cff import *
from hgcSimHits_cff import *
from hgcSimTracks_cff import *
from trackSimHits_cff import *
from hgcRecHits_cff import *
from simClusters_cff import *
from caloParticles_cff import *
from trackingParticles_cff import *
from tracks_cff import *
from genparticles_cff import genParticleTable
from genVertex_cff import *
from pfCands_cff import *

nanoMetadata = cms.EDProducer("UniqueStringProducer",
    strings = cms.PSet(
        tag = cms.string("untagged"),
    )
)

genParticleTable.src = "genParticles"
genParticleTable.variables = cms.PSet(genParticleTable.variables,
    charge = CandVars.charge)

nanoHGCMLSequence = cms.Sequence(nanoMetadata+genVertexTables+genParticleTable+
        trackingParticleTable+caloParticleTable+simClusterTables+
        simTrackTables+hgcSimHitsSequence+trackerSimHitTables)

def customizeReco(process):
    process.nanoHGCMLSequence.insert(1, hgcRecHitsSequence)
    process.nanoHGCMLSequence.insert(2, pfCandTable)
    process.nanoHGCMLSequence.insert(3, pfTICLCandTable)
    process.nanoHGCMLSequence.insert(4, trackTables)
    return process

def customizeNoSimClusters(process):
    process.nanoHGCMLSequence.remove(simClusterTable)
    process.nanoHGCMLSequence.remove(simClusterToCaloPartTable)
    process.nanoHGCMLSequence.remove(hgcEEHitsToSimClusterTable)
    process.nanoHGCMLSequence.remove(hgcHEfrontHitsToSimClusterTable)
    process.nanoHGCMLSequence.remove(hgcHEbackHitsToSimClusterTable)
    process.nanoHGCMLSequence.remove(simTrackToSimClusterTable)
    return process

def customizeMergedSimClusters(process):
    process.nanoHGCMLSequence.insert(1, mergedSimClusterTables)
    return process
