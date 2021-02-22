from __future__ import print_function
import FWCore.ParameterSet.Config as cms
from PhysicsTools.NanoAOD.common_cff import *
from PhysicsTools.NanoAOD.genparticles_cff import genParticleTable
from PhysicsTools.NanoAOD.genVertex_cff import *
from DPGAnalysis.HGCalNanoAOD.hgcSimHits_cff import *
from DPGAnalysis.HGCalNanoAOD.hgcSimTracks_cff import *
from DPGAnalysis.HGCalNanoAOD.trackSimHits_cff import *
from DPGAnalysis.HGCalNanoAOD.hgcRecHits_cff import *
from DPGAnalysis.HGCalNanoAOD.simClusters_cff import *
from DPGAnalysis.HGCalNanoAOD.caloParticles_cff import *
from DPGAnalysis.TrackNanoAOD.trackingParticles_cff import *
from DPGAnalysis.TrackNanoAOD.tracks_cff import *
from DPGAnalysis.PFNanoAOD.pfCands_cff import *

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

def customizeNoMergedCaloTruth(process):
    process.nanoHGCMLSequence.remove(simClusterTable)
    process.nanoHGCMLSequence.remove(simClusterToCaloPartTable)
    process.nanoHGCMLSequence.remove(hgcEEHitsToSimClusterTable)
    process.nanoHGCMLSequence.remove(hgcHEfrontHitsToSimClusterTable)
    process.nanoHGCMLSequence.remove(hgcHEbackHitsToSimClusterTable)
    process.nanoHGCMLSequence.remove(simTrackToSimClusterTable)

    process.nanoHGCMLSequence.remove(caloParticleTable)
    return process

def customizeMergedSimClusters(process):
    process.nanoHGCMLSequence.insert(1, mergedSimClusterTables)
    return process
