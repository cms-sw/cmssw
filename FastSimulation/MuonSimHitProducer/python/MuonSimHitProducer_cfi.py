import FWCore.ParameterSet.Config as cms

from RecoMuon.TrackingTools.MuonServiceProxy_cff import *
from FastSimulation.MaterialEffects.MaterialEffects_cfi import *
MuonSimHits = cms.EDProducer("MuonSimHitProducer",
    # Material effects (Multiple scattering)
    MaterialEffectsForMuonsBlock,
    # Services
    MuonServiceProxy,
    # Muons
    MUONS = cms.PSet(
        # The muon simtrack's must be taken from there
        simModuleLabel = cms.string('famosSimHits'),
        simModuleProcess = cms.string('MuonSimTracks'),
        # The reconstruted tracks must be taken from there
        trackModuleLabel = cms.string('generalTracks'),
        simHitDTIneffParameters  = cms.vdouble(0.342, -4.597),
        simHitCSCIneffParameters = cms.vdouble(0.200, -3.199)
    ),
    Chi2EstimatorCut = cms.double(1000.0),
    TRACKS = cms.PSet(
        # Set to true if the full pattern recognition was used
        # to reconstruct tracks in the tracker
        FullPatternRecognition = cms.untracked.bool(False)
    )
)


