import FWCore.ParameterSet.Config as cms

cosmicInTracker = cms.EDFilter("CosmicGenFilterHelix",
    src = cms.InputTag("generator"),
    pdgIds = cms.vint32(-13, 13), ## only generated particles of these IDs are considered
    charges = cms.vint32(1, -1),  ## needs to be parallel to pdgIds

    propagator = cms.string('SteppingHelixPropagatorAlong'),

    # defines dimensions of target cylinder in cm (full tracker: r=112, z=+/- 270)
    radius = cms.double(80.0), ## i.e. at least for barrel layers
    minZ = cms.double(-212.0), ## i.e. at least four TEC discs
    maxZ = cms.double(212.0),  ## dito

    # momentum cuts after propagation:
    minPt = cms.double(0.0),
    minP = cms.double(0.0),

    doMonitor = cms.untracked.bool(False) # Fill monitoring histograms? Needs TFileService!
)
