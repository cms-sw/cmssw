# The following comments couldn't be translated into the new config version:

# This uses all the information (also the rechits)
# Set updatedState to combinedState to exclude rechits (usefull to evaluate pulls and residuals)

import FWCore.ParameterSet.Config as cms

modTIFNtupleMaker = cms.EDFilter("TIFNtupleMaker",
    SeedsLabel = cms.string('cosmicseedfinder'),
    # Objects Labels
    TracksLabel = cms.string('cosmictrackfinder'),
    TEC_ON = cms.bool(True),
    TID_ON = cms.bool(True),
    TOB_ON = cms.bool(True),
    TIB_ON = cms.bool(True),
    #  string Fitter = "KFFittingSmoother"   
    #  string Propagator = "PropagatorWithMaterial" 
    #  string src = "ctfWithMaterialTracks"
    fileName = cms.string('test_TAC.root'),
    SINGLE_DETECTORS = cms.bool(False),
    TTRHBuilder = cms.string('WithTrackAngle'),
    TrackInfoLabel = cms.InputTag("trackinfo","updatedState"),
    dCrossTalkErr = cms.untracked.double(0.1),
    oSiStripDigisLabel = cms.untracked.string('SiStripDigis'),
    oSiStripDigisProdInstName = cms.untracked.string('ZeroSuppressed')
)


