import FWCore.ParameterSet.Config as cms

SiPixelHitEfficiencySource = cms.EDAnalyzer("SiPixelHitEfficiencySource",
    src = cms.InputTag("siPixelHitEfficiency"),
    debug = cms.untracked.bool(False),                          
    saveFile = cms.untracked.bool(True),
    outputFile = cms.string('Pixel_DQM_HitEfficiency.root'),
    modOn = cms.untracked.bool(False),
    ladOn = cms.untracked.bool(True),
    layOn = cms.untracked.bool(False),
    phiOn = cms.untracked.bool(False),
    ringOn = cms.untracked.bool(False),
    bladeOn = cms.untracked.bool(True),
    diskOn = cms.untracked.bool(False),
    updateEfficiencies = cms.untracked.bool(False), 

    trajectoryInput = cms.InputTag('rsWithMaterialTracksP5'),  
    applyEdgeCut = cms.untracked.bool(False),
    nSigma_EdgeCut = cms.untracked.double(2.)             
)
