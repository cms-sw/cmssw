import FWCore.ParameterSet.Config as cms

SiPixelDigiSource = cms.EDFilter("SiPixelDigiSource",
    src = cms.InputTag("siPixelDigis"),
    outputFile = cms.string('Pixel_DQM_Digi.root'),
    saveFile = cms.untracked.bool(False),
    isPIB = cms.untracked.bool(False),
    slowDown = cms.untracked.bool(False),
    modOn = cms.untracked.bool(True),
    twoDimOn = cms.untracked.bool(True),
	reducedSet = cms.untracked.bool(True),	
    hiRes = cms.untracked.bool(False),                            
    ladOn = cms.untracked.bool(False),
    layOn = cms.untracked.bool(False),
    phiOn = cms.untracked.bool(False),
    ringOn = cms.untracked.bool(False),
    bladeOn = cms.untracked.bool(False),
    diskOn = cms.untracked.bool(False)
)


