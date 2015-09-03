import FWCore.ParameterSet.Config as cms

SiPixelDigiSource = cms.EDAnalyzer("SiPixelDigiSource",
    TopFolderName = cms.string('Pixel'),
    src = cms.InputTag("siPixelDigis"),
    outputFile = cms.string('Pixel_DQM_Digi.root'),
    saveFile = cms.untracked.bool(False),
    isPIB = cms.untracked.bool(False),
    slowDown = cms.untracked.bool(False),
    modOn = cms.untracked.bool(True),
    twoDimOn = cms.untracked.bool(True),	
    twoDimModOn = cms.untracked.bool(True),     
    #allows to have no twoD plots on Mod level (but possibly on other levels),
    #if !twoDimOn no plots on module level anyway, no projections if twoDimOn and !twoDimModOn
    twoDimOnlyLayDisk = cms.untracked.bool(False), 
    #allows to have only twoD plots on lay/disk level (even if layOn/diskOn), no others (and no projections)
    #only possible if !reducedSet, twoD has no impact, for SiPixelMonitorClient hiRes must be true
    reducedSet = cms.untracked.bool(True),
    hiRes = cms.untracked.bool(False), 
    ladOn = cms.untracked.bool(False),
    layOn = cms.untracked.bool(False),
    phiOn = cms.untracked.bool(False),
    ringOn = cms.untracked.bool(False),
    bladeOn = cms.untracked.bool(False),
    diskOn = cms.untracked.bool(False),
    isOffline = cms.untracked.bool(False),
    bigEventSize = cms.untracked.int32(1000)
)


