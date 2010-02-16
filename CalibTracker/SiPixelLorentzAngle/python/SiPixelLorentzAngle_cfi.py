# The following comments couldn't be translated into the new config version:

# 	bool MTCCtrack= false

import FWCore.ParameterSet.Config as cms

read = cms.EDAnalyzer("SiPixelLorentzAngle",
    #what type of tracks should be used: 
    #   	string src = "generalTracks"
    src = cms.string('globalMuons'),
    binsDepth = cms.int32(50),
    Fitter = cms.string('KFFittingSmoother'),
    fileNameFit = cms.string('lorentzFit.txt'),
    #in case of MC set this to true to save the simhits
    simData = cms.bool(False),
    TTRHBuilder = cms.string('WithTrackAngle'),
    ptMin = cms.double(3.0),
    fileName = cms.string('lorentzangle.root'),
    Propagator = cms.string('PropagatorWithMaterial'),
    binsDrift = cms.int32(60)
)


