import FWCore.ParameterSet.Config as cms

from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
SiPixelLorentzAnglePCLWorker = DQMEDAnalyzer( 
    "SiPixelLorentzAnglePCLWorker",
    folder = cms.string('AlCaReco/SiPixelLorentzAngle'),
    src = cms.InputTag("TrackRefitter"),    
    binsDepth	= cms.int32(50),
    binsDrift =	cms.int32(200),
    ptMin = cms.double(3),
    normChi2Max = cms.double(2),
    clustSizeYMin = cms.int32(4),
    residualMax = cms.double(0.005),
    clustChargeMax = cms.double(120000)
)
