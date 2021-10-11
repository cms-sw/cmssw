import FWCore.ParameterSet.Config as cms

from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
SiPixelLorentzAnglePCLWorker = DQMEDAnalyzer(
    "SiPixelLorentzAnglePCLWorker",
    folder = cms.string('AlCaReco/SiPixelLorentzAngle'),
    notInPCL = cms.bool(False),
    fileName = cms.string('testrun.root'),
    newmodulelist = cms.vstring("BPix_BmI_SEC7_LYR2_LDR12F_MOD1",
                                "BPix_BmI_SEC8_LYR2_LDR14F_MOD1",
                                "BPix_BmO_SEC3_LYR2_LDR5F_MOD1",
                                "BPix_BmO_SEC3_LYR2_LDR5F_MOD2",
                                "BPix_BmO_SEC3_LYR2_LDR5F_MOD3",
                                "BPix_BpO_SEC1_LYR2_LDR1F_MOD1",
                                "BPix_BpO_SEC1_LYR2_LDR1F_MOD2",
                                "BPix_BpO_SEC1_LYR2_LDR1F_MOD3"),
    src = cms.InputTag("TrackRefitter"),
    binsDepth    = cms.int32(50),
    binsDrift =    cms.int32(200),
    ptMin = cms.double(3),
    normChi2Max = cms.double(2),
    clustSizeYMin = cms.int32(4),
    clustSizeYMinL4 = cms.int32(3),
    clustSizeXMax = cms.int32(5),
    residualMax = cms.double(0.005),
    clustChargeMaxPerLength = cms.double(50000)
)
