import FWCore.ParameterSet.Config as cms

CaloTPGTranscoder = cms.ESSource("CaloTPGTranscoderULUTs",
    hcalLUT1 = cms.FileInPath('CalibCalorimetry/CaloTPG/data/outputLUTtranscoder_physics.dat'),
    hcalLUT2 = cms.FileInPath('CalibCalorimetry/CaloTPG/data/TPGcalcDecompress2.txt')
)


