import FWCore.ParameterSet.Config as cms

CaloTPGTranscoder = cms.ESSource("CaloTPGTranscoderULUTs",
    hcalLUT1 = cms.FileInPath('CalibCalorimetry/CaloTPG/data/outputLUTtranscoder.dat'),
    hcalLUT2 = cms.FileInPath('CalibCalorimetry/CaloTPG/data/TPGcalcDecompress2.txt')
)


