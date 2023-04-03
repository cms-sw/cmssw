import FWCore.ParameterSet.Config as cms

from CalibCalorimetry.CaloTPG.tpScales_cff import tpScales

CaloTPGTranscoder = cms.ESProducer("CaloTPGTranscoderULUTs",
    hcalLUT1 = cms.FileInPath('CalibCalorimetry/CaloTPG/data/outputLUTtranscoder_physics.dat'),
    hcalLUT2 = cms.FileInPath('CalibCalorimetry/CaloTPG/data/TPGcalcDecompress2.txt'),
    read_Ascii_Compression_LUTs = cms.bool(False),
    read_Ascii_RCT_LUTs = cms.bool(False),
    ietaLowerBound = cms.vint32( 1,18,27,29),
    ietaUpperBound = cms.vint32(17,26,28,32),
    ZS = cms.vint32(4,2,1,0),
    LUTfactor = cms.vint32(1,2,5,0),
    linearLUTs = cms.bool(False),
    nominal_gain = cms.double(0.177),
    RCTLSB = cms.double(0.25),
    tpScales = tpScales,
)

from Configuration.Eras.Modifier_run2_HCAL_2018_cff import run2_HCAL_2018
run2_HCAL_2018.toModify(CaloTPGTranscoder, linearLUTs=True)
