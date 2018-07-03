import FWCore.ParameterSet.Config as cms

tpScales = cms.PSet(
    HF=cms.PSet(
        NCTShift=cms.int32(1),
        RCTShift=cms.int32(3),
    ),
    HBHE=cms.PSet(
        LSBQIE8=cms.double(1/8.),
        LSBQIE11=cms.double(1/16.),
        LSBQIE11Overlap=cms.double(1/8.),
    )
)

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
